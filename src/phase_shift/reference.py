# src/phase_shift/reference.py
"""Resolving the +/-phi sign branch between phase maps: a sample and its reference, or a stack."""

import warnings
from dataclasses import dataclass

import numpy as np

from .backend import get_array_module, to_device
from .utils import _estimation_weight, wrap, wrap_add, wrap_sub


@dataclass
class DifferenceResult:
    """Output of :func:`subtract_reference`.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Resolved phase difference, in radians, wrapped to ``[-pi, pi]``.
    sign : int
        ``+1`` if ``phi - phi_ref`` was kept, ``-1`` if ``phi_ref``'s sign
        branch was flipped first (``phi + phi_ref``).
    spread_same : float
        Weighted circular RMS spread of ``phi - phi_ref``, in radians.
    spread_flipped : float
        Weighted circular RMS spread of ``phi + phi_ref``, in radians.
    ambiguous : bool
        True if the two spreads were too close to pick a branch confidently.
    """

    phi: np.ndarray
    sign: int
    spread_same: float
    spread_flipped: float
    ambiguous: bool


def subtract_reference(phi: np.ndarray, phi_ref: np.ndarray,
                       weight: np.ndarray | None = None,
                       mask: np.ndarray | None = None,
                       ambiguous_ratio: float = 1.5,
                       device: str = "auto") -> DifferenceResult:
    """Subtract a reference phase map, resolving its sign branch first.

    A solve recovers ``phi`` only up to ``(phi, delta) -> (-phi, -delta)``, so
    two independent solves need not share a branch; see
    ``docs/gauge_conventions.md`` §"Reference subtraction". Both
    ``wrap(phi -+ phi_ref)`` are formed and the one with the lower weighted
    circular spread is kept, since the aberration shared by the two
    measurements cancels only in the correctly signed combination.

    Parameters
    ----------
    phi : np.ndarray, shape (H, W)
        Sample phase map, in radians.
    phi_ref : np.ndarray, shape (H, W)
        Reference phase map, in radians.
    weight : np.ndarray, shape (H, W), optional
        Per-pixel reliability, e.g. :attr:`phase_shift.result.PhaseResult.b`.
        Weights the spread comparison only, not the returned ``phi``. Negative
        values are clipped to 0.
    mask : np.ndarray, shape (H, W), optional
        Pixels where it is falsey are excluded from the comparison; combined
        with ``weight``.
    ambiguous_ratio : float, default 1.5
        Report ``ambiguous`` when the larger spread is below this factor times
        the smaller one. Must be at least 1.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Device to run on; see :func:`phase_shift.backend.to_device`.

    Returns
    -------
    DifferenceResult
        Resolved difference and the two spreads it was chosen from.

    Raises
    ------
    ValueError
        If ``phi`` is not 2-D, if ``phi_ref``, ``weight`` or ``mask`` does not
        have ``phi``'s shape, or if ``ambiguous_ratio`` is below 1.

    Warns
    -----
    UserWarning
        If the two spreads are too close to resolve the branch.
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D (H, W), got shape {phi.shape}")
    if phi.shape != phi_ref.shape:
        raise ValueError(f"phi_ref must have shape {phi.shape}, got {phi_ref.shape}")
    if ambiguous_ratio < 1:
        raise ValueError(f"ambiguous_ratio must be at least 1, got {ambiguous_ratio}")

    phi = to_device(phi, device=device)
    phi_ref = to_device(phi_ref, device=device)
    w = _estimation_weight(phi, weight, mask, device)

    diff_same = wrap_sub(phi, phi_ref)
    diff_flipped = wrap_add(phi, phi_ref)
    spread_same, spread_flipped = _spread(diff_same, w), _spread(diff_flipped, w)

    if spread_same <= spread_flipped:
        phi_out, sign = diff_same, 1
    else:
        phi_out, sign = diff_flipped, -1

    ambiguous = _is_ambiguous(spread_same, spread_flipped, ambiguous_ratio)
    if ambiguous:
        warnings.warn(
            f"subtract_reference: spread_same={spread_same:.4f} and "
            f"spread_flipped={spread_flipped:.4f} are too close to resolve the "
            f"sign branch (ratio {hi / lo:.2f} < {ambiguous_ratio}); check phi "
            f"visually before trusting it.",
            stacklevel=2,
        )

    return DifferenceResult(phi=phi_out, sign=sign, spread_same=spread_same,
                            spread_flipped=spread_flipped, ambiguous=ambiguous)


@dataclass
class SignAlignment:
    """Output of :func:`align_signs`.

    Attributes
    ----------
    phi : np.ndarray, shape (L, H, W)
        Maps on the sign branch of ``phis[0]``, in radians, wrapped to
        ``[-pi, pi]``.
    signs : np.ndarray, shape (L,)
        ``+1`` where the input map was kept, ``-1`` where it was negated.
    ambiguous : np.ndarray, shape (L,)
        True where the two spreads were too close to pick a branch
        confidently; such maps are not used as a reference for later ones.
    reference : np.ndarray, shape (L,)
        Index of the map each one was compared with; ``-1`` for ``phis[0]``.
    """

    phi: np.ndarray
    signs: np.ndarray
    ambiguous: np.ndarray
    reference: np.ndarray


def align_signs(phis: np.ndarray, weight: np.ndarray | None = None,
                mask: np.ndarray | None = None, ambiguous_ratio: float = 1.5,
                device: str = "auto") -> SignAlignment:
    """Put every map of a stack, e.g. a wavelength scan, on the sign branch of ``phis[0]``.

    Each solve recovers its map only up to ``phi -> -phi``. Map ``k`` is compared
    with the nearest earlier map that was resolved confidently, using
    :func:`subtract_reference`'s discriminant, and negated if ``-phis[k]`` agrees
    better. The maps must share their carrier (or have none) and be spatially
    aligned; see ``docs/gauge_conventions.md`` §"Reference subtraction".

    Parameters
    ----------
    phis : np.ndarray, shape (L, H, W)
        Phase maps, in radians, ordered so neighbours are similar (e.g. by
        wavelength).
    weight : np.ndarray, shape (H, W) or (L, H, W), optional
        Per-pixel reliability, shared or per map; the comparison of two maps
        uses the product of their weights. Negative values are clipped to 0.
    mask : np.ndarray, shape (H, W), optional
        Pixels where it is falsey are excluded from every comparison.
    ambiguous_ratio : float, default 1.5
        Report a map as ambiguous when the larger spread is below this factor
        times the smaller one. Must be at least 1.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Device to run on; see :func:`phase_shift.backend.to_device`.

    Returns
    -------
    SignAlignment
        Aligned stack, the applied signs, and which comparisons were ambiguous.

    Raises
    ------
    ValueError
        If ``phis`` is not 3-D, if ``weight`` is neither ``(H, W)`` nor
        ``phis``' shape, if ``mask`` is not ``(H, W)``, or if
        ``ambiguous_ratio`` is below 1.
    """
    if phis.ndim != 3:
        raise ValueError(f"phis must be 3-D (L, H, W), got shape {phis.shape}")
    if weight is not None and weight.shape not in (phis.shape, phis.shape[1:]):
        raise ValueError(f"weight must have shape {phis.shape[1:]} or {phis.shape}, "
                         f"got {weight.shape}")
    if ambiguous_ratio < 1:
        raise ValueError(f"ambiguous_ratio must be at least 1, got {ambiguous_ratio}")

    phis = to_device(phis, device=device)
    xp = get_array_module(phis)
    L = phis.shape[0]

    def w_of(k: int) -> np.ndarray:
        wk = weight if weight is None or weight.ndim == 2 else weight[k]
        return _estimation_weight(phis[k], wk, mask, device)

    out = phis.copy()
    signs = np.ones(L, dtype=int)
    ambiguous = np.zeros(L, dtype=bool)
    reference = np.full(L, -1, dtype=int)
    last, w_last = 0, w_of(0)
    for k in range(1, L):
        w_k = w_of(k)
        w = w_last * w_k
        if float(w.sum()) <= 0:
            w = xp.ones(phis.shape[1:])
        spread_same = _spread(wrap_sub(phis[k], out[last]), w)
        spread_flipped = _spread(wrap_add(phis[k], out[last]), w)
        reference[k] = last
        ambiguous[k] = _is_ambiguous(spread_same, spread_flipped, ambiguous_ratio)
        if spread_flipped < spread_same:
            signs[k] = -1
            out[k] = -phis[k]
        if not ambiguous[k]:
            last, w_last = k, w_k

    return SignAlignment(phi=out, signs=signs, ambiguous=ambiguous, reference=reference)


def _spread(d: np.ndarray, w: np.ndarray) -> float:
    """Weighted circular RMS spread of ``d`` about its mean angle."""
    xp = get_array_module(d)
    mean_angle = float(xp.angle(xp.sum(w * xp.exp(1j * d))))
    resid = wrap(d - mean_angle)
    return float(xp.sqrt(xp.sum(w * resid**2) / xp.sum(w)))


def _is_ambiguous(spread_same: float, spread_flipped: float, ratio: float) -> bool:
    """True if the larger spread is below ``ratio`` times the smaller one."""
    lo, hi = sorted([spread_same, spread_flipped])
    return hi < ratio * lo
