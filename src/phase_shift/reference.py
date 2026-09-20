# src/phase_shift/reference.py
"""Resolving the +/-phi sign branch between a sample and a reference phase map."""

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
    xp = get_array_module(phi, phi_ref)
    w = _estimation_weight(phi, weight, mask, device)

    def spread(d: np.ndarray) -> float:
        """Weighted circular RMS spread of ``d`` about its mean angle."""
        mean_angle = float(xp.angle(xp.sum(w * xp.exp(1j * d))))
        resid = wrap(d - mean_angle)
        return float(xp.sqrt(xp.sum(w * resid**2) / xp.sum(w)))

    diff_same = wrap_sub(phi, phi_ref)
    diff_flipped = wrap_add(phi, phi_ref)
    spread_same = spread(diff_same)
    spread_flipped = spread(diff_flipped)

    if spread_same <= spread_flipped:
        phi_out, sign = diff_same, 1
    else:
        phi_out, sign = diff_flipped, -1

    lo, hi = sorted([spread_same, spread_flipped])
    ambiguous = hi < ambiguous_ratio * lo
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
