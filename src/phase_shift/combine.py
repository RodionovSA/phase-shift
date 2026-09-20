# src/phase_shift/combine.py
"""Averaging repeated, independent phase acquisitions of the same object."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .backend import get_array_module, to_device
from .carrier import remove_carrier
from .reference import subtract_reference
from .utils import _estimation_weight


@dataclass
class CombinedResult:
    """Output of :func:`combine_acquisitions`.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Combined phase map, in radians, wrapped to ``[-pi, pi]``.
    scatter : np.ndarray, shape (H, W)
        Per-pixel circular standard deviation across the acquisitions, in
        radians, ``sqrt(-2*log(R))``.
    mean_resultant : np.ndarray, shape (H, W)
        Per-pixel mean resultant length ``R`` in ``[0, 1]``; 1 is perfect
        agreement. Also usable as a reliability weight.
    n : int
        Number of acquisitions combined.
    sign_flips : list of int
        Indices into ``phis`` whose sign branch was flipped before averaging.
    """

    phi: np.ndarray
    scatter: np.ndarray
    mean_resultant: np.ndarray
    n: int
    sign_flips: list[int]


def combine_acquisitions(phis: Sequence[np.ndarray],
                         weights: Sequence[np.ndarray] | None = None,
                         align_carrier: bool = True, reference: int = 0,
                         carrier_kwargs: dict | None = None,
                         device: str = "auto") -> CombinedResult:
    """Average independent phase measurements of the same object.

    Averaging ``k`` acquisitions reduces the random, acquisition-to-
    acquisition component of the error as ``1/sqrt(k)``; it does not help
    against per-frame model error, which is systematic within a run.

    Each map is first oriented to agree with ``phis[reference]`` using
    :func:`phase_shift.reference.subtract_reference`'s discriminant, run on
    the raw maps before carrier removal; then, when ``align_carrier`` is set,
    passed through :func:`phase_shift.carrier.remove_carrier`; then combined
    as the weighted complex mean. See ``docs/gauge_conventions.md``
    §"Multi-acquisition combination".

    Parameters
    ----------
    phis : sequence of np.ndarray, each shape (H, W)
        Phase maps of the same object, in radians, e.g.
        :attr:`phase_shift.result.PhaseResult.phi` from separate fits.
    weights : sequence of np.ndarray, each shape (H, W), optional
        Per-acquisition reliability, e.g. each fit's
        :attr:`phase_shift.result.PhaseResult.b`. Used both for carrier
        removal and for the final mean. Negative values are clipped to 0.
        Defaults to uniform weight.
    align_carrier : bool, default True
        Remove each map's own carrier before combining. Turn off when the
        maps already share one carrier.
    reference : int, default 0
        Index into ``phis`` fixing the sign convention.
    carrier_kwargs : dict, optional
        Keyword arguments forwarded to
        :func:`phase_shift.carrier.remove_carrier`, overriding its defaults.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Device to run on; see :func:`phase_shift.backend.to_device`.

    Returns
    -------
    CombinedResult
        Combined phase, its per-pixel scatter, and which maps were flipped.

    Raises
    ------
    ValueError
        If fewer than 2 acquisitions are given, if a map is not 2-D or the
        maps differ in shape, if ``reference`` is out of range, or if
        ``weights`` does not have one entry per acquisition.
    """
    n = len(phis)
    if n < 2:
        raise ValueError(f"phis must hold at least 2 acquisitions, got {n}")
    if not 0 <= reference < n:
        raise ValueError(f"reference must be in [0, {n}), got {reference}")
    if weights is not None and len(weights) != n:
        raise ValueError(f"weights must have one entry per acquisition ({n}), "
                         f"got {len(weights)}")

    phis = [to_device(p, device=device) for p in phis]
    xp = get_array_module(*phis)
    if phis[0].ndim != 2:
        raise ValueError(f"each map in phis must be 2-D (H, W), got shape {phis[0].shape}")
    for i, p in enumerate(phis):
        if p.shape != phis[0].shape:
            raise ValueError(f"phis[{i}] has shape {p.shape}, expected {phis[0].shape}")

    ws = [_estimation_weight(phis[i], None if weights is None else weights[i],
                             device=device, uniform_if_empty=False)
          for i in range(n)]

    sign_flips = []
    resolved = list(phis)
    for i in range(n):
        if i == reference:
            continue
        dr = subtract_reference(phis[reference], phis[i], weight=ws[reference], device=device)
        if dr.sign == -1:
            # Left unwrapped: every later use goes through exp(1j*phi).
            resolved[i] = -phis[i]
            sign_flips.append(i)

    aligned = resolved
    if align_carrier:
        ckw = dict(carrier_kwargs) if carrier_kwargs else {}
        aligned = [remove_carrier(p, weight=ws[i], device=device, **ckw).phi
                   for i, p in enumerate(aligned)]

    # Accumulate the weighted circular mean instead of stacking (n, H, W).
    mean_field = total_w = None
    for w, p in zip(ws, aligned):
        term = w * xp.exp(1j * p)
        mean_field = term if mean_field is None else mean_field + term
        total_w = w if total_w is None else total_w + w
    # Keep each floor in its array's own dtype, so a float32 input is not
    # promoted by a float64 scalar.
    total_w = xp.where(total_w > 0, total_w,
                       xp.asarray(xp.finfo(total_w.dtype).eps, total_w.dtype))
    mean_field = mean_field / total_w

    R = xp.clip(xp.abs(mean_field), 0, 1)
    R_floor = xp.asarray(xp.finfo(R.dtype).eps, R.dtype)
    scatter = xp.sqrt(xp.clip(-2 * xp.log(xp.maximum(R, R_floor)), 0, None))

    return CombinedResult(phi=xp.angle(mean_field), scatter=scatter, mean_resultant=R,
                          n=n, sign_flips=sign_flips)
