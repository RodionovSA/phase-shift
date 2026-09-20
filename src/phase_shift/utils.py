# src/phase_shift/utils.py
"""Shared utilities: phase wrapping, estimation weights, and value formatting."""

import numpy as np

from .backend import get_array_module, to_device


def wrap(x: np.ndarray) -> np.ndarray:
    """Wrap ``x`` (radians) into ``[-pi, pi]``."""
    xp = get_array_module(x)
    two_pi = 2 * xp.pi
    return x - two_pi * xp.round(x / two_pi)


def _wrap_combine(phi_a: np.ndarray, phi_b: np.ndarray, sign: float) -> np.ndarray:
    """Return ``phi_a + sign*phi_b`` wrapped into ``[-pi, pi]``."""
    xp = get_array_module(phi_a, phi_b)
    return xp.angle(xp.exp(1j * phi_a) * xp.exp(1j * sign * phi_b))


def wrap_add(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """Return ``phi_a + phi_b`` (radians) wrapped into ``[-pi, pi]``."""
    return _wrap_combine(phi_a, phi_b, 1.0)


def wrap_sub(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """Return ``phi_a - phi_b`` (radians) wrapped into ``[-pi, pi]``."""
    return _wrap_combine(phi_a, phi_b, -1.0)


def _estimation_weight(phi: np.ndarray, weight: np.ndarray | None = None,
                       mask: np.ndarray | None = None, device: str = "auto",
                       uniform_if_empty: bool = True) -> np.ndarray:
    """Per-pixel weight for estimating a quantity from the phase map ``phi``.

    Parameters
    ----------
    phi : np.ndarray, shape (H, W)
        Phase map the weight applies to; sets the expected shape.
    weight : np.ndarray, shape (H, W), optional
        Per-pixel reliability, e.g. a modulation map. Negative values are
        clipped to 0. Defaults to uniform weight.
    mask : np.ndarray, shape (H, W), optional
        Pixels where it is falsey get zero weight; combined with ``weight``.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Device to move ``weight`` and ``mask`` to; see
        :func:`phase_shift.backend.to_device`.
    uniform_if_empty : bool, default True
        Fall back to uniform weight when no pixel is left with positive weight.

    Returns
    -------
    np.ndarray, shape (H, W)

    Raises
    ------
    ValueError
        If ``weight`` or ``mask`` does not have ``phi``'s shape.
    """
    xp = get_array_module(phi)
    if weight is None:
        w = xp.ones(phi.shape)
    else:
        w = xp.clip(to_device(weight, device=device), 0, None)
        if w.shape != phi.shape:
            raise ValueError(f"weight must have shape {phi.shape}, got {w.shape}")
    if mask is not None:
        mask = to_device(mask, device=device, dtype=bool)
        if mask.shape != phi.shape:
            raise ValueError(f"mask must have shape {phi.shape}, got {mask.shape}")
        w = w * mask
    if uniform_if_empty and float(w.sum()) <= 0:
        w = xp.ones(phi.shape)
    return w


def format_value(value: object) -> str:
    """Format ``value`` for printing: floats to 6 significant digits, else ``str``."""
    return f"{value:.6g}" if isinstance(value, float) else str(value)
