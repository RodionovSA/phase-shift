# src/phase_shift/interference_model.py
"""Interference model of ``docs/interference_model.md``."""

import numpy as np

from .backend import get_array_module


def model_stack(a: np.ndarray, b: np.ndarray, phi: np.ndarray, delta: np.ndarray,
                g: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Evaluate the frame stack of ``docs/interference_model.md`` Eq. (17).

    ``I_n = alpha_n * (a + g_n * b * cos(phi + delta_n))``.

    Parameters
    ----------
    a, b, phi : np.ndarray, shape (H, W)
        Background, fringe amplitude, and phase in radians.
    delta : np.ndarray, shape (N,) or (N, H, W)
        Phase step per frame, or per frame and pixel, in radians.
    g, alpha : np.ndarray, shape (N,)
        Per-frame fringe gain and source-power factor.

    Returns
    -------
    np.ndarray, shape (N, H, W)

    Raises
    ------
    ValueError
        If ``delta`` is neither 1-D nor 3-D.
    """
    if delta.ndim not in (1, 3):
        raise ValueError(f"delta must have shape (N,) or (N, H, W), got {delta.shape}")
    xp = get_array_module(a, b, phi, delta, g, alpha)
    if delta.ndim == 1:
        delta = delta[:, xp.newaxis, xp.newaxis]
    carrier = g[:, xp.newaxis, xp.newaxis] * b[xp.newaxis, :, :] \
        * xp.cos(phi[xp.newaxis, :, :] + delta)
    return alpha[:, xp.newaxis, xp.newaxis] * (a[xp.newaxis, :, :] + carrier)
