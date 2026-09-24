# src/simulator/separable.py
"""Time-dependent beam intensities and coherence as static maps times per-frame factors."""

import numpy as np

from .shapes import check_shape


def beam_intensity(I_map: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Build a beam intensity of ``docs/interference_model.md`` Eq. (6).

    ``I = alpha * I_map``.

    Parameters
    ----------
    I_map : np.ndarray, shape (1, H, W)
        Static beam intensity.
    alpha : np.ndarray, shape (N,)
        Source-power factor per frame, non-negative.

    Returns
    -------
    np.ndarray, shape (N, H, W)

    Raises
    ------
    ValueError
        If an input does not have the shape above, or ``alpha`` has a
        negative value.
    """
    check_shape("I_map", I_map, (1, None, None))
    check_shape("alpha", alpha, (None,))
    if bool((alpha < 0).any()):
        raise ValueError(f"alpha must be non-negative, got minimum {float(alpha.min())}")
    return alpha[:, None, None] * I_map


def coherence(gamma_map: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Build the coherence of ``docs/interference_model.md`` Eq. (12).

    ``gamma = g * gamma_map``.

    Parameters
    ----------
    gamma_map : np.ndarray, shape (1, H, W)
        Static coherence envelope.
    g : np.ndarray, shape (N,)
        Contrast factor per frame, in ``[0, 1]``.

    Returns
    -------
    np.ndarray, shape (N, H, W)

    Raises
    ------
    ValueError
        If an input does not have the shape above, or ``g`` leaves ``[0, 1]``.
    """
    check_shape("gamma_map", gamma_map, (1, None, None))
    check_shape("g", g, (None,))
    if bool(((g < 0) | (g > 1)).any()):
        raise ValueError(f"g must lie in [0, 1], got range [{float(g.min())}, {float(g.max())}]")
    return g[:, None, None] * gamma_map
