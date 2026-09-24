# src/simulator/core.py
"""Two-beam interference stack from background, amplitude, and phase arrays."""

import numpy as np

from phase_shift.backend import get_array_module

from .shapes import check_shapes


def simulate(a: np.ndarray, b: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Evaluate the interference stack of ``docs/interference_model.md`` Eq. (3).

    ``I = a + b * cos(psi)``.

    Parameters
    ----------
    a, b : np.ndarray, 3-D
        Background and fringe amplitude.
    psi : np.ndarray, 3-D
        Total phase in radians.

    Each input has axes ``(N, H, W)``; an axis of size 1 broadcasts, e.g.
    ``(1, H, W)`` for a static field or ``(N, 1, 1)`` for a per-frame value.

    Returns
    -------
    np.ndarray, shape (N, H, W)

    Raises
    ------
    ValueError
        If an input is not 3-D, or the shapes do not broadcast together.
    """
    check_shapes(a=a, b=b, psi=psi)
    xp = get_array_module(a, b, psi)
    return a + b * xp.cos(psi)
