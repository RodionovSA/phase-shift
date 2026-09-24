# src/simulator/beams.py
"""Background and fringe amplitude from the two beam intensities."""

import numpy as np

from phase_shift.backend import get_array_module

from .shapes import check_shapes


def fringe_terms(I1: np.ndarray, I2: np.ndarray, gamma: np.ndarray,
                 theta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build ``a`` and ``b`` of ``docs/interference_model.md`` Eq. (2).

    ``a = I1 + I2``, ``b = 2 * gamma * sqrt(I1 * I2) * cos(theta)``.

    Parameters
    ----------
    I1, I2 : np.ndarray, 3-D
        Beam intensities at the sensor, non-negative.
    gamma : np.ndarray, 3-D
        Degree of coherence, in ``[0, 1]``.
    theta : np.ndarray, 3-D, optional
        Angle between the polarization vectors in radians. None means
        co-polarized beams, ``theta = 0`` (Eq. 4).

    Each input has axes ``(N, H, W)``; an axis of size 1 broadcasts.

    Returns
    -------
    a, b : np.ndarray
        Background and fringe amplitude, shaped as the inputs broadcast.

    Raises
    ------
    ValueError
        If an input is not 3-D, the shapes do not broadcast together,
        ``I1`` or ``I2`` has a negative value, or ``gamma`` leaves ``[0, 1]``.
    """
    arrays = {"I1": I1, "I2": I2, "gamma": gamma}
    if theta is not None:
        arrays["theta"] = theta
    check_shapes(**arrays)
    for name, x in (("I1", I1), ("I2", I2)):
        if bool((x < 0).any()):
            raise ValueError(f"{name} must be non-negative, got minimum {float(x.min())}")
    if bool(((gamma < 0) | (gamma > 1)).any()):
        raise ValueError(
            f"gamma must lie in [0, 1], got range [{float(gamma.min())}, {float(gamma.max())}]"
        )
    xp = get_array_module(*arrays.values())
    a = I1 + I2
    b = 2 * gamma * xp.sqrt(I1 * I2)
    if theta is not None:
        b = b * xp.cos(theta)
    return a, b
