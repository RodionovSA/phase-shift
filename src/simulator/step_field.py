# src/simulator/step_field.py
"""Phase step ``delta_n + Delta_n(x, y)`` with ``Delta_n`` expanded on spatial modes.

``Delta_n = sum_j c_jn H_j`` as in ``docs/interference_model.md`` Eq. (9b) and
``docs/vp_aia.md`` Eq. (6). The modes ``H_j`` are the unit-norm rows of
:func:`phase_shift.basis.spatial_basis`, so the coefficients ``c`` are in the
units of VP-AIA's fitted ``coeffs``. Every generator returns coefficients with
zero frame mean, ``<c_jn>_n = 0``.
"""

import numpy as np

from phase_shift.backend import get_array_module, to_device

from .basis import poly_modes
from .shapes import check_shape


def _per_mode(name: str, value: float | np.ndarray, J: int) -> np.ndarray:
    """Broadcast a scalar or ``(J,)`` non-negative parameter to shape ``(J, 1)``."""
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        value = np.full(J, value)
    check_shape(name, value, (J,))
    if np.any(value < 0):
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value[:, None]


def _zero_frame_mean(c: np.ndarray) -> np.ndarray:
    """Subtract each mode's mean over frames, Eq. (9b)."""
    return c - c.mean(axis=1, keepdims=True)


def random_coeffs(J: int, N: int, rms: float | np.ndarray,
                  rng: int | np.random.Generator | None = None,
                  device: str = "cpu") -> np.ndarray:
    """Independent Gaussian coefficients per mode and frame, then zero frame mean.

    Parameters
    ----------
    J, N : int
        Number of modes and frames.
    rms : float or np.ndarray, shape (J,)
        Standard deviation per mode, before the frame mean is removed.
    rng : int or np.random.Generator, optional
        Seed or generator for reproducible coefficients.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned array.

    Returns
    -------
    np.ndarray, shape (J, N), float64

    Raises
    ------
    ValueError
        If ``rms`` is negative or not scalar or ``(J,)``.
    """
    scale = _per_mode("rms", rms, J)
    c = scale * np.random.default_rng(rng).standard_normal((J, N))
    return to_device(_zero_frame_mean(c), device)


def random_walk_coeffs(J: int, N: int, sigma: float | np.ndarray,
                       rng: int | np.random.Generator | None = None,
                       device: str = "cpu") -> np.ndarray:
    """Coefficients drifting as a Gaussian random walk over frames, then zero frame mean.

    Parameters
    ----------
    J, N : int
        Number of modes and frames.
    sigma : float or np.ndarray, shape (J,)
        Standard deviation of the frame-to-frame increment per mode.
    rng : int or np.random.Generator, optional
        Seed or generator for reproducible coefficients.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned array.

    Returns
    -------
    np.ndarray, shape (J, N), float64

    Raises
    ------
    ValueError
        If ``sigma`` is negative or not scalar or ``(J,)``.
    """
    scale = _per_mode("sigma", sigma, J)
    c = np.cumsum(scale * np.random.default_rng(rng).standard_normal((J, N)), axis=1)
    return to_device(_zero_frame_mean(c), device)


def linear_coeffs(N: int, slopes: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Coefficients changing linearly over frames, ``c_jn = slope_j * (n - (N-1)/2)``.

    Parameters
    ----------
    N : int
        Number of frames.
    slopes : array-like, shape (J,)
        Change per frame of each mode's coefficient.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned array.

    Returns
    -------
    np.ndarray, shape (J, N), float64

    Raises
    ------
    ValueError
        If ``slopes`` is not 1-D.
    """
    slopes = np.asarray(slopes, dtype=np.float64)
    check_shape("slopes", slopes, (None,))
    c = slopes[:, None] * (np.arange(N) - (N - 1) / 2)
    return to_device(c, device)


def proportional_coeffs(piston: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Coefficients following the piston step, ``c_jn = k_j * (delta_n - <delta>_n)``.

    Parameters
    ----------
    piston : np.ndarray, shape (N,)
        Piston step per frame in radians.
    gains : array-like, shape (J,)
        Coefficient per radian of piston, ``k_j``, for each mode.

    Returns
    -------
    np.ndarray, shape (J, N), float64
        On the device of ``piston``.

    Raises
    ------
    ValueError
        If ``piston`` or ``gains`` is not 1-D.
    """
    check_shape("piston", piston, (None,))
    xp = get_array_module(piston)
    gains = xp.asarray(gains, dtype=xp.float64)
    check_shape("gains", gains, (None,))
    return gains[:, None] * (piston - piston.mean())[None, :]


def step_field(H: int, W: int, piston: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Phase step ``delta_n + sum_j c_jn H_j``, ``docs/interference_model.md`` Eq. (9a).

    Parameters
    ----------
    H, W : int
        Field height and width.
    piston : np.ndarray, shape (N,)
        Piston step ``delta_n`` in radians.
    coeffs : np.ndarray, shape (J, N)
        Mode coefficients ``c_jn`` in VP-AIA units; ``J = (M+1)(M+2)/2 - 1``
        for degree ``M``, e.g. 0, 2, 5, 9.

    Returns
    -------
    np.ndarray, shape (N, H, W)
        Phase step in radians, ready for :func:`simulator.total_phase`.

    Raises
    ------
    ValueError
        If ``piston`` is not 1-D, ``coeffs`` is not ``(J, N)``, or ``J``
        matches no degree.
    """
    check_shape("piston", piston, (None,))
    N = piston.shape[0]
    check_shape("coeffs", coeffs, (None, N))
    xp = get_array_module(piston, coeffs)
    modes = poly_modes(H, W, coeffs.shape[0], xp)                           # (J, H*W)
    Delta = (coeffs.T @ modes).reshape(N, H, W)
    return piston[:, None, None] + Delta
