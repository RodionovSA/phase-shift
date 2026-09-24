# src/simulator/sequences.py
"""Per-frame ``(N,)`` sequences: source-power factor ``alpha``, contrast ``g``, piston step.

Combine ``alpha`` and ``g`` sequences by multiplying them.
"""

import numpy as np

from phase_shift.backend import to_device


def _check_frames(N: int) -> None:
    """Raise ``ValueError`` unless ``N`` is positive."""
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")


def linear(N: int, start: float, stop: float, device: str = "cpu") -> np.ndarray:
    """Linear drop or grow from ``start`` at the first frame to ``stop`` at the last.

    Parameters
    ----------
    N : int
        Number of frames, positive.
    start, stop : float
        Values at the first and last frame.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned sequence.

    Returns
    -------
    np.ndarray, shape (N,), float64

    Raises
    ------
    ValueError
        If ``N`` is not positive.
    """
    _check_frames(N)
    return to_device(np.linspace(start, stop, N), device)


def random_dips(N: int, rate: float, depth: float,
                rng: int | np.random.Generator | None = None, base: float = 1.0,
                two_sided: bool = False, device: str = "cpu") -> np.ndarray:
    """Irregular disturbances: each frame is hit independently with probability ``rate``.

    A hit frame gets ``base * (1 - depth * u)``, ``u`` uniform in ``[0, 1]``;
    with ``two_sided`` the sign of ``depth * u`` is random. Other frames get
    ``base``.

    Parameters
    ----------
    N : int
        Number of frames, positive.
    rate : float
        Probability that a frame is disturbed, in ``[0, 1]``.
    depth : float
        Largest relative change of a disturbed frame, in ``[0, 1]``.
    rng : int or np.random.Generator, optional
        Seed or generator for reproducible sequences.
    base : float, default 1.0
        Value of undisturbed frames.
    two_sided : bool, default False
        Allow disturbances above ``base`` as well as below.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned sequence.

    Returns
    -------
    np.ndarray, shape (N,), float64

    Raises
    ------
    ValueError
        If ``N`` is not positive, or ``rate`` or ``depth`` leaves ``[0, 1]``.
    """
    _check_frames(N)
    if not 0 <= rate <= 1:
        raise ValueError(f"rate must lie in [0, 1], got {rate}")
    if not 0 <= depth <= 1:
        raise ValueError(f"depth must lie in [0, 1], got {depth}")
    rng = np.random.default_rng(rng)
    hit = rng.random(N) < rate
    change = depth * rng.random(N)
    if two_sided:
        change *= rng.choice([-1.0, 1.0], N)
    return to_device(base * (1 - np.where(hit, change, 0.0)), device)


def uniform_steps(N: int, offset: float = 0.0, device: str = "cpu") -> np.ndarray:
    """Piston steps evenly spread over one period, ``2 pi n / N + offset``.

    Parameters
    ----------
    N : int
        Number of frames, positive.
    offset : float, default 0.0
        Step of the first frame in radians.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned sequence.

    Returns
    -------
    np.ndarray, shape (N,), float64

    Raises
    ------
    ValueError
        If ``N`` is not positive.
    """
    _check_frames(N)
    return to_device(2 * np.pi * np.arange(N) / N + offset, device)


def linear_steps(N: int, step: float, sigma: float = 0.0,
                 rng: int | np.random.Generator | None = None, offset: float = 0.0,
                 device: str = "cpu") -> np.ndarray:
    """Piston steps of fixed size with independent Gaussian noise per frame.

    ``step * n + offset + e_n``, ``e_n ~ N(0, sigma^2)``.

    Parameters
    ----------
    N : int
        Number of frames, positive.
    step : float
        Nominal step between frames in radians.
    sigma : float, default 0.0
        Standard deviation of the noise in radians, non-negative.
    rng : int or np.random.Generator, optional
        Seed or generator for reproducible sequences.
    offset : float, default 0.0
        Nominal step of the first frame in radians.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned sequence.

    Returns
    -------
    np.ndarray, shape (N,), float64

    Raises
    ------
    ValueError
        If ``N`` is not positive or ``sigma`` is negative.
    """
    _check_frames(N)
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    noise = sigma * np.random.default_rng(rng).standard_normal(N)
    return to_device(step * np.arange(N) + offset + noise, device)


def random_steps(N: int, rng: int | np.random.Generator | None = None,
                 device: str = "cpu") -> np.ndarray:
    """Piston steps drawn independently and uniformly from ``[0, 2 pi)``.

    Parameters
    ----------
    N : int
        Number of frames, positive.
    rng : int or np.random.Generator, optional
        Seed or generator for reproducible sequences.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned sequence.

    Returns
    -------
    np.ndarray, shape (N,), float64

    Raises
    ------
    ValueError
        If ``N`` is not positive.
    """
    _check_frames(N)
    return to_device(2 * np.pi * np.random.default_rng(rng).random(N), device)
