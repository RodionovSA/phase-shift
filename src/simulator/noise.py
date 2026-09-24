# src/simulator/noise.py
"""Zero-mean camera noise models, ``docs/aia_noise.md`` Eq. (1).

Every model has ``variance(I)``, the per-pixel, per-frame variance
``sigma_n^2(x, y)``, and ``model(I, rng)``, a noisy copy of ``I``. Noise is
drawn on the CPU from ``rng``, so a seed gives the same stack on every
device; the result is returned on the device of ``I``.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from phase_shift.backend import asnumpy, get_array_module

from .shapes import check_shape


class NoiseModel(Protocol):
    """Interface shared by the noise models."""

    def variance(self, I: np.ndarray) -> np.ndarray:
        """Noise variance of each pixel and frame of ``I``, shape of ``I``."""
        ...

    def __call__(self, I: np.ndarray,
                 rng: int | np.random.Generator | None = None) -> np.ndarray:
        """Noisy copy of ``I``."""
        ...


def _check_stack(I: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``I`` is an ``(N, H, W)`` array."""
    check_shape("I", I, (None, None, None))


@dataclass(frozen=True)
class GaussianNoise:
    """Additive Gaussian noise of standard deviation ``sigma``.

    Attributes
    ----------
    sigma : float or np.ndarray
        Standard deviation in the units of ``I``, non-negative; a scalar or a
        NumPy array broadcastable to ``(N, H, W)``.
    """

    sigma: float | np.ndarray

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.sigma) < 0):
            raise ValueError(f"sigma must be non-negative, got minimum "
                             f"{float(np.min(self.sigma))}")

    def variance(self, I: np.ndarray) -> np.ndarray:
        """``sigma^2`` broadcast to the shape of ``I``.

        Raises
        ------
        ValueError
            If ``I`` is not 3-D, or ``sigma`` does not broadcast to it.
        """
        _check_stack(I)
        xp = get_array_module(I)
        sigma = xp.asarray(self.sigma, dtype=xp.float64)
        return xp.broadcast_to(sigma ** 2, I.shape)

    def __call__(self, I: np.ndarray,
                 rng: int | np.random.Generator | None = None) -> np.ndarray:
        """``I + sigma * e``, ``e`` standard normal per pixel and frame.

        Raises
        ------
        ValueError
            If ``I`` is not 3-D, or ``sigma`` does not broadcast to it.
        """
        _check_stack(I)
        sigma = np.broadcast_to(np.asarray(self.sigma, dtype=np.float64), I.shape)
        noise = sigma * np.random.default_rng(rng).standard_normal(I.shape)
        return I + get_array_module(I).asarray(noise)


@dataclass(frozen=True)
class CameraNoise:
    """Shot noise plus Gaussian read noise of a linear camera, ``I`` in DN.

    ``I_meas = Poisson(gain * I) / gain + sigma_read * e``, so
    ``variance = I / gain + sigma_read^2``.

    Attributes
    ----------
    sigma_read : float
        Read noise in DN, non-negative.
    gain : float
        Conversion gain in electrons per DN, positive.
    """

    sigma_read: float
    gain: float

    def __post_init__(self) -> None:
        if self.sigma_read < 0:
            raise ValueError(f"sigma_read must be non-negative, got {self.sigma_read}")
        if self.gain <= 0:
            raise ValueError(f"gain must be positive, got {self.gain}")

    def _check(self, I: np.ndarray) -> None:
        """Raise ``ValueError`` unless ``I`` is a non-negative ``(N, H, W)`` stack."""
        _check_stack(I)
        if bool((I < 0).any()):
            raise ValueError(f"I must be non-negative, got minimum {float(I.min())}")

    def variance(self, I: np.ndarray) -> np.ndarray:
        """``I / gain + sigma_read^2``, shape of ``I``, in DN².

        Raises
        ------
        ValueError
            If ``I`` is not 3-D or has a negative value.
        """
        self._check(I)
        return I / self.gain + self.sigma_read ** 2

    def __call__(self, I: np.ndarray,
                 rng: int | np.random.Generator | None = None) -> np.ndarray:
        """Noisy stack in DN, not quantized or clipped.

        Raises
        ------
        ValueError
            If ``I`` is not 3-D or has a negative value.
        """
        self._check(I)
        rng = np.random.default_rng(rng)
        electrons = rng.poisson(self.gain * asnumpy(I).astype(np.float64))
        noisy = electrons / self.gain + self.sigma_read * rng.standard_normal(I.shape)
        return get_array_module(I).asarray(noisy)
