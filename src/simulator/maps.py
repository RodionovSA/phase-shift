# src/simulator/maps.py
"""Static ``(1, H, W)`` maps: beam intensities, coherence envelope, sample phase."""

import numpy as np

from phase_shift.backend import get_array_module

from .coords import pixel_coords
from .shapes import check_shape


def gaussian_map(H: int, W: int, fwhm: tuple[float, float], peak: float = 1.0,
                 center: tuple[float, float] | None = None,
                 device: str = "cpu") -> np.ndarray:
    """Elliptical Gaussian map.

    ``peak * exp(-4 ln2 * ((x - x0)^2 / fwhm_x^2 + (y - y0)^2 / fwhm_y^2))``.

    Parameters
    ----------
    H, W : int
        Field height and width.
    fwhm : (float, float)
        Full width at half maximum ``(fwhm_x, fwhm_y)`` in pixels, positive.
    peak : float, default 1.0
        Value at ``center``.
    center : (float, float), optional
        Center ``(x0, y0)`` in pixels from the top-left pixel. None means the
        field center.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned map.

    Returns
    -------
    np.ndarray, shape (1, H, W), float64

    Raises
    ------
    ValueError
        If a ``fwhm`` component is not positive.
    """
    fx, fy = fwhm
    if fx <= 0 or fy <= 0:
        raise ValueError(f"fwhm must be positive, got {fwhm}")
    x, y = pixel_coords(H, W, center, device)
    xp = get_array_module(x)
    return peak * xp.exp(-4 * np.log(2) * ((x / fx) ** 2 + (y / fy) ** 2))


def masked_map(mask: np.ndarray, inside: float, outside: float) -> np.ndarray:
    """Map equal to ``inside`` within ``mask`` and ``outside`` elsewhere.

    Parameters
    ----------
    mask : np.ndarray, shape (1, H, W), bool
        Region set to ``inside``, e.g. from :mod:`simulator.masks`.
    inside, outside : float
        Values within and outside the mask, e.g. intensities or phases in
        radians.

    Returns
    -------
    np.ndarray, shape (1, H, W), float64
        On the device of ``mask``.

    Raises
    ------
    ValueError
        If ``mask`` is not a boolean ``(1, H, W)`` array.
    """
    check_shape("mask", mask, (1, None, None))
    if mask.dtype != bool:
        raise ValueError(f"mask must be boolean, got dtype {mask.dtype}")
    xp = get_array_module(mask)
    return xp.where(mask, float(inside), float(outside))
