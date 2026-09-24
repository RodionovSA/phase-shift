# src/simulator/masks.py
"""Hard-edged boolean masks on an ``(H, W)`` field.

Combine masks with ``|``, ``&``, and ``~``.
"""

import numpy as np

from .coords import pixel_coords


def circle(H: int, W: int, radius: float, center: tuple[float, float] | None = None,
           device: str = "cpu") -> np.ndarray:
    """Mask of the pixels within ``radius`` of ``center``.

    Parameters
    ----------
    H, W : int
        Field height and width.
    radius : float
        Radius in pixels, positive.
    center : (float, float), optional
        Center ``(x0, y0)`` in pixels from the top-left pixel. None means the
        field center.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned mask.

    Returns
    -------
    np.ndarray, shape (1, H, W), bool

    Raises
    ------
    ValueError
        If ``radius`` is not positive.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    x, y = pixel_coords(H, W, center, device)
    return x ** 2 + y ** 2 <= radius ** 2


def rectangle(H: int, W: int, width: float, height: float,
              center: tuple[float, float] | None = None, device: str = "cpu") -> np.ndarray:
    """Mask of an axis-aligned rectangle around ``center``.

    Parameters
    ----------
    H, W : int
        Field height and width.
    width, height : float
        Rectangle size along ``x`` and ``y`` in pixels, positive.
    center : (float, float), optional
        Center ``(x0, y0)`` in pixels from the top-left pixel. None means the
        field center.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned mask.

    Returns
    -------
    np.ndarray, shape (1, H, W), bool

    Raises
    ------
    ValueError
        If ``width`` or ``height`` is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}, {height}")
    x, y = pixel_coords(H, W, center, device)
    return (abs(x) <= width / 2) & (abs(y) <= height / 2)
