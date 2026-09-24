# src/simulator/coords.py
"""Pixel coordinates of an ``(H, W)`` field."""

import numpy as np

from phase_shift.backend import to_device


def pixel_coords(H: int, W: int, center: tuple[float, float] | None = None,
                 device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates relative to ``center``.

    ``x`` is the column index and ``y`` the row index, from the top-left pixel.

    Parameters
    ----------
    H, W : int
        Field height and width, positive.
    center : (float, float), optional
        Origin ``(x0, y0)`` in pixels. None means the field center,
        ``((W - 1) / 2, (H - 1) / 2)``.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned arrays.

    Returns
    -------
    x : np.ndarray, shape (1, 1, W), float64
    y : np.ndarray, shape (1, H, 1), float64

    Raises
    ------
    ValueError
        If ``H`` or ``W`` is not positive.
    """
    if H < 1 or W < 1:
        raise ValueError(f"H and W must be positive, got H={H}, W={W}")
    x0, y0 = ((W - 1) / 2, (H - 1) / 2) if center is None else center
    x = to_device(np.arange(W, dtype=np.float64) - x0, device)
    y = to_device(np.arange(H, dtype=np.float64) - y0, device)
    return x.reshape(1, 1, W), y.reshape(1, H, 1)
