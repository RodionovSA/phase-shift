# src/phase_shift/backend.py
"""NumPy/CuPy array-module dispatch and device placement."""

from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

try:
    import cupy as _cp
except ImportError:  # pragma: no cover
    _cp = None

CUPY_AVAILABLE = _cp is not None


def get_array_module(*arrays: np.ndarray) -> ModuleType:
    """Return ``numpy`` or ``cupy``, whichever module ``arrays`` live on.

    Returns ``numpy`` when CuPy is not installed. Arrays must all live on one
    device.
    """
    if CUPY_AVAILABLE:
        return _cp.get_array_module(*arrays)
    return np


def to_device(x: ArrayLike, device: str = "auto", dtype: DTypeLike = None) -> np.ndarray:
    """Move ``x`` to the requested device.

    Parameters
    ----------
    x : array-like
        Data to move.
    device : {"auto", "cpu", "cuda"}, default "auto"
        ``"cpu"`` returns a NumPy array, ``"cuda"`` a CuPy array. ``"auto"``
        uses CuPy when installed, NumPy otherwise.
    dtype : dtype, optional
        Cast while moving.

    Returns
    -------
    ndarray
        NumPy or CuPy array.

    Raises
    ------
    RuntimeError
        If ``device="cuda"`` and CuPy is not installed.
    ValueError
        If ``device`` is not one of the options above.
    """
    if device == "cpu":
        xp = np
    elif device == "cuda":
        if not CUPY_AVAILABLE:
            raise RuntimeError("device='cuda' requested but cupy is not installed")
        xp = _cp
    elif device == "auto":
        xp = _cp if CUPY_AVAILABLE else np
    else:
        raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {device!r}")

    return xp.asarray(x, dtype=dtype) if dtype is not None else xp.asarray(x)


def asnumpy(x: ArrayLike) -> np.ndarray:
    """Return ``x`` as a NumPy array, copying from the GPU if needed."""
    if CUPY_AVAILABLE and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return np.asarray(x)


def default_dtype(xp: ModuleType, complex_: bool = False) -> type[np.generic]:
    """Return the working dtype for large arrays: ``float32`` or ``complex64``.

    Parameters
    ----------
    xp : module
        ``numpy`` or ``cupy``.
    complex_ : bool, default False
        Return the complex dtype.
    """
    return xp.complex64 if complex_ else xp.float32
