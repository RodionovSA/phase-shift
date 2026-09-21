# src/phase_shift/backend.py
"""NumPy/CuPy array-module dispatch, device placement, and working precision."""

from dataclasses import dataclass
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

try:
    import cupy as _cp
except ImportError:  # pragma: no cover
    _cp = None

CUPY_AVAILABLE = _cp is not None

PRECISIONS = ("single", "double", "fast")

_PRESETS = {
    "single": (np.float32, np.float64),
    "double": (np.float64, np.float64),
    "fast": (np.float32, np.float32),
}


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


@dataclass(frozen=True)
class Precision:
    """Dtypes a solve runs in.

    Small linear algebra, ``(N,)`` vectors, and scalar reductions stay float64
    whatever the precision; only the two dtypes below vary. Build one from a
    preset name with :meth:`of`, or from a pair of dtypes directly.

    Attributes
    ----------
    work : np.dtype
        Dtype of the large ``(N, H, W)`` / ``(N, P)`` arrays and of the fields
        recovered from them.
    accum : np.dtype
        Dtype the operands of reductions and matrix products over the full
        stack are cast to. Above ``work`` it costs a temporary copy of the
        stack and buys back the precision a float32 sum over ``P`` pixels
        loses.
    """

    work: np.dtype
    accum: np.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "work", np.dtype(self.work))
        object.__setattr__(self, "accum", np.dtype(self.accum))

    @classmethod
    def of(cls, precision: "str | Precision | None" = None) -> "Precision":
        """Resolve a preset name, a :class:`Precision`, or None to a precision.

        Parameters
        ----------
        precision : str or Precision, optional
            One of :data:`PRECISIONS`, an explicit :class:`Precision`, or None
            for the package default, :func:`get_precision`.

        Returns
        -------
        Precision

        Raises
        ------
        ValueError
            If ``precision`` is neither a registered preset name nor a
            :class:`Precision`.
        """
        if precision is None:
            return _precision
        if isinstance(precision, cls):
            return precision
        if isinstance(precision, str) and precision in _PRESETS:
            return cls(*_PRESETS[precision])
        raise ValueError(
            f"unknown precision {precision!r}, expected one of {PRECISIONS} "
            f"or a Precision instance"
        )


_precision = Precision.of("single")


def get_precision() -> Precision:
    """Return the package-wide default precision, ``"single"`` unless set."""
    return _precision


def set_precision(precision: "str | Precision") -> Precision:
    """Set the package-wide default precision.

    Applies to every function and solver built afterwards that is not given an
    explicit ``precision``. A :class:`phase_shift.solver.PhaseSolver` resolves
    it once, at construction.

    Parameters
    ----------
    precision : str or Precision
        One of :data:`PRECISIONS`, or an explicit :class:`Precision`.

    Returns
    -------
    Precision
        The resolved precision now in effect.

    Raises
    ------
    ValueError
        If ``precision`` is not recognized.
    """
    global _precision
    if precision is None:
        raise ValueError(f"precision must be one of {PRECISIONS} or a Precision, got None")
    _precision = Precision.of(precision)
    return _precision
