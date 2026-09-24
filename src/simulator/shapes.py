# src/simulator/shapes.py
"""Shape validation for ``(N, H, W)`` input arrays."""

import numpy as np


def check_shape(name: str, x: np.ndarray, expected: tuple[int | None, ...]) -> None:
    """Check that ``x`` has exactly the ``expected`` shape.

    Parameters
    ----------
    name : str
        Argument name, for the error message.
    x : np.ndarray
        Array to check.
    expected : tuple of int or None
        Required shape; ``None`` accepts any size on that axis.

    Raises
    ------
    ValueError
        If ``x`` does not have the ``expected`` shape.
    """
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) != len(expected) or any(
        e is not None and s != e for s, e in zip(shape, expected)
    ):
        want = "(" + ", ".join("any" if e is None else str(e) for e in expected) + ")"
        got = shape if shape is not None else type(x).__name__
        raise ValueError(f"{name} must have shape {want}, got {got}")


def check_shapes(**arrays: np.ndarray) -> tuple[int, int, int]:
    """Check that arrays are 3-D and broadcast together.

    Parameters
    ----------
    **arrays : np.ndarray, 3-D
        Named arrays with axes ``(N, H, W)``; an axis of size 1 broadcasts.

    Returns
    -------
    tuple of int
        The broadcast shape ``(N, H, W)``.

    Raises
    ------
    ValueError
        If an array is not 3-D, or the shapes do not broadcast together.
    """
    for name, x in arrays.items():
        if getattr(x, "ndim", None) != 3:
            raise ValueError(
                f"{name} must be a 3-D array of shape (N, H, W) with size-1 axes allowed, "
                f"got {getattr(x, 'shape', type(x).__name__)}"
            )
    shapes = [x.shape for x in arrays.values()]
    try:
        return np.broadcast_shapes(*shapes)
    except ValueError:
        raise ValueError(
            f"{', '.join(arrays)} shapes do not broadcast: {', '.join(map(str, shapes))}"
        ) from None
