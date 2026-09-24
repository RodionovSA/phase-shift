# src/simulator/basis.py
"""Polynomial modes of :func:`phase_shift.basis.spatial_basis`, sized by their count."""

from types import ModuleType

import numpy as np

from phase_shift.basis import spatial_basis


def poly_degree(L: int) -> int:
    """Degree ``M`` with ``(M+1)(M+2)/2 - 1 = L`` modes.

    Raises
    ------
    ValueError
        If no degree has ``L`` modes.
    """
    M = 0
    while (M + 1) * (M + 2) // 2 - 1 < L:
        M += 1
    if (M + 1) * (M + 2) // 2 - 1 != L:
        raise ValueError(
            f"number of coefficients must be (M+1)(M+2)/2 - 1 (0, 2, 5, 9, 14, ...), got {L}"
        )
    return M


def poly_modes(H: int, W: int, L: int, xp: ModuleType) -> np.ndarray:
    """First ``L`` polynomial modes, zero-mean and orthonormal over the field.

    Returns
    -------
    np.ndarray, shape (L, H*W), float64
        Unit-norm rows of :func:`phase_shift.basis.spatial_basis`.

    Raises
    ------
    ValueError
        If no degree has ``L`` modes.
    """
    return spatial_basis(H, W, "poly", xp, "double", degree=poly_degree(L))
