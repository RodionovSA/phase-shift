# src/phase_shift/basis.py
"""Spatial basis families for the phase-step error field.

The field is expanded as ``Delta_n = sum_j c_jn * p_j``,
``docs/interference_model.md`` Eq. (9b) and ``docs/sf_aia.md`` Eq. (T1). Every
family is centered and orthonormalized here, so each basis function has zero
spatial mean (Eq. 9a, Eq. T3) whatever the family. ``docs/vp_aia.md`` Eq. (13)
uses the same functions as its modes ``H_j``.

Add a family by registering one builder of its raw functions in
:data:`BASIS_REGISTRY`; the conventions are applied by :func:`spatial_basis`.
"""

from functools import lru_cache
from types import ModuleType
from typing import Callable

import numpy as np


def _coords(H: int, W: int, xp: ModuleType) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates centered on the field and scaled to about ``[-1, 1]``.

    The convention of ``docs/sf_aia.md`` §1.2(i).

    Parameters
    ----------
    H, W : int
        Frame height and width.
    xp : module
        ``numpy`` or ``cupy``.

    Returns
    -------
    x, y : np.ndarray, shape (H*W,), float64
        Flattened row-major, matching the pixel flattening of the stack.
    """
    yy, xx = xp.meshgrid(xp.arange(H, dtype=xp.float64), xp.arange(W, dtype=xp.float64),
                         indexing="ij")
    x = ((xx - xx.mean()) / max(W / 2.0, 1.0)).ravel()
    y = ((yy - yy.mean()) / max(H / 2.0, 1.0)).ravel()
    return x, y


def _poly_terms(H: int, W: int, xp: ModuleType, degree: int = 1) -> np.ndarray:
    """Monomials of total degree 1 through ``degree``, before centering.

    ``docs/sf_aia.md`` Eq. (T1): ``x, y, x^2, xy, y^2, ...`` in ascending
    degree, so lower orders are fixed before higher ones build on them.
    Degree 0 is excluded, being the piston.

    Parameters
    ----------
    H, W : int
        Frame height and width.
    xp : module
        ``numpy`` or ``cupy``.
    degree : int, default 1
        Highest total degree ``M``. ``0`` gives an empty basis, the piston
        model.

    Returns
    -------
    np.ndarray, shape (J, H*W), float64
        ``J = (degree+1)*(degree+2)//2 - 1`` raw rows.

    Raises
    ------
    ValueError
        If ``degree`` is negative.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    exponents = [(d - i, i) for d in range(1, degree + 1) for i in range(d + 1)]
    x, y = _coords(H, W, xp)
    rows = xp.empty((len(exponents), H * W), dtype=xp.float64)
    for j, (ex, ey) in enumerate(exponents):
        rows[j] = (x ** ex) * (y ** ey)
    return rows


BASIS_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "poly": _poly_terms,
}

BASES = list(BASIS_REGISTRY)


def _centered_orthonormal(rows: np.ndarray, xp: ModuleType) -> np.ndarray:
    """Center and orthonormalize ``rows`` in place, in order.

    Subtracts each row's spatial mean (``docs/interference_model.md`` Eq. 9a,
    ``docs/sf_aia.md`` Eq. T3), then applies modified Gram-Schmidt against the
    rows already fixed, so a truncation of the basis is itself a basis.

    Parameters
    ----------
    rows : np.ndarray, shape (J, P), float64
        Raw basis functions; overwritten.
    xp : module
        ``numpy`` or ``cupy``, matching ``rows``.

    Returns
    -------
    np.ndarray, shape (J, P), float64
        The same array, centered and orthonormalized.
    """
    for j in range(rows.shape[0]):
        col = rows[j] - rows[j].mean()
        for k in range(j):
            col = col - (col @ rows[k]) * rows[k]
        rows[j] = col / float(xp.sqrt(xp.sum(col * col)))
    return rows


@lru_cache(maxsize=32)
def spatial_basis(H: int, W: int, kind: str = "poly", xp: ModuleType = np,
                  **kwargs) -> np.ndarray:
    """Build an orthonormal, zero-spatial-mean basis on an ``(H, W)`` field.

    Cached per argument set, since a solve rebuilds the same basis every
    refinement round.

    Parameters
    ----------
    H, W : int
        Frame height and width.
    kind : str, default "poly"
        Basis family, one of :data:`BASES`.
    xp : module, default numpy
        ``numpy`` or ``cupy``.
    **kwargs
        Passed to the family builder, e.g. ``degree`` for ``"poly"``.

    Returns
    -------
    np.ndarray, shape (J, H*W), float64
        Basis functions, flattened row-major. ``J`` depends on the family and
        its arguments, and may be 0.

    Raises
    ------
    ValueError
        If ``kind`` is not registered, or the family rejects ``kwargs``.
    """
    if kind not in BASIS_REGISTRY:
        raise ValueError(f"unknown basis {kind!r}, expected one of {BASES}")
    return _centered_orthonormal(BASIS_REGISTRY[kind](H, W, xp, **kwargs), xp)
