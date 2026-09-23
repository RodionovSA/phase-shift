# src/phase_shift/basis.py
"""Spatial basis families for the phase-step error field.

The field is expanded as ``Delta_n = sum_j c_jn * p_j``,
``docs/interference_model.md`` Eq. (9b) and ``docs/sf_aia.md`` Eq. (T1). Every
family is centered and orthonormalized here, so each basis function has zero
spatial mean (Eq. 9a, Eq. T3) whatever the family. ``docs/vp_aia.md`` Eq. (6)
uses the same functions as its modes ``H_j``.

Add a family by registering one builder of its raw functions in
:data:`BASIS_REGISTRY`; the conventions are applied by :func:`spatial_basis`.
"""

from functools import lru_cache
from types import ModuleType
from typing import Callable

import numpy as np

from .backend import Precision


def _coords(H: int, W: int, xp: ModuleType) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates centered on the field and scaled to about ``[-1, 1]``.

    The convention of ``docs/sf_aia.md`` §"Algorithm" step 2.

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


def _centered_orthonormal(rows: np.ndarray, xp: ModuleType, context: str) -> np.ndarray:
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
    context : str
        Field shape and family arguments, for the error message.

    Returns
    -------
    np.ndarray, shape (J, P), float64
        The same array, centered and orthonormalized.

    Raises
    ------
    ValueError
        If a row is constant over the field, or a combination of the rows
        before it, and so cannot be normalized.
    """
    for j in range(rows.shape[0]):
        raw_norm = float(xp.sqrt(xp.sum(rows[j] * rows[j])))
        col = rows[j] - rows[j].mean()
        for k in range(j):
            col = col - (col @ rows[k]) * rows[k]
        norm = float(xp.sqrt(xp.sum(col * col)))
        if norm <= 1e-8 * max(raw_norm, 1.0):
            raise ValueError(
                f"{context}: basis function {j} is constant over the field, or a "
                f"combination of the functions before it, so it cannot be normalized; "
                f"the field is too small or too thin for this family"
            )
        rows[j] = col / norm
    return rows


@lru_cache(maxsize=32)
def spatial_basis(H: int, W: int, kind: str = "poly", xp: ModuleType = np,
                  precision: "str | Precision | None" = None, **kwargs) -> np.ndarray:
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
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. The rows
        are returned in ``precision.accum``, the dtype they are multiplied
        against the stack in. Centering and orthonormalization run in float64
        whatever the precision: they are a one-off, cached, and modified
        Gram-Schmidt in float32 loses orthogonality by about ``1e-5`` over a
        megapixel field.
    **kwargs
        Passed to the family builder, e.g. ``degree`` for ``"poly"``.

    Returns
    -------
    np.ndarray, shape (J, H*W)
        Basis functions in ``precision.accum``, flattened row-major. ``J``
        depends on the family and its arguments, and may be 0.

    Raises
    ------
    ValueError
        If ``kind`` is not registered, if the family rejects ``kwargs``, or if
        the field is too small for the family to produce independent
        functions on it.
    """
    if kind not in BASIS_REGISTRY:
        raise ValueError(f"unknown basis {kind!r}, expected one of {BASES}")
    context = f"basis {kind!r} with {kwargs} on a {H}x{W} field"
    rows = _centered_orthonormal(BASIS_REGISTRY[kind](H, W, xp, **kwargs), xp, context)
    return rows.astype(Precision.of(precision).accum, copy=False)
