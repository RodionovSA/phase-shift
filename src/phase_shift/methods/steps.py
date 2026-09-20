# src/phase_shift/methods/steps.py
"""Alternating least-squares steps of ``docs/aia.md``.

The pixel step solves every pixel's ``(a, u, v)`` with the phase steps fixed
(§"Pixel step", Eq. 5-6); the frame step solves every frame's
``(c_n, P_n, Q_n)`` with the fields fixed (§"Frame step", Eq. 7-9). Alternating
them is AIA; SF-AIA and VP-AIA reuse both.
"""

from types import ModuleType

import numpy as np
from numpy.typing import DTypeLike

from .. import backend as _backend
from ..backend import get_array_module


def pixel_design(delta: np.ndarray, g: np.ndarray, xp: ModuleType) -> np.ndarray:
    """Assemble the pixel-step design ``[1, g*cos(delta), g*sin(delta)]``.

    ``docs/aia.md`` Eq. (6); its columns are ``(1, P_n, Q_n)``.

    Parameters
    ----------
    delta, g : np.ndarray, shape (N,)
        Per-frame phase step, in radians, and fringe gain.
    xp : module
        ``numpy`` or ``cupy``, matching the inputs.

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    return xp.column_stack([xp.ones_like(delta), g * xp.cos(delta), g * xp.sin(delta)])


def pixel_step(stack: np.ndarray, delta: np.ndarray, g: np.ndarray | None = None,
               dtype: DTypeLike = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve every pixel's background and quadrature fields.

    With the phase steps and gains fixed, each pixel is the linear regression
    of ``docs/aia.md`` Eq. (5) on the design of Eq. (6), solved for all pixels
    at once through one shared pseudoinverse. Here ``u = b*cos(phi)`` and
    ``v = -b*sin(phi)``, so recovering ``phi`` needs a phase-step estimate
    too, e.g. from :func:`frame_step`.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P`` pixels each.
    delta : np.ndarray, shape (N,)
        Per-frame phase step, in radians.
    g : np.ndarray, shape (N,), optional
        Per-frame fringe gain. Defaults to all ones.
    dtype : dtype, optional
        Working dtype of the returned fields. Defaults to
        :func:`phase_shift.backend.default_dtype`; the design matrix and its
        pseudoinverse are always float64.

    Returns
    -------
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components, in ``dtype``.

    Raises
    ------
    ValueError
        If ``stack`` is not 2-D, ``delta`` is not 1-D, or ``delta``/``g`` do
        not match ``stack``'s frame count.
    """
    if stack.ndim != 2:
        raise ValueError(f"stack must be 2-D (N, P), got shape {stack.shape}")
    if delta.ndim != 1:
        raise ValueError(f"delta must be 1-D (N,), got shape {delta.shape}")
    if delta.shape[0] != stack.shape[0]:
        raise ValueError(f"delta must have length {stack.shape[0]}, got {delta.shape[0]}")
    if g is not None and len(g) != len(delta):
        raise ValueError(f"g must have length {len(delta)}, got {len(g)}")

    xp = get_array_module(stack)
    N = stack.shape[0]
    work_dtype = dtype if dtype is not None else _backend.default_dtype(xp)
    delta = xp.asarray(delta, dtype=xp.float64)
    g = xp.ones(N, dtype=xp.float64) if g is None else xp.asarray(g, dtype=xp.float64)

    A = pixel_design(delta, g, xp)                                  # (N, 3) float64
    X = xp.linalg.pinv(A).astype(work_dtype) @ stack                # (3, P)
    return X[0], X[1], X[2]


def frame_step(stack: np.ndarray, u: np.ndarray, v: np.ndarray,
               precise_reduce: bool = True
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve every frame's phase step, gain, and offset.

    With ``(u, v)`` fixed, each frame is the linear regression of
    ``docs/aia.md`` Eq. (7) on the design of Eq. (8), giving
    ``delta_n = atan2(Q_n, P_n)`` and ``g_n = hypot(P_n, Q_n)``, Eq. (9). The
    background field is replaced by the scalar offset ``c_n``; §"Frame step",
    "Dropping the background field", shows this is exact up to a bounded,
    frame-independent displacement.

    The returned ``delta`` is not referenced to a phase origin and ``g`` is
    not normalized; see :mod:`phase_shift.methods.gauge`.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P`` pixels each.
    u, v : np.ndarray, shape (P,)
        Quadrature components, e.g. from :func:`pixel_step`.
    precise_reduce : bool, default True
        See :attr:`phase_shift.config.PhaseConfig.precise_reduce`.

    Returns
    -------
    delta : np.ndarray, shape (N,), float64
        Per-frame phase step, in radians.
    g : np.ndarray, shape (N,), float64
        Per-frame fringe gain, unnormalized.
    c : np.ndarray, shape (N,), float64
        Per-frame offset, unnormalized.

    Raises
    ------
    ValueError
        If ``stack`` is not 2-D, ``u``/``v`` are not 1-D, or they do not match
        ``stack``'s pixel count.
    """
    if stack.ndim != 2:
        raise ValueError(f"stack must be 2-D (N, P), got shape {stack.shape}")
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError(f"u and v must be 1-D (P,), got shapes {u.shape} and {v.shape}")
    if len(u) != stack.shape[1] or len(v) != stack.shape[1]:
        raise ValueError(f"u and v must have length {stack.shape[1]}, "
                         f"got {len(u)} and {len(v)}")

    xp = get_array_module(stack, u, v)
    P = stack.shape[1]
    u64 = xp.asarray(u, dtype=xp.float64)
    v64 = xp.asarray(v, dtype=xp.float64)

    # (P,)-sized, so float64 regardless of precise_reduce: never touches stack.
    Su, Sv = float(xp.sum(u64)), float(xp.sum(v64))
    Suu = float(xp.sum(u64 * u64))
    Svv = float(xp.sum(v64 * v64))
    Suv = float(xp.sum(u64 * v64))
    BtB = xp.asarray([[float(P), Su, Sv], [Su, Suu, Suv], [Sv, Suv, Svv]])

    # The one reduction over the full stack, and so the one place
    # precise_reduce changes the memory cost.
    u_mm = u64 if precise_reduce else xp.asarray(u, dtype=stack.dtype)
    v_mm = v64 if precise_reduce else xp.asarray(v, dtype=stack.dtype)
    IB = xp.stack([xp.sum(stack, axis=1).astype(xp.float64),
                   (stack @ u_mm).astype(xp.float64),
                   (stack @ v_mm).astype(xp.float64)], axis=1)       # (N, 3)

    x = xp.linalg.solve(BtB, IB.T)                                   # (3, N)
    c, Pn, Qn = x[0], x[1], x[2]
    delta = xp.arctan2(Qn, Pn)
    g = xp.sqrt(Pn * Pn + Qn * Qn)
    return delta, g, c
