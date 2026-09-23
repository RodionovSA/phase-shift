# src/phase_shift/methods/steps.py
"""Alternating least-squares steps of ``docs/aia.md``.

The pixel step solves every pixel's ``(a, u, v)`` with the phase steps fixed
(§"Pixel step", Eq. 5-6); the frame step solves every frame's
``(c_n, P_n, Q_n)`` with the fields fixed (§"Frame step", Eq. 7-9). Alternating
them is AIA; VP-AIA reuses both.
"""

from types import ModuleType

import numpy as np

from ..backend import Precision, get_array_module


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
               precision: str | Precision | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. The
        returned fields are in ``precision.work``; the design matrix and its
        pseudoinverse are float64 regardless.

    Returns
    -------
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components, in ``precision.work``.

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
    p = Precision.of(precision)
    delta = xp.asarray(delta, dtype=xp.float64)
    g = xp.ones(N, dtype=xp.float64) if g is None else xp.asarray(g, dtype=xp.float64)

    A = pixel_design(delta, g, xp)                                  # (N, 3) float64
    X = xp.linalg.pinv(A).astype(p.work) @ stack                    # (3, P)
    return X[0], X[1], X[2]


def frame_step(stack: np.ndarray, u: np.ndarray, v: np.ndarray,
               precision: str | Precision | None = None
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
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. Only
        ``precision.accum`` matters here, as the dtype ``u`` and ``v`` are
        cast to for the one reduction over the full stack.

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
    p = Precision.of(precision)

    # Scalar pixel sums: accumulated in precision.accum, so no (P,)-sized copy
    # of u or v is made.
    Su, Sv = float(xp.sum(u, dtype=p.accum)), float(xp.sum(v, dtype=p.accum))
    Suu = float(xp.sum(u * u, dtype=p.accum))
    Svv = float(xp.sum(v * v, dtype=p.accum))
    Suv = float(xp.sum(u * v, dtype=p.accum))
    BtB = xp.asarray([[float(P), Su, Sv], [Su, Suu, Suv], [Sv, Suv, Svv]])

    # The one reduction over the full stack, and so the one place
    # precision.accum changes the memory cost.
    u_mm = xp.asarray(u, dtype=p.accum)
    v_mm = xp.asarray(v, dtype=p.accum)
    IB = xp.stack([xp.sum(stack, axis=1, dtype=p.accum).astype(xp.float64),
                   (stack @ u_mm).astype(xp.float64),
                   (stack @ v_mm).astype(xp.float64)], axis=1)       # (N, 3)

    x = xp.linalg.solve(BtB, IB.T)                                   # (3, N)
    c, Pn, Qn = x[0], x[1], x[2]
    delta = xp.arctan2(Qn, Pn)
    g = xp.sqrt(Pn * Pn + Qn * Qn)
    return delta, g, c
