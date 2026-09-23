# src/phase_shift/methods/gauge.py
"""Gauge conventions of ``docs/gauge_conventions.md`` applied by the solves."""

from types import ModuleType

import numpy as np

from ..backend import Precision


def whitening_matrix(u: np.ndarray, v: np.ndarray, xp: ModuleType,
                     precision: str | Precision | None = None) -> np.ndarray:
    """Return the symmetric 2x2 transform that whitens ``(u, v)``.

    ``T = s * G^(-1/2)``, with ``G = [[sum u^2, sum uv], [sum uv, sum v^2]]``
    and ``s = sqrt((sum u^2 + sum v^2) / 2)``: applying ``T`` to ``(u, v)``
    imposes ``sum(u**2) == sum(v**2)`` and ``sum(u*v) == 0`` while preserving
    their total energy, ``docs/aia.md`` Eq. (15) and ``docs/vp_aia.md``
    §"Normalization" step 2.

    The model is unchanged by applying ``T`` to ``(u, v)`` and ``T^-1`` to
    ``(P_n, Q_n)``, ``docs/vp_aia.md`` §"Quadrature frame". A caller holding no
    ``(P_n, Q_n)`` to transform wants :func:`whiten_uv` instead.

    Parameters
    ----------
    u, v : np.ndarray, shape (P,)
        Quadrature components.
    xp : module
        ``numpy`` or ``cupy``, matching ``u``/``v``.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. The three
        pixel sums accumulate in ``precision.accum``.

    Returns
    -------
    np.ndarray, shape (2, 2), float64
        Symmetric; a host NumPy array whatever ``xp`` is.
    """
    acc = Precision.of(precision).accum
    Suu = float(xp.sum(u * u, dtype=acc))
    Svv = float(xp.sum(v * v, dtype=acc))
    Suv = float(xp.sum(u * v, dtype=acc))
    eps = np.finfo(float).eps
    scale = np.sqrt(max((Suu + Svv) / 2, eps))

    G = np.array([[Suu, Suv], [Suv, Svv]])
    w, V = np.linalg.eigh(G)
    w = np.maximum(w, eps)
    return (V * (scale / np.sqrt(w))) @ V.T           # symmetric, scale * G^(-1/2)


def whiten_uv(u: np.ndarray, v: np.ndarray, xp: ModuleType,
              precision: str | Precision | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotate and shear ``(u, v)`` to equal energy and zero correlation.

    Imposes ``sum(u**2) == sum(v**2)`` and ``sum(u*v) == 0``, ``docs/aia.md``
    Eq. (15), fixing the shear and anisotropic scaling of the quadrature-basis
    freedom Eq. (13). Applied after every pixel step that fits the gain, which
    refits ``(P_n, Q_n)`` afterwards rather than transforming them.

    Parameters
    ----------
    u, v : np.ndarray, shape (P,)
        Quadrature components.
    xp : module
        ``numpy`` or ``cupy``, matching ``u``/``v``.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`.

    Returns
    -------
    u, v : np.ndarray, shape (P,)
        Whitened components, in the input dtype.
    """
    T = whitening_matrix(u, v, xp, precision)
    m00, m01, m11 = float(T[0, 0]), float(T[0, 1]), float(T[1, 1])
    return u * m00 + v * m01, u * m01 + v * m11


def pin_phase_origin(delta: np.ndarray) -> np.ndarray:
    """Re-reference the phase steps so ``delta[0] = 0``.

    ``docs/aia.md`` §"Phase-origin convention": a constant added to every
    ``delta_n`` and subtracted from ``Phi`` leaves the model unchanged.

    Parameters
    ----------
    delta : np.ndarray, shape (N,)
        Per-frame phase step, in radians.

    Returns
    -------
    np.ndarray, shape (N,)
    """
    return delta - delta[0]


def normalize_gain(g: np.ndarray, xp: ModuleType) -> np.ndarray:
    """Scale the per-frame gain to ``median(g) = 1``.

    ``docs/aia.md`` Eq. (16), fixing the remaining scale of the
    quadrature-basis freedom.

    Parameters
    ----------
    g : np.ndarray, shape (N,)
        Per-frame fringe gain.
    xp : module
        ``numpy`` or ``cupy``, matching ``g``.

    Returns
    -------
    np.ndarray, shape (N,)
    """
    return g / max(float(xp.median(g)), np.finfo(float).eps)


def center_offsets(c: np.ndarray, xp: ModuleType) -> np.ndarray:
    """Center the per-frame offsets to ``mean(c) = 0``.

    ``docs/aia.md`` Eq. (16), fixing the constant shared between the
    background field and the per-frame offset.

    Parameters
    ----------
    c : np.ndarray, shape (N,)
        Per-frame offset.
    xp : module
        ``numpy`` or ``cupy``, matching ``c``.

    Returns
    -------
    np.ndarray, shape (N,)
    """
    return c - xp.mean(c)


def normalize_quadrature_frame(a: np.ndarray, u: np.ndarray, v: np.ndarray,
                               P_n: np.ndarray, Q_n: np.ndarray, xp: ModuleType,
                               precision: str | Precision | None = None
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """Put a quadrature solution into the conventional frame.

    Applies ``docs/vp_aia.md`` §"Normalization" steps 1-4 in order -- shift,
    whitening, phase origin, contrast scale -- each preserving the conditions
    the previous ones established. Afterwards the four conditions of
    §"Quadrature frame" hold, together with ``delta[0] = 0`` and
    ``median(g) = 1``::

        sum((a - mean(a)) * u) == sum((a - mean(a)) * v) == 0
        sum(u**2) == sum(v**2),  sum(u*v) == 0

    Every step is a reparametrization of ``docs/vp_aia.md`` Eq. (3), so
    ``a + P_n u + Q_n v`` is unchanged.

    Parameters
    ----------
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature fields, flattened.
    P_n, Q_n : np.ndarray, shape (N,)
        Per-frame quadrature coefficients, ``docs/vp_aia.md`` Eq. (3):
        ``P_n = g_n cos(delta_n)``, ``Q_n = g_n sin(delta_n)``.
    xp : module
        ``numpy`` or ``cupy``, matching the inputs.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. The
        pixel sums accumulate in ``precision.accum``; the 2x2 solves are
        float64.

    Returns
    -------
    a, u, v : np.ndarray, shape (P,)
        Normalized fields, in the input dtype.
    P_n, Q_n : np.ndarray, shape (N,), float64
        Normalized per-frame coefficients.
    """
    acc = Precision.of(precision).accum
    eps = np.finfo(float).eps
    P = u.shape[0]
    P_n = xp.asarray(P_n, dtype=xp.float64)
    Q_n = xp.asarray(Q_n, dtype=xp.float64)

    # Step 1, shift: move the part of a that correlates with the quadratures
    # into (P_n, Q_n). Raw moments, so no centered (P,)-sized copy is made.
    Su = float(xp.sum(u, dtype=acc))
    Sv = float(xp.sum(v, dtype=acc))
    Sa = float(xp.sum(a, dtype=acc))
    Suu = float(xp.sum(u * u, dtype=acc))
    Svv = float(xp.sum(v * v, dtype=acc))
    Suv = float(xp.sum(u * v, dtype=acc))
    Sau = float(xp.sum(a * u, dtype=acc))
    Sav = float(xp.sum(a * v, dtype=acc))
    Cuv = Suv - Su * Sv / P
    C = np.array([[Suu - Su * Su / P, Cuv], [Cuv, Svv - Sv * Sv / P]])
    rhs = np.array([Sau - Sa * Su / P, Sav - Sa * Sv / P])
    p, q = (np.linalg.pinv(C) @ rhs).tolist()
    a = a - p * u - q * v
    P_n = P_n + p
    Q_n = Q_n + q

    # Step 2, whitening: T on (u, v), its inverse on (P_n, Q_n), which leaves
    # the model unchanged (docs/vp_aia.md §"Quadrature frame").
    T = whitening_matrix(u, v, xp, precision)
    Ti = np.linalg.inv(T)
    u, v = (u * float(T[0, 0]) + v * float(T[0, 1]),
            u * float(T[1, 0]) + v * float(T[1, 1]))
    P_n, Q_n = (P_n * float(Ti[0, 0]) + Q_n * float(Ti[1, 0]),
                P_n * float(Ti[0, 1]) + Q_n * float(Ti[1, 1]))

    # Step 3, phase origin: one rotation applied to both pairs.
    d0 = float(xp.arctan2(Q_n[0], P_n[0]))
    # Python floats, not NumPy scalars: a np.float64 scalar is not weak under
    # NEP 50 and would promote the working-dtype fields.
    c, s = float(np.cos(d0)), float(np.sin(d0))
    P_n, Q_n = P_n * c + Q_n * s, -P_n * s + Q_n * c
    u, v = u * c + v * s, -u * s + v * c

    # Step 4, contrast scale: median(g) = 1.
    scale = max(float(xp.median(xp.sqrt(P_n * P_n + Q_n * Q_n))), eps)
    return a, u * scale, v * scale, P_n / scale, Q_n / scale
