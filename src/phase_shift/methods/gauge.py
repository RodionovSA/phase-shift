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


def center_coeffs(coeffs: np.ndarray) -> np.ndarray:
    """Center each basis term's coefficients to zero frame mean.

    ``docs/sf_aia.md`` Eq. (T3b)/(E4) and ``docs/interference_model.md``
    Eq. (9b): a coefficient with a nonzero frame mean adds the same static
    pattern to every frame and belongs to the phase, not to the step field.

    Parameters
    ----------
    coeffs : np.ndarray, shape (J, N)
        Per-frame step-field coefficients.

    Returns
    -------
    np.ndarray, shape (J, N)
    """
    return coeffs - coeffs.mean(axis=1, keepdims=True)
