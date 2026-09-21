# src/phase_shift/methods/diagnostics.py
"""Accuracy diagnostics of ``docs/aia.md`` §"Accuracy diagnostics"."""

import warnings
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from ..backend import Precision
from ..errors import aia_phi_error_parts
from ..utils import format_value
from .base import MethodParam
from .steps import pixel_design


@dataclass
class AIAParam(MethodParam):
    """Diagnostics of an AIA solve, carried on ``PhaseResult.method_param``.

    Attributes
    ----------
    kappa_p : float
        Condition number of the pixel-step normal matrix ``A_p``. Above about
        20 the solve is unreliable; use more, more evenly spaced phase shifts.
    kappa_ps : float
        Condition number of the normalized frame-step design, bounded below by
        2. Large values mean the field spans less than about one fringe, so
        the frame step cannot separate ``delta_n`` from noise even when the
        loop reports convergence.
    predicted_rms : float
        Predicted RMS phase error, in radians, ``docs/aia.md`` Eq. (11). An
        empirical fit; §"Phase-error covariance" gives the exact expression,
        reported as :attr:`phase_shift.result.PhaseResult.phi_error`.
    iters_run : int
        Number of alternating least-squares iterations run.
    converged : bool
        Whether the loop stopped on ``tol`` rather than exhausting ``iters``.
    g_fit : np.ndarray, shape (N,)
        Per-frame fringe gain that ``(a, u, v)`` were fit against, one
        iteration behind the returned ``g`` when ``fit_gain`` is set.
    c_fit : np.ndarray, shape (N,)
        Per-frame residual offset that ``(a, u, v)`` were fit against, with
        ``mean(c_fit) = 0``. All zero when ``fit_gain=False``; large values
        mean ``PhaseConfig.use_alpha`` has not removed the brightness drift.
    g_min_ratio : float
        ``min(g_fit) / median(g_fit)``. Small values flag a frame nearly
        uncorrelated with the fringe pattern, whose ``delta_n`` is poorly
        determined.
    precision : Precision
        Dtypes the solve ran in, carried so :meth:`phi_error` reproduces them.
    """

    kappa_p: float
    kappa_ps: float
    predicted_rms: float
    iters_run: int
    converged: bool
    g_fit: np.ndarray
    c_fit: np.ndarray
    g_min_ratio: float
    precision: Precision

    def phi_error(self, b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                  fit_gain: bool, noise_std: np.ndarray, simplified: bool,
                  xp: ModuleType) -> np.ndarray:
        """Return ``sigma_Phi`` of ``docs/aia.md`` §"Phase-error covariance".

        See :meth:`phase_shift.methods.base.MethodParam.phi_error` for the
        arguments.

        Returns
        -------
        np.ndarray, shape (H, W)
            Per-pixel phase standard deviation, in radians.
        """
        phi_var, _ = aia_phi_error_parts(b, phi, delta, g, fit_gain, noise_std,
                                         simplified, xp, precision=self.precision)
        return xp.sqrt(phi_var)

    def print_summary(self) -> None:
        """Print convergence, both condition numbers, and the gain ratio."""
        print(f"converged:     {format_value(self.converged)}")
        print(f"kappa_p:       {format_value(self.kappa_p)}")
        print(f"kappa_ps:      {format_value(self.kappa_ps)}")
        print(f"predicted_rms: {format_value(self.predicted_rms)}")
        print(f"g_min_ratio:   {format_value(self.g_min_ratio)}")


def cond2(M: np.ndarray, xp: ModuleType) -> np.ndarray:
    """2-norm condition number of a square matrix, or of a batch of them.

    Parameters
    ----------
    M : np.ndarray, shape (k, k) or (..., k, k)
        Matrix or batch of matrices.
    xp : module
        ``numpy`` or ``cupy``, matching ``M``.

    Returns
    -------
    np.ndarray
        Condition number per matrix, ``inf`` where the matrix is singular;
        0-d for a single matrix.
    """
    s = xp.linalg.svd(M, compute_uv=False)
    smin = s.min(axis=-1)
    smax = s.max(axis=-1)
    return xp.where(smin > 0, smax / xp.where(smin > 0, smin, 1.0), xp.inf)


def chunked_sigma(I: np.ndarray, A: np.ndarray, X: np.ndarray, xp: ModuleType,
                  c: np.ndarray | None = None, chunk: int = 1_000_000,
                  precision: str | Precision | None = None) -> float:
    """RMS of the pixel-step residual ``I - c - A @ X``, over pixel chunks.

    The residual is accumulated chunk by chunk in ``precision.accum``, so no
    second ``(N, P)`` array is held.

    Parameters
    ----------
    I : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P`` pixels each.
    A : np.ndarray, shape (N, 3), float64
        Pixel-step design matrix.
    X : np.ndarray, shape (3, P)
        Fitted ``(a, u, v)``, in ``I``'s dtype.
    xp : module
        ``numpy`` or ``cupy``, matching the inputs.
    c : np.ndarray, shape (N,), optional
        Per-frame offset to subtract; ``None`` leaves ``I`` as is.
    chunk : int, default 1_000_000
        Pixels reduced per chunk; bounds memory, not the result.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`.

    Returns
    -------
    float
        Residual RMS, in ``I``'s units.
    """
    acc = Precision.of(precision).accum
    N, P = I.shape
    A_work = A.astype(I.dtype)
    c_work = None if c is None else c.astype(I.dtype)[:, None]
    ssq = 0.0
    for s in range(0, P, chunk):
        resid = I[:, s:s + chunk] - A_work @ X[:, s:s + chunk]
        if c_work is not None:
            resid = resid - c_work
        ssq += float(xp.sum(resid * resid, dtype=acc))
    return float(np.sqrt(ssq / (N * P)))


def aia_diagnostics(I: np.ndarray, delta_fit: np.ndarray, g: np.ndarray, a: np.ndarray,
                    u: np.ndarray, v: np.ndarray, N: int, xp: ModuleType, iters_run: int,
                    converged: bool, c: np.ndarray | None = None,
                    precision: str | Precision | None = None) -> AIAParam:
    """Measure the accuracy diagnostics of a solved ``(a, u, v)``.

    Computes ``kappa_p``, ``kappa_ps`` and ``predicted_rms`` of
    ``docs/aia.md`` §"Accuracy diagnostics" against the phase steps and gains
    that ``(a, u, v)`` were fit with.

    Parameters
    ----------
    I : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P`` pixels each.
    delta_fit, g : np.ndarray, shape (N,)
        Per-frame phase step, in radians, and fringe gain that ``(a, u, v)``
        were fit against.
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components.
    N : int
        Number of frames.
    xp : module
        ``numpy`` or ``cupy``, matching the inputs.
    iters_run : int
        Value to report as :attr:`AIAParam.iters_run`.
    converged : bool
        Value to report as :attr:`AIAParam.converged`.
    c : np.ndarray, shape (N,), optional
        Per-frame offset from the joint-gain frame step, reported as
        :attr:`AIAParam.c_fit` and removed before measuring the residual.
        ``None`` reports an all-zero ``c_fit``.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`.

    Returns
    -------
    AIAParam

    Warns
    -----
    UserWarning
        If ``kappa_p`` or ``kappa_ps`` exceeds 20, or ``g_min_ratio`` falls
        below 0.1.
    """
    P = I.shape[1]
    acc = Precision.of(precision).accum

    A = pixel_design(delta_fit, g, xp)
    kappa_p = float(cond2(A.T @ A, xp))

    # kappa_ps on the normalized design (unit-circle directions, b divided
    # out), from five scalar reductions rather than a (P, 3) design matrix.
    r = xp.maximum(xp.hypot(u, v), float(np.finfo(np.float64).eps))      # (P,)
    cphi, sphi = u / r, -v / r
    Scp, Ssp = float(xp.sum(cphi, dtype=acc)), float(xp.sum(sphi, dtype=acc))
    Scc = float(xp.sum(cphi * cphi, dtype=acc))
    Sss = float(xp.sum(sphi * sphi, dtype=acc))
    Scs = float(xp.sum(cphi * sphi, dtype=acc))
    CtC = xp.asarray([[float(P), Scp, Ssp], [Scp, Scc, Scs], [Ssp, Scs, Sss]])
    kappa_ps = float(cond2(CtC, xp))

    sigma = chunked_sigma(I, A, xp.vstack([a, u, v]), xp, c=c, precision=precision)
    b_amp = max(float(xp.median(r)), np.finfo(float).eps)
    predicted_rms = 0.42 * (np.sqrt(kappa_p) + 2) * (sigma / b_amp) / np.sqrt(N)

    if kappa_p > 20:
        warnings.warn(
            f"aia: poorly conditioned phase-shift distribution "
            f"(kappa_p={kappa_p:.1f}); accuracy is unreliable. Consider "
            f"more evenly-spaced phase shifts and/or more frames.",
            stacklevel=2,
        )
    if kappa_ps > 20:
        warnings.warn(
            f"aia: poor phase coverage (kappa_ps={kappa_ps:.1f}); the "
            f"field spans too little phase (roughly less than one fringe) "
            f"for the frame step to reliably separate delta_n from noise, "
            f"even though the iteration converged. Consider adding phase "
            f"diversity (e.g. tilt/carrier fringes) or using calibrated "
            f"phase steps instead of blind estimation.",
            stacklevel=2,
        )

    c_fit = xp.zeros(N, dtype=xp.float64) if c is None else xp.asarray(c, dtype=xp.float64)
    g_min_ratio = float(xp.min(g)) / max(float(xp.median(g)), np.finfo(float).eps)
    if g_min_ratio < 0.1:
        warnings.warn(
            f"aia: at least one frame's fitted gain is far below the "
            f"median (g_min_ratio={g_min_ratio:.3f}); that frame is nearly "
            f"uncorrelated with the recovered fringe pattern and its "
            f"delta_n is poorly determined. Consider dropping it from the "
            f"acquisition.",
            stacklevel=2,
        )

    return AIAParam(
        kappa_p=kappa_p, kappa_ps=kappa_ps, predicted_rms=predicted_rms,
        iters_run=iters_run, converged=converged,
        g_fit=xp.asarray(g, dtype=xp.float64), c_fit=c_fit, g_min_ratio=g_min_ratio,
        precision=Precision.of(precision),
    )
