# src/phase_shift/methods/sf_aia.py
"""Spatial-Field AIA (SF-AIA) of ``docs/sf_aia.md``.

Recovers the spatially varying part ``Delta_n(x, y)`` of each frame's phase
step, which the piston-only AIA solve leaves in its residual. ``aia_step_field``
alternates an AIA solve with the per-frame coefficient fit of
:func:`fit_step_field` until the score of
:func:`phase_shift.methods.step_field.step_field_quality` stops improving;
see §"Algorithm".
"""

from dataclasses import dataclass

import numpy as np

from ..backend import Precision, get_array_module
from ..basis import BASES, spatial_basis
from ..utils import format_value
from .aia import aia
from .diagnostics import aia_diagnostics, cond2
from .gauge import (center_coeffs, center_offsets, normalize_gain, pin_phase_origin,
                    whiten_uv)
from .step_field import StepFieldParam, step_field_quality
from .steps import frame_step, pixel_step


def fit_step_field(stack: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
                   delta: np.ndarray, basis: np.ndarray, g: np.ndarray | None = None,
                   chunk: int = 1_000_000, precision: str | Precision | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Fit each frame's step-field coefficients from the AIA residual.

    Solves ``G^(n) c = -h^(n)`` independently per frame, ``docs/sf_aia.md``
    Eq. (E1): a spatially varying phase-step error leaves a residual in the
    piston-model fit that is linear in that frame's coefficients. ``G^(n)`` is
    accumulated over pixel chunks, so the basis pair products are never held
    for the whole field at once.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P = H*W`` pixels each.
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components of the piston-model solution.
    delta : np.ndarray, shape (N,)
        Piston phase step of each frame, in radians.
    basis : np.ndarray, shape (J, P)
        Step-field basis, e.g. from :func:`phase_shift.basis.spatial_basis`.
    g : np.ndarray, shape (N,), optional
        Per-frame fringe gain, as used in the pixel step. Defaults to ones.
    chunk : int, default 1_000_000
        Pixels accumulated per chunk; bounds memory, not the result.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. Its
        ``accum`` is the dtype of the per-chunk ``(N, Pc)`` arrays.

    Returns
    -------
    coeffs : np.ndarray, shape (J, N), float64
        Per-frame coefficients ``c_jn``.
    cond : np.ndarray, shape (N,), float64
        Per-frame condition number of ``G^(n)``, Eq. (E3). Large values mark a
        frame whose fit is unreliable however small its residual.

    Raises
    ------
    ValueError
        If the shapes of ``stack``, ``a``/``u``/``v``, ``delta``, ``basis`` or
        ``g`` are inconsistent.
    """
    if stack.ndim != 2:
        raise ValueError(f"stack must be 2-D (N, P), got shape {stack.shape}")
    if a.shape != u.shape or a.shape != v.shape:
        raise ValueError(f"a, u and v must have the same shape, got {a.shape}, "
                         f"{u.shape} and {v.shape}")
    if a.ndim != 1 or a.shape[0] != stack.shape[1]:
        raise ValueError(f"a, u and v must be 1-D of length {stack.shape[1]}, "
                         f"got shape {a.shape}")
    if delta.ndim != 1 or delta.shape[0] != stack.shape[0]:
        raise ValueError(f"delta must be 1-D of length {stack.shape[0]}, "
                         f"got shape {delta.shape}")
    if basis.ndim != 2 or basis.shape[1] != stack.shape[1]:
        raise ValueError(f"basis must be 2-D (J, {stack.shape[1]}), got shape {basis.shape}")
    if g is not None and len(g) != len(delta):
        raise ValueError(f"g must have length {len(delta)}, got {len(g)}")

    xp = get_array_module(stack, a, u, v, delta, basis)
    N, P = stack.shape
    J = basis.shape[0]
    acc = Precision.of(precision).accum
    delta = xp.asarray(delta, dtype=xp.float64)
    g = xp.ones(N, dtype=xp.float64) if g is None else xp.asarray(g, dtype=xp.float64)
    # Every operand of the (N, Pc) blocks below reaches accum first, so the
    # blocks are never wider than the precision asks for.
    basis = basis.astype(acc, copy=False)
    g_a = g.astype(acc)
    c, s = xp.cos(delta).astype(acc), xp.sin(delta).astype(acc)

    # Upper-triangle (j, j') pairs of G, in plain Python since J is small.
    iu = [jj for jj in range(J) for _ in range(jj, J)]
    ju = [kk for jj in range(J) for kk in range(jj, J)]
    iu, ju = xp.asarray(iu), xp.asarray(ju)
    K = len(iu)

    G_flat = xp.zeros((N, K), dtype=xp.float64)
    h = xp.zeros((N, J), dtype=xp.float64)

    for start in range(0, P, chunk):
        sl = slice(start, start + chunk)
        a_c, u_c, v_c = a[sl], u[sl], v[sl]
        basis_c = basis[:, sl]                                          # (J, Pc)
        stack_c = stack[:, sl].astype(acc, copy=False)                  # (N, Pc)

        model_c = a_c[None, :] + g_a[:, None] * (xp.outer(c, u_c) + xp.outer(s, v_c))
        resid_c = stack_c - model_c                                     # (N, Pc)
        w_c = g_a[:, None] * (xp.outer(s, u_c) - xp.outer(c, v_c))      # (N, Pc)

        pp_c = basis_c[iu, :] * basis_c[ju, :]                          # (K, Pc)
        G_flat += (w_c * w_c) @ pp_c.T                                  # (N, K)
        h += (w_c * resid_c) @ basis_c.T                                # (N, J)

    G = xp.zeros((N, J, J), dtype=xp.float64)
    G[:, iu, ju] = G_flat
    G[:, ju, iu] = G_flat
    cond = cond2(G, xp)

    # The batched solve needs the right-hand side's last two axes as its core
    # dimensions; the singleton is squeezed off again.
    sol = xp.linalg.solve(G, -h[..., None])[..., 0]                     # (N, J)
    return sol.T, cond


@dataclass
class SFAIAParam(StepFieldParam):
    """Diagnostics of an SF-AIA solve.

    The step-field fields are documented on
    :class:`phase_shift.methods.step_field.StepFieldParam`; ``coeffs`` and
    ``kappa_fit`` are those of the best round. The refinement loop adds:

    Attributes
    ----------
    rms_frac : float
        The best round's :func:`phase_shift.methods.step_field.step_field_quality`
        score.
    rms_frac_history : list of float
        Score of every round, in order, including rounds that made it worse.
    refine_iters_run : int
        Number of refinement rounds run.
    refine_converged : bool
        Whether the loop stopped on ``refine_tol`` rather than exhausting
        ``refine_iters``.
    best_iter : int
        Round the reported coefficients come from, not necessarily the last;
        ``-1`` when no round ran.
    """

    rms_frac: float
    rms_frac_history: list[float]
    refine_iters_run: int
    refine_converged: bool
    best_iter: int

    def print_summary(self) -> None:
        """Print the shared step-field diagnostics, then the refinement ones."""
        super().print_summary()
        print(f"refine_converged: {format_value(self.refine_converged)}")
        print(f"refine_iters_run: {format_value(self.refine_iters_run)}")
        print(f"best_iter:        {format_value(self.best_iter)}")
        print(f"rms_frac:         {format_value(self.rms_frac)}")


def aia_step_field(stack: np.ndarray, g: np.ndarray, fit_gain: bool = False,
                   delta0: np.ndarray | None = None, iters: int = 30, tol: float = 1e-4,
                   precision: str | Precision | None = None, basis: str = "poly",
                   basis_kwargs: dict | None = None, refine_iters: int = 5,
                   refine_tol: float = 1e-3, crop: int = 100
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                              SFAIAParam]:
    """Recover phase with a spatially varying phase step.

    Runs :func:`phase_shift.methods.aia.aia` to convergence, then repeats the
    rounds of ``docs/sf_aia.md`` §"Algorithm": fit the per-frame coefficients
    (:func:`fit_step_field`), score them (:func:`step_field_quality`),
    gauge-fix them (Eq. E4), remove their contribution from the *original*
    stack, and re-solve. Stops when the round-over-round improvement falls
    below ``refine_tol`` or the round budget is spent, and reports the
    best-scoring round.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted frames; see :func:`phase_shift.methods.aia.aia`.
    g : np.ndarray, shape (N,)
        Per-frame fringe gain; see :func:`phase_shift.methods.aia.aia`.
    fit_gain : bool, default False
        Recover ``g`` jointly, in the initial solve and in every round.
    delta0, iters, tol, precision
        Passed to the initial :func:`phase_shift.methods.aia.aia` call;
        ``precision`` applies to the refinement rounds too.
    basis : str, default "poly"
        Step-field basis family, one of :data:`phase_shift.basis.BASES`.
    basis_kwargs : dict, optional
        Arguments for that family, e.g. ``{"degree": 2}``. The default builds
        a degree-1 polynomial, a pure tilt; an empty basis (``degree=0``)
        returns the plain AIA result. §"Conditioning" advises keeping the
        basis small.
    refine_iters : int, default 5
        Maximum number of refinement rounds; ``0`` returns the plain AIA
        result with zero coefficients.
    refine_tol : float, default 1e-3
        Stop once a round's non-negative improvement in ``rms_frac`` falls
        below this. A round that makes the score worse keeps the loop running
        but is never reported.
    crop : int, default 100
        Pixels excluded from each edge when scoring; see
        :func:`step_field_quality`.

    Returns
    -------
    a, b, phi, delta, g, method_param
        As :func:`phase_shift.methods.aia.aia`, with an :class:`SFAIAParam`.

    Raises
    ------
    ValueError
        If ``basis`` is not registered, or ``refine_iters``/``refine_tol`` is
        negative.
    """
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}, expected one of {BASES}")
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be non-negative, got {refine_iters}")
    if refine_tol < 0:
        raise ValueError(f"refine_tol must be non-negative, got {refine_tol}")

    xp = get_array_module(stack)
    N, H, W = stack.shape
    p = Precision.of(precision)
    I = stack.reshape(N, -1).astype(p.work, copy=False)            # (N, P)
    g = xp.asarray(g, dtype=xp.float64)
    basis_kwargs = dict(basis_kwargs) if basis_kwargs else {}
    basis_rows = spatial_basis(H, W, basis, xp, precision=p, **basis_kwargs)  # (J, P)
    basis_work = basis_rows.astype(p.work, copy=False)             # (J, P)
    J = basis_rows.shape[0]

    a_map, b0, phi0, delta, g, aia_param0 = aia(stack, g, fit_gain=fit_gain, delta0=delta0,
                                                iters=iters, tol=tol, precision=p)
    a = a_map.reshape(-1)
    u = (b0 * xp.cos(phi0)).reshape(-1)
    v = (-b0 * xp.sin(phi0)).reshape(-1)
    # (a, u, v) were fit against I minus aia()'s own c_fit, zero unless
    # fit_gain; the regression below must match.
    c = aia_param0.c_fit

    coeffs = xp.zeros((J, N), dtype=xp.float64)
    kappa_fit = float("nan")
    rms_frac = float("nan")
    rms_history: list[float] = []
    prev_rms = None
    refine_converged = False
    best = None            # fields and score of the best round
    best_iter = -1
    it = -1

    # An empty basis leaves nothing to fit, so the plain aia() result passes
    # through unchanged, as with refine_iters=0.
    for it in range(refine_iters if J > 0 else 0):
        Ic = I - c.astype(p.work)[:, None] if fit_gain else I
        coeffs_it, cond = fit_step_field(Ic, a, u, v, delta, basis_rows, g=g,
                                         precision=p)
        kappa_it = float(xp.max(cond))
        rms_frac_it, _ = step_field_quality(Ic, a, u, v, delta, coeffs_it, basis_rows,
                                            H, W, g=g, crop=crop, precision=p)
        rms_history.append(rms_frac_it)

        # The snapshot holds the fields this round was fit and scored against,
        # before its own correction, so it stays self-consistent.
        if best is None or rms_frac_it < best[-1]:
            best = (a, u, v, delta, g, c, coeffs_it, kappa_it, rms_frac_it)
            best_iter = it

        # Only a non-negative improvement counts as converged, so a worse
        # round cannot stop the loop.
        if prev_rms is not None and 0 <= (prev_rms - rms_frac_it) < refine_tol:
            refine_converged = True
            break
        prev_rms = rms_frac_it

        coeffs_fixed = center_coeffs(coeffs_it)

        # Correct the original stack: each round estimates the total field,
        # not an increment. Eq. (E1) fits resid ~= -w*Delta, so the correction
        # adds it back. Every operand is in the working dtype, so the
        # corrected stack costs no more than the stack itself.
        gs = (g * xp.sin(delta)).astype(p.work)                       # (N,)
        gc = (g * xp.cos(delta)).astype(p.work)
        w = xp.outer(gs, u) - xp.outer(gc, v)                         # (N, P)
        corrected = I + w * (coeffs_fixed.T.astype(p.work) @ basis_work)
        corrected_Ic = corrected - c.astype(p.work)[:, None] if fit_gain else corrected

        a, u, v = pixel_step(corrected_Ic, delta, g, precision=p)
        if fit_gain:
            u, v = whiten_uv(u, v, xp, precision=p)
        new_delta, new_g, new_c = frame_step(corrected, u, v, precision=p)
        delta = pin_phase_origin(new_delta)
        if fit_gain:
            c = center_offsets(new_c, xp)
            g = normalize_gain(new_g, xp)

    refine_iters_run = it + 1

    if best is not None:
        a, u, v, delta, g, c, coeffs, kappa_fit, rms_frac = best

    aia_param = aia_diagnostics(I, delta, g, a, u, v, N, xp, aia_param0.iters_run,
                                aia_param0.converged, c=(c if fit_gain else None),
                                precision=p)
    method_param = SFAIAParam(
        aia_param=aia_param, basis=basis, basis_kwargs=basis_kwargs, coeffs=coeffs,
        coeffs_rms=xp.sqrt(xp.mean(coeffs ** 2, axis=1)), kappa_fit=kappa_fit,
        rms_frac=rms_frac, rms_frac_history=rms_history,
        refine_iters_run=refine_iters_run, refine_converged=refine_converged,
        best_iter=best_iter, precision=p,
    )

    phi = xp.arctan2(-v, u).reshape(H, W)
    b = xp.hypot(u, v).reshape(H, W)
    a_map = a.reshape(H, W)
    return a_map, b, phi, delta, g, method_param
