# src/phase_shift/methods/vp_aia.py
"""Variable-Projection AIA (VP-AIA) of ``docs/vp_aia.md``.

Recovers the spatially varying part ``Delta_n(x, y)`` of each frame's phase
step as a first-order correction to the AIA solution, in one pass: the pixel
fields are eliminated by projection (§"Removing the pixel corrections"), the
frame corrections and mode coefficients are fitted jointly
(:func:`phase_shift.methods.vp_system.fit_frame_and_coeffs`), and the pixel
corrections follow in closed form from Eq. (26). Unlike SF-AIA's alternation
it reaches the joint least-squares solution without a refinement loop, and so
without that loop's attenuation bias; see §"Comparison with SF-AIA".
"""

from dataclasses import dataclass
from types import ModuleType

import numpy as np

from ..backend import Precision, get_array_module
from ..basis import BASES, spatial_basis
from ..errors import vp_phi_error
from ..utils import format_value
from .aia import aia
from .diagnostics import aia_diagnostics
from .gauge import normalize_quadrature_frame
from .step_field import StepFieldParam, step_field_quality
from .steps import pixel_step
from .vp_system import fit_frame_and_coeffs


@dataclass
class VPAIAParam(StepFieldParam):
    """Diagnostics of a VP-AIA solve.

    The step-field fields are documented on
    :class:`phase_shift.methods.step_field.StepFieldParam`. For this method
    ``kappa_fit`` is the condition number of the one joint system of
    ``docs/vp_aia.md`` Eq. (25), not a per-frame value: of order 100 for a
    well-posed fit, and independent of the pixel count. The first-order pass
    adds:

    Attributes
    ----------
    P_corr, Q_corr : np.ndarray, shape (N,)
        First-order frame corrections ``P_n^(1)``, ``Q_n^(1)`` of Eq. (24),
        already folded into the reported ``delta`` and ``g``. Large values
        relative to ``(P_n, Q_n)`` mean the AIA baseline was far from the
        first-order solution, so the expansion is being stretched.
    beta_cov_unit : np.ndarray, shape (2N+NJ, 2N+NJ)
        Covariance of the fitted unknowns per unit ``sigma_0^2``,
        §"Noise of the fitted unknowns". Of the last round when
        ``rounds_run > 1``; the document derives the noise for a single pass.
    rms_frac : float
        :func:`phase_shift.methods.step_field.step_field_quality` score of the
        corrected solution: residual RMS over the cropped field, divided by
        the data's.
    rounds_run : int
        Number of relinearization rounds run, ``docs/vp_aia.md`` Eq. (28).
        ``1`` is the single first-order pass the method exists for.
    increment_rms : list of float
        RMS over frames and pixels of each round's increment to ``Delta_n``,
        in radians, in order. A second entry well below the first means the
        first pass had already converged; one comparable to it means the
        expansion was being stretched and the extra rounds earned their cost.
        The sequence is not monotone -- each round re-runs the AIA baseline,
        which re-partitions a little between the piston and the field -- while
        the phase error it leaves does fall round over round.
    """

    P_corr: np.ndarray
    Q_corr: np.ndarray
    beta_cov_unit: np.ndarray
    rms_frac: float
    rounds_run: int
    increment_rms: list[float]

    def phi_error(self, b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                  fit_gain: bool, noise_std: np.ndarray, simplified: bool,
                  xp: ModuleType) -> np.ndarray:
        """Return ``sigma_Phi`` of ``docs/vp_aia.md`` Eq. (29).

        The pixel-step variance plus the variance the fitted step field adds.
        See :meth:`phase_shift.methods.base.MethodParam.phi_error` for the
        arguments.

        Returns
        -------
        np.ndarray, shape (H, W)
            Per-pixel phase standard deviation, in radians.
        """
        H, W = phi.shape
        basis = spatial_basis(H, W, self.basis, xp, precision=self.precision,
                              **self.basis_kwargs)
        return vp_phi_error(b, phi, g * xp.cos(delta), g * xp.sin(delta), noise_std,
                            simplified, self.beta_cov_unit, basis, xp,
                            precision=self.precision)

    def print_summary(self) -> None:
        """Print the shared step-field diagnostics, then the first-order ones."""
        super().print_summary()
        print(f"rms_frac:         {format_value(self.rms_frac)}")
        print(f"rounds_run:       {format_value(self.rounds_run)}")
        print(f"increment_rms:    {format_value(self.increment_rms)}")
        print(f"|P_corr|max:      {format_value(float(np.max(np.abs(self.P_corr))))}")
        print(f"|Q_corr|max:      {format_value(float(np.max(np.abs(self.Q_corr))))}")


def _pixel_corrections(alpha: np.ndarray, u: np.ndarray, v: np.ndarray,
                       P_n: np.ndarray, Q_n: np.ndarray, basis: np.ndarray,
                       acc: np.dtype, xp: ModuleType
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-field corrections of ``docs/vp_aia.md`` Eq. (26).

    The three right-hand sides ``sum_n w_n Delta_n``, ``sum_n P_n w_n Delta_n``
    and ``sum_n Q_n w_n Delta_n`` factor through five ``(J,)`` vectors, since
    ``w_n = P_n v - Q_n u`` and ``Delta_n = sum_j alpha_nj H_j``. Evaluating
    them that way needs three ``(P,)`` fields and no ``(N, P)`` array, which
    is why §"cost" can say ``Delta_n`` need not be stored.

    Parameters
    ----------
    alpha : np.ndarray, shape (N, J)
        Fitted mode coefficients.
    u, v : np.ndarray, shape (P,)
        Zeroth-order quadrature fields.
    P_n, Q_n : np.ndarray, shape (N,)
        Zeroth-order per-frame coefficients.
    basis : np.ndarray, shape (J, P)
        Spatial modes ``H_j``.
    acc : dtype
        ``precision.accum``.
    xp : module
        ``numpy`` or ``cupy``, matching the inputs.

    Returns
    -------
    a1, u1, v1 : np.ndarray, shape (P,)
        Corrections ``a^(1)``, ``u^(1)``, ``v^(1)``, in ``acc``.
    """
    N = P_n.shape[0]
    A = xp.stack([xp.ones(N, dtype=xp.float64), P_n, Q_n], axis=1)        # (N, 3)
    A_p = (A.T @ A).astype(acc)                                           # (3, 3), Eq. (7)

    mP = (P_n @ alpha).astype(acc)                                        # (J,)
    mQ = (Q_n @ alpha).astype(acc)
    mPP = ((P_n * P_n) @ alpha).astype(acc)
    mPQ = ((P_n * Q_n) @ alpha).astype(acc)
    mQQ = ((Q_n * Q_n) @ alpha).astype(acc)

    u_a, v_a = u.astype(acc, copy=False), v.astype(acc, copy=False)
    rhs = xp.stack([v_a * (mP @ basis) - u_a * (mQ @ basis),
                    v_a * (mPP @ basis) - u_a * (mPQ @ basis),
                    v_a * (mPQ @ basis) - u_a * (mQQ @ basis)])           # (3, P)
    a1, u1, v1 = -xp.linalg.solve(A_p, rhs)
    return a1, u1, v1


def _relinearize(I: np.ndarray, out: np.ndarray, u: np.ndarray, v: np.ndarray,
                 delta: np.ndarray, g: np.ndarray, alpha: np.ndarray,
                 basis: np.ndarray, chunk: int, acc: np.dtype,
                 xp: ModuleType) -> np.ndarray:
    """Remove the accumulated step field from the stack, ``docs/vp_aia.md`` Eq. (28).

    ``I_n' = I_n - g_n b [cos(Phi + delta_n + Delta_n) - cos(Phi + delta_n)]``,
    written in quadrature variables: ``b cos(Phi + theta) = u cos(theta) +
    v sin(theta)``, so the current ``(u, v)`` serve for both terms and no
    ``b``/``Phi`` pair is rebuilt. Written into ``out`` one pixel block at a
    time.

    Parameters
    ----------
    I : np.ndarray, shape (N, P)
        The measured stack, unchanged between rounds.
    out : np.ndarray, shape (N, P)
        Destination buffer, reused across rounds.
    u, v : np.ndarray, shape (P,)
        Current quadrature fields.
    delta, g : np.ndarray, shape (N,)
        Current piston steps and gains.
    alpha : np.ndarray, shape (N, J)
        Accumulated coefficients of ``Delta_n``.
    basis : np.ndarray, shape (J, P)
        The modes ``H_j``.
    chunk : int
        Pixels per block.
    acc : dtype
        ``precision.accum``.
    xp : module

    Returns
    -------
    np.ndarray, shape (N, P)
        ``out``.
    """
    P = I.shape[1]
    g_a = g.astype(acc)
    d_a = delta.astype(acc)
    alpha_a = alpha.astype(acc)
    gc, gs = g_a * xp.cos(d_a), g_a * xp.sin(d_a)
    for start in range(0, P, chunk):
        sl = slice(start, start + chunk)
        u_c, v_c = u[sl].astype(acc, copy=False), v[sl].astype(acc, copy=False)
        dn = d_a[:, None] + alpha_a @ basis[:, sl]                        # (N, C)
        shifted = g_a[:, None] * (u_c[None, :] * xp.cos(dn)
                                  + v_c[None, :] * xp.sin(dn))
        piston = gc[:, None] * u_c[None, :] + gs[:, None] * v_c[None, :]
        out[:, sl] = (I[:, sl] - (shifted - piston)).astype(out.dtype, copy=False)
    return out


def aia_variable_projection(stack: np.ndarray, g: np.ndarray, fit_gain: bool = False,
                            delta0: np.ndarray | None = None, iters: int = 30,
                            tol: float = 1e-4,
                            precision: str | Precision | None = None,
                            basis: str = "poly", basis_kwargs: dict | None = None,
                            rounds: int = 1, round_tol: float = 1e-4,
                            crop: int = 100, chunk: int = 1_000_000
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, VPAIAParam]:
    """Recover phase with a spatially varying phase step, in one first-order pass.

    Runs the six steps of ``docs/vp_aia.md`` §"VP-AIA algorithm": the AIA
    baseline finished with a pixel step and normalized, then the residual and
    projector of Eqs. (16)-(18), the joint fit of Eq. (25), the pixel
    corrections of Eq. (26), the corrected fields of Eq. (27), and a second
    normalization.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted frames; see :func:`phase_shift.methods.aia.aia`.
    g : np.ndarray, shape (N,)
        Starting per-frame fringe gain.
    fit_gain : bool, default False
        Must be True. ``docs/vp_aia.md`` derives Eq. (24) with ``(P_n, Q_n)``
        free in the plane; the fixed-gain case is not derived.
    delta0, iters, tol, precision
        Passed to the AIA baseline; ``precision`` applies to the first-order
        pass too.
    basis : str, default "poly"
        Mode family for ``H_j`` of Eq. (13), one of
        :data:`phase_shift.basis.BASES`.
    basis_kwargs : dict, optional
        Arguments for that family, e.g. ``{"degree": 2}``. An empty basis
        (``degree=0``) fits the frame corrections alone.
    rounds : int, default 1
        Maximum relinearization rounds, ``docs/vp_aia.md`` Eq. (28). ``1`` is
        the single first-order pass; raise it when ``max|Delta_n|`` is large
        enough for the second-order error to matter, since each round reduces
        what remains by a factor of order ``max|Delta_n|``.
    round_tol : float, default 1e-4
        Stop once a round's increment to ``Delta_n`` falls below this RMS, in
        radians. Ignored when ``rounds`` is 1.
    crop : int, default 100
        Pixels excluded from each edge when scoring; see
        :func:`phase_shift.methods.step_field.step_field_quality`.
    chunk : int, default 1_000_000
        Pixels accumulated per chunk in the one pass over the field.

    Returns
    -------
    a, b, phi, delta, g, method_param
        As :func:`phase_shift.methods.aia.aia`, with a :class:`VPAIAParam`.

    Raises
    ------
    ValueError
        If ``fit_gain`` is False, if ``basis`` is not registered, or if the
        frame count or mode count is outside what Eq. (15) allows.

    References
    ----------
    R. Frisch and F. V. Waugh, "Partial time regressions as compared with
    individual trends," Econometrica 1(4), 387-401 (1933).

    G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses and
    nonlinear least squares problems whose variables separate," SIAM Journal
    on Numerical Analysis 10(2), 413-432 (1973).
    """
    if not fit_gain:
        raise ValueError(
            "vp_aia fits the per-frame gain jointly: use gain_mode='joint' and "
            "leave PhaseConfig.g unset. docs/vp_aia.md derives Eq. (24) with "
            "(P_n, Q_n) free in the plane, and does not cover a fixed gain."
        )
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}, expected one of {BASES}")
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    if round_tol < 0:
        raise ValueError(f"round_tol must be non-negative, got {round_tol}")

    xp = get_array_module(stack)
    N, H, W = stack.shape
    p = Precision.of(precision)
    basis_kwargs = dict(basis_kwargs) if basis_kwargs else {}
    modes = spatial_basis(H, W, basis, xp, precision=p, **basis_kwargs)   # (J, P)
    J = modes.shape[0]

    I = stack.reshape(N, -1).astype(p.work, copy=False)                   # (N, P)
    K = I.shape[1]
    alpha = xp.zeros((N, J), dtype=xp.float64)            # accumulated Delta_n
    increment_rms: list[float] = []
    buffer = xp.empty_like(I) if rounds > 1 else None
    delta_start = delta0
    it = 0

    for it in range(rounds):
        # Eq. (28): every round but the first re-linearizes about the
        # accumulated field, so what AIA sees is the remaining error only.
        I_lin = I if it == 0 else _relinearize(I, buffer, u, v, delta, g, alpha,
                                               modes, chunk, p.accum, xp)

        # Step 1: AIA baseline, a final pixel step at the converged (P_n, Q_n),
        # and normalization. Eq. (20) needs the pixel-step residual, so the
        # final pixel step is part of the algorithm, not a tidy-up.
        _, _, _, delta, g, aia_param = aia(I_lin.reshape(N, H, W), g, fit_gain=True,
                                           delta0=delta_start, iters=iters, tol=tol,
                                           precision=p)
        a0, u0, v0 = pixel_step(I_lin, delta, g, precision=p)
        P_n = g * xp.cos(delta)                                           # (N,), Eq. (5)
        Q_n = g * xp.sin(delta)
        a0, u0, v0, P_n, Q_n = normalize_quadrature_frame(a0, u0, v0, P_n, Q_n, xp,
                                                          precision=p)

        # Steps 2-4: residual, projector, and the joint fit of Eq. (25).
        sol = fit_frame_and_coeffs(I_lin, a0, u0, v0, P_n, Q_n, modes, chunk=chunk,
                                   precision=p)

        # Step 5: pixel corrections, Eq. (26), and the corrected fields, Eq. (27).
        a1, u1, v1 = _pixel_corrections(sol.alpha, u0, v0, P_n, Q_n, modes, p.accum, xp)
        a = (a0 + a1).astype(p.work, copy=False)
        u = (u0 + u1).astype(p.work, copy=False)
        v = (v0 + v1).astype(p.work, copy=False)

        # Step 6: normalization of the corrected fields.
        a, u, v, P_n, Q_n = normalize_quadrature_frame(a, u, v, P_n + sol.P_corr,
                                                       Q_n + sol.Q_corr, xp, precision=p)
        delta = xp.arctan2(Q_n, P_n)                                      # delta[0] = 0
        g = xp.sqrt(P_n * P_n + Q_n * Q_n)                                # median(g) = 1

        # The increments carry Eq. (13)'s two zero means, so the accumulated
        # field keeps both conventions. The modes are orthonormal over the
        # field, so the RMS of this round's increment needs no (N, P) array.
        alpha = alpha + sol.alpha
        delta_start = delta
        increment_rms.append(float(xp.sqrt(xp.sum(sol.alpha ** 2) / (N * K))))
        if increment_rms[-1] < round_tol:
            break

    coeffs = alpha.T                                                      # (J, N)
    rms_frac, _ = step_field_quality(I, a, u, v, delta, coeffs, modes, H, W, g=g,
                                     crop=crop, precision=p)

    aia_param = aia_diagnostics(I, delta, g, a, u, v, N, xp, aia_param.iters_run,
                                aia_param.converged, precision=p)
    method_param = VPAIAParam(
        aia_param=aia_param, basis=basis, basis_kwargs=basis_kwargs, coeffs=coeffs,
        coeffs_rms=xp.sqrt(xp.mean(coeffs ** 2, axis=1)) if J else xp.zeros(0),
        kappa_fit=sol.kappa_vp, precision=p, P_corr=sol.P_corr, Q_corr=sol.Q_corr,
        beta_cov_unit=sol.beta_cov_unit, rms_frac=rms_frac, rounds_run=it + 1,
        increment_rms=increment_rms,
    )

    phi = xp.arctan2(-v, u).reshape(H, W)
    b = xp.hypot(u, v).reshape(H, W)
    return a.reshape(H, W), b, phi, delta, g, method_param
