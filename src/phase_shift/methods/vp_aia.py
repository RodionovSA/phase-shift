# src/phase_shift/methods/vp_aia.py
"""Variable-Projection AIA (VP-AIA) of ``docs/vp_aia.md``.

Recovers the spatially varying part ``Delta_n(x, y)`` of each frame's phase
step as corrections to the AIA solution, order by order in the size of the
step error (§"Expansion around the AIA solution"). At every order the pixel
fields are eliminated by projection (§"Removing the pixel corrections"), the
frame corrections and mode coefficients are fitted with one fixed matrix
(:mod:`phase_shift.methods.vp_system`), and the pixel corrections follow in
closed form from Eq. (17).
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
from .vp_system import build_system, order_rhs, solve_order


@dataclass
class VPAIAParam(StepFieldParam):
    """Diagnostics of a VP-AIA solve.

    The step-field fields are documented on
    :class:`phase_shift.methods.step_field.StepFieldParam`. For this method
    ``kappa_fit`` is the condition number of the one normal system of
    ``docs/vp_aia.md`` Eq. (16), shared by all orders: of order 100 for a
    well-posed fit, and independent of the pixel count. The orders add:

    Attributes
    ----------
    P_corr, Q_corr : np.ndarray, shape (N,)
        Frame corrections summed over the orders run, Eq. (18), before the
        final normalization. Large values relative to ``(P_n, Q_n)`` mean the
        AIA solution was far from the corrected one, so the expansion is being
        stretched.
    beta_cov_unit : np.ndarray, shape (2N+NJ, 2N+NJ)
        Covariance of the fitted unknowns per unit ``sigma_0^2``,
        §"Noise of the fitted unknowns"; it holds for any number of orders.
    rms_frac : float
        :func:`phase_shift.methods.step_field.step_field_quality` score of the
        corrected solution: residual RMS over the cropped field, divided by
        the data's.
    orders_run : int
        Number of orders computed, ``k_max`` of §"VP-AIA algorithm and cost"
        unless ``order_tol`` stopped earlier.
    increment_rms : list of float
        RMS over frames and pixels of each order's increment ``Delta_n^(k)``,
        in radians, in order. Each entry should be smaller than the previous
        one by a factor of order ``max|Delta_n|``.
    """

    P_corr: np.ndarray
    Q_corr: np.ndarray
    beta_cov_unit: np.ndarray
    rms_frac: float
    orders_run: int
    increment_rms: list[float]

    def phi_error(self, b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                  fit_gain: bool, noise_std: np.ndarray, simplified: bool,
                  xp: ModuleType) -> np.ndarray:
        """Return ``sigma_Phi`` of ``docs/vp_aia.md`` Eq. (19).

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
        """Print the shared step-field diagnostics, then those of the orders."""
        super().print_summary()
        print(f"rms_frac:         {format_value(self.rms_frac)}")
        print(f"orders_run:       {format_value(self.orders_run)}")
        print(f"increment_rms:    {format_value(self.increment_rms)}")
        print(f"|P_corr|max:      {format_value(float(np.max(np.abs(self.P_corr))))}")
        print(f"|Q_corr|max:      {format_value(float(np.max(np.abs(self.Q_corr))))}")


def _pixel_corrections(alpha: np.ndarray, Atr: np.ndarray, u: np.ndarray, v: np.ndarray,
                       P_n: np.ndarray, Q_n: np.ndarray, basis: np.ndarray,
                       acc: np.dtype, xp: ModuleType
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-field corrections of one order, ``docs/vp_aia.md`` Eq. (17).

    ``A_p^-1 A' (r^(k) - d^(k))`` with ``d_n^(k) = w_n Delta_n^(k)``. The three
    entries of ``A' d^(k)``, ``sum_n w_n Delta_n``, ``sum_n P_n w_n Delta_n``
    and ``sum_n Q_n w_n Delta_n``, factor through five ``(J,)`` vectors, since
    ``w_n = P_n v - Q_n u`` and ``Delta_n^(k) = sum_j alpha_nj^(k) H_j``, so no
    ``(N, P)`` array is formed.

    Parameters
    ----------
    alpha : np.ndarray, shape (N, J)
        Mode coefficients ``alpha_nj^(k)`` of this order.
    Atr : np.ndarray, shape (3, P)
        ``A' r^(k)``, from :func:`phase_shift.methods.vp_system.order_rhs`.
    u, v : np.ndarray, shape (P,)
        AIA-solution quadrature fields.
    P_n, Q_n : np.ndarray, shape (N,)
        AIA-solution per-frame coefficients.
    basis : np.ndarray, shape (J, P)
        Spatial modes ``H_j``.
    acc : dtype
        ``precision.accum``.
    xp : module
        ``numpy`` or ``cupy``, matching the inputs.

    Returns
    -------
    ak, uk, vk : np.ndarray, shape (P,)
        Corrections ``a^(k)``, ``u^(k)``, ``v^(k)``, in ``acc``.
    """
    N = P_n.shape[0]
    A = xp.stack([xp.ones(N, dtype=xp.float64), P_n, Q_n], axis=1)        # (N, 3)
    A_p = (A.T @ A).astype(acc)                                           # (3, 3), Eq. (6)

    mP = (P_n @ alpha).astype(acc)                                        # (J,)
    mQ = (Q_n @ alpha).astype(acc)
    mPP = ((P_n * P_n) @ alpha).astype(acc)
    mPQ = ((P_n * Q_n) @ alpha).astype(acc)
    mQQ = ((Q_n * Q_n) @ alpha).astype(acc)

    u_a, v_a = u.astype(acc, copy=False), v.astype(acc, copy=False)
    Atd = xp.stack([v_a * (mP @ basis) - u_a * (mQ @ basis),
                    v_a * (mPP @ basis) - u_a * (mPQ @ basis),
                    v_a * (mPQ @ basis) - u_a * (mQQ @ basis)])           # (3, P)
    ak, uk, vk = xp.linalg.solve(A_p, Atr - Atd)
    return ak, uk, vk


def aia_variable_projection(stack: np.ndarray, g: np.ndarray, fit_gain: bool = False,
                            delta0: np.ndarray | None = None, iters: int = 30,
                            tol: float = 1e-4,
                            precision: str | Precision | None = None,
                            basis: str = "poly", basis_kwargs: dict | None = None,
                            orders: int = 1, order_tol: float = 0.0,
                            crop: int = 100, chunk: int = 1_000_000
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, VPAIAParam]:
    """Recover phase with a spatially varying phase step, order by order.

    Runs ``docs/vp_aia.md`` §"VP-AIA algorithm and cost": the AIA solution,
    finished with a pixel step and normalized; the normal system of
    Eqs. (14)-(16), formed once; then, at each order, the full-model residual
    of Eq. (8), the solve of Eq. (16), the pixel corrections of Eq. (17), and
    the accumulation of Eq. (18); and a final normalization (Appendix B).

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted frames; see :func:`phase_shift.methods.aia.aia`.
    g : np.ndarray, shape (N,)
        Starting per-frame fringe gain.
    fit_gain : bool, default False
        Must be True. ``docs/vp_aia.md`` derives Eq. (15) with ``(P_n, Q_n)``
        free in the plane; the fixed-gain case is not derived.
    delta0, iters, tol, precision
        Passed to the AIA solve; ``precision`` applies to the orders too.
    basis : str, default "poly"
        Mode family for ``H_j`` of Eq. (2), one of
        :data:`phase_shift.basis.BASES`.
    basis_kwargs : dict, optional
        Arguments for that family, e.g. ``{"degree": 2}``. An empty basis
        (``degree=0``) fits the frame corrections alone.
    orders : int, default 1
        Maximum number of orders ``k_max``. ``1`` is the first-order
        correction; each further order reduces the error in ``Delta_n`` by a
        factor of order ``max|Delta_n|`` and costs one more pass over the
        stack.
    order_tol : float, default 0.0
        Stop before ``orders`` once an order's increment to ``Delta_n`` falls
        below this RMS, in radians. ``0`` runs every order.
    crop : int, default 100
        Pixels excluded from each edge when scoring; see
        :func:`phase_shift.methods.step_field.step_field_quality`.
    chunk : int, default 1_000_000
        Pixels accumulated per chunk in each pass over the stack.

    Returns
    -------
    a, b, phi, delta, g, method_param
        As :func:`phase_shift.methods.aia.aia`, with a :class:`VPAIAParam`.

    Raises
    ------
    ValueError
        If ``fit_gain`` is False, if ``basis`` is not registered, if
        ``orders`` or ``order_tol`` is out of range, or if the frame count or
        mode count is outside what Eq. (3) allows.

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
            "leave PhaseConfig.g unset. docs/vp_aia.md derives Eq. (15) with "
            "(P_n, Q_n) free in the plane, and does not cover a fixed gain."
        )
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}, expected one of {BASES}")
    if not isinstance(orders, (int, np.integer)) or orders < 1:
        raise ValueError(f"orders must be an integer of at least 1, got {orders!r}")
    if order_tol < 0:
        raise ValueError(f"order_tol must be non-negative, got {order_tol}")

    xp = get_array_module(stack)
    N, H, W = stack.shape
    p = Precision.of(precision)
    basis_kwargs = dict(basis_kwargs) if basis_kwargs else {}
    modes = spatial_basis(H, W, basis, xp, precision=p, **basis_kwargs)   # (J, P)
    J = modes.shape[0]

    I = stack.reshape(N, -1).astype(p.work, copy=False)                   # (N, P)
    K = I.shape[1]

    # Step 1: AIA solution, a final pixel step at the converged (P_n, Q_n),
    # and normalization. The expansion is taken about this solution.
    _, _, _, delta, g, aia_param = aia(I.reshape(N, H, W), g, fit_gain=True, delta0=delta0,
                                       iters=iters, tol=tol, precision=p)
    a0, u0, v0 = pixel_step(I, delta, g, precision=p)
    a0, u0, v0, P0, Q0 = normalize_quadrature_frame(a0, u0, v0, g * xp.cos(delta),
                                                    g * xp.sin(delta), xp, precision=p)

    # Step 2: the normal system, the same at every order.
    system = build_system(u0, v0, P0, Q0, modes, chunk=chunk, precision=p)

    # Step 3: the orders, Eqs. (8), (16), (17), (18).
    a, u, v, P_n, Q_n = a0, u0, v0, P0, Q0
    alpha = xp.zeros((N, J), dtype=xp.float64)
    increment_rms: list[float] = []
    for _ in range(orders):
        rho, Atr = order_rhs(I, a, u, v, P_n, Q_n, alpha, system, modes, chunk=chunk,
                             precision=p)
        sol = solve_order(system, rho)
        ak, uk, vk = _pixel_corrections(sol.alpha, Atr, u0, v0, P0, Q0, modes,
                                        p.accum, xp)
        a = (a + ak).astype(p.work, copy=False)
        u = (u + uk).astype(p.work, copy=False)
        v = (v + vk).astype(p.work, copy=False)
        P_n, Q_n = P_n + sol.P_corr, Q_n + sol.Q_corr
        alpha = alpha + sol.alpha
        # The modes are orthonormal over the field, so the RMS of the
        # increment needs no (N, P) array.
        increment_rms.append(float(xp.sqrt(xp.sum(sol.alpha ** 2) / (N * K))))
        if increment_rms[-1] < order_tol:
            break

    P_corr, Q_corr = P_n - P0, Q_n - Q0

    # Step 4: normalization (Appendix B). Each alpha^(k) has the zero frame
    # mean of Eq. (15), so the accumulated field keeps the conventions of
    # Eq. (2).
    a, u, v, P_n, Q_n = normalize_quadrature_frame(a, u, v, P_n, Q_n, xp, precision=p)
    delta = xp.arctan2(Q_n, P_n)                                          # delta[0] = 0
    g = xp.sqrt(P_n * P_n + Q_n * Q_n)                                    # median(g) = 1

    coeffs = alpha.T                                                      # (J, N)
    rms_frac, _ = step_field_quality(I, a, u, v, delta, coeffs, modes, H, W, g=g,
                                     crop=crop, precision=p)

    aia_param = aia_diagnostics(I, delta, g, a, u, v, N, xp, aia_param.iters_run,
                                aia_param.converged, precision=p)
    method_param = VPAIAParam(
        aia_param=aia_param, basis=basis, basis_kwargs=basis_kwargs, coeffs=coeffs,
        coeffs_rms=xp.sqrt(xp.mean(coeffs ** 2, axis=1)) if J else xp.zeros(0),
        kappa_fit=system.kappa_vp, precision=p, P_corr=P_corr, Q_corr=Q_corr,
        beta_cov_unit=system.beta_cov_unit, rms_frac=rms_frac,
        orders_run=len(increment_rms), increment_rms=increment_rms,
    )

    phi = xp.arctan2(-v, u).reshape(H, W)
    b = xp.hypot(u, v).reshape(H, W)
    return a.reshape(H, W), b, phi, delta, g, method_param
