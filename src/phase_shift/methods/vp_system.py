# src/phase_shift/methods/vp_system.py
"""Normal system of the VP-AIA fit at each order, ``docs/vp_aia.md`` Eqs. (13)-(16).

Eq. (12) holds at every pixel and contains only the ``2N + NJ`` unknowns
``P_n^(k)``, ``Q_n^(k)`` and ``alpha_nj^(k)``, so the least-squares fit reduces
to one dense system whose size does not grow with the pixel count. Its matrix
depends only on the AIA solution and the modes, so :func:`build_system` forms
and inverts it once; :func:`order_rhs` accumulates the right-hand side of one
order in a chunked pass, and :func:`solve_order` solves for that order's
unknowns. Appendix E reduces every frame-indexed pixel sum to sums without a
frame index.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from ..backend import Precision, get_array_module, weighted_gram
from .diagnostics import cond2


@dataclass
class VPSystem:
    """Fixed normal system of ``docs/vp_aia.md`` Eqs. (14)-(16), shared by all orders.

    Attributes
    ----------
    u, v : np.ndarray, shape (P,)
        AIA-solution quadrature fields the columns of Eq. (12) are built from;
        the arrays passed to :func:`build_system`, not copies.
    P_n, Q_n : np.ndarray, shape (N,), float64
        AIA-solution per-frame coefficients the system is built at.
    Pi : np.ndarray, shape (N, N), float64
        Residual-maker of the pixel step, Eq. (11).
    M_reg_inv : np.ndarray, shape (2N+NJ, 2N+NJ), float64
        ``(M + C'C)^-1`` of Eq. (16), in the mode convention ``<H_j H_k> =
        delta_jk``.
    beta_cov_unit : np.ndarray, shape (2N+NJ, 2N+NJ), float64
        ``(M + C'C)^-1 M (M + C'C)^-1``, the covariance of ``beta`` per unit
        ``sigma_0^2``; see §"Noise of the fitted unknowns". Ordered as
        ``beta``: the ``P_n^(k)``, then the ``Q_n^(k)``, then ``alpha_nj^(k)``
        with ``j`` fastest, in the caller's mode convention.
    kappa_vp : float
        Condition number of ``M + C'C``, of order 100 for a well-posed fit and
        independent of the pixel count. Large values mean Eq. (12) has a
        direction the modes cannot resolve -- the basis has outrun what the
        fringe pattern carries.
    s_mode : float
        ``sqrt(K)``, the scale between the caller's unit-norm modes and the
        convention ``M_reg_inv`` is built in.
    """

    u: np.ndarray
    v: np.ndarray
    P_n: np.ndarray
    Q_n: np.ndarray
    Pi: np.ndarray
    M_reg_inv: np.ndarray
    beta_cov_unit: np.ndarray
    kappa_vp: float
    s_mode: float


@dataclass
class VPSolution:
    """Unknowns of one order, ``docs/vp_aia.md`` Eq. (16).

    Attributes
    ----------
    P_corr, Q_corr : np.ndarray, shape (N,), float64
        Frame corrections ``P_n^(k)``, ``Q_n^(k)``, satisfying the six
        conditions of Eq. (15).
    alpha : np.ndarray, shape (N, J), float64
        Mode coefficients ``alpha_nj^(k)`` of Eq. (2), with zero frame mean
        by Eq. (15).
    """

    P_corr: np.ndarray
    Q_corr: np.ndarray
    alpha: np.ndarray


def max_modes(N: int, K: int) -> int:
    """Largest identifiable number of spatial modes, ``docs/vp_aia.md`` Eq. (3).

    ``floor((N-3)K/(N-1)) - 2``, the count bound on ``J``. It is necessary,
    not sufficient: the fit must still determine every unknown uniquely.

    Parameters
    ----------
    N : int
        Number of frames.
    K : int
        Number of pixels.

    Returns
    -------
    int
        ``0`` for ``N < 4``; identifying any mode also needs ``N >= 5``, see
        §"Spatial modes".
    """
    if N < 4:
        return 0
    return max(int(((N - 3) * K) // (N - 1)) - 2, 0)


def _field_sums(u: np.ndarray, v: np.ndarray, basis: np.ndarray, chunk: int,
                acc: np.dtype, xp) -> tuple:
    """Accumulate the frame-free sums ``F`` of ``docs/vp_aia.md`` Appendix E.

    Returns
    -------
    Suu, Svv, Suv : float
        The three scalar sums.
    uuH, vvH, uvH : np.ndarray, shape (J,), float64
        The three sums per spatial function.
    uuHH, vvHH, uvHH : np.ndarray, shape (J, J), float64
        The three sums per pair of spatial functions.
    """
    J, P = basis.shape
    Suu = Svv = Suv = 0.0
    uuH = xp.zeros(J, dtype=xp.float64)
    vvH = xp.zeros(J, dtype=xp.float64)
    uvH = xp.zeros(J, dtype=xp.float64)
    uuHH = xp.zeros((J, J), dtype=xp.float64)
    vvHH = xp.zeros((J, J), dtype=xp.float64)
    uvHH = xp.zeros((J, J), dtype=xp.float64)

    for start in range(0, P, chunk):
        sl = slice(start, start + chunk)
        u_c = u[sl].astype(acc, copy=False)
        v_c = v[sl].astype(acc, copy=False)
        H_c = basis[:, sl]                                                # (J, C)
        uu_c, vv_c, uv_c = u_c * u_c, v_c * v_c, u_c * v_c
        Suu += float(xp.sum(uu_c, dtype=acc))
        Svv += float(xp.sum(vv_c, dtype=acc))
        Suv += float(xp.sum(uv_c, dtype=acc))
        uuH += (H_c @ uu_c).astype(xp.float64)
        vvH += (H_c @ vv_c).astype(xp.float64)
        uvH += (H_c @ uv_c).astype(xp.float64)
        uuHH += weighted_gram(H_c, uu_c, H_c).astype(xp.float64)
        vvHH += weighted_gram(H_c, vv_c, H_c).astype(xp.float64)
        uvHH += weighted_gram(H_c, uv_c, H_c).astype(xp.float64)

    return Suu, Svv, Suv, uuH, vvH, uvH, uuHH, vvHH, uvHH


def build_system(u: np.ndarray, v: np.ndarray, P_n: np.ndarray, Q_n: np.ndarray,
                 basis: np.ndarray, chunk: int = 1_000_000,
                 precision: str | Precision | None = None) -> VPSystem:
    """Form and invert the normal system shared by every order.

    Builds ``M`` and ``C`` of ``docs/vp_aia.md`` Eqs. (14)-(15) from one pass
    over the pixels (Appendix E, Eq. E2) and inverts ``M + C'C``, Eq. (16).

    Parameters
    ----------
    u, v : np.ndarray, shape (P,)
        AIA-solution quadrature fields, from a pixel step at ``(P_n, Q_n)``
        normalized by :func:`phase_shift.methods.gauge.normalize_quadrature_frame`
        (§"VP-AIA algorithm and cost", step 1).
    P_n, Q_n : np.ndarray, shape (N,)
        AIA-solution per-frame coefficients, ``docs/vp_aia.md`` Eq. (4).
    basis : np.ndarray, shape (J, P)
        Spatial modes ``H_j`` of Eq. (2), e.g. from
        :func:`phase_shift.basis.spatial_basis`.
    chunk : int, default 1_000_000
        Pixels accumulated per chunk; bounds memory, not the result.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. Its
        ``accum`` carries the per-chunk blocks; the ``(2N+NJ)``-sized system
        is float64.

    Returns
    -------
    VPSystem

    Raises
    ------
    ValueError
        If the shapes are inconsistent, if ``N < 5`` while ``J > 0`` (no mode
        is identifiable below five frames, §"Spatial modes"), or if ``J``
        exceeds :func:`max_modes`.

    Warns
    -----
    UserWarning
        If ``M + C'C`` is close to singular, so Eq. (12) has an undetermined
        direction beyond those Eq. (15) fixes.
    """
    if u.ndim != 1 or u.shape != v.shape:
        raise ValueError(f"u and v must be 1-D of the same length, got shapes "
                         f"{u.shape} and {v.shape}")
    if P_n.ndim != 1 or P_n.shape != Q_n.shape:
        raise ValueError(f"P_n and Q_n must be 1-D of the same length, got shapes "
                         f"{P_n.shape} and {Q_n.shape}")
    if basis.ndim != 2 or basis.shape[1] != u.shape[0]:
        raise ValueError(f"basis must be 2-D (J, {u.shape[0]}), got shape {basis.shape}")

    xp = get_array_module(u, v, basis)
    N, K, J = P_n.shape[0], u.shape[0], basis.shape[0]
    if J > 0 and N < 5:
        raise ValueError(f"vp_aia needs at least 5 frames to identify a spatial mode, "
                         f"got N={N}; see docs/vp_aia.md §'Spatial modes'")
    if J > max_modes(N, K):
        raise ValueError(f"basis has J={J} modes, above the identifiable maximum "
                         f"{max_modes(N, K)} for N={N} frames and K={K} pixels "
                         f"(docs/vp_aia.md Eq. 3)")

    acc = Precision.of(precision).accum
    P_n = xp.asarray(P_n, dtype=xp.float64)
    Q_n = xp.asarray(Q_n, dtype=xp.float64)

    Suu, Svv, Suv, uuH, vvH, uvH, uuHH, vvHH, uvHH = _field_sums(u, v, basis, chunk,
                                                                 acc, xp)

    # Solve in the mode convention of Eq. (20), <H_j H_k> = delta_jk, rather
    # than the unit-norm rows spatial_basis returns. Without it the frame block
    # of M grows as sum u^2 ~ K while the coefficient block stays O(1), and
    # kappa_vp tracks the pixel count instead of the fit. Scaling the sums is
    # the same as scaling the rows, and costs no (J, P) copy.
    s_mode = float(np.sqrt(K))
    uuH, vvH, uvH = uuH * s_mode, vvH * s_mode, uvH * s_mode
    uuHH, vvHH, uvHH = uuHH * s_mode ** 2, vvHH * s_mode ** 2, uvHH * s_mode ** 2

    # Eq. (11): the residual-maker of the pixel-step fit.
    A = xp.stack([xp.ones(N, dtype=xp.float64), P_n, Q_n], axis=1)        # (N, 3)
    Pi = xp.eye(N, dtype=xp.float64) - A @ xp.linalg.pinv(A.T @ A) @ A.T  # (N, N)

    # Eq. (E2): every sum carrying a frame index, from the sums above.
    du = P_n[:, None] * uvH[None, :] - Q_n[:, None] * uuH[None, :]        # (N, J)
    dv = P_n[:, None] * vvH[None, :] - Q_n[:, None] * uvH[None, :]        # (N, J)
    PQ = xp.outer(P_n, Q_n)
    E_nm = (xp.einsum('n,m,jk->nmjk', P_n, P_n, vvHH)
            - xp.einsum('nm,jk->nmjk', PQ + PQ.T, uvHH)
            + xp.einsum('n,m,jk->nmjk', Q_n, Q_n, uuHH))                  # (N, N, J, J)

    S = 2 * N + N * J
    M = xp.zeros((S, S), dtype=xp.float64)
    M[:N, :N] = Pi * Suu
    M[:N, N:2 * N] = Pi * Suv
    M[N:2 * N, :N] = Pi * Suv
    M[N:2 * N, N:2 * N] = Pi * Svv
    Du = (Pi[:, :, None] * du[None, :, :]).reshape(N, N * J)              # (N, NJ)
    Dv = (Pi[:, :, None] * dv[None, :, :]).reshape(N, N * J)
    M[:N, 2 * N:] = Du
    M[2 * N:, :N] = Du.T
    M[N:2 * N, 2 * N:] = Dv
    M[2 * N:, N:2 * N] = Dv.T
    M[2 * N:, 2 * N:] = (Pi[:, :, None, None] * E_nm).transpose(0, 2, 1, 3) \
        .reshape(N * J, N * J)

    # Eq. (15): the 6 + J directions Pi leaves undetermined, as extra rows.
    C = xp.zeros((6 + J, S), dtype=xp.float64)
    C[0:3, :N] = A.T
    C[3:6, N:2 * N] = A.T
    idx = 2 * N + xp.arange(N) * J
    for j in range(J):
        C[6 + j, idx + j] = 1.0

    # Eq. (15)'s rows only select among solutions of equal loss, so their
    # weight is free; at weight 1 against an M of order sum u^2 they would set
    # the smallest singular values and dominate kappa_vp. Matching M's scale
    # leaves the solution alone and conditions the system.
    w_c = np.sqrt(max(float(xp.max(xp.abs(xp.diag(M)))), np.finfo(float).eps))
    C = C * w_c

    M_reg = M + C.T @ C
    kappa_vp = float(cond2(M_reg, xp))
    if not np.isfinite(kappa_vp) or kappa_vp > 1e8:
        warnings.warn(
            f"vp_aia: the normal system is close to singular "
            f"(kappa_vp={kappa_vp:.3g}); Eq. (12) has a direction the basis "
            f"cannot resolve. Use fewer modes or more phase diversity.",
            stacklevel=2,
        )
    M_reg_inv = xp.linalg.pinv(M_reg)

    D = xp.concatenate([xp.ones(2 * N, dtype=xp.float64),
                        xp.full(N * J, s_mode, dtype=xp.float64)])
    return VPSystem(
        u=u, v=v, P_n=P_n, Q_n=Q_n, Pi=Pi, M_reg_inv=M_reg_inv,
        beta_cov_unit=(M_reg_inv @ M @ M_reg_inv) * D[:, None] * D[None, :],
        kappa_vp=kappa_vp, s_mode=s_mode,
    )


def order_rhs(stack: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
              P_n: np.ndarray, Q_n: np.ndarray, alpha: np.ndarray, system: VPSystem,
              basis: np.ndarray, chunk: int = 1_000_000,
              precision: str | Precision | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Right-hand sides of one order, ``docs/vp_aia.md`` Eqs. (14) and (17).

    Forms the full-model residual of Eq. (8) at the accumulated estimates one
    pixel block at a time, projects it with ``Pi``, and accumulates the sums
    ``L`` of Appendix E, Eq. (E2), against the AIA-solution fields of
    ``system``. The same pass returns ``A' r^(k)`` at every pixel, which the
    pixel corrections of Eq. (17) need. No ``(N, P)`` array is held.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P`` pixels each.
    a, u, v : np.ndarray, shape (P,)
        Accumulated pixel fields ``a_{k-1}``, ``u_{k-1}``, ``v_{k-1}``.
    P_n, Q_n : np.ndarray, shape (N,)
        Accumulated frame coefficients ``P_{n,k-1}``, ``Q_{n,k-1}``.
    alpha : np.ndarray, shape (N, J)
        Accumulated mode coefficients of ``Delta_{n,k-1}``; zeros at ``k = 1``.
    system : VPSystem
        From :func:`build_system`.
    basis : np.ndarray, shape (J, P)
        The modes ``H_j`` the system was built with.
    chunk : int, default 1_000_000
        Pixels accumulated per chunk; bounds memory, not the result.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. Its
        ``accum`` carries the per-chunk ``(N, C)`` blocks.

    Returns
    -------
    rho : np.ndarray, shape (2N+NJ,), float64
        Right-hand side of Eq. (14), in the mode convention of ``system``.
    Atr : np.ndarray, shape (3, P)
        ``A' r^(k)`` at every pixel, with ``A`` of rows ``(1, P_n, Q_n)`` at the
        AIA solution, in ``precision.accum``; zero up to rounding at ``k = 1``.

    Raises
    ------
    ValueError
        If the shapes do not match the system.
    """
    if stack.ndim != 2:
        raise ValueError(f"stack must be 2-D (N, P), got shape {stack.shape}")
    N, P = stack.shape
    J = basis.shape[0]
    if system.P_n.shape != (N,):
        raise ValueError(f"stack has N={N} frames, the system {system.P_n.shape[0]}")
    if system.u.shape != (P,):
        raise ValueError(f"stack has P={P} pixels, the system {system.u.shape[0]}")
    if a.shape != (P,) or u.shape != (P,) or v.shape != (P,):
        raise ValueError(f"a, u and v must have shape ({P},), got {a.shape}, "
                         f"{u.shape} and {v.shape}")
    if alpha.shape != (N, J):
        raise ValueError(f"alpha must have shape ({N}, {J}), got {alpha.shape}")

    xp = get_array_module(stack, a, u, v, basis)
    acc = Precision.of(precision).accum
    P_a = xp.asarray(P_n, dtype=xp.float64).astype(acc)
    Q_a = xp.asarray(Q_n, dtype=xp.float64).astype(acc)
    alpha_a = xp.asarray(alpha, dtype=xp.float64).astype(acc)
    Pi_a = system.Pi.astype(acc)

    bu = xp.zeros(N, dtype=xp.float64)
    bv = xp.zeros(N, dtype=xp.float64)
    ruH = xp.zeros((N, J), dtype=xp.float64)
    rvH = xp.zeros((N, J), dtype=xp.float64)
    A_a = xp.stack([xp.ones(N, dtype=xp.float64), system.P_n, system.Q_n]).astype(acc)
    Atr = xp.empty((3, P), dtype=acc)

    for start in range(0, P, chunk):
        sl = slice(start, start + chunk)
        u_c = u[sl].astype(acc, copy=False)
        v_c = v[sl].astype(acc, copy=False)
        u0_c = system.u[sl].astype(acc, copy=False)
        v0_c = system.v[sl].astype(acc, copy=False)
        H_c = basis[:, sl]                                                # (J, C)
        D_c = alpha_a @ H_c                                               # (N, C)
        piston = xp.outer(P_a, u_c) + xp.outer(Q_a, v_c)
        w_c = xp.outer(P_a, v_c) - xp.outer(Q_a, u_c)
        r_c = stack[:, sl] - (a[sl].astype(acc, copy=False)[None, :]
                              + piston * xp.cos(D_c) + w_c * xp.sin(D_c))  # (N, C), Eq. (8)
        Atr[:, sl] = A_a @ r_c
        r_c = Pi_a @ r_c
        bu += (r_c @ u0_c).astype(xp.float64)
        bv += (r_c @ v0_c).astype(xp.float64)
        ruH += weighted_gram(r_c, u0_c, H_c).astype(xp.float64)
        rvH += weighted_gram(r_c, v0_c, H_c).astype(xp.float64)

    # Eq. (E2), in the mode convention of the system.
    s = system.s_mode
    rho = xp.concatenate([bu, bv, (system.P_n[:, None] * rvH
                                   - system.Q_n[:, None] * ruH).reshape(-1) * s])
    return rho, Atr


def solve_order(system: VPSystem, rho: np.ndarray) -> VPSolution:
    """Solve one order, ``docs/vp_aia.md`` Eq. (16), ``beta = (M + C'C)^-1 rho``.

    By the Frisch-Waugh-Lovell theorem this gives the same unknowns as a joint
    fit of Eq. (9) that includes the pixel corrections; those follow from
    Eq. (17).

    Parameters
    ----------
    system : VPSystem
        From :func:`build_system`.
    rho : np.ndarray, shape (2N+NJ,)
        The first output of :func:`order_rhs`.

    Returns
    -------
    VPSolution
    """
    N = system.P_n.shape[0]
    beta = system.M_reg_inv @ rho
    J = (beta.shape[0] - 2 * N) // N
    return VPSolution(P_corr=beta[:N], Q_corr=beta[N:2 * N],
                      alpha=(beta[2 * N:] * system.s_mode).reshape(N, J))
