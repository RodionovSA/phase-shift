# src/phase_shift/methods/vp_system.py
"""Normal system of the VP-AIA first-order fit, ``docs/vp_aia.md`` Eqs. (22)-(25).

Eq. (20) holds at every pixel and contains only the ``2N + NJ`` unknowns
``P_n^(1)``, ``Q_n^(1)`` and ``alpha_nj``, so the least-squares fit reduces to
one dense system whose size does not grow with the pixel count. Eq. (23) turns
every frame-indexed pixel sum into sums without a frame index, which
:func:`fit_frame_and_coeffs` accumulates in a single chunked pass.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from ..backend import Precision, get_array_module
from .diagnostics import cond2


@dataclass
class VPSolution:
    """Solution of ``docs/vp_aia.md`` Eq. (25).

    Attributes
    ----------
    P_corr, Q_corr : np.ndarray, shape (N,), float64
        First-order frame corrections ``P_n^(1)``, ``Q_n^(1)``, satisfying the
        six conditions of Eq. (24).
    alpha : np.ndarray, shape (N, J), float64
        Phase-step coefficients ``alpha_nj`` of Eq. (13), with zero frame mean
        by Eq. (24).
    beta_cov_unit : np.ndarray, shape (2N+NJ, 2N+NJ), float64
        ``(M + C'C)^-1 M (M + C'C)^-1``, the covariance of ``beta`` per unit
        ``sigma_0^2``; see §"Noise of the fitted unknowns". Ordered as
        ``beta``: the ``P_n^(1)``, then the ``Q_n^(1)``, then ``alpha_nj``
        with ``j`` fastest, and in the same mode convention as ``alpha``.
    kappa_vp : float
        Condition number of ``M + C'C``, of order 100 for a well-posed fit and
        independent of the pixel count. Large values mean Eq. (20) has a
        direction the modes cannot resolve -- the basis has outrun what the
        fringe pattern carries.
    """

    P_corr: np.ndarray
    Q_corr: np.ndarray
    alpha: np.ndarray
    beta_cov_unit: np.ndarray
    kappa_vp: float


def max_modes(N: int, K: int) -> int:
    """Largest identifiable number of spatial modes, ``docs/vp_aia.md`` Eq. (15).

    ``floor((N-3)(K-2)/(N-1))``, the count bound on ``J``. It is necessary,
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
    return int(((N - 3) * (K - 2)) // (N - 1))


def _pixel_sums(stack: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
                P_n: np.ndarray, Q_n: np.ndarray, basis: np.ndarray, chunk: int,
                acc: np.dtype, xp) -> tuple:
    """Accumulate every sum of ``docs/vp_aia.md`` Eqs. (22)-(23) in one pass.

    The residual of Eq. (16) is formed one pixel block at a time, so no
    ``(N, P)`` array is held. Each block is built in ``acc``.

    Returns
    -------
    Suu, Svv, Suv : float
        The three scalar sums.
    uuH, vvH, uvH : np.ndarray, shape (J,), float64
        The three sums per spatial function.
    uuHH, vvHH, uvHH : np.ndarray, shape (J, J), float64
        The three sums per pair of spatial functions.
    bu, bv : np.ndarray, shape (N,), float64
        ``sum u r_n`` and ``sum v r_n``.
    ruH, rvH : np.ndarray, shape (N, J), float64
        ``sum u H_j r_n`` and ``sum v H_j r_n``.
    """
    N, P = stack.shape
    J = basis.shape[0]
    # The (N,) coefficients stay float64, but the blocks below are pixel-sized:
    # accum copies keep a float64 (N,) operand from widening them.
    P_a, Q_a = P_n.astype(acc), Q_n.astype(acc)

    Suu = Svv = Suv = 0.0
    uuH = xp.zeros(J, dtype=xp.float64)
    vvH = xp.zeros(J, dtype=xp.float64)
    uvH = xp.zeros(J, dtype=xp.float64)
    uuHH = xp.zeros((J, J), dtype=xp.float64)
    vvHH = xp.zeros((J, J), dtype=xp.float64)
    uvHH = xp.zeros((J, J), dtype=xp.float64)
    bu = xp.zeros(N, dtype=xp.float64)
    bv = xp.zeros(N, dtype=xp.float64)
    ruH = xp.zeros((N, J), dtype=xp.float64)
    rvH = xp.zeros((N, J), dtype=xp.float64)

    for start in range(0, P, chunk):
        sl = slice(start, start + chunk)
        u_c, v_c, H_c = u[sl], v[sl], basis[:, sl]                   # (C,), (J, C)
        uu_c, vv_c, uv_c = u_c * u_c, v_c * v_c, u_c * v_c
        r_c = stack[:, sl] - (a[sl][None, :] + xp.outer(P_a, u_c)
                              + xp.outer(Q_a, v_c))                  # (N, C), Eq. (16)

        Suu += float(xp.sum(uu_c, dtype=acc))
        Svv += float(xp.sum(vv_c, dtype=acc))
        Suv += float(xp.sum(uv_c, dtype=acc))
        uuH += (H_c @ uu_c).astype(xp.float64)
        vvH += (H_c @ vv_c).astype(xp.float64)
        uvH += (H_c @ uv_c).astype(xp.float64)
        # (H_c * w) @ H_c.T, never H_c[:, None] * H_c[None]: the latter is a
        # (J, J, C) temporary.
        uuHH += ((H_c * uu_c) @ H_c.T).astype(xp.float64)
        vvHH += ((H_c * vv_c) @ H_c.T).astype(xp.float64)
        uvHH += ((H_c * uv_c) @ H_c.T).astype(xp.float64)
        bu += (r_c @ u_c).astype(xp.float64)
        bv += (r_c @ v_c).astype(xp.float64)
        ruH += ((r_c * u_c[None, :]) @ H_c.T).astype(xp.float64)
        rvH += ((r_c * v_c[None, :]) @ H_c.T).astype(xp.float64)

    return Suu, Svv, Suv, uuH, vvH, uvH, uuHH, vvHH, uvHH, bu, bv, ruH, rvH


def fit_frame_and_coeffs(stack: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
                         P_n: np.ndarray, Q_n: np.ndarray, basis: np.ndarray,
                         chunk: int = 1_000_000,
                         precision: str | Precision | None = None) -> VPSolution:
    """Fit the first-order frame corrections and phase-step coefficients.

    Solves ``docs/vp_aia.md`` Eq. (25), ``beta = (M + C'C)^-1 b``, the
    least-squares fit of Eq. (20) over all pixels. By the Frisch-Waugh-Lovell
    theorem this gives the same unknowns as a joint fit of Eq. (17) that
    includes the pixel corrections; those follow from Eq. (26).

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P = H*W`` pixels each.
    a, u, v : np.ndarray, shape (P,)
        Zeroth-order background and quadrature fields. They must come from a
        pixel step at ``(P_n, Q_n)`` (:func:`phase_shift.methods.steps.pixel_step`)
        and be normalized by
        :func:`phase_shift.methods.gauge.normalize_quadrature_frame`, as
        §"VP-AIA algorithm" step 1 prescribes. Eq. (21) assumes ``Pi`` leaves
        the residual column unchanged, which holds only for the pixel-step
        residual; fields that merely fit the data well do not satisfy it, and
        the coefficients come out wrong rather than merely noisy.
    P_n, Q_n : np.ndarray, shape (N,)
        Zeroth-order per-frame coefficients, ``docs/vp_aia.md`` Eq. (5).
    basis : np.ndarray, shape (J, P)
        Spatial modes ``H_j`` of Eq. (13), e.g. from
        :func:`phase_shift.basis.spatial_basis`.
    chunk : int, default 1_000_000
        Pixels accumulated per chunk; bounds memory, not the result.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. Its
        ``accum`` carries the per-chunk ``(N, C)`` blocks; the
        ``(2N+NJ)``-sized system is float64.

    Returns
    -------
    VPSolution

    Raises
    ------
    ValueError
        If the shapes are inconsistent, if ``N < 5`` while ``J > 0`` (no mode
        is identifiable below five frames, §"Spatial modes"), or if ``J``
        exceeds :func:`max_modes`.

    Warns
    -----
    UserWarning
        If ``M + C'C`` is close to singular, so Eq. (20) has an undetermined
        direction beyond those Eq. (24) fixes.
    """
    if stack.ndim != 2:
        raise ValueError(f"stack must be 2-D (N, P), got shape {stack.shape}")
    if a.shape != u.shape or a.shape != v.shape:
        raise ValueError(f"a, u and v must have the same shape, got {a.shape}, "
                         f"{u.shape} and {v.shape}")
    if a.ndim != 1 or a.shape[0] != stack.shape[1]:
        raise ValueError(f"a, u and v must be 1-D of length {stack.shape[1]}, "
                         f"got shape {a.shape}")
    if P_n.shape != (stack.shape[0],) or Q_n.shape != (stack.shape[0],):
        raise ValueError(f"P_n and Q_n must have shape ({stack.shape[0]},), got "
                         f"{P_n.shape} and {Q_n.shape}")
    if basis.ndim != 2 or basis.shape[1] != stack.shape[1]:
        raise ValueError(f"basis must be 2-D (J, {stack.shape[1]}), got shape {basis.shape}")

    xp = get_array_module(stack, a, u, v, basis)
    N, K = stack.shape
    J = basis.shape[0]
    if J > 0 and N < 5:
        raise ValueError(f"vp_aia needs at least 5 frames to identify a spatial mode, "
                         f"got N={N}; see docs/vp_aia.md §'Spatial modes'")
    if J > max_modes(N, K):
        raise ValueError(f"basis has J={J} modes, above the identifiable maximum "
                         f"{max_modes(N, K)} for N={N} frames and K={K} pixels "
                         f"(docs/vp_aia.md Eq. 15)")

    acc = Precision.of(precision).accum
    P_n = xp.asarray(P_n, dtype=xp.float64)
    Q_n = xp.asarray(Q_n, dtype=xp.float64)

    (Suu, Svv, Suv, uuH, vvH, uvH, uuHH, vvHH, uvHH,
     bu, bv, ruH, rvH) = _pixel_sums(stack, a, u, v, P_n, Q_n, basis, chunk, acc, xp)

    # Solve in the mode convention of Eq. (30), <H_j H_k> = delta_jk, rather
    # than the unit-norm rows spatial_basis returns. Without it the frame block
    # of M grows as sum u^2 ~ K while the coefficient block stays O(1), and
    # kappa_vp tracks the pixel count instead of the fit. Scaling the sums is
    # the same as scaling the rows, and costs no (J, P) copy; the coefficients
    # are scaled back on the way out.
    s_mode = float(np.sqrt(K))
    uuH, vvH, uvH = uuH * s_mode, vvH * s_mode, uvH * s_mode
    uuHH, vvHH, uvHH = uuHH * s_mode ** 2, vvHH * s_mode ** 2, uvHH * s_mode ** 2
    ruH, rvH = ruH * s_mode, rvH * s_mode

    # Eq. (18): the residual-maker of the pixel-step fit.
    A = xp.stack([xp.ones(N, dtype=xp.float64), P_n, Q_n], axis=1)        # (N, 3)
    Pi = xp.eye(N, dtype=xp.float64) - A @ xp.linalg.pinv(A.T @ A) @ A.T  # (N, N)

    # Eq. (23): every sum carrying a frame index, from the sums above.
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

    b = xp.concatenate([bu, bv,
                        (P_n[:, None] * rvH - Q_n[:, None] * ruH).reshape(-1)])

    # Eq. (24): the 6 + J directions Pi leaves undetermined, as extra rows.
    C = xp.zeros((6 + J, S), dtype=xp.float64)
    C[0:3, :N] = A.T
    C[3:6, N:2 * N] = A.T
    idx = 2 * N + xp.arange(N) * J
    for j in range(J):
        C[6 + j, idx + j] = 1.0

    # Eq. (24)'s rows only select among solutions of equal loss, so their
    # weight is free; at weight 1 against an M of order sum u^2 they would set
    # the smallest singular values and dominate kappa_vp. Matching M's scale
    # leaves the solution alone and conditions the system.
    w_c = np.sqrt(max(float(xp.max(xp.abs(xp.diag(M)))), np.finfo(float).eps))
    C = C * w_c

    M_reg = M + C.T @ C
    kappa_vp = float(cond2(M_reg, xp))
    if not np.isfinite(kappa_vp) or kappa_vp > 1e8:
        warnings.warn(
            f"vp_aia: the first-order system is close to singular "
            f"(kappa_vp={kappa_vp:.3g}); Eq. (20) has a direction the basis "
            f"cannot resolve. Use fewer modes or more phase diversity.",
            stacklevel=2,
        )
    M_reg_inv = xp.linalg.pinv(M_reg)
    beta = M_reg_inv @ b                                                  # Eq. (25)

    # Back to the caller's mode convention: alpha is fit against H * s_mode.
    D = xp.concatenate([xp.ones(2 * N, dtype=xp.float64),
                        xp.full(N * J, s_mode, dtype=xp.float64)])
    return VPSolution(
        P_corr=beta[:N], Q_corr=beta[N:2 * N],
        alpha=(beta[2 * N:] * D[2 * N:]).reshape(N, J),
        beta_cov_unit=(M_reg_inv @ M @ M_reg_inv) * D[:, None] * D[None, :],
        kappa_vp=kappa_vp,
    )
