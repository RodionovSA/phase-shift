# src/phase_shift/errors.py
"""Shared phase-error derivations for the AIA family.

Computed from a method's output (``b, phi, delta, g``) rather than inside the
solve: ``docs/aia.md`` §"Phase-error covariance" for the Eq. (26) baseline and
the exact Stage-2/3 corrections, Eqs. (40) and (45); ``docs/sf_aia.md``
§"Noise of the corrected solve" for the step-field term. Each method exposes
its own map through :meth:`phase_shift.methods.base.MethodParam.phi_error`;
the functions here are the pieces those share.
"""

from types import ModuleType

import numpy as np

from .backend import Precision


def _frame_side(delta: np.ndarray, g: np.ndarray, xp: ModuleType) -> tuple[np.ndarray, ...]:
    """Frame-side quantities of ``docs/aia.md`` Eqs. (23)-(25) and (34).

    All are ``(2, 2)`` or ``(2, N)`` and stay in float64 whatever the working
    dtype of the fields, as the methods' own normal-equation solves do.

    Parameters
    ----------
    delta, g : np.ndarray, shape (N,)
        Per-frame phase step in radians, and per-frame gain.
    xp : module
        ``numpy`` or ``cupy``.

    Returns
    -------
    C_inv : np.ndarray, shape (2, 2)
        Inverse of Eq. (25)'s frame-side covariance.
    k : np.ndarray, shape (2, N)
        Eq. (34)'s leverage vectors, one column per frame.
    w_n : np.ndarray, shape (2, N)
        ``(-sin delta_n, cos delta_n)``, the tangential direction of Eq. (36).
    """
    d = xp.stack([xp.cos(delta), xp.sin(delta)])                          # (2, N)
    w_n = xp.stack([-d[1], d[0]])                                         # (2, N)
    R = xp.mean(g[None, :] * d, axis=1)                                   # (2,)
    C = xp.mean(g[None, :] ** 2 * d[:, None, :] * d[None, :, :], axis=2) \
        - xp.outer(R, R)                                                  # (2, 2), Eq. (25)
    C_inv = xp.linalg.pinv(C)
    k = (C_inv @ (g[None, :] * d - R[:, None])) / delta.shape[0]          # (2, N), Eq. (34)
    return C_inv, k, w_n


def _frame_step_weights(u: np.ndarray, v: np.ndarray, sigma0_sq: np.ndarray,
                        xp: ModuleType, acc: np.dtype
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``docs/aia.md`` Eq. (38): the frame step's per-pixel weight and its covariance.

    Parameters
    ----------
    u, v : np.ndarray, shape (H, W)
        Quadrature fields, ``docs/aia.md`` Eq. (2), in the working dtype.
    sigma0_sq : np.ndarray, shape (H, W)
        Per-pixel noise variance, Eq. (32), in the working dtype.
    xp : module
    acc : dtype
        ``precision.accum``; the dtype the pixel sums accumulate in.

    Returns
    -------
    l0, l1 : np.ndarray, shape (H, W)
        Components of ``ell(x, y)``, in the working dtype.
    M : np.ndarray, shape (2, 2), float64
        ``sum ell ell^T sigma_0^2``, the ``{P, Q}`` block of Eq. (35).
    """
    ones = xp.ones((), dtype=acc)
    su = xp.sum(u, dtype=acc)
    sv = xp.sum(v, dtype=acc)
    A_ps = xp.stack([
        xp.stack([ones * u.size, su, sv]),
        xp.stack([su, xp.sum(u * u, dtype=acc), xp.sum(u * v, dtype=acc)]),
        xp.stack([sv, xp.sum(u * v, dtype=acc), xp.sum(v * v, dtype=acc)]),
    ]).astype(xp.float64)                                                 # (3, 3)
    Z = xp.linalg.pinv(A_ps)[1:].astype(u.dtype)                          # (2, 3)

    l0 = Z[0, 0] + Z[0, 1] * u + Z[0, 2] * v                              # (H, W)
    l1 = Z[1, 0] + Z[1, 1] * u + Z[1, 2] * v                              # (H, W)
    m00 = xp.sum(l0 * l0 * sigma0_sq, dtype=acc)
    m01 = xp.sum(l0 * l1 * sigma0_sq, dtype=acc)
    m11 = xp.sum(l1 * l1 * sigma0_sq, dtype=acc)
    M = xp.stack([xp.stack([m00, m01]), xp.stack([m01, m11])]).astype(xp.float64)
    return l0, l1, M


def aia_phi_error_parts(b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                        fit_gain: bool, sigma0: np.ndarray, simplified: bool,
                        xp: ModuleType, precision: str | Precision | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """``docs/aia.md``'s AIA phase-error map, exact to first order in the noise.

    The Eq. (26) baseline when ``simplified``; otherwise Eq. (45) when the
    gain was fitted with the steps, and Eq. (40) when only the steps were.
    Returns Eq. (34)'s leverage vectors alongside, so a method that builds on
    this map does not rebuild them.

    Parameters
    ----------
    b, phi : np.ndarray, shape (H, W)
        Fitted fringe amplitude and phase in radians.
    delta, g : np.ndarray, shape (N,)
        Fitted per-frame phase step and gain.
    fit_gain : bool
        Whether ``g`` was fitted jointly with the steps: Stage 3 rather than
        Stage 2.
    sigma0 : np.ndarray, shape (H, W), or float
        Per-pixel noise standard deviation, Eq. (32).
    simplified : bool
        Return the Eq. (26) baseline alone, dropping the ``O(1/N_p)``
        fitted-step correction.
    xp : module
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`.

    Returns
    -------
    phi_var : np.ndarray, shape (H, W)
        ``sigma_Phi(x, y)^2``, in squared radians.
    k : np.ndarray, shape (2, N), float64
        Eq. (34)'s leverage vectors.
    """
    work = b.dtype
    acc = Precision.of(precision).accum
    delta64 = xp.asarray(delta, dtype=xp.float64)
    g64 = xp.asarray(g, dtype=xp.float64)
    N = delta64.shape[0]
    C_inv, k, w_n = _frame_side(delta64, g64, xp)

    eps = xp.asarray(xp.finfo(xp.float64).eps, dtype=work)
    b2 = xp.maximum(b ** 2, eps)
    w = xp.stack([xp.sin(phi), xp.cos(phi)])                              # (2, H, W)

    # Eq. (26) split into its noise-free factor and sigma_0^2, so the
    # Stage-2/3 corrections below need no division by the noise.
    unit_var = xp.einsum('ahw,ab,bhw->hw', w, C_inv.astype(work), w) / (N * b2)
    sigma0_sq = sigma0 ** 2
    if simplified:
        return unit_var * sigma0_sq, k

    u = b * xp.cos(phi)                                                   # (H, W), Eq. (2)
    v = -b * xp.sin(phi)
    l0, l1, M = _frame_step_weights(u, v, sigma0_sq * xp.ones_like(b), xp, acc)
    M_w = M.astype(work)

    # The correction is O(1/N_p), so it is accumulated on its own rather than
    # as the `1 - 2 q.ell` factor of Eqs. (40)/(45), which would cancel away
    # in the working dtype on a large field.
    if fit_gain:
        # Eq. (45): Pi_n = I_2, so Gamma does not depend on the frame.
        q_dot_l = u * l0 + v * l1
        qMq = (M_w[0, 0] * u + M_w[0, 1] * v) * u + (M_w[1, 0] * u + M_w[1, 1] * v) * v
        return unit_var * (sigma0_sq + (qMq - 2 * sigma0_sq * q_dot_l)), k

    # Eq. (40) with Pi_n = w_n w_n^T: one frame at a time, no (N, H, W) array.
    w_n_w = w_n.astype(work)
    k_w = k.astype(work)
    mu = xp.einsum('an,ab,bn->n', w_n, M, w_n).astype(work)               # (N,), w_n^T M w_n
    correction = xp.zeros_like(b2)
    for n in range(N):
        t = w_n_w[0, n] * u + w_n_w[1, n] * v                             # (H, W), w_n^T q
        r = w_n_w[0, n] * l0 + w_n_w[1, n] * l1                           # (H, W), w_n^T ell
        w_dot_k = w[0] * k_w[0, n] + w[1] * k_w[1, n]                     # (H, W)
        correction += w_dot_k ** 2 * (t ** 2 * mu[n] - 2 * sigma0_sq * t * r)
    return unit_var * sigma0_sq + correction / b2, k


def _step_field_sensitivity(u: np.ndarray, v: np.ndarray, P_n: np.ndarray,
                            Q_n: np.ndarray, sl: slice) -> np.ndarray:
    """``docs/sf_aia.md`` Eq. (T7)'s ``w_n = u Q_n - v P_n`` on one pixel block.

    Parameters
    ----------
    u, v : np.ndarray, shape (P,)
        Quadrature fields, flattened, ``docs/aia.md`` Eq. (2).
    P_n, Q_n : np.ndarray, shape (N,)
        Per-frame quadrature coefficients, ``docs/aia.md`` Eq. (3), in ``u``'s
        dtype so the returned block is not widened.
    sl : slice
        Pixels of this block.

    Returns
    -------
    np.ndarray, shape (N, C)
    """
    return Q_n[:, None] * u[None, sl] - P_n[:, None] * v[None, sl]


def step_field_phi_error(b: np.ndarray, phi: np.ndarray, delta: np.ndarray,
                         g: np.ndarray, fit_gain: bool, sigma0: np.ndarray,
                         simplified: bool, basis: np.ndarray, xp: ModuleType,
                         chunk: int = 65_536,
                         precision: str | Precision | None = None) -> np.ndarray:
    """``docs/sf_aia.md``'s step-field phase-error map, Eq. (E7).

    The AIA map of :func:`aia_phi_error_parts` plus the variance the fitted
    step field adds, §"Noise of the corrected solve". Both terms are exact to
    first order in the noise; §"Validity" states what their sum leaves out.

    Parameters
    ----------
    b, phi : np.ndarray, shape (H, W)
        Fitted fringe amplitude and phase in radians.
    delta, g : np.ndarray, shape (N,)
        Fitted per-frame phase step and gain.
    fit_gain : bool
        Whether ``g`` was fitted jointly with the steps.
    sigma0 : np.ndarray, shape (H, W), or float
        Per-pixel noise standard deviation, ``docs/aia.md`` Eq. (32).
    simplified : bool
        Return the Eq. (26) baseline alone, dropping both the fitted-step
        correction and the step-field term; both are ``O(1/N_p)``.
    basis : np.ndarray, shape (J, P)
        The step field's basis, from :func:`phase_shift.basis.spatial_basis`.
        An empty basis reduces this to the plain-``aia`` result.
    xp : module
    chunk : int, default 65536
        Pixels reduced per block; bounds memory, not the result.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. The term
        is a difference of larger quantities, so ``precision.accum`` carries
        every pixel-sized block; the ``(N*J, N*J)`` matrices stay float64.

    Notes
    -----
    Two passes over the pixels: one for Eq. (E7)'s matrices ``E_nm`` and
    ``G^(n)``, one for its quadratic form. No ``(N, P)`` array is held.

    Returns
    -------
    np.ndarray, shape (H, W)
        ``sigma_Phi(x, y)``, in radians.
    """
    acc = Precision.of(precision).accum
    phi_var, k = aia_phi_error_parts(b, phi, delta, g, fit_gain, sigma0, simplified, xp,
                                     precision=precision)
    if simplified:
        return xp.sqrt(phi_var)

    H, W = phi.shape
    basis = xp.asarray(basis, dtype=acc)                                  # (J, P)
    J = basis.shape[0]
    if J == 0:
        return xp.sqrt(phi_var)

    N = delta.shape[0]
    delta64 = xp.asarray(delta, dtype=xp.float64)
    g64 = xp.asarray(g, dtype=xp.float64)
    P_n = g64 * xp.cos(delta64)                                           # (N,), Eq. (3)
    Q_n = g64 * xp.sin(delta64)
    phi_a = xp.asarray(phi, dtype=acc).reshape(-1)                        # (P,)
    b_a = xp.asarray(b, dtype=acc).reshape(-1)
    u = b_a * xp.cos(phi_a)                                               # (P,), Eq. (2)
    v = -b_a * xp.sin(phi_a)
    del phi_a, b_a
    P_tot = u.shape[0]
    sigma0_sq = xp.asarray(sigma0, dtype=acc) ** 2
    sigma0_sq = (sigma0_sq.reshape(-1) if sigma0_sq.ndim else sigma0_sq) \
        * xp.ones(P_tot, dtype=acc)                                       # (P,)
    # The (N,) coefficients stay float64, but the blocks below are pixel-sized:
    # they take accum copies so a float64 (N,) operand cannot widen them.
    P_a, Q_a = P_n.astype(acc), Q_n.astype(acc)
    blocks = [slice(s, min(s + chunk, P_tot)) for s in range(0, P_tot, chunk)]

    # First pass: Eq. (E7)'s noise-weighted cross-frame matrices D_n^T L D_m,
    # and the unweighted blocks that are Eq. (E1)'s own normal matrices G^(n).
    DLD = xp.zeros((N * J, N * J), dtype=xp.float64)
    G = xp.zeros((N, J, J), dtype=xp.float64)
    for sl in blocks:
        w_field = _step_field_sensitivity(u, v, P_a, Q_a, sl)             # (N, C)
        rows = (w_field[:, None, :] * basis[None, :, sl]).reshape(N * J, -1)
        DLD += ((rows * sigma0_sq[None, sl]) @ rows.T).astype(xp.float64)
        for n in range(N):
            block = rows[n * J:(n + 1) * J]
            G[n] += (block @ block.T).astype(xp.float64)
    G_inv = xp.linalg.pinv(G)                                             # (N, J, J)
    K = xp.einsum('nja,namb,mbk->njmk', G_inv, DLD.reshape(N, J, N, J), G_inv)

    # Pi removes from a column over frames its pixel-step fit, docs/vp_aia.md Eq. (18).
    A = xp.stack([xp.ones(N, dtype=xp.float64), P_n, Q_n], axis=1)        # (N, 3)
    Pi = xp.eye(N, dtype=xp.float64) - A @ xp.linalg.pinv(A.T @ A) @ A.T  # (N, N)
    weight = (Pi[:, None, :, None] * K).reshape(N * J, N * J).astype(acc)  # (N*J, N*J)

    # Second pass: t_n = s_n w_n, frame-centered by Eq. (E4), then Eq. (E7)'s
    # quadratic form. With w = (sin Phi, cos Phi) = (-v, u)/b, Eq. (E7)'s
    # s_n = -(w . k_n)/b is (k_n0 v - k_n1 u)/b^2.
    k_a = xp.asarray(k, dtype=acc)
    floor = xp.asarray(xp.finfo(xp.float64).eps, dtype=acc)
    extra = xp.empty(P_tot, dtype=acc)
    for sl in blocks:
        w_field = _step_field_sensitivity(u, v, P_a, Q_a, sl)             # (N, C)
        b2 = xp.maximum(u[sl] ** 2 + v[sl] ** 2, floor)                   # (C,)
        s_n = (k_a[0][:, None] * v[None, sl] - k_a[1][:, None] * u[None, sl]) / b2[None, :]
        t = s_n * w_field                                                 # (N, C)
        t = t - xp.mean(t, axis=0, keepdims=True)
        z = (t[:, None, :] * basis[None, :, sl]).reshape(N * J, -1)
        extra[sl] = xp.sum(z * (weight @ z), axis=0)

    return xp.sqrt(xp.maximum(phi_var + extra.reshape(H, W).astype(b.dtype), 0))
