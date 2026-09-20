# src/phase_shift/errors.py
"""Shared phase-error derivations for the AIA family.

Computed from a method's output (``b, phi, delta, g``) rather than inside the
solve: ``docs/aia.md`` §"Phase-error covariance" for the baseline and the
Stage-2/3 corrections, ``docs/sf_aia.md`` §"Noise of the corrected solve" for
the step-field term. Each method exposes its own map through
:meth:`phase_shift.methods.base.MethodParam.phi_error`; the functions here are
the pieces those share.
"""

from types import ModuleType

import numpy as np


def aia_phi_error_parts(b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                         fit_gain: bool, sigma0: np.ndarray, simplified: bool,
                         xp: ModuleType) -> tuple[np.ndarray, np.ndarray | None]:
    """``docs/aia.md``'s AIA phase-error map, Eq. (26) plus the Stage-2/3 correction.

    Returns ``w_dot_k`` alongside the variance so
    :func:`step_field_phi_error` can reuse Eq. (34)'s leverage vectors
    instead of rebuilding them.

    Parameters
    ----------
    b, phi : np.ndarray, shape (H, W)
    delta, g : np.ndarray, shape (N,)
    fit_gain : bool
        Include the fitted-gain sensitivity of Eq. (39) (Stage 3) as well as
        the phase-step one of Eq. (33) (Stage 2).
    sigma0 : np.ndarray, shape (H, W), or float
    simplified : bool
        Return the Eq. (26) baseline alone.
    xp : module

    Returns
    -------
    phi_var : np.ndarray, shape (H, W)
        ``sigma_Phi(x, y)^2``: Eq. (26) alone when ``simplified``, else with
        the Stage-2/3 correction.
    w_dot_k : np.ndarray, shape (N, H, W), or None
        Eq. (34)'s leverage vectors dotted with ``(sin(phi), cos(phi))``, in
        the working dtype; ``None`` when ``simplified``.
    """
    work = b.dtype
    delta64 = xp.asarray(delta, dtype=xp.float64)
    g64 = xp.asarray(g, dtype=xp.float64)
    N = delta64.shape[0]

    # Frame-side quantities (Eq. 23-25) -- small, float64 regardless
    # of b/phi's working dtype, same convention as phase.methods.aia's own
    # normal-equation solves. Cast to `work` right before use against a big
    # (H, W)/(N, H, W) array, same as pixel_step's own
    # `pinv(A).astype(work_dtype) @ stack` -- otherwise ordinary promotion
    # silently turns every big array below float64 too.
    d = xp.stack([xp.cos(delta64), xp.sin(delta64)])                     # (2, N)
    R = xp.mean(g64[None, :] * d, axis=1)                                # (2,)
    C = xp.mean(g64[None, :] ** 2 * d[:, None, :] * d[None, :, :], axis=2) \
        - xp.outer(R, R)                                                 # (2, 2)
    C_inv = xp.linalg.pinv(C)
    C_inv_w = C_inv.astype(work)

    eps = xp.asarray(xp.finfo(xp.float64).eps, dtype=work)
    b2 = xp.maximum(b ** 2, eps)
    w = xp.stack([xp.sin(phi), xp.cos(phi)])                             # (2, H, W)

    # Eq. (26): Stage-1 baseline.
    phi_var = sigma0 ** 2 / (N * b2) * xp.einsum('ahw,ab,bhw->hw', w, C_inv_w, w)

    if simplified:
        return phi_var, None

    Np = b.size
    b2_mean = xp.mean(b2)
    sigma_eff_sq = xp.mean(sigma0 ** 2 * b2) / b2_mean                    # Eq. (37)
    g_w = g64.astype(work)
    delta_w = delta64.astype(work)

    r = g64[None, :] * d - R[:, None]                                    # (2, N)
    k = ((C_inv @ r) / N).astype(work)                                   # (2, N), Eq. (34)
    w_dot_k = xp.einsum('ahw,an->nhw', w, k)                             # (N, H, W)

    dI_ddelta = -g_w[:, None, None] * b[None, :, :] \
        * xp.sin(phi[None, :, :] + delta_w[:, None, None])               # (N, H, W), Eq. (33)
    term = dI_ddelta ** 2 / g_w[:, None, None] ** 2
    if fit_gain:
        dI_dg = b[None, :, :] * xp.cos(phi[None, :, :] + delta_w[:, None, None])  # Eq. (39)
        term = term + dI_dg ** 2

    # Eq. (33)/(39) sensitivities against the Eq. (37)/(40) variances.
    correction = (2 * sigma_eff_sq / (Np * b2_mean * b2)) \
        * xp.sum(term * w_dot_k ** 2, axis=0)
    return phi_var + correction, w_dot_k


def step_field_phi_error(b: np.ndarray, phi: np.ndarray, delta: np.ndarray,
                               g: np.ndarray, fit_gain: bool, sigma0,
                               simplified: bool, basis: np.ndarray, xp) -> np.ndarray:
    """``docs/sf_aia.md``'s step-field phase-error map.

    The AIA map of :func:`aia_phi_error_parts` combined with the step field's
    own contribution, §"Noise of the corrected solve". Reuses Eq. (34)'s
    leverage vectors ``k`` rather than a per-pixel 3x3 solve.

    Parameters
    ----------
    b, phi : np.ndarray, shape (H, W)
    delta, g : np.ndarray, shape (N,)
    fit_gain : bool
    sigma0 : np.ndarray, shape (H, W), or float
    simplified : bool
        Return the Eq. (26) baseline alone, dropping both the Stage-2/3
        correction and the step-field term; both are ``O(1/N_p)``.
    basis : np.ndarray, shape (J, P)
        The step field's basis, from :func:`phase_shift.basis.spatial_basis`.
        An empty basis makes the discount identically zero, reducing this to
        the plain-``aia`` result.
    xp : module

    Returns
    -------
    np.ndarray, shape (H, W)
        ``sigma_Phi(x, y)``, in radians.
    """
    phi_var, w_dot_k = aia_phi_error_parts(b, phi, delta, g, fit_gain, sigma0, simplified, xp)
    if simplified:
        return xp.sqrt(phi_var)

    H, W = phi.shape
    work = b.dtype
    basis = basis.astype(work, copy=False)                                # (J, P)
    J = basis.shape[0]
    if J == 0:
        return xp.sqrt(phi_var)

    delta64 = xp.asarray(delta, dtype=xp.float64)
    g64 = xp.asarray(g, dtype=xp.float64)
    P_n = (g64 * xp.cos(delta64)).astype(work)
    Q_n = (g64 * xp.sin(delta64)).astype(work)
    u = (b * xp.cos(phi)).reshape(-1)
    v = (-b * xp.sin(phi)).reshape(-1)
    w_n = Q_n[:, None] * u[None, :] - P_n[:, None] * v[None, :]           # (N, P), Eq. (T7)
    w_n_sq = w_n ** 2

    # Eq. (E1)'s w_n^2-weighted Gram matrix, one (J, J) solve per frame, in
    # the working dtype: G reduces over the big (N, P) data, so its inputs
    # are cast down rather than up.
    G = xp.einsum('np,jp,kp->njk', w_n_sq, basis, basis)                  # (N, J, J)
    G_inv = xp.linalg.pinv(G)
    h = w_n_sq * xp.einsum('jp,njk,kp->np', basis, G_inv, basis)          # (N, P)
    h = h.reshape(-1, H, W)

    b2 = xp.maximum(b ** 2, xp.asarray(xp.finfo(xp.float64).eps, dtype=work))
    discount = (sigma0 ** 2 / b2) * xp.sum(h * w_dot_k ** 2, axis=0)

    return xp.sqrt(xp.maximum(phi_var - discount, 0))
