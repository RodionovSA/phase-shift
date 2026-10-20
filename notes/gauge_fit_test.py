"""Test: fitting the 4 gauge parameters (shift p,q; symmetric det-1 shear S) in VP-AIA.

Variants, noiseless, physical truth:
  conv   : current conventions (Appendix B), 1st and 2nd order pass
  gauge  : 1st order bilinear (VarPro over gamma), 2nd order with linear gamma^(1)
  gauge0 : same 1st order, 2nd order without gamma^(1)
"""
import numpy as np
from phase_shift.methods.aia import aia
from phase_shift.methods.steps import pixel_step
from phase_shift.methods.gauge import normalize_quadrature_frame
from phase_shift.basis import spatial_basis

rng = np.random.default_rng(1)
H, W, N = 30, 40, 7
Y, X = np.mgrid[0:H, 0:W].astype(float)
X = (X - W / 2) / (W / 2); Y = (Y - H / 2) / (H / 2)
Hm = spatial_basis(H, W, "poly", np, precision="double", degree=2)  # (5, K)
J, K = Hm.shape

a_t = (1.0 + 0.2 * X + 0.15 * Y ** 2 + 0.1 * X * Y).ravel()
b_t = (0.8 + 0.1 * X - 0.08 * Y + 0.05 * X ** 2).ravel()
Phi_t = (2 * np.pi * (1.6 * X + 0.9 * Y) + 1.2 * X ** 2 - 0.7 * X * Y).ravel()
delta_t = 2 * np.pi * np.arange(N) / N + rng.uniform(-0.3, 0.3, N); delta_t -= delta_t[0]
g_t = 1 + 0.08 * rng.standard_normal(N); g_t /= np.median(g_t)
D_shape = rng.standard_normal((N, J)); D_shape -= D_shape.mean(0)
D_max = np.abs(D_shape @ Hm).max()

u_t, v_t = b_t * np.cos(Phi_t), -b_t * np.sin(Phi_t)
P_t, Q_t = g_t * np.cos(delta_t), g_t * np.sin(delta_t)
w_t = P_t[:, None] * v_t - Q_t[:, None] * u_t


def gauge(f, gam):
    """Apply shift (p,q) and symmetric det-1 S to (a,u,v,P,Q); piston unchanged."""
    a, u, v, P, Q = f
    p, q, s1, s2 = gam
    h = np.sqrt(1 + s1 ** 2 + s2 ** 2)
    S = np.array([[h + s1, s2], [s2, h - s1]])
    Si = np.array([[h - s1, -s2], [-s2, h + s1]])
    P2, Q2 = S @ np.vstack([P, Q])
    u2, v2 = Si @ np.vstack([u, v])
    P2, Q2 = P2 + p, Q2 + q
    return (a - p * u2 - q * v2, u2, v2, P2, Q2)


def wfun(f):
    _, u, v, P, Q = f
    return P[:, None] * v - Q[:, None] * u


def proj(P, Q):
    A = np.stack([np.ones(N), P, Q], 1)
    Ap = A.T @ A
    return A, Ap, np.eye(N) - A @ np.linalg.solve(Ap, A.T)


def fit(f, target, extra_cols=()):
    """Fit target (N,K) = Pi z with z columns: P_m u, Q_m v, w_m H_j, extra (N,K) cols.
    Returns beta pieces, model z (N,K), Pi residual vector."""
    a, u, v, P, Q = f
    A, Ap, Pi = proj(P, Q)
    w = wfun(f)
    cols = []
    for m in range(N):
        z = np.zeros((N, K)); z[m] = u; cols.append(z)
    for m in range(N):
        z = np.zeros((N, K)); z[m] = v; cols.append(z)
    for m in range(N):
        for j in range(J):
            z = np.zeros((N, K)); z[m] = w[m] * Hm[j]; cols.append(z)
    cols += list(extra_cols)
    Z = np.stack(cols)                                  # (nb, N, K)
    Xd = np.einsum("nm,cmk->cnk", Pi, Z).reshape(len(cols), -1).T
    nb = len(cols)
    C = np.zeros((6 + J, nb))
    C[0:3, 0:N] = A.T; C[3:6, N:2 * N] = A.T
    for j in range(J):
        C[6 + j, 2 * N + j:2 * N + N * J:J] = 1
    t = (Pi @ target).ravel()
    beta = np.linalg.lstsq(np.vstack([Xd, C]), np.concatenate([t, np.zeros(6 + J)]),
                           rcond=None)[0]
    zmod = np.einsum("c,cnk->nk", beta, Z)
    return beta, zmod, Xd @ beta - t


def pixel_corr(f, rhs):
    _, u, v, P, Q = f
    A, Ap, _ = proj(P, Q)
    return np.linalg.solve(Ap, A.T @ rhs)              # (3, K)


def first_order(f, I):
    a, u, v, P, Q = f
    r = I - (a + P[:, None] * u + Q[:, None] * v)
    beta, z, res = fit(f, r)
    P1, Q1 = beta[:N], beta[N:2 * N]
    al = beta[2 * N:2 * N + N * J].reshape(N, J)
    a1, u1, v1 = pixel_corr(f, r - z)
    r1 = z + a1 + P[:, None] * u1 + Q[:, None] * v1
    return dict(P1=P1, Q1=Q1, al=al, a1=a1, u1=u1, v1=v1, r=r, r1=r1, res=res)


def fit_gamma(f0, I, iters=30):
    """Variable projection: Gauss-Newton on the reduced residual over gamma."""
    gam = np.zeros(4)
    for _ in range(iters):
        r0 = first_order(gauge(f0, gam), I)["res"]
        Jac = np.empty((r0.size, 4))
        for k in range(4):
            e = np.zeros(4); e[k] = 1e-6
            Jac[:, k] = (first_order(gauge(f0, gam + e), I)["res"] - r0) / 1e-6
        step = np.linalg.lstsq(Jac, -r0, rcond=None)[0]
        gam += step
        if np.abs(step).max() < 1e-12:
            break
    return gam


def second_order(f, I, s1, gam=None, f0=None):
    a, u, v, P, Q = f
    D1 = s1["al"] @ Hm
    k = (s1["P1"][:, None] * s1["u1"] + s1["Q1"][:, None] * s1["v1"]
         + (P[:, None] * s1["v1"] - Q[:, None] * s1["u1"]
            + v * s1["P1"][:, None] - u * s1["Q1"][:, None]) * D1
         - 0.5 * (P[:, None] * u + Q[:, None] * v) * D1 ** 2)
    r2 = s1["r"] - s1["r1"]
    extra = []
    if gam is not None:                                 # gamma^(1) columns
        for kk in range(4):
            e = np.zeros(4); e[kk] = 1e-6
            dw = (wfun(gauge(f0, gam + e)) - wfun(gauge(f0, gam - e))) / 2e-6
            extra.append(dw * D1)
    beta, z, _ = fit(f, r2 - k, extra)
    al2 = beta[2 * N:2 * N + N * J].reshape(N, J)
    return s1["al"] + al2, beta[2 * N + N * J:]


def run(Dmax):
    al_t = D_shape * (Dmax / D_max)
    Dt = al_t @ Hm
    I = a_t + (P_t[:, None] * u_t + Q_t[:, None] * v_t) * np.cos(Dt) + w_t * np.sin(Dt)
    _, _, _, dl, g, _ = aia(I.reshape(N, H, W), np.ones(N), fit_gain=True,
                            iters=2000, tol=1e-14, precision="double")
    a0, u0, v0 = pixel_step(I, dl, g, precision="double")
    f0 = normalize_quadrature_frame(a0, u0, v0, g * np.cos(dl), g * np.sin(dl), np,
                                    precision="double")
    err = lambda al: np.linalg.norm(al - al_t) / np.linalg.norm(al_t)
    werr = lambda f: np.linalg.norm(wfun(f) - w_t) / np.linalg.norm(w_t)
    out = {}
    # conventions
    s1 = first_order(f0, I)
    out["conv1"] = err(s1["al"])
    out["conv2"] = err(second_order(f0, I, s1)[0])
    # fitted gauge
    gam = fit_gamma(f0, I)
    fg = gauge(f0, gam)
    s1g = first_order(fg, I)
    out["gauge1"] = err(s1g["al"])
    al2, g1 = second_order(fg, I, s1g, gam, f0)
    out["gauge2"] = err(al2)
    out["gauge2_no_g1"] = err(second_order(fg, I, s1g)[0])
    out["w_conv"] = werr(f0)
    out["w_gauge"] = werr(fg)
    out["w_gauge+g1"] = werr(gauge(f0, gam + g1))
    out["gamma"] = gam
    return out


if __name__ == "__main__":
    keys = ["conv1", "conv2", "gauge1", "gauge2", "gauge2_no_g1", "w_conv", "w_gauge",
            "w_gauge+g1"]
    print("Dmax   " + " ".join(f"{k:>12s}" for k in keys))
    for Dmax in [0.2, 0.1, 0.05, 0.025, 0.0125]:
        o = run(Dmax)
        print(f"{Dmax:<6} " + " ".join(f"{o[k]:12.2e}" for k in keys),
              " gamma=", np.array2string(o["gamma"], precision=4))
