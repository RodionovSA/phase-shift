"""Monte Carlo: phase and alpha error with noise, fitted gauge vs conventions, k_max orders."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gauge_fit_test as g
from order_k_test import model
from phase_shift.methods.aia import aia
from phase_shift.methods.steps import pixel_step
from phase_shift.methods.gauge import normalize_quadrature_frame
N, J, K, Hm = g.N, g.J, g.K, g.Hm
Phi_t = np.arctan2(-g.v_t, g.u_t)

def solve(I, fit_gauge, kmax=3):
    _, _, _, dl, gg, _ = aia(I.reshape(N, g.H, g.W), np.ones(N), fit_gain=True, iters=2000, tol=1e-12, precision="double")
    a0, u0, v0 = pixel_step(I, dl, gg, precision="double")
    f0 = normalize_quadrature_frame(a0, u0, v0, gg * np.cos(dl), gg * np.sin(dl), np, precision="double")
    gam = g.fit_gamma(f0, I, iters=15) if fit_gauge else np.zeros(4)
    fb = g.gauge(f0, gam); s1 = g.first_order(fb, I); D1 = s1["al"] @ Hm
    gcols = []
    if fit_gauge:
        for kk in range(4):
            e = np.zeros(4); e[kk] = 1e-6
            gcols.append((g.wfun(g.gauge(fb, e)) - g.wfun(g.gauge(fb, -e))) / 2e-6 * D1)
    F = tuple(x + d for x, d in zip(fb, (s1["a1"], s1["u1"], s1["v1"], s1["P1"], s1["Q1"])))
    al = s1["al"].copy()
    for k in range(2, kmax + 1):
        R = I - model(F, al)
        beta, z, _ = g.fit(fb, R, gcols)
        F = tuple(x + d for x, d in zip(F, (*g.pixel_corr(fb, R - z), beta[:N], beta[N:2 * N])))
        if fit_gauge: F = g.gauge(F, beta[2 * N + N * J:])
        al = al + beta[2 * N:2 * N + N * J].reshape(N, J)
    a, u, v, P, Q = F
    d1 = np.arctan2(Q[0], P[0]); c, s_ = np.cos(d1), np.sin(d1)
    u, v = c * u + s_ * v, -s_ * u + c * v                     # rotate so delta_1 = 0
    phi = np.arctan2(-v, u)
    return phi, al

rng = np.random.default_rng(0)
sigma = 0.02                                                   # b ~ 0.8, a ~ 1
print(f"sigma={sigma}, K={K}, N={N}, J={J}, k_max=3, 20 trials")
print("Dmax    method   phase bias  phase noise   alpha rel bias  alpha rel noise")
for Dmax in [0.2, 0.05, 0.0125]:
    al_t = g.D_shape * (Dmax / g.D_max); Dt = al_t @ Hm
    I0 = g.a_t + (g.P_t[:, None] * g.u_t + g.Q_t[:, None] * g.v_t) * np.cos(Dt) + g.w_t * np.sin(Dt)
    for fit_gauge in (False, True):
        ph, als = [], []
        for t in range(20):
            p_, a_ = solve(I0 + sigma * rng.standard_normal(I0.shape), fit_gauge)
            ph.append(np.angle(np.exp(1j * (p_ - Phi_t)))); als.append(a_)
        ph, als = np.array(ph), np.array(als)
        pbias = np.sqrt(np.mean(ph.mean(0) ** 2)); pnoise = np.sqrt(np.mean(ph.var(0)))
        abias = np.linalg.norm(als.mean(0) - al_t) / np.linalg.norm(al_t)
        anoise = np.sqrt(np.mean(np.sum((als - als.mean(0)) ** 2, axis=(1, 2)))) / np.linalg.norm(al_t)
        print(f"{Dmax:<7} {'fitted' if fit_gauge else 'conv':8s} {pbias:10.2e}  {pnoise:10.2e}   {abias:12.2e}   {anoise:12.2e}")
