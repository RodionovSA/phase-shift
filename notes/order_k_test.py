"""Test: order-k VP-AIA as a fixed-operator fit of the full-model residual.

Order 1: bilinear fit with the gauge gamma (notes/gauge_fit_test.py).
Order k >= 2: fit R = I - model(accumulated estimates) with the operator of
order 1 (Pi, w at the moved baseline) plus 4 gauge columns (dw/dgamma) Delta^(1);
update the estimates and apply the gauge increment to them.
Also compares k = 2 with the explicit second-order form (known terms k_n written out).
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gauge_fit_test as g
from phase_shift.methods.aia import aia
from phase_shift.methods.steps import pixel_step
from phase_shift.methods.gauge import normalize_quadrature_frame

N, J, K, Hm = g.N, g.J, g.K, g.Hm


def model(f, al):
    a, u, v, P, Q = f
    D = al @ Hm
    c = P[:, None] * u + Q[:, None] * v
    return a + c * np.cos(D) + g.wfun(f) * np.sin(D)


def run(Dmax, orders=5):
    al_t = g.D_shape * (Dmax / g.D_max)
    Dt = al_t @ Hm
    I = g.a_t + (g.P_t[:, None] * g.u_t + g.Q_t[:, None] * g.v_t) * np.cos(Dt) + g.w_t * np.sin(Dt)
    _, _, _, dl, gg, _ = aia(I.reshape(N, g.H, g.W), np.ones(N), fit_gain=True,
                             iters=2000, tol=1e-14, precision="double")
    a0, u0, v0 = pixel_step(I, dl, gg, precision="double")
    f0 = normalize_quadrature_frame(a0, u0, v0, gg * np.cos(dl), gg * np.sin(dl), np,
                                    precision="double")
    err = lambda al: np.linalg.norm(al - al_t) / np.linalg.norm(al_t)

    # order 1, bilinear in gamma
    gam = g.fit_gamma(f0, I)
    fb = g.gauge(f0, gam)                                   # moved baseline
    s1 = g.first_order(fb, I)
    errs = [err(s1["al"])]
    D1 = s1["al"] @ Hm
    # explicit k_n form at k = 2, for comparison
    al2_explicit = g.second_order(fb, I, s1, gam, f0)[0]

    # gauge columns at the moved baseline, relative to its own frame
    gcols = []
    for kk in range(4):
        e = np.zeros(4); e[kk] = 1e-6
        dw = (g.wfun(g.gauge(fb, e)) - g.wfun(g.gauge(fb, -e))) / 2e-6
        gcols.append(dw * D1)

    F = tuple(x + d for x, d in zip(fb, (s1["a1"], s1["u1"], s1["v1"], s1["P1"], s1["Q1"])))
    al = s1["al"].copy()
    al_k2 = None
    for k in range(2, orders + 1):
        R = I - model(F, al)
        beta, z, _ = g.fit(fb, R, gcols)
        Pk, Qk = beta[:N], beta[N:2 * N]
        alk = beta[2 * N:2 * N + N * J].reshape(N, J)
        dgam = beta[2 * N + N * J:]
        ak, uk, vk = g.pixel_corr(fb, R - z)
        F = tuple(x + d for x, d in zip(F, (ak, uk, vk, Pk, Qk)))
        F = g.gauge(F, dgam)
        al = al + alk
        errs.append(err(al))
        if k == 2:
            al_k2 = al.copy()
    diff2 = np.linalg.norm(al_k2 - al2_explicit) / np.linalg.norm(al_t)
    return errs, err(al2_explicit), diff2


if __name__ == "__main__":
    orders = 5
    print("Dmax    " + " ".join(f"{'k=' + str(k):>9s}" for k in range(1, orders + 1))
          + "   explicit k=2   |chord-explicit| k=2")
    for Dmax in [0.4, 0.2, 0.1, 0.05, 0.025]:
        errs, e2x, d2 = run(Dmax, orders)
        print(f"{Dmax:<7} " + " ".join(f"{e:9.2e}" for e in errs) + f"   {e2x:12.2e}   {d2:12.2e}")
