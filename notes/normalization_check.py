"""Final normalization: rotation + scale only vs the full Appendix B normalization,
order by order, noiseless. Errors against the true parameters moved to the conventions."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gauge_fit_test as g
from order_k_test import model
from phase_shift.methods.aia import aia
from phase_shift.methods.steps import pixel_step
from phase_shift.methods.gauge import normalize_quadrature_frame as nqf
N, J, Hm = g.N, g.J, g.Hm

def rot_scale(f):
    a, u, v, P, Q = f
    d1 = np.arctan2(Q[0], P[0]); c, s = np.cos(d1), np.sin(d1)
    P, Q = c * P + s * Q, -s * P + c * Q; u, v = c * u + s * v, -s * u + c * v
    m = np.median(np.hypot(P, Q))
    return a, u * m, v * m, P / m, Q / m

def errs(f, ref):
    a, u, v, P, Q = f; ar, ur, vr, Pr, Qr = ref
    dphi = np.angle(np.exp(1j * (np.arctan2(-v, u) - np.arctan2(-vr, ur))))
    dd = np.angle(np.exp(1j * (np.arctan2(Q, P) - np.arctan2(Qr, Pr))))
    return np.sqrt(np.mean(dphi ** 2)), np.abs(dd).max(), np.abs(np.hypot(P, Q) - np.hypot(Pr, Qr)).max()

def run(truth, Dmax, kmax=4):
    a_t, u_t, v_t, P_t, Q_t = truth
    al_t = g.D_shape * (Dmax / g.D_max); Dt = al_t @ Hm
    I = model(truth, al_t)
    ref = nqf(*truth, np, precision="double")            # conventional true parameters
    _, _, _, dl, gg, _ = aia(I.reshape(N, g.H, g.W), np.ones(N), fit_gain=True, iters=3000, tol=1e-14, precision="double")
    a0, u0, v0 = pixel_step(I, dl, gg, precision="double")
    fb = nqf(a0, u0, v0, gg * np.cos(dl), gg * np.sin(dl), np, precision="double")
    F, al = fb, np.zeros((N, J)); out = []
    for k in range(1, kmax + 1):
        R = I - model(F, al)
        beta, z, _ = g.fit(fb, R)
        ak, uk, vk = g.pixel_corr(fb, R - z)
        F = (F[0] + ak, F[1] + uk, F[2] + vk, F[3] + beta[:N], F[4] + beta[N:2 * N])
        al = al + beta[2 * N:2 * N + N * J].reshape(N, J)
        out.append((errs(rot_scale(F), ref), errs(nqf(*F, np, precision="double"), ref)))
    return out

physical = (g.a_t, g.u_t, g.v_t, g.P_t, g.Q_t)
conventional = nqf(*physical, np, precision="double")
for name, truth in [("truth satisfies conventions", conventional), ("physical truth", physical)]:
    print(f"\n{name}:  errors vs true parameters moved to the conventions")
    print("Dmax   k   rot+scale: phase RMS  step max  gain max  |  full: phase RMS  step max  gain max")
    for Dmax in (0.2, 0.1, 0.05):
        for k, (e1, e2) in enumerate(run(truth, Dmax), 1):
            print(f"{Dmax:<5} {k}   {e1[0]:10.2e} {e1[1]:9.2e} {e1[2]:9.2e}  |  {e2[0]:10.2e} {e2[1]:9.2e} {e2[2]:9.2e}")
