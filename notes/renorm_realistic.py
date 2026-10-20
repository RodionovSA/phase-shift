"""End-only vs every-order normalization, realistic synthetic case with noise.
Elliptical fringes, Gaussian beam, smooth background, random steps, gain spread."""
import sys
import numpy as np
from phase_shift.basis import spatial_basis
from phase_shift.methods.aia import aia
from phase_shift.methods.steps import pixel_step
from phase_shift.methods.gauge import normalize_quadrature_frame as nqf
from phase_shift.methods.vp_system import build_system, order_rhs, solve_order
from phase_shift.methods.vp_aia import _pixel_corrections

H, W, N, KMAX, TRIALS = 200, 240, 8, 4, 10
rng = np.random.default_rng(0)
Y, X = np.mgrid[0:H, 0:W].astype(float); X = (X - W / 2) / (W / 2); Y = (Y - H / 2) / (H / 2)
r2 = (X - 0.1) ** 2 + ((Y + 0.05) * 1.3) ** 2
Phi = 2 * np.pi * 8 * r2 / r2.max() + 2.0 * X + 0.3
b = 0.8 * np.exp(-((X - 0.05) ** 2 + (Y + 0.03) ** 2) / 1.2)
a = 1.0 + 0.1 * X - 0.05 * Y ** 2 + 0.2 * np.exp(-(X ** 2 + Y ** 2) / 0.8)
delta = 2 * np.pi * np.arange(N) / N + rng.uniform(-0.4, 0.4, N); delta -= delta[0]
g = 1 + 0.05 * rng.standard_normal(N); g /= np.median(g)
basis = spatial_basis(H, W, "poly", np, degree=2); J, K = basis.shape
shape = rng.standard_normal((N, J)); shape -= shape.mean(0)
shape /= np.abs(shape @ basis).max()
Phi_f = Phi.ravel()

def solve(I, renorm):
    _, _, _, dl, gg, _ = aia(I.reshape(N, H, W), np.ones(N), fit_gain=True, iters=200, tol=1e-10, precision="double")
    a0, u0, v0 = pixel_step(I, dl, gg, precision="double")
    base = nqf(a0, u0, v0, gg * np.cos(dl), gg * np.sin(dl), np, precision="double")
    sy = build_system(*base[1:], basis, precision="double")
    F, al, out = base, np.zeros((N, J)), []
    for k in range(KMAX):
        rho, Atr = order_rhs(I, *F, al, sy, basis, precision="double")
        s = solve_order(sy, rho)
        ak, uk, vk = _pixel_corrections(s.alpha, Atr, *base[1:], basis, np.float64, np)
        F = (F[0] + ak, F[1] + uk, F[2] + vk, F[3] + s.P_corr, F[4] + s.Q_corr); al = al + s.alpha
        if renorm: F = nqf(*F, np, precision="double")
        Fn = nqf(*F, np, precision="double")
        phi = np.arctan2(-Fn[2], Fn[1])
        d = np.angle(np.exp(1j * (phi - Phi_f))); d -= np.angle(np.mean(np.exp(1j * d)))
        stp = np.angle(np.exp(1j * (np.arctan2(Fn[4], Fn[3]) - delta)))
        out.append((al.copy(), d, stp))
    return out

sig_list = [0.0, 0.01, 0.03]
print(f"{H}x{W}, K={K}, N={N}, J={J}, 8 elliptical rings + tilt, Gaussian beam; errors vs physical truth")
print("Dmax  sigma  norm    order:  alpha bias   alpha noise | phase bias  phase noise | step bias")
for Dmax in (0.2, 0.05):
    al_t = shape * Dmax; Dt = al_t @ basis
    I0 = (a.ravel() + g[:, None] * b.ravel() * np.cos(Phi_f + delta[:, None] + Dt))
    for sigma in sig_list:
        nt = 1 if sigma == 0 else TRIALS
        noises = [sigma * np.random.default_rng(100 + t).standard_normal(I0.shape) for t in range(nt)]
        for renorm in (False, True):
            runs = [solve(I0 + e, renorm) for e in noises]
            for k in range(KMAX):
                als = np.array([r[k][0] for r in runs]); ds = np.array([r[k][1] for r in runs]); st = np.array([r[k][2] for r in runs])
                ab = np.linalg.norm(als.mean(0) - al_t) / np.linalg.norm(al_t)
                an = np.sqrt(np.mean(np.sum((als - als.mean(0)) ** 2, (1, 2)))) / np.linalg.norm(al_t)
                pb = np.sqrt(np.mean(ds.mean(0) ** 2)); pn = np.sqrt(np.mean(ds.var(0)))
                sb = np.abs(st.mean(0)).max()
                print(f"{Dmax:<5} {sigma:<6} {'every' if renorm else 'end':6s} {k+1}:      {ab:9.2e}   {an:9.2e}  | {pb:9.2e}  {pn:9.2e}  | {sb:9.2e}", flush=True)
