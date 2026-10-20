"""Check: with the gamma columns, Appendix E's normal system needs only the
sums  S_xy[j,j'] = sum u_x u_y H_j H_j'  (H_0 = 1)  and  T_x[n,j] = sum x H_j (Pi r)_n."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gauge_fit_test as g
rng = np.random.default_rng(5)
N, J, K, Hm = g.N, g.J, g.K, g.Hm
a, u, v = rng.standard_normal((3, K)); P, Q = rng.standard_normal((2, N))
f = (a, u, v, P, Q); _, _, Pi = g.proj(P, Q)
al1 = rng.standard_normal((N, J)); r = rng.standard_normal((N, K))
G = np.vstack([np.ones(K), Hm])                                   # (J+1, K)
# columns as (lambda_n, mu_n, g_n[j]) : z_n = (lambda_n u + mu_n v) sum_j g_n[j] G_j
cols = []
for m in range(N):
    e = np.eye(N)[m]; cols.append((0 * e, 0 * e, np.outer(e, np.eye(J + 1)[0])))   # placeholder
cols = []
E0 = np.eye(J + 1)
for m in range(N):                                                # P_m: u at frame m
    e = np.eye(N)[m]; cols.append((e, 0 * e, np.outer(e, E0[0])))
for m in range(N):                                                # Q_m: v at frame m
    e = np.eye(N)[m]; cols.append((0 * e, e, np.outer(e, E0[0])))
for m in range(N):                                                # alpha_mj: w_m H_j
    for j in range(J):
        e = np.eye(N)[m]; cols.append((-Q * e, P * e, np.outer(e, E0[j + 1])))
ga = np.hstack([np.zeros((N, 1)), al1])                           # Delta^(1) in G basis
dw = [(0 * P, 1 + 0 * P), (-1 + 0 * P, 0 * P), (2 * Q, 2 * P), (-2 * P, 2 * Q)]   # (c_u, c_v) of dw/dgamma
for cu, cv in dw:
    cols.append((cu, cv, ga))
# dense
Z = np.stack([(lam[:, None] * u + mu[:, None] * v) * (gg @ G) for lam, mu, gg in cols])
Md = np.einsum("anx,nm,bmx->ab", Z, Pi, Z); rd = np.einsum("anx,nx->a", Z, Pi @ r)
# reduced
S = {k: np.einsum("x,jx,kx->jk", x, G, G) for k, x in {"uu": u * u, "uv": u * v, "vv": v * v}.items()}
Tu, Tv = np.einsum("x,jx,nx->nj", u, G, Pi @ r), np.einsum("x,jx,nx->nj", v, G, Pi @ r)
def blk(c1, c2):
    l1, m1, g1 = c1; l2, m2, g2 = c2
    out = 0.0
    for n in range(N):
        for m in range(N):
            if Pi[n, m] == 0: continue
            q = l1[n] * l2[m] * S["uu"] + (l1[n] * m2[m] + m1[n] * l2[m]) * S["uv"] + m1[n] * m2[m] * S["vv"]
            out += Pi[n, m] * g1[n] @ q @ g2[m]
    return out
Mr = np.array([[blk(c1, c2) for c2 in cols] for c1 in cols])
rr = np.array([sum(g1[n] @ (l1[n] * Tu[n] + m1[n] * Tv[n]) for n in range(N)) for l1, m1, g1 in cols])
# dw/dgamma coefficients vs finite difference of the move
fd = [(g.wfun(g.gauge(f, e)) - g.wfun(g.gauge(f, -e))) / 2e-6 for e in 1e-6 * np.eye(4)]
an = [cu[:, None] * u + cv[:, None] * v for cu, cv in dw]
print("M rel err", np.abs(Md - Mr).max() / np.abs(Md).max(), " rho rel err", np.abs(rd - rr).max() / np.abs(rd).max())
print("dw/dgamma rel err", max(np.abs(x - y).max() / np.abs(y).max() for x, y in zip(an, fd)))
