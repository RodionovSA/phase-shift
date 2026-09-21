# notes/validate_exact_stages.py
"""CPU Monte Carlo for PLAN.md Steps 2 and 5; writes measurements as JSON.

Checks the exact per-pixel phase variance of ``docs/aia.md`` Eqs. (40) and
(45), and the step-field term of ``docs/sf_aia.md`` Eq. (E7), against the
estimators they describe: one pass taken from the true fields, with the frame
step read at the noise-free quadratures. Variances are measured about each
estimator's own mean through antithetic noise pairs, never as a mean squared
error about the truth.

``--part step_bias`` measures instead the bias those variances sit beside: the
displacement the full alternation puts into ``delta_n`` because its frame step
regresses on quadratures its own pixel step estimated from the same noise. The
same antithetic pairs isolate it, the bias being the even part of the noise and
the variance the odd one.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phase_shift.basis import spatial_basis
from phase_shift.errors import aia_phi_error_parts, step_field_phi_error
from phase_shift.methods import aia


def make_case(seed: int = 2, N: int = 7, H: int = 16, W: int = 16, degree: int = 1,
              ideal: bool = False, orthogonal_background: bool = True) -> dict:
    """Synthetic fields and steps, with the exact quantities the formulas need.

    ``orthogonal_background`` removes the background's overlap with
    ``{1, u, v}``; that overlap is the frame-independent displacement of
    ``docs/aia.md`` §"Dropping the background field", which biases a fitted
    ``delta_n`` and is outside the variances measured here.
    """
    rng = np.random.default_rng(seed)
    Np = H * W
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    if ideal:
        phi = (2 * np.pi * np.arange(Np) / Np).reshape(H, W)
        b = np.ones((H, W))
        a = np.ones((H, W))
        delta = 2 * np.pi * np.arange(N) / N
        g = np.ones(N)
        sigma = np.full((H, W), 0.05)
    else:
        phi = np.angle(np.exp(1j * (0.9 * xx + 0.5 * yy + 0.7 * np.sin(xx / 3.0))))
        b = 0.6 + 0.8 * rng.random((H, W))
        a = 1.5 + 0.5 * rng.random((H, W))
        delta = np.sort(rng.uniform(0, 2 * np.pi, N))
        delta -= delta[0]
        g = 0.7 + 0.6 * rng.random(N)
        sigma = 0.03 + 0.04 * rng.random((H, W))
    u = b * np.cos(phi)
    v = -b * np.sin(phi)
    if orthogonal_background:
        B = np.stack([np.ones(Np), u.ravel(), v.ravel()], axis=1)
        a = (a.ravel() - B @ np.linalg.lstsq(B, a.ravel(), rcond=None)[0]).reshape(H, W)
    P = g * np.cos(delta)
    Q = g * np.sin(delta)
    I = a[None] + P[:, None, None] * u[None] + Q[:, None, None] * v[None]
    return dict(phi=phi, b=b, delta=delta, g=g, sigma=sigma, u=u, v=v, I=I, P=P, Q=Q,
                N=N, Np=Np, shape=(H, W), basis=spatial_basis(H, W, "poly", np, degree=degree))


def _pixel_step(frames: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(A, frames, rcond=None)[0]


def solve_aia(case: dict, eps: np.ndarray, stage: int) -> np.ndarray:
    """One conditional pass: frame step from the true ``(u, v)``, then a pixel step."""
    N = case["N"]
    meas = (case["I"] + eps).reshape(N, -1)
    delta, g = case["delta"], case["g"]
    if stage >= 2:
        B = np.stack([np.ones(case["Np"]), case["u"].ravel(), case["v"].ravel()], axis=1)
        coef = _pixel_step(meas.T, B)                                    # (3, N)
        delta = np.arctan2(coef[2], coef[1])
        g = np.hypot(coef[1], coef[2]) if stage >= 3 else g
    A = np.stack([np.ones(N), g * np.cos(delta), g * np.sin(delta)], axis=1)
    X = _pixel_step(meas, A)
    return np.arctan2(-X[2], X[1])


def solve_step_field(case: dict, eps: np.ndarray, rounds: int) -> np.ndarray:
    """``docs/sf_aia.md`` §"Algorithm" with the steps and gains held at the truth."""
    N = case["N"]
    A = np.stack([np.ones(N), case["P"], case["Q"]], axis=1)
    meas = (case["I"] + eps).reshape(N, -1)
    p = case["basis"]
    X = _pixel_step(meas, A)
    for _ in range(rounds):
        resid = meas - A @ X
        w_field = case["Q"][:, None] * X[1][None] - case["P"][:, None] * X[2][None]
        D = w_field[:, None, :] * p[None]                                # (N, J, P)
        G = np.einsum('njp,nkp->njk', D, D)
        rhs = np.einsum('njp,np->nj', D, resid)
        c = -np.squeeze(np.linalg.solve(G, rhs[:, :, None]), -1)         # (N, J), Eq. (E1)
        c = c - c.mean(axis=0, keepdims=True)                            # Eq. (E4)
        X = _pixel_step(meas + w_field * (c @ p), A)
    return np.arctan2(-X[2], X[1])


def measure(case: dict, solver, trials: int, seed: int) -> np.ndarray:
    """Per-pixel variance about the estimator's own mean, from antithetic pairs."""
    rng = np.random.default_rng(seed)
    ref = case["phi"].ravel()
    plus, minus = [], []
    for _ in range(trials):
        z = rng.standard_normal((case["N"],) + case["shape"]) * case["sigma"][None]
        plus.append(np.angle(np.exp(1j * (solver(z) - ref))))
        minus.append(np.angle(np.exp(1j * (solver(-z) - ref))))
    plus, minus = np.array(plus), np.array(minus)
    even = (plus + minus) / 2
    odd = (plus - minus) / 2
    return (odd ** 2).mean(0) + even.var(0, ddof=1)


def run_aia(trials: int, seed: int) -> list[dict]:
    """Eqs. (40) and (45) against the estimators they describe."""
    rows = []
    for N, H, W in [(5, 16, 16), (7, 16, 16), (9, 24, 24)]:
        case = make_case(N=N, H=H, W=W)
        args = (case["b"], case["phi"], case["delta"], case["g"])
        base, _ = aia_phi_error_parts(*args, False, case["sigma"], True, np)
        got_1 = measure(case, lambda z: solve_aia(case, z, 1), trials, seed)
        for stage, fit_gain in ((2, False), (3, True)):
            pred, _ = aia_phi_error_parts(*args, fit_gain, case["sigma"], False, np)
            got = measure(case, lambda z, s=stage: solve_aia(case, z, s), trials, seed)
            rows.append(dict(N=N, Np=case["Np"], stage=stage,
                             baseline_ratio=float(got_1.mean() / base.mean()),
                             predicted_delta=float((pred - base).mean()),
                             measured_delta=float((got - got_1).mean()),
                             ratio=float((got - got_1).mean() / (pred - base).mean())))
    return rows


def run_step_field(trials: int, seed: int) -> list[dict]:
    """Eq. (E7) at one pass, then the refinement loop's drift toward Eq. (30)."""
    rows = []
    for degree in (1, 2):
        case = make_case(seed=3, N=9, H=16, W=16, degree=degree)
        J, Np = case["basis"].shape[0], case["Np"]
        args = (case["b"], case["phi"], case["delta"], case["g"])
        base, _ = aia_phi_error_parts(*args, False, case["sigma"], True, np)
        pred = step_field_phi_error(*args, False, case["sigma"], False,
                                    case["basis"], np) ** 2
        aia_part, _ = aia_phi_error_parts(*args, False, case["sigma"], False, np)
        got_0 = measure(case, lambda z: solve_step_field(case, z, 0), trials, seed)
        for rounds in (1, 2, 3, 4, 6, 8, 12, 20):
            got = measure(case, lambda z, r=rounds: solve_step_field(case, z, r),
                          trials, seed)
            rows.append(dict(J=J, Np=Np, rounds=rounds,
                             measured_relative_times_Np=float(((got - got_0) / got_0).mean() * Np),
                             eq_E7_relative_times_Np=float(((pred - aia_part) / base).mean() * Np),
                             eq_E8=J / 4, vp_eq_30=J))
    return rows


def resampled_case(side: int, seed: int = 0, N: int = 7, sigma: float = 0.05) -> dict:
    """The same fringe content at a different sampling density.

    Only ``N_p`` changes: the fringe count, the shapes of ``a`` and ``b``, the
    steps, the gains and ``sigma`` are all held fixed, so a quantity that
    varies across these cases varies with the field size alone.
    """
    rng = np.random.default_rng(seed)
    delta = np.sort(rng.uniform(0, 2 * np.pi, N))
    delta -= delta[0]
    g = 0.7 + 0.6 * rng.random(N)
    x, y = np.meshgrid(np.arange(side) / side, np.arange(side) / side, indexing="xy")
    phi = np.angle(np.exp(1j * (2 * np.pi * (3 * x + 2 * y) + 0.7 * np.sin(6 * np.pi * x))))
    b = 0.8 + 0.4 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    a = 1.5 + 0.5 * np.cos(4 * np.pi * y)
    u, v = b * np.cos(phi), -b * np.sin(phi)
    B = np.stack([np.ones(u.size), u.ravel(), v.ravel()], axis=1)
    a = (a.ravel() - B @ np.linalg.lstsq(B, a.ravel(), rcond=None)[0]).reshape(side, side)
    P, Q = g * np.cos(delta), g * np.sin(delta)
    I = a[None] + P[:, None, None] * u[None] + Q[:, None, None] * v[None]
    return dict(phi=phi, b=b, delta=delta, g=g, u=u, v=v, I=I, P=P, Q=Q, N=N,
                Np=side * side, shape=(side, side), sigma=np.full((side, side), sigma))


def _step_bias(case: dict, pairs: int, seed: int, noise_scale: float,
               alternate: bool) -> tuple[float, float]:
    """Bias of ``delta_n``, from the even part of antithetic noise pairs.

    The bias is even in the noise and the variance is odd, so averaging
    ``(delta(+z) + delta(-z))/2`` removes the variance term exactly instead of
    averaging it down. ``alternate`` runs the full AIA loop; otherwise a single
    frame step is taken from the true ``(u, v)``, which is the estimator
    ``aia.md`` Eqs. (35)-(36) describe.
    """
    N = case["N"]
    B = np.stack([np.ones(case["Np"]), case["u"].ravel(), case["v"].ravel()], axis=1)
    rng = np.random.default_rng(seed)
    evens = []
    for _ in range(pairs):
        z = rng.standard_normal((N,) + case["shape"]) * case["sigma"][None] * noise_scale
        signed = []
        for sign in (+1, -1):
            meas = (case["I"] + sign * z).astype(np.float64)
            if alternate:
                fitted = aia(meas, case["g"].copy(), fit_gain=False, dtype=np.float64,
                             iters=400, tol=1e-12)[3]
            else:
                coef = np.linalg.lstsq(B, meas.reshape(N, -1).T, rcond=None)[0]
                fitted = np.arctan2(coef[2], coef[1])
            err = np.angle(np.exp(1j * (fitted - case["delta"])))
            signed.append(err - err[0])                          # delta_1 = 0 gauge
        evens.append((signed[0] + signed[1]) / 2)
    evens = np.array(evens)                                      # (pairs, N)
    mean = evens.mean(0)
    se = evens.std(0, ddof=1) / np.sqrt(pairs)
    return float(np.sqrt(np.mean(mean ** 2))), float(np.sqrt(np.mean(se ** 2)))


def run_step_bias(pairs: int, seed: int) -> list[dict]:
    """The alternation's bias in ``delta_n``, against ``N_p`` and against sigma."""
    rows = []
    for side, scale, alternate in [(16, 1.0, True), (32, 1.0, True), (64, 1.0, True),
                                   (128, 1.0, True), (256, 1.0, True),
                                   (64, 0.5, True), (64, 0.25, True),
                                   (16, 1.0, False), (64, 1.0, False), (256, 1.0, False)]:
        case = resampled_case(side)
        count = max(60, pairs // max(1, (side // 16) ** 2))
        bias, se = _step_bias(case, count, seed, scale, alternate)
        noise_free = aia(case["I"].astype(np.float64), case["g"].copy(), fit_gain=False,
                         dtype=np.float64, iters=3000, tol=1e-14)[3]
        exact = np.angle(np.exp(1j * (noise_free - case["delta"])))
        rows.append(dict(Np=case["Np"], noise_scale=scale, alternating=alternate,
                         pairs=count, bias_rms=bias, bias_se=se,
                         bias_over_sigma_sq=bias / scale ** 2,
                         noise_free_err_rms=float(np.sqrt(np.mean(
                             (exact - exact[0]) ** 2)))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("notes/exact_stages.json"))
    parser.add_argument("--part", choices=("aia", "step_field", "step_bias", "both"),
                        default="both")
    args = parser.parse_args()

    result = {}
    if args.part in ("aia", "both"):
        result["aia"] = run_aia(args.trials, args.seed)
    if args.part in ("step_field", "both"):
        result["step_field"] = run_step_field(min(args.trials, 4_000), args.seed)
    if args.part == "step_bias":
        result["step_bias"] = run_step_bias(min(args.trials, 1_500), args.seed)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
