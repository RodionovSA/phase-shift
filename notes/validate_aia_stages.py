# notes/validate_aia_stages.py
"""CPU Monte Carlo for PLAN.md Step 1; writes measurements as JSON."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase_shift.methods import aia, frame_step, pixel_step
from phase_shift.utils import wrap


def _moments(plus: np.ndarray, minus: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Antithetic even mean and squared even/odd norms."""
    even = (plus + minus) / 2
    odd = (plus - minus) / 2
    return even, float(even @ even), float(odd @ odd)


@dataclass
class _Accumulator:
    sums: np.ndarray
    even_sq: np.ndarray
    odd_sq: np.ndarray
    counts: np.ndarray

    @classmethod
    def create(cls, blocks: int, size: int) -> "_Accumulator":
        return cls(np.zeros((blocks, size)), np.zeros(blocks), np.zeros(blocks),
                   np.zeros(blocks, dtype=int))

    def add(self, group: int, plus: np.ndarray, minus: np.ndarray) -> None:
        mean, even_sq, odd_sq = _moments(plus, minus)
        self.sums[group] += mean
        self.even_sq[group] += even_sq
        self.odd_sq[group] += odd_sq
        self.counts[group] += 1


def _statistics(groups: _Accumulator) -> dict:
    """Variance, bias, and paired delete-group jackknife confidence interval."""
    sums, even_sq, odd_sq, counts = groups.sums, groups.even_sq, groups.odd_sq, groups.counts

    def evaluate(total: np.ndarray, e2: float, o2: float, count: int) -> tuple:
        mean = total / count
        even_var = max((e2 - count * (mean @ mean)) / (count - 1), 0.0)
        variance = (o2 / count + even_var) / mean.size
        bias = float(np.sqrt(np.mean(mean ** 2)))
        bias_se = float(np.sqrt(even_var / (count * mean.size)))
        return variance, bias, bias_se

    total, e2, o2, count = sums.sum(0), even_sq.sum(), odd_sq.sum(), counts.sum()
    variance, bias, bias_se = evaluate(total, e2, o2, count)
    leave_one_out = [evaluate(total - s, e2 - e, o2 - o, count - c)[0]
                     for s, e, o, c in zip(sums, even_sq, odd_sq, counts)]
    return dict(variance=variance, bias_rms=bias, bias_mean_rms_se=bias_se,
                leave_one_out=leave_one_out)


def _report_pair(base: dict, fitted: dict, pixels: int) -> dict:
    coefficient = pixels * (fitted["variance"] / base["variance"] - 1)
    jackknife = pixels * (np.array(fitted["leave_one_out"])
                         / np.array(base["leave_one_out"]) - 1)
    count = len(jackknife)
    se = np.sqrt((count - 1) / count * np.sum((jackknife - jackknife.mean()) ** 2))
    return dict(k=coefficient, k_ci95_half_width=float(1.96 * se),
                baseline_variance=base["variance"], fitted_variance=fitted["variance"],
                bias_rms=fitted["bias_rms"], bias_mean_rms_se=fitted["bias_mean_rms_se"],
                baseline_bias_rms=base["bias_rms"])


def measure(pixels: int, frames: int, pairs: int, blocks: int, sigma: float,
            seed: int, estimator: str, background_overlap: float = 0.0) -> dict:
    """Compare frozen-step, conditional-frame, and converged phase estimates."""
    rng = np.random.default_rng(seed)
    phi = 2 * np.pi * np.arange(pixels) / pixels + 0.173
    delta = 2 * np.pi * np.arange(frames) / frames
    g = np.ones(frames)
    u, v = np.cos(phi), -np.sin(phi)
    truth = 2 + background_overlap * u + np.cos(phi[None, :] + delta[:, None])
    names = ["baseline", "stage2", "stage3"]
    gauges = ["truth_origin", "spatial_mean_removed"]
    data = {(name, gauge): _Accumulator.create(blocks, pixels)
            for name in names for gauge in gauges}
    step_data = {name: _Accumulator.create(blocks, frames) for name in names[1:]}
    failed = {name: 0 for name in names[1:]}
    max_iterations = {name: 0 for name in names[1:]}

    for trial in range(pairs):
        noise = sigma * rng.standard_normal(truth.shape)
        phase_errors = {name: [] for name in names}
        step_errors = {name: [] for name in names[1:]}
        for sign in (1, -1):
            stack = truth + sign * noise
            _, ub, vb = pixel_step(stack, delta, g, precision="double")
            phase_errors["baseline"].append(wrap(np.arctan2(-vb, ub) - phi))
            if estimator == "conditional":
                steps, gains, _ = frame_step(stack, u, v)
            for name, fit_gain in (("stage2", False), ("stage3", True)):
                if estimator == "conditional":
                    _, uf, vf = pixel_step(stack, steps, gains if fit_gain else g,
                                          precision="double")
                    recovered = np.arctan2(-vf, uf)
                    fitted_steps = steps
                else:
                    _, _, recovered, fitted_steps, _, diagnostics = aia(
                        stack.reshape(frames, 1, pixels), g, fit_gain=fit_gain,
                        delta0=delta, iters=300, tol=1e-11, precision="double")
                    recovered = recovered.ravel()
                    failed[name] += int(not diagnostics.converged)
                    max_iterations[name] = max(max_iterations[name], diagnostics.iters_run)
                phase_errors[name].append(wrap(recovered - phi))
                step_errors[name].append(wrap(fitted_steps - delta))
        group = trial % blocks
        for name in names:
            plus, minus = phase_errors[name]
            data[name, "truth_origin"].add(group, plus, minus)
            data[name, "spatial_mean_removed"].add(
                group, plus - plus.mean(), minus - minus.mean())
        for name in names[1:]:
            step_data[name].add(group, *step_errors[name])

    measurements = {}
    for gauge in gauges:
        baseline = _statistics(data["baseline", gauge])
        measurements[gauge] = {
            name: _report_pair(baseline, _statistics(data[name, gauge]), pixels)
            for name in names[1:]
        }
    steps_report = {name: {key: value for key, value in _statistics(groups).items()
                           if key != "leave_one_out"}
                    for name, groups in step_data.items()}
    return dict(pixels=pixels, frames=frames, sigma=sigma, seed=seed,
                background_overlap=background_overlap,
                independent_pairs=pairs, blocks=blocks, estimator=estimator,
                nonconverged=failed, max_iterations=max_iterations,
                phase=measurements, steps=steps_report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=2048)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--background-overlap", type=float, default=0.0)
    parser.add_argument("--estimator", choices=["conditional", "aia"], default="conditional")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.blocks < 4 or args.pairs < 2 * args.blocks or args.pairs % args.blocks
            or args.frames < 5 or min(args.pixels) < 8 or args.sigma <= 0):
        parser.error("require N >= 5, pixels >= 8, sigma > 0, and equal blocks of >= 2 pairs")
    report = dict(coefficient="variance_fitted / variance_baseline = 1 + k / N_p",
                  confidence="95% normal, paired delete-group jackknife; independent noise pairs",
                  results=[])
    for pixels in args.pixels:
        result = measure(pixels, args.frames, args.pairs, args.blocks, args.sigma,
                         args.seed + pixels, args.estimator, args.background_overlap)
        report["results"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
