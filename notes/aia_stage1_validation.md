# AIA covariance: Step 1 measurements

2026-09-20. Numerical investigation for `PLAN.md`; this is not an amendment to
the settled theory or an approved implementation specification.

## Experiment

NumPy float64, eight uniform steps, unit gain and fringe amplitude,
`Phi[p] = 2*pi*p/N_p + 0.173`, constant background 2, independent Gaussian
intensity noise with standard deviation 0.01. This discrete phase grid has
exactly zero first and second circular moments. The baseline uses the true
steps and gains in the package's `pixel_step`.

Two estimators are measured separately:

1. **Conditional:** `frame_step` uses the true, noise-free quadratures. The
   same noisy stack is then passed to `pixel_step`, with fitted steps alone
   (Stage 2), or fitted steps and gains (Stage 3). There is no whitening,
   phase-origin pinning, gain normalization, or offset subtraction in this
   conditional experiment: it isolates the sensitivities in Eqs. (33)/(39).
2. **AIA:** the actual package solver, including its gauges, whitening and
   offsets, initialized at the true steps. Both stages use `tol=1e-11`,
   `iters=300`. This measures the local converged estimator; it does not test
   convergence from an inaccurate initialization.

Each independent Gaussian noise draw is evaluated with both signs. Baseline
and fitted arms share that draw. If their wrapped phase errors are `e+` and
`e-`, define `m=(e+ + e-)/2` and `h=(e+ - e-)/2`. Symmetry gives `E[h]=0`.
For `R` independent pairs, estimate each pixel's variance by

```
mean(h**2) + sum((m - mean(m))**2)/(R - 1).
```

Average these variances across pixels, then form
`variance_fitted / variance_baseline = 1 + k/N_p`. The variance is about each
estimator's own mean, never its mean squared error about the truth. Bias is
estimated independently as the per-pixel mean of `m`. Report its spatial RMS
and the RMS sampling standard error of that mean; an RMS at its sampling
floor is not evidence of nonzero bias.

Use 64 equal independent groups and paired delete-group jackknife standard
errors for `k`; the intervals below are normal 95% intervals. The two signs
are **not** counted as independent trials. The script accumulates group
moments, retaining no ensemble-sized stacks. Memory grows with the pixel
count and number of groups, not the trial count.

## Conditional calculation

8,192 independent pairs per size. With the true quadratures supplying the
absolute phase reference, retain that reference in both arms:

| Pixels | Stage 2: k (95% interval half-width) | Stage 3: k (95% interval half-width) |
|---:|---:|---:|
| 256 | -1.530 +/- 0.040 | -2.044 +/- 0.048 |
| 1,024 | -1.477 +/- 0.041 | -1.988 +/- 0.044 |
| 4,096 | -1.511 +/- 0.047 | -2.020 +/- 0.053 |

These agree with -3/2 and -2, respectively. The sign withdrawal in `PLAN.md`
stands: these measurements do not refute the document's conditional ideal
calculation. Baseline variance is within 0.08% of `2*sigma**2/N` throughout.
The conditional step variance divided by `2*sigma**2/N_p` is 0.9942, 1.0019,
and 1.0000.

Phase bias RMS is about `2.8e-7` rad, at its sampling floor. Conditional step
bias RMS ranges from `4.6e-10` to `8.9e-9` rad, also consistent with its
sampling floor. These ideal data do not reproduce the earlier configuration's
reported `1e-4` to `5e-4` rad bias.

## Identical removal of the global phase mode

Subtract the spatial mean of each wrapped phase-error map, separately in
every realization and in **both** arms. This removes only the constant phase
mode. It changes the baseline as well as the fitted result, so the preceding
coefficients cannot be compared directly with this table.

| Pixels | Conditional Stage 2 | Conditional Stage 3 | AIA Stage 2 | AIA Stage 3 |
|---:|---:|---:|---:|---:|
| 256 | -0.515 +/- 0.027 | -1.031 +/- 0.038 | +1.021 +/- 0.080 | -1.021 +/- 0.062 |
| 1,024 | -0.477 +/- 0.028 | -0.988 +/- 0.037 | +1.033 +/- 0.070 | -0.977 +/- 0.056 |
| 4,096 | -0.504 +/- 0.032 | -1.013 +/- 0.039 | +1.032 +/- 0.084 | -1.038 +/- 0.055 |

AIA uses 4,096 independent pairs per size. All 49,152 AIA solves converged;
the maximum iteration counts were 29 for Stage 2 and 4 for Stage 3. The
coefficients are stable within the reported precision over this size range.
Mean-removed phase bias RMS is `3.8e-7` to `4.0e-7` rad, at its sampling floor.

At ten times lower noise (`sigma=0.001`, 1,024 pixels, 1,024 independent
pairs), AIA gives `k=+0.958 +/- 0.149` and `-0.969 +/- 0.115`; all 4,096
solves converge. The Stage-2 difference persists in the small-noise regime.

The JSON also retains uncentered AIA measurements under `truth_origin` to
make the effect of phase-origin handling visible. AIA pins the first fitted
step, whereas the baseline knows that step exactly. Those raw measurements
therefore contain different global-mode uncertainty and are **not** the
matched comparison used for the conclusions above.

## Bias control

Repeat the conditional 1,024-pixel experiment with
`a[p] = 2 + 0.0005*u[p]`. The background projection in `aia.md`,
"Dropping the background field", gives a noise-free fitted frame phasor
`(cos(delta_n)+0.0005, sin(delta_n))`. Its known angular bias is
`atan2(sin(delta_n), cos(delta_n)+0.0005) - delta_n`, wrapped.

Measured step bias RMS is `3.53553368e-4` rad; its RMS sampling standard
error is `2.16e-9` rad. Step variance is `1.95679977e-7` rad squared, versus
`1.95679417e-7` without the overlap. Bias squared adds approximately 0.639
times the variance to the MSE, while the variance itself barely changes.
This is a controlled example of the pitfall; the original experiment's
configuration was not recorded in the plan and has not been reconstructed.

## Consequence for Step 2

Eq. (35) explicitly conditions on noise-free quadratures. The package's
converged AIA does not. Even after identical global-mode removal, their
Stage-2 corrections differ in sign and magnitude. It is therefore unsafe to
implement the conditional propagation as the solver's covariance. Agreement
of the two Stage-3 coefficients in this ideal configuration does not establish
their equivalence for arbitrary phase, gain, background, or step coverage.

Proposed next substep, for agreement before changing `docs/aia.md`: let `V_n`
be the `(P,Q)` covariance block of Eq. (35), `r_n=(cos(delta_n),sin(delta_n))`,
and `w_n=(-sin(delta_n),cos(delta_n))`. Add the conditional first-order results
`Var(g_n)=r_n.T @ V_n @ r_n` and
`Cov(delta_n,g_n)=w_n.T @ V_n @ r_n / g_n`. Then derive the coupled response
of the actual alternating solves, with its gauges, for runtime phase errors.
The scalar coefficients alone are not a substitute for that derivation.

Here "exact" must retain the section's first-order meaning: the quadrature
regression covariance is exact, but propagation through `atan2`, `hypot`, and
the nonlinear fitted estimator is a small-noise linearization.

Scope still unvalidated: irregular or poorly conditioned step/phase coverage,
nonuniform gains/amplitudes/noise, SF-AIA, and measurements on the NVIDIA
Quadro 2200. No package code, public API, or settled theory was changed.

## Reproduction and checks

No environment synchronization or CUDA installation is needed:

```sh
uv run --no-sync python notes/validate_aia_stages.py --pairs 8192 --blocks 64 --output notes/aia_stage1_conditional.json
uv run --no-sync python notes/validate_aia_stages.py --estimator aia --pairs 4096 --blocks 64 --output notes/aia_stage1_solver.json
uv run --no-sync python notes/validate_aia_stages.py --pairs 8192 --blocks 64 --pixels 1024 --background-overlap 0.0005 --output notes/aia_stage1_bias_control.json
uv run --no-sync python notes/validate_aia_stages.py --estimator aia --pairs 1024 --blocks 32 --pixels 1024 --sigma 0.001 --output notes/aia_stage1_low_noise.json
uv run --no-sync pytest -k 'not device_cuda'
```

The existing CPU suite passes: 28 tests, one CUDA-specific check deselected.
The measurements use the existing NumPy 2.5.1 environment, Python 3.12.13.
Additional numerical checks passed for the antithetic variance/bias estimator,
the identical-arm paired ratio, the analytic ideal baseline, inclusion of the
conditional coefficients in their reported intervals, convergence of every
solver run, and the analytic background-overlap bias. `git diff --check` passed.
