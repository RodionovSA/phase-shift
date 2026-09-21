# Exact fitted-step phase error: derivation and measurement

Covers PLAN.md Steps 2–6. The document changes are in `docs/aia.md`
§"Phase-error covariance", `docs/sf_aia.md` §"Noise of the corrected solve",
and `docs/vp_aia.md` §"Noise of the corrected estimates"; the implementation
is `src/phase_shift/errors.py`; the experiment is
[`validate_exact_stages.py`](validate_exact_stages.py).

## What was derived

Writing the frame-step solution out, rather than treating its error as an
independent perturbation, closes the Stage-2/3 gap in one step. The frame
step's error is a pixel-weighted sum of that frame's own noise,

```
(e_Pn, e_Qn) = sum_{x,y} ell(x,y) eps_n(x,y),
ell(x,y)     = [A_ps^-1 (1, u, v)^T]_{P,Q}                       aia.md Eq. (38)
```

which is Eq. (34) transposed: `k_n` weights frames at one pixel, `ell` weights
pixels within one frame. Two consequences follow immediately. The frame-step
errors are uncorrelated across frames, because each involves one frame's noise
alone. And the correlation between `e_delta_n` and the pixel's own noise —
the term the previous text could not sign — is just `ell(x,y)` evaluated at
that pixel.

Collecting both displacements of `(u, v)` gives one expression covering both
stages, with `q = (u, v)` at the pixel:

```
(e_u, e_v) = sum_n k_n eta_n,
eta_n      = eps_n(x,y) - q^T Pi_n (e_Pn, e_Qn)                  aia.md Eq. (39)
sigma_Phi^2 = (1/b^2) sum_n (w.k_n)^2 Gamma_n,
Gamma_n     = sigma^2 [1 - 2 q^T Pi_n ell] + q^T Pi_n M Pi_n q   aia.md Eq. (40)
```

`Pi_n = w_n w_n^T` when only `delta_n` is fitted, `I_2` when `g_n` is fitted
too, and `0` recovers Stage 1 exactly since `sum_n k_n k_n^T = S`. At
`Pi_n = I_2` the result no longer depends on the frame, so Stage 3 is Stage 1
times one scalar per pixel (Eq. 45). The gain variance and the step–gain
covariance are the radial and mixed projections of the same `M` that Eq. (36)
projects tangentially (Eq. 43), with `g_n^2 sigma_delta_n^2 + sigma_g_n^2 =
tr M` as the frame-side counterpart of Eq. (27).

## What was measured

All variances are about each estimator's own mean, from antithetic `±` noise
pairs; no mean squared error about the truth is used anywhere.

**Ideal configuration, analytic.** Eqs. (40) and (45) reduce to the
documented `1 - 3/(2 N_p)` and `1 - 2/N_p` at every pixel and for every
`N >= 3`, to 13 digits. The one exception is Stage 2 at `N = 4`, where the
coefficient is `-3/2 - (1/2) cos(4 Phi)`; its field average is again `-3/2`.
Eq. (38) and Eq. (41) of the previous numbering are therefore exact readings
of the new results, not approximations, which is consistent with the Step-1
measurement that validated them.

**Non-ideal configuration, Monte Carlo.** Irregular steps, per-frame gains, a
contrast varying by a factor of two across the field, and a spatially varying
`sigma`. Measured over `N` in {5, 7, 9} and `N_p` in {256, 576}, 25 000
antithetic pairs each, the ratio of the measured `O(1/N_p)` correction to the
predicted one is `1.00 ± 0.02` in every case. The prediction is insensitive to
the noise scale over a factor of 10, so the agreement is not a second-order
artefact.

**One trap worth recording.** The first non-ideal run put Stage 2 29% above
prediction while Stage 3 agreed. The cause is the frame-independent
displacement `(a_u, a_v)` of `aia.md` §"Dropping the background field": when
the background overlaps the fringe pattern, the frame step returns
`(P_n + a_u, Q_n + a_v)`. With `g_n` fitted that displacement is a gauge
direction and leaves `Phi` untouched; with `g_n` held fixed, projecting the
radius back onto the true `g_n` breaks the invariance and the displacement
leaks into `Phi`. It is a bias, not a variance, and it is the same effect
Step 1 controlled for. Removing the background's overlap with `{1, u, v}`
brings Stage 2 to `1.015`. This is now stated in `aia.md` §"Validity".

**SF-AIA, Eq. (E7).** One corrected pass with `delta_n`, `g_n` at the truth:
with constant noise, the measured added variance matches Eq. (E7) to within
`3-6%` across `N` in {7, 9} and `J` in {2, 5}, and Eq. (E8)'s `J/(4 N_p)`
reading about as well. The sign is positive — the fitted field adds noise. The
previous implementation subtracted a leverage discount, so both its sign and
its magnitude were wrong; it has been replaced by Eq. (E7) itself.

**Eq. (E7) was homoscedastic.** With `sigma_0` varying across the field by a
factor of two, the measured term ran `27-38%` above the formula, while the
AIA Stage-2/3 results held. Tracing the derivation, the covariance of the
coefficients is

```
Cov(c_n, c_m) = Pi_nm G^(n)^-1 [sum_p sigma_0^2(p) w_n w_m p_j p_j'] G^(m)^-1
```

so the middle Gram carries the noise weight while `G^(n)` stays unweighted,
Eq. (E1) being an unweighted fit. The old form pulled a single `sigma_0^2`
out of both. Writing the weighted `D_n^T L D_m` brings the same configurations to
`0.97, 0.98, 1.05`. `sf_aia.md` Eq. (E7) and `errors.py` now carry it;
for constant `sigma_0` nothing changes, since it is `sigma_0^2 D_n^T D_m`.
The AIA results needed no such fix: Eqs. (38)-(40) carry `sigma^2(x,y)` per
pixel from the start, which is why they matched on a varying-`sigma` field.

**SF-AIA's refinement loop.** Measured over rounds at `N_p = 256`, the added
variance keeps growing: at `N = 9, J = 5` it runs
`1.84, 3.62, 5.12, 6.37, 8.19, 9.34, 10.49, 11.12` over rounds
`1, 2, 3, 4, 6, 8, 12, 20`, against Eq. (E7)'s one-pass `1.85`, Eq. (E8)'s
`J/4 = 1.25` and `vp_aia.md` Eq. (30)'s `J = 5`. It passes Eq. (30) after
about three rounds and saturates two to three times above it; at `N = 7,
J = 2` it is still climbing at twenty rounds. So the loop does not converge to
the unbiased one-pass estimator's noise — the alternation's fixed point is a
different, noisier one. Eq. (E9) is a floor for a multi-round solve, not an
estimate of it, and the algorithm's keep-the-best-round rule is what limits
this in practice. Deriving the loop's fixed-point noise is left open in
PLAN.md.

**The SF-AIA/VP-AIA factor of four.** `sf_aia.md` Eq. (E8) gives
`1 + J/(4 N_p)` where `vp_aia.md` Eq. (30) gives `1 + J/K`. Working both
through by hand, each reduces to a quadratic form in the `N x N` matrix with
entries `Pi_nm cos(delta_n - delta_m)`, whose eigenvalue on the second
temporal harmonic is `1/2`. SF-AIA's per-frame fit multiplies by that
eigenvalue; VP-AIA's joint fit inverts it. The ratio is `2 / (1/2) = 4`. It is
the square of the factor of two in `sf_aia.md` §"Bias of a single pass": one
pass recovers the second harmonic of `c_jn` at half size and attenuates its
noise by the same half. Both derivations are correct; the difference is the
estimator.

## The bias these variances sit beside

Everything above is a variance of order `1/N_p`. Measuring the accompanying
bias turned up the one result here that does not shrink with the field.

Run `--part step_bias`. The configuration holds the fringe content fixed —
same fringe count, same `a` and `b` shapes, same steps, gains and `sigma` —
and changes only the sampling density, so `N_p` is the only variable. The bias
is read off the even part of antithetic noise pairs, which cancels the
variance term exactly rather than averaging it down.

| `N_p` | bias rms | `+/- se` | `sigma_delta` | bias / `sigma_delta` |
|---|---|---|---|---|
| 256 | 8.116e-4 | 1.0e-5 | 5.5e-3 | 0.15 |
| 1 024 | 8.127e-4 | 5.0e-6 | 2.8e-3 | 0.29 |
| 4 096 | 8.143e-4 | 3.4e-6 | 1.4e-3 | 0.59 |
| 16 384 | 8.109e-4 | 2.7e-6 | 6.9e-4 | 1.17 |
| 65 536 | 8.138e-4 | 2.0e-6 | 3.5e-4 | 2.36 |

Flat to four digits over a 256x range. Three further checks fix the
mechanism:

- **It is exactly `O(sigma^2)`.** At `N_p = 4096`, `bias/sigma^2` is
  `8.143e-4, 8.149e-4, 8.151e-4` over a 16x range in `sigma^2`.
- **It is not model error.** A noise-free solve of the same stack recovers
  `delta_n` to `2.5e-12` rad at every `N_p`, converged. The data is the pure
  AIA model of Eq. (3), and `g_n` is handed to the solver at its true value.
- **It is the alternation, not the frame step.** One frame step from the
  *true* `(u, v)`, same noise, gives `1.9e-6, 8.1e-8, 1.9e-9` at
  `N_p = 256, 4096, 65536`, each at or below its own standard error. The
  estimator Eqs. (35)-(36) describe is unbiased; the loop around it is not.

`O(sigma^2)`, created by regressing on estimated quadratures, and independent
of sample size: that is errors-in-variables, and it is *inconsistent* — more
pixels never remove it, because every pixel added carries the same relative
regressor error. Since `sigma_delta` falls as `1/sqrt(N_p)` and the bias does
not, `delta_n` on a full frame is bias-dominated, roughly `9 sigma_delta` at a
megapixel. The direction of the bias has not been characterised, so
"attenuation" is not claimed.

Neither this nor the background-overlap bias is a convergence artefact:
tightening `tol` from `1e-4` to `1e-11` leaves the converged bias unchanged
(`1.80e-2` both, within 0.2%), and only `tol = 1e-3` adds a visible ~15%.

## Limits

- Everything in the variance sections is first order in the noise and
  conditions the frame step on noise-free `(u, v)`, which is Eq. (35)'s own
  assumption. It describes one pass taken from the true fields, not the fixed
  point of a converged alternation. Step 1 already showed that the converged
  AIA solver has a different `O(1/N_p)` coefficient, and the iteration sweep
  recorded in PLAN.md shows the coefficient moving continuously between the
  two; the bias section above is the other half of that same gap.
- Eq. (E9) adds two exact terms and omits their cross-correlation, which is of
  the same `O(1/N_p)` order.
- `errors.py` reproduces the reference implementations to machine precision in
  float64. In float32 the field mean of the SF-AIA term matches to five
  digits; individual pixels where the term is three orders of magnitude below
  the field mean can be off by tens of percent of their own tiny value.
