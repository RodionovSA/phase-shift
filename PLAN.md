# Plan: exact phase-error model, gauge unification, docs, tests

The `methods/` restructure is done and committed (`702cff2`, `08e21ba`). This
plan covers the phase-error work that followed, plus the gauge unification and
the documentation and test gaps the restructure exposed.

**Direction.** `PhaseResult.phi_error` is computed from the *exact* covariance
forms, for every method. The asymptotic and averaged forms stay in the
documents as analysis, each with the conditions under which it is valid,
rather than as the basis of a computed error.

Each document already separates the two, and each stops at the same place:

| method | exact form | approximation | missing |
|---|---|---|---|
| AIA | Eq. (26) pixel, Eq. (36) step | Eq. (29), (30), (37), (40); (38)/(41) | exact `sigma_g_n^2`; exact `sigma_Phi^2` for fitted steps (Stages 2, 3) |
| SF-AIA | Eq. (E7) | Eq. (E8), `1 + J/(4*N_p)` | fitted `P_n`, `Q_n` |
| VP-AIA | Eq. (29) | Eq. (30), `1 + J/K` | fitted `P_n`, `Q_n` |

Both field-fitting methods state that they "treat `P_n`, `Q_n` as exact; the
noise of their estimates is not included here", deferring it to `aia.md` —
which supplies it only through the Stage-2/3 asymptotics this plan cannot
validate. So the exact fitted-step covariance of step 2 is the missing piece
for all three, and each method's error is then that term composed with its own
field-fit term.

## Validation status

Monte Carlo against the document, all with `delta`/`g` held at the truth so
each formula is tested on its own terms:

| Equation | what it gives | measured / predicted |
|---|---|---|
| (26) | `sigma_Phi^2`, `sigma_b^2` per pixel | 0.9993, 0.9999 |
| (27) | `b^2 sigma_Phi^2 + sigma_b^2 = tr(C^-1) sigma^2/N` | 0.9997 |
| (29) | pixel-averaged `sigma_Phi^2` | 0.9999 |
| (30) | ideal case `sqrt(2/N) sigma/b` | 1.0007 |
| (35) | `Cov(c_n, P_n, Q_n)` | within 1% |
| (36) | `sigma_delta_n^2` | 0.9996 |
| (37) | isotropic `sigma_delta_n^2` | 1.0000 at >= 1 fringe |

Every exactly testable statement in §"Phase-error covariance" holds, and
`errors.py`'s `simplified=True` path reproduces Eq. (26) to 0.1%.

**Withdrawn.** The earlier finding that Eq. (38)/(41) have the wrong sign is
*not supported*. Those measurements compared the mean squared error about the
true `delta`, not a variance about the estimator's own mean. The frame step's
`delta_n` carries a bias of `1-5e-4` rad in that configuration — `bias^2` up to
0.43 of the variance — which inflated the comparison by 29% where Eq. (36) is
in fact correct to 0.04%. A `1/N_p` effect is far smaller than that artefact.

## Step 1. Re-measure Stages 2 and 3 correctly

Variance about each estimator's own mean, with the bias reported separately
rather than folded in, and the global phase mode handled identically in both
arms. Sizes from `N_p = 256` up to where the coefficient stabilises — Stage 3
was still 18% from its asymptote at `N_p = 144`, so the small fields used
before are unusable for this.

Deliverable: `k` with an error bar for Stage 2 and Stage 3, and a separate
statement of the bias, which matters for an MSE-based error budget even though
Eq. (36) is a variance.

## Step 2. Derive the missing exact formulas for Stages 2 and 3

`aia.md` gives an exact per-frame step variance (Eq. 36) and nothing else
exact beyond it: the gain variance appears only in its isotropic form
(Eq. 40), and the phase error under fitted steps exists only as the
ideal-configuration readings Eq. (38) and Eq. (41). Three exact results have
to be derived and added, in the style of Eq. (26)/(36):

1. **Exact `sigma_g_n^2`**, the Eq. (36) counterpart for the gain: project
   Eq. (35)'s `Cov(c_n, P_n, Q_n)` onto `r_n = (cos delta_n, sin delta_n)`
   instead of `w_n`. Eq. (40) then becomes its isotropic reading, as Eq. (37)
   is for Eq. (36). The cross term `Cov(delta_n, g_n)` comes from the same
   projection and is needed by 2 and 3.
2. **Exact `sigma_Phi^2` for Stage 2**, fitted steps and known gains:
   propagate `e_delta_n` into the pixel step through the exact sensitivity of
   Eq. (33), keeping its correlation with the pixel noise, since `delta_n` is
   fitted from the same frames. Eq. (38) becomes the ideal-case reading of
   this.
3. **Exact `sigma_Phi^2` for Stage 3**, fitted steps and gains: as 2, with
   `e_g_n` through Eq. (39) and the `Cov(delta_n, g_n)` cross term added.
   Eq. (41) becomes its ideal-case reading.

That correlation with the pixel noise is where the present Stage-2/3 argument
does its work, and it is the part this plan could not validate, so the
derivation states it explicitly rather than asserting a factor. The result
should be a per-pixel quadratic form in small `(2N)`-sized matrices,
computable alongside Eq. (26) without an `(N, H, W)` temporary.

*Deliverable:* three new numbered equations in §"Phase-error covariance", with
Eq. (38)/(40)/(41) retained as their approximate readings under stated
conditions (step 4). New theory in a settled document, so written and agreed a
step at a time per `docs/AGENTS.md`, and checked against step 1's measurement
before it is implemented.

## Step 3. Rewrite `errors.py` on the exact forms

`phi_error` becomes Eq. (26) plus step 2's exact result — its Stage-2 form
when `fit_gain` is False, its Stage-3 form when True — replacing the present
Eq. (37)/(40)-based correction. `PhaseConfig.phi_error_simplified` keeps its
meaning, baseline only or baseline plus the fitted-step contribution, but both
branches are then exact rather than asymptotic.

*Verification:* against step 1's Monte Carlo at several `N`, `N_p` and step
distributions, including the poorly conditioned ones where the approximations
fail.

## Step 4. Document the approximations' validity

Keep the approximate forms as analysis, each with the measured conditions
under which it may be used. In `aia.md`:

- **Eq. (29)** needs the second harmonic of `Phi` to average out, not "many
  fringes": it is exact for `Phi` spanning any interval of length `pi`
  (measured 1.0000), 25% off at `pi/2`, and a factor 2 off for a nearly flat
  field. It separately needs `Phi` uncorrelated with `sigma/b` — with
  `b = 1 + 0.6 cos(Phi)` it is 16% low even for a well-spread `Phi`.
- **Eq. (37)** is exact to 4 digits at one fringe or more, but 3.3x off at half
  a fringe and 22x at a quarter. The condition that fails first is not the
  stated `Suu ~ Svv`, `Suv ~ 0` — both hold exactly at half a fringe — but the
  unstated `Su ~ Sv ~ 0`, the quadratures being orthogonal to the design's
  constant column.
- **Eq. (38), (40) and (41)** become the approximate readings of step 2's
  exact results, each with its regime of validity, measured in step 1. Stage 3
  was still 18% from its asymptote at `N_p = 144`, so that regime is not
  academic: it is where the small fields of a cropped ROI actually sit.

## Step 5. SF-AIA: re-measure, rewrite to Eq. (E7), then compose

Same treatment as AIA, in three parts.

1. **Re-measure as a variance.** The earlier SF-AIA result used the same MSE
   quantity as the withdrawn Stage-2 finding. It is on firmer ground — it
   compared two solves of the same estimator with `delta`/`g` fixed, and
   reproduced Eq. (E8)'s `J` scaling at two basis sizes — but it is not
   trusted until re-run.
2. **Implement the exact Eq. (E7)**, with `Pi`, `t_n` and `D_n' D_m`, in place
   of the present leverage-discount construction, which has the opposite sign
   and six times the magnitude (`3*J/(2*N_p)` against `J/(4*N_p)`); a sign
   flip alone would not fix it. Eq. (E8) moves to §"Validity" as the
   uniform-steps reading, with the conditions it needs: `N >= 5`, constant
   `g_n` and `b`, many fringes, and an orthonormal basis.
3. **Compose with the fitted-step term** of step 2, which Eq. (E7) explicitly
   leaves out. The SF-AIA error is the Eq. (26) baseline, plus the fitted
   `delta_n`/`g_n` contribution, plus Eq. (E7)'s field-fit contribution —
   each exact, with the correlations between them stated rather than assumed
   away.

## Step 6. VP-AIA: the same treatment in the document

No implementation yet, so this step is `vp_aia.md` only, and it mirrors
step 5's structure: Eq. (29) is already the exact form and Eq. (30) the
uniform-steps approximation, so the work is to state Eq. (30)'s validity
conditions (`N >= 5`, constant `g_n` and `b`, many fringes, orthonormal
`H_j`), and to note that Eq. (29) excludes the fitted-step noise that step 2
supplies.

One cross-document check belongs here. For the same quantity — a `J`-mode
phase-step field fitted from the same data — `sf_aia.md` Eq. (E8) gives
`1 + J/(4*N_p)` and `vp_aia.md` Eq. (30) gives `1 + J/K`, a factor of 4 apart
with `K = N_p`. The estimators differ (VP-AIA projects out the pixel
corrections `a1, u1, v1`, SF-AIA does not), so the coefficients may legitimately
differ, and the measured SF-AIA value sits at `J/4`. Worth confirming that the
factor is the estimator and not an error in one of the two derivations, since
the two documents otherwise describe the same physical effect.

## Step 7. Unify the contrast-scale gauge on the median

`vp_aia.md` fixes the contrast scale with `mean(g_n) = 1`; `aia.md` Eq. (16),
`sf_aia.md` and the code use `median(g_n) = 1`. Adopt the median: it is what
the implementation does, and it is robust to a single frame whose gain
collapses — the vibration-corrupted frame `g_min_ratio` exists to flag, which
would drag a mean.

Four local places in `vp_aia.md`: the definition in §"Gauge conventions",
normalization step 4, the sentence in §"Quadrature frame" naming the
condition, and the two degree-of-freedom counts, which need only that the
contrast scale is one scalar condition and are checked rather than changed.
Nothing differentiates through the scale, so the substitution is free.
`gauge_conventions.md`'s VP-AIA row then loses its "differs from AIA" note.

## Step 8. README

Delete the `ripple.py` bullet; add `basis.py`, `methods/steps.py`,
`methods/gauge.py`, `methods/diagnostics.py`; correct `README.md:46` from
Eq. (8) to Eq. (17); document `StepFieldParam` alongside `AIAParam`.

## Step 9. Invariant tests for the shared modules

Direct tests for what the methods now rely on: `whiten_uv`'s Eq. (15)
conditions; `normalize_gain`'s `median(g) = 1`; `center_offsets` and
`center_coeffs`' zero means (Eq. 16, T3b); `pin_phase_origin`'s
`delta[0] = 0`; `spatial_basis`' zero spatial mean and orthonormality for
every registered family, and its `ValueError` on an unknown one;
`MethodParam`'s default `phi_error` of `None`. Plus a cheap analytic
regression test of Eq. (26) against Eq. (30) in the ideal configuration.

## Step 10. Degenerate bases

`spatial_basis` returns silent all-NaN rows when a coordinate is constant over
the field (1 pixel wide or tall, and `2x3`/`3x2` at higher degree). Raise
`ValueError` naming the shape and the family arguments instead.

## Open question

The frame step's `delta_n` is biased by `1-5e-4` rad in the configuration
tested, varying with `delta_n`, with `bias^2` reaching 0.43 of the variance.
Eq. (36) is a variance and is correct as such, so an error budget quoted as an
RMS deviation from the true step is larger than Eq. (36) suggests. Worth
deciding whether the bias belongs in the documented error model.

## Deferred

- **Working dtype**, your own pass: `aia` returns `b` as float64 while
  `a`/`phi` are float32, which is why `phi_error` comes back float64;
  `utils._estimation_weight` is the one place the carrier, reference and
  combine weight dtype is decided.
- **`aia`'s joint-gain buffer**: `pinv(A) @ (I - c*1')` rearranged to remove
  the `(N, P)` temporary, ~340 MB at 12 frames and 7 Mpx. Not bit-identical.
- **`step_field_quality`'s residual**: chunked scoring would drop another
  full-size array per refine round.
- **`aia_step_field` -> `sf_aia`**, `StepFieldParam` -> `SFAIAParam`.
- **VP-AIA implementation**: its own plan, on the shared modules.
- **Carrier removal joint fit** (`carrier_removal.md` §2-3): unapproved
  theory; `carrier.py` keeps today's algorithm.
