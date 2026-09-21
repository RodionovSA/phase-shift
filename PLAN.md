# VP-AIA implementation plan

Implements `docs/vp_aia.md`: Variable-Projection AIA recovers the spatially
varying phase-step error `Delta_n(x, y)` as a first-order correction to the AIA
solution, in one pass, without SF-AIA's refinement loop and without its
attenuation bias (`vp_aia.md` §"Comparison with SF-AIA").

Status: awaiting agreement. Nothing implemented.

## 1. What the method does

| step | `vp_aia.md` | content |
|---|---|---|
| 1 | §"VP-AIA algorithm", Eqs. (5)-(11) | AIA baseline, final pixel step, normalization 1-4 |
| 2 | Eqs. (16)-(18) | residual `r_n`, sensitivity `w_n`, projector `Pi` |
| 3 | Eq. (23) | one chunked pass over pixels for every sum |
| 4 | Eqs. (22), (24), (25) | assemble `M`, `b`, `C`; solve `beta = (M + C'C)^-1 b` |
| 5 | Eqs. (26)-(27) | pixel corrections, corrected fields |
| 6 | §"Normalization" | steps 1-4 again on the corrected fields |

`beta` has length `2N + NJ` and does not grow with the pixel count, so step 4 is
a small dense solve whatever the frame size.

## 2. Verified before planning

A throwaway prototype of Eqs. (16)-(27) on a synthetic `9 x 64 x 80` stack with
`max|Delta_n| = 0.075 rad` and a degree-1 basis (`J = 2`):

- `alpha` recovered to `8.2e-4` relative error;
- phase RMS `0.174 deg` (AIA) -> `0.0147 deg` (VP-AIA), a factor of 12;
- `cond(M + C'C) = 1.7e6` at size 36.

Also checked numerically: `Pi` has rank `N-3` with `Pi A = 0` and `Pi^2 = Pi`
(Eq. 18); all three reductions of Eq. (23); and the Eq. (26) factorization of
§6.2 below. The prototype is not part of the deliverable.

## 3. The open question

**What should VP-AIA do when the user fixes the gain?**

`PhaseConfig(gain_mode="none")`, or passing `g=` directly, tells a method that
`g_n` is known and must not be fitted. `vp_aia.md` does not cover that case.
The first-order fit gives each frame two free corrections `P_n^(1), Q_n^(1)`,
and Eq. (24) pins six of those `2N` directions. With `g_n` fixed, `(P_n, Q_n)`
must stay on the circle of radius `g_n`, so only motion along the circle is
allowed -- one unknown per frame, the step angle, not two. That is a different
system from Eq. (22) and is not derived.

**Settled: raise `ValueError` asking for `gain_mode="joint"`.** VP-AIA invents
no mathematics for the fixed-gain case. Making it work means adding a section to
`vp_aia.md` first.

### Decided, stated here so they can be vetoed

- **Phase-noise map.** `StepFieldParam.phi_error` currently hardcodes
  `sf_aia.md` Eq. (E7); VP-AIA needs `vp_aia.md` Eq. (29). Move `phi_error` down
  into `SFAIAParam`, leave `phase_step_field` and the coefficient fields on the
  base, give `VPAIAParam` its own. Internal; `"sf_aia"` behavior unchanged.
- **Relinearization, Eq. (28).** Implemented, as `rounds: int = 1`, so the
  default is the single pass the method exists for.
- **Names.** `methods/vp_aia.py`, registry key `"vp_aia"`, function
  `aia_variable_projection`, matching `sf_aia.py` / `"sf_aia"` /
  `aia_step_field`.

## 4. Reused as is

- `methods/steps.py` — `pixel_step` (Eq. 8), `pixel_design` (Eq. 7).
- `methods/aia.py` — `aia` for the whole zeroth order (step 1).
- `basis.py` — `spatial_basis` is already `H_j` of Eq. (13): centered
  (`<H_j> = 0`) and orthonormal. Its module docstring already says so.
- `methods/step_field.py` — `StepFieldParam` as the base of `VPAIAParam`, and
  `step_field_quality` to score the recovered `Delta_n`. This is the payoff of
  the split that just landed.
- `methods/diagnostics.py` — `aia_diagnostics`, `cond2`.
- `backend.py` — `Precision`; `p.work` for the `(N, P)` and `(H, W)` arrays,
  `p.accum` for the pixel-sum pass, float64 for everything of size
  `2N + NJ`.

## 5. New and changed files

| file | change |
|---|---|
| `methods/gauge.py` | add `whitening_matrix`, `normalize_quadrature_frame` |
| `methods/vp_system.py` | **new** — Eqs. (22)-(25): pixel sums, `M`, `b`, `C`, solve |
| `methods/vp_aia.py` | **new** — `VPAIAParam`, `aia_variable_projection` |
| `errors.py` | add `vp_phi_error` (Eq. 29) |
| `methods/step_field.py` | move `phi_error` out to `SFAIAParam` (decision 2) |
| `methods/sf_aia.py` | receive that `phi_error` |
| `methods/__init__.py` | registry entry, re-exports |
| `docs/AGENTS.md`, `README.md` | VP-AIA in the method list |

`vp_system.py` is split from `vp_aia.py` because assembling and solving a
constrained normal system is its own topic; if it lands under ~120 lines it
should collapse back into `vp_aia.py` instead.

Import direction stays one-way: `solver -> vp_aia -> {vp_system, step_field,
aia, steps, gauge, diagnostics} -> {backend, basis, errors, utils}`.

## 6. Numerics

### 6.1 Never materialize `(N, P)`

`r_n` (Eq. 16) is stack-sized. Every sum of Eqs. (22)-(23) is accumulated over
pixel chunks, as `fit_step_field` and `step_field_phi_error` already do, so
`r_n` exists only one chunk at a time. Per chunk:

- 3 scalars: `sum u^2`, `sum v^2`, `sum uv`;
- `3J`: `sum u^2 H_j`, `sum v^2 H_j`, `sum uv H_j`;
- `3 J(J+1)/2`: the same weights against `H_j H_j'`;
- `2N + 2NJ` right-hand sides: `sum u r_n`, `sum v r_n`, `sum u H_j r_n`,
  `sum v H_j r_n`.

Eq. (23) turns every frame-indexed sum into these, so nothing is accumulated
per `(n, j)` pair over pixels.

### 6.2 Two traps

**The `(J, J, P)` temporary.** Writing `sum_xy u^2 H_j H_j'` as
`(H[:, None, :] * H[None, :, :]) @ (u*u)` allocates `J^2 P` — 0.8 GB at
`J = 10`, `P = 1e6`. Write it `(H * (u*u)) @ H.T`: one `(J, P)` temporary,
80 MB, and chunkable. Verified equal to 1.8e-11.

**Eq. (26) does not need `Delta_n`.** The three right-hand sides
`sum_n w_n Delta_n`, `sum_n P_n w_n Delta_n`, `sum_n Q_n w_n Delta_n` factor
through five `(J,)` vectors. With `w_n = P_n v - Q_n u` and
`Delta_n = sum_j alpha_nj H_j`:

```
m_P = P . alpha    m_Q = Q . alpha    m_PP = (P*P) . alpha
m_PQ = (P*Q) . alpha    m_QQ = (Q*Q) . alpha          # each (J,)

sum_n w_n Delta_n       = v (m_P  @ H) - u (m_Q  @ H)
sum_n P_n w_n Delta_n   = v (m_PP @ H) - u (m_PQ @ H)
sum_n Q_n w_n Delta_n   = v (m_PQ @ H) - u (m_QQ @ H)
```

Three `(P,)` fields, no `(N, P)` array anywhere. Verified to 1.4e-14. The
document's cost note already says `Delta_n` need not be stored; this is how.

### 6.3 Sign convention

`vp_aia.md` Eq. (17) has `w_n = P_n v - Q_n u`. `errors._step_field_sensitivity`
has `sf_aia.md` Eq. (T7)'s `w_n = u Q_n - v P_n` — the exact negative. Eq. (29)
is even in `w_n` so the noise map is unaffected, but Eqs. (22), (23) and (26)
are not. Either give the shared helper an explicit sign argument or keep the
two separate; do not reuse it silently.

### 6.4 Validation

`N >= 5` (§"Spatial modes": at `N = 4` no mode is identifiable) and
`J <= floor((N-3)(K-2)/(N-1))` with `K = H*W` (Eq. 15), both `ValueError`
naming the argument and the value. `M + C'C` singular means an undetermined
direction beyond Eq. (24) survived; warn with `cond2` rather than returning
silently, as `aia_diagnostics` warns on `kappa_p`.

### 6.5 Eq. (30) is not Eq. (29)

`spatial_basis` normalizes to `sum_xy H_j H_j' = delta_jj'`, while Eq. (30)'s
asymptotic reading assumes `<H_j H_j'>_xy = delta_jj'` — a factor `K` apart.
Eq. (29) is what gets implemented and is unaffected; only a docstring quoting
the `1 + J/K` rule of thumb would need the conversion.

## 7. Steps

Each step is one reviewable change, run against CPU tests, reported before the
next starts.

1. **`gauge.py`** — `whitening_matrix(u, v, xp) -> (2, 2)`, with the existing
   `whiten_uv` rewritten to call it (no signature change); and
   `normalize_quadrature_frame(a, u, v, P, Q, xp)` applying §"Normalization"
   steps 1-4 in order. Tests: each of the four conditions of Eq. (24) and
   §"Quadrature frame" holds afterwards; idempotent on an already-normalized
   set.
2. **`vp_system.py`** — the chunked pixel-sum pass and the assembly and solve of
   Eqs. (22)-(25). Tests: recovery of a planted `alpha` on a noise-free
   synthetic stack; Eq. (24) satisfied by the solution; chunked equals
   unchunked; `Pi` identities.
3. **`vp_aia.py`** — `aia_variable_projection` wiring steps 1-6, plus
   `VPAIAParam(StepFieldParam)` carrying `kappa_vp`, `P_corr`, `Q_corr`,
   `rms_frac` from `step_field_quality`, and `rounds_run`. Registry entry.
   Tests: end-to-end phase improvement over plain `aia` on the §2 stack;
   `Delta_n` has zero spatial and frame mean; `N < 5` and `J > J_max` raise.
4. **`errors.py`** — `vp_phi_error` for Eq. (29), reusing the chunked second
   pass of `step_field_phi_error`. `VPAIAParam.phi_error` overrides the base
   (decision 2). Tests: agrees with a Monte-Carlo phase variance on a small
   stack to within its own standard error; reduces to `aia.md` Eq. (26) at
   `J = 0`.
5. **Relinearization** (decision 3) — the `rounds` loop of Eq. (28), reusing
   `interference_model.model_stack`. Test: on a stack with `max|Delta| ~ 0.5`
   rad, two rounds beat one.
6. **Docs** — `README.md` method list and `docs/AGENTS.md` status line.

Steps 1-2 are independent of decisions 1-3 and can start once names are agreed.

## 8. Cost

`vp_aia.md` §"cost": `O(NKJ)` for the residual and right-hand sides, `O(KJ^2)`
for Eq. (23), `O((2N+NJ)^3)` for Eq. (25) — about `J + J^2/N` AIA iterations for
the whole first-order pass, against SF-AIA's `refine_iters` full AIA solves.
Storage above AIA: the `(J, P)` basis and matrices of size `2N + NJ`.

## 9. Not covered by `docs/vp_aia.md`

Raise before implementing, do not choose silently:

- fixed `g_n` (decision 1);
- the cross-correlation between the two terms of Eq. (29), which
  §"Noise of the zeroth-order steps" explicitly leaves underived — `phi_error`
  will add them, matching `sf_aia.md` Eq. (E9), and the docstring must say so;
- a convergence criterion for the Eq. (28) loop beyond "stop when the increment
  becomes negligible".
