# Gauge Conventions

Several steps in this pipeline have exact ambiguities — transformations of
the recovered fields that leave the fitted data unchanged. This is a
reference for which convention pins each one down, and where. It documents
existing choices; it does not introduce new ones.

## Shared model (`interference_model.md` Eq. 17, `aia.md`)

| Freedom | Convention | Where |
|---|---|---|
| `(Φ, δ_n) → (Φ−c, δ_n+c)` for any constant `c` | `δ_n[0] = 0`, re-applied every iteration (`aia.md` §"Phase-origin convention"). `Φ` itself still carries an unresolved additive constant — harmless for relative phase maps. | `methods/gauge.py` (`pin_phase_origin`), applied by `methods/aia.py` (`aia`) and `methods/sf_aia.py` (`aia_step_field`) |
| `(Φ, δ) → (−Φ, −δ)` (cosine is even) | **Not** pinned by the solve — two independent solves may land on opposite branches (`aia.md` §"Identifiability and gauge", *Discrete*). Resolved downstream (see Reference/Combine below). | `reference.py` |
| `α_n` (per-frame source-power scale): `α→λα`, `(a,b)→(a,b)/λ` | `median(alpha) = 1` | `solver.py` (`PhaseSolver._alpha_norm`) |
| `g_n` (fringe-contrast scale): only enters as `g_n·b` | `median(g) = 1` (`aia.md` Eq. 16) | `methods/gauge.py` (`normalize_gain`); `frame_contrast.py` (`measure_frame_contrast`) |
| `u, v` sign labeling (not a free gauge — a naming choice) | `u = b·cos(Φ)`, `v = −b·sin(Φ)` ⇒ `Φ = atan2(−v, u)`, `b = hypot(u, v)` | `methods/steps.py` (`pixel_step`), `methods/aia.py` (`aia`); `interference_model.md` Eq. 18 |

## AIA joint-gain fit (`fit_gain=True`)

| Freedom | Convention | Where |
|---|---|---|
| `a(x) + c_n` invariant under `a→a+k`, `c_n→c_n−k` | `mean(c_n) = 0` (`aia.md` Eq. 16) | `methods/gauge.py` (`center_offsets`); `c_n` fitted in `methods/steps.py` (`frame_step`) |
| With `g_n` free, `(u,v)` determined only up to any invertible `M` (`GL(2)`, `aia.md` Eq. 13), not just a rotation | Whiten `(u,v)` so `Σu² = Σv²`, `Σu·v = 0` — collapses `GL(2)` down to `O(2)`, the same residual ambiguity plain (`g≡1`) AIA already has | `methods/gauge.py` (`whiten_uv`); derivation in `aia.md` §"Identifiability and gauge", Eq. 15 |
| `(g_n, δ_n) → (−g_n, δ_n+π)` | `g_n = hypot(P_n, Q_n) ≥ 0` always (`aia.md` Eq. 9) | `methods/steps.py` (`frame_step`) |

## SF-AIA (method `"sf_aia"`)

The step field `Δ_n(x,y) = Σ_j c_jn·p_j(x,y)` is expanded on a basis chosen by
`basis`/`basis_kwargs` (`BASES` lists the families). Both conventions below
hold for every family: `spatial_basis` applies the first to whatever functions
a family supplies, and the second constrains the fitted coefficients.

| Freedom | Convention | Where |
|---|---|---|
| Spatial split of `δ_n(x,y)` into piston `δ_n` + field `Δ_n(x,y)` | Zero spatial mean per basis function, `⟨p_j⟩ = 0`, so `δ_n` is the field average of the step; the built-in `"poly"` family builds its monomials on coordinates centred at the field centroid and scaled to ≈`[-1,1]`, orthonormalized in ascending degree | `basis.py` (`spatial_basis`); `sf_aia.md` §"Conventions" Eq. T1/T3, §"Algorithm" step 2; `interference_model.md` Eq. 9a |
| `c_jn ↔ Φ`: a per-frame-constant basis coefficient is indistinguishable from part of `Φ` | Frame-mean-zero: `c_jn ← c_jn − mean_n(c_jn)`, applied before correcting the data each refine round. **`StepFieldParam.coeffs` is reported un-gauge-fixed** — subtract the frame mean yourself before reading a row as physical per-frame drift. | `methods/gauge.py` (`center_coeffs`), applied by `methods/sf_aia.py` (`aia_step_field`); `sf_aia.md` §"Conventions" Eq. T3b, §"Gauge fixing" Eq. E4; `interference_model.md` Eq. 9b |

## VP-AIA (`vp_aia.md`, not yet implemented)

Theory only: no method is registered for it, so the `Where` column cites
`vp_aia.md` alone. VP-AIA fixes the same phase-step split as SF-AIA, and adds
conventions for the four AIA parametrization freedoms; the normalization steps
below are those of Appendix D §"Normalization", applied both to the
zeroth-order solution and to the corrected fields.

| Freedom | Convention | Where |
|---|---|---|
| A spatially uniform part of `Δ_n` is indistinguishable from the piston `δ_n` | `⟨Δ_n⟩_{x,y} = 0`, the uniform part assigned to `δ_n` | §"Gauge conventions"; `interference_model.md` Eq. 9a |
| A pattern shared by all frames is indistinguishable from the static phase `Φ` | `⟨Δ_n(x,y)⟩_n = 0`, the shared pattern assigned to `Φ`; equivalently `⟨α_nj⟩_n = 0` for the mode amplitudes | §"Gauge conventions", Eq. 6; `interference_model.md` Eq. 9b |
| `(Φ, δ_n) → (Φ−c, δ_n+c)` | `δ_1 = 0`, as a rotation of `(P_n,Q_n)` and `(u,v)` | §"Gauge conventions"; normalization step 3 |
| `g_n·b` scale | `median(g_n) = 1`, with `s = median_n hypot(P_n,Q_n)` — the same convention as `aia.md` Eq. 16 | §"Gauge conventions"; normalization step 4 |
| With `g_n` free, `(u,v) → T(u,v)`, `(P_n,Q_n) → T^{-⊤}(P_n,Q_n)` for invertible `T`, plus the shifts `(P_n,Q_n) → (P_n+p, Q_n+q)`, `a → a−pu−qv` | Shifts fixed by the scalar intercept: `Σ(a−ā)u = Σ(a−ā)v = 0`. Shear and anisotropic scaling fixed by whitening `(u,v)` after every pixel step: `Σu² = Σv²`, `Σu·v = 0`. The remaining rotations, reflections, and common scalings follow from `δ_1 = 0`, the step direction, and `⟨g_n⟩_n = 1`. | §"Quadrature frame"; normalization steps 1–2 |

The shift conditions have no AIA counterpart: `aia.md` §"Identifiability and
gauge" leaves those two directions unpinned, noting the iteration does not
drift along them. Fixing them matters here because shifts, shears, and
anisotropic scalings change the first-order sensitivity `w_n = P_n·v − Q_n·u`
the phase-step fit is built on.

## Carrier removal (`remove_carrier`)

| Freedom | Convention | Where |
|---|---|---|
| Spatial origin for the tilt/curvature/piston split (origin-dependent — differs from SF-AIA's centroid convention above) | Pixel `(0,0)`, unnormalized `x,y` | `carrier.py` |
| Global piston of the output | Weighted circular mean set to zero: `arg(Σ w·e^{iφ}) = 0` | `carrier.py` |
| Carrier frequency `(fx,fy)`, defined only modulo 1 cycle/pixel | FFT-bin peak picks the representative; the refine step tracks the nearest branch to the current estimate | `carrier.py`; `carrier_removal.md` §4 |

## Reference subtraction (`subtract_reference`)

| Freedom | Convention | Where |
|---|---|---|
| `(Φ,δ)→(−Φ,−δ)` sign branch between `phi` and `phi_ref` | Compute both `wrap(phi∓phi_ref)`; keep whichever has lower weighted circular spread. `phi_ref` is the one flipped; `phi` is kept as-is. Ties go to `phi − phi_ref`. | `reference.py` (`subtract_reference`) |

## Multi-acquisition combination (`combine_acquisitions`)

| Freedom | Convention | Where |
|---|---|---|
| Sign branch across `k` acquisitions | Every map reoriented to agree with `phis[reference]` (default index `0`), using `subtract_reference`'s discriminant — run on the *raw* maps, before carrier removal (a stronger, unambiguous discriminant there) | `combine.py` (`combine_acquisitions`) |
| Averaging convention for wrapped angles | Weighted complex mean, `angle(Σ w·e^{iφ} / Σw)` — never an arithmetic mean of wrapped angles | `combine.py` |
