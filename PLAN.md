# Restructuring plan: `src/phase_shift/methods`

Bring `methods/` to `src/phase_shift/AGENTS.md`: one method per module, shared
topics in their own modules, module and function structure mirroring the docs.

## Ground rules

- Every step preserves results **bit-identically**. Each step is verified by
  comparing against the `git show HEAD:` version of the touched modules on
  synthetic stacks, field by field, plus `pytest tests/`.
- One step at a time; summary and test report after each, then wait for
  approval before the next.
- Mathematics comes from `docs/`. No equation is changed by this plan; only
  where code lives, what it is called, and how it is documented.

## Decisions taken

1. **Error model moves onto `MethodParam`.** Each method owns its phase-error
   model as a `MethodParam` method, mirroring `phase_step_field`. `errors.py`
   keeps the shared derivations; the method-name dispatch goes away.
2. **`"sf_aia"` replaces `"aia_step_field"`; the `"aia_tilt"` alias is dropped.**
   Registry keys become `"aia"` and `"sf_aia"`, matching `docs/sf_aia.md` and
   the module name. Configs naming the old keys must be updated.
3. **VP-AIA is not implemented here.** `docs/vp_aia.md` gets its own plan
   afterwards, built on the shared modules this plan creates.
4. **Building blocks stay public** and are re-exported from
   `phase_shift.methods`; `spatial_basis` is re-exported from the package root.
5. **The spatial basis is a family, not a function.** `basis.py` holds a
   registry of basis families with the gauge conditions applied in one place,
   so a new family is one registered builder and no change to any method.

## Target layout

| Module | Contents | Doc |
|---|---|---|
| `methods/base.py` | `MethodParam`: `print_summary`, `phase_step_field`, `phi_error` | `interference_model.md` Eq. (17)/(20) |
| `methods/steps.py` | `pixel_step`, `frame_step`, `_pixel_design` | `aia.md` §"Pixel step", §"Frame step" |
| `methods/gauge.py` | `whiten_uv`, `pin_phase_origin`, `normalize_gain`, `center_offsets`, `center_coeffs` | `gauge_conventions.md`; `sf_aia.md` Eq. (E4); `vp_aia.md` §"Gauge conventions" |
| `methods/diagnostics.py` | `AIAParam`, `aia_diagnostics`, `chunked_sigma`, `cond2` | `aia.md` §"Accuracy diagnostics" |
| `methods/aia.py` | `aia` | `aia.md` |
| `methods/sf_aia.py` | `aia_step_field`, `fit_step_field`, `step_field_quality`, `StepFieldParam` | `sf_aia.md` |
| `basis.py` (root) | `spatial_basis`, `BASIS_REGISTRY`, `BASES`, one builder per family | `interference_model.md` Eq. (9a)/(9b); `sf_aia.md` §1.2(i), Eq. (T1)/(T3); `vp_aia.md` Eq. (13) |
| `errors.py` (root) | shared phase-error derivations, no dispatch | `aia.md` Eq. (22)/(34)/(38); `sf_aia.md` Eq. (E9) |

Import direction stays one-way: `solver` → `methods` → shared modules →
`backend`. No method module imports another method module.

## Why this split

`sf_aia.py` currently imports `AIAParam`, `_aia_diagnostics`, `_whiten_uv`,
`aia_pixel_step` and `aia_frame_step` from `aia.py`, and `errors.py` imports
`_poly_basis` from `sf_aia.py` — both banned by AGENTS §Structure, which names
"gauge fixing, error computation, polynomial bases" as the shared topics it
expects. `docs/vp_aia.md` needs the same five: its algorithm step 1 is the
pixel step, whitening and frame step with normalizations 1–4, and its Eq. (13)
spatial modes carry SF-AIA's zero-spatial-mean/zero-frame-mean conventions.

## Steps

### 1. `methods/diagnostics.py`

Move `AIAParam`, `_aia_diagnostics` → `aia_diagnostics`, `_chunked_sigma` →
`chunked_sigma`. Replace `_cond3` (aia) and `_cond_batch` (sf_aia) with one
`cond2` handling both a single matrix and a batch. `aia.py` and `sf_aia.py`
import from here. Pure move; no caller-visible change.

### 2. `methods/gauge.py`

Move `_whiten_uv` → `whiten_uv` (`aia.md` §"A gauge freedom that only appears
once g_n is free"). Add the one-line conventions currently inlined and
duplicated in `aia()` and `aia_step_field()`: `pin_phase_origin`
(`delta - delta[0]`), `normalize_gain` (`median(g) = 1`), `center_offsets`
(`mean(c) = 0`), `center_coeffs` (frame-mean-zero, Eq. T3b/E4). Each call site
becomes a named call; every expression is carried over unchanged.

### 3. `basis.py` (package root)

The spatial basis becomes a family of bases behind one entry point, so adding
a family is a registered builder rather than an edit to any method. Root
placement because `errors.py` needs it now and `carrier_removal.md` §1.3
specifies the same basis for carrier removal.

- `spatial_basis(H, W, kind="poly", xp=np, **kwargs) -> (J, P) float64`, the
  single entry point, cached per argument set as `_poly_basis` is today.
  It validates `kind`, dispatches to the family builder, and then applies the
  conventions every estimator relies on, in one place: subtract each row's
  spatial mean (`interference_model.md` Eq. 9a, `sf_aia.md` Eq. T3) and
  orthonormalize the rows in order.
- `BASIS_REGISTRY = {"poly": _poly_terms}` and `BASES = list(BASIS_REGISTRY)`,
  mirroring `METHOD_REGISTRY`/`METHODS`.
- A family builder returns only its raw, un-centered rows, built from the
  shared centered, unit-scaled coordinates of `sf_aia.md` §1.2(i).
  `_poly_terms(degree)` is today's monomials of total degree 1 through `M`;
  `degree=0` keeps meaning the empty basis, i.e. the piston model.
- The frame-mean condition (Eq. 9b, T3b) stays in `methods/gauge.py`, since it
  constrains the fitted coefficients rather than the basis.

Selection travels as `basis: str` plus `basis_kwargs: dict`, the same pair as
`PhaseConfig.method`/`method_kwargs`, so a configuration stays YAML-safe
through `PhaseConfig.to_yaml` — a typed spec object would not survive
`yaml.safe_dump`.

*API effect:* `aia_step_field(degree=1)` becomes
`aia_step_field(basis="poly", basis_kwargs={"degree": 1})`;
`StepFieldParam.degree` becomes `basis`/`basis_kwargs`, which is what
`phase_step_field` and the error model rebuild the basis from.
`tests/test_phase.py` imports `spatial_basis` instead of `_poly_basis`.

### 4. `methods/steps.py`

Move `aia_pixel_step` → `pixel_step` and `aia_frame_step` → `frame_step`, with
`_pixel_design`. Rewrite the four validation messages ("Stack shape must be
have 2 dims") to name the argument and the received value.

### 5. `methods/aia.py`

Left holding `aia()` alone. Path header; `np.ndarray | None` and `DTypeLike`
instead of `Optional`/untyped `dtype`; return type hint; docstrings trimmed to
NumPy style with the derivations cited from `aia.md` instead of restated;
`phase.*` → `phase_shift.*`; validation of `stack.ndim`, `iters`, `tol`. The
per-iteration `I - c_fit[:, None]` full `(N, P)` temporary is reviewed against
AGENTS §"Backend and numerics" and either justified or removed.

### 6. `methods/sf_aia.py`

Same style pass. `step_field_quality` stops returning the `(N, P)` `resid`
array its only caller discards. `StepFieldParam` gets `list[float]` and
`DTypeLike` field types, and carries `basis`/`basis_kwargs` from step 3.
`fit_step_field` and `step_field_quality` keep taking a prebuilt `(J, P)`
basis array, so they stay family-agnostic. Validation messages rewritten as in
step 4.

### 7. Error model onto `MethodParam`

`MethodParam.phi_error(...)` returns `None` by default; `AIAParam` and
`StepFieldParam` override it. `errors.py` keeps `_aia_phi_error_parts` and
`_aia_step_field_phi_error` as shared derivations — `StepFieldParam` reuses the
AIA parts through them, not through `methods/aia.py` — and loses
`compute_phi_error`'s name dispatch. `PhaseSolver._phi_error` calls
`method_param.phi_error(...)`. The step-field leverage term (`sf_aia.md`
Eq. E7/E9) takes the basis from `StepFieldParam`'s `basis`/`basis_kwargs`
instead of rebuilding it from `degree`.

*API effect:* `compute_phi_error` is removed; a new method with an error model
now needs no edit outside its own module.

### 8. Registry rename

`METHOD_REGISTRY` becomes `{"aia": aia, "sf_aia": aia_step_field}`. Update
`config.py`'s docstring, `tests/test_phase.py`, and the README.

*API effect:* `PhaseConfig(method="aia_step_field")` and `"aia_tilt"` stop
working; `"sf_aia"` replaces both, with `degree=1` still the default.

### 9. `methods/__init__.py` and root exports

Re-export `pixel_step`, `frame_step`, `fit_step_field`, `step_field_quality`,
`whiten_uv`, `aia_diagnostics`, `AIAParam`, `StepFieldParam`, `MethodParam`,
`METHODS`, `METHOD_REGISTRY` from `phase_shift.methods`; add `spatial_basis`
and `BASES` to `phase_shift/__init__.py`. Everything not listed gets a `_`
prefix.

## Open question: two sign discrepancies in `errors.py`

Found while moving the error model in step 7, and **left as the code has
them** — `docs/AGENTS.md` says to raise a code/doc disagreement rather than
reconcile it. Both affect only ``phi_error_simplified=False``, so the default
output is unchanged.

1. **Stage-2/3 correction.** `aia.md` Eq. (38)/(41) make the correction a
   *reduction* of ``sigma_Phi^2``, and the text warns that treating the step
   error as independent of the pixel noise "gives the right size but the wrong
   sign". `aia_phi_error_parts` returns ``phi_var + correction``. Measured in
   the ideal configuration of those equations, the code's relative change is
   ``+3.66e-04`` where Eq. (38) predicts ``-3.66e-04`` (Stage 2), and
   ``+4.88e-04`` where Eq. (41) predicts ``-4.88e-04`` (Stage 3): the
   magnitudes agree to 1.00, only the sign differs.
2. **Step-field term.** `sf_aia.md` §"Noise of the corrected solve" states the
   correction "cannot reduce the noise; it adds a variance term" (Eq. E7), and
   Eq. (E8) gives ``1 + J/(4*N_p)``. `step_field_phi_error` computes a
   leverage *discount* and returns ``phi_var - discount``.

Both look like the docs were revised without the code following: the equation
numbers throughout `errors.py` referred to an older numbering of `aia.md`
(Eq. 22 -> 26, 28 -> 33, 29 -> 34, 33 -> 37, 35 -> 39), and `sf_aia.md`'s
Eq. (E5)/(E9) and §9, and `aia.md` §"Direct phase-error computation", no
longer exist. The citations are corrected; the arithmetic is not.

## Deferred

- **VP-AIA** (`docs/vp_aia.md`): its own plan, after this one. Its Eq. (13)
  modes `H_j` are any fixed, linearly independent spatial functions, so it
  takes `basis`/`basis_kwargs` as SF-AIA does.
- **Further basis families** (Zernike, Fourier, spline, …): one registered
  builder each, no change to `spatial_basis` or the methods. The docs
  currently prescribe a *polynomial* basis (`interference_model.md` Eq. 9b,
  `sf_aia.md` Eq. T1), so per `docs/AGENTS.md` a new family needs the theory
  updated first — the module makes that a doc question, not a code one.
- **Carrier removal joint fit** (`docs/carrier_removal.md` §2–3, Eq. C1–C9):
  unapproved theory; `carrier.py` keeps today's algorithm.
- **Fixed working dtype** across the package: planned separately by the user;
  `utils._estimation_weight` is the single place the weight dtype is decided.
- **Stale doc references to `ripple.py`**: `README.md` and
  `gauge_conventions.md` §"Phase ripple" still describe the removed module.
