# phase-shift

Phase extraction from phase-shifted interferogram stacks, with interchangeable
algorithms and NumPy/CuPy backends. Extracted from `holoeml-processing`.

## Overview

This repository extracts phase information from stacks of phase-shifted
interferograms — recovering the wrapped phase map, fringe amplitude, and
background from frames where the exact phase-shift steps aren't precisely
known.

## Project structure

- `src/phase_shift/` — phase-processing library.
  - `solver.py` — `PhaseSolver`, the main entry point. Call
    `PhaseSolver(config).fit(stack)` to recover phase.
  - `config.py` — `PhaseConfig`: which algorithm to run and how, with YAML
    save/load.
  - `result.py` — `PhaseResult`: the recovered fields, `solver.result`.
  - `interference_model.py` — `model_stack`, the frame stack of the
    interference model evaluated from its fields.
  - `methods/` — one module per phase-recovery algorithm, registered in
    `methods/__init__.py`'s `METHOD_REGISTRY` (`phase_shift.METHODS` lists
    its names), plus the modules they share.
    - `aia.py` — an Advanced Iterative Algorithm implementation for blind
      phase-shift extraction (Wang & Han 2004; enhanced per Chen & Kemao,
      *Optics Express* 27(26), 37634-37651, 2019).
    - `sf_aia.py` — Spatial-Field AIA, registered as `"sf_aia"` and
      implemented by `aia_step_field`, refining that solve against a
      spatially-varying phase-step error expanded on a basis
      (`basis_kwargs={"degree": 1}`, the default, is a pure linear tilt).
    - `base.py` — `MethodParam`, the diagnostics interface every method
      returns.
    - `steps.py` — `pixel_step`/`frame_step`, the two alternating
      least-squares solves the methods are built from.
    - `gauge.py` — the gauge conventions applied between those steps;
      `docs/gauge_conventions.md` is the reference for all of them.
    - `diagnostics.py` — `AIAParam` and the conditioning/residual
      diagnostics it reports.
  - `basis.py` — `spatial_basis`, `BASES`: the spatial basis families a
    phase-step error field is expanded on, each centered and orthonormalized
    to the conventions of `docs/sf_aia.md`.
  - `errors.py` — the phase-error derivations of `docs/aia.md`
    §"Phase-error covariance" and `docs/sf_aia.md`, shared by the methods'
    `phi_error` maps.
  - `frame_contrast.py` — `measure_frame_contrast`/`measure_frame_visibility`,
    per-frame fringe gain and visibility estimated from the spatial carrier.
  - `carrier.py` — `remove_carrier`, estimating/removing a spatial
    carrier and (optionally) defocus from a wrapped phase map.
  - `reference.py` — `subtract_reference`, resolving the phase sign-branch
    ambiguity between a sample and reference phase map.
  - `combine.py` — `combine_acquisitions`, averaging repeated independent
    acquisitions of the same object.
  - `backend.py` — NumPy/CuPy array-module dispatch shared by all of the
    above (see GPU section below).
  - `utils.py` — `wrap`/`wrap_add`/`wrap_sub` phase wrapping and `format_value`,
    shared helpers.
- `docs/interference_model.md` — the interferometry model (Eq. 17) every
  module in `src/phase_shift/` is written against.
- `docs/aia.md` — AIA and its accuracy analysis; backs `methods/aia.py`,
  `methods/steps.py`, `methods/gauge.py`, and `errors.py`.
- `docs/sf_aia.md` — Spatial-Field AIA (SF-AIA), the method that uses AIA
  to recover spatially varying phase-step errors; backs `methods/sf_aia.py`.
- `docs/vp_aia.md` — Variable-Projection AIA, the same correction in one
  pass; theory only, not yet implemented.
- `docs/gauge_conventions.md` — which convention pins each exact ambiguity
  of the model, and where in the code it is applied.
- `docs/AGENTS.md` — how the theory documents relate and which are settled.
- `tests/` — unit tests (synthetic data; run in seconds).
- `examples/synthetic_stack.py` — a runnable example requiring no experimental data.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/RodionovSA/phase-shift.git
cd phase-shift
uv sync --locked
uv run pytest
uv run python examples/synthetic_stack.py
```

Installs `phase_shift` in editable mode, its runtime dependencies (NumPy and
PyYAML), and the pytest development group into `.venv`. Python 3.12 is selected
by `.python-version`; Python >= 3.12 is supported by the package metadata.
`uv.lock` preserves the source project's versions for retained dependencies.
Use `uv sync --locked --no-dev` for runtime dependencies only. After deliberately
changing dependencies, run `uv lock` and commit both dependency files.

The distribution is named **phase-shift**; Python imports use **phase_shift**.
The `PhaseSolver`, `PhaseConfig`, and `PhaseResult` class names are retained.
This package does not install a `phase` compatibility alias, so it can coexist
with the original `holoeml-processing` package.

## Quick usage

```python
from phase_shift import PhaseSolver, PhaseConfig

# stack: np.ndarray, shape (N, H, W) — N phase-shifted interferogram frames
solver = PhaseSolver(PhaseConfig()).fit(stack)

phi = solver.result.phi   # wrapped phase map, (H, W), in (-pi, pi]
b   = solver.result.b     # fringe modulation amplitude map, (H, W)
a   = solver.result.a     # background intensity map, (H, W)

print(solver.result.reconstruction_error)      # RMSE of the fit, method-agnostic
print(solver.result.method_param)              # diagnostics specific to the method used
```

`PhaseConfig` selects and configures the algorithm (`method="aia"` by
default; see `phase_shift.METHODS` for what's registered) and controls the
shared normalization/gain steps (`use_alpha`, `gain_mode`, `g`) —
`gain_mode="joint"` (the default) fits each frame's fringe contrast jointly
with its phase step inside the method's own iteration, making no
assumption about the fringe pattern's spatial frequency (unlike the older
FFT-based `phase_shift.frame_contrast.measure_frame_contrast`, which needs a linear
spatial carrier and fails on circular or otherwise carrier-free fringes);
`gain_mode="none"` fixes every frame's gain at 1, and passing `g` directly
uses it as a fixed value regardless of `gain_mode`.
`solver.result.method_param` carries whatever diagnostics that method reports —
for `"aia"`, an `AIAParam` with `kappa_p`/`kappa_ps` (condition-number
diagnostics from Chen & Kemao 2019: large values flag a poorly conditioned
acquisition whose result shouldn't be trusted, even if `converged` is
`True`), `predicted_rms` (the paper's predicted phase error in radians),
`iters_run`, `converged`, and (when `gain_mode="joint"`) `g_fit`/`c_fit`/
`g_min_ratio` describing the joint-gain fit. For `"sf_aia"`, a
`StepFieldParam` wrapping that `AIAParam` alongside the step field itself:
`basis`/`basis_kwargs` naming the family it was expanded on, `coeffs`
(shape `(J, N)`, not gauge-fixed — subtract the frame mean before reading a
row as per-frame drift) and `coeffs_rms`, `kappa_fit` flagging a basis that
has outrun what the fringe pattern resolves, and `rms_frac`,
`rms_frac_history`, `best_iter`, `refine_iters_run`, `refine_converged`
describing the refinement loop. Every method also reports a per-pixel
`solver.result.phi_error` map in radians, from `docs/aia.md`
§"Phase-error covariance". See `src/phase_shift/config.py`,
`src/phase_shift/result.py`, `src/phase_shift/methods/aia.py`, and
`src/phase_shift/methods/sf_aia.py` for full parameter/field documentation.

## GPU (CuPy)

The solver and device-selecting processing entry points accept
`device="auto"|"cpu"|"cuda"`. Array-level helpers use their input array's backend.
`"auto"` selects CuPy if it imports successfully, otherwise NumPy. Use
`device="cpu"` explicitly on a machine with CuPy installed but no usable CUDA
device; automatic CUDA runtime recovery is not implemented. Result arrays
stay on the backend that performed the computation. Use
`phase_shift.asnumpy(x)` to move a GPU result to the host before plotting or
saving with NumPy.

To enable it:

```bash
uv sync --locked --extra cuda
```

This installs the optional `cupy-cuda12x` backend retained from the source
project. It requires a compatible NVIDIA GPU, CUDA runtime, and driver. CPU
installation does not require CuPy or CUDA.

## Precision

Every entry point takes `precision=`, alongside `device=`, and
`phase_shift.set_precision(...)` sets the package-wide default for the ones
that are left alone:

```python
import phase_shift as ps

ps.set_precision("double")                       # package-wide default
ps.PhaseSolver(cfg, device="cuda", precision="single").fit(stack)
```

A precision fixes two dtypes. `work` is the dtype of the large `(N, H, W)` /
`(N, P)` arrays and of the recovered fields `phi`, `a`, `b` and `phi_error`.
`accum` is the dtype the operands of a reduction or matrix product over the
full stack are cast to; above `work` it costs a temporary copy of the stack
and buys back the precision a float32 sum over the pixels loses.

| preset | `work` | `accum` | |
|---|---|---|---|
| `"single"` | float32 | float64 | the default |
| `"double"` | float64 | float64 | reference numbers, ~2x the memory |
| `"fast"` | float32 | float32 | lowest memory, least accurate reductions |

Everything else — the per-iteration linear algebra, condition numbers,
normal-equation solves, the `(N,)` vectors `delta`, `g` and `alpha`, and the
polynomial bases — is `float64` whatever the precision. A solver resolves its
precision once, at construction, and records it on
`solver.result.precision`. Numerical conventions and algorithms are retained
from the source.

## Status

Research software; APIs may evolve. Phase unwrapping is outside the current scope.

## License

MIT; the original copyright notice is preserved in [LICENSE](LICENSE).
