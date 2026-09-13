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
  - `solver.py` — `PhaseSolver`/`PhaseConfig`/`PhaseResult`: the main entry
    point. Configure a `PhaseConfig` (which algorithm to run and how) and
    call `PhaseSolver(config).fit(stack)` to recover phase.
  - `methods/` — one module per phase-recovery algorithm, registered in
    `methods/__init__.py`'s `METHOD_REGISTRY` (`phase_shift.solver.METHODS` is
    derived from it). `aia.py` — an Advanced Iterative Algorithm
    implementation for blind phase-shift extraction (Wang & Han 2004;
    enhanced per Chen & Kemao, *Optics Express* 27(26), 37634-37651, 2019).
    `step_field.py` — `aia_step_field`, refining that solve against an
    arbitrary-degree spatially-varying phase-step error (`degree=1` is a
    pure linear tilt, registered separately as `"aia_tilt"`).
  - `utils.py` — `measure_frame_contrast`/`measure_frame_visibility`,
    per-frame fringe-gain estimation shared by every method.
  - `carrier.py` — `remove_carrier`, estimating/removing a spatial
    carrier and (optionally) defocus from a wrapped phase map.
  - `reference.py` — `subtract_reference`, resolving the phase sign-branch
    ambiguity between a sample and reference phase map.
  - `combine.py` — `combine_acquisitions`, averaging repeated independent
    acquisitions of the same object.
  - `ripple.py` — `estimate_phase_ripple` / `apply_phase_ripple`, correcting
    a phase-locked error `eps(phi)`.
  - `backend.py` — NumPy/CuPy array-module dispatch shared by all of the
    above (see GPU section below).
- `docs/interference_model.md` — the interferometry model (Eq. 8) every
  module in `src/phase_shift/` is written against.
- `docs/frame_moments.md`, `docs/step_field_residuals.md` — derivations of how
  non-ideal phase steps and per-frame contrast leak into frame statistics and
  into the AIA solve, respectively; the latter backs `methods/step_field.py`.
- `tests/` — unit tests (synthetic data; run in seconds).
- `examples/synthetic_stack.py` — a runnable example requiring no experimental data.
- [Documentation index](docs/README.md) — theory, development, and extraction notes.

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

phi = solver.phi_   # wrapped phase map, (H, W), in (-pi, pi]
b   = solver.b_     # fringe modulation amplitude map, (H, W)
a   = solver.a_     # background intensity map, (H, W)

print(solver.reconstruction_error_)      # RMSE of the fit, method-agnostic
print(solver.method_param_)              # diagnostics specific to the method used
```

`PhaseConfig` selects and configures the algorithm (`method="aia"` by
default; see `phase_shift.solver.METHODS` for what's registered) and controls the
shared normalization/gain steps (`use_alpha`, `gain_mode`, `g`) —
`gain_mode="joint"` (the default) fits each frame's fringe contrast jointly
with its phase step inside the method's own iteration, making no
assumption about the fringe pattern's spatial frequency (unlike the older
FFT-based `phase_shift.utils.measure_frame_contrast`, which needs a linear
spatial carrier and fails on circular or otherwise carrier-free fringes);
`gain_mode="none"` fixes every frame's gain at 1, and passing `g` directly
uses it as a fixed value regardless of `gain_mode`.
`solver.method_param_` carries whatever diagnostics that method reports —
for `"aia"`, an `AIAParam` with `kappa_p`/`kappa_ps` (condition-number
diagnostics from Chen & Kemao 2019: large values flag a poorly conditioned
acquisition whose result shouldn't be trusted, even if `converged` is
`True`), `predicted_rms` (the paper's predicted phase error in radians),
`iters_run`, `converged`, and (when `gain_mode="joint"`) `g_fit`/`c_fit`/
`g_min_ratio` describing the joint-gain fit. See `src/phase_shift/solver.py` and
`src/phase_shift/methods/aia.py` for full parameter/field documentation.

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

The large `(N, P)`-shaped arrays default to `float32` (`dtype=` on
`PhaseSolver`/`measure_frame_contrast` overrides this); the small
per-iteration linear algebra (condition numbers, normal-equation solves,
reductions) use `float64` by default; `precise_reduce=False` enables the
existing faster reduction option. Numerical conventions and algorithms are
retained from the source; see
[extraction notes](docs/extraction.md) for scope and verification details.

## Status

Research software; APIs may evolve. Phase unwrapping is outside the current scope.

## License

MIT; the original copyright notice is preserved in [LICENSE](LICENSE).
