# Documentation

Start with the [package README](../README.md) for installation and usage.

## Theory and conventions

- [Interference model](interference_model.md): the per-frame intensity model.
- [Gauge conventions](gauge_conventions.md): phase sign, piston, and gain conventions.
- [Advanced Iterative Algorithm](aia.md): the baseline reconstruction method.
- [Step-field residuals](step_field_residuals.md): spatially varying phase-step errors and their refinement.
- [Frame moments](frame_moments.md): how phase sampling and contrast affect stack statistics.

## Development

- [Extraction notes](extraction.md): source revision, package boundaries, and verification.
- [Synthetic example](../examples/synthetic_stack.py): a complete CPU reconstruction.

Run `uv sync --locked` and `uv run pytest` from the repository root. The inherited
suite primarily exercises CPU behavior; CUDA execution needs separate validation
on suitable hardware. Read the interference model and
the relevant derivation before changing an algorithm. Read gauge conventions
before changing phase offsets, sign selection, or gain normalization.

To add a method, implement the callable contract described in
`src/phase_shift/methods/__init__.py` and add it to `METHOD_REGISTRY`.
`phase_shift.solver.METHODS` is derived from that registry.

Use backend-dispatched array operations for CPU/GPU calculations. Preserve the
existing distinction between large float32 arrays and small float64 numerical
solves, and use wrapped-angle helpers for phase arithmetic.
