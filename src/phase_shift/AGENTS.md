# Code guidance

Rules for the `phase_shift` package. The root `AGENTS.md` still applies.

## Coding steps

1. Read the relevant section in `docs/`, the ground truth for all mathematics
   (see `docs/AGENTS.md`); note the equations the change implements. Code
   follows the docs, never the reverse.
2. Read the module being changed and its callers (`solver.py`,
   `methods/__init__.py`, `errors.py`) before editing.
3. State the plan as a sequence of small steps: files touched, equations used,
   effect on public API. Wait for agreement on the plan.
4. Implement one step at a time, each small enough to review quickly (e.g. one
   function or one module move). Reuse existing helpers instead of duplicating
   logic.
5. After each step, summarize what changed and how it maps to the docs, run
   relevant CPU tests, and report what passed and what still needs the GPU
   machine. Wait for the user's approval before the next step.

## Structure

- Mirror the logical structure of the docs where possible: modules, function
  order, and names follow the sections and steps they implement.
- Keep files short and focused on one topic: one method per module in
  `methods/`, and separate modules for shared topics (e.g. gauge fixing, error
  computation, polynomial bases).
- Code used by more than one method, or not specific to any method, goes into a
  shared module, never into one method's file for others to import.
- Structure never costs efficiency: splitting code must not add array copies,
  device transfers, repeated passes over the stack, or lost caching.
- Imports point one way: `solver` -> `methods` -> shared modules, `backend`.
  Method modules never import `solver` or each other's private helpers.
- A new method is a function
  `(stack, g, fit_gain=False, dtype=None, precise_reduce=True, **kwargs)
  -> (a, b, phi, delta, g, method_param)`, a `MethodParam` subclass for its
  diagnostics, and one `METHOD_REGISTRY` entry. `solver.py` stays unchanged.
- Re-export public names in `__init__.py`; prefix everything else with `_`.
- No compatibility aliases, shims, or dead code paths unless the user asks.

## Backend and numerics

- Only `backend.py` imports CuPy. Array functions start with
  `xp = get_array_module(...)` and use `xp.` throughout. Call `to_device` only
  at public entry points; return results on the device they were computed on.
- Everything must run with NumPy alone; no CuPy-only code without a NumPy path.
- Large `(N, H, W)` / `(N, P)` arrays use the working dtype (`default_dtype`,
  float32). Small linear algebra, `(N,)` vectors, and scalar reductions use
  float64.
- Avoid extra full-size temporaries: prefer scalar sums or chunked reductions.
- Minimize host syncs (`float(...)`, `asnumpy`): at most a few per iteration,
  never per pixel or per frame inside a loop.
- Combine wrapped phases with `wrap`, `wrap_add`, `wrap_sub`, not raw arithmetic.

## Style

- Every `.py` file starts with its path from the repository root as a comment,
  then a short module docstring stating the file's purpose:
  ```python
  # src/phase_shift/methods/aia.py
  """Advanced Iterative Algorithm (AIA) phase extraction."""
  ```
- Names follow theory notation: `phi`, `delta`, `g`, `alpha`, `a`, `b`, `u`,
  `v`; dimensions `N, H, W, P`; matrices uppercase as in the docs.
- Type hints on all function arguments and return values, private functions
  included (e.g. `x: np.ndarray`, `-> np.ndarray`); dataclasses for configs,
  results, and diagnostics.
- Write public, general-purpose code: docstrings and comments describe current
  behavior for users, not how the code came to be.
- Keep docstrings small, NumPy-style: one-line summary, then `Parameters` /
  `Returns` / `Raises` with shapes and units. Refer to the docs for mathematics
  and rationale, e.g. ``docs/aia.md`` Eq. (22) or §"Pixel step", instead of
  explaining it inline.
- Never leave comments or docstring text about design decisions, alternatives
  tried, previous versions, or numerical verifications.
- Keep comments short and rare. Add trailing shape comments (`# (N, 3)`) on
  non-obvious array lines.
- Validate public inputs early with `ValueError` naming the argument and the
  received value. Report poor numerical quality with
  `warnings.warn("<method>: ...", stacklevel=2)`, never `print`.
- Match surrounding formatting; keep lines under ~100 characters.

## Tests

- Fast synthetic tests in `tests/`; never depend on files in `data/`.
- Seed every RNG, compare against known ground truth with wrap-aware errors and
  explicit tolerances.
- Guard GPU-specific checks with `CUPY_AVAILABLE` so they skip on the Mac.
