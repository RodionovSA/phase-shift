# Extraction validation

Validation performed during extraction from source commit
`e67d89f1e61ef716e362359a5672cf3d5c0f33db`.

| Check | Result |
| --- | --- |
| Algorithm preservation | All 12 library modules have identical Python ASTs after removing docstrings. |
| Python compilation | All extracted Python files compile. |
| Documentation | Local Markdown links resolve. |
| Lock consistency | `uv lock --offline --check` passes, resolving 11 packages including the project and optional/development dependencies. |
| Inherited CPU checks | 19 existing test methods passed when invoked directly with their original assertions. |
| Remaining inherited checks | 11 methods requiring pytest helpers or fixtures were not run. |
| Synthetic example | Reconstruction RMSE `1.59955e-07`; sign/piston-aligned phase RMS `9.67331e-08` rad. |
| Source repository | `main` was rechecked and still points to the source commit. No GitHub writes were made to it. |

## Environment and limits

The CPU checks used the preinstalled Python 3.12.14, NumPy 2.3.5, and PyYAML
6.0.3. This is not the locked dependency environment: the package retains the
source minimum NumPy version of 2.5.1. These checks therefore confirm useful
extraction behavior but do not establish full supported-environment validation.

`uv sync --locked` was attempted but the NumPy wheel download timed out under
the workspace's network restrictions. Pytest and Hatchling were unavailable,
so the complete pytest suite, editable installation, and wheel/sdist build
could not be verified. No test assertions or tolerances were relaxed, and no
replacement pytest implementation was used. GPU execution and real experimental
data were not tested.

On a machine with package-index access, complete verification with:

```bash
uv sync --locked
uv run pytest
uv run python examples/synthetic_stack.py
uv build
```
