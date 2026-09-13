# Extraction from holoeml-processing

Source: [RodionovSA/holoeml-processing](https://github.com/RodionovSA/holoeml-processing)
at commit `e67d89f1e61ef716e362359a5672cf3d5c0f33db` on `main`.

This is an independent package snapshot. No source-repository files or GitHub
references are changed by the extraction. Full Git history is not included;
the source revision and original MIT copyright are retained for provenance.

## Included

| Source | Destination |
| --- | --- |
| `phase/` | `src/phase_shift/` |
| `tests/` | `tests/`, with updated imports |
| `docs/*.md` | `docs/`, with updated module/path references |
| `LICENSE`, `.python-version` | Preserved |
| `pyproject.toml`, `uv.lock` | Adapted for the independent distribution |

All existing phase modules are retained, including AIA, the polynomial
step-field extension and its `aia_tilt` alias, backend dispatch, phase
configuration/results, carrier removal, reference subtraction, repeated
acquisition combination, ripple correction, and frame statistics. Numerical
implementations and method names are preserved. Documentation references and
Python import names are updated; no `phase` compatibility alias is installed.

`PhaseSolver`, `PhaseConfig`, and `PhaseResult` remain the public class names.
Existing code can use the standalone library by replacing `from phase ...`
with `from phase_shift ...`. That migration has not been applied to
`holoeml-processing`.

## Dependency scope

Runtime dependencies are NumPy and PyYAML. CuPy remains the optional `cuda`
extra, and pytest remains in the `dev` group. Minimum versions are retained
from the source project. The lockfile retains source versions and artifact
hashes for the remaining dependency graph, removing unused packages.

Torch, OpenCV, Jupyter, Matplotlib, and nbstripout are excluded because the
extracted library and synthetic tests do not use them. Exploratory acquisition
notebooks and their local data/configuration, the entry-point stub, and editor
workspace state remain in the original project. The standalone example uses
synthetic data and needs no plotting or notebook dependencies.

## Verification

The inherited synthetic suite exercises the methods and phase utilities.
The new synthetic example demonstrates phase recovery with sign and piston
alignment for comparison to known truth. The numerical code is checked
against the source after removing docstrings to confirm the extraction does
not alter algorithms. Actual validation results are recorded in the
[validation report](validation.md); GPU and real-acquisition validation require
suitable hardware/data.
