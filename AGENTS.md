# Repository guidance

Phase extraction from interferogram stacks using NumPy/CuPy. Use `uv` for
environment management and Python commands; setup: `uv sync --locked`.

## Agree on scope

- Before modifying existing work, ask whether the user wants a small, focused
  change or a broader revision, and whether to preserve the current flow and
  logic. Do not ask again when the request or conversation already establishes
  these preferences.
- Discuss every public API change with the user before implementing it. APIs
  may evolve, but require agreement. If mathematics also changes, settle the
  mathematics first, then the API.
- Keep changes focused, logic concise, and responsibilities clearly split
  across files. Avoid mixing unrelated changes or adding unnecessary abstractions.

## Mathematics and documentation

- Read `docs/AGENTS.md` for the theory reading order and document status.
- Treat `docs/` as the source of truth for mathematics. Read the relevant theory
  before implementing or modifying numerical behavior. If the required math is
  missing or needs changing, discuss it with the user, then create or update the
  theory document before implementation. Surface discrepancies rather than
  silently choosing between code and docs.
- Explain mathematical reasoning and implementation tradeoffs in conversation.
  Optional temporary Markdown notes may hold detailed implementation discussion;
  keep them separate from package documentation.
- Keep docstrings short and focused on current, user-facing behavior. Link to
  theory documents for mathematics. Keep docstrings and package docs free of
  version-history commentary, leftover change notes, internal implementation
  discussion, and incidental numerical justifications.

## Validation

- Run relevant local tests for small changes; run `uv run pytest` for significant
  changes. Add or update tests as needed to validate changed behavior.
- Development usually happens on a Mac; do not run GPU tests there. Real GPU and
  measurement validation happens later on the NVIDIA Quadro 2200 machine.
  Report what was tested and what still requires that machine.

Keep this file compact and repository-wide; place specialized rules in folder
`AGENTS.md` files and detailed theory in `docs/`.
