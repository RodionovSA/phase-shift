# Theory reading guide

## Main scope

- `interference_model.md`: established physical model and shared definitions.
  Read first for mathematical work.
- `aia.md`: established, standalone AIA method built on that model. Read for
  AIA work and before studying the method below.
- `aia_noise.md`: phase-error covariance of the AIA solution, split from
  `aia.md`. Read for noise and error-map work.
- `vp_aia.md`: Variable-Projection AIA (VP-AIA), the user's own method, which
  recovers spatially varying phase-step errors as a first-order correction to
  AIA in one pass. This theory may still be revised through discussion with the
  user.

The first two documents are settled foundations. Do not revise their mathematics
without explicit agreement; discuss proposed VP-AIA changes before updating its
theory and implementation. VP-AIA is implemented (`methods/vp_aia.py` with
`methods/vp_system.py`), so revising its theory means revising the code that
follows it.

## Reading rules

- Read only the main documents and sections relevant to the task, including the
  definitions, assumptions, and referenced derivations needed to interpret them.
- Other documents are supplementary, outside the established main scope.
  `carrier_removal.md` is agent-authored and has not been fully reviewed by the
  user; do not treat it as approved theory.
- If documents disagree or required theory is missing, raise the issue with the
  user before implementing a mathematical choice. Do not silently reconcile it.

## Writing and editing documents

- Split every new document or update into steps. Write one step, then wait for
  the user's approval before the next.
- Check math (algebra, small numerical test) before writing it.
- Publication-ready text: no remarks about earlier versions.
- Use the document's notation; define every new symbol.
- Concise, strict, elementary math: explicit matrices and sums, normal
  equations. No Kronecker, KKT, or abstract operator notation.
- Explain the central idea fully; keep the rest short.
- Logical flow: model, conventions, general analysis, method. Justify
  assumptions before using them; refer back instead of repeating.
- Keep equation numbering and references consistent; cite borrowed methods.
- Long explanations for the user go to `notes/`.
