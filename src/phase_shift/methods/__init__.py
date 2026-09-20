# src/phase_shift/methods/__init__.py
"""Registry of phase-recovery methods for :class:`phase_shift.solver.PhaseSolver`.

Each entry maps a name to a callable
``(stack, g, fit_gain=False, dtype=None, precise_reduce=True, **method_kwargs)
-> (a, b, phi, delta, g, method_param)``, the fields of
``docs/interference_model.md`` Eq. (17). The returned ``g`` is the input passed
through unchanged when ``fit_gain`` is False, else the gain fitted alongside
``delta`` with ``median(g) = 1``.

Add a method as its own module here plus one entry below; :data:`METHODS`
follows from the keys and nothing else changes. The building blocks the
methods share are re-exported here too: the alternating least-squares steps
(:mod:`phase_shift.methods.steps`), the gauge conventions
(:mod:`phase_shift.methods.gauge`), and the accuracy diagnostics
(:mod:`phase_shift.methods.diagnostics`).
"""

from .aia import aia
from .base import MethodParam
from .diagnostics import AIAParam, aia_diagnostics, chunked_sigma, cond2
from .gauge import (center_coeffs, center_offsets, normalize_gain, pin_phase_origin,
                    whiten_uv)
from .sf_aia import StepFieldParam, aia_step_field, fit_step_field, step_field_quality
from .steps import frame_step, pixel_design, pixel_step

METHOD_REGISTRY = {
    "aia": aia,
    "sf_aia": aia_step_field,
}

METHODS = list(METHOD_REGISTRY)

__all__ = [
    "METHOD_REGISTRY",
    "METHODS",
    "MethodParam",
    "aia",
    "aia_step_field",
    "AIAParam",
    "StepFieldParam",
    "pixel_step",
    "frame_step",
    "pixel_design",
    "fit_step_field",
    "step_field_quality",
    "whiten_uv",
    "pin_phase_origin",
    "normalize_gain",
    "center_offsets",
    "center_coeffs",
    "aia_diagnostics",
    "chunked_sigma",
    "cond2",
]
