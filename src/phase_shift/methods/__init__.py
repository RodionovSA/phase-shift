# src/phase_shift/methods/__init__.py
"""Registry of phase-recovery methods for :class:`phase_shift.solver.PhaseSolver`.

Each entry maps a name to a callable
``(stack, g, fit_gain=False, precision=None, **method_kwargs)
-> (a, b, phi, delta, g, method_param)``, the fields of
``docs/interference_model.md`` Eq. (17). The returned ``g`` is the input passed
through unchanged when ``fit_gain`` is False, else the gain fitted alongside
``delta`` with ``median(g) = 1``.

Add a method as its own module here plus one entry below; :data:`METHODS`
follows from the keys and nothing else changes. The building blocks the
methods share are re-exported here too: the alternating least-squares steps
(:mod:`phase_shift.methods.steps`), the gauge conventions
(:mod:`phase_shift.methods.gauge`), the accuracy diagnostics
(:mod:`phase_shift.methods.diagnostics`), and the spatially varying phase step
(:mod:`phase_shift.methods.step_field`).
"""

from .aia import aia
from .base import MethodParam
from .diagnostics import AIAParam, aia_diagnostics, chunked_sigma, cond2
from .gauge import (center_coeffs, center_offsets, normalize_gain,
                    normalize_quadrature_frame, pin_phase_origin, whiten_uv,
                    whitening_matrix)
from .sf_aia import SFAIAParam, aia_step_field, fit_step_field
from .step_field import StepFieldParam, step_field_quality
from .vp_aia import VPAIAParam, aia_variable_projection
from .steps import frame_step, pixel_design, pixel_step
from .vp_aia import VPAIAParam, aia_variable_projection
from .vp_system import VPSolution, fit_frame_and_coeffs, max_modes

METHOD_REGISTRY = {
    "aia": aia,
    "sf_aia": aia_step_field,
    "vp_aia": aia_variable_projection,
}

METHODS = list(METHOD_REGISTRY)

__all__ = [
    "METHOD_REGISTRY",
    "METHODS",
    "MethodParam",
    "aia",
    "aia_step_field",
    "aia_variable_projection",
    "AIAParam",
    "StepFieldParam",
    "SFAIAParam",
    "VPAIAParam",
    "VPSolution",
    "pixel_step",
    "frame_step",
    "pixel_design",
    "fit_step_field",
    "step_field_quality",
    "fit_frame_and_coeffs",
    "max_modes",
    "whiten_uv",
    "whitening_matrix",
    "normalize_quadrature_frame",
    "pin_phase_origin",
    "normalize_gain",
    "center_offsets",
    "center_coeffs",
    "aia_diagnostics",
    "chunked_sigma",
    "cond2",
]
