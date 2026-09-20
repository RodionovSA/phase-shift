# src/phase_shift/__init__.py
"""Phase processing: interferogram stack -> wrapped phase -> object phase.

Public API, re-exported here regardless of which submodule a name lives in:

- :mod:`phase_shift.solver`, :mod:`phase_shift.config`,
  :mod:`phase_shift.result` -- ``PhaseSolver``, ``PhaseConfig``,
  ``PhaseResult``: the primary entry point. Configure a ``PhaseSolver`` with a
  ``PhaseConfig`` (which algorithm to run and how) and call ``.fit(stack)`` to
  recover phase; see :class:`phase_shift.result.PhaseResult` for the recovered
  fields.
- :mod:`phase_shift.methods` -- ``METHODS``, ``MethodParam``, and the building
  blocks the methods share: the least-squares steps, the gauge conventions,
  and the accuracy diagnostics.
- :mod:`phase_shift.basis` -- ``spatial_basis``, ``BASES``: the basis families
  a phase-step error field is expanded on.
- :mod:`phase_shift.interference_model` -- ``model_stack``
- :mod:`phase_shift.frame_contrast` -- ``measure_frame_contrast``,
  ``measure_frame_visibility``
- :mod:`phase_shift.carrier` -- ``remove_carrier``, ``CarrierResult``
- :mod:`phase_shift.reference` -- ``subtract_reference``, ``DifferenceResult``
- :mod:`phase_shift.combine` -- ``combine_acquisitions``, ``CombinedResult``
- :mod:`phase_shift.backend` -- NumPy/CuPy array-module dispatch shared by all
  of the above; every function accepts a ``device="auto"|"cpu"|"cuda"``
  argument and returns result arrays on whichever device it ran on. Use
  ``phase_shift.backend.asnumpy`` to bring a result field back to the host
  explicitly.
"""

from . import backend
from .backend import asnumpy
from .basis import BASES, spatial_basis
from .carrier import CarrierResult, remove_carrier
from .combine import CombinedResult, combine_acquisitions
from .config import PhaseConfig
from .frame_contrast import measure_frame_contrast, measure_frame_visibility
from .interference_model import model_stack
from .methods import METHODS, MethodParam
from .reference import DifferenceResult, subtract_reference
from .result import PhaseResult
from .solver import PhaseSolver

__all__ = [
    "PhaseSolver",
    "PhaseConfig",
    "PhaseResult",
    "MethodParam",
    "METHODS",
    "spatial_basis",
    "BASES",
    "model_stack",
    "measure_frame_contrast",
    "measure_frame_visibility",
    "CarrierResult",
    "remove_carrier",
    "DifferenceResult",
    "subtract_reference",
    "CombinedResult",
    "combine_acquisitions",
    "backend",
    "asnumpy",
]
