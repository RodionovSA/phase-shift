"""Phase processing: interferogram stack -> wrapped phase -> object phase.

Public API, re-exported here regardless of which submodule a name lives in:

- :mod:`phase.solver`, :mod:`phase.config`, :mod:`phase.result` --
  ``PhaseSolver``, ``PhaseConfig``, ``PhaseResult``: the primary entry point.
  Configure a ``PhaseSolver`` with a ``PhaseConfig`` (which algorithm to run
  and how) and call ``.fit(stack)`` to recover phase; see
  :class:`phase.result.PhaseResult` for the recovered fields.
- :mod:`phase.methods` -- ``METHODS``, ``MethodParam``
- :mod:`phase.interference_model` -- ``model_stack``
- :mod:`phase.frame_contrast` -- ``measure_frame_contrast``, ``measure_frame_visibility``
- :mod:`phase.carrier` -- ``remove_carrier``, ``CarrierResult``
- :mod:`phase.reference` -- ``subtract_reference``, ``DifferenceResult``
- :mod:`phase.combine` -- ``combine_acquisitions``, ``CombinedResult``
- :mod:`phase.backend` -- NumPy/CuPy array-module dispatch shared by all of
  the above; every function accepts a ``device="auto"|"cpu"|"cuda"``
  argument and returns result arrays on whichever device it ran on. Use
  ``phase.backend.asnumpy`` to bring a result field back to the host
  explicitly.
"""

from . import backend
from .backend import asnumpy
from .carrier import CarrierResult, remove_carrier
from .combine import CombinedResult, combine_acquisitions
from .reference import DifferenceResult, subtract_reference
from .config import PhaseConfig
from .frame_contrast import measure_frame_contrast, measure_frame_visibility
from .interference_model import model_stack
from .methods import METHODS, MethodParam
from .result import PhaseResult
from .solver import PhaseSolver

__all__ = [
    "PhaseSolver",
    "PhaseConfig",
    "PhaseResult",
    "MethodParam",
    "METHODS",
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
