# src/phase_shift/result.py
"""Phase extraction result."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike

from .backend import to_device
from .methods import MethodParam
from .utils import format_value


@dataclass
class PhaseResult:
    """Fields recovered by :class:`phase_shift.solver.PhaseSolver`.

    Fields follow ``docs/interference_model.md`` Eq. (17). Arrays stay on the
    device the solver ran on; use :func:`phase_shift.backend.asnumpy` to bring
    them to the host.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase, in radians.
    a : np.ndarray, shape (H, W)
        Background intensity.
    b : np.ndarray, shape (H, W)
        Fringe amplitude.
    delta : np.ndarray, shape (N,)
        Per-frame phase step, in radians, with ``delta[0] = 0``.
    g : np.ndarray, shape (N,)
        Per-frame fringe gain; ``median(g) = 1`` when fitted.
    alpha : np.ndarray, shape (N,)
        Per-frame source-power factor, with ``median(alpha) = 1``.
    method_param : MethodParam
        Diagnostics of the method that produced this result.
    reconstruction_error : float
        RMS difference between the input stack and Eq. (17) evaluated at the
        fitted fields, in input units.
    phi_error : np.ndarray, shape (H, W), optional
        Per-pixel phase standard deviation, in radians; see ``docs/aia.md``
        §"Phase-error covariance". ``None`` for methods without an error model.
    """

    phi: np.ndarray
    a: np.ndarray
    b: np.ndarray
    delta: np.ndarray
    g: np.ndarray
    alpha: np.ndarray
    method_param: MethodParam
    reconstruction_error: float
    phi_error: np.ndarray | None = None

    def to_device(self, device: str = "auto", dtype: DTypeLike = None) -> "PhaseResult":
        """Return a copy with the array fields moved to ``device``.

        Parameters
        ----------
        device : {"auto", "cpu", "cuda"}, default "auto"
            Target device; see :func:`phase_shift.backend.to_device`.
        dtype : dtype, optional
            Cast while moving.

        Returns
        -------
        PhaseResult
            New result. ``method_param`` is carried over unchanged, including
            any arrays it holds.
        """
        return PhaseResult(
            phi=to_device(self.phi, device=device, dtype=dtype),
            a=to_device(self.a, device=device, dtype=dtype),
            b=to_device(self.b, device=device, dtype=dtype),
            delta=to_device(self.delta, device=device, dtype=dtype),
            g=to_device(self.g, device=device, dtype=dtype),
            alpha=to_device(self.alpha, device=device, dtype=dtype),
            method_param=self.method_param,
            reconstruction_error=self.reconstruction_error,
            phi_error=to_device(self.phi_error, device=device, dtype=dtype)
            if self.phi_error is not None else None,
        )

    def print_summary(self) -> None:
        """Print the method diagnostics, then ``reconstruction_error``."""
        self.method_param.print_summary()
        print(f"reconstruction_error: {format_value(self.reconstruction_error)}")
