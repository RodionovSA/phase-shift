# src/phase_shift/methods/base.py
"""Shared base type for per-method diagnostics."""

from dataclasses import dataclass, fields
from types import ModuleType

import numpy as np

from ..utils import format_value


@dataclass
class MethodParam:
    """Base type for a method's per-fit diagnostics.

    Each registered method returns its own subclass carrying what that
    algorithm reports, e.g. :class:`phase_shift.methods.aia.AIAParam`. Stored
    on :attr:`phase_shift.result.PhaseResult.method_param`.
    """

    def print_summary(self) -> None:
        """Print each diagnostic field on its own line, in declaration order."""
        for f in fields(self):
            print(f"{f.name}: {format_value(getattr(self, f.name))}")

    def phi_error(self, b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                  fit_gain: bool, noise_std: np.ndarray, simplified: bool,
                  xp: ModuleType) -> np.ndarray | None:
        """Return the per-pixel phase standard deviation, or ``None``.

        Default: this method has no error model. A method with one overrides
        this, e.g. :meth:`phase_shift.methods.diagnostics.AIAParam.phi_error`.

        Parameters
        ----------
        b, phi : np.ndarray, shape (H, W)
            Fitted fringe amplitude and phase, in radians.
        delta, g : np.ndarray, shape (N,)
            Fitted per-frame phase step and gain.
        fit_gain : bool
            Whether ``g`` was fitted jointly with the phase steps.
        noise_std : np.ndarray, shape (H, W)
            Per-pixel noise standard deviation ``sigma_0``,
            ``docs/aia_noise.md`` Eq. (16).
        simplified : bool
            Drop the corrections that are ``O(1/N_p)`` relative to the
            baseline.
        xp : module
            ``numpy`` or ``cupy``, matching the inputs.

        Returns
        -------
        np.ndarray, shape (H, W), or None
            ``sigma_Phi``, in radians.
        """
        return None

    def phase_step_field(self, delta: np.ndarray, H: int, W: int,
                         xp: ModuleType) -> np.ndarray:
        """Return the per-frame, per-pixel phase step ``delta_n + Delta_n(x, y)``.

        This is the phase step of ``docs/interference_model.md`` Eq. (17).
        Broadcasting ``delta`` alone is its uniform-piston limit, Eq. (20); a
        method that recovers a spatially varying step overrides this, e.g.
        :class:`phase_shift.methods.step_field.StepFieldParam`.

        Parameters
        ----------
        delta : np.ndarray, shape (N,)
            Per-frame phase step, in radians.
        H, W : int
            Frame height and width.
        xp : module
            ``numpy`` or ``cupy``, matching ``delta``.

        Returns
        -------
        np.ndarray, shape (N, H, W)
            Phase step, in radians. May be a read-only broadcast view.
        """
        N = delta.shape[0]
        return xp.broadcast_to(delta[:, None, None], (N, H, W))
