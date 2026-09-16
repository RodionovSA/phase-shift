# src/phase_shift/solver.py
"""Phase extraction front end dispatching to registered methods."""

import numpy as np
from numpy.typing import DTypeLike

from .backend import get_array_module, to_device
from .config import PhaseConfig
from .errors import compute_phi_error
from .interference_model import model_stack
from .methods import METHOD_REGISTRY, MethodParam
from .result import PhaseResult


class PhaseSolver:
    """Recover phase from interferogram stacks with a registered method.

    Parameters
    ----------
    config : PhaseConfig
        Method and options.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Device to run on; see :func:`phase_shift.backend.to_device`.
    dtype : dtype, optional
        Working dtype. If None, the stack keeps its dtype and the method uses
        :func:`phase_shift.backend.default_dtype`.

    Attributes
    ----------
    result : PhaseResult or None
        Result of the last :meth:`fit`; ``None`` before fitting.
    """

    def __init__(self, config: PhaseConfig, device: str = "auto",
                 dtype: DTypeLike = None) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.result: PhaseResult | None = None

    def fit(self, stack: np.ndarray) -> "PhaseSolver":
        """Recover phase from an interferogram stack.

        Divides out ``alpha_n`` when ``config.use_alpha`` is set, runs the
        configured method, then computes ``reconstruction_error`` and
        ``phi_error``.

        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W)
            Phase-shifted interferograms.

        Returns
        -------
        PhaseSolver
            This solver, for chaining, e.g. ``solver.fit(stack).result.phi``.

        Raises
        ------
        ValueError
            If ``stack`` is not 3-D, or a frame has non-positive mean intensity
            while ``config.use_alpha`` is set.
        """
        if stack.ndim != 3:
            raise ValueError(f"stack must be 3-D (N, H, W), got shape {stack.shape}")
        stack = to_device(stack, device=self.device, dtype=self.dtype)
        xp = get_array_module(stack)
        
        normalized_stack, alpha = self._alpha_norm(stack, self.config.use_alpha)
        g, fit_gain = self._g_fit(stack, self.config.gain_mode, self.config.g)
        a, b, phi, delta, g, method_param = self._solve(normalized_stack, g, fit_gain)
        residual_sq = self._rec_error(stack, a, b, phi, method_param, delta, g, alpha)
        rmse = float(xp.sqrt(xp.mean(residual_sq)))
        if self.config.noise_std is not None:
            noise_std = to_device(self.config.noise_std, device=self.device, dtype=self.dtype)
        else:
            noise_std = xp.sqrt(xp.mean(residual_sq, axis=0))
        phi_error = compute_phi_error(self.config.method, b, phi, delta, g, fit_gain,
                                      noise_std, self.config.phi_error_simplified,
                                      method_param, xp)

        self.result = PhaseResult(phi, a, b, delta, g, alpha, method_param, rmse, phi_error)
        return self

    def _solve(self, stack: np.ndarray, g: np.ndarray, fit_gain: bool
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          MethodParam]:
        """Run the configured method on the normalized stack.

        Returns
        -------
        a, b, phi, delta, g, method_param
            Fitted fields and diagnostics; see
            :class:`phase_shift.result.PhaseResult`.
        """
        solve_fn = METHOD_REGISTRY[self.config.method.lower()]
        return solve_fn(stack, g, fit_gain=fit_gain, dtype=self.dtype,
                        precise_reduce=self.config.precise_reduce, **self.config.method_kwargs)

    def _alpha_norm(self, stack: np.ndarray, use_alpha: bool) -> tuple[np.ndarray, np.ndarray]:
        """Divide each frame by its source-power factor ``alpha_n``.

        ``alpha_n`` is the frame mean scaled to ``median(alpha) = 1``; see
        ``docs/aia.md`` §"Starting point".

        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W)
            Interferogram stack.

        Returns
        -------
        normalized_stack : np.ndarray, shape (N, H, W)
            ``stack`` with each frame divided by ``alpha_n``.
        alpha : np.ndarray, shape (N,)
            Per-frame source-power factor.

        Raises
        ------
        ValueError
            If a frame has non-positive mean intensity.
        """
        xp = get_array_module(stack)
        m = xp.mean(stack, axis=(1, 2))
        if not float(xp.min(m)) > 0:
            raise ValueError("every frame must have a positive mean intensity")
        
        alpha = m / xp.median(m) if use_alpha else xp.ones(stack.shape[0], dtype=xp.float64)
        return stack / alpha[:, None, None], alpha
    
    def _g_fit(self, stack: np.ndarray, gain_mode: str, g: list| None = None) -> tuple[np.ndarray, bool]:
        xp = get_array_module(stack)
        if g is not None:
            g = xp.asarray(g, dtype=xp.float64)
            fit_gain = False
        else:
            g = xp.ones(stack.shape[0])
            fit_gain = gain_mode == "joint"
            
        return g, fit_gain
    
    def _rec_error(self, stack: np.ndarray, a: np.ndarray, b: np.ndarray, phi: np.ndarray, 
               method_param: MethodParam, delta: np.ndarray, g: np.ndarray, 
               alpha: np.ndarray) -> np.ndarray:
        
        xp = get_array_module(stack)
        H, W = phi.shape
        model = model_stack(a, b, phi, method_param.phase_step_field(delta, H, W, xp), g, alpha)
        residual_sq = (stack - model) ** 2
        return residual_sq 
    
    def _phi_error():
        ...
