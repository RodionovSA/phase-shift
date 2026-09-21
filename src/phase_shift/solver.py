# src/phase_shift/solver.py
"""Phase extraction front end dispatching to registered methods."""

import numpy as np

from .backend import Precision, get_array_module, to_device
from .config import PhaseConfig
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
    precision : {"single", "double", "fast"} or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. Resolved
        once here, so a later :func:`phase_shift.backend.set_precision` does
        not change this solver. Defaults to
        :func:`phase_shift.backend.get_precision`.

    Attributes
    ----------
    precision : Precision
        The resolved precision this solver runs in.
    result : PhaseResult or None
        Result of the last :meth:`fit`; ``None`` before fitting.

    Raises
    ------
    ValueError
        If ``precision`` is not recognized.
    """

    def __init__(self, config: PhaseConfig, device: str = "auto",
                 precision: str | Precision | None = None) -> None:
        self.config = config
        self.device = device
        self.precision = Precision.of(precision)
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
            If ``stack`` is not 3-D, if ``config.g`` does not have shape
            ``(N,)``, or if a frame has non-positive mean intensity while
            ``config.use_alpha`` is set.
        """
        if stack.ndim != 3:
            raise ValueError(f"stack must be 3-D (N, H, W), got shape {stack.shape}")
        stack = to_device(stack, device=self.device, dtype=self.precision.work)
        xp = get_array_module(stack)

        normalized_stack, alpha = self._alpha_norm(stack, self.config.use_alpha)
        g, fit_gain = self._g_fit(stack, self.config.gain_mode, self.config.g)
        a, b, phi, delta, g, method_param = self._solve(normalized_stack, g, fit_gain)
        residual_mean_sq = self._residual_mean_sq(stack, a, b, phi, delta, g, alpha,
                                                  method_param)
        rmse = float(xp.sqrt(xp.mean(residual_mean_sq, dtype=xp.float64)))
        phi_error = self._phi_error(b, phi, delta, g, fit_gain, residual_mean_sq,
                                    method_param)

        self.result = PhaseResult(phi, a, b, delta, g, alpha, method_param, rmse,
                                  self.precision, phi_error)
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
        return solve_fn(stack, g, fit_gain=fit_gain, precision=self.precision,
                        **self.config.method_kwargs)

    def _alpha_norm(self, stack: np.ndarray, use_alpha: bool) -> tuple[np.ndarray, np.ndarray]:
        """Divide each frame by its source-power factor ``alpha_n``.

        ``alpha_n`` is the frame mean scaled to ``median(alpha) = 1``; see
        ``docs/aia.md`` §"Starting point".

        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W)
            Interferogram stack.
        use_alpha : bool
            If False, return ``stack`` unchanged and ``alpha_n = 1``.

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
        if not use_alpha:
            return stack, xp.ones(stack.shape[0], dtype=xp.float64)

        m = xp.mean(stack, axis=(1, 2), dtype=xp.float64)                   # (N,)
        if not float(xp.min(m)) > 0:
            raise ValueError("every frame must have a positive mean intensity")
        alpha = m / xp.median(m)
        return stack / alpha.astype(stack.dtype)[:, None, None], alpha

    def _g_fit(self, stack: np.ndarray, gain_mode: str,
               g: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Resolve the per-frame fringe gain ``g_n`` the method starts from.

        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W)
            Interferogram stack; only its frame count and array module are used.
        gain_mode : {"none", "joint"}
            ``"joint"`` lets the method fit ``g_n``; ignored when ``g`` is given.
        g : np.ndarray, shape (N,), optional
            Fixed gain. If None, start from ``g_n = 1``.

        Returns
        -------
        g : np.ndarray, shape (N,), float64
            Fixed or initial per-frame gain.
        fit_gain : bool
            Whether the method fits ``g_n`` jointly with the phase steps.

        Raises
        ------
        ValueError
            If ``g`` does not have shape ``(N,)``.
        """
        xp = get_array_module(stack)
        N = stack.shape[0]
        if g is None:
            return xp.ones(N, dtype=xp.float64), gain_mode == "joint"

        g = xp.asarray(g, dtype=xp.float64)
        if g.shape != (N,):
            raise ValueError(f"g must have shape ({N},), got {g.shape}")
        return g, False

    def _residual_mean_sq(self, stack: np.ndarray, a: np.ndarray, b: np.ndarray,
                          phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                          alpha: np.ndarray, method_param: MethodParam) -> np.ndarray:
        """Mean squared difference between ``stack`` and the fitted model, per pixel.

        Evaluates ``docs/interference_model.md`` Eq. (17) at the fitted fields
        via :func:`phase_shift.interference_model.model_stack`, one frame at a
        time so no ``(N, H, W)`` temporary is held. The per-frame step comes
        from :meth:`phase_shift.methods.base.MethodParam.phase_step_field`, so
        a spatially varying step is reconstructed as the method recovered it.

        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W)
            Input stack, before ``alpha_n`` normalization.
        a, b, phi : np.ndarray, shape (H, W)
            Fitted background, fringe amplitude, and phase in radians.
        delta, g, alpha : np.ndarray, shape (N,)
            Fitted phase steps and gain, and the source-power factor.
        method_param : MethodParam
            Diagnostics of the method that produced the fit.

        Returns
        -------
        np.ndarray, shape (H, W)
            Residual mean square, in squared input units.
        """
        xp = get_array_module(stack)
        work = self.precision.work
        N, H, W = stack.shape
        delta_field = method_param.phase_step_field(delta.astype(work), H, W, xp)
        g_w = g.astype(work)
        alpha_w = alpha.astype(work)

        residual_sq_sum = xp.zeros((H, W), dtype=work)
        for n in range(N):
            model = model_stack(a, b, phi, delta_field[n:n + 1], g_w[n:n + 1],
                                alpha_w[n:n + 1])                           # (1, H, W)
            residual_sq_sum += (stack[n] - model[0]) ** 2
        return residual_sq_sum / N

    def _phi_error(self, b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                   fit_gain: bool, residual_mean_sq: np.ndarray,
                   method_param: MethodParam) -> np.ndarray | None:
        """Per-pixel phase standard deviation of the fit.

        Uses ``config.noise_std`` as ``sigma_0`` when given, otherwise the
        per-pixel RMS fit residual, and asks the method for its own map; see
        :meth:`phase_shift.methods.base.MethodParam.phi_error`.

        Parameters
        ----------
        b, phi : np.ndarray, shape (H, W)
            Fitted fringe amplitude and phase in radians.
        delta, g : np.ndarray, shape (N,)
            Fitted per-frame phase steps and gain.
        fit_gain : bool
            Whether ``g`` was fitted jointly with the phase steps.
        residual_mean_sq : np.ndarray, shape (H, W)
            Per-pixel residual mean square from :meth:`_residual_mean_sq`.
        method_param : MethodParam
            Diagnostics of the method that produced the fit.

        Returns
        -------
        np.ndarray, shape (H, W), or None
            ``sigma_Phi`` in radians, or None for a method without an error
            model.

        Raises
        ------
        ValueError
            If ``config.noise_std``'s shape does not match ``phi``.
        """
        xp = get_array_module(b)
        if self.config.noise_std is not None:
            noise_std = to_device(self.config.noise_std, device=self.device,
                                  dtype=self.precision.work)
        else:
            noise_std = xp.sqrt(residual_mean_sq)
        if noise_std.shape != phi.shape:
            raise ValueError(
                f"noise_std shape {noise_std.shape} does not match phi shape {phi.shape}"
            )
        return method_param.phi_error(b, phi, delta, g, fit_gain, noise_std,
                                      self.config.phi_error_simplified, xp)
