# src/phase_shift/methods/step_field.py
"""Shared pieces of a spatially varying phase step ``Delta_n(x, y)``.

A method that expands the step field on a spatial basis reports its
coefficients through :class:`StepFieldParam` and scores them with
:func:`step_field_quality`; neither depends on how the coefficients were
fitted. See ``docs/sf_aia.md`` §"Algorithm" and ``docs/vp_aia.md``.
"""

from dataclasses import dataclass
from types import ModuleType

import numpy as np

from ..backend import Precision, get_array_module
from ..basis import spatial_basis
from ..errors import step_field_phi_error
from ..utils import format_value
from .base import MethodParam
from .diagnostics import AIAParam


def step_field_quality(stack: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
                       delta: np.ndarray, coeffs: np.ndarray, basis: np.ndarray,
                       H: int, W: int, g: np.ndarray | None = None, crop: int = 100,
                       precision: str | Precision | None = None
                       ) -> tuple[float, np.ndarray]:
    """Score how much of the residual the fitted step field explains.

    Step 4 of ``docs/sf_aia.md`` §"Algorithm": rebuild every frame at the
    corrected step ``delta_n + Delta_n`` and compare the residual RMS with the
    data RMS, over a border-cropped region where the fit is most reliable. A
    falling score round over round is what marks the fit as a real correction
    rather than fitted noise.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P = H*W`` pixels each.
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components, as passed to
        :func:`fit_step_field`.
    delta : np.ndarray, shape (N,)
        Piston phase step of each frame, in radians.
    coeffs : np.ndarray, shape (J, N)
        Per-frame coefficients, e.g. from :func:`fit_step_field`.
    basis : np.ndarray, shape (J, P)
        Step-field basis matching ``coeffs``.
    H, W : int
        Frame height and width, ``P = H*W``.
    g : np.ndarray, shape (N,), optional
        Per-frame fringe gain. Defaults to ones.
    crop : int, default 100
        Pixels excluded from each edge before either RMS; ``0`` uses the full
        field.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. Its
        ``accum`` is the dtype of every operand feeding the ``(N, P)``
        reconstruction, and so of ``resid``.

    Returns
    -------
    rms_frac : float
        Residual RMS over the cropped region, divided by ``stack``'s.
    resid : np.ndarray, shape (N, P)
        Residual of the corrected model, uncropped, in ``precision.accum``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, or ``crop`` leaves no pixels.
    """
    if stack.ndim != 2:
        raise ValueError(f"stack must be 2-D (N, P), got shape {stack.shape}")
    if a.shape != u.shape or a.shape != v.shape:
        raise ValueError(f"a, u and v must have the same shape, got {a.shape}, "
                         f"{u.shape} and {v.shape}")
    if a.ndim != 1 or a.shape[0] != stack.shape[1]:
        raise ValueError(f"a, u and v must be 1-D of length {stack.shape[1]}, "
                         f"got shape {a.shape}")
    if delta.ndim != 1 or delta.shape[0] != stack.shape[0]:
        raise ValueError(f"delta must be 1-D of length {stack.shape[0]}, "
                         f"got shape {delta.shape}")
    if coeffs.shape[1] != len(delta):
        raise ValueError(f"coeffs must have shape (J, {len(delta)}), got {coeffs.shape}")
    if basis.shape[0] != coeffs.shape[0] or basis.shape[1] != stack.shape[1]:
        raise ValueError(f"basis must have shape ({coeffs.shape[0]}, {stack.shape[1]}), "
                         f"got {basis.shape}")
    if stack.shape[1] != H * W:
        raise ValueError(f"stack's second dimension must equal H*W ({H * W}), "
                         f"got {stack.shape[1]}")
    if g is not None and len(g) != len(delta):
        raise ValueError(f"g must have length {len(delta)}, got {len(g)}")
    if crop < 0 or 2 * crop >= H or 2 * crop >= W:
        raise ValueError(f"crop={crop} leaves no pixels for frame size {H}x{W}")

    xp = get_array_module(stack, a, u, v, delta, coeffs, basis)
    N = stack.shape[0]
    # Every operand must reach calc_dtype before the (N, P) arrays are built;
    # casting resid afterwards is too late. basis is float64 from
    # spatial_basis, and u/v are float64 in this module's own use.
    calc_dtype = Precision.of(precision).accum
    delta = xp.asarray(delta, dtype=calc_dtype)
    coeffs = xp.asarray(coeffs, dtype=calc_dtype)
    g = xp.ones(N, dtype=calc_dtype) if g is None else xp.asarray(g, dtype=calc_dtype)
    basis = basis.astype(calc_dtype, copy=False)
    a = a.astype(calc_dtype, copy=False)
    u = u.astype(calc_dtype, copy=False)
    v = v.astype(calc_dtype, copy=False)

    dn = delta[:, None] + coeffs.T @ basis                               # (N, P)
    model = a[None, :] + g[:, None] * (u[None, :] * xp.cos(dn) + v[None, :] * xp.sin(dn))
    resid = stack - model

    # Border crop as a flat boolean mask; guarded because `mask[-0:]` would
    # cover the whole field.
    mask = xp.ones((H, W), dtype=bool)
    if crop > 0:
        mask[:crop] = mask[-crop:] = mask[:, :crop] = mask[:, -crop:] = False
    mask = mask.ravel()

    rms_frac = float(xp.std(resid[:, mask], dtype=calc_dtype)
                     / xp.std(stack[:, mask], dtype=calc_dtype))
    return rms_frac, resid

@dataclass
class StepFieldParam(MethodParam):
    """Diagnostics of a solve that recovered a spatially varying phase step.

    Base for the methods of ``docs/sf_aia.md`` and ``docs/vp_aia.md``: both
    expand ``Delta_n(x, y)`` on a spatial basis and report per-frame
    coefficients. A subclass adds its own convergence fields, e.g.
    :class:`phase_shift.methods.sf_aia.SFAIAParam`.

    Attributes
    ----------
    aia_param : AIAParam
        Accuracy diagnostics of the underlying AIA solve, recomputed against
        the final fields. Its ``iters_run``/``converged`` describe that solve,
        not the step-field fit.
    basis : str
        Basis family of the step field, one of
        :data:`phase_shift.basis.BASES`.
    basis_kwargs : dict
        Arguments the basis was built with, e.g. ``{"degree": 2}``.
    coeffs : np.ndarray, shape (J, N)
        Per-frame coefficients ``c_jn``, fit against the fields this result
        reports. Not gauge-fixed (``docs/sf_aia.md`` Eq. T3b): subtract the
        frame mean before reading a row as per-frame drift.
    coeffs_rms : np.ndarray, shape (J,)
        RMS of each row of ``coeffs`` across frames.
    kappa_fit : float
        Largest per-frame condition number of the coefficient fit,
        ``docs/sf_aia.md`` Eq. (E3). Large values mean the basis has outrun
        what the fringe pattern resolves, however good the fit's own score
        looks; see §"Conditioning".
    precision : Precision
        Dtypes the solve ran in, carried so :meth:`phase_step_field` rebuilds
        the step field the way :func:`step_field_quality` scored it.
    """

    aia_param: AIAParam
    basis: str
    basis_kwargs: dict
    coeffs: np.ndarray
    coeffs_rms: np.ndarray
    kappa_fit: float
    precision: Precision

    def phi_error(self, b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                  fit_gain: bool, noise_std: np.ndarray, simplified: bool,
                  xp: ModuleType) -> np.ndarray:
        """Return ``sigma_Phi`` including the step field's own contribution.

        ``docs/sf_aia.md`` §"Noise of the corrected solve", on top of the AIA
        map. See :meth:`phase_shift.methods.base.MethodParam.phi_error` for
        the arguments.

        Returns
        -------
        np.ndarray, shape (H, W)
            Per-pixel phase standard deviation, in radians.
        """
        H, W = phi.shape
        basis = spatial_basis(H, W, self.basis, xp, precision=self.precision,
                              **self.basis_kwargs)
        return step_field_phi_error(b, phi, delta, g, fit_gain, noise_std, simplified,
                                    basis, xp, precision=self.precision)

    def print_summary(self) -> None:
        """Print the AIA diagnostics, then the step-field ones."""
        self.aia_param.print_summary()
        print(f"basis:            {self.basis} {self.basis_kwargs}")
        print(f"kappa_fit:        {format_value(self.kappa_fit)}")
        print(f"coeffs_rms:       {format_value(self.coeffs_rms)}")

    def phase_step_field(self, delta: np.ndarray, H: int, W: int,
                         xp: ModuleType) -> np.ndarray:
        """Return ``delta_n`` plus the fitted step field ``coeffs[:, n] @ basis``.

        Overrides the piston-only broadcast of
        :meth:`phase_shift.methods.base.MethodParam.phase_step_field` with the
        spatially varying step, ``docs/sf_aia.md`` Eq. (T1). Built in
        ``precision.accum``, as :func:`step_field_quality` does.

        Parameters
        ----------
        delta : np.ndarray, shape (N,)
            Per-frame piston step, in radians.
        H, W : int
            Frame height and width.
        xp : module
            ``numpy`` or ``cupy``, matching ``delta``.

        Returns
        -------
        np.ndarray, shape (N, H, W)
        """
        calc_dtype = self.precision.accum
        basis = spatial_basis(H, W, self.basis, xp, precision=self.precision,
                              **self.basis_kwargs)                       # (J, P)
        coeffs = xp.asarray(self.coeffs, dtype=calc_dtype)
        N = delta.shape[0]
        field = delta[:, None] + coeffs.T @ basis                        # (N, P)
        return field.reshape(N, H, W)
