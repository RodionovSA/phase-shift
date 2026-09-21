# src/phase_shift/methods/aia.py
"""Advanced Iterative Algorithm (AIA) of ``docs/aia.md``."""

import numpy as np

from ..backend import Precision, get_array_module
from ..utils import wrap
from .diagnostics import AIAParam, aia_diagnostics
from .gauge import center_offsets, normalize_gain, pin_phase_origin, whiten_uv
from .steps import frame_step, pixel_step


def aia(stack: np.ndarray, g: np.ndarray, fit_gain: bool = False,
        delta0: np.ndarray | None = None, iters: int = 30, tol: float = 1e-4,
        precision: str | Precision | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, AIAParam]:
    """Recover phase and phase steps from a stack with unknown steps.

    Alternates the pixel step (:func:`phase_shift.methods.steps.pixel_step`)
    with the frame step (:func:`phase_shift.methods.steps.frame_step`) until
    the phase steps stop moving; see ``docs/aia.md`` §"Alternating least
    squares". With ``fit_gain``, the gain is recovered alongside them and the
    quadrature fields are whitened after each pixel step
    (:func:`phase_shift.methods.gauge.whiten_uv`).

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted frames, already divided by ``alpha_n`` and on the target
        device, as :meth:`phase_shift.solver.PhaseSolver.fit` leaves them.
    g : np.ndarray, shape (N,)
        Per-frame fringe gain: held fixed when ``fit_gain`` is False, the
        starting value when True.
    fit_gain : bool, default False
        Recover ``g_n`` jointly with ``delta_n``. Preferable to an out-of-band
        estimate such as
        :func:`phase_shift.frame_contrast.measure_frame_contrast`, which needs
        a spatial carrier, whenever the contrast drifts between frames.
    delta0 : np.ndarray, shape (N,), optional
        Starting phase steps, in radians. Defaults to evenly spaced steps,
        which minimize ``kappa_ps``.
    iters : int, default 30
        Maximum number of alternations.
    tol : float, default 1e-4
        Convergence tolerance on the largest per-frame change in ``delta``,
        and in ``g`` when ``fit_gain`` is set.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`. Sets the
        dtype of the ``(N, P)`` arrays and of the returned fields; the
        per-iteration linear algebra is float64 regardless.

    Returns
    -------
    a, b, phi : np.ndarray, shape (H, W)
        Background, fringe amplitude, and wrapped phase in radians.
    delta, g : np.ndarray, shape (N,)
        Per-frame phase step, in radians, with ``delta[0] = 0``, and fringe
        gain: the input ``g`` when ``fit_gain`` is False, else the fitted gain
        with ``median(g) = 1``.
    method_param : AIAParam
        Convergence and accuracy diagnostics.

    Raises
    ------
    ValueError
        If ``stack`` is not 3-D, ``iters`` is below 1, or ``tol`` is negative.

    References
    ----------
    Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of
    randomly phase-shifted interferograms," Optics and Lasers in Engineering
    (2004).

    Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase extraction:
    performance evaluation and enhancement," Optics Express 27(26),
    37634-37651 (2019).
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (N, H, W), got shape {stack.shape}")
    if iters < 1:
        raise ValueError(f"iters must be at least 1, got {iters}")
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")

    xp = get_array_module(stack)
    N, H, W = stack.shape
    p = Precision.of(precision)
    I = stack.reshape(N, -1).astype(p.work, copy=False)              # (N, P)

    if delta0 is None:
        delta0 = xp.arange(N) * 2 * xp.pi / N
    delta = xp.asarray(delta0, dtype=xp.float64).copy()
    g = xp.asarray(g, dtype=xp.float64)
    c = xp.zeros(N, dtype=xp.float64)
    # Only the joint-gain solve offsets the stack, and then into one reused
    # buffer rather than a fresh (N, P) array per iteration.
    I_pixel = xp.empty_like(I) if fit_gain else I

    u = v = a = None
    delta_fit = delta
    g_fit = g
    c_fit = c
    converged = False
    for it in range(iters):
        delta_fit = delta
        g_fit = g
        c_fit = c
        if fit_gain:
            xp.subtract(I, c_fit.astype(p.work)[:, None], out=I_pixel)
        a, u, v = pixel_step(I_pixel, delta_fit, g_fit, precision=p)
        if fit_gain:
            u, v = whiten_uv(u, v, xp, precision=p)
        new_delta, new_g, new_c = frame_step(I, u, v, precision=p)

        new_delta = pin_phase_origin(new_delta)
        step = float(xp.abs(wrap(new_delta - delta)).max())

        if fit_gain:
            new_c = center_offsets(new_c, xp)
            new_g = normalize_gain(new_g, xp)
            step = max(step, float(xp.abs(new_g - g).max()))
            g = new_g
            c = new_c

        delta = new_delta
        if step < tol:
            converged = True
            break

    phi = xp.arctan2(-v, u).reshape(H, W)
    b = xp.hypot(u, v).reshape(H, W)
    a_map = a.reshape(H, W)

    # Diagnostics describe what (a, u, v) were fit against, not the last
    # frame step's update.
    method_param = aia_diagnostics(I, delta_fit, g_fit, a, u, v, N, xp, it + 1, converged,
                                   c=(c_fit if fit_gain else None), precision=p)
    return a_map, b, phi, delta, g, method_param
