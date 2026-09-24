# src/phase_shift/carrier.py
"""Estimating and removing a low-order carrier from a wrapped phase map."""

import warnings
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from .backend import Precision, asnumpy, get_array_module, to_device
from .basis import spatial_basis
from .utils import _estimation_weight, wrap


@dataclass
class CarrierResult:
    """Output of :func:`remove_carrier`.

    Conventions follow ``docs/gauge_conventions.md`` §"Carrier removal".

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Phase with the carrier removed, in radians, wrapped to ``[-pi, pi]``.
        Its weighted circular mean is zero.
    eta : np.ndarray, shape (L+1,)
        Carrier coefficients of ``docs/carrier_removal.md`` Eq. (1), piston
        first, float64.
    piston : float
        ``eta[0]``, in radians.
    n_iter : int
        Newton steps taken on the full grid.
    cond : float
        Condition number of ``S`` (Eq. 6) at the last step; ``inf`` if ``S``
        is not positive definite.
    """

    phi: np.ndarray
    eta: np.ndarray
    piston: float
    n_iter: int
    cond: float


def _carrier_basis(H: int, W: int, degree: int, xp: ModuleType,
                   p: Precision) -> np.ndarray:
    """Basis ``p_0 = 1, p_1, ..., p_L`` of ``docs/carrier_removal.md`` §"Basis".

    The ``"poly"`` rows are rescaled to unit mean square, the norm of ``p_0``.

    Returns
    -------
    np.ndarray, shape (L+1, H, W), ``p.accum``
    """
    rows = spatial_basis(H, W, "poly", xp, p, degree=degree)                # (L, H*W)
    P = xp.empty((rows.shape[0] + 1, H * W), dtype=p.accum)
    P[0] = 1
    P[1:] = rows * np.sqrt(H * W)
    return P.reshape(-1, H, W)


def _window_sum(a: np.ndarray, K: int) -> np.ndarray:
    """Sum of ``a`` over the ``K x K`` window centred on each pixel, truncated
    to the array, as in ``docs/carrier_removal.md`` Eq. (8)."""
    xp = get_array_module(a)
    h, w = a.shape
    r = K // 2
    c = xp.zeros((h + K, w + K), dtype=a.dtype)
    c[r + 1:r + 1 + h, r + 1:r + 1 + w] = a
    c = c.cumsum(0).cumsum(1)
    return c[K:K + h, K:K + w] - c[:h, K:K + w] - c[K:K + h, :w] + c[:h, :w]


def _start(zeta: np.ndarray, P: np.ndarray, window: int) -> np.ndarray:
    """Start from window-summed neighbour products, ``docs/carrier_removal.md``
    Eqs. (8)-(10), with the piston from Eq. (4).

    Parameters
    ----------
    zeta : np.ndarray, shape (h, w), complex
        Weighted complex field, Eq. (2).
    P : np.ndarray, shape (L+1, h, w)
        Basis sampled on the same grid.
    window : int
        Window size ``K``.

    Returns
    -------
    np.ndarray, shape (L+1,), float64
    """
    xp = get_array_module(zeta)
    L1 = P.shape[0]
    eta = np.zeros(L1)
    if L1 > 1:
        c_x = _window_sum(zeta[:, 1:] * zeta[:, :-1].conj(), window)         # (h, w-1)
        c_y = _window_sum(zeta[1:, :] * zeta[:-1, :].conj(), window)         # (h-1, w)
        P_x = (P[1:, :, 1:] - P[1:, :, :-1]).reshape(L1 - 1, -1)
        P_y = (P[1:, 1:, :] - P[1:, :-1, :]).reshape(L1 - 1, -1)
        w_x, w_y = xp.abs(c_x).ravel(), xp.abs(c_y).ravel()
        G = (P_x * w_x) @ P_x.T + (P_y * w_y) @ P_y.T
        r = P_x @ (w_x * xp.angle(c_x).ravel()) + P_y @ (w_y * xp.angle(c_y).ravel())
        eta[1:] = np.linalg.solve(asnumpy(G), asnumpy(r))
    psi = xp.asarray(eta, dtype=P.dtype) @ P.reshape(L1, -1)
    eta[0] = float(xp.angle(xp.sum(zeta.ravel() * xp.exp(-1j * psi))))
    return eta


def _newton(zeta: np.ndarray, P: np.ndarray, eta: np.ndarray,
            max_iter: int) -> tuple[np.ndarray, int, float, bool]:
    """Newton steps of ``docs/carrier_removal.md`` Eqs. (5)-(7).

    Stops when no coefficient moves by more than the square root of the
    basis dtype's resolution.

    Returns
    -------
    eta : np.ndarray, shape (L+1,), float64
    n_iter : int
        Steps taken.
    cond : float
        Condition number of ``S`` at the last step; ``inf`` if ``S`` is not
        positive definite.
    converged : bool
    """
    xp = get_array_module(zeta)
    P_f = P.reshape(P.shape[0], -1)                                          # (L+1, h*w)
    z_f = zeta.ravel()
    tol = float(np.sqrt(np.finfo(P.dtype).eps))
    converged = False
    for n_iter in range(1, max_iter + 1):
        d = z_f * xp.exp(-1j * (xp.asarray(eta, dtype=P.dtype) @ P_f))      # omega e^{i chi}
        h = asnumpy(P_f @ d.imag)
        S = asnumpy((P_f * d.real) @ P_f.T)
        step = np.linalg.solve(S, h)
        eta = eta + step
        if np.abs(step).max() < tol:
            converged = True
            break
    ev = np.linalg.eigvalsh(S)
    cond = float(ev[-1] / ev[0]) if ev[0] > 0 else np.inf
    return eta, n_iter, cond, converged


def remove_carrier(phi: np.ndarray, weight: np.ndarray | None = None,
                   mask: np.ndarray | None = None, degree: int = 2, window: int = 5,
                   subsample: int = 4, max_iter: int = 20, device: str = "auto",
                   precision: "str | Precision | None" = None) -> CarrierResult:
    """Fit and remove a low-order carrier from a wrapped phase map.

    Maximizes ``docs/carrier_removal.md`` Eq. (3) over a polynomial carrier
    of total degree up to ``degree`` plus a piston, working on
    ``weight * exp(1j * phi)`` so the map is never unwrapped. The start
    (Eqs. 8-10) and the first Newton steps (Eq. 7) run on every
    ``subsample``-th pixel; the fit is finished on the full grid
    (§"Subsampled start").

    Parameters
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase map, in radians, e.g.
        :attr:`phase_shift.result.PhaseResult.phi`.
    weight : np.ndarray, shape (H, W), optional
        Per-pixel reliability ``omega``, e.g.
        :attr:`phase_shift.result.PhaseResult.b`. Negative values are clipped
        to 0.
    mask : np.ndarray, shape (H, W), optional
        Pixels where it is falsey are excluded; combined with ``weight``.
    degree : int, default 2
        Highest total degree ``M`` of the carrier; ``0`` removes only the
        piston.
    window : int, default 5
        Window size ``K`` of Eq. (8), a positive odd number of pixels.
    subsample : int, default 4
        Stride of the grid the start runs on; ``1`` runs everything on the
        full grid. The phase must change by less than ``pi`` over
        ``subsample`` pixels.
    max_iter : int, default 20
        Maximum Newton steps on each grid.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Device to run on; see :func:`phase_shift.backend.to_device`.
    precision : str or Precision, optional
        Dtypes to run in; see :class:`phase_shift.backend.Precision`.

    Returns
    -------
    CarrierResult

    Raises
    ------
    ValueError
        If ``phi`` is not 2-D, if ``weight`` or ``mask`` does not have
        ``phi``'s shape, if ``degree`` is negative, if ``window`` is not a
        positive odd number, if ``subsample`` or ``max_iter`` is below 1, or
        if the subsampled grid has fewer than 2 pixels along an axis.

    Warns
    -----
    UserWarning
        If the Newton steps on the full grid do not converge, or end where
        ``S`` is not positive definite.
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D (H, W), got shape {phi.shape}")
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd number, got {window}")
    if subsample < 1:
        raise ValueError(f"subsample must be at least 1, got {subsample}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    H, W = phi.shape
    if -(-H // subsample) < 2 or -(-W // subsample) < 2:
        raise ValueError(
            f"subsample={subsample} leaves fewer than 2 pixels along an axis of "
            f"the {H}x{W} field"
        )

    p = Precision.of(precision)
    phi = to_device(phi, device=device)
    xp = get_array_module(phi)
    w = _estimation_weight(phi, weight, mask, device)
    zeta = (w * xp.exp(1j * phi)).astype(np.result_type(p.accum, np.complex64))  # Eq. (2)
    P = _carrier_basis(H, W, degree, xp, p)                                  # (L+1, H, W)

    if subsample > 1:
        zeta_s, P_s = zeta[::subsample, ::subsample], P[:, ::subsample, ::subsample]
        eta, *_ = _newton(zeta_s, P_s, _start(zeta_s, P_s, window), max_iter)
    else:
        eta = _start(zeta, P, window)
    eta, n_iter, cond, converged = _newton(zeta, P, eta, max_iter)

    if not converged:
        warnings.warn(f"remove_carrier: Newton steps did not converge in {max_iter} "
                      f"iterations", stacklevel=2)
    if not np.isfinite(cond):
        warnings.warn("remove_carrier: the fit ended where S is not positive definite, "
                      "so it is not at a maximum; check that the phase changes by less "
                      "than pi over subsample pixels", stacklevel=2)

    psi = (xp.asarray(eta, dtype=p.accum) @ P.reshape(P.shape[0], -1)).reshape(H, W)
    return CarrierResult(phi=wrap(phi - psi).astype(p.work), eta=eta,
                         piston=float(eta[0]), n_iter=n_iter, cond=cond)
