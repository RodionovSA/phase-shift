# src/phase_shift/carrier.py
"""Estimating and removing the spatial carrier and defocus of a phase map."""

import cmath
import math
import warnings
from dataclasses import dataclass

import numpy as np

from .backend import get_array_module, to_device
from .utils import _estimation_weight


@dataclass
class CarrierResult:
    """Output of :func:`remove_carrier`.

    Conventions follow ``docs/gauge_conventions.md`` §"Carrier removal": the
    origin is pixel ``(0, 0)`` with unnormalized ``x, y``, and the output's
    weighted circular mean is zero.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Phase with the carrier, the curvature term (when fitted), and the
        piston removed, in radians, wrapped to ``[-pi, pi]``.
    kx, ky : float
        Carrier spatial frequency along the column (x) and row (y) axes, in
        rad/pixel.
    fx, fy : float
        The same carrier in cycles/pixel, ``kx = 2*pi*fx``.
    kxx, kyy, kxy : float
        Curvature coefficients of ``kxx*x^2 + kyy*y^2 + kxy*x*y``, in
        rad/pixel^2. All ``0.0`` when ``defocus=False``.
    piston : float
        Constant phase offset removed after demodulation, in radians.
    """

    phi: np.ndarray
    kx: float
    ky: float
    fx: float
    fy: float
    kxx: float
    kyy: float
    kxy: float
    piston: float


def _next_smooth(n: int, factors: tuple[int, ...] = (2, 3, 5, 7)) -> int:
    """Return the smallest ``m >= n`` whose prime factors all lie in ``factors``.

    Parameters
    ----------
    n : int
        Lower bound, in samples.
    factors : tuple of int, default (2, 3, 5, 7)
        Allowed prime factors.

    Returns
    -------
    int
        FFT length to zero-pad to.
    """
    k = n
    while True:
        m = k
        for f in factors:
            while m % f == 0:
                m //= f
        if m == 1:
            return k
        k += 1


def _coarse_peak(field: np.ndarray) -> tuple[float, float]:
    """Locate the dominant peak of ``field``'s spectrum, to one FFT bin.

    The transform is zero-padded to an efficient length (:func:`_next_smooth`).

    Parameters
    ----------
    field : np.ndarray, shape (H, W)
        Complex field to transform.

    Returns
    -------
    fy, fx : float
        Peak frequency along the row and column axes, in cycles/pixel.
    """
    xp = get_array_module(field)
    H, W = field.shape
    Hp, Wp = _next_smooth(H), _next_smooth(W)
    if (Hp, Wp) != (H, W):
        padded = xp.zeros((Hp, Wp), dtype=field.dtype)
        padded[:H, :W] = field
    else:
        padded = field
    F = xp.fft.fft2(padded)
    iy, ix = xp.unravel_index(xp.argmax(xp.abs(F)), F.shape)
    fy = float(xp.fft.fftfreq(Hp)[int(iy)])
    fx = float(xp.fft.fftfreq(Wp)[int(ix)])
    return fy, fx


def _estimate_tilt(c: np.ndarray, w: np.ndarray, window: bool,
                   refine_iters: int) -> tuple[float, float]:
    """Estimate the dominant linear carrier of a complex field.

    A coarse FFT peak (:func:`_coarse_peak`) is refined on the vector-averaged
    phase difference of neighboring pixels,
    ``Sx = sum(w_pair * c[:, 1:] * conj(c[:, :-1]))`` along columns and ``Sy``
    along rows. ``Sx``/``Sy`` do not depend on the current estimate, so they
    are formed once and the refinement iterates on their angles, which keeps
    the estimate wrap-safe. See ``docs/carrier_removal.md`` §5.

    Parameters
    ----------
    c : np.ndarray, shape (H, W)
        Complex field, ``exp(1j * phi)``.
    w : np.ndarray, shape (H, W)
        Per-pixel estimation weight.
    window : bool
        Apply a 2-D Hann window before the coarse FFT.
    refine_iters : int
        Maximum number of refinement iterations.

    Returns
    -------
    fx, fy : float
        Carrier frequency along the column and row axes, in cycles/pixel.
    """
    xp = get_array_module(c, w)
    H, W = c.shape
    if window and H > 1 and W > 1:
        win = xp.outer(xp.hanning(H), xp.hanning(W))
    else:
        win = xp.ones((H, W))
    fy, fx = _coarse_peak(w * c * win)

    Sx = Sy = None
    if W > 1:
        wpx = w[:, 1:] * w[:, :-1]
        Sx = complex(xp.sum(wpx * c[:, 1:] * xp.conj(c[:, :-1])))
    if H > 1:
        wpy = w[1:, :] * w[:-1, :]
        Sy = complex(xp.sum(wpy * c[1:, :] * xp.conj(c[:-1, :])))
    ax = cmath.phase(Sx) if Sx is not None else 0.0
    ay = cmath.phase(Sy) if Sy is not None else 0.0

    two_pi = 2 * math.pi
    for _ in range(refine_iters):
        dfx = cmath.phase(cmath.exp(1j * (ax - two_pi * fx))) / two_pi if Sx is not None else 0.0
        dfy = cmath.phase(cmath.exp(1j * (ay - two_pi * fy))) / two_pi if Sy is not None else 0.0
        fx += dfx
        fy += dfy
        if abs(dfx) < 1e-8 and abs(dfy) < 1e-8:
            break

    return fx, fy


def _estimate_curvature(c: np.ndarray, w: np.ndarray, window: bool, refine_iters: int,
                        n_blocks: int) -> tuple[float, float, float]:
    """Estimate quadratic curvature from the local tilt of a block grid.

    For ``phi = piston + kx*x + ky*y + kxx*x^2 + kyy*y^2 + kxy*x*y`` the local
    frequency is linear in position,
    ``fx(x, y) = fx0 + (kxx/pi)*x + (kxy/(2*pi))*y`` and
    ``fy(x, y) = fy0 + (kxy/(2*pi))*x + (kyy/pi)*y``. The field is split into
    an ``n_blocks x n_blocks`` grid of equal-size blocks, cropped to the
    largest size divisible by ``n_blocks``; every block's local frequency is
    estimated as in :func:`_estimate_tilt`, batched over one FFT, and the two
    relations above are fitted jointly by least squares over all blocks.

    Parameters
    ----------
    c : np.ndarray, shape (H, W)
        Complex field, ``exp(1j * phi)``.
    w : np.ndarray, shape (H, W)
        Per-pixel estimation weight.
    window : bool
        Apply a 2-D Hann window before each block's coarse FFT.
    refine_iters : int
        Number of refinement iterations per block.
    n_blocks : int
        Grid size along each axis.

    Returns
    -------
    kxx, kyy, kxy : float
        Curvature coefficients, in rad/pixel^2; all ``0.0`` if the fit is
        under-determined.

    Warns
    -----
    UserWarning
        If the blocks are smaller than 2x2, or fewer than 4 of them carry
        usable weight for the 5-unknown fit.
    """
    xp = get_array_module(c, w)
    H, W = c.shape
    bh, bw = H // n_blocks, W // n_blocks
    if bh < 2 or bw < 2:
        warnings.warn(
            "remove_carrier: n_blocks too large for this image size; "
            "cannot fit a curvature term, skipping it (kxx=kyy=kxy=0). "
            "Try a smaller n_blocks.",
            stacklevel=3,
        )
        return 0.0, 0.0, 0.0
    Hc, Wc = bh * n_blocks, bw * n_blocks

    def to_blocks(a: np.ndarray) -> np.ndarray:
        """Reshape ``a`` into ``(n_blocks**2, bh, bw)`` equal-size blocks."""
        return (a[:Hc, :Wc].reshape(n_blocks, bh, n_blocks, bw)
                 .transpose(0, 2, 1, 3).reshape(n_blocks * n_blocks, bh, bw))

    c_b = to_blocks(c)
    w_b = to_blocks(w)
    nblk = n_blocks * n_blocks

    win = (xp.outer(xp.hanning(bh), xp.hanning(bw)) if (window and bh > 1 and bw > 1)
           else xp.ones((bh, bw)))

    # Coarse peak per block, from one padded batched FFT.
    bhp, bwp = _next_smooth(bh), _next_smooth(bw)
    field = w_b * c_b * win
    if (bhp, bwp) != (bh, bw):
        padded = xp.zeros((nblk, bhp, bwp), dtype=field.dtype)
        padded[:, :bh, :bw] = field
    else:
        padded = field
    F = xp.fft.fft2(padded, axes=(1, 2))
    Fabs = xp.abs(F).reshape(nblk, -1)
    peak = xp.argmax(Fabs, axis=1)
    iy, ix = peak // bwp, peak % bwp
    fy = xp.fft.fftfreq(bhp)[iy].astype(xp.float64)                      # (nblk,)
    fx = xp.fft.fftfreq(bwp)[ix].astype(xp.float64)                      # (nblk,)

    # Per-block Sx/Sy and refinement, vectorized over the block axis.
    have_x, have_y = bw > 1, bh > 1
    if have_x:
        wpx = w_b[:, :, 1:] * w_b[:, :, :-1]
        Sx = (wpx * c_b[:, :, 1:] * xp.conj(c_b[:, :, :-1])).sum(axis=(1, 2))
        ax = xp.angle(Sx)
    if have_y:
        wpy = w_b[:, 1:, :] * w_b[:, :-1, :]
        Sy = (wpy * c_b[:, 1:, :] * xp.conj(c_b[:, :-1, :])).sum(axis=(1, 2))
        ay = xp.angle(Sy)

    two_pi = 2 * xp.pi
    for _ in range(refine_iters):
        if have_x:
            fx = fx + xp.angle(xp.exp(1j * (ax - two_pi * fx))) / two_pi
        if have_y:
            fy = fy + xp.angle(xp.exp(1j * (ay - two_pi * fy))) / two_pi

    block_row = xp.repeat(xp.arange(n_blocks), n_blocks)
    block_col = xp.tile(xp.arange(n_blocks), n_blocks)
    yc = block_row.astype(xp.float64) * bh + (bh - 1) / 2.0              # (nblk,)
    xc = block_col.astype(xp.float64) * bw + (bw - 1) / 2.0              # (nblk,)

    total_w = float(w.sum())
    floor = 0.01 * total_w / max(nblk, 1)
    w_sum = w_b.reshape(nblk, -1).sum(axis=1)
    usable = w_sum >= floor

    n = int(xp.sum(usable))
    if n < 4:
        warnings.warn(
            "remove_carrier: fewer than 4 blocks had usable weight; "
            "cannot fit a curvature term, skipping it (kxx=kyy=kxy=0). "
            "Try a smaller n_blocks or check weight/mask coverage.",
            stacklevel=3,
        )
        return 0.0, 0.0, 0.0

    xc_u, yc_u = xc[usable], yc[usable]
    fx_u, fy_u = fx[usable], fy[usable]

    # Unknowns [fx0, fy0, kxx/pi, kxy/(2*pi), kyy/pi], two rows per block.
    M = xp.zeros((2 * n, 5), dtype=xp.float64)
    rhs = xp.zeros(2 * n, dtype=xp.float64)
    M[0::2, 0] = 1.0
    M[0::2, 2] = xc_u
    M[0::2, 3] = yc_u
    rhs[0::2] = fx_u
    M[1::2, 1] = 1.0
    M[1::2, 3] = xc_u
    M[1::2, 4] = yc_u
    rhs[1::2] = fy_u
    sol = xp.linalg.lstsq(M, rhs, rcond=None)[0]
    kxx = float(np.pi * float(sol[2]))
    kyy = float(np.pi * float(sol[4]))
    kxy = float(2 * np.pi * float(sol[3]))
    return kxx, kyy, kxy


def remove_carrier(phi: np.ndarray, weight: np.ndarray | None = None,
                   mask: np.ndarray | None = None, refine_iters: int = 10,
                   window: bool = True, defocus: bool = True,
                   n_blocks: int = 10, device: str = "auto") -> CarrierResult:
    """Estimate and remove the spatial carrier of a wrapped phase map.

    Removes the linear ramp ``kx*x + ky*y``, the ``piston``, and, when
    ``defocus`` is set, the curvature ``kxx*x^2 + kyy*y^2 + kxy*x*y`` from

        ``phi = wrap(phase_obj + kx*x + ky*y
                     + kxx*x^2 + kyy*y^2 + kxy*x*y + piston)``

    All estimation runs on ``exp(1j * phi)``, so the map is never unwrapped
    and the number of carrier fringes is unrestricted. The steps are: fit the
    curvature from a block grid (:func:`_estimate_curvature`) and demodulate
    it; fit the remaining tilt (:func:`_estimate_tilt`) and demodulate it;
    remove the piston as the weighted circular mean. See
    ``docs/carrier_removal.md`` §5 for the method and ``docs/gauge_conventions.md``
    §"Carrier removal" for the origin, piston, and frequency conventions.

    Parameters
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase map, in radians, e.g.
        :attr:`phase_shift.result.PhaseResult.phi`.
    weight : np.ndarray, shape (H, W), optional
        Per-pixel reliability, e.g. :attr:`phase_shift.result.PhaseResult.b`.
        Used for estimation only; the returned ``phi`` is computed from the
        unweighted field. Negative values are clipped to 0.
    mask : np.ndarray, shape (H, W), optional
        Pixels where it is falsey are excluded from estimation; combined with
        ``weight``.
    refine_iters : int, default 10
        Maximum refinement iterations after each coarse FFT, for the
        full-field estimate and for every block.
    window : bool, default True
        Apply a 2-D Hann window before each coarse FFT; skipped when the field
        has a size-1 axis.
    defocus : bool, default True
        Also fit and remove the quadratic curvature term.
    n_blocks : int, default 10
        Grid size for the curvature estimate, used only when ``defocus`` is
        set. Needs at least 4 blocks with usable weight; otherwise the
        curvature is skipped with a warning.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Device to run on; see :func:`phase_shift.backend.to_device`.

    Returns
    -------
    CarrierResult
        Flattened phase and the removed carrier, curvature, and piston.

    Raises
    ------
    ValueError
        If ``phi`` is not 2-D, if ``weight`` or ``mask`` does not have
        ``phi``'s shape, if ``n_blocks`` is below 1, or if ``refine_iters`` is
        negative.

    Warns
    -----
    UserWarning
        If the curvature fit is under-determined; see
        :func:`_estimate_curvature`.
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D (H, W), got shape {phi.shape}")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be at least 1, got {n_blocks}")
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be non-negative, got {refine_iters}")

    phi = to_device(phi, device=device)
    xp = get_array_module(phi)
    H, W = phi.shape
    c = xp.exp(1j * phi)
    w = _estimation_weight(phi, weight, mask, device)

    x = xp.arange(W, dtype=xp.float64)[None, :]                          # (1, W)
    y = xp.arange(H, dtype=xp.float64)[:, None]                          # (H, 1)

    kxx = kyy = kxy = 0.0
    if defocus:
        kxx, kyy, kxy = _estimate_curvature(c, w, window, refine_iters, n_blocks)
        if kxx or kyy or kxy:
            c = c * xp.exp(-1j * (kxx * x**2 + kyy * y**2 + kxy * x * y))

    fx, fy = _estimate_tilt(c, w, window, refine_iters)

    d = c * xp.exp(-1j * 2 * xp.pi * (fx * x + fy * y))
    piston = float(xp.angle(xp.sum(w * d)))
    d = d * xp.exp(-1j * piston)

    return CarrierResult(phi=xp.angle(d), kx=2 * np.pi * fx, ky=2 * np.pi * fy,
                         fx=fx, fy=fy, kxx=kxx, kyy=kyy, kxy=kxy, piston=piston)
