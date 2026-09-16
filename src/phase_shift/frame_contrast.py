# src/phase_shift/frame_contrast.py
"""Per-frame fringe contrast and visibility estimation."""

import warnings

import numpy as np
from numpy.typing import DTypeLike

from .backend import default_dtype, get_array_module


def _carrier_dc_amplitudes(stack: np.ndarray, dc_radius: int = 8,
                           halfwin: tuple[int, int] = (3, 4), frame_chunk: int = 8,
                           dtype: DTypeLike = None) -> tuple[np.ndarray, np.ndarray]:
    """Return each frame's carrier-peak and DC amplitudes from its 2-D spectrum.

    The carrier peak is located once, in the frame-summed Hann-windowed
    spectrum. A frame's carrier amplitude is the root of ``|F|^2`` summed over
    ``halfwin`` around that peak; its DC amplitude is ``|F[0, 0]|``.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Interferogram frames.
    dc_radius : int, default 8
        Half-size, in FFT bins, of the region around DC excluded from the peak
        search.
    halfwin : (int, int), default (3, 4)
        Half-size, in FFT bins along (row, column), of the window summed around
        the peak.
    frame_chunk : int, default 8
        Number of frames transformed at once; bounds memory, not the result.
    dtype : dtype, optional
        Working dtype for the FFT input. Defaults to
        :func:`phase_shift.backend.default_dtype`.

    Returns
    -------
    carrier_amp, dc_amp : np.ndarray, shape (N,)
        Unnormalized carrier-peak and DC amplitudes, float64.

    Warns
    -----
    UserWarning
        If the located peak lies on the ``dc_radius`` boundary, making the
        amplitudes unreliable.
    """
    xp = get_array_module(stack)
    work_dtype = dtype if dtype is not None else default_dtype(xp)
    N, H, W = stack.shape
    win = (xp.outer(xp.hanning(H), xp.hanning(W)) if H > 1 and W > 1
           else xp.ones((H, W))).astype(work_dtype)
    Wc = W // 2 + 1

    # Locate the carrier peak in the frame-summed spectrum.
    Psum = xp.zeros((H, Wc), dtype=xp.float64)
    for s in range(0, N, frame_chunk):
        block = stack[s:s + frame_chunk].astype(work_dtype) * win
        Psum += xp.abs(xp.fft.rfft2(block, axes=(1, 2))).astype(xp.float64).sum(0)
    Psum[:dc_radius, :dc_radius] = 0
    Psum[-dc_radius:, :dc_radius] = 0
    iy, ix = xp.unravel_index(xp.argmax(Psum), Psum.shape)
    iy, ix = int(iy), int(ix)

    if ix <= dc_radius and (iy <= dc_radius or iy >= H - dc_radius - 1):
        warnings.warn(
            f"frame_contrast: carrier peak at (row={iy}, col={ix}) lies on the "
            f"dc_radius={dc_radius} exclusion boundary; amplitudes are unreliable. "
            f"Use a smaller dc_radius or verify the carrier location.",
            stacklevel=3,
        )

    hy, hx = halfwin
    rows = xp.asarray([(iy + k) % H for k in range(-hy, hy + 1)])
    c0, c1 = max(ix - hx, 0), min(ix + hx + 1, Wc)

    # Carrier and DC amplitude of each frame.
    amp_sq = xp.empty(N, dtype=xp.float64)
    dc_amp = xp.empty(N, dtype=xp.float64)
    for s in range(0, N, frame_chunk):
        block = stack[s:s + frame_chunk].astype(work_dtype) * win
        Fc = xp.fft.rfft2(block, axes=(1, 2))
        amp_sq[s:s + block.shape[0]] = (
            xp.abs(Fc[:, rows, :][:, :, c0:c1]).astype(xp.float64) ** 2
        ).sum(axis=(1, 2))
        dc_amp[s:s + block.shape[0]] = xp.abs(Fc[:, 0, 0]).astype(xp.float64)

    return xp.sqrt(amp_sq), dc_amp


def measure_frame_contrast(stack: np.ndarray, dc_radius: int = 8,
                           halfwin: tuple[int, int] = (3, 4), frame_chunk: int = 8,
                           dtype: DTypeLike = None) -> np.ndarray:
    """Measure each frame's fringe gain ``g_n`` from its spatial carrier.

    ``g_n`` is defined in ``docs/interference_model.md`` Eq. (17). Requires a
    linear spatial carrier; fringes without one (e.g. circular) give
    unreliable results.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Interferogram frames.
    dc_radius : int, default 8
        Half-size, in FFT bins, of the region around DC excluded from the peak
        search.
    halfwin : (int, int), default (3, 4)
        Half-size, in FFT bins along (row, column), of the window summed around
        the peak.
    frame_chunk : int, default 8
        Number of frames transformed at once; bounds memory, not the result.
    dtype : dtype, optional
        Working dtype for the FFT input. Defaults to
        :func:`phase_shift.backend.default_dtype`.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-frame gain, normalized to ``median(g) = 1`` within this stack.

    Warns
    -----
    UserWarning
        If the carrier peak lies on the ``dc_radius`` boundary.
    """
    xp = get_array_module(stack)
    amp, _ = _carrier_dc_amplitudes(stack, dc_radius, halfwin, frame_chunk, dtype)
    return amp / xp.median(amp)


def measure_frame_visibility(stack: np.ndarray, dc_radius: int = 8,
                             halfwin: tuple[int, int] = (3, 4), frame_chunk: int = 8,
                             dtype: DTypeLike = None) -> np.ndarray:
    """Measure each frame's fringe visibility from its spatial carrier.

    Returns ``2 * carrier_amp / dc_amp``, proportional to the visibility
    ``g_n * b / a`` of ``docs/interference_model.md`` Eq. (20). The factor
    depends on the setup and region of interest, so values are comparable
    across stacks from the same setup and region, unlike
    :func:`measure_frame_contrast`.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Interferogram frames.
    dc_radius : int, default 8
        Half-size, in FFT bins, of the region around DC excluded from the peak
        search.
    halfwin : (int, int), default (3, 4)
        Half-size, in FFT bins along (row, column), of the window summed around
        the peak.
    frame_chunk : int, default 8
        Number of frames transformed at once; bounds memory, not the result.
    dtype : dtype, optional
        Working dtype for the FFT input. Defaults to
        :func:`phase_shift.backend.default_dtype`.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-frame visibility, unnormalized.

    Warns
    -----
    UserWarning
        If the carrier peak lies on the ``dc_radius`` boundary.
    """
    xp = get_array_module(stack)
    amp, dc_amp = _carrier_dc_amplitudes(stack, dc_radius, halfwin, frame_chunk, dtype)
    dc_amp = xp.where(dc_amp > 0, dc_amp, xp.asarray(xp.finfo(xp.float64).eps))
    return 2.0 * amp / dc_amp


def frame_visibility_from_fit(g: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Return each frame's fringe visibility from fitted model fields.

    Computes ``g_n * median(b / a)``, the visibility of
    ``docs/interference_model.md`` Eq. (20).

    Parameters
    ----------
    g : np.ndarray, shape (N,)
        Per-frame fringe gain.
    b : np.ndarray, shape (H, W)
        Fringe amplitude map.
    a : np.ndarray, shape (H, W)
        Background intensity map.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-frame visibility, float64.
    """
    xp = get_array_module(g, b, a)
    a_floor = xp.maximum(a, xp.asarray(xp.finfo(xp.float64).eps))
    ratio = float(xp.median(b / a_floor))
    return xp.asarray(g, dtype=xp.float64) * ratio
