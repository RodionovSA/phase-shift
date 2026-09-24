# src/simulator/carrier.py
"""Static carrier phase from coefficients in the basis used by carrier removal."""

import numpy as np

from phase_shift.backend import get_array_module, to_device

from .basis import poly_modes


def carrier_map(H: int, W: int, eta: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Carrier ``sum_l eta_l p_l`` without piston, ``docs/carrier_removal.md`` Eq. (1).

    The basis ``p_1, ..., p_L`` is that of §"Basis", so ``eta`` is in the units
    of :attr:`phase_shift.carrier.CarrierResult.eta` without its piston,
    ``eta[1:]``, for a field of the same ``(H, W)``.

    Parameters
    ----------
    H, W : int
        Field height and width.
    eta : array-like, shape (L,)
        Coefficients in radians, ``L = (M+1)(M+2)/2 - 1`` for degree ``M``,
        in ascending degree: ``x, y`` (tilts), then ``x^2, xy, y^2``, and so on.
    device : {"cpu", "cuda", "auto"}, default "cpu"
        Device of the returned map.

    Returns
    -------
    np.ndarray, shape (1, H, W), float64
        Carrier phase in radians, not wrapped.

    Raises
    ------
    ValueError
        If ``eta`` is not 1-D or its length matches no degree.
    """
    eta = to_device(np.asarray(eta, dtype=np.float64), device)
    if eta.ndim != 1:
        raise ValueError(f"eta must be 1-D, got shape {eta.shape}")
    rows = poly_modes(H, W, eta.shape[0], get_array_module(eta))             # (L, H*W)
    return (np.sqrt(H * W) * (eta @ rows)).reshape(1, H, W)
