# src/simulator/phase.py
"""Total interference phase from its static and time-dependent parts."""

import numpy as np

from .shapes import check_shape


def total_phase(phi: np.ndarray, carrier: np.ndarray, delta: np.ndarray,
                phi_inst: np.ndarray | None = None) -> np.ndarray:
    """Build the total phase of ``docs/interference_model.md`` Eq. (10).

    ``psi = phi + phi_inst + carrier + delta``.

    Parameters
    ----------
    phi : np.ndarray, shape (1, H, W)
        Sample phase in radians.
    carrier : np.ndarray, shape (1, H, W)
        Carrier phase in radians.
    delta : np.ndarray, shape (N, H, W)
        Phase step in radians, per frame and pixel.
    phi_inst : np.ndarray, shape (1, H, W), optional
        Static instrumental phase in radians. None means zero.

    Returns
    -------
    np.ndarray, shape (N, H, W)
        Total phase in radians, not wrapped.

    Raises
    ------
    ValueError
        If an input does not have the shape above.
    """
    check_shape("phi", phi, (1, None, None))
    _, H, W = phi.shape
    check_shape("carrier", carrier, (1, H, W))
    check_shape("delta", delta, (None, H, W))
    psi = phi + carrier + delta
    if phi_inst is not None:
        check_shape("phi_inst", phi_inst, (1, H, W))
        psi = psi + phi_inst
    return psi
