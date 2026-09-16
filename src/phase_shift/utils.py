# src/phase_shift/utils.py
"""Shared utilities: phase wrapping and value formatting."""

import numpy as np

from .backend import get_array_module


def wrap(x: np.ndarray) -> np.ndarray:
    """Wrap ``x`` (radians) into ``[-pi, pi]``."""
    xp = get_array_module(x)
    two_pi = 2 * xp.pi
    return x - two_pi * xp.round(x / two_pi)


def _wrap_combine(phi_a: np.ndarray, phi_b: np.ndarray, sign: float) -> np.ndarray:
    """Return ``phi_a + sign*phi_b`` wrapped into ``[-pi, pi]``."""
    xp = get_array_module(phi_a, phi_b)
    return xp.angle(xp.exp(1j * phi_a) * xp.exp(1j * sign * phi_b))


def wrap_add(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """Return ``phi_a + phi_b`` (radians) wrapped into ``[-pi, pi]``."""
    return _wrap_combine(phi_a, phi_b, 1.0)


def wrap_sub(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """Return ``phi_a - phi_b`` (radians) wrapped into ``[-pi, pi]``."""
    return _wrap_combine(phi_a, phi_b, -1.0)


def format_value(value: object) -> str:
    """Format ``value`` for printing: floats to 6 significant digits, else ``str``."""
    return f"{value:.6g}" if isinstance(value, float) else str(value)
