# src/phase_shift/config.py
"""Phase extraction configuration and its YAML serialization."""

import os
from dataclasses import dataclass, field

import numpy as np
import yaml

from .methods import METHODS

GAIN_MODES = ("none", "joint")


@dataclass
class PhaseConfig:
    """Configuration for :class:`phase_shift.solver.PhaseSolver`.

    Attributes
    ----------
    use_alpha : bool, default True
        Estimate and divide out the per-frame source-power factor ``alpha_n``
        (``docs/aia.md`` §"Starting point"). If False, ``alpha_n = 1``.
    gain_mode : {"none", "joint"}, default "joint"
        ``"joint"`` fits the per-frame fringe gain ``g_n`` together with the
        phase steps; ``"none"`` fixes ``g_n = 1``. Ignored when ``g`` is given.
    g : np.ndarray, shape (N,), optional
        Fixed per-frame fringe gain, e.g. from
        :func:`phase_shift.frame_contrast.measure_frame_contrast`.
    method : str, default "aia"
        Registered method name, one of :data:`phase_shift.METHODS`;
        case-insensitive.
    method_kwargs : dict, default {}
        Keyword arguments passed to the method, e.g. ``{"iters": 50}`` for
        ``"aia"``.
    noise_std : np.ndarray, shape (H, W), optional
        Per-pixel camera noise standard deviation ``sigma_0``,
        ``docs/aia_noise.md`` Eq. (16). Defaults to the per-pixel RMS fit residual.
    phi_error_simplified : bool, default True
        Compute ``PhaseResult.phi_error`` for known steps and gains,
        ``docs/aia_noise.md`` Eq. (10). If False, add the exact correction for the
        steps and gains the method fitted, ``docs/aia_noise.md`` Eq. (24) or (29),
        and for ``"vp_aia"`` the step field's own term, ``docs/vp_aia.md``
        Eq. (19). The correction is ``O(1/N_p)``, so it matters on a small
        region and not on a full frame.

    Raises
    ------
    ValueError
        If ``method`` or ``gain_mode`` is not recognized.
    """

    use_alpha: bool = True
    gain_mode: str = "joint"
    g: np.ndarray | None = None
    method: str = "aia"
    method_kwargs: dict = field(default_factory=dict)
    noise_std: np.ndarray | None = None
    phi_error_simplified: bool = True

    def __post_init__(self) -> None:
        if self.method.lower() not in METHODS:
            raise ValueError(f"unknown method {self.method!r}, expected one of {METHODS}")
        if self.gain_mode not in GAIN_MODES:
            raise ValueError(
                f"unknown gain_mode {self.gain_mode!r}, expected one of {GAIN_MODES}"
            )

    def to_dict(self) -> dict:
        """Return this configuration as a plain dict.

        Arrays become lists; ``method_kwargs`` is included as is.

        Returns
        -------
        dict
            One key per field, in declaration order.
        """
        data = {
            "use_alpha": self.use_alpha,
            "gain_mode": self.gain_mode,
            "g": self.g.tolist() if self.g is not None else None,
            "method": self.method,
            "method_kwargs": self.method_kwargs,
            "noise_std": self.noise_std.tolist() if self.noise_std is not None else None,
            "phi_error_simplified": self.phi_error_simplified,
        }

        return data

    def to_yaml(self, path: str | os.PathLike) -> None:
        """Save this configuration as a YAML file.

        ``method_kwargs`` values must be plain data (numbers, strings, lists,
        dicts); arrays are not supported there.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path.
        """
        data = self.to_dict()
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "PhaseConfig":
        """Load a configuration written by :meth:`to_yaml`.

        Missing keys take their default values.

        Parameters
        ----------
        path : str or os.PathLike
            Source file path.

        Returns
        -------
        PhaseConfig

        Raises
        ------
        TypeError
            If the file contains an unknown key.
        ValueError
            If ``method`` or ``gain_mode`` is not recognized.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if data.get("g") is not None:
            data["g"] = np.asarray(data["g"], dtype=float)
        if data.get("noise_std") is not None:
            data["noise_std"] = np.asarray(data["noise_std"], dtype=float)
        return cls(**data)
