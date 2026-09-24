# src/simulator/scene.py
"""Interferogram stack of ``docs/interference_model.md`` Eq. (17) from prebuilt parts."""

from dataclasses import dataclass
from functools import cached_property
from types import ModuleType

import numpy as np

from phase_shift.backend import get_array_module

from .beams import fringe_terms
from .carrier import carrier_map
from .core import simulate
from .noise import NoiseModel
from .phase import total_phase
from .separable import beam_intensity, coherence
from .step_field import step_field


@dataclass(frozen=True)
class Scene:
    """All inputs of one acquisition; chains the simulator layers.

    ``I_n = alpha_n [a + g_n b cos(phi + carrier + delta_n + Delta_n)]`` plus
    noise. ``H, W`` come from ``phi`` and ``N`` from ``piston``. Change one
    input with :func:`dataclasses.replace`.

    Attributes
    ----------
    I1_map, I2_map : np.ndarray, shape (1, H, W)
        Static beam intensities, in DN when used with
        :class:`simulator.CameraNoise`.
    gamma_map : np.ndarray, shape (1, H, W)
        Static coherence envelope.
    phi : np.ndarray, shape (1, H, W)
        Sample phase in radians.
    piston : np.ndarray, shape (N,)
        Piston step in radians.
    coeffs : np.ndarray, shape (J, N), optional
        Step-field coefficients, see :func:`simulator.step_field`. None means
        a uniform piston step.
    carrier : np.ndarray, shape (L,) or (1, H, W), optional
        Carrier as coefficients for :func:`simulator.carrier_map`, or as a
        phase map in radians. None means no carrier.
    alpha, g : np.ndarray, shape (N,), optional
        Source-power and contrast factors per frame. None means ones.
    noise : NoiseModel, optional
        Camera noise. None means a noise-free stack.
    """

    I1_map: np.ndarray
    I2_map: np.ndarray
    gamma_map: np.ndarray
    phi: np.ndarray
    piston: np.ndarray
    coeffs: np.ndarray | None = None
    carrier: np.ndarray | None = None
    alpha: np.ndarray | None = None
    g: np.ndarray | None = None
    noise: NoiseModel | None = None

    @cached_property
    def _ideal(self) -> np.ndarray:
        """Noise-free stack, computed once."""
        xp = get_array_module(self.phi, self.piston)
        N = self.piston.shape[0]
        _, H, W = self.phi.shape
        ones = xp.ones(N)
        alpha = ones if self.alpha is None else self.alpha
        g = ones if self.g is None else self.g
        a, b = fringe_terms(beam_intensity(self.I1_map, alpha),
                            beam_intensity(self.I2_map, alpha),
                            coherence(self.gamma_map, g))
        coeffs = xp.zeros((0, N)) if self.coeffs is None else self.coeffs
        psi = total_phase(self.phi, self._carrier_map(xp, H, W),
                          step_field(H, W, self.piston, coeffs))
        return simulate(a, b, psi)

    def _carrier_map(self, xp: ModuleType, H: int, W: int) -> np.ndarray:
        """Carrier as a ``(1, H, W)`` map, from coefficients if given as 1-D.

        Raises
        ------
        ValueError
            If ``carrier`` is neither 1-D nor 3-D.
        """
        if self.carrier is None:
            return xp.zeros((1, H, W))
        ndim = np.ndim(self.carrier)
        if ndim == 1:
            return carrier_map(H, W, self.carrier, device="cpu" if xp is np else "cuda")
        if ndim == 3:
            return self.carrier
        raise ValueError(f"carrier must be coefficients (L,) or a map (1, H, W), "
                         f"got {ndim}-D")

    def ideal(self) -> np.ndarray:
        """Noise-free stack.

        Returns
        -------
        np.ndarray, shape (N, H, W)

        Raises
        ------
        ValueError
            If an input has the wrong shape or an out-of-range value.
        """
        return self._ideal.copy()

    def stack(self, rng: int | np.random.Generator | None = None) -> np.ndarray:
        """Stack with one noise realization drawn from ``rng``.

        Parameters
        ----------
        rng : int or np.random.Generator, optional
            Seed or generator; a different seed, or repeated calls with one
            generator, give different realizations.

        Returns
        -------
        np.ndarray, shape (N, H, W)

        Raises
        ------
        ValueError
            If an input has the wrong shape or an out-of-range value.
        """
        if self.noise is None:
            return self.ideal()
        return self.noise(self._ideal, rng)

    def variance(self) -> np.ndarray:
        """Noise variance per pixel and frame, ``docs/aia_noise.md`` Eq. (1).

        Returns
        -------
        np.ndarray, shape (N, H, W)
            Zeros without a noise model.
        """
        if self.noise is None:
            return get_array_module(self._ideal).zeros_like(self._ideal)
        return self.noise.variance(self._ideal)
