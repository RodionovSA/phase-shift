# src/simulator/__init__.py
"""Simulation of phase-shifted interferogram stacks."""

from .beams import fringe_terms
from .carrier import carrier_map
from .core import simulate
from .maps import gaussian_map, masked_map
from .masks import circle, rectangle
from .noise import CameraNoise, GaussianNoise, NoiseModel
from .phase import total_phase
from .scene import Scene
from .separable import beam_intensity, coherence
from .sequences import linear, linear_steps, random_dips, random_steps, uniform_steps
from .step_field import (linear_coeffs, proportional_coeffs, random_coeffs,
                         random_walk_coeffs, step_field)

__all__ = [
    "CameraNoise", "GaussianNoise", "NoiseModel", "Scene", "beam_intensity", "carrier_map", "circle", "coherence", "fringe_terms", "gaussian_map",
    "linear", "linear_coeffs", "linear_steps", "masked_map", "proportional_coeffs",
    "random_coeffs", "random_dips", "random_steps", "random_walk_coeffs", "rectangle",
    "simulate", "step_field", "total_phase", "uniform_steps",
]
