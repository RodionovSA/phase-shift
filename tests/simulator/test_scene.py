from dataclasses import replace

import numpy as np
import pytest

from simulator import (CameraNoise, Scene, carrier_map, circle, gaussian_map, masked_map,
                       random_coeffs, random_dips, step_field, uniform_steps)

H, W, N = 24, 32, 6


def _scene(**kwargs) -> Scene:
    base = dict(
        I1_map=masked_map(circle(H, W, radius=10), inside=100.0, outside=20.0),
        I2_map=np.full((1, H, W), 80.0),
        gamma_map=gaussian_map(H, W, fwhm=(40, 30), peak=0.9),
        phi=masked_map(circle(H, W, radius=6), inside=1.0, outside=0.0),
        piston=uniform_steps(N),
    )
    base.update(kwargs)
    return Scene(**base)


class TestScene:
    def test_matches_eq17(self):
        coeffs = random_coeffs(2, N, rms=2.0, rng=0)
        eta = np.array([3.0, -2.0])
        alpha = np.linspace(1.0, 0.9, N)
        g = random_dips(N, 0.5, 0.4, rng=1)
        s = _scene(coeffs=coeffs, eta=eta, alpha=alpha, g=g)
        a = s.I1_map + s.I2_map
        b = 2 * s.gamma_map * np.sqrt(s.I1_map * s.I2_map)
        psi = s.phi + carrier_map(H, W, eta) + step_field(H, W, s.piston, coeffs)
        al, gl = alpha[:, None, None], g[:, None, None]
        expected = al * (a + gl * b * np.cos(psi))
        np.testing.assert_allclose(s.ideal(), expected, rtol=1e-12)

    def test_defaults_are_noise_free_uniform_piston(self):
        s = _scene()
        a = s.I1_map + s.I2_map
        b = 2 * s.gamma_map * np.sqrt(s.I1_map * s.I2_map)
        expected = a + b * np.cos(s.phi + s.piston[:, None, None])
        np.testing.assert_allclose(s.ideal(), expected, rtol=1e-12)
        np.testing.assert_array_equal(s.stack(rng=0), s.ideal())
        np.testing.assert_array_equal(s.variance(), 0.0)

    def test_seeds_and_generator(self):
        s = _scene(noise=CameraNoise(sigma_read=2.0, gain=0.5))
        np.testing.assert_array_equal(s.stack(rng=3), s.stack(rng=3))
        assert not np.array_equal(s.stack(rng=3), s.stack(rng=4))
        rng = np.random.default_rng(5)
        assert not np.array_equal(s.stack(rng), s.stack(rng))

    def test_variance_from_noise_model(self):
        cam = CameraNoise(sigma_read=2.0, gain=0.5)
        s = _scene(noise=cam)
        np.testing.assert_allclose(s.variance(), cam.variance(s.ideal()))

    def test_ideal_is_a_copy(self):
        s = _scene()
        s.ideal()[:] = 0.0
        assert s.ideal().max() > 0

    def test_replace_changes_one_input(self):
        s = _scene()
        s2 = replace(s, phi=np.zeros((1, H, W)))
        assert not np.allclose(s.ideal(), s2.ideal())
        assert s2.I1_map is s.I1_map

    def test_propagates_layer_validation(self):
        with pytest.raises(ValueError, match="piston must have shape"):
            _scene(piston=np.zeros((N, 1))).ideal()
