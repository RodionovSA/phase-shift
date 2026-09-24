import numpy as np
import pytest

from simulator import beam_intensity, coherence, fringe_terms

N, H, W = 4, 5, 6


class TestSeparable:
    def test_beam_intensity_scales_map_per_frame(self):
        rng = np.random.default_rng(0)
        I_map, alpha = rng.random((1, H, W)), rng.random(N)
        I = beam_intensity(I_map, alpha)
        assert I.shape == (N, H, W)
        np.testing.assert_allclose(I[2], alpha[2] * I_map[0])

    def test_coherence_scales_map_per_frame(self):
        rng = np.random.default_rng(1)
        gamma_map, g = rng.random((1, H, W)), rng.random(N)
        gamma = coherence(gamma_map, g)
        assert gamma.shape == (N, H, W)
        np.testing.assert_allclose(gamma[3], g[3] * gamma_map[0])

    def test_chain_gives_eq14_terms(self):
        rng = np.random.default_rng(2)
        I1, I2, gamma_map = (rng.random((1, H, W)) for _ in range(3))
        alpha, g = rng.random(N), rng.random(N)
        a, b = fringe_terms(beam_intensity(I1, alpha), beam_intensity(I2, alpha),
                            coherence(gamma_map, g))
        al, gl = alpha[:, None, None], g[:, None, None]
        np.testing.assert_allclose(a, al * (I1 + I2))
        np.testing.assert_allclose(b, al * gl * 2 * gamma_map * np.sqrt(I1 * I2))

    @pytest.mark.parametrize("I_map, alpha, match", [
        (np.ones((N, H, W)), np.ones(N), "I_map must have shape"),
        (np.ones((1, H, W)), np.ones((N, 1, 1)), "alpha must have shape"),
        (np.ones((1, H, W)), np.array([1.0, -0.1]), "alpha must be non-negative"),
    ])
    def test_beam_intensity_rejects(self, I_map, alpha, match):
        with pytest.raises(ValueError, match=match):
            beam_intensity(I_map, alpha)

    @pytest.mark.parametrize("gamma_map, g, match", [
        (np.ones((H, W)), np.ones(N), "gamma_map must have shape"),
        (np.ones((1, H, W)), np.ones((N, 1)), "g must have shape"),
        (np.ones((1, H, W)), np.array([0.5, 1.1]), r"g must lie in \[0, 1\]"),
        (np.ones((1, H, W)), np.array([-0.1, 0.5]), r"g must lie in \[0, 1\]"),
    ])
    def test_coherence_rejects(self, gamma_map, g, match):
        with pytest.raises(ValueError, match=match):
            coherence(gamma_map, g)
