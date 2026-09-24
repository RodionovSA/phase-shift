import numpy as np
import pytest

from simulator import fringe_terms


class TestFringeTerms:
    def test_matches_formula_and_broadcasts(self):
        N, H, W = 4, 5, 6
        rng = np.random.default_rng(0)
        I1 = rng.random((1, H, W))
        I2 = rng.random((N, 1, 1))
        gamma = rng.random((N, H, W))
        a, b = fringe_terms(I1, I2, gamma)
        assert a.shape == b.shape == (N, H, W)
        np.testing.assert_allclose(a, I1 + I2)
        np.testing.assert_allclose(b, 2 * gamma * np.sqrt(I1 * I2))

    def test_theta_scales_amplitude(self):
        rng = np.random.default_rng(1)
        I1, I2, gamma = (rng.random((3, 5, 6)) for _ in range(3))
        theta = rng.random((1, 1, 1))
        _, b0 = fringe_terms(I1, I2, gamma)
        _, b = fringe_terms(I1, I2, gamma, theta)
        np.testing.assert_allclose(b, b0 * np.cos(theta))

    def test_equal_beams_full_coherence(self):
        ones = np.ones((2, 3, 4))
        a, b = fringe_terms(ones, ones, ones)
        np.testing.assert_allclose(a, 2.0)
        np.testing.assert_allclose(b, 2.0)

    def test_rejects_negative_intensity(self):
        I2 = np.ones((1, 3, 4))
        I2[0, 1, 2] = -0.1
        with pytest.raises(ValueError, match="I2 must be non-negative"):
            fringe_terms(np.ones((1, 3, 4)), I2, np.ones((1, 1, 1)))

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_gamma_out_of_range(self, value):
        with pytest.raises(ValueError, match=r"gamma must lie in \[0, 1\]"):
            fringe_terms(np.ones((1, 3, 4)), np.ones((1, 3, 4)), np.full((2, 1, 1), value))

    def test_rejects_bad_theta_shape(self):
        ones = np.ones((1, 3, 4))
        with pytest.raises(ValueError, match="theta must be a 3-D array"):
            fringe_terms(ones, ones, ones, theta=0.5)
