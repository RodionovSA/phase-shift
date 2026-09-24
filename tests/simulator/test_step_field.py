import numpy as np
import pytest

from phase_shift.basis import spatial_basis
from simulator import (linear_coeffs, linear_steps, proportional_coeffs, random_coeffs,
                       random_steps, random_walk_coeffs, step_field, total_phase,
                       uniform_steps)

H, W, N = 16, 20, 8


class TestPistonSteps:
    def test_uniform_steps(self):
        np.testing.assert_allclose(uniform_steps(4, offset=0.5),
                                   0.5 + np.array([0, 0.5, 1, 1.5]) * np.pi)

    def test_linear_steps_noise_free_and_noisy(self):
        np.testing.assert_allclose(linear_steps(5, 1.2, offset=0.1), 0.1 + 1.2 * np.arange(5))
        noise = linear_steps(20_000, 1.0, sigma=0.05, rng=0) - np.arange(20_000)
        assert np.std(noise) == pytest.approx(0.05, rel=0.03)

    def test_random_steps_in_period_and_reproducible(self):
        s = random_steps(1000, rng=3)
        assert np.all((s >= 0) & (s < 2 * np.pi))
        np.testing.assert_array_equal(s, random_steps(1000, rng=3))

    def test_rejects(self):
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            linear_steps(5, 1.0, sigma=-1.0)
        with pytest.raises(ValueError, match="N must be positive"):
            uniform_steps(0)


class TestCoeffGenerators:
    @pytest.mark.parametrize("c", [
        random_coeffs(5, N, rms=[1, 2, 3, 4, 5], rng=0),
        random_walk_coeffs(5, N, sigma=0.5, rng=1),
        linear_coeffs(N, [1.0, -2.0, 0.5, 0.0, 3.0]),
        proportional_coeffs(uniform_steps(N), [1.0, -2.0, 0.5, 0.0, 3.0]),
    ])
    def test_shape_and_zero_frame_mean(self, c):
        assert c.shape == (5, N)
        np.testing.assert_allclose(c.mean(axis=1), 0.0, atol=1e-12)

    def test_random_coeffs_scale_per_mode(self):
        c = random_coeffs(2, 20_000, rms=[0.1, 2.0], rng=2)
        np.testing.assert_allclose(c.std(axis=1), [0.1, 2.0], rtol=0.03)

    def test_random_walk_increments(self):
        c = random_walk_coeffs(1, 20_000, sigma=0.3, rng=4)
        assert np.std(np.diff(c[0])) == pytest.approx(0.3, rel=0.03)

    def test_linear_and_proportional_values(self):
        np.testing.assert_allclose(linear_coeffs(3, [2.0]), [[-2.0, 0.0, 2.0]])
        np.testing.assert_allclose(proportional_coeffs(np.array([0.0, 1.0, 5.0]), [2.0]),
                                   [[-4.0, -2.0, 6.0]])

    def test_rejects_bad_scale(self):
        with pytest.raises(ValueError, match="rms must be non-negative"):
            random_coeffs(2, N, rms=[1.0, -1.0])
        with pytest.raises(ValueError, match="sigma must have shape"):
            random_walk_coeffs(2, N, sigma=[1.0, 1.0, 1.0])


class TestStepField:
    def test_matches_vp_aia_reconstruction(self):
        piston = uniform_steps(N)
        coeffs = random_coeffs(5, N, rms=3.0, rng=5)
        field = step_field(H, W, piston, coeffs)
        modes = spatial_basis(H, W, "poly", np, "double", degree=2)
        assert field.shape == (N, H, W)
        np.testing.assert_allclose(field, piston[:, None, None]
                                   + (coeffs.T @ modes).reshape(N, H, W), atol=1e-12)

    def test_piston_is_field_mean_and_coeff_is_norm(self):
        piston = uniform_steps(N)
        coeffs = random_coeffs(2, N, rms=3.0, rng=6)
        field = step_field(H, W, piston, coeffs)
        np.testing.assert_allclose(field.mean(axis=(1, 2)), piston, atol=1e-12)
        Delta = field - piston[:, None, None]
        np.testing.assert_allclose(np.sqrt((Delta ** 2).sum(axis=(1, 2))),
                                   np.linalg.norm(coeffs, axis=0), atol=1e-10)

    def test_no_modes_gives_uniform_piston(self):
        piston = linear_steps(N, 0.7)
        field = step_field(H, W, piston, np.zeros((0, N)))
        np.testing.assert_allclose(field, np.broadcast_to(piston[:, None, None], (N, H, W)))

    def test_feeds_total_phase(self):
        field = step_field(H, W, uniform_steps(N), random_coeffs(2, N, rms=1.0, rng=7))
        zero = np.zeros((1, H, W))
        np.testing.assert_allclose(total_phase(zero, zero, field), field)

    @pytest.mark.parametrize("coeffs, match", [
        (np.zeros((3, N)), "number of coefficients must be"),
        (np.zeros((2, N + 1)), "coeffs must have shape"),
    ])
    def test_rejects(self, coeffs, match):
        with pytest.raises(ValueError, match=match):
            step_field(H, W, uniform_steps(N), coeffs)
