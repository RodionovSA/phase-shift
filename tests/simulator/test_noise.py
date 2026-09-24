import numpy as np
import pytest

from simulator import CameraNoise, GaussianNoise

SHAPE = (50, 40, 50)


class TestGaussianNoise:
    def test_scalar_sigma_statistics(self):
        I = np.full(SHAPE, 3.0)
        model = GaussianNoise(0.2)
        noisy = model(I, rng=0)
        assert noisy.shape == SHAPE
        assert np.mean(noisy - I) == pytest.approx(0.0, abs=0.002)
        assert np.var(noisy - I) == pytest.approx(0.04, rel=0.02)
        np.testing.assert_allclose(model.variance(I), 0.04)

    def test_map_sigma_and_reproducible(self):
        sigma = np.array([0.1, 0.5])[None, None, :].repeat(SHAPE[2] // 2, axis=2)
        I = np.zeros(SHAPE)
        model = GaussianNoise(sigma)
        noisy = model(I, rng=1)
        np.testing.assert_allclose(np.var(noisy, axis=(0, 1)), sigma[0, 0] ** 2, rtol=0.1)
        np.testing.assert_array_equal(noisy, model(I, rng=1))
        np.testing.assert_allclose(model.variance(I), np.broadcast_to(sigma ** 2, SHAPE))

    def test_rejects(self):
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            GaussianNoise(-0.1)
        with pytest.raises(ValueError, match="I must have shape"):
            GaussianNoise(0.1)(np.zeros((4, 4)))


class TestCameraNoise:
    @pytest.mark.parametrize("level", [5.0, 200.0])
    def test_mean_and_variance_match_model(self, level):
        model = CameraNoise(sigma_read=1.5, gain=4.0)
        I = np.full(SHAPE, level)
        noisy = model(I, rng=2)
        expected = level / 4.0 + 1.5 ** 2
        assert np.mean(noisy) == pytest.approx(level, rel=0.005)
        assert np.var(noisy) == pytest.approx(expected, rel=0.02)
        np.testing.assert_allclose(model.variance(I), expected)

    def test_shot_noise_is_quantized_in_electrons(self):
        noisy = CameraNoise(sigma_read=0.0, gain=2.0)(np.full((2, 3, 4), 10.0), rng=3)
        np.testing.assert_allclose(noisy * 2.0, np.round(noisy * 2.0))

    def test_zero_intensity_gives_read_noise_only(self):
        noisy = CameraNoise(sigma_read=0.7, gain=1.0)(np.zeros(SHAPE), rng=4)
        assert np.std(noisy) == pytest.approx(0.7, rel=0.02)

    @pytest.mark.parametrize("kwargs, match", [
        (dict(sigma_read=-1.0, gain=1.0), "sigma_read must be non-negative"),
        (dict(sigma_read=1.0, gain=0.0), "gain must be positive"),
    ])
    def test_rejects_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CameraNoise(**kwargs)

    def test_rejects_negative_intensity(self):
        I = np.ones((2, 3, 4))
        I[0, 0, 0] = -1.0
        with pytest.raises(ValueError, match="I must be non-negative"):
            CameraNoise(1.0, 1.0)(I)
