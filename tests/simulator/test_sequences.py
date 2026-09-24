import numpy as np
import pytest

from simulator import linear, random_dips


class TestLinear:
    def test_endpoints_and_shape(self):
        s = linear(5, 1.0, 0.8)
        assert s.shape == (5,)
        np.testing.assert_allclose(s, [1.0, 0.95, 0.9, 0.85, 0.8])


class TestRandomDips:
    def test_reproducible_with_seed(self):
        np.testing.assert_array_equal(random_dips(50, 0.3, 0.5, rng=7),
                                      random_dips(50, 0.3, 0.5, rng=7))

    def test_dips_stay_within_depth_below_base(self):
        s = random_dips(10_000, 0.2, 0.4, rng=0, base=0.9)
        assert np.all(s <= 0.9) and np.all(s >= 0.9 * 0.6)
        assert np.mean(s < 0.9) == pytest.approx(0.2, abs=0.02)

    def test_two_sided_goes_both_ways(self):
        s = random_dips(10_000, 0.5, 0.3, rng=1, two_sided=True)
        assert s.max() > 1.0 and s.min() < 1.0
        assert np.all(np.abs(s - 1.0) <= 0.3)

    @pytest.mark.parametrize("rate, s_expected", [(0.0, 1.0), (1.0, None)])
    def test_rate_limits(self, rate, s_expected):
        s = random_dips(100, rate, 0.5, rng=2)
        if s_expected is not None:
            np.testing.assert_array_equal(s, s_expected)
        else:
            assert np.all(s <= 1.0)

    @pytest.mark.parametrize("kwargs, match", [
        (dict(N=0, rate=0.1, depth=0.1), "N must be positive"),
        (dict(N=5, rate=1.5, depth=0.1), r"rate must lie in \[0, 1\]"),
        (dict(N=5, rate=0.1, depth=-0.1), r"depth must lie in \[0, 1\]"),
    ])
    def test_rejects(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            random_dips(**kwargs)
