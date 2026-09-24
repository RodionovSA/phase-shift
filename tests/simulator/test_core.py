import numpy as np
import pytest

from simulator import simulate


class TestSimulate:
    def test_full_fields_match_formula(self):
        rng = np.random.default_rng(0)
        a, b, psi = (rng.random((4, 5, 6)) for _ in range(3))
        np.testing.assert_allclose(simulate(a, b, psi), a + b * np.cos(psi))

    def test_broadcasts_static_and_per_frame(self):
        N, H, W = 4, 5, 6
        rng = np.random.default_rng(1)
        a = rng.random((1, H, W))
        b = rng.random((N, 1, 1))
        psi = rng.random((N, H, W))
        I = simulate(a, b, psi)
        assert I.shape == (N, H, W)
        np.testing.assert_allclose(I[2], a[0] + b[2, 0, 0] * np.cos(psi[2]))

    @pytest.mark.parametrize("bad", [np.ones((5, 6)), 1.0, np.ones((1, 1, 5, 6))])
    def test_rejects_non_3d(self, bad):
        with pytest.raises(ValueError, match="b must be a 3-D array"):
            simulate(np.ones((1, 5, 6)), bad, np.ones((4, 1, 1)))

    def test_rejects_incompatible_shapes(self):
        with pytest.raises(ValueError, match="do not broadcast"):
            simulate(np.ones((1, 5, 6)), np.ones((4, 1, 1)), np.ones((3, 5, 6)))
