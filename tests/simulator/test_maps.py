import numpy as np
import pytest

from simulator import circle, gaussian_map, masked_map, total_phase


class TestGaussianMap:
    def test_peak_and_half_maximum(self):
        g = gaussian_map(21, 41, fwhm=(10, 4), peak=0.8, center=(15, 8))
        assert g.shape == (1, 21, 41)
        assert g[0, 8, 15] == pytest.approx(0.8)
        assert g[0, 8, 20] == pytest.approx(0.4)
        assert g[0, 10, 15] == pytest.approx(0.4)

    def test_default_center_is_field_center(self):
        g = gaussian_map(5, 7, fwhm=(3, 3))
        assert np.unravel_index(np.argmax(g[0]), (5, 7)) == (2, 3)

    def test_rejects_bad_fwhm(self):
        with pytest.raises(ValueError, match="fwhm must be positive"):
            gaussian_map(5, 5, fwhm=(0, 3))


class TestMaskedMap:
    def test_values_inside_and_outside(self):
        m = circle(9, 9, radius=2)
        I = masked_map(m, inside=1.0, outside=0.1)
        assert I.shape == (1, 9, 9) and I.dtype == np.float64
        np.testing.assert_array_equal(I[m], 1.0)
        np.testing.assert_array_equal(I[~m], 0.1)

    def test_phase_map_lower_inside(self):
        H, W, N = 9, 9, 3
        m = circle(H, W, radius=2)
        phi = masked_map(m, inside=-0.7, outside=0.3)
        np.testing.assert_array_equal(phi[m], -0.7)
        zero = np.zeros((1, H, W))
        psi = total_phase(phi, zero, np.zeros((N, H, W)))
        np.testing.assert_array_equal(psi, np.broadcast_to(phi, (N, H, W)))

    @pytest.mark.parametrize("mask, match", [
        (np.ones((1, 3, 3)), "mask must be boolean"),
        (np.ones((3, 3), bool), "mask must have shape"),
    ])
    def test_rejects(self, mask, match):
        with pytest.raises(ValueError, match=match):
            masked_map(mask, 1.0, 0.0)
