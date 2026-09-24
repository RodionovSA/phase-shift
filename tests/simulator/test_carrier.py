import numpy as np
import pytest

from phase_shift.carrier import remove_carrier
from simulator import carrier_map

H, W = 64, 80


class TestCarrierMap:
    def test_shape_zero_mean_and_linear_in_eta(self):
        e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 2.0])
        c1, c2 = carrier_map(H, W, e1), carrier_map(H, W, e2)
        assert c1.shape == (1, H, W)
        assert abs(c1.mean()) < 1e-12
        np.testing.assert_allclose(carrier_map(H, W, e1 + e2), c1 + c2, atol=1e-12)

    def test_unit_coefficient_has_unit_mean_square(self):
        for l in range(5):
            eta = np.zeros(5)
            eta[l] = 1.0
            assert np.mean(carrier_map(H, W, eta) ** 2) == pytest.approx(1.0)

    @pytest.mark.parametrize("eta", [[4.0, -3.0], [6.0, -4.0, 2.0, 1.0, -1.5]])
    def test_round_trip_through_carrier_removal(self, eta):
        carrier = carrier_map(H, W, eta)[0]
        degree = {2: 1, 5: 2}[len(eta)]
        result = remove_carrier(carrier, degree=degree, device="cpu", precision="double")
        np.testing.assert_allclose(result.eta[1:], eta, atol=1e-6)
        assert abs(result.piston) < 1e-6

    @pytest.mark.parametrize("eta, match", [
        ([1.0], "number of coefficients must be"), ([1.0] * 3, "number of coefficients must be"),
        (np.ones((1, 2)), "eta must be 1-D"),
    ])
    def test_rejects_bad_eta(self, eta, match):
        with pytest.raises(ValueError, match=match):
            carrier_map(H, W, eta)
