import numpy as np
import pytest

from simulator import total_phase

N, H, W = 4, 5, 6


def _fields(seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    return (rng.random((1, H, W)), rng.random((1, H, W)), rng.random((N, H, W)),
            rng.random((1, H, W)))


class TestTotalPhase:
    def test_sums_components(self):
        phi, carrier, delta, phi_inst = _fields()
        np.testing.assert_allclose(total_phase(phi, carrier, delta), phi + carrier + delta)
        psi = total_phase(phi, carrier, delta, phi_inst)
        assert psi.shape == (N, H, W)
        np.testing.assert_allclose(psi, phi + carrier + delta + phi_inst)

    @pytest.mark.parametrize("name, shape", [
        ("phi", (N, H, W)), ("phi", (H, W)),
        ("carrier", (1, H, W + 1)), ("carrier", (N, H, W)),
        ("delta", (N, 1, 1)), ("delta", (N, H)),
        ("phi_inst", (N, H, W)),
    ])
    def test_rejects_wrong_shape(self, name, shape):
        args = dict(zip(("phi", "carrier", "delta", "phi_inst"), _fields()))
        args[name] = np.zeros(shape)
        with pytest.raises(ValueError, match=f"{name} must have shape"):
            total_phase(**args)
