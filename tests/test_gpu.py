# tests/test_gpu.py
"""``device="cuda"`` against ``device="cpu"`` on the same inputs; skipped without CuPy."""

import numpy as np
import pytest

from phase_shift import (PhaseConfig, PhaseSolver, asnumpy, combine_acquisitions,
                         remove_carrier, subtract_reference)
from phase_shift.backend import get_array_module, to_device, weighted_gram
from phase_shift.utils import wrap

pytestmark = pytest.mark.gpu


def _fringes(H: int = 64, W: int = 80, seed: int = 0) -> np.ndarray:
    """Wrapped tilt + curvature phase map."""
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    rng = np.random.default_rng(seed)
    phi = 0.31 * X - 0.19 * Y + 4e-4 * X**2 + 0.02 * rng.standard_normal((H, W))
    return np.angle(np.exp(1j * phi))


def _stack(N: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = _fringes(seed=seed)
    delta = np.sort(rng.uniform(0, 2 * np.pi, N))
    g = 0.8 + 0.4 * rng.random(N)
    stack = 2.0 + g[:, None, None] * np.cos(phi[None] + delta[:, None, None])
    return stack + 0.01 * rng.standard_normal(stack.shape)


def _max_phase_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(wrap(asnumpy(a).astype(np.float64) - asnumpy(b))).max())


@pytest.mark.parametrize("method,precision,tol", [("aia", "double", 1e-8),
                                                  ("aia", "single", 1e-3),
                                                  ("vp_aia", "double", 1e-8)])
def test_solver_matches_cpu(method: str, precision: str, tol: float) -> None:
    stack = _stack()
    kw = {"iters": 40} if method == "aia" else {"iters": 40, "crop": 5}
    cfg = PhaseConfig(method=method, method_kwargs=kw)
    cpu = PhaseSolver(cfg, device="cpu", precision=precision).fit(stack).result
    gpu = PhaseSolver(cfg, device="cuda", precision=precision).fit(stack).result
    assert get_array_module(gpu.phi) is not np
    assert gpu.phi.dtype == cpu.phi.dtype
    assert _max_phase_diff(gpu.phi, cpu.phi) < tol
    np.testing.assert_allclose(asnumpy(gpu.delta), cpu.delta, atol=tol)
    np.testing.assert_allclose(asnumpy(gpu.g), cpu.g, rtol=tol)
    np.testing.assert_allclose(asnumpy(gpu.b), cpu.b, rtol=tol)
    np.testing.assert_allclose(asnumpy(gpu.phi_error), cpu.phi_error, rtol=tol)


@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-12), (np.float32, 1e-4)])
@pytest.mark.parametrize("M,K", [(10, 5), (5, 5), (0, 5), (10, 0)])
def test_weighted_gram_matches_cpu(dtype: type, rtol: float, M: int, K: int) -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, 100_000)).astype(dtype)
    w = rng.standard_normal(100_000).astype(dtype)
    B = rng.standard_normal((K, 100_000)).astype(dtype)
    G = weighted_gram(*(to_device(x, "cuda") for x in (A, w, B)))
    assert get_array_module(G) is not np
    assert G.shape == (M, K) and G.dtype == dtype
    ref = (A.astype(np.float64) * w) @ B.T.astype(np.float64)
    np.testing.assert_allclose(asnumpy(G), ref, rtol=rtol, atol=rtol * np.abs(ref).max(initial=0))


def test_remove_carrier_matches_cpu() -> None:
    phi = _fringes()
    cpu = remove_carrier(phi, device="cpu", precision="double")
    gpu = remove_carrier(phi, device="cuda", precision="double")
    assert _max_phase_diff(gpu.phi, cpu.phi) < 1e-8
    np.testing.assert_allclose(asnumpy(gpu.eta), cpu.eta, atol=1e-10)


def test_subtract_reference_matches_cpu() -> None:
    phi, phi_ref = _fringes(seed=1), -_fringes(seed=2)
    cpu = subtract_reference(phi, phi_ref, device="cpu")
    gpu = subtract_reference(phi, phi_ref, device="cuda")
    assert gpu.sign == cpu.sign == -1
    assert _max_phase_diff(gpu.phi, cpu.phi) < 1e-10


def test_combine_acquisitions_matches_cpu() -> None:
    phis = [s * _fringes(seed=k) for k, s in enumerate((1, -1, 1, -1))]
    cpu = combine_acquisitions(phis, device="cpu")
    gpu = combine_acquisitions(phis, device="cuda")
    assert gpu.sign_flips == cpu.sign_flips
    assert _max_phase_diff(gpu.phi, cpu.phi) < 1e-8
