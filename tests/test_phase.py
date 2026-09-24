"""Fast unit tests for `phase_shift`, on small synthetic stacks (no data/ files needed).

These exist to catch regressions in seconds during the CPU<->GPU backend
work (see src/phase_shift/backend.py) without loading the ~1.5 GB real acquisitions in
data/ -- run with real data (scripts/test/*.ipynb) remains the authority on
physical correctness; these check the math and the numpy/cupy dispatch.
"""

import numpy as np
import pytest

from phase_shift import (
    PhaseConfig,
    PhaseSolver,
    combine_acquisitions,
    measure_frame_contrast,
    remove_carrier,
    subtract_reference,
)
from phase_shift import METHODS
from phase_shift.backend import CUPY_AVAILABLE, Precision, get_precision, set_precision
from phase_shift.basis import BASES, spatial_basis
from phase_shift.methods import MethodParam
from phase_shift.methods.gauge import (center_offsets, normalize_gain,
                                       normalize_quadrature_frame, pin_phase_origin,
                                       whiten_uv, whitening_matrix)
from phase_shift.errors import aia_phi_error_parts, vp_phi_error
from phase_shift.methods.steps import pixel_step
from phase_shift.methods.step_field import step_field_quality
from phase_shift.methods.vp_system import fit_frame_and_coeffs, max_modes
from phase_shift.utils import wrap


def circ_rms_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Wrapped RMS difference between two phase arrays, in degrees."""
    return float(np.degrees(np.sqrt(np.mean(wrap(np.asarray(a) - np.asarray(b)) ** 2))))


def make_stack(H=48, W=64, N=12, seed=0, sign=1.0, dtype=np.float32):
    """Synthetic phase-shifted interferogram stack with known ground truth.

    Phase steps are irregular (not evenly spaced) and per-frame contrast
    varies, so this exercises AIA's blind estimation the way real data does
    rather than the trivial evenly-spaced/unit-gain case.
    """
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    phi_true = sign * np.angle(np.exp(1j * (0.31 * X + 0.19 * Y + 0.4 * np.sin(X / 11.0))))
    b_true = 1.0 + 0.3 * rng.random((H, W))
    a_true = 2.0 + 0.2 * rng.random((H, W))
    delta_true = np.sort(rng.uniform(0, 2 * np.pi, N))
    delta_true -= delta_true[0]
    g_true = 0.6 + 0.8 * rng.random(N)

    stack = np.empty((N, H, W), dtype=np.float64)
    for n in range(N):
        stack[n] = a_true + g_true[n] * b_true * np.cos(sign * phi_true + delta_true[n])
    stack += 0.01 * rng.standard_normal(stack.shape)
    return stack.astype(dtype), dict(phi=phi_true, b=b_true, a=a_true, delta=delta_true, g=g_true)


class TestPhaseConfig:
    def test_bad_gain_mode_raises(self):
        with pytest.raises(ValueError):
            PhaseConfig(gain_mode="fft")

    def test_from_yaml_rejects_unknown_keys(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("use_g: false\ndc_radius: 8\n")
        with pytest.raises(TypeError):
            PhaseConfig.from_yaml(p)

    def test_yaml_roundtrip(self, tmp_path):
        p = tmp_path / "config.yaml"
        cfg = PhaseConfig(use_alpha=False, gain_mode="none", method="vp_aia",
                           method_kwargs={"iters": 5})
        cfg.to_yaml(p)
        cfg2 = PhaseConfig.from_yaml(p)
        assert cfg2.use_alpha is False
        assert cfg2.gain_mode == "none"
        assert cfg2.method == "vp_aia"
        assert cfg2.method_kwargs == {"iters": 5}


class TestPrecision:
    def test_presets(self):
        assert Precision.of("single") == Precision(np.float32, np.float64)
        assert Precision.of("double") == Precision(np.float64, np.float64)
        assert Precision.of("fast") == Precision(np.float32, np.float32)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            Precision.of("half")

    def test_default_is_single(self):
        assert get_precision() == Precision.of("single")

    def test_set_precision_applies_to_new_solvers_only(self):
        stack, truth = make_stack()
        solver_before = PhaseSolver(PhaseConfig(gain_mode="none"))
        try:
            set_precision("double")
            assert get_precision() == Precision.of("double")
            # Resolved at construction, so the earlier solver is unaffected.
            assert solver_before.precision == Precision.of("single")
            assert PhaseSolver(PhaseConfig()).precision == Precision.of("double")
            assert solver_before.fit(stack).result.phi.dtype == np.float32
        finally:
            set_precision("single")

    def test_result_records_precision_and_field_dtypes(self):
        stack, truth = make_stack()
        for name, work in (("single", np.float32), ("double", np.float64),
                           ("fast", np.float32)):
            for method in METHODS:
                kw = {"iters": 10} if method == "aia" else {"iters": 10, "crop": 5}
                gain = "joint" if method == "vp_aia" else "none"
                cfg = PhaseConfig(method=method, gain_mode=gain, method_kwargs=kw)
                r = PhaseSolver(cfg, precision=name).fit(stack).result
                assert r.precision == Precision.of(name), method
                assert r.phi.dtype == r.a.dtype == r.b.dtype == work
                assert r.phi_error.dtype == work
                # (N,) vectors stay float64 whatever the precision.
                assert r.delta.dtype == r.g.dtype == r.alpha.dtype == np.float64

    def test_explicit_precision_instance_accepted(self):
        stack, truth = make_stack()
        p = Precision(np.float64, np.float64)
        r = PhaseSolver(PhaseConfig(gain_mode="none"), precision=p).fit(stack).result
        assert r.precision == p
        assert r.phi.dtype == np.float64


class TestAIA:
    def test_recovers_known_phase(self):
        stack, truth = make_stack()
        # supply the exact known gain (PhaseConfig(g=...)) to isolate the
        # AIA solve's accuracy from gain-estimation accuracy.
        solver = PhaseSolver(PhaseConfig(g=truth["g"])).fit(stack)
        assert solver.result.method_param.converged
        # aia has an exact (phi, delta) -> (-phi, -delta) sign ambiguity
        # (I_n = a + b*cos(phi+delta_n) is invariant under it) -- accept
        # either branch.
        err_same = circ_rms_deg(solver.result.phi, truth["phi"])
        err_flip = circ_rms_deg(solver.result.phi, -truth["phi"])
        assert min(err_same, err_flip) < 0.5

    def test_single_close_to_double(self):
        stack, truth = make_stack()
        # gain_mode="none" (fixed g=1) isolates precision effects in the solve
        # itself from any precision-dependent variation in gain estimation.
        config = PhaseConfig(gain_mode="none")
        r64 = PhaseSolver(config, precision="double").fit(stack)
        r32 = PhaseSolver(config, precision="single").fit(stack)
        assert circ_rms_deg(r32.result.phi, r64.result.phi) < 1e-2
        assert r32.result.method_param.iters_run == r64.result.method_param.iters_run
        assert r32.result.method_param.converged == r64.result.method_param.converged

    def test_fast_close_to_single(self):
        # "fast" differs from "single" only in the dtype frame_step's
        # stack-scale reduction runs in (src/phase_shift/methods/steps.py) --
        # the recovered phase should be indistinguishable at the same
        # threshold test_single_close_to_double uses for a genuine work-dtype
        # change.
        stack, truth = make_stack()
        config = PhaseConfig(gain_mode="none")
        r_single = PhaseSolver(config, precision="single").fit(stack)
        r_fast = PhaseSolver(config, precision="fast").fit(stack)
        assert circ_rms_deg(r_fast.result.phi, r_single.result.phi) < 1e-2
        assert r_fast.result.method_param.iters_run == r_single.result.method_param.iters_run
        assert r_fast.result.method_param.converged == r_single.result.method_param.converged

    def test_gain_auto_matches_supplied_gain_ranking(self):
        stack, truth = make_stack(seed=1)
        g = measure_frame_contrast(stack)
        # recovered gain should correlate strongly with the true per-frame
        # contrast (both normalized to median 1)
        g_true_norm = truth["g"] / np.median(truth["g"])
        assert np.corrcoef(g, g_true_norm)[0, 1] > 0.9

    def test_fit_gain_false_returns_g_unchanged(self):
        stack, truth = make_stack(seed=6)
        from phase_shift.methods.aia import aia
        a, b, phi, delta, g_out, mp = aia(stack, truth["g"], fit_gain=False)
        assert np.array_equal(g_out, truth["g"])
        assert np.all(mp.c_fit == 0)

    def test_joint_gain_recovers_true_gain(self):
        # make_stack's g_true already varies frame-to-frame (see its
        # docstring) -- gain_mode="joint" should recover it from the data
        # alone, without measure_frame_contrast's carrier-peak assumption.
        stack, truth = make_stack(seed=8)
        solver = PhaseSolver(PhaseConfig(gain_mode="joint")).fit(stack)
        g_true_norm = truth["g"] / np.median(truth["g"])
        assert np.corrcoef(solver.result.g, g_true_norm)[0, 1] > 0.99

    def test_joint_gain_improves_phase_accuracy_under_gain_drift(self):
        stack, truth = make_stack(seed=9)
        r_none = PhaseSolver(PhaseConfig(gain_mode="none")).fit(stack)
        r_joint = PhaseSolver(PhaseConfig(gain_mode="joint")).fit(stack)
        err_none = min(circ_rms_deg(r_none.result.phi, truth["phi"]),
                        circ_rms_deg(r_none.result.phi, -truth["phi"]))
        err_joint = min(circ_rms_deg(r_joint.result.phi, truth["phi"]),
                         circ_rms_deg(r_joint.result.phi, -truth["phi"]))
        assert err_joint < err_none

    def test_joint_gain_iteration_is_monotone(self):
        """With fit_gain, the pixel and frame steps must minimize the same
        joint residual -- pin this by replicating aia()'s own loop body
        with its public building blocks and checking the residual never
        increases round over round (the c_n-consistency fix, see aia.py).
        """
        from phase_shift.methods.steps import frame_step, pixel_step

        stack, _ = make_stack(seed=10, N=14)
        N = stack.shape[0]
        I = stack.reshape(N, -1).astype(np.float64)

        def joint_cost(a, u, v, c, delta, g):
            model = (a[None, :] + c[:, None]
                      + g[:, None] * (np.outer(np.cos(delta), u) + np.outer(np.sin(delta), v)))
            return float(np.sqrt(np.mean((I - model) ** 2)))

        delta = np.arange(N) * 2 * np.pi / N
        g = np.ones(N)
        c = np.zeros(N)
        costs = []
        for _ in range(20):
            a, u, v = pixel_step(I - c[:, None], delta, g)
            costs.append(joint_cost(a, u, v, c, delta, g))
            new_delta, new_g, new_c = frame_step(I, u, v)
            delta = new_delta - new_delta[0]
            c = new_c - new_c.mean()
            g = new_g / np.median(new_g)

        assert all(costs[i + 1] <= costs[i] + 1e-9 for i in range(len(costs) - 1))

    def test_device_cuda_without_cupy_raises(self):
        if CUPY_AVAILABLE:
            pytest.skip("cupy is installed in this environment")
        stack, _ = make_stack(H=8, W=8, N=6)
        with pytest.raises(RuntimeError):
            PhaseSolver(PhaseConfig(), device="cuda").fit(stack)

    def test_bad_device_raises(self):
        stack, _ = make_stack(H=8, W=8, N=6)
        with pytest.raises(ValueError):
            PhaseSolver(PhaseConfig(), device="tpu").fit(stack)


class TestStepField:
    def test_poly_basis_orthonormal_and_zero_mean(self):
        basis = spatial_basis(20, 24, "poly", np, degree=2)
        assert basis.shape[0] == 5   # x, y, x^2, xy, y^2
        assert np.allclose(basis @ basis.T, np.eye(5), atol=1e-8)
        assert np.allclose(basis.mean(axis=1), 0, atol=1e-10)

    def test_step_field_quality_resid_dtype_is_accum(self):
        # step_field_quality's own regression test: resid must come out in
        # precision.accum, even though none of a/u/v/basis/coeffs/delta
        # arrive pre-cast to it (basis in particular is always float64 from
        # spatial_basis).
        rng = np.random.default_rng(2)
        N, H, W = 10, 40, 50
        P = H * W
        stack = (rng.random((N, P)).astype(np.float32) + 1)
        a = rng.random(P).astype(np.float32)
        u = rng.random(P).astype(np.float32)
        v = rng.random(P).astype(np.float32)
        delta = np.sort(rng.uniform(0, 2 * np.pi, N))
        basis = spatial_basis(H, W, "poly", np, degree=1)
        coeffs = rng.standard_normal((basis.shape[0], N)) * 0.01

        rms_single, resid_single = step_field_quality(
            stack, a, u, v, delta, coeffs, basis, H, W, crop=5, precision="single")
        rms_fast, resid_fast = step_field_quality(
            stack, a, u, v, delta, coeffs, basis, H, W, crop=5, precision="fast")

        assert resid_single.dtype == np.float64
        assert resid_fast.dtype == np.float32
        assert rms_fast == pytest.approx(rms_single, abs=1e-4)

def _carrier_case(H=96, W=128, sigma=0.0, seed=5):
    """Wrapped tilt + curvature carrier, smooth modulation ``b``, complex noise."""
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    carrier = (0.3 * X - 0.2 * Y + 4e-4 * X**2 - 2e-4 * Y**2 + 1e-4 * X * Y + 0.6)
    b = 0.6 + 0.3 * np.cos(X / W * 2.0) * np.cos(Y / H * 1.5)
    rng = np.random.default_rng(seed)
    noise = sigma * (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
    zeta = b * np.exp(1j * carrier) + noise
    return np.angle(zeta), np.abs(zeta), carrier, b


class TestCarrier:
    def test_removes_noise_free_carrier(self):
        phi, w, _, _ = _carrier_case()
        r = remove_carrier(phi, w, precision="double")
        assert np.abs(r.phi).max() < 1e-8
        assert r.eta.shape == (6,)
        assert r.cond < 2

    def test_recovers_carrier_under_noise(self):
        phi, w, carrier, b = _carrier_case(sigma=0.3)
        r = remove_carrier(phi, w, precision="double")
        err = wrap(phi - r.phi - carrier)
        err = wrap(err - np.angle(np.sum(b * np.exp(1j * err))))
        assert np.abs(err).max() < 0.05

    def test_invariant_to_2pi_wraps(self):
        phi, w, _, _ = _carrier_case(sigma=0.1)
        m = np.random.default_rng(6).integers(-3, 4, phi.shape)
        r1 = remove_carrier(phi, w, precision="double")
        r2 = remove_carrier(phi + 2 * np.pi * m, w, precision="double")
        np.testing.assert_allclose(r1.eta, r2.eta, atol=1e-10)

    def test_subsampled_start_matches_full(self):
        phi, w, _, _ = _carrier_case(sigma=0.1)
        r1 = remove_carrier(phi, w, subsample=1, precision="double")
        r4 = remove_carrier(phi, w, subsample=4, precision="double")
        assert np.abs(wrap(r1.phi - r4.phi)).max() < 1e-8

    def test_degree_zero_removes_piston_only(self):
        phi = np.full((32, 40), 2.5)
        r = remove_carrier(phi, degree=0, precision="double")
        assert r.eta.shape == (1,)
        assert r.piston == pytest.approx(2.5)
        assert np.abs(r.phi).max() < 1e-12

    @pytest.mark.parametrize("kwargs", [dict(window=4), dict(window=0), dict(degree=-1),
                                        dict(subsample=0), dict(subsample=40),
                                        dict(max_iter=0)])
    def test_rejects_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            remove_carrier(np.zeros((32, 40)), **kwargs)


class TestReference:
    def test_resolves_sign_branch(self):
        H, W = 48, 48
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        common = np.angle(np.exp(1j * (0.3 * X + 0.2 * Y)))
        rng = np.random.default_rng(3)
        phi = np.angle(np.exp(1j * (common + 0.02 * rng.standard_normal((H, W)))))
        # reference measured on the opposite sign branch
        phi_ref = np.angle(np.exp(-1j * (common + 0.02 * rng.standard_normal((H, W)))))
        r = subtract_reference(phi, phi_ref)
        assert r.sign == -1
        assert not r.ambiguous
        # phi_ref = -common (opposite branch), so phi + phi_ref cancels the
        # common aberration while phi - phi_ref doubles it -- the chosen
        # (flipped) branch should have much lower spread.
        assert r.spread_flipped < r.spread_same


class TestCombine:
    def test_reduces_scatter_and_resolves_flips(self):
        H, W = 48, 64
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        phi_true = np.angle(np.exp(1j * (0.25 * X + 0.15 * Y)))
        rng = np.random.default_rng(4)

        def acquisition(sign, noise):
            return np.angle(np.exp(1j * sign * (phi_true + noise * rng.standard_normal((H, W)))))

        phis2 = [acquisition(1, 0.05), acquisition(-1, 0.05)]
        phis5 = phis2 + [acquisition(1, 0.05), acquisition(-1, 0.05), acquisition(1, 0.05)]

        r2 = combine_acquisitions(phis2, align_carrier=False)
        r5 = combine_acquisitions(phis5, align_carrier=False)
        assert set(r2.sign_flips) | set(r5.sign_flips)  # some flips detected
        assert circ_rms_deg(r5.phi, phi_true) <= circ_rms_deg(r2.phi, phi_true) + 0.5

    def test_raises_on_single_acquisition(self):
        with pytest.raises(ValueError):
            combine_acquisitions([np.zeros((4, 4))])


class TestWrap:
    def test_wrap_matches_angle_exp(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(-50, 50, 10_000)
        ref = np.angle(np.exp(1j * x))
        assert np.max(np.abs(wrap(x) - ref)) < 1e-12


def _quadrature_case(H=40, W=50, N=7, seed=20, dtype=np.float64):
    """A quadrature solution off the conventional frame in every way at once.

    Sheared and anisotropic `(u, v)`, a background correlated with both, a
    nonzero first step and a gain whose median is not 1 -- so each of
    `docs/vp_aia.md` §"Normalization" steps 1-4 has something to do.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    phi = (0.35 * xx + 0.22 * yy).ravel()
    b = (40 + 5 * np.sin(0.02 * yy)).ravel()
    u = (b * np.cos(phi)).astype(dtype)
    v = (-0.7 * b * np.sin(phi) + 0.3 * b * np.cos(phi)).astype(dtype)   # sheared
    a = (120 + 0.4 * u + 0.25 * v).astype(dtype)                         # correlated
    delta = np.sort(rng.uniform(0.3, 2 * np.pi, N))
    g = 2.5 * (1 + 0.1 * rng.standard_normal(N))                         # median != 1
    return a, u, v, g * np.cos(delta), g * np.sin(delta)


class TestGaugeConventions:
    """Each convention of `docs/gauge_conventions.md`, on its own helper."""

    def test_whiten_uv_equalizes_energy_and_decorrelates(self):
        rng = np.random.default_rng(11)
        P = 2000
        u = rng.standard_normal(P) * 3.0
        v = 0.4 * u + rng.standard_normal(P) * 0.5      # sheared and anisotropic
        u_w, v_w = whiten_uv(u, v, np)
        Suu, Svv, Suv = np.sum(u_w ** 2), np.sum(v_w ** 2), np.sum(u_w * v_w)
        assert abs(Suu - Svv) < 1e-8 * Suu              # Eq. (15), equal energy
        assert abs(Suv) < 1e-8 * Suu                    # Eq. (15), decorrelated

    def test_whiten_uv_preserves_total_energy_and_spanned_subspace(self):
        rng = np.random.default_rng(12)
        u = rng.standard_normal(500) * 2.0
        v = 0.3 * u + rng.standard_normal(500)
        u_w, v_w = whiten_uv(u, v, np)
        total = np.sum(u ** 2) + np.sum(v ** 2)
        assert abs(np.sum(u_w ** 2) + np.sum(v_w ** 2) - total) < 1e-8 * total
        # A re-basing of span{u, v}: projecting u onto {u_w, v_w} loses nothing.
        M = np.stack([u_w, v_w], axis=1)
        resid = u - M @ np.linalg.lstsq(M, u, rcond=None)[0]
        assert np.max(np.abs(resid)) < 1e-8 * np.max(np.abs(u))

    def test_pin_phase_origin_zeroes_first_step(self):
        delta = np.array([0.7, 1.9, 3.4, 5.1])
        pinned = pin_phase_origin(delta)
        assert pinned[0] == 0.0
        assert np.allclose(np.diff(pinned), np.diff(delta))   # differences untouched

    def test_normalize_gain_sets_median_to_one(self):
        g = np.array([0.4, 1.3, 2.2, 5.0, 0.9])
        g_n = normalize_gain(g, np)
        assert abs(float(np.median(g_n)) - 1.0) < 1e-12       # Eq. (16)
        assert np.allclose(g_n / g_n[0], g / g[0])            # ratios untouched

    def test_center_offsets_sets_mean_to_zero(self):
        c = np.array([1.5, -0.2, 3.3, 0.8])
        c_c = center_offsets(c, np)
        assert abs(float(np.mean(c_c))) < 1e-12               # Eq. (16)
        assert np.allclose(np.diff(c_c), np.diff(c))

    def test_whitening_matrix_paired_transform_preserves_model(self):
        # docs/vp_aia.md §"Quadrature frame": T on (u, v) and T^-1 on
        # (P_n, Q_n) leave a + P_n u + Q_n v unchanged.
        rng = np.random.default_rng(21)
        P, N = 3000, 7
        u = rng.standard_normal(P) * 2.0
        v = 0.6 * u + rng.standard_normal(P) * 0.4
        P_n, Q_n = rng.standard_normal(N), rng.standard_normal(N)
        before = np.outer(P_n, u) + np.outer(Q_n, v)
        T = whitening_matrix(u, v, np)
        Ti = np.linalg.inv(T)
        u_w, v_w = u * T[0, 0] + v * T[0, 1], u * T[1, 0] + v * T[1, 1]
        P_w, Q_w = P_n * Ti[0, 0] + Q_n * Ti[1, 0], P_n * Ti[0, 1] + Q_n * Ti[1, 1]
        after = np.outer(P_w, u_w) + np.outer(Q_w, v_w)
        assert np.max(np.abs(after - before)) < 1e-10 * np.max(np.abs(before))

    def test_normalize_quadrature_frame_imposes_all_conditions(self):
        a, u, v, P_n, Q_n = _quadrature_case()
        a, u, v, P_n, Q_n = normalize_quadrature_frame(a, u, v, P_n, Q_n, np)
        ac = a - a.mean()
        scale = np.sum(u ** 2)
        assert abs(np.sum(ac * u)) < 1e-8 * scale            # step 1, shift
        assert abs(np.sum(ac * v)) < 1e-8 * scale
        assert abs(np.sum(u ** 2) - np.sum(v ** 2)) < 1e-8 * scale   # step 2
        assert abs(np.sum(u * v)) < 1e-8 * scale
        assert abs(float(np.arctan2(Q_n[0], P_n[0]))) < 1e-12        # step 3
        assert abs(float(np.median(np.hypot(P_n, Q_n))) - 1.0) < 1e-12   # step 4

    def test_normalize_quadrature_frame_preserves_the_model(self):
        # Every step is a reparametrization of docs/vp_aia.md Eq. (3).
        a, u, v, P_n, Q_n = _quadrature_case()
        before = a[None, :] + np.outer(P_n, u) + np.outer(Q_n, v)
        out = normalize_quadrature_frame(a, u, v, P_n, Q_n, np)
        a2, u2, v2, P2, Q2 = out
        after = a2[None, :] + np.outer(P2, u2) + np.outer(Q2, v2)
        assert np.max(np.abs(after - before)) < 1e-9 * np.max(np.abs(before))

    def test_normalize_quadrature_frame_is_idempotent(self):
        first = normalize_quadrature_frame(*_quadrature_case(), np)
        second = normalize_quadrature_frame(*first, np)
        for x, y in zip(first, second):
            assert np.allclose(x, y, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("precision,work", [("single", np.float32),
                                                ("fast", np.float32),
                                                ("double", np.float64)])
    def test_normalize_quadrature_frame_dtypes(self, precision, work):
        a, u, v, P_n, Q_n = _quadrature_case(dtype=work)
        a, u, v, P_n, Q_n = normalize_quadrature_frame(a, u, v, P_n, Q_n, np,
                                                       precision=precision)
        assert a.dtype == u.dtype == v.dtype == work     # fields keep work dtype
        assert P_n.dtype == Q_n.dtype == np.float64      # (N,) vectors stay float64

class TestSpatialBasis:
    @pytest.mark.parametrize("kind", BASES)
    def test_every_family_is_orthonormal_and_zero_mean(self, kind):
        basis = spatial_basis(23, 29, kind, np)
        assert basis.ndim == 2 and basis.shape[1] == 23 * 29
        J = basis.shape[0]
        assert np.allclose(basis @ basis.T, np.eye(J), atol=1e-8)
        assert np.allclose(basis.mean(axis=1), 0, atol=1e-10)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="unknown basis"):
            spatial_basis(16, 16, "not_a_family", np)

    def test_truncation_of_a_basis_is_a_basis(self):
        full = spatial_basis(20, 24, "poly", np, degree=3)
        short = spatial_basis(20, 24, "poly", np, degree=1)
        assert np.allclose(full[: short.shape[0]], short, atol=1e-10)

    @pytest.mark.parametrize("H,W,degree", [(1, 16, 1), (16, 1, 2), (2, 3, 2), (3, 2, 2)])
    def test_field_too_small_for_family_raises(self, H, W, degree):
        with pytest.raises(ValueError, match="cannot be normalized"):
            spatial_basis(H, W, "poly", np, degree=degree)


class TestMethodParamDefaults:
    def test_base_phi_error_is_none(self):
        param = MethodParam()
        assert param.phi_error(np.ones((4, 4)), np.zeros((4, 4)), np.zeros(5),
                               np.ones(5), False, np.ones((4, 4)), False, np) is None

    def test_base_phase_step_field_broadcasts_the_piston(self):
        delta = np.array([0.0, 1.0, 2.0])
        field = MethodParam().phase_step_field(delta, 4, 5, np)
        assert field.shape == (3, 4, 5)
        assert np.allclose(field, delta[:, None, None])


class TestPhaseError:
    def test_ideal_configuration_matches_closed_form(self):
        """Eq. (10) reduces to Eq. (14) for evenly spaced steps and unit gain."""
        from phase_shift.errors import aia_phi_error_parts

        N, H, W = 7, 12, 10
        delta = 2 * np.pi * np.arange(N) / N
        g = np.ones(N)
        rng = np.random.default_rng(3)
        phi = rng.uniform(-np.pi, np.pi, (H, W))
        b = 1.0 + 0.5 * rng.random((H, W))
        sigma0 = np.full((H, W), 0.02)

        phi_var, _ = aia_phi_error_parts(b, phi, delta, g, False, sigma0, True, np)
        expected = (2.0 / N) * (sigma0 / b) ** 2                  # Eq. (14)
        assert np.allclose(phi_var, expected, rtol=1e-10)

    @pytest.mark.parametrize("fit_gain,coeff", [(False, -1.5), (True, -2.0)])
    def test_ideal_fitted_step_correction_matches_documented_coefficient(self, fit_gain, coeff):
        """Eqs. (25)/(30): the ideal-case reading of the exact Eqs. (24)/(29)."""
        from phase_shift.errors import aia_phi_error_parts

        N, H, W = 7, 16, 16
        Np = H * W
        delta = 2 * np.pi * np.arange(N) / N
        g = np.ones(N)
        phi = (2 * np.pi * np.arange(Np) / Np).reshape(H, W)   # spread evenly over 2*pi
        b = np.ones((H, W))
        sigma0 = np.full((H, W), 0.05)

        base, _ = aia_phi_error_parts(b, phi, delta, g, fit_gain, sigma0, True, np)
        full, _ = aia_phi_error_parts(b, phi, delta, g, fit_gain, sigma0, False, np)
        measured = (full / base - 1.0) * Np
        assert np.allclose(measured, coeff, atol=1e-9)

    def test_correction_scales_as_one_over_pixel_count(self):
        """The fitted-step term of Eqs. (24)/(29) is O(1/N_p)."""
        from phase_shift.errors import aia_phi_error_parts

        N = 7
        delta = np.sort(np.random.default_rng(4).uniform(0, 2 * np.pi, N))
        delta -= delta[0]
        g = np.ones(N)
        sizes = []
        for side in (16, 32):
            Y, X = np.mgrid[0:side, 0:side].astype(float)
            phi = np.angle(np.exp(1j * (0.7 * X + 0.4 * Y)))
            b = np.ones((side, side))
            sigma0 = np.full((side, side), 0.05)
            base, _ = aia_phi_error_parts(b, phi, delta, g, False, sigma0, True, np)
            full, _ = aia_phi_error_parts(b, phi, delta, g, False, sigma0, False, np)
            sizes.append(float(np.mean(full / base - 1.0)) * side ** 2)
        assert abs(sizes[0] - sizes[1]) < 0.35 * abs(sizes[0])   # N_p * correction is stable

    def test_solver_reports_a_phi_error_map(self):
        stack, truth = make_stack()
        solver = PhaseSolver(PhaseConfig(use_alpha=False, gain_mode="joint")).fit(stack)
        err = solver.result.phi_error
        assert err is not None and err.shape == truth["phi"].shape
        assert np.all(np.isfinite(err)) and np.all(err > 0)


def _vp_case(H=40, W=50, N=9, seed=31, degree=1, amp=0.02, dtype=np.float64):
    """A stack with a planted first-order step field, already in the conventional frame.

    Returns the flattened fields VP-AIA's first-order pass takes as input,
    together with the planted `alpha` (docs/vp_aia.md Eq. 6).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    phi = (0.35 * xx + 0.22 * yy).ravel()
    a = (120 + 8 * np.cos(0.01 * xx)).ravel()
    b = (45 + 3 * np.sin(0.02 * yy)).ravel()
    delta = np.sort(rng.uniform(0, 2 * np.pi, N))
    delta -= delta[0]
    g = 1 + 0.05 * rng.standard_normal(N)
    g /= np.median(g)
    basis = spatial_basis(H, W, "poly", np, degree=degree)
    alpha = rng.standard_normal((N, basis.shape[0])) * amp * np.sqrt(H * W)
    alpha -= alpha.mean(0)                                   # Eq. (14)
    Delta = alpha @ basis                                    # (N, P)
    stack = (a[None] + g[:, None] * b[None]
             * np.cos(phi[None] + delta[:, None] + Delta)).astype(dtype)
    # Zeroth order as docs/vp_aia.md §"VP-AIA algorithm" step 1 builds it: a
    # pixel step at the true (P_n, Q_n), then normalization. Eq. (11) needs the
    # pixel-step residual, not just fields that fit.
    P_n, Q_n = g * np.cos(delta), g * np.sin(delta)
    a0, u0, v0 = pixel_step(stack, delta, g, precision="double")
    a0, u0, v0, P_n, Q_n = normalize_quadrature_frame(a0, u0, v0, P_n, Q_n, np)
    return stack, a0, u0, v0, P_n, Q_n, basis, alpha


class TestVPSystem:
    """`docs/vp_aia.md` Eqs. (13)-(15), the first-order normal system."""

    def test_recovers_planted_coefficients(self):
        stack, a, u, v, P_n, Q_n, basis, alpha_true = _vp_case()
        sol = fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, basis)
        rel = np.abs(sol.alpha - alpha_true).max() / np.abs(alpha_true).max()
        assert rel < 1e-2, rel

    def test_unfitted_fields_are_rejected_by_the_math(self):
        # Eq. (12) assumes Pi leaves the residual unchanged, which holds only
        # for the pixel-step residual. Fields that merely fit the data -- here
        # the exact truth -- break the fit, so the docstring's precondition is
        # a real one, not a formality.
        stack, _, _, _, _, _, basis, alpha_true = _vp_case()
        H, W, N = 40, 50, 9
        rng = np.random.default_rng(31)
        yy, xx = np.mgrid[0:H, 0:W]
        phi = (0.35 * xx + 0.22 * yy).ravel()
        a_t = (120 + 8 * np.cos(0.01 * xx)).ravel()
        b_t = (45 + 3 * np.sin(0.02 * yy)).ravel()
        delta = np.sort(rng.uniform(0, 2 * np.pi, N)); delta -= delta[0]
        g = 1 + 0.05 * rng.standard_normal(N); g /= np.median(g)
        sol = fit_frame_and_coeffs(stack, a_t, b_t * np.cos(phi), -b_t * np.sin(phi),
                                   g * np.cos(delta), g * np.sin(delta), basis)
        rel = np.abs(sol.alpha - alpha_true).max() / np.abs(alpha_true).max()
        assert rel > 0.5, rel

    def test_solution_satisfies_eq24(self):
        stack, a, u, v, P_n, Q_n, basis, _ = _vp_case()
        sol = fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, basis)
        N = len(P_n)
        A = np.column_stack([np.ones(N), P_n, Q_n])
        # A' P1 sums N terms of size |P1|, so that product is the scale the
        # six conditions are satisfied against.
        scale = N * max(np.abs(sol.P_corr).max(), np.abs(sol.Q_corr).max())
        assert np.max(np.abs(A.T @ sol.P_corr)) < 1e-10 * scale    # six conditions
        assert np.max(np.abs(A.T @ sol.Q_corr)) < 1e-10 * scale
        assert np.max(np.abs(sol.alpha.mean(0))) < 1e-10 * np.abs(sol.alpha).max()

    def test_chunking_does_not_change_the_result(self):
        stack, a, u, v, P_n, Q_n, basis, _ = _vp_case()
        whole = fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, basis, chunk=10 ** 9)
        pieces = fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, basis, chunk=137)
        # Chunking only reorders the pixel sums, so the difference is the
        # float64 summation order alone. It lands on each vector as a whole,
        # so the tolerance goes on the vector's scale, not on each component's
        # own magnitude.
        assert whole.kappa_vp == pytest.approx(pieces.kappa_vp, rel=1e-9)
        for got, ref in ((pieces.alpha, whole.alpha), (pieces.P_corr, whole.P_corr),
                         (pieces.Q_corr, whole.Q_corr)):
            assert np.max(np.abs(got - ref)) < 1e-10 * np.max(np.abs(ref))

    def test_conditioning_does_not_track_the_pixel_count(self):
        # Regression: the modes are solved in Eq. (20)'s <H_j H_k> = delta
        # convention and Eq. (14)'s rows are weighted to M's scale. Without
        # either, kappa_vp grew linearly with K and reported the frame size
        # rather than the quality of the fit.
        kappas = []
        for H, W in ((30, 40), (60, 80), (120, 160)):
            stack, a, u, v, P_n, Q_n, basis, _ = _vp_case(H=H, W=W)
            kappas.append(fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, basis).kappa_vp)
        assert max(kappas) < 1e3, kappas                 # not 1e6 and rising
        assert max(kappas) / min(kappas) < 2.0, kappas   # K grew 16x

    def test_projector_identities(self):
        # docs/vp_aia.md Eq. (10): Pi A = 0, Pi^2 = Pi, rank N-3.
        rng = np.random.default_rng(5)
        N = 8
        P_n, Q_n = rng.standard_normal(N), rng.standard_normal(N)
        A = np.column_stack([np.ones(N), P_n, Q_n])
        Pi = np.eye(N) - A @ np.linalg.pinv(A.T @ A) @ A.T
        assert np.max(np.abs(Pi @ A)) < 1e-12
        assert np.max(np.abs(Pi @ Pi - Pi)) < 1e-12
        assert np.linalg.matrix_rank(Pi) == N - 3

    def test_too_few_frames_raises(self):
        stack, a, u, v, P_n, Q_n, basis, _ = _vp_case(N=4)
        with pytest.raises(ValueError, match="at least 5 frames"):
            fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, basis)

    def test_too_many_modes_raises(self):
        stack, a, u, v, P_n, Q_n, basis, _ = _vp_case(H=4, W=4, N=6, degree=3)
        with pytest.raises(ValueError, match="identifiable maximum"):
            fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, basis)

    def test_max_modes_matches_eq15(self):
        for N, K in ((5, 1000), (9, 2000), (32, 10 ** 6)):
            assert max_modes(N, K) == ((N - 3) * (K - 2)) // (N - 1)
        assert max_modes(3, 1000) == 0

    def test_empty_basis_gives_only_frame_corrections(self):
        stack, a, u, v, P_n, Q_n, _, _ = _vp_case()
        empty = spatial_basis(40, 50, "poly", np, degree=0)
        sol = fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, empty)
        assert sol.alpha.shape == (len(P_n), 0)
        assert sol.P_corr.shape == (len(P_n),)

    @pytest.mark.parametrize("precision", ["single", "fast", "double"])
    def test_precision_presets_agree(self, precision):
        stack, a, u, v, P_n, Q_n, basis, alpha_true = _vp_case(dtype=np.float32)
        sol = fit_frame_and_coeffs(stack, a, u, v, P_n, Q_n, basis, precision=precision)
        rel = np.abs(sol.alpha - alpha_true).max() / np.abs(alpha_true).max()
        assert rel < 5e-2, rel
        assert sol.alpha.dtype == np.float64        # the (2N+NJ) system is float64


def make_vp_stack(H=90, W=110, N=12, seed=11, amp=0.03, degree=1, dtype=np.float32):
    """Stack whose phase step carries a first-order spatial error on top of the piston.

    Built from the same mode family VP-AIA fits, with `docs/vp_aia.md`
    Eq. (6)'s zero frame mean, so the planted field is exactly representable.
    """
    rng = np.random.default_rng(seed)
    K = H * W
    yy, xx = np.mgrid[0:H, 0:W]
    phi = np.angle(np.exp(1j * (0.35 * xx + 0.22 * yy)))
    a = 120 + 8 * np.cos(0.01 * xx)
    b = 45 + 3 * np.sin(0.02 * yy)
    delta = np.sort(rng.uniform(0, 2 * np.pi, N))
    delta -= delta[0]
    g = 1 + 0.05 * rng.standard_normal(N)
    g /= np.median(g)
    basis = spatial_basis(H, W, "poly", np, degree=degree)
    alpha = rng.standard_normal((N, basis.shape[0])) * amp * np.sqrt(K)
    alpha -= alpha.mean(0)                                   # Eq. (14)
    Delta = (alpha @ basis).reshape(N, H, W)
    stack = (a[None] + g[:, None, None] * b[None]
             * np.cos(phi[None] + delta[:, None, None] + Delta))
    return stack.astype(dtype), dict(phi=phi, delta=delta, g=g, alpha=alpha,
                                     Delta=Delta, basis=basis)


class TestVPAIA:
    """`docs/vp_aia.md` end to end, through `PhaseSolver`."""

    @staticmethod
    def _solve(stack, method, **kw):
        cfg = PhaseConfig(method=method, gain_mode="joint", use_alpha=False,
                          method_kwargs={"iters": 60, "tol": 1e-10, **kw})
        return PhaseSolver(cfg, precision="double").fit(stack).result

    def test_beats_plain_aia_on_a_step_field(self):
        stack, truth = make_vp_stack()
        base = self._solve(stack, "aia")
        vp = self._solve(stack, "vp_aia", basis_kwargs={"degree": 1}, crop=10)
        e_base = circ_rms_deg(base.phi - np.median(base.phi - truth["phi"]), truth["phi"])
        e_vp = circ_rms_deg(vp.phi - np.median(vp.phi - truth["phi"]), truth["phi"])
        assert e_vp < e_base / 5, (e_vp, e_base)

    def test_recovers_the_planted_step_field(self):
        stack, truth = make_vp_stack()
        r = self._solve(stack, "vp_aia", basis_kwargs={"degree": 1}, crop=10)
        alpha = r.method_param.coeffs.T                      # (N, J)
        rel = np.abs(alpha - truth["alpha"]).max() / np.abs(truth["alpha"]).max()
        assert rel < 0.1, rel

    def test_step_field_keeps_both_zero_means(self):
        # docs/vp_aia.md Eq. (6): <H_j>_xy = 0 by construction, and Eq. (14)
        # supplies <alpha_nj>_n = 0.
        stack, truth = make_vp_stack()
        r = self._solve(stack, "vp_aia", basis_kwargs={"degree": 2}, crop=10)
        mp = r.method_param
        assert np.max(np.abs(mp.coeffs.mean(axis=1))) < 1e-9 * np.abs(mp.coeffs).max()
        field = mp.phase_step_field(r.delta, *r.phi.shape, np)
        spatial_mean = field.reshape(len(r.delta), -1).mean(axis=1) - r.delta
        assert np.max(np.abs(spatial_mean)) < 1e-9

    def test_fixed_gain_raises(self):
        # docs/vp_aia.md derives Eq. (14) with (P_n, Q_n) free; the fixed-gain
        # case is not derived, so the method refuses rather than inventing it.
        stack, _ = make_vp_stack()
        for cfg in (PhaseConfig(method="vp_aia", gain_mode="none"),
                    PhaseConfig(method="vp_aia", g=np.ones(stack.shape[0]))):
            with pytest.raises(ValueError, match="gain_mode='joint'"):
                PhaseSolver(cfg).fit(stack)

    def test_too_few_frames_raises(self):
        stack, _ = make_vp_stack(N=4)
        cfg = PhaseConfig(method="vp_aia", gain_mode="joint", use_alpha=False,
                          method_kwargs={"basis_kwargs": {"degree": 1}, "crop": 5})
        with pytest.raises(ValueError, match="at least 5 frames"):
            PhaseSolver(cfg).fit(stack)

    def test_empty_basis_leaves_the_aia_solution(self):
        stack, truth = make_vp_stack(amp=0.0)
        base = self._solve(stack, "aia")
        vp = self._solve(stack, "vp_aia", basis_kwargs={"degree": 0}, crop=10)
        assert vp.method_param.coeffs.shape == (0, stack.shape[0])
        assert circ_rms_deg(vp.phi, base.phi) < 1e-3

    def test_kappa_fit_is_the_joint_system(self):
        stack, _ = make_vp_stack()
        r = self._solve(stack, "vp_aia", basis_kwargs={"degree": 1}, crop=10)
        # The one system of Eq. (15), not a per-frame value: O(100), and it
        # does not track the pixel count.
        assert 1.0 < r.method_param.kappa_fit < 1e3

    @pytest.mark.parametrize("precision,work", [("single", np.float32),
                                                ("fast", np.float32),
                                                ("double", np.float64)])
    def test_precision_presets(self, precision, work):
        stack, truth = make_vp_stack()
        cfg = PhaseConfig(method="vp_aia", gain_mode="joint", use_alpha=False,
                          method_kwargs={"iters": 60, "tol": 1e-10, "crop": 10,
                                         "basis_kwargs": {"degree": 1}})
        r = PhaseSolver(cfg, precision=precision).fit(stack).result
        assert r.phi.dtype == r.a.dtype == r.b.dtype == work
        assert r.delta.dtype == r.g.dtype == np.float64
        d = wrap(np.asarray(r.phi, dtype=float) - truth["phi"])
        assert float(np.degrees(np.sqrt(np.mean((d - np.median(d)) ** 2)))) < 0.1


def make_uniform_case(H=80, W=90, N=9, deg=1, seed=4, amp=0.01):
    """A stack meeting the four conditions of `docs/vp_aia.md` Eq. (20).

    Uniform steps with `N >= 5`, constant `g_n` and `b`, many fringes, and the
    orthonormal modes `spatial_basis` already provides -- so Eq. (19) should
    reduce to Eq. (20)'s `1 + J/K`.
    """
    rng = np.random.default_rng(seed)
    K = H * W
    yy, xx = np.mgrid[0:H, 0:W]
    phi = np.angle(np.exp(1j * (0.9 * xx + 0.7 * yy)))
    a, b = np.full((H, W), 100.0), np.full((H, W), 40.0)
    delta = np.arange(N) * 2 * np.pi / N
    g = np.ones(N)
    basis = spatial_basis(H, W, "poly", np, degree=deg)
    alpha = rng.standard_normal((N, basis.shape[0])) * amp * np.sqrt(K)
    alpha -= alpha.mean(0)
    Delta = (alpha @ basis).reshape(N, H, W)
    stack = (a[None] + g[:, None, None] * b[None]
             * np.cos(phi[None] + delta[:, None, None] + Delta))
    return stack, phi, basis.shape[0], K


class TestVPPhaseError:
    """`docs/vp_aia.md` Eq. (19), VP-AIA's phase-error map."""

    @staticmethod
    def _fit(stack, deg, crop=5, precision="double"):
        cfg = PhaseConfig(method="vp_aia", gain_mode="joint", use_alpha=False,
                          method_kwargs={"iters": 80, "tol": 1e-12, "crop": crop,
                                         "basis_kwargs": {"degree": deg}})
        return PhaseSolver(cfg, precision=precision).fit(stack).result

    def test_first_term_is_aia_eq26(self):
        # docs/vp_aia.md §"Noise of the baseline steps": "The first term of
        # Eq. (19) is aia_noise.md Eq. (10)". With no modes the two must agree
        # exactly, which pins the s_n derivation.
        stack, _, _, _ = make_uniform_case(H=60, W=70)
        r = self._fit(stack, 1)
        H, W = r.phi.shape
        sigma0 = np.full((H, W), 0.5)
        empty = spatial_basis(H, W, "poly", np, degree=0)
        vp0 = vp_phi_error(r.b, r.phi, r.g * np.cos(r.delta), r.g * np.sin(r.delta),
                           sigma0, False, r.method_param.beta_cov_unit, empty, np)
        aia_var, _ = aia_phi_error_parts(r.b, r.phi, r.delta, r.g, True, sigma0, True, np)
        assert np.max(np.abs(vp0 - np.sqrt(aia_var))) < 1e-12 * float(vp0.max())

    def test_simplified_drops_the_correction(self):
        stack, _, _, _ = make_uniform_case(H=60, W=70)
        r = self._fit(stack, 2)
        mp, (H, W) = r.method_param, r.phi.shape
        sigma0 = np.full((H, W), 0.5)
        full = mp.phi_error(r.b, r.phi, r.delta, r.g, True, sigma0, False, np)
        simp = mp.phi_error(r.b, r.phi, r.delta, r.g, True, sigma0, True, np)
        assert np.all(full >= simp - 1e-12)              # the correction adds variance
        assert np.max(full - simp) > 0

    def test_reduces_to_eq30_under_its_four_conditions(self):
        # Uniform steps, constant g_n and b, many fringes, orthonormal modes:
        # Eq. (19) must give Eq. (20)'s 1 + J/K.
        for H, W, N, deg in ((80, 90, 9, 1), (80, 90, 9, 2), (120, 130, 13, 2)):
            stack, _, J, K = make_uniform_case(H=H, W=W, N=N, deg=deg)
            r = self._fit(stack, deg)
            mp = r.method_param
            sigma0 = np.full(r.phi.shape, 0.5)
            full = mp.phi_error(r.b, r.phi, r.delta, r.g, True, sigma0, False, np)
            simp = mp.phi_error(r.b, r.phi, r.delta, r.g, True, sigma0, True, np)
            got = float((full ** 2 / simp ** 2 - 1).mean())
            assert got == pytest.approx(J / K, rel=0.02), (got, J / K, H, W, N, deg)

    def test_matches_a_monte_carlo_phase_variance(self):
        # The map is a prediction about repeated noise realizations, so check
        # it against one. 300 trials give a 4.1% standard error on a standard
        # deviation, which is the tolerance below.
        H, W, N, deg, trials, sigma0 = 40, 44, 9, 1, 300, 0.30
        stack, phi_t, J, K = make_uniform_case(H=H, W=W, N=N, deg=deg)
        rng = np.random.default_rng(7)
        errs, preds = [], []
        s0 = np.full((H, W), sigma0)
        for _ in range(trials):
            r = self._fit(stack + sigma0 * rng.standard_normal(stack.shape), deg, crop=4)
            d = wrap(np.asarray(r.phi, dtype=float) - phi_t)
            errs.append(wrap(d - np.median(d)))
            preds.append(r.method_param.phi_error(r.b, r.phi, r.delta, r.g, True,
                                                  s0, False, np))
        c = (slice(4, -4), slice(4, -4))
        emp = np.array(errs).std(axis=0, ddof=1)[c].mean()
        pred = np.array(preds).mean(axis=0)[c].mean()
        se = 1.0 / np.sqrt(2 * (trials - 1))
        assert emp / pred == pytest.approx(1.0, abs=3 * se), (emp, pred)

    def test_solver_reports_the_map(self):
        stack, _, _, _ = make_uniform_case(H=60, W=70)
        r = self._fit(stack, 1)
        assert r.phi_error is not None and r.phi_error.shape == r.phi.shape
        assert np.all(r.phi_error > 0)


class TestVPRelinearization:
    """`docs/vp_aia.md` Eq. (18), the relinearization loop."""

    @staticmethod
    def _solve(stack, rounds, round_tol=1e-12, deg=1):
        cfg = PhaseConfig(method="vp_aia", gain_mode="joint", use_alpha=False,
                          method_kwargs={"iters": 80, "tol": 1e-12, "crop": 4,
                                         "basis_kwargs": {"degree": deg},
                                         "rounds": rounds, "round_tol": round_tol})
        return PhaseSolver(cfg, precision="double").fit(stack).result

    def test_one_round_is_the_plain_first_order_pass(self):
        stack, _, _, _ = make_uniform_case(H=50, W=56, N=9, amp=0.01)
        cfg = PhaseConfig(method="vp_aia", gain_mode="joint", use_alpha=False,
                          method_kwargs={"iters": 80, "tol": 1e-12, "crop": 4,
                                         "basis_kwargs": {"degree": 1}})
        default = PhaseSolver(cfg, precision="double").fit(stack).result
        explicit = self._solve(stack, rounds=1)
        assert default.method_param.rounds_run == 1
        assert np.allclose(default.method_param.coeffs, explicit.method_param.coeffs,
                           rtol=1e-12, atol=1e-14)

    def test_extra_rounds_help_when_the_field_is_large(self):
        # A single pass is first order in Delta_n, so a large field leaves a
        # second-order error that Eq. (18) is there to remove.
        stack, phi_t, _, _ = make_uniform_case(H=50, W=56, N=9, amp=0.12)
        one = circ_rms_deg(self._solve(stack, 1).phi, phi_t)
        many = circ_rms_deg(self._solve(stack, 5).phi, phi_t)
        assert many < one / 10, (one, many)

    def test_phase_error_falls_round_over_round(self):
        stack, phi_t, _, _ = make_uniform_case(H=50, W=56, N=9, amp=0.12)
        errs = [circ_rms_deg(self._solve(stack, r).phi, phi_t) for r in (1, 2, 3, 4)]
        assert all(b < a for a, b in zip(errs, errs[1:])), errs

    def test_round_tol_stops_early(self):
        # A field the first pass already resolves: the second round's increment
        # falls under round_tol and the loop stops rather than spending its
        # budget.
        stack, _, _, _ = make_uniform_case(H=50, W=56, N=9, amp=0.005)
        r = self._solve(stack, rounds=6, round_tol=1e-4)
        assert r.method_param.rounds_run < 6
        assert len(r.method_param.increment_rms) == r.method_param.rounds_run
        assert r.method_param.increment_rms[-1] < 1e-4

    def test_accumulated_field_keeps_both_zero_means(self):
        # "The increments have zero spatial and frame means, so the accumulated
        # Delta_n keeps both conventions."
        stack, _, _, _ = make_uniform_case(H=50, W=56, N=9, amp=0.12, deg=2)
        mp = self._solve(stack, rounds=4, deg=2).method_param
        assert mp.rounds_run == 4
        assert np.max(np.abs(mp.coeffs.mean(axis=1))) < 1e-9 * np.abs(mp.coeffs).max()

    def test_bad_round_arguments_raise(self):
        stack, _, _, _ = make_uniform_case(H=50, W=56, N=9)
        with pytest.raises(ValueError, match="rounds must be at least 1"):
            self._solve(stack, rounds=0)
        with pytest.raises(ValueError, match="round_tol must be non-negative"):
            self._solve(stack, rounds=2, round_tol=-1.0)
