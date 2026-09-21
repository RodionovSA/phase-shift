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
from phase_shift.backend import CUPY_AVAILABLE, Precision, get_precision, set_precision
from phase_shift.basis import BASES, spatial_basis
from phase_shift.methods import MethodParam
from phase_shift.methods.gauge import (center_coeffs, center_offsets, normalize_gain,
                                       pin_phase_origin, whiten_uv)
from phase_shift.methods.step_field import step_field_quality
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


def make_step_field_stack(H=40, W=48, N=16, seed=7, kind="quadratic", dtype=np.float32,
                           gain_std=0.0):
    """Synthetic stack whose per-frame phase step carries a spatial error on top of the piston.

    ``kind`` selects the shape of that error:

    - ``"linear"``: a pure tilt, ``k_xn*x`` (no curvature) -- a degree-1 step field exactly.
    - ``"quadratic"``: a pure curvature term with zero mean and no linear part -- a
      degree-1 (tilt-only) fit cannot represent this at all.
    - ``"static_quadratic"``: the same curvature shape, but identical every frame (no
      frame-to-frame variation) -- should be invisible to the step-field mechanism and
      fully absorbed into the recovered phase instead (docs/sf_aia.md §9.3).

    ``gain_std`` adds random per-frame contrast on top of the default ``g_true = 1``
    (still normalized to ``median(g_true) = 1``), for exercising ``fit_gain=True``
    alongside the step-field refinement.
    """
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    Xc, Yc = X - X.mean(), Y - Y.mean()
    phi_true = np.angle(np.exp(1j * (0.25 * X + 0.18 * Y)))
    b_true = 1.0 + 0.2 * rng.random((H, W))
    a_true = 2.0 + 0.1 * rng.random((H, W))
    delta_true = np.arange(N) * 2 * np.pi / N
    g_true = np.ones(N) if gain_std == 0.0 else 1.0 + gain_std * rng.standard_normal(N)
    g_true = g_true / np.median(g_true)

    scale = 1.0 / max(np.abs(Xc).max(), np.abs(Yc).max()) ** 2
    Xc2 = (Xc ** 2 - (Xc ** 2).mean()) * scale
    drift = (np.arange(N) - (N - 1) / 2.0) / N               # zero-mean, monotone in n

    if kind == "linear":
        kx = 0.3 * drift / max(np.abs(Xc).max(), 1.0)
        Delta = kx[:, None, None] * Xc[None, :, :]
    elif kind == "quadratic":
        Delta = 0.8 * drift[:, None, None] * Xc2[None, :, :]
    elif kind == "static_quadratic":
        Delta = np.broadcast_to(0.5 * Xc2, (N, H, W)).copy()
    else:
        raise ValueError(f"unknown kind {kind!r}")

    step = delta_true[:, None, None] + Delta
    stack = np.empty((N, H, W), dtype=np.float64)
    for n in range(N):
        stack[n] = a_true + g_true[n] * b_true * np.cos(phi_true + step[n])
    stack += 0.005 * rng.standard_normal(stack.shape)
    return stack.astype(dtype), dict(phi=phi_true, delta=delta_true, Delta=Delta, g=g_true)


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
        cfg = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                           method_kwargs={"iters": 5})
        cfg.to_yaml(p)
        cfg2 = PhaseConfig.from_yaml(p)
        assert cfg2.use_alpha is False
        assert cfg2.gain_mode == "none"
        assert cfg2.method == "sf_aia"
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
            for method in ("aia", "sf_aia"):
                kw = {"iters": 10} if method == "aia" else {"iters": 10, "crop": 5}
                cfg = PhaseConfig(method=method, gain_mode="none", method_kwargs=kw)
                r = PhaseSolver(cfg, precision=name).fit(stack).result
                assert r.precision == Precision.of(name)
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

    def test_quadratic_step_field_needs_higher_degree(self):
        stack, _ = make_step_field_stack(kind="quadratic")
        kw = dict(iters=40, tol=1e-6, refine_iters=8, refine_tol=1e-8, crop=5)
        cfg1 = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                            method_kwargs=dict(basis_kwargs=dict(degree=1), **kw))
        cfg2 = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                            method_kwargs=dict(basis_kwargs=dict(degree=2), **kw))
        r1 = PhaseSolver(cfg1).fit(stack)
        r2 = PhaseSolver(cfg2).fit(stack)
        assert r2.result.reconstruction_error < 0.5 * r1.result.reconstruction_error

    def test_linear_tilt_recovered_at_degree1(self):
        stack, _ = make_step_field_stack(kind="linear")
        cfg_plain = PhaseConfig(use_alpha=False, gain_mode="none", method="aia")
        cfg_tilt = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                                method_kwargs=dict(iters=40, tol=1e-6, basis_kwargs=dict(degree=1),
                                                    refine_iters=8, refine_tol=1e-8, crop=5))
        r_plain = PhaseSolver(cfg_plain).fit(stack)
        r_tilt = PhaseSolver(cfg_tilt).fit(stack)
        assert r_tilt.result.reconstruction_error < 0.3 * r_plain.result.reconstruction_error

    def test_linear_tilt_recovered_with_joint_gain(self):
        # Same linear-tilt field as test_linear_tilt_recovered_at_degree1,
        # but with per-frame gain drift added -- gain_mode="joint" must
        # recover both the tilt and the drift together, plumbed through the
        # step-field refine loop's own re-fitted g (see sf_aia).
        stack, truth = make_step_field_stack(kind="linear", gain_std=0.3)
        cfg_tilt = PhaseConfig(use_alpha=False, gain_mode="joint", method="sf_aia",
                                method_kwargs=dict(iters=40, tol=1e-6, basis_kwargs=dict(degree=1),
                                                    refine_iters=8, refine_tol=1e-8, crop=5))
        r_tilt = PhaseSolver(cfg_tilt).fit(stack)
        assert r_tilt.result.reconstruction_error < 0.05
        assert np.corrcoef(r_tilt.result.g, truth["g"])[0, 1] > 0.99

    def test_frame_independent_aberration_absorbed_into_phase(self):
        stack, _ = make_step_field_stack(kind="static_quadratic")
        cfg = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                           method_kwargs=dict(iters=40, tol=1e-6, basis_kwargs=dict(degree=2),
                                               refine_iters=8, refine_tol=1e-8, crop=5))
        r = PhaseSolver(cfg).fit(stack)
        assert r.result.reconstruction_error < 0.05

    def test_default_basis_matches_explicit_degree1(self):
        # the default basis must stay a degree-1 polynomial, a pure tilt
        stack, _ = make_step_field_stack(kind="linear")
        kw = dict(iters=40, tol=1e-6, refine_iters=8, refine_tol=1e-8, crop=5)
        cfg_default = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                                   method_kwargs=kw)
        cfg_explicit = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                                    method_kwargs=dict(basis_kwargs=dict(degree=1), **kw))
        r_default = PhaseSolver(cfg_default).fit(stack).result
        r_explicit = PhaseSolver(cfg_explicit).fit(stack).result
        assert r_default.method_param.basis_kwargs.get("degree", 1) == 1
        assert r_default.reconstruction_error == pytest.approx(r_explicit.reconstruction_error)

    def test_degree0_matches_plain_aia(self):
        stack, _ = make_step_field_stack(kind="quadratic")
        cfg_plain = PhaseConfig(use_alpha=False, gain_mode="none", method="aia")
        cfg_deg0 = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                                method_kwargs=dict(basis_kwargs=dict(degree=0), refine_iters=5, crop=5))
        r_plain = PhaseSolver(cfg_plain).fit(stack).result
        r_deg0 = PhaseSolver(cfg_deg0).fit(stack).result
        mp = r_deg0.method_param

        assert r_deg0.reconstruction_error == pytest.approx(r_plain.reconstruction_error)
        assert mp.coeffs.shape == (0, stack.shape[0])
        assert mp.refine_iters_run == 0
        assert mp.best_iter == -1

    def test_fast_close_to_single(self):
        # Same equivalence check as TestAIA's, for sf_aia's own
        # stack-scale reductions (frame_step inside the refine loop,
        # step_field_quality's model/resid reconstruction, and its RMS
        # ratio -- all in src/phase_shift/methods/sf_aia.py).
        stack, truth = make_step_field_stack(kind="linear", gain_std=0.3)
        kw = dict(iters=40, tol=1e-6, basis_kwargs=dict(degree=1), refine_iters=8, refine_tol=1e-8, crop=5)
        cfg = PhaseConfig(use_alpha=False, gain_mode="joint", method="sf_aia",
                          method_kwargs=kw)
        r_single = PhaseSolver(cfg, precision="single").fit(stack).result
        r_fast = PhaseSolver(cfg, precision="fast").fit(stack).result
        assert circ_rms_deg(r_fast.phi, r_single.phi) < 1e-2
        assert r_fast.method_param.rms_frac == pytest.approx(r_single.method_param.rms_frac, abs=1e-4)
        assert np.allclose(r_fast.method_param.coeffs, r_single.method_param.coeffs, atol=1e-3)
        # method_param carries the precision it was solved with, so
        # phase_step_field (called generically by PhaseSolver.fit's
        # reconstruction-error check) rebuilds the field the same way.
        assert r_single.method_param.precision == Precision.of("single")
        assert r_fast.method_param.precision == Precision.of("fast")
        assert r_fast.reconstruction_error == pytest.approx(r_single.reconstruction_error, abs=1e-4)

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

    def test_refine_loop_keeps_best_round_not_last(self, monkeypatch):
        """A round that makes rms_frac worse must not stop the loop early or be returned.

        Patches step_field_quality to return an engineered rms sequence with
        a regression in the middle (round 1 worse than round 0), so the
        control flow of the refinement loop is tested in isolation from the
        actual per-frame fit quality.
        """
        import phase_shift.methods.sf_aia as sf

        stack, _ = make_step_field_stack(kind="quadratic", H=16, W=16, N=6)
        rms_seq = iter([0.20, 0.35, 0.19, 0.19])

        def fake_quality(*args, **kwargs):
            return next(rms_seq), None

        monkeypatch.setattr(sf, "step_field_quality", fake_quality)

        cfg = PhaseConfig(use_alpha=False, gain_mode="none", method="sf_aia",
                           method_kwargs=dict(basis_kwargs=dict(degree=2), refine_iters=4, refine_tol=1e-3, crop=2))
        r = PhaseSolver(cfg).fit(stack)
        mp = r.result.method_param

        assert mp.rms_frac_history == pytest.approx([0.20, 0.35, 0.19, 0.19])
        assert mp.best_iter == 2
        assert mp.rms_frac == pytest.approx(0.19)


class TestCarrier:
    def test_recovers_pure_carrier_and_curvature(self):
        H, W = 80, 96
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        kx, ky = 0.05, -0.03
        kxx, kyy, kxy = 2e-4, -1e-4, 5e-5
        piston = 0.6
        phi = np.angle(np.exp(1j * (kx * X + ky * Y + kxx * X**2 + kyy * Y**2
                                     + kxy * X * Y + piston)))
        r = remove_carrier(phi, defocus=True, refine_iters=10, n_blocks=6)
        assert abs(r.kx - kx) < 1e-6
        assert abs(r.ky - ky) < 1e-6
        assert abs(r.kxx - kxx) < 1e-8
        assert abs(r.kyy - kyy) < 1e-8
        assert abs(r.kxy - kxy) < 1e-8
        assert circ_rms_deg(r.phi, np.zeros((H, W))) < 1e-4

    def test_defaults_match_defocus_true_call(self):
        H, W = 64, 64
        rng = np.random.default_rng(2)
        phi = np.angle(np.exp(1j * (0.02 * np.arange(W))[None, :] * np.ones((H, 1))
                              + 1j * 0.05 * rng.standard_normal((H, W))))
        r_default = remove_carrier(phi)
        r_explicit = remove_carrier(phi, defocus=True, refine_iters=10, n_blocks=10)
        assert r_default.kx == r_explicit.kx
        assert r_default.ky == r_explicit.ky


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

    def test_center_coeffs_sets_frame_mean_to_zero_per_term(self):
        rng = np.random.default_rng(13)
        coeffs = rng.standard_normal((3, 7)) + np.array([[2.0], [-1.0], [0.5]])
        centered = center_coeffs(coeffs)
        assert np.allclose(centered.mean(axis=1), 0, atol=1e-12)   # Eq. (T3b)
        assert np.allclose(centered - centered[:, :1], coeffs - coeffs[:, :1])


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
        """Eq. (26) reduces to Eq. (30) for evenly spaced steps and unit gain."""
        from phase_shift.errors import aia_phi_error_parts

        N, H, W = 7, 12, 10
        delta = 2 * np.pi * np.arange(N) / N
        g = np.ones(N)
        rng = np.random.default_rng(3)
        phi = rng.uniform(-np.pi, np.pi, (H, W))
        b = 1.0 + 0.5 * rng.random((H, W))
        sigma0 = np.full((H, W), 0.02)

        phi_var, _ = aia_phi_error_parts(b, phi, delta, g, False, sigma0, True, np)
        expected = (2.0 / N) * (sigma0 / b) ** 2                  # Eq. (30)
        assert np.allclose(phi_var, expected, rtol=1e-10)

    @pytest.mark.parametrize("fit_gain,coeff", [(False, -1.5), (True, -2.0)])
    def test_ideal_fitted_step_correction_matches_documented_coefficient(self, fit_gain, coeff):
        """Eqs. (41)/(46): the ideal-case reading of the exact Eqs. (40)/(45)."""
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
        """The fitted-step term of Eqs. (40)/(45) is O(1/N_p)."""
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

    def test_step_field_term_adds_noise_and_grows_with_basis_size(self):
        """`docs/sf_aia.md` Eq. (E7): the fitted field can only add variance."""
        from phase_shift.errors import aia_phi_error_parts, step_field_phi_error

        N, H, W = 9, 20, 24
        rng = np.random.default_rng(6)
        delta = np.sort(rng.uniform(0, 2 * np.pi, N))
        delta -= delta[0]
        g = np.ones(N)
        Y, X = np.mgrid[0:H, 0:W].astype(float)
        phi = np.angle(np.exp(1j * (0.9 * X + 0.5 * Y)))
        b = 0.6 + 0.8 * rng.random((H, W))
        sigma0 = np.full((H, W), 0.05)

        aia_var, _ = aia_phi_error_parts(b, phi, delta, g, False, sigma0, False, np)
        extras = []
        for degree in (1, 2):
            basis = spatial_basis(H, W, "poly", np, degree=degree)
            total = step_field_phi_error(b, phi, delta, g, False, sigma0, False,
                                         basis, np) ** 2
            extras.append(float(np.mean(total - aia_var)))
        assert extras[0] > 0                       # adds variance, never subtracts
        assert extras[1] > extras[0]               # and more of it with more modes

    def test_step_field_term_weights_noise_by_where_it_sits(self):
        """Eq. (E7)'s `E_nm` carries `sigma_0^2` under the sum over pixels."""
        from phase_shift.errors import aia_phi_error_parts, step_field_phi_error

        N, H, W = 9, 20, 24
        delta = 2 * np.pi * np.arange(N) / N
        g = np.ones(N)
        Y, X = np.mgrid[0:H, 0:W].astype(float)
        phi = np.angle(np.exp(1j * (0.9 * X + 0.5 * Y)))
        b = np.ones((H, W))
        basis = spatial_basis(H, W, "poly", np, degree=1)

        def extra(sigma0):
            aia_var, _ = aia_phi_error_parts(b, phi, delta, g, False, sigma0, False, np)
            total = step_field_phi_error(b, phi, delta, g, False, sigma0, False,
                                         basis, np) ** 2
            return total - aia_var

        flat = np.full((H, W), 0.05)
        # Same mean square, concentrated on a disc at the centre of the field.
        disc = np.where((X - W / 2) ** 2 + (Y - H / 2) ** 2 < (W / 4) ** 2, 3.0, 1.0)
        piled = 0.05 * disc / np.sqrt(np.mean(disc ** 2))
        assert np.isclose(np.mean(piled ** 2), np.mean(flat ** 2))
        assert np.isclose(np.mean(extra(2 * flat)), 4 * np.mean(extra(flat)))
        # A single sigma_0 pulled out of Eq. (E7) would make these equal.
        assert np.mean(extra(piled)) < 0.75 * np.mean(extra(flat))

    def test_solver_reports_a_phi_error_map(self):
        stack, truth = make_stack()
        solver = PhaseSolver(PhaseConfig(use_alpha=False, gain_mode="joint")).fit(stack)
        err = solver.result.phi_error
        assert err is not None and err.shape == truth["phi"].shape
        assert np.all(np.isfinite(err)) and np.all(err > 0)
