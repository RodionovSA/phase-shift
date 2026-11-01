# scripts/analysis/extract_phase.py
"""Extract sample-minus-reference phase for a wavelength stack and save it as .npy.

For each wavelength: fit the sample and reference interferogram stacks, subtract the
reference phase (sign branch resolved, no carrier removal), and optionally remove the
fringe-leakage ripple by fitting harmonics of the reference phase on bare glass.

Usage:
    uv run python scripts/analysis/extract_phase.py --config scripts/analysis/extract_phase.yaml
"""

import argparse
import re
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from phase_shift import PhaseConfig, PhaseSolver, subtract_reference
from phase_shift.backend import asnumpy

try:
    import cupy as cp
except ImportError:
    cp = None


def rob(x, k=5):
    """Robust outlier mask, unbiased by outliers."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))*1.4826
    return np.abs(x - med) >= k*mad


def correct(stack, mask):
    """Replace hot/dead pixels by the median of their good 3x3 neighbours."""
    ys, xs = np.where(mask)
    corr = stack.astype(float)
    for y, x in zip(ys, xs):
        sl = (slice(max(0, y - 1), y + 2), slice(max(0, x - 1), x + 2))
        good = ~mask[sl]
        if good.any():
            corr[:, y, x] = np.median(stack[:, *sl][:, good], axis=1)
    return corr


def box_mean(x, r):
    """Mean over a (2r+1)^2 window, edges padded."""
    for axis in (0, 1):
        p = np.pad(x, [(r + 1, r) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        c = np.cumsum(p, axis=axis)
        x = (np.take(c, range(2*r + 1, c.shape[axis]), axis=axis)
             - np.take(c, range(0, c.shape[axis] - 2*r - 1), axis=axis))/(2*r + 1)
    return x


def glass_mask(I_s, I_r, thr, grow):
    """Bare glass from the transmission map I_s/I_r; structures grown by `grow` px."""
    T = box_mean(I_s/np.maximum(I_r, 1e-6), 2)
    T /= np.median(T)
    return box_mean((np.abs(T - 1) > thr).astype(float), grow) == 0


def remove_leakage(phi, psi, glass, K, poly_deg, step):
    """Fit sum_k (cos k psi, sin k psi) x polynomial(x, y) on glass and subtract it from phi."""
    H, W = phi.shape

    def terms(sl):
        yy, xx = np.mgrid[0:H, 0:W]
        Y, X = (yy[sl] - H/2)/(H/2), (xx[sl] - W/2)/(W/2)
        mono = [X**i*Y**j for i in range(poly_deg + 1) for j in range(poly_deg + 1 - i)]
        yield np.ones_like(X)
        for k in range(1, K + 1):
            for f in (np.cos, np.sin):
                h = f(k*psi[sl])
                for mo in mono:
                    yield h*mo

    z = np.exp(1j*phi)
    x = np.angle(z*np.conj(z[glass].mean()))          # glass piston removed
    sub = (slice(0, H, step), slice(0, W, step))
    g = glass[sub]
    A = np.stack([t[g] for t in terms(sub)], -1)      # (P_glass, n_terms)
    c, *_ = np.linalg.lstsq(A, x[sub][g], rcond=None)
    ripple = np.zeros((H, W))
    for i, (ci, t) in enumerate(zip(c, terms((slice(None), slice(None))))):
        if i:                                         # piston stays in phi
            ripple += ci*t
    return np.angle(np.exp(1j*(phi - ripple))), np.degrees(ripple[glass].std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("extract_phase.yaml"))
    cfg = yaml.safe_load(ap.parse_args().config.read_text())
    data, sol, leak = cfg["data"], cfg["solver"], cfg["leakage"]

    root = Path(data["root"])
    files = {int(re.search(r"_(\d+)nm_", p.name).group(1)): p for p in root.glob("*nm_*.npz")}
    wavelengths = sorted(files) if data["wavelengths"] is None else list(data["wavelengths"])
    N = data["N"]

    black = np.load(next(root.glob("*_black_*.npz")))["black"]
    m = black.mean(0)
    bad = rob(m) | rob(black.std(0))

    config = PhaseConfig(use_alpha=sol["use_alpha"], gain_mode=sol["gain_mode"],
                         method=sol["method"], method_kwargs=sol["method_kwargs"])
    solver = PhaseSolver(config, device=sol["device"], precision=sol["precision"])

    out = Path(data["out"])
    np.save(out.with_name(out.stem + "_wvl.npy"), np.array(wavelengths))
    stack = None
    for w, wl in enumerate(tqdm(wavelengths)):
        with np.load(files[wl]) as d:
            sample = d[d.files[1]][0, :N]
            ref = d[d.files[4]][0, :N]
            if leak["enabled"]:
                I_s = correct(d[d.files[3]] - m, bad).mean(0)   # sample arm at the sample position
                I_r = correct(d[d.files[6]] - m, bad).mean(0)   # sample arm at the reference position

        rs = solver.fit(correct(sample - m, bad)).result
        rr = solver.fit(correct(ref - m, bad)).result
        phi_r = asnumpy(rr.phi)
        phi = asnumpy(subtract_reference(rs.phi, rr.phi, rr.b).phi)
        del rs, rr
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()

        if leak["enabled"]:
            glass = glass_mask(I_s, I_r, leak["mask_thr"], leak["mask_grow"])
            phi, ripple_std = remove_leakage(phi, phi_r, glass, leak["K"], leak["poly_deg"], leak["step"])
            tqdm.write(f"{wl} nm: glass fraction {glass.mean():.2f}, removed ripple rms {ripple_std:.2f} deg")

        if stack is None:
            stack = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                              shape=(len(wavelengths),) + phi.shape)
        stack[w] = phi
    stack.flush()
    print(f"saved {out} {stack.shape} and {out.with_name(out.stem + '_wvl.npy')}")


if __name__ == "__main__":
    main()
