# scripts/analysis/compare_carrier_removal.py
"""Compare three carrier-removal orders for sample/reference phase subtraction.

current     : remove_carrier on sample and reference (no mask), then subtract_reference
masked      : remove_carrier on sample and reference with structures masked, then subtract
sub_first   : subtract_reference on the raw maps, then remove_carrier on the difference (masked)

Each approach is scored on bare glass around the patch array: peak-to-valley of a fitted
tilt and quadratic, the residual rms, and the spread of the mean phase over four glass
spots around the array.

Usage:
    uv run python scripts/analysis/compare_carrier_removal.py --root <data folder> \
        --wavelengths 500 550 590 650 700 --out <output folder>
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from phase_shift import PhaseConfig, PhaseSolver, remove_carrier, subtract_reference
from phase_shift.backend import asnumpy


APPROACHES = ("current", "masked", "sub_first")


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


def glass_mask(b_s, b_r, thr=0.03, grow=15):
    """Bare glass: sample/reference fringe-amplitude ratio flat; structures grown by `grow` px."""
    ratio = box_mean(np.abs(b_s)/np.maximum(np.abs(b_r), 1e-6), 2)
    struct = np.abs(ratio/np.median(ratio) - 1) > thr
    struct |= np.abs(b_s) < 0.2*np.median(np.abs(b_s))
    return box_mean(struct.astype(float), grow) == 0


def fit_background(phi, mask, step=4):
    """Tilt and quadratic over masked pixels: (tilt p-v, quadratic p-v, residual rms), degrees."""
    H, W = phi.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    m = mask[::step, ::step]
    z = np.exp(1j*phi[::step, ::step][m])
    x = np.angle(z*np.conj(z.mean()))
    Y, X = (ys[m] - H/2)/(H/2), (xs[m] - W/2)/(W/2)
    A = np.stack([np.ones_like(X), X, Y, X*X, X*Y, Y*Y], -1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ c
    return (np.degrees(np.ptp(A[:, 1:3] @ c[1:3])), np.degrees(np.ptp(A[:, 3:] @ c[3:])),
            np.degrees(r.std()))


def spot_spread(phi, mask, center, radius=160, side=30):
    """Std (deg) of the mean glass phase in four spots at +-radius around center."""
    cy, cx = center
    means = []
    for dy, dx in ((0, -radius), (0, radius), (-radius, 0), (radius, 0)):
        sl = (slice(cy + dy - side//2, cy + dy + side//2), slice(cx + dx - side//2, cx + dx + side//2))
        good = mask[sl]
        if good.mean() > 0.5:
            means.append(np.exp(1j*phi[sl][good]).mean())
    means = np.array(means)
    return np.degrees(np.angle(means*np.conj(means.mean())).std()), len(means)


def array_center(mask, guess, half=150):
    """Centroid of the non-glass pixels in a box around guess (the patch array)."""
    gy, gx = guess
    sub = ~mask[gy - half:gy + half, gx - half:gx + half]
    ys, xs = np.nonzero(sub)
    return gy - half + int(ys.mean()), gx - half + int(xs.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--wavelengths", type=int, nargs="+", default=[500, 550, 590, 650, 700])
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--method", default="vp_aia", choices=["aia", "vp_aia"])
    ap.add_argument("--array-guess", type=int, nargs=2, default=[1005, 1295], help="y x of the patch array")
    ap.add_argument("--window", type=int, default=400, help="half size of the scoring window around the array")
    ap.add_argument("--out", type=Path, default=Path("carrier_compare"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")

    black = np.load(next(args.root.glob("*_black_*.npz")))["black"]
    m, s = black.mean(0), black.std(0)
    bad = rob(m) | rob(s)

    kwargs = {"iters": 30, "tol": 1e-4}
    if args.method == "vp_aia":
        kwargs |= {"basis": "poly", "basis_kwargs": {"degree": 2}, "orders": 1, "order_tol": 0.0, "crop": 100}
    solver = PhaseSolver(PhaseConfig(use_alpha=True, gain_mode="joint", method=args.method,
                                     method_kwargs=kwargs, phi_error_simplified=True), precision="single")

    rows, maps = [], {}
    for wl in args.wavelengths:
        f = next(args.root.glob(f"*_{wl}nm_*.npz"))
        with np.load(f) as d:
            sample = d[d.files[1]][0, :args.N]
            ref = d[d.files[4]][0, :args.N]
        rs = solver.fit(correct(sample - m, bad)).result
        rr = solver.fit(correct(ref - m, bad)).result
        phi_s, b_s = asnumpy(rs.phi), asnumpy(rs.b)
        phi_r, b_r = asnumpy(rr.phi), asnumpy(rr.b)

        glass = glass_mask(b_s, b_r)
        out = {}
        out["current"] = subtract_reference(remove_carrier(phi_s, b_s, degree=2).phi,
                                            remove_carrier(phi_r, b_r, degree=2).phi, b_r).phi
        out["masked"] = subtract_reference(remove_carrier(phi_s, b_s, mask=glass, degree=2).phi,
                                           remove_carrier(phi_r, b_r, mask=glass, degree=2).phi,
                                           b_r, mask=glass).phi
        out["sub_first"] = remove_carrier(subtract_reference(phi_s, phi_r, b_r, mask=glass).phi,
                                          b_s, mask=glass, degree=2).phi

        cy, cx = array_center(glass, args.array_guess)
        h = args.window
        win = np.zeros_like(glass)
        win[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h] = True
        for name in APPROACHES:
            phi = asnumpy(out[name])
            tilt, quad, rms = fit_background(phi, glass & win)
            spread, n_spots = spot_spread(phi, glass, (cy, cx))
            rows.append(dict(wavelength=wl, approach=name, tilt_pv=tilt, quad_pv=quad,
                             resid_rms=rms, spot_std=spread, n_spots=n_spots))
            print(f"{wl} nm {name:9s} tilt p-v {tilt:6.2f}  quad p-v {quad:6.2f}  "
                  f"resid rms {rms:5.2f}  spot std {spread:5.2f} ({n_spots} spots)  [deg]", flush=True)
            maps[(wl, name)] = (phi, glass, (cy, cx))

    with open(args.out/"metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # glass-background maps around the array, piston removed on glass
    h = args.window
    fig, ax = plt.subplots(len(APPROACHES), len(args.wavelengths),
                           figsize=(3.2*len(args.wavelengths), 3*len(APPROACHES)), squeeze=False)
    for j, wl in enumerate(args.wavelengths):
        for i, name in enumerate(APPROACHES):
            phi, glass, (cy, cx) = maps[(wl, name)]
            sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
            z = np.exp(1j*phi[sl])
            x = np.degrees(np.angle(z*np.conj(z[glass[sl]].mean())))
            im = ax[i, j].imshow(np.where(glass[sl], x, np.nan), cmap="RdBu", vmin=-15, vmax=15)
            ax[i, j].set_title(f"{name}, {wl} nm", fontsize=9)
            ax[i, j].axis("off")
    fig.colorbar(im, ax=ax, label="glass phase (deg)", shrink=0.6)
    fig.savefig(args.out/"glass_maps.png", dpi=90)

    # masks, to check that structures are excluded
    fig, ax = plt.subplots(1, len(args.wavelengths), figsize=(3.2*len(args.wavelengths), 3), squeeze=False)
    for j, wl in enumerate(args.wavelengths):
        ax[0, j].imshow(maps[(wl, "current")][1], cmap="gray")
        ax[0, j].set_title(f"glass mask, {wl} nm", fontsize=9)
        ax[0, j].axis("off")
    fig.savefig(args.out/"masks.png", dpi=90)


if __name__ == "__main__":
    main()
