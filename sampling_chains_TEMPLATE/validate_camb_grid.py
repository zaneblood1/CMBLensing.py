import argparse
import time
import numpy as np
import camb

from cmb_lensing.camb_grid_interp import CambGrid

#Acceptance test for a merged CAMB grid: draw random points strictly inside the box, run
#CAMB directly there, and compare against what the spline predicts. Every grid node is
#reproduced exactly by construction, so the only honest check is at points the grid has
#never seen.
#
#The points are drawn in the grid's own coordinates (H0, logA, ns, ombh2, omch2), then the
#query is made in the sampler's coordinates (theta_MC_100, ...) using the theta CAMB
#reports. That exercises the whole path the sampler uses, theta -> H0 inversion included.
#
#Usage:
#   python validate_camb_grid.py --grid multi_param_CAMB_grid/camb_grid_spline.npz -n 20
#
#What to expect: the interpolation error should sit below CAMB's own accuracy floor, which
#at default boosts is ~9e-4 in lnCl^TT and ~1.6e-3 in lnCl^PP (measured against an
#AccuracyBoost=2 reference). If TT is much worse than that, the H0 axis is too coarse -
#that is the axis the node count is most sensitive to, because moving it shifts the
#acoustic peaks in ell.

parser = argparse.ArgumentParser()
parser.add_argument("--grid", type = str, required = True,
                    help = "merged camb_grid_spline.npz")
parser.add_argument("-n", "--n_points", type = int, default = 20,
                    help = "random interior points to test")
parser.add_argument("--seed", type = int, default = 0)
parser.add_argument("--margin", type = float, default = 0.05,
                    help = "stay this fraction of each axis away from its edges")
parser.add_argument("--ell_max", type = int, default = None,
                    help = "only compare ells below this (default: all)")
args = parser.parse_args()

grid = CambGrid(args.grid)
meta = np.load(args.grid)
tau, mnu = float(meta["tau"]), float(meta["mnu"])
lmax, k_pivot, alens = int(meta["lmax"]), float(meta["k_pivot"]), float(meta["alens"])
accuracy = {"AccuracyBoost": float(meta["accuracy_boost"]),
            "lSampleBoost": float(meta["l_sample_boost"]),
            "lAccuracyBoost": float(meta["l_accuracy_boost"])}
ells = grid.ells
sl = slice(None) if args.ell_max is None else slice(0, int(np.searchsorted(ells, args.ell_max)))

print(f"grid  : {args.grid}")
print(f"shape : {' x '.join(f'{name}={axis.size}' for name, axis in zip(['H0', 'logA', 'ns', 'ombh2', 'omch2'], grid.axes))}")
print(f"CAMB  : tau={tau} mnu={mnu} lmax={lmax} accuracy={accuracy}")
print(f"comparing ells {ells[sl][0]:.0f}..{ells[sl][-1]:.0f} at {args.n_points} random "
      f"interior points\n")


#every splined spectrum the grid file holds (an old TT/PP-only file is still validated
#on what it has). Unlensed BB is exact zeros by construction, so there is nothing to
#compare there. te_rho is sign-changing, so its error is reported as the ABSOLUTE
#|delta rho| (which is exactly the error metric that matters for the T/E block) rather
#than |delta lnCl|
CHECKED_SPECTRA = [name for name in ["tt", "ee", "tt_lensed", "ee_lensed", "bb_lensed",
                                     "pp", "te_rho"]
                   if grid.has_spectrum(name)]
print(f"checking spectra: {CHECKED_SPECTRA}\n")


def camb_cls_at(h0, log_a, ns, ombh2, omch2):
    """Identical settings to run_single_camb_grid.py, so any difference is interpolation
    error and not a configuration mismatch."""
    pars = camb.set_params(
        H0 = h0, ombh2 = ombh2, omch2 = omch2, cosmomc_theta = None,
        mnu = mnu, As = np.exp(log_a) * 1e-10, ns = ns, lmax = lmax,
        tau = tau, pivot_scalar = k_pivot, pivot_tensor = k_pivot, Alens = alens,
        **accuracy
    )
    pars.WantScalars = True
    pars.WantTensors = False
    pars.DoLensing = True
    pars.set_nonlinear_lensing(True)
    results = camb.get_results(pars)
    power_spectra = results.get_cmb_power_spectra(pars, lmax = lmax - 1, CMB_unit = "muK")
    lens_potential = results.get_lens_potential_cls(lmax = lmax - 1)[:, 0]
    ell = np.arange(2, lmax).astype(np.float64)
    cl = {}
    for col, stokes in enumerate(["tt", "ee", "bb", "te"]):
        cl[stokes] = power_spectra["unlensed_scalar"][2:, col] * 2 * np.pi / (ell * (ell + 1))
        cl[f"{stokes}_lensed"] = power_spectra["total"][2:, col] * 2 * np.pi / (ell * (ell + 1))
    cl["pp"] = lens_potential[2:] * 2 * np.pi / ell**4
    cl["te_rho"] = cl["te"] / np.sqrt(cl["tt"] * cl["ee"])
    return cl, results.cosmomc_theta() * 100


rng = np.random.default_rng(args.seed)
errors = {name: [] for name in CHECKED_SPECTRA}
t_start = time.time()
for i in range(args.n_points):
    #draw inside the box, keeping clear of the edges so this measures interpolation rather
    #than end-condition behaviour
    point = [rng.uniform(axis[0] + args.margin * (axis[-1] - axis[0]),
                         axis[-1] - args.margin * (axis[-1] - axis[0]))
             for axis in grid.axes]
    h0, log_a, ns, ombh2, omch2 = point
    try:
        cl, theta = camb_cls_at(h0, log_a, ns, ombh2, omch2)
    except Exception as exc:
        print(f"[{i + 1}/{args.n_points}] CAMB failed at H0={h0:.3f}: {exc}")
        continue

    query = np.array([[theta, log_a, ns, ombh2, omch2]])
    got = {name: grid.cl(name, query)[0] for name in CHECKED_SPECTRA}
    if not np.all(np.isfinite(got["tt"])):
        print(f"[{i + 1}/{args.n_points}] spline returned NaN at theta={theta:.5f} "
              f"(H0={h0:.3f}) - outside the box?")
        continue

    point_errors = {name: (np.abs(got[name][sl] - cl[name][sl])
                           if name == "te_rho"
                           else np.abs(np.log(got[name][sl]) - np.log(cl[name][sl])))
                    for name in CHECKED_SPECTRA}
    for name in CHECKED_SPECTRA:
        errors[name].append(point_errors[name])
    summary = "  ".join(f"{name} {point_errors[name].max():.2e}"
                        for name in CHECKED_SPECTRA)
    print(f"[{i + 1}/{args.n_points}] H0={h0:7.3f} theta={theta:.5f} logA={log_a:.4f} "
          f"ns={ns:.4f} ombh2={ombh2:.5f} omch2={omch2:.5f}  {summary}", flush = True)

if not errors["tt"]:
    raise SystemExit("no points could be compared")

print(f"\n{'':24s}{'max':>10s}{'median':>10s}{'  worst ell':>14s}")
for name in CHECKED_SPECTRA:
    err = np.array(errors[name])
    worst_ell = ells[sl][np.unravel_index(err.argmax(), err.shape)[1]]
    label = "|drho| te_rho" if name == "te_rho" else f"lnCl {name}"
    print(f"{label:24s}{err.max():10.2e}{np.median(err):10.2e}{worst_ell:14.0f}")
print(f"\n{len(errors['tt'])} points compared in {(time.time() - t_start) / 60:.1f} min")
