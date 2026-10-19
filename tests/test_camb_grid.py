"""Acceptance test for the merged 5D CAMB grid: draw random points strictly inside the
box, run CAMB directly there, and compare against what the spline predicts. Every grid
node is reproduced exactly by construction, so the only honest check is at points the
grid has never seen.

The points are drawn in the grid's own coordinates (H0, logA, ns, ombh2, omch2), then
queried in the sampler's coordinates (theta_MC_100, ...) using the theta CAMB reports, so
the theta -> H0 inversion is exercised too.

    pytest tests/test_camb_grid.py                      #a few points, skipped without the grid
    python tests/test_camb_grid.py --grid <...>/camb_grid_spline.npz -n 20

What to expect: interpolation error below CAMB's own accuracy floor, which at default
boosts is ~9e-4 in lnCl^TT and ~1.6e-3 in lnCl^PP (measured against an AccuracyBoost = 2
reference). If TT is much worse, the H0 axis is too coarse: moving H0 shifts the acoustic
peaks in ell, so it is the axis the node count is most sensitive to.

Memory: one spectrum's spline is ~2.1 GB, so the spectra are evaluated one at a time and
each spline is dropped before the next is built.
"""

import os
import time
import argparse
import numpy as np
import pytest

from cmb_lensing.camb_grid_interp import CambGrid
from cmb_lensing.constants import CAMB_GRID_PATH

#unlensed BB is exact zeros by construction, so there is nothing to compare there.
#te_rho crosses zero, so its error is the absolute |delta rho| rather than |delta lnCl|
CHECKED_SPECTRA = ["tt", "ee", "tt_lensed", "ee_lensed", "bb_lensed", "pp", "te_rho"]

#pass mark of the pytest run (seed 0, 5 points), per spectrum, on the max error over all
#points and ells: ~2x what the real grid measured on 2026-10-19 (tt 1.3e-3, ee 4.7e-3 at
#an EE trough near ell 470, tt_lensed 1.1e-3, ee_lensed 4.3e-3, bb_lensed 3.0e-4,
#pp 4.7e-5, te_rho 2.4e-3; medians 2e-5 - 2e-4)
MAX_ERROR = {"tt": 3e-3, "ee": 1e-2, "tt_lensed": 3e-3, "ee_lensed": 1e-2,
             "bb_lensed": 1e-3, "pp": 1e-4, "te_rho": 5e-3}


def camb_cls_at(meta, h0, log_a, ns, ombh2, omch2):
    """CAMB with the grid's own settings (identical to run_single_camb_grid.py), so any
    difference is interpolation error and not a configuration mismatch."""
    import camb
    lmax = int(meta["lmax"].item())
    pars = camb.set_params(
        H0 = h0, ombh2 = ombh2, omch2 = omch2, cosmomc_theta = None,
        mnu = meta["mnu"].item(), As = np.exp(log_a) * 1e-10, ns = ns, lmax = lmax,
        tau = meta["tau"].item(), pivot_scalar = meta["k_pivot"].item(),
        pivot_tensor = meta["k_pivot"].item(), Alens = meta["alens"].item(),
        AccuracyBoost = meta["accuracy_boost"].item(),
        lSampleBoost = meta["l_sample_boost"].item(),
        lAccuracyBoost = meta["l_accuracy_boost"].item()
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


def grid_errors(grid_path, n_points, seed = 0, margin = 0.05, ell_max = None):
    """Per spectrum, an (n_points, n_ell) array of |spline - CAMB| (in lnCl, or in rho for
    te_rho), plus the ells compared."""
    grid = CambGrid(grid_path)
    ells = grid.ells
    if ell_max is None:
        n_ell = ells.size
    else:
        n_ell = int(np.searchsorted(ells, ell_max))

    #CAMB at every point first, so only one spline is ever in memory below
    rng = np.random.default_rng(seed)
    queries = []
    camb_cls = []
    for i in range(n_points):
        #stay clear of the edges, so this measures interpolation and not end conditions
        point = []
        for axis in grid.axes:
            span = axis[-1] - axis[0]
            point.append(rng.uniform(axis[0] + margin * span, axis[-1] - margin * span))
        h0, log_a, ns, ombh2, omch2 = point
        cl, theta = camb_cls_at(grid.data, h0, log_a, ns, ombh2, omch2)
        queries.append([theta, log_a, ns, ombh2, omch2])
        camb_cls.append(cl)
        print(f"[{i + 1}/{n_points}] CAMB at H0={h0:7.3f} theta={theta:.5f} logA={log_a:.4f} "
              f"ns={ns:.4f} ombh2={ombh2:.5f} omch2={omch2:.5f}", flush = True)
    queries = np.array(queries)

    errors = {}
    for name in CHECKED_SPECTRA:
        got = grid.cl(name, queries)[:, :n_ell]
        want = np.array([cl[name][:n_ell] for cl in camb_cls])
        if name == "te_rho":
            errors[name] = np.abs(got - want)
        else:
            errors[name] = np.abs(np.log(got) - np.log(want))
        #drop the ~2.1 GB spline before the next spectrum's is built
        grid.spline_cache.clear()
        print(f"  {name}: max error {errors[name].max():.2e}", flush = True)
    return errors, ells[:n_ell]


@pytest.mark.skipif(not os.path.exists(CAMB_GRID_PATH), reason = "no merged CAMB grid")
def test_grid_matches_camb():
    errors, ells = grid_errors(CAMB_GRID_PATH, n_points = 5)
    for name in CHECKED_SPECTRA:
        #NaN would mean a point fell outside the box or theta was not inverted
        assert np.all(np.isfinite(errors[name])), name
        assert errors[name].max() < MAX_ERROR[name], (name, errors[name].max())


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type = str, default = CAMB_GRID_PATH,
                        help = "merged camb_grid_spline.npz")
    parser.add_argument("-n", "--n_points", type = int, default = 20,
                        help = "random interior points to test")
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--margin", type = float, default = 0.05,
                        help = "stay this fraction of each axis away from its edges")
    parser.add_argument("--ell_max", type = int, default = None,
                        help = "only compare ells below this (default: all)")
    args = parser.parse_args()

    t_start = time.time()
    errors, ells = grid_errors(args.grid, args.n_points, args.seed, args.margin, args.ell_max)
    print(f"\n{'':24s}{'max':>10s}{'median':>10s}{'  worst ell':>14s}")
    for name in CHECKED_SPECTRA:
        err = errors[name]
        worst_ell = ells[np.unravel_index(np.nanargmax(err), err.shape)[1]]
        if name == "te_rho":
            label = "|drho| te_rho"
        else:
            label = f"lnCl {name}"
        print(f"{label:24s}{np.nanmax(err):10.2e}{np.nanmedian(err):10.2e}{worst_ell:14.0f}")
    print(f"\n{args.n_points} points compared in {(time.time() - t_start) / 60:.1f} min")
