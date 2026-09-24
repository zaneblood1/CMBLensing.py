"""How far the 5D CAMB grid's spectrum DERIVATIVES are from direct CAMB's, and what that
does to the 1st-principles forecast.

The chains run their likelihood through the grid spline (camb_grid_interp) while their data
come from direct CAMB, so the grid's dC/dtheta sets the posterior width and its value offset
at the fiducial point sets the bias. This script measures both against direct CAMB at
GROUND_TRUTH, for scalar TT, lensed TT and phiphi:

  1. value offset      C_grid / C_CAMB - 1 at the fiducial point (-> bias, not width)
  2. derivatives       dlnC/dtheta by central differences at the forecast's own step
                       (FD_STEP_FRAC * PARAM_SIGMA) from both sources, compared per band as
                       ||dlnC_grid - dlnC_CAMB|| / ||dlnC_CAMB||
  3. FD sanity check   for logA and ns (which do not touch the theta -> H0 inversion) the
                       grid's central difference against its ANALYTIC spline derivative, so
                       item 2 measures the grid and not the finite difference
  4. the forecast      forecast_from_1st_principles with cl_source = "camb" and "grid",
                       marginalized sigmas side by side

Memory: the grid is read through CambGrid.cl_local (memory-mapped, ~16 MB per spectrum per
point), so the whole run stays well under 1 GB beyond CAMB itself.

Usage:
    python -m cmb_lensing.compare_grid_derivatives
    python -m cmb_lensing.compare_grid_derivatives --spectra unlensed --ell_max 3000

Writes grid_vs_camb_derivatives.png and grid_vs_camb_derivatives.npz into
cmb_lensing/fisher_output/.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER, PARAM_SIGMA
from cmb_lensing.camb_grid_interp import load_camb_grid, GRID_AXES
from cmb_lensing.fisher_forecast import (camb_cls_at_params, grid_cls_at_params,
                                         CAMB_GRID_PATH, GRID_CLS_KEYS, FD_STEP_FRAC,
                                         covariance_from_fisher, output_dir)
from cmb_lensing.fisher_forecast_from_1st_principles import forecast_from_1st_principles

#the multipole bands the per-band comparison is reported in
BAND_EDGES = [2, 30, 100, 300, 600, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
#parameters whose grid derivative is a plain spline partial (no theta -> H0 inversion)
ANALYTIC_PARAMS = ("logA", "ns")


def stencil(name, step):
    up, down = dict(GROUND_TRUTH), dict(GROUND_TRUTH)
    up[name] += step
    down[name] -= step
    return up, down


def log_derivatives(source_fn, step_fracs):
    """{param: {cls key: dlnC/dtheta}} by central differences of ln C."""
    out = {}
    for name in PARAM_ORDER:
        step = step_fracs[name] * PARAM_SIGMA[name]
        up, down = stencil(name, step)
        plus, minus = source_fn(up), source_fn(down)
        out[name] = {key: (np.log(np.asarray(plus[key])) - np.log(np.asarray(minus[key])))
                          / (2 * step) for key in GRID_CLS_KEYS}
    return out


def band_error(ells, reference, other):
    """Per band ||other - reference|| / ||reference|| (RMS over the band's multipoles)."""
    errors = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        band = (ells >= lo) & (ells < hi)
        errors.append(np.sqrt(np.sum((other[band] - reference[band])**2)
                              / np.sum(reference[band]**2)))
    return np.array(errors)


def main():
    parser = argparse.ArgumentParser(description = "5D CAMB grid vs direct CAMB spectrum "
                                                   "derivatives and 1st-principles forecast")
    parser.add_argument("--nside", type = int, default = 128)
    parser.add_argument("--theta_pix", type = float, default = 2.5)
    parser.add_argument("--noise", type = float, default = 5.0)
    parser.add_argument("--spectra", choices = ("lensed", "unlensed"), default = "lensed")
    parser.add_argument("--ell_max", type = int, default = None,
                        help = "forecast ell_max (default CAMB's 3999, the grid's own end)")
    parser.add_argument("--sampled", nargs = "+", default = ["omch2", "theta_MC_100", "logA"],
                        help = "parameters forecast in item 4 (default: the chains' trio)")
    parser.add_argument("--skip_forecast", action = "store_true")
    args = parser.parse_args()

    grid = load_camb_grid(CAMB_GRID_PATH)
    ells = np.asarray(grid.ells, dtype = np.float64)
    print(f"grid axes: " + ", ".join(
        f"{name} {axis.size} nodes (spacing {(axis[1] - axis[0]) / PARAM_SIGMA.get(name, np.nan):.2f} sigma)"
        if name in PARAM_SIGMA else f"{name} {axis.size} nodes (spacing {axis[1] - axis[0]:.3g})"
        for name, axis in zip(GRID_AXES, grid.axes)))

    #1. value offset at the fiducial point
    camb_fid, grid_fid = camb_cls_at_params(GROUND_TRUTH), grid_cls_at_params(GROUND_TRUTH)
    print("\n1. value offset C_grid / C_CAMB - 1 at GROUND_TRUTH, band RMS:")
    offsets = {}
    for key in GRID_CLS_KEYS:
        ratio = np.asarray(grid_fid[key]) / np.asarray(camb_fid[key]) - 1
        offsets[key] = ratio
        rms = [np.sqrt(np.mean(ratio[(ells >= lo) & (ells < hi)]**2))
               for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
        print(f"   {key:10s} " + " ".join(f"{r:8.1e}" for r in rms))
    print("   bands: " + " ".join(f"{lo}-{hi}" for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])))

    #2. derivatives at the forecast's step
    fracs = dict(FD_STEP_FRAC)
    d_camb = log_derivatives(camb_cls_at_params, fracs)
    d_grid = log_derivatives(grid_cls_at_params, fracs)
    print("\n2. ||dlnC_grid - dlnC_CAMB|| / ||dlnC_CAMB|| per band (step 0.05 sigma):")
    errors = {}
    for key in GRID_CLS_KEYS:
        print(f"   {key}")
        for name in PARAM_ORDER:
            errors[(name, key)] = band_error(ells, d_camb[name][key], d_grid[name][key])
            print(f"     {name:13s} " + " ".join(f"{e:8.1e}" for e in errors[(name, key)]))

    #3. the grid FD against its analytic spline partial
    print("\n3. grid central difference vs analytic spline derivative (max |diff| / max |d|):")
    point = np.array([[GROUND_TRUTH[name] for name in PARAM_ORDER]])
    for name in ANALYTIC_PARAMS:
        axis = GRID_AXES.index(name)
        for key, spectrum in GRID_CLS_KEYS.items():
            analytic = grid.cl_local(spectrum, point, grid_axis = axis)[0]
            fd = d_grid[name][key]
            print(f"   {name:5s} {key:10s} {np.max(np.abs(fd - analytic)) / np.max(np.abs(analytic)):.1e}")

    np.savez(os.path.join(output_dir(), "grid_vs_camb_derivatives.npz"),
             ells = ells, band_edges = BAND_EDGES, params = PARAM_ORDER,
             keys = list(GRID_CLS_KEYS),
             offsets = np.array([offsets[k] for k in GRID_CLS_KEYS]),
             d_camb = np.array([[d_camb[n][k] for k in GRID_CLS_KEYS] for n in PARAM_ORDER]),
             d_grid = np.array([[d_grid[n][k] for k in GRID_CLS_KEYS] for n in PARAM_ORDER]))

    figure, axes = plt.subplots(len(GRID_CLS_KEYS), len(PARAM_ORDER),
                                figsize = (4 * len(PARAM_ORDER), 3.2 * len(GRID_CLS_KEYS)),
                                sharex = True)
    for row, key in enumerate(GRID_CLS_KEYS):
        for col, name in enumerate(PARAM_ORDER):
            axis = axes[row, col]
            axis.plot(ells, d_camb[name][key] * PARAM_SIGMA[name], lw = 1.2, label = "CAMB")
            axis.plot(ells, d_grid[name][key] * PARAM_SIGMA[name], lw = 0.8, ls = "--",
                      label = "5D grid")
            twin = axis.twinx()
            twin.plot(ells, (d_grid[name][key] - d_camb[name][key]) * PARAM_SIGMA[name],
                      color = "C3", lw = 0.5, alpha = 0.7)
            twin.tick_params(axis = "y", colors = "C3", labelsize = 7)
            axis.set_title(f"{key}: sigma dlnC/d{name}", fontsize = 9)
            axis.grid(alpha = 0.3)
            if row == len(GRID_CLS_KEYS) - 1:
                axis.set_xlabel(r"$\ell$")
    axes[0, 0].legend(fontsize = 8)
    figure.suptitle("Direct CAMB vs 5D grid spline, per-sigma log derivative at GROUND_TRUTH "
                    "(red, right axis: grid - CAMB)", fontsize = 10)
    figure.tight_layout()
    path = os.path.join(output_dir(), "grid_vs_camb_derivatives.png")
    figure.savefig(path, dpi = 130)
    plt.close(figure)
    print(f"\nwrote {path}")

    if args.skip_forecast:
        return
    #4. the forecast itself
    is_sampled = {name: name in args.sampled for name in PARAM_ORDER}
    sigmas = {}
    for source in ("camb", "grid"):
        fisher, names, _ = forecast_from_1st_principles(
            args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
            spectra = args.spectra, ell_max = args.ell_max, verbose = False,
            cl_source = source)
        covariance = covariance_from_fisher(fisher, names)
        sigmas[source] = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(sigmas[source], sigmas[source])
        sigmas[source + "_corr"] = correlation
    print(f"\n4. 1st-principles forecast, nside {args.nside} / {args.theta_pix}' / "
          f"{args.noise} uK, {args.spectra} TT + phiphi:")
    print(f"   {'param':13s} {'sigma CAMB':>11s} {'sigma grid':>11s} {'grid/CAMB-1':>12s}")
    for i, name in enumerate(names):
        print(f"   {name:13s} {sigmas['camb'][i]:11.4e} {sigmas['grid'][i]:11.4e} "
              f"{sigmas['grid'][i] / sigmas['camb'][i] - 1:+12.2%}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"   r({names[i]}, {names[j]}): CAMB {sigmas['camb_corr'][i, j]:+.3f}  "
                  f"grid {sigmas['grid_corr'][i, j]:+.3f}")


if __name__ == "__main__":
    main()
