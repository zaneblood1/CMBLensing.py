"""Precompute CAMB TT/PP Cls on a fixed 1D grid of any single LCDM parameter.

The 1D theta conditional in sample_lcdm.py only ever queries Cls along one
parameter's 50-point grid (all other parameters fixed at ground truth), and the Cls are
map-independent, so a single one-time sweep of CAMB calls serves every chain of an HPC
trial. This script runs that sweep for a chosen parameter (theta_MC_100, logA, ns, ombh2
or omch2), saves the results to camb_<param>_grid.npz, and provides
load_camb_spline_predictors(param_name) which returns jit-safe drop-in replacements for
the emulator's predict_tt/predict_pp batch functions (Cls evaluated via a cubic spline in
the swept parameter through the CAMB nodes).

Usage:
    python -m cmb_lensing.precompute_camb_1d ombh2          #full 50-point sweep (~10-20 min)
    python -m cmb_lensing.precompute_camb_1d ns --test      #3-point smoke test
    python -m cmb_lensing.precompute_camb_1d all            #sweep all five parameters

In the sampler (USE_CAMB_SPLINE = True in sample_lcdm.py), these replace the 5D grid
predictors for a single sampled parameter:
    from cmb_lensing.precompute_camb_1d import load_camb_spline_predictors
    predict_tt, predict_pp = load_camb_spline_predictors("ns")
    #same f(params_batch) -> (M, n_ell) signature as the camb_grid_interp predictors, so
    #every downstream call site (make_eval_logpdf_batch, _recompute_cosmo_matrices,
    #get_new_cf_matrix) works unchanged

The Cls are computed at ells 2..DEFAULT_MAX_ELL-1 (all real CAMB values, no
extrapolation) - the same support as load_sim's CAMB data-map path, so the model and the
data share the same log-log extrapolation anchor on the 2D grid. The covariance-build
call sites in sample_lcdm.py size their ells arrays from the predictor output.
"""

import argparse
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

from cmb_lensing.simulate import _camb_via_callback, _extract_all_cls
from cmb_lensing.constants import DEFAULT_MAX_ELL

#parameter ordering used by the sampler (must match sample_lcdm.PARAM_ORDER)
PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_ORDER)}

#ground-truth parameter values
GROUND_TRUTH = {
    "theta_MC_100": 1.031732,
    "logA": 3.218387,
    "ns": 0.959814,
    "ombh2": 0.022386,
    "omch2": 0.109381,
}

#Search ranges for each parameter
PARAM_BOUNDS = {
    "theta_MC_100": (0.9328, 1.1452),
    "logA": (2.661635, 3.782861),
    "ns": (0.867143, 1.042186),
    "ombh2": (0.020413, 0.024389),
    "omch2": (0.079704, 0.155541),
}
GRID_SIZE = 50
PARAM_GRIDS = {name: np.linspace(lo, hi, GRID_SIZE) for name, (lo, hi) in PARAM_BOUNDS.items()}

#sigma of each parameter (the bounds above are roughly mean +/- 5 sigma); used
#by sample_lcdm's progress plot to normalize the chain traces
PARAM_SIGMA = {
    "theta_MC_100": 0.0262018,
    "logA": 0.1128948,
    "ns": 0.0164744,
    "ombh2": 0.0004006,
    "omch2": 0.0059354,
}

#lmax = lmax_prime = DEFAULT_MAX_ELL makes dl2cl's ell grid arange(2, lmax) land on
#ells 2..DEFAULT_MAX_ELL-1 with identity interpolation - pure CAMB, no extrapolation.
#IMPORTANT: this must stay at DEFAULT_MAX_ELL (not +1) so the cache shares the exact
#ell support of load_sim's CAMB data-map path (also 2..DEFAULT_MAX_ELL-1). The model
#and the data then log-log extrapolate the 2D grid tail from the SAME anchor interval;
#a one-multipole support mismatch was measured to shift the clamped ombh2 conditional
#peak by ~+5e-5. The covariance call sites size their ells from the predictor output
CAMB_LMAX = DEFAULT_MAX_ELL

def default_cache_path(param_name):
    #cache lives next to this module so the same relative layout works on the laptop and
    #the cluster - no hardcoded absolute path to flip per machine
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "camb_splines", 
                        f"camb_{param_name}_grid.npz")

def camb_cls_at(param_name, value):
    """One CAMB run with param_name set to value and the other four parameters at their
    GROUND_TRUTH values. Returns (cl_tt, cl_pp) on ells 2..DEFAULT_MAX_ELL-1."""
    params = dict(GROUND_TRUTH)
    params[param_name] = float(value)
    cosmomc_theta = params["theta_MC_100"] / 100
    As = np.exp(params["logA"]) * 1e-10
    #fixed (non-sampled) CAMB parameters: H0=None, r=0, mnu=0.06, tau=0.05, nt=0,
    #k_pivot=0.05, Alens=1 - same as load_sim defaults
    unlensed_scalar, tensor, total, lens_potential = _camb_via_callback(
        None, params["ombh2"], params["omch2"], cosmomc_theta, 0.0, 0.06, 0.05,
        As, 0, params["ns"], CAMB_LMAX, 0.05, 1
    )
    cls = _extract_all_cls(unlensed_scalar, tensor, total, lens_potential,
                           CAMB_LMAX, CAMB_LMAX)
    return np.asarray(cls["scalar_TT"]), np.asarray(cls["phi"])


def run_sweep(param_name, grid, out_path):
    ells = np.arange(2, CAMB_LMAX).astype(np.float64)
    n_ell = ells.size
    kept_values = []
    kept_tt = []
    kept_pp = []

    t_start = time.time()
    for i, value in enumerate(grid):
        t0 = time.time()
        tt, pp = camb_cls_at(param_name, value)
        elapsed = time.time() - t0
        remaining = (grid.size - i - 1) * (time.time() - t_start) / (i + 1)
        if not (np.all(np.isfinite(tt)) and np.all(np.isfinite(pp))):
            #CAMB cannot solve some corners of the +/- 5 sigma box with the other four
            #parameters pinned at ground truth (e.g. theta_MC_100 > ~1.117 needs H0 > 100,
            #outside CAMB's H0 search range, so _camb_callback_fn returns NaNs). the
            #sampler's direct-CAMB path turns those NaNs into a rejected proposal, so
            #drop the node here; load_camb_spline_predictors reproduces the NaN behavior
            #for queries beyond the surviving grid
            print(f"[{i + 1:2d}/{grid.size}] {param_name} = {value:.6f}  SKIPPED "
                  f"(CAMB unsolvable at this value)", flush = True)
            continue
        if tt.size != n_ell or pp.size != n_ell:
            raise RuntimeError(f"unexpected Cl length {tt.size} (wanted {n_ell})")
        kept_values.append(value)
        kept_tt.append(tt)
        kept_pp.append(pp)
        print(f"[{i + 1:2d}/{grid.size}] {param_name} = {value:.6f}  "
              f"({elapsed:.1f}s, ~{remaining / 60:.1f} min remaining)", flush = True)

    if len(kept_values) < 4:
        raise RuntimeError(f"only {len(kept_values)} CAMB-solvable nodes on the "
                           f"{param_name} grid - not enough for a cubic spline")
    kept_grid = np.array(kept_values)
    cl_tt = np.array(kept_tt)
    cl_pp = np.array(kept_pp)

    #the grid is stored under "<param>_grid" and the fixed values under "fixed_<name>",
    #matching the layout of the original ombh2-only cache so old files keep loading
    save_kwargs = {f"{param_name}_grid": kept_grid, "ells": ells,
                   "cl_tt": cl_tt, "cl_pp": cl_pp,
                   "swept_param": param_name, "emulator_max_ell": DEFAULT_MAX_ELL}
    for name in PARAM_ORDER:
        if name != param_name:
            save_kwargs[f"fixed_{name}"] = GROUND_TRUTH[name]
    np.savez(out_path, **save_kwargs)
    print(f"saved {out_path} ({kept_grid.size}/{grid.size} nodes)")
    return kept_grid, cl_tt, cl_pp


def spline_accuracy_check(grid, cl_tt, cl_pp):
    """Leave-half-out check: spline through the even-index nodes, evaluate at the odd
    nodes, and report the worst |delta lnCl|. Needs no extra CAMB calls."""
    for name, cl in (("TT", cl_tt), ("PP", cl_pp)):
        spline = CubicSpline(grid[::2], np.log(cl[::2]), axis = 0)
        err = np.abs(spline(grid[1::2]) - np.log(cl[1::2]))
        print(f"spline check {name}: max |delta lnCl| at held-out nodes = {err.max():.2e}")


def load_camb_spline_predictors(param_name, path = None):
    """Returns (predict_tt, predict_pp) with the grid predictor signature
    f(params_batch) -> (M, n_ell), backed by cubic splines in param_name
    through the cached CAMB nodes. jit-safe via jax.pure_callback.

    Only the param_name column of params_batch is used - the cache is only valid while
    the other four parameters sit at their cached fixed values, so we assert they match."""
    if param_name not in PARAM_INDEX:
        raise ValueError(f"unknown parameter {param_name!r}; expected one of {PARAM_ORDER}")
    if path is None:
        path = default_cache_path(param_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no cached CAMB grid for {param_name} at {path} - "
                                f"generate it with: python -m cmb_lensing.precompute_camb_1d {param_name}")
    data = np.load(path)
    grid = data[f"{param_name}_grid"]
    n_ell = data["cl_tt"].shape[1]
    #spline in lnCl for positivity and smoothness; queries beyond the cached grid return
    #NaN (see _eval) rather than extrapolating
    spline_tt = CubicSpline(grid, np.log(data["cl_tt"]), axis = 0)
    spline_pp = CubicSpline(grid, np.log(data["cl_pp"]), axis = 0)
    other_names = [name for name in PARAM_ORDER if name != param_name]
    other_columns = [PARAM_INDEX[name] for name in other_names]
    fixed = np.array([float(data[f"fixed_{name}"]) for name in other_names])

    def _eval(spline, params_batch):
        pb = np.asarray(params_batch, dtype = np.float64)
        others = pb[:, other_columns]
        if not np.allclose(others, fixed[None, :], rtol = 0, atol = 1e-12):
            raise ValueError(f"cached CAMB grid only covers {param_name} variation but a "
                             f"non-{param_name} parameter moved off its cached fixed value")
        values = pb[:, PARAM_INDEX[param_name]]
        cls = np.exp(spline(values))
        #run_sweep drops CAMB-unsolvable nodes, so the cached grid can end short of
        #PARAM_BOUNDS (e.g. theta_MC_100 > ~1.117 needs H0 > 100). mirror
        #_camb_callback_fn beyond the surviving grid by returning NaN: the logpdf goes
        #non-finite and the proposal is rejected, exactly as the direct-CAMB path would
        cls[(values < grid[0]) | (values > grid[-1])] = np.nan
        return cls

    def make_predictor(spline):
        def predict(params_batch):
            M = params_batch.shape[0]
            return jax.pure_callback(
                lambda pb: _eval(spline, pb),
                jax.ShapeDtypeStruct((M, n_ell), jnp.float64),
                params_batch, vmap_method = "sequential"
            )
        return predict

    return make_predictor(spline_tt), make_predictor(spline_pp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Precompute CAMB Cls on a 1D parameter grid")
    parser.add_argument("param", choices = PARAM_ORDER + ["all"],
                        help = "parameter to sweep, or 'all' for every parameter")
    parser.add_argument("--test", action = "store_true",
                        help = "3-point smoke test written to a temporary file")
    parser.add_argument("--out", default = None,
                        help = "output .npz path (default camb_<param>_grid.npz next to "
                               "this module; ignored when sweeping 'all')")
    args = parser.parse_args()

    param_names = PARAM_ORDER if args.param == "all" else [args.param]
    for param_name in param_names:
        full_grid = PARAM_GRIDS[param_name]
        if args.test:
            grid = full_grid[[0, GRID_SIZE // 2, GRID_SIZE - 1]]
            out = args.out if (args.out and args.param != "all") \
                else f"/tmp/camb_{param_name}_grid_test.npz"
        else:
            grid = full_grid
            out = args.out if (args.out and args.param != "all") \
                else default_cache_path(param_name)
        kept_grid, cl_tt, cl_pp = run_sweep(param_name, grid, out)
        if not args.test:
            spline_accuracy_check(kept_grid, cl_tt, cl_pp)
