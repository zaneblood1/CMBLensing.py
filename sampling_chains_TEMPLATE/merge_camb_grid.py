import os
import glob
import zipfile
import argparse
import numpy as np
from scipy.interpolate import make_interp_spline

#Merges the per-job .npz slabs written by run_single_camb_grid.py into a single 5D cubic
#spline over (H0, logA, ns, ombh2, omch2), and writes it to one file that
#cmb_lensing.camb_grid_interp loads at sample time.
#
#Each job wrote one H0 sweep at a fixed (logA, ns, ombh2, omch2). Stacking them gives
#   lnCl[h0, logA, ns, ombh2, omch2, ell]
#
#Splined spectra: unlensed TT / EE (the T + P sampler's field covariances), lensed
#TT / EE / BB (the theta-dependent quadratic-estimate / mass-matrix recompute), and PP -
#all in lnCl. Unlensed scalar BB is identically zero with r = 0 and tensors off, so it
#cannot go through the lnCl spline: it is verified to be zero at every node and recorded
#as the flag bb_is_zero instead of a coefficient table (the interpolator then returns
#exact zeros). If a future grid is generated with tensor power, the BB values spline
#like any other positive spectrum. TE crosses zero, so it is splined LINEARLY as the
#correlation ratio te_rho = Cl_TE / sqrt(Cl_TT * Cl_EE): bounded in (-1, 1), smooth in
#every parameter, and - because the sampler reconstructs
#Cl_TE = rho * sqrt(Cl_TT * Cl_EE) from the same splined TT/EE - the 2x2 T/E block
#stays positive semi-definite by construction wherever |rho| < 1.
#What gets saved is not the raw grid but its cubic B-spline *coefficients*: prefiltering
#here (once) turns evaluation into a purely local 4-point-per-axis contraction, which is
#what makes the interpolator fast enough to sit inside the sampler's inner loop. scipy's
#RegularGridInterpolator(method="cubic") was measured at minutes per call on this grid
#shape, so it is not an option.
#
#Usage:
#   python merge_camb_grid.py --grid_dir multi_param_CAMB_grid
#   python merge_camb_grid.py --grid_dir multi_param_CAMB_grid --out camb_grid_spline.npz

parser = argparse.ArgumentParser()
parser.add_argument("--grid_dir", type = str, required = True,
                    help = "folder holding the grid_logA*_ns*_ombh2*_omch2*.npz slabs")
parser.add_argument("--out", type = str, default = None,
                    help = "output file (default camb_grid_spline.npz inside grid_dir)")
parser.add_argument("--ell_chunk", type = int, default = 256,
                    help = "ells prefiltered at a time; lower this if memory is tight")
args = parser.parse_args()

out_path = args.out or os.path.join(args.grid_dir, "camb_grid_spline.npz")
slabs = sorted(glob.glob(os.path.join(args.grid_dir, "grid_logA*_ns*_ombh2*_omch2*.npz")))
if not slabs:
    raise RuntimeError(f"no grid slabs found in {args.grid_dir}")
print(f"found {len(slabs)} slabs in {args.grid_dir}")

#first pass: read only the small arrays to work out the grid shape and the axes. Indices
#are read from inside each file rather than parsed out of the file name so a renamed file
#can never silently land in the wrong slot
first = np.load(slabs[0])
h0_grid = first["h0_grid"]
ells = first["ells"]
n_h0, n_ell = h0_grid.size, ells.size

index_keys = ["log_a_index", "ns_index", "ombh2_index", "omch2_index"]
value_keys = ["log_a", "ns", "ombh2", "omch2"]
meta_keys = ["tau", "mnu", "lmax", "k_pivot", "alens",
             "accuracy_boost", "l_sample_boost", "l_accuracy_boost"]

indices = np.zeros((len(slabs), 4), dtype = int)
values = np.zeros((len(slabs), 4))
for s, path in enumerate(slabs):
    z = np.load(path)
    indices[s] = [int(z[k]) for k in index_keys]
    values[s] = [float(z[k]) for k in value_keys]
    if not np.array_equal(z["h0_grid"], h0_grid) or not np.array_equal(z["ells"], ells):
        raise RuntimeError(f"{os.path.basename(path)} has a different H0 or ell axis than "
                           f"{os.path.basename(slabs[0])} - the slabs are not from one run")
    for k in meta_keys:
        if not np.isclose(float(z[k]), float(first[k])):
            raise RuntimeError(f"{os.path.basename(path)} has {k} = {float(z[k])} but "
                               f"{os.path.basename(slabs[0])} has {float(first[k])} - "
                               f"the slabs are not from one run")

shape_4 = tuple(indices[:, a].max() + 1 for a in range(4))
n_log_a, n_ns, n_ombh2, n_omch2 = shape_4
expected = int(np.prod(shape_4))
print(f"grid shape: H0={n_h0} x logA={n_log_a} x ns={n_ns} x ombh2={n_ombh2} "
      f"x omch2={n_omch2}  ({n_h0 * expected} points, {n_ell} ells)")
if len(slabs) != expected:
    missing = expected - len(slabs)
    raise RuntimeError(f"expected {expected} slabs for that shape but found {len(slabs)} "
                       f"- {missing} job(s) never wrote output. Re-run them before merging")

#recover each parameter axis from the slab values and check every slab agrees on it.
#GRID_AXES is the axis ordering of the merged array and must match camb_grid_interp
GRID_AXES = ["H0", "logA", "ns", "ombh2", "omch2"]
axes = {"H0": h0_grid}
for a, name in enumerate(["logA", "ns", "ombh2", "omch2"]):
    axis = np.full(shape_4[a], np.nan)
    for s in range(len(slabs)):
        i = indices[s, a]
        if np.isnan(axis[i]):
            axis[i] = values[s, a]
        elif not np.isclose(axis[i], values[s, a], rtol = 0, atol = 1e-12):
            raise RuntimeError(f"slabs disagree on {name} node {i}: "
                               f"{axis[i]} vs {values[s, a]}")
    axes[name] = axis
    #the evaluator maps a query straight onto a fractional grid index, which is only valid
    #for a uniformly spaced axis. camb_grid.sh builds them all with linspace
    step = np.diff(axis)
    if axis.size > 2 and not np.allclose(step, step[0], rtol = 1e-9, atol = 0):
        raise RuntimeError(f"the {name} axis is not uniformly spaced - the interpolator "
                           f"assumes uniform axes")

#second pass: fill the 5D arrays. float32 keeps the merged file at ~1.1 GB per observable
#and carries ~3e-6 absolute error on lnCl, far below the spline and CAMB error budgets.
#SPLINED_SPECTRA go through the lnCl spline; unlensed BB is handled separately below.
#
#MEMORY: each spectrum's grid is ~1.1 GB at the production shape and there are six of
#them, so nothing is ever held for more than one spectrum at a time - the slabs are
#re-read once per spectrum (cheap: np.load on an npz reads only the requested member),
#the prefilter runs IN PLACE, and each finished coefficient table is appended to the
#output npz and freed before the next spectrum starts. Holding all inputs + outputs at
#once (the old flow) peaked at ~13 GB and OOM-killed 16 GB machines
SPLINED_SPECTRA = ["tt", "ee", "tt_lensed", "ee_lensed", "bb_lensed", "pp"]
#linearly-splined quantities, appended after the lnCl spectra: the TE correlation ratio,
#computed per node from three slab members
LINEAR_SPECTRA = ["te_rho"]
missing_keys = [f"cl_{name}" for name in SPLINED_SPECTRA + ["bb", "te"]
                if f"cl_{name}" not in first]
if missing_keys:
    raise RuntimeError(f"{os.path.basename(slabs[0])} is missing {missing_keys} - these "
                       f"slabs are from a pre-polarization run of "
                       f"run_single_camb_grid.py. Re-run camb_grid.sh (all jobs) so every "
                       f"slab records the unlensed EE/BB and lensed TT/EE/BB spectra")

grid_shape = (n_h0,) + shape_4
bb_max = 0.0
theta_5d = np.full(grid_shape, np.nan)
ok = np.zeros(grid_shape, dtype = bool)

#small-array pass: ok flags, theta, and the unlensed-BB zero check
for s, path in enumerate(slabs):
    z = np.load(path)
    a, b, c, d = indices[s]
    bb_max = max(bb_max, np.nanmax(np.abs(z["cl_bb"])))
    theta_5d[:, a, b, c, d] = z["theta_MC_100"]
    ok[:, a, b, c, d] = z["ok"]
    if (s + 1) % 100 == 0 or s + 1 == len(slabs):
        print(f"  read {s + 1}/{len(slabs)} slabs", flush = True)

#unlensed scalar BB: with r = 0 and tensors off it is exactly zero everywhere, and the
#interpolator then returns exact zeros (bb_is_zero flag) rather than log-splining it.
#A grid generated with tensor power would spline it like the others - not implemented
#until such a grid exists, so fail loudly rather than silently mishandling it
if bb_max > 0:
    raise RuntimeError(f"unlensed BB is nonzero (max {bb_max:.3e}) - this merge only "
                       f"supports the r = 0 / tensors-off grids where scalar BB is "
                       f"identically zero. Extend SPLINED_SPECTRA to cover nonzero BB")
bb_is_zero = True

n_bad = int((~ok).sum())
if n_bad:
    #a cubic spline cannot be built across a hole, so this is fatal rather than a warning
    bad = np.argwhere(~ok)[:10]
    raise RuntimeError(f"{n_bad} grid points are missing (CAMB failed there). A tensor "
                       f"product spline cannot be built over holes. First few "
                       f"(h0, logA, ns, ombh2, omch2) indices:\n{bad}")

#theta_MC_100 is a background quantity, so it must not depend on logA or ns. Collapsing it
#to (H0, ombh2, omch2) both shrinks the table and checks that assumption
theta_3d = theta_5d[:, 0, 0, :, :]
spread = np.nanmax(np.abs(theta_5d - theta_3d[:, None, None, :, :]))
if spread > 1e-10:
    raise RuntimeError(f"theta_MC_100 varies by {spread:.2e} across the logA/ns axes; it "
                       f"should depend only on H0, ombh2 and omch2")
print(f"theta_MC_100 spans [{theta_3d.min():.4f}, {theta_3d.max():.4f}] over the grid "
      f"(independent of logA/ns to {spread:.1e})")

#Turn the sampled values into cubic B-spline coefficients along the five parameter axes
#(never along ell - the ell axis is only ever indexed, not interpolated).
#
#make_interp_spline's default not-a-knot end condition is what makes this accurate. The
#obvious alternative, scipy.ndimage.spline_filter1d, is faster but every boundary mode it
#offers folds the data back on itself at the edges, which is not exact even for a straight
#line: measured at 1.9e-4 in lnCl on a 4-node axis, i.e. the size of the whole error
#budget. With axes this short the boundary is never far away, so that error would
#contaminate the entire axis rather than just its edges.
#
#Chunked over ell so the float64 working copy stays small; the prefilter overwrites the
#input array chunk by chunk (each ell chunk is independent), so no second ~1.1 GB output
#allocation is ever made.
def build_coefficients_inplace(data, label):
    knots = None
    for start in range(0, n_ell, args.ell_chunk):
        block = data[..., start:start + args.ell_chunk].astype(np.float64)
        axis_knots = []
        for axis in range(5):
            spline = make_interp_spline(axes[GRID_AXES[axis]],
                                        np.moveaxis(block, axis, 0), k = 3)
            block = np.moveaxis(spline.c, 0, axis)
            axis_knots.append(spline.t)
        knots = axis_knots
        data[..., start:start + args.ell_chunk] = block.astype(np.float32)
        print(f"  {label}: splined ells {start}..{min(start + args.ell_chunk, n_ell)}",
              flush = True)
    return knots

#An npz is a zip of .npy members, so it can be written INCREMENTALLY: each coefficient
#table is appended as soon as it is built and freed before the next spectrum's grid is
#allocated. np.load reads the result exactly as if np.savez had written it in one shot
def zip_write_array(zf, name, arr):
    #a 0-d array is already contiguous, and np.ascontiguousarray would promote it to shape
    #(1,) - which then breaks float()/int() on the reader side under numpy 2
    arr = np.asarray(arr)
    if arr.ndim > 0:
        arr = np.ascontiguousarray(arr)
    with zf.open(name + ".npy", "w", force_zip64 = True) as f:
        np.lib.format.write_array(f, arr, allow_pickle = False)

print("building spline coefficients")
knots = None
coeff_shape = None
with zipfile.ZipFile(out_path, "w", compression = zipfile.ZIP_STORED,
                     allowZip64 = True) as zf:
    for name in SPLINED_SPECTRA + LINEAR_SPECTRA:
        #fill this spectrum's grid from the slabs (only the needed members are read per
        #file); lnCl for the positive spectra, the raw correlation ratio for te_rho
        lncl = np.full(grid_shape + (n_ell,), np.nan, dtype = np.float32)
        for s, path in enumerate(slabs):
            a, b, c, d = indices[s]
            z = np.load(path)
            if name == "te_rho":
                rho = z["cl_te"] / np.sqrt(z["cl_tt"] * z["cl_ee"])
                lncl[:, a, b, c, d, :] = rho.astype(np.float32)
            else:
                lncl[:, a, b, c, d, :] = np.log(z[f"cl_{name}"]).astype(np.float32)
        if name == "te_rho":
            rho_max = float(np.nanmax(np.abs(lncl)))
            if rho_max >= 1:
                raise RuntimeError(f"|te_rho| reaches {rho_max:.6f} >= 1 somewhere on "
                                   f"the grid - the T/E block would not be positive "
                                   f"definite there")
            print(f"  te_rho: max |rho| over the grid = {rho_max:.4f}", flush = True)
        knots = build_coefficients_inplace(lncl, name)
        coeff_shape = lncl.shape
        zip_write_array(zf, f"coeff_{name}", lncl)
        del lncl
        print(f"  {name}: written", flush = True)

    #theta_grid is saved raw, not prefiltered: the interpolator densifies it along H0
    #with a plain 1D cubic (the ombh2/omch2 slots are exact grid nodes) before
    #inverting it
    zip_write_array(zf, "bb_is_zero", np.asarray(bb_is_zero))
    for axis_name in GRID_AXES:
        zip_write_array(zf, f"axis_{axis_name}", axes[axis_name])
    for i, axis_name in enumerate(GRID_AXES):
        zip_write_array(zf, f"knots_{axis_name}", knots[i])
    zip_write_array(zf, "ells", ells)
    zip_write_array(zf, "theta_grid", theta_3d)
    for k in meta_keys:
        zip_write_array(zf, k, np.asarray(first[k]))

size_gb = os.path.getsize(out_path) / 1024**3
print(f"\nwrote {out_path} ({size_gb:.2f} GB)")
print(f"  spectra {SPLINED_SPECTRA} + {LINEAR_SPECTRA} + zero unlensed BB, coefficients "
      f"{coeff_shape}, ells {ells[0]:.0f}..{ells[-1]:.0f}")
print(f"  load it with cmb_lensing.camb_grid_interp.load_camb_grid_predictors("
      f"\"{out_path}\")")
