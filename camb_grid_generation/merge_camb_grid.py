import os
import glob
import zipfile
import argparse
import numpy as np
from cmb_lensing.camb_grid_interp import GRID_AXES, spline_coefficients

#Merges the per-job .npz slabs written by run_single_camb_grid.py into one file holding
#the 5D cubic spline of every spectrum over (H0, logA, ns, ombh2, omch2), which
#cmb_lensing.camb_grid_interp loads at sample time.
#
#Each job wrote one H0 sweep at a fixed (logA, ns, ombh2, omch2). Stacking them gives
#   lnCl[h0, logA, ns, ombh2, omch2, ell]
#and spline_coefficients (camb_grid_interp.py) turns that into the coefficients of the
#tensor-product cubic spline through it - a make_interp_spline along each of the five
#axes in turn. The coefficients and knots are what get saved, so loading the grid
#needs no further work.
#
#Splined in lnCl: unlensed TT / EE, lensed TT / EE / BB, and PP. TE crosses zero, so it
#is splined linearly as the correlation ratio te_rho = Cl_TE / sqrt(Cl_TT * Cl_EE); the
#sampler rebuilds Cl_TE = rho * sqrt(Cl_TT * Cl_EE) from its own TT / EE, which keeps the
#T/E block positive semi-definite. Unlensed BB is identically zero at r = 0 with tensors
#off, so it is not splined at all: the file records bb_is_zero = True instead.
#
#Usage:
#   python merge_camb_grid.py --grid_dir multi_param_CAMB_grid
#   python merge_camb_grid.py --grid_dir multi_param_CAMB_grid --out camb_grid_spline.npz

parser = argparse.ArgumentParser()
parser.add_argument("--grid_dir", type = str, required = True, help = "folder holding the grid_logA*_ns*_ombh2*_omch2*.npz slabs")
parser.add_argument("--out", type = str, default = None, help = "output file (default camb_grid_spline.npz inside grid_dir)")
parser.add_argument("--ell_chunk", type = int, default = 256, help = "ells splined at a time; lower this if memory is tight")
args = parser.parse_args()

out_path = args.out or os.path.join(args.grid_dir, "camb_grid_spline.npz")
slabs = sorted(glob.glob(os.path.join(args.grid_dir, "grid_logA*_ns*_ombh2*_omch2*.npz")))
print(f"found {len(slabs)} slabs in {args.grid_dir}")

#first pass: each slab's place in the grid and its parameter values. The indices are read
#from inside the file, not parsed from its name
first = np.load(slabs[0])
h0_grid = first["h0_grid"]
ells = first["ells"]
n_ell = ells.size
meta_keys = ["tau", "mnu", "lmax", "k_pivot", "alens",
             "accuracy_boost", "l_sample_boost", "l_accuracy_boost"]
ORDER = {"log_a": 0, "ns": 1, "ombh2": 2, "omch2": 3}
index_keys = ["log_a_index", "ns_index", "ombh2_index", "omch2_index"]
value_keys = ["log_a", "ns", "ombh2", "omch2"]

indices = np.zeros((len(slabs), 4), dtype = int)
values = np.zeros((len(slabs), 4))
for s in range(len(slabs)):
    z = np.load(slabs[s])
    for a in range(len(index_keys)):
        indices[s, a] = int(z[index_keys[a]])
        values[s, a] = float(z[value_keys[a]])

job_shape = tuple(indices[:, a].max() + 1 for a in range(len(index_keys)))
grid_shape = (h0_grid.size,) + job_shape
print(f"grid shape: H0 x logA x ns x ombh2 x omch2 = {grid_shape}, {n_ell} ells")

#the parameter axes: node i of axis a is the value of any slab sitting at index i
axes = [h0_grid]
for a in range(len(index_keys)):
    axis = np.zeros(job_shape[a])
    for s in range(len(slabs)):
        axis[indices[s, a]] = values[s, a]
    axes.append(axis)

#theta_MC_100 is a background quantity: it depends on H0, ombh2 and omch2 but not on logA
#or ns, so the logA = ns = node-0 slice holds all of it
theta_grid = np.zeros((h0_grid.size, job_shape[ORDER["ombh2"]], job_shape[ORDER["omch2"]]))
for s in range(len(slabs)):
    a, b, c, d = indices[s]
    if a == 0 and b == 0:
        theta_grid[:, c, d] = np.load(slabs[s])["theta_MC_100"]

#An npz is a zip of .npy files, so it can be written one member at a time: each
#coefficient table (~1.1 GB) is appended and freed before the next spectrum is read,
#so the tables never all sit in memory at once
def zip_write_array(zf, name, arr):
    with zf.open(name + ".npy", "w", force_zip64 = True) as f:
        np.lib.format.write_array(f, np.asarray(arr), allow_pickle = False)

SPLINED_SPECTRA = ["tt", "ee", "tt_lensed", "ee_lensed", "bb_lensed", "pp", "te_rho"]

print("building spline coefficients")
with zipfile.ZipFile(out_path, "w", compression = zipfile.ZIP_STORED, allowZip64 = True) as zf:
    for name in SPLINED_SPECTRA:
        #float32 keeps a table at ~1.1 GB and carries ~3e-6 absolute error on lnCl
        grid = np.zeros(grid_shape + (n_ell,), dtype = np.float32)
        for s in range(len(slabs)):
            a, b, c, d = indices[s]
            z = np.load(slabs[s])
            if name == "te_rho":
                grid[:, a, b, c, d, :] = z["cl_te"] / np.sqrt(z["cl_tt"] * z["cl_ee"])
            else:
                grid[:, a, b, c, d, :] = np.log(z[f"cl_{name}"])

        #the ell axis is never interpolated, so each block of ells is splined on its own
        #(in float64) and written back over the values it came from... We do this "chunking"
        #in ell to avoid possible OOM issues that may arise if it was all done in one shot
        for start in range(0, n_ell, args.ell_chunk):
            block = grid[..., start:start + args.ell_chunk].astype(np.float64)
            #knots depend only on the node positions (axes), never on the values, so every
            #chunk and spectrum returns the same knots; the last one is saved below
            coeff, knots = spline_coefficients(axes, block)
            grid[..., start:start + args.ell_chunk] = coeff
        print(f"  {name}: splined", flush = True)
        zip_write_array(zf, f"coeff_{name}", grid)
        del grid

    zip_write_array(zf, "bb_is_zero", True)
    for a in range(len(GRID_AXES)):
        zip_write_array(zf, f"axis_{GRID_AXES[a]}", axes[a])
        zip_write_array(zf, f"knots_{GRID_AXES[a]}", knots[a])
    zip_write_array(zf, "ells", ells)
    #theta_grid is stored as values, not coefficients: camb_grid_interp splines it at load
    zip_write_array(zf, "theta_grid", theta_grid)
    for k in meta_keys:
        zip_write_array(zf, k, first[k])

print(f"\nwrote {out_path} ({os.path.getsize(out_path) / 1024**3:.2f} GB)")
