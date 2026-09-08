import os
import argparse
import time
import numpy as np
import camb

#Computes one H0 sweep of the 5D CAMB grid at a fixed (logA, ns, ombh2, omch2) and writes
#it to a single .npz in the shared grid folder. One slurm job per call; camb_grid.sh
#spawns one of these per (logA, ns, ombh2, omch2) combination.
#
#Deliberately imports only numpy and camb - no JAX, no cmb_lensing - so 875 concurrent
#jobs start fast and stay small in memory.

#Parse the arguments of the slurm job
parser = argparse.ArgumentParser()
parser.add_argument("--log_a_index", type = int)
parser.add_argument("--log_a", type = float)
parser.add_argument("--ns_index", type = int)
parser.add_argument("--ns", type = float)
parser.add_argument("--ombh2_index", type = int)
parser.add_argument("--ombh2", type = float)
parser.add_argument("--omch2_index", type = int)
parser.add_argument("--omch2", type = float)
parser.add_argument("--h0_min", type = float)
parser.add_argument("--h0_max", type = float)
parser.add_argument("--h0_nodes", type = int)
parser.add_argument("--lmax", type = int)
parser.add_argument("--tau", type = float)
parser.add_argument("--mnu", type = float)
parser.add_argument("--k_pivot", type = float)
parser.add_argument("--alens", type = float)
parser.add_argument("--accuracy_boost", type = float)
parser.add_argument("--l_sample_boost", type = float)
parser.add_argument("--l_accuracy_boost", type = float)
parser.add_argument("--out_dir", type = str)
args = parser.parse_args()

#Dl -> Cl on ells 2..lmax-1. This is simulate.dl2cl specialized to lmax == lmax_prime,
#where its log-log interpolation onto the target ell grid is the identity and drops out
#(checked against the real dl2cl: agrees to 1e-15). Kept in numpy so this script never
#imports JAX. Because lmax matches EMULATOR_MAX_ELL, the grid lands on exactly the ell
#support of load_sim's CAMB data-map path - model and data then share the same log-log
#extrapolation anchor, and a one-multipole mismatch here measurably biases the conditional
def dl2cl(dl_xx, ells, is_phi = False):
    if is_phi:
        return dl_xx[2:] * 2 * np.pi / ells**4
    return dl_xx[2:] * 2 * np.pi / (ells * (ells + 1))

#One CAMB call at the given H0. Mirrors simulate._run_camb with two changes: H0 is passed
#directly instead of solving for it from cosmomc_theta, and the tensor sector is off since
#r = 0 (it was being computed and then multiplied by zero). Checked against the existing
#theta-solve path at the fiducial cosmology: agrees to 2.7e-5 in lnCl^TT and 7.6e-6 in
#lnCl^PP, well below both the spline budget and CAMB's own accuracy floor.
#
#Beyond TT and PP this now records the full unlensed-scalar AND lensed (total) spectra so
#the merged grid can serve polarization sampling (EE, BB, TE) and theta-dependent
#quadratic-estimate / mass-matrix recomputes (lensed TT, EE, BB) from the same
#interpolation engine. With WantTensors = False and r = 0, unlensed scalar BB is exactly
#zero at every node - it is recorded anyway so the merge step can verify and flag that.
#TE (unlensed and lensed) crosses zero so it cannot go through the lnCl spline; the
#merge splines the unlensed TE as the correlation ratio te_rho = TE / sqrt(TT * EE)
#(linear, not log); the lensed TE is stored raw only for now
SPECTRA = ["tt", "ee", "bb", "te", "tt_lensed", "ee_lensed", "bb_lensed", "te_lensed",
           "pp"]
#spectra that must be strictly positive at every ell for the lnCl spline to exist.
#bb is identically zero (checked separately), te changes sign, so neither is here
POSITIVE_SPECTRA = ["tt", "ee", "tt_lensed", "ee_lensed", "bb_lensed", "pp"]

def camb_cls_at(h0, ells):
    pars = camb.set_params(
        H0 = h0, ombh2 = args.ombh2, omch2 = args.omch2, cosmomc_theta = None,
        mnu = args.mnu, As = np.exp(args.log_a) * 1e-10, ns = args.ns, lmax = args.lmax,
        tau = args.tau, pivot_scalar = args.k_pivot, pivot_tensor = args.k_pivot,
        Alens = args.alens, AccuracyBoost = args.accuracy_boost,
        lSampleBoost = args.l_sample_boost, lAccuracyBoost = args.l_accuracy_boost
    )
    pars.WantScalars = True
    pars.WantTensors = False
    pars.DoLensing = True
    pars.set_nonlinear_lensing(True)

    results = camb.get_results(pars)
    power_spectra = results.get_cmb_power_spectra(pars, lmax = args.lmax - 1, CMB_unit = "muK")
    lens_potential = results.get_lens_potential_cls(lmax = args.lmax - 1)[:, 0]
    #column order matches simulate._CAMB_COLS: TT, EE, BB, TE. "total" is what
    #simulate._extract_all_cls uses for the lensed spectra (== lensed_scalar here since
    #tensors are off), so the grid stays consistent with load_sim's data-map path
    unlensed = power_spectra["unlensed_scalar"]
    total = power_spectra["total"]
    cl = {}
    for col, stokes in enumerate(["tt", "ee", "bb", "te"]):
        cl[stokes] = dl2cl(unlensed[:, col], ells)
        cl[f"{stokes}_lensed"] = dl2cl(total[:, col], ells)
    cl["pp"] = dl2cl(lens_potential, ells, is_phi = True)
    #the sampler works in theta_MC_100 but this grid is laid out in H0, so the
    #interpolator needs theta(H0, ombh2, omch2) to convert. The background is already
    #solved at this point so reading it back costs ~0.3 ms
    theta_MC_100 = results.cosmomc_theta() * 100
    return cl, theta_MC_100

ells = np.arange(2, args.lmax).astype(np.float64)
h0_grid = np.linspace(args.h0_min, args.h0_max, args.h0_nodes)


#a CAMB failure leaves its row NaN and clears the ok flag rather than killing the job, so
#one bad corner does not cost the whole H0 sweep. The merge step can then see the hole
cls = {name: np.full((args.h0_nodes, ells.size), np.nan) for name in SPECTRA}
theta_MC_100 = np.full(args.h0_nodes, np.nan)
ok = np.zeros(args.h0_nodes, dtype = bool)

t_start = time.time()
for i, h0 in enumerate(h0_grid):
    try:
        cl, theta = camb_cls_at(h0, ells)
        for name in POSITIVE_SPECTRA:
            if not np.all(cl[name] > 0):
                raise RuntimeError(f"non-positive Cl^{name}")
        if not np.all(cl["bb"] >= 0):
            raise RuntimeError("negative unlensed BB")
        if not (np.all(np.isfinite(cl["te"])) and np.all(np.isfinite(cl["te_lensed"]))):
            raise RuntimeError("non-finite TE")
        for name in SPECTRA:
            cls[name][i] = cl[name]
        theta_MC_100[i] = theta
        ok[i] = True
    except Exception as exc:
        print(f"[{i + 1}/{args.h0_nodes}] H0 = {h0:.4f} FAILED: {exc}", flush = True)
        continue
    elapsed = time.time() - t_start
    remaining = (args.h0_nodes - i - 1) * elapsed / (i + 1)
    print(f"[{i + 1}/{args.h0_nodes}] H0 = {h0:.4f}  "
          f"({elapsed / (i + 1):.1f}s/call, ~{remaining / 60:.1f} min remaining)", flush = True)

#indices are in the file name so the merge step can place this slab in the 5D grid
#without re-deriving the axes from the parameter values
os.makedirs(args.out_dir, exist_ok = True)
file_name = (f"grid_logA{args.log_a_index}_ns{args.ns_index}"
             f"_ombh2{args.ombh2_index}_omch2{args.omch2_index}.npz")
np.savez(os.path.join(args.out_dir, file_name),
         h0_grid = h0_grid, ells = ells, ok = ok,
         **{f"cl_{name}": cls[name] for name in SPECTRA},
         theta_MC_100 = theta_MC_100,
         log_a = args.log_a, ns = args.ns, ombh2 = args.ombh2, omch2 = args.omch2,
         log_a_index = args.log_a_index, ns_index = args.ns_index,
         ombh2_index = args.ombh2_index, omch2_index = args.omch2_index,
         tau = args.tau, mnu = args.mnu, lmax = args.lmax, k_pivot = args.k_pivot,
         alens = args.alens, accuracy_boost = args.accuracy_boost,
         l_sample_boost = args.l_sample_boost, l_accuracy_boost = args.l_accuracy_boost)
print(f"saved {file_name} ({ok.sum()}/{args.h0_nodes} nodes ok, "
      f"{(time.time() - t_start) / 60:.1f} min)", flush = True)
