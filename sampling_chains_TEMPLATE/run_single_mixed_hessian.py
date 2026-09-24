"""One realization of the mixed-coordinate Hessian: simulate, mix, finite-difference, save.

Spawned once per realization by mixed_hessian.sh. Each job draws (f, phi) at the fiducial
cosmology, mixes them ONCE with D and G at that cosmology, then finite-differences
statistics.mixed_logpdf in theta at that fixed mixed pair, and writes -Hessian to a single
small npz. fisher_forecast_from_mixed_logpdf.py --hessian_dir averages the collected files.

Fanning this out matters here: mixed_logpdf costs TWO lensing solves per evaluation (an
inverse one inside unmix, a forward one inside logpdf), and nothing cancels across the
stencil because D, G and the unmixed fields all move with theta. Sequentially that is
~3800 lensing solves for 100 realizations at 19 stencil points; one job per realization
turns it into 38 apiece.

With --louis the job instead runs ONE sub-chain of Louis's observed information for its
realization (louis_realization): (f, phi) are sampled from their posterior given the data
at theta_0 with the MCMC randomness of --sub_chain_index, and the stencil is run at every
sweep. mixed_hessian.sh spawns several such jobs per map_seed. The file holds the RAW
chain - every sweep from the first, no burn-in, no thinning - as per-sweep `hessians`,
`scores` and `phi_accepts`; the merge (--hessian_dir, louis_information_from_chains) groups
the sub-chains by map_seed, cuts the burn-in, measures R-hat, thins and pools. The files
carry method "mixed_louis" and are named hessian_<index>_chain_<sub>.npz.

A louis job REWRITES its file after every sweep (atomically, `finished = False` until the
last write), so a job killed at the wall-clock limit loses at most the sweep in progress,
and a running chain can be scp'd off and diagnosed - chain_analysis.louis_chain_diagnostics
and the merge both accept unfinished sub-chains - to kill a pathological run early.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import numpy as np

from cmb_lensing.fisher_forecast_from_mixed_logpdf import (mixed_hessian_realization,
                                                           louis_realization,
                                                           louis_information_from_chains,
                                                           METHOD, LOUIS_METHOD,
                                                           DEFAULT_LOUIS_DRAWS)
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER

parser = argparse.ArgumentParser()
parser.add_argument("--realization_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--params", nargs = "+", required = True)
parser.add_argument("--out_dir", type = str, required = True)
parser.add_argument("--louis", type = int, default = 0,
                    help = "1 for Louis's observed information, 0 for the plain Hessian")
#raw sweeps, ALL of them saved: burn-in and thinning are post-processing choices, never
#made here
parser.add_argument("--louis_draws", type = int, default = DEFAULT_LOUIS_DRAWS)
#which of the realization's independent sub-chains this job is (same data map, different
#MCMC randomness); ignored without --louis
parser.add_argument("--sub_chain_index", type = int, default = 0)
args = parser.parse_args()

unknown = [name for name in args.params if name not in PARAM_ORDER]
if unknown:
    raise ValueError(f"unknown parameters {unknown}; choose from {PARAM_ORDER}")

os.makedirs(args.out_dir, exist_ok = True)

is_sampled = {name: (name in args.params) for name in PARAM_ORDER}

def write_npz(file_name, hessian, names, steps, extra):
    """Write one result file ATOMICALLY: a hidden temp file, then os.replace over the target.

    Louis jobs call this after every sweep, so the file on disk is always a complete,
    loadable snapshot - never half written, whether it is scp'd off mid-write or the job is
    killed mid-write. The temp name starts with "." so the merge's hessian_*.npz glob never
    sees it, and os.replace is atomic on the same filesystem.
    """
    out_path = os.path.join(args.out_dir, file_name)
    temp_path = os.path.join(args.out_dir, f".{file_name}.tmp")
    #an open file handle, because np.savez given a path appends ".npz" to the temp name
    with open(temp_path, "wb") as handle:
        np.savez(handle,
                 hessian = hessian,
                 names = np.array(names),
                 method = LOUIS_METHOD if args.louis else METHOD,
                 realization_index = args.realization_index,
                 map_seed = args.map_seed,
                 nside = args.nside,
                 theta_pix = args.theta_pix,
                 noise_level = args.noise_level,
                 l_knee = args.l_knee,
                 steps = np.array(steps),
                 **extra)
    os.replace(temp_path, out_path)
    return out_path


def louis_payload(result):
    """(hessian, extra) for a louis checkpoint or final result."""
    #a PROVISIONAL single-sub-chain value with no burn-in, only so every file carries a
    #`hessian` (NaN until there are two sweeps); the merge never reads it for louis files and
    #rebuilds the information from the raw per-sweep arrays of all the realization's sub-chains
    if len(result["scores"]) >= 2:
        hessian, _ = louis_information_from_chains([(result["hessians"], result["scores"])],
                                                   burn_in = 0)
    else:
        hessian = np.full(result["hessian_truth"].shape, np.nan)
    extra = {key: result[key] for key in ("hessian_truth", "hessians", "scores",
                                          "phi_accepts", "phi_acceptance", "n_sweeps",
                                          "sub_chain_index", "finished")}
    #n_burn = 0 tells the merge nothing was cut here, unlike files from before this rework
    extra.update(louis_draws = args.louis_draws, n_burn = 0)
    return hessian, extra


#the index is in the filename and the seed is in the payload; load_hessian_directory
#rejects duplicate seeds (duplicate (seed, sub-chain) pairs for louis files), so a mis-set
#map_prefix cannot silently double count
if args.louis:
    file_name = f"hessian_{args.realization_index:04d}_chain_{args.sub_chain_index:02d}.npz"

    def checkpoint(snapshot):
        #rewrite the whole file after every sweep (`finished = False`): a job killed at its
        #wall-clock limit keeps every sweep it finished, and a running chain can be copied
        #off and run through chain_analysis.louis_chain_diagnostics before it ends. The
        #payload is ~100 bytes per sweep, negligible next to a 19-point stencil
        hessian, extra = louis_payload(snapshot)
        write_npz(file_name, hessian, snapshot["names"], snapshot["steps"], extra)

    result = louis_realization(args.nside, args.theta_pix, args.noise_level, is_sampled,
                               GROUND_TRUTH, args.map_seed, n_draws = args.louis_draws,
                               sub_chain_index = args.sub_chain_index, l_knee = args.l_knee,
                               on_sweep = checkpoint)
    hessian, extra = louis_payload(result)
    names, steps = result["names"], result["steps"]
else:
    #a plain Hessian is one stencil (~2 minutes), with nothing to checkpoint part way
    file_name = f"hessian_{args.realization_index:04d}.npz"
    hessian, names, steps = mixed_hessian_realization(
        args.nside, args.theta_pix, args.noise_level, is_sampled, GROUND_TRUTH,
        args.map_seed, l_knee = args.l_knee)
    extra = {}

out_path = write_npz(file_name, hessian, names, steps, extra)

print(f"realization {args.realization_index} (seed {args.map_seed}"
      + (f", sub-chain {args.sub_chain_index}" if args.louis else "")
      + f"): diagonal {np.array2string(np.diag(hessian), precision = 4)}")
print(f"wrote {out_path}")
