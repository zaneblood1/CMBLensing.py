"""One data realization of the marginal-Fisher calculation: draw fields, save scores.

Spawned once per map by marginal_fisher.sh. Simulates a map at the ground-truth cosmology,
Gibbs-samples (f, phi) from p(f, phi | d, theta_0) with theta held completely fixed, and
writes the per-draw complete-data scores plus the mean normalized power to a single npz.
merge_marginal_fisher.py combines those into the final Fisher by both estimators.

Only the n-vector score per draw and one array per prior block are kept, so a map's output
is a few hundred KB no matter how many sweeps it ran - the fields are never written.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import numpy as np
import jax.numpy as jnp

from cmb_lensing.util import get_fourier_weights
from cmb_lensing.marginal_fisher import (log_cl_derivatives, marginal_fisher_one_map,
                                         PRIOR_BLOCKS)
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER

parser = argparse.ArgumentParser()
parser.add_argument("--map_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--n_draws", type = int, required = True)
parser.add_argument("--burn_in", type = int, required = True)
parser.add_argument("--params", nargs = "+", required = True)
parser.add_argument("--out_dir", type = str, required = True)
args = parser.parse_args()

unknown = [name for name in args.params if name not in PARAM_ORDER]
if unknown:
    raise ValueError(f"unknown parameters {unknown}; choose from {PARAM_ORDER}")

os.makedirs(args.out_dir, exist_ok = True)

weights = jnp.broadcast_to(
    jnp.real(get_fourier_weights((args.nside, args.nside // 2 + 1))),
    (args.nside, args.nside // 2 + 1))

#the Cl derivative stencil is identical for every map, so every job recomputes the same
#19 CAMB runs. that is ~1 minute against a multi-hour field chain, and it keeps the jobs
#independent - no shared cache file to race on
derivatives, steps, fiducial, good = log_cl_derivatives(
    args.params, GROUND_TRUTH, args.nside, args.theta_pix)

result = marginal_fisher_one_map(
    args.nside, args.theta_pix, args.noise_level, args.params, GROUND_TRUTH,
    derivatives, fiducial, good, weights, args.map_seed,
    args.n_draws, args.burn_in, l_knee = args.l_knee)

out_path = os.path.join(args.out_dir, f"map_{args.map_index}_scores.npz")
np.savez(out_path,
         scores = result["scores"],
         information = result["information"],
         hessian_term = result["hessian_term"],
         score_covariance = result["score_covariance"],
         phi_acceptance = result["phi_acceptance"],
         autocorrelation = result["autocorrelation"],
         names = np.array(args.params),
         map_index = args.map_index, map_seed = args.map_seed,
         nside = args.nside, theta_pix = args.theta_pix,
         noise_level = args.noise_level, l_knee = args.l_knee,
         n_draws = args.n_draws, burn_in = args.burn_in)

survived = np.diag(result["information"]) / np.diag(result["hessian_term"])
print(f"map {args.map_index}: phi acceptance {result['phi_acceptance']:.2f}, "
      f"tau {np.array2string(result['autocorrelation'], precision = 1)}, "
      f"{np.array2string(survived * 100, precision = 0)}% of E[-H] survives")
print(f"wrote {out_path}")
