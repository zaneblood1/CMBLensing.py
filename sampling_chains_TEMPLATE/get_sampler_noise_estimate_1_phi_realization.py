"""One phi realization of the sampler noise estimate: fix phi, draw data, accumulate the score.

Spawned once per seed by get_sampler_noise_estimate.sh. The job takes load_sim's phi at its
seed (at GROUND_TRUTH), draws --num_draws fresh (f, n) data sets at that phi, and for each one
evaluates the score grad_phi logpdf(f_WF(d, phi), phi) at the Wiener-filtered field. It stores
running per-mode SUMS of the score and of |score|^2 - not the variance - so every checkpoint
(every few draws, written atomically) is a valid partial measurement; the merge forms the
variance, the Fisher information F_phi, and N_bound = 1 / <F_phi>. With --calibration 1 the
same is done at phi = 0, where the bound must reproduce the unlensed-filter QE N0 exactly.

See cmb_lensing/sampler_noise_estimate.py for the algebra.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse

from cmb_lensing.sampler_noise_estimate import run_job, DEFAULT_TOL, DEFAULT_N_DRAWS

parser = argparse.ArgumentParser()
parser.add_argument("--realization_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--num_draws", type = int, default = DEFAULT_N_DRAWS)
parser.add_argument("--tol", type = float, default = DEFAULT_TOL)
parser.add_argument("--calibration", type = int, choices = (0, 1), default = 0,
                    help = "1: phi = 0 calibration job (written as "
                           "sampler_noise_calibration.npz)")
parser.add_argument("--out_dir", type = str, required = True)
args = parser.parse_args()

#the index is in the filename and the seed is in the payload; load_score_directory rejects
#duplicate seeds, so a mis-set seed_prefix cannot silently double count a phi
run_job(args.out_dir, args.realization_index, args.map_seed, args.nside, args.theta_pix,
        args.noise_level, n_draws = args.num_draws, l_knee = args.l_knee, tol = args.tol,
        calibration = bool(args.calibration))
