"""One chain of the sampler effective noise: sample (f, phi) | d with theta at ground truth.

Spawned once per seed by get_sampler_effective_noise.sh. The job builds load_sim's (f, phi, d)
realization at its seed (at GROUND_TRUTH), then runs sample_lcdm.sample_joint's Gibbs sweep
without its theta step - f from its exact conditional, one HMC step on phi - starting from
phi = 0, and writes EVERY phi sample to a preallocated .npy as the chain produces it:

    sampler_phi_NNNN_samples.npy   (num_sweeps, nside, nside // 2 + 1) complex64
    sampler_phi_NNNN.npz           n_done, phi_true, C_phi, HMC acceptances, configuration

The samples are flushed and the npz rewritten (atomically) every few sweeps, so
merge_sampler_effective_noise.py can analyse the chain while it is still running and a job
killed at the wall clock keeps what it finished.

See cmb_lensing/sampler_effective_noise.py for the estimator.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse

from cmb_lensing.sampler_effective_noise import run_job, DEFAULT_N_SWEEPS, DEFAULT_FLUSH_EVERY

parser = argparse.ArgumentParser()
parser.add_argument("--realization_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--num_sweeps", type = int, default = DEFAULT_N_SWEEPS,
                    help = "Gibbs sweeps stored, INCLUDING the analysis's burn-in")
parser.add_argument("--flush_every", type = int, default = DEFAULT_FLUSH_EVERY,
                    help = "flush the samples and rewrite the sidecar every this many sweeps")
parser.add_argument("--out_dir", type = str, required = True)
args = parser.parse_args()

#the index is in the filename and the seed is in the payload; the analysis rejects duplicate
#seeds, so a mis-set seed_prefix cannot silently double count a realization
run_job(args.out_dir, args.realization_index, args.map_seed, args.nside, args.theta_pix,
        args.noise_level, n_sweeps = args.num_sweeps, l_knee = args.l_knee,
        flush_every = args.flush_every)
