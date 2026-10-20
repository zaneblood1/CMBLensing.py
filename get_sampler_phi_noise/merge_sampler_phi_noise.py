"""Burn in and average the sampler phi chains into one per-mode N_phi.

Run through merge_sampler_phi_noise.sh on the HPC (the samples are ~200 MB per chain at nside
128 / 3000 sweeps), pointing --chain_dir at the out_dir get_sampler_phi_noise.sh wrote to. The
jobs may still be running: only the sweeps each one reports as complete are read.

    python merge_sampler_phi_noise.py --chain_dir ABSOLUTE_PATH_TO/sampler_phi_noise_output --burn_in 200

Writes sampler_phi_noise.npz into that directory: `nphi` (the per-mode rfft2 matrix in
covar_matrix_from_cls units, NaN on unmeasured modes, 0 at the origin), its jackknife error
`nphi_error`, `cphi`, the mean moments and the box / cosmology metadata.
"""

import argparse
import os
import numpy as np
from cmb_lensing.sampler_phi_noise import merge_phi_noise, MERGED_NAME

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain_dir", type = str,
                        default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "sampler_phi_noise_output"),
                        help = "the (scp'd) out_dir get_sampler_phi_noise.sh wrote its chains "
                               "to (default: sampler_phi_noise_output/ next to this script)")
    parser.add_argument("--burn_in", type = int, required = True)
    args = parser.parse_args()

    merged = merge_phi_noise(args.chain_dir, args.burn_in)
    out_path = os.path.join(args.chain_dir, MERGED_NAME)
    np.savez(out_path, **merged)

if __name__ == "__main__":
    main()
