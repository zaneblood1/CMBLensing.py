"""Average the per-realization moments into one per-mode map_joint N_phi.

Run after get_map_joint_phi_noise.sh has finished, pointing --noise_dir at the out_dir that
script wrote to:

    python merge_map_joint_phi_noise.py --noise_dir ABSOLUTE_PATH_TO/map_joint_phi_noise_output

Writes map_joint_phi_noise.npz into that directory: `nphi` (the per-mode rfft2 matrix in
covar_matrix_from_cls units, NaN on unmeasured modes), its jackknife error `nphi_error`, `cphi`,
the mean moments and the box / cosmology metadata. Use it as
fisher_forecast --nphi_source map_joint --phi_noise <npz>, or as the delensed-covariance jobs'
phi_noise. See cmb_lensing/map_joint_phi_noise.py.
"""

import argparse
import os
import numpy as np
from cmb_lensing.map_joint_phi_noise import merge_phi_noise, MERGED_NAME

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_dir", type = str,
                        default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "map_joint_phi_noise_output"),
                        help = "the (scp'd) out_dir get_map_joint_phi_noise.sh wrote its jobs "
                               "to (default: map_joint_phi_noise_output/ next to this script)")
    parser.add_argument("--out_name", type = str, default = MERGED_NAME)
    args = parser.parse_args()

    merged = merge_phi_noise(args.noise_dir)
    out_path = os.path.join(args.noise_dir, args.out_name)
    np.savez(out_path, **merged)

if __name__ == "__main__":
    main()
