"""Average the per-realization delensed covariances into the 1 + 2k stencil matrices.

Run locally after get_delensed_field_covariance.sh has finished on the HPC and its out_dir has been
scp'd over:

    python merge_delensed_field_covariance.py --covariance_dir <local copy of out_dir>

Writes delensed_covariance.npz into that directory: the realization mean of the delensed
per-mode covariance at theta_0 and theta_0 +/- h_i. The npz is what fisher_forecast.py's
--delensed_covariance reads. Refuses files that disagree on the box, the stencil or the
delensing Wiener weight, and refuses duplicate seeds.
"""

import argparse
import os
import numpy as np
from cmb_lensing.delensed_covariance import merge_delensed_covariance, MERGED_NAME


def main():
    parser = argparse.ArgumentParser(
        description = "Average per-realization delensed covariances into the stencil")
    parser.add_argument("--covariance_dir", type = str,
                        help = "the (scp'd) out_dir get_delensed_field_covariance.sh wrote to")
    parser.add_argument("--out_name", type = str, default = MERGED_NAME)
    args = parser.parse_args()

    merged = merge_delensed_covariance(args.covariance_dir)
    out_path = os.path.join(args.covariance_dir, args.out_name)
    np.savez(out_path, **merged)

if __name__ == "__main__":
    main()
