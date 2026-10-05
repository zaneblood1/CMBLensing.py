"""Average the per-phi Fisher information into one N_phi matrix and compare it.

Run after get_sampler_noise_estimate.sh has finished, pointing --noise_dir at (a local copy
of) the out_dir that script wrote to:

    python merge_sampler_noise_estimate.py --noise_dir ABSOLUTE_PATH_TO/sampler_noise_estimate_output \
        --map_joint_noise <merge_delensed_covariance.py or merge_phi_noise.py npz, same box>

Writes into that directory:
    sampler_noise_estimate.npz          `nphi` is the full rfft-grid N_phi matrix
                                        (fisher_forecast --nphi_source score --phi_noise <it>)
    sampler_noise_estimate_matrix.png   the three N_phi on the rfft grid and their ratios
    sampler_noise_estimate_spectra.png  the same as band spectra, with C_phi and the
                                        calibration ratio

The three N_phi compared: the sampler bound 1/<F_phi> from these jobs; map_joint's empirical
N_eff (optional, from --map_joint_noise); and the 1st-principles quadratic estimator, i.e.
load_sim's own quadratic_estimate without the NPHI_FAC preconditioning factor. If the jobs
stored map_joint's phi_MAP fields (map_joint_steps > 0 in get_sampler_noise_estimate.sh), the
band report and the spectra plot also carry the average over phi realizations of the per-mode
variance of phi_MAP over each job's (f, n) draws, raw and divided by the measured response^2
(`phi_map_variance` / `phi_map_response` / `phi_map_noise` in the npz). Refuses job
files that disagree on the box, the noise, the CG tolerance or the cosmology, and duplicate
seeds; see cmb_lensing/sampler_noise_estimate.py.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/ except for
the default --noise_dir.
"""

import argparse
import os

import numpy as np

from cmb_lensing.sampler_noise_estimate import (merge_sampler_noise, load_map_joint_noise,
                                                plot_matrices, plot_spectra, MERGED_NAME,
                                                DEFAULT_DELTA_ELL)


def main():
    parser = argparse.ArgumentParser(
        description = "Merge the sampler noise estimate jobs into one N_phi matrix")
    parser.add_argument("--noise_dir", type = str,
                        required = True,
                        help = "the out_dir get_sampler_noise_estimate.sh wrote its jobs to")
    parser.add_argument("--map_joint_noise", type = str, default = None,
                        help = "optional: map_joint's empirical N_eff at the SAME box, from a "
                               "merge_delensed_covariance.py npz (per mode, phi_noise_fid) or "
                               "a merge_phi_noise.py npz (bands); left out of the plots if "
                               "not given")
    parser.add_argument("--delta_ell", type = float, default = DEFAULT_DELTA_ELL,
                        help = "width of the |L| bands for the spectra and the band report")
    args = parser.parse_args()

    merged = merge_sampler_noise(args.noise_dir, delta_ell = args.delta_ell)
    out_path = os.path.join(args.noise_dir, MERGED_NAME)
    np.savez(out_path, **merged)

    map_joint = None
    if args.map_joint_noise:
        map_joint = load_map_joint_noise(args.map_joint_noise, int(merged["nside"]),
                                         float(merged["theta_pix"]),
                                         float(merged["noise_level"]))
    stem = os.path.splitext(out_path)[0]
    plot_matrices(merged, map_joint, stem + "_matrix.png")
    plot_spectra(merged, map_joint, stem + "_spectra.png")

    print(f"\nwrote {out_path}")
    print(f"wrote {stem}_matrix.png")
    print(f"wrote {stem}_spectra.png")
    print(f"\nUse it with:\n"
          f"    python -m cmb_lensing.fisher_forecast --nphi_source score "
          f"--phi_noise {out_path} --nside {int(merged['nside'])} "
          f"--theta_pix {float(merged['theta_pix']):g} "
          f"--noise {float(merged['noise_level']):g} ...")


if __name__ == "__main__":
    main()
