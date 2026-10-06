"""Analyse the sampler effective noise chains: burn in, average, cross correlate with truth.

Run on the HPC (the samples are ~200 MB per chain at nside 128 / 3000 sweeps), pointing
--chain_dir at the out_dir get_sampler_effective_noise.sh wrote to. The jobs may still be
running: only the sweeps each one reports as complete are read.

    python merge_sampler_effective_noise.py --chain_dir ABSOLUTE_PATH_TO/sampler_effective_noise_output \
        --burn_in 200 --burn_in_scan 100 400 800

For every chain: cut --burn_in sweeps, measure the integrated autocorrelation time (IAT) of
every phi mode, and average the remaining UN-THINNED samples into the posterior mean phi.
Then, per mode and with the moments summed over the data maps before the ratio is taken,

    r_L^2 = <phi_mean phi_true*>^2 / (<|phi_mean|^2> <|phi_true|^2>),   N_L = C_L (1 / r_L^2 - 1)

with a delete-one-map jackknife error. Writes into --chain_dir:
    sampler_effective_noise.npz           `nphi` is the full rfft-grid N_phi matrix
                                          (fisher_forecast --nphi_source score --phi_noise <it>)
    sampler_effective_noise_spectra.png   band spectra of the four estimates vs C_phi and the QE
    sampler_effective_noise_iat.png       IAT per mode (mean over chains) and per band per chain
    iat_per_chain/iat_chain_NNNN.png      IAT per phi mode, one image per chain

The report prints, per L band and as ratios to the QE's N0: the headline "cross" estimate
(|phi_mean|^2 from the cross power of disjoint chain blocks, which carries no Monte Carlo
noise bias) with its jackknife error; the "naive", "b_only" and "variance" estimates that
must agree with it; the same from the first and second half of every chain; the IAT and the
smallest effective sample count. --burn_in_scan recomputes the headline at other burn-ins.
See cmb_lensing/sampler_effective_noise.py.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import numpy as np

from cmb_lensing.sampler_effective_noise import (merge_effective_noise, write_outputs,
                                                 DEFAULT_BURN_IN, DEFAULT_DELTA_ELL,
                                                 DEFAULT_N_BLOCKS)


def main():
    parser = argparse.ArgumentParser(
        description = "Analyse the sampler effective noise chains into one N_phi matrix")
    parser.add_argument("--chain_dir", type = str, required = True,
                        help = "the out_dir get_sampler_effective_noise.sh wrote its chains to")
    parser.add_argument("--burn_in", type = int, default = DEFAULT_BURN_IN,
                        help = "sweeps cut from the start of every chain")
    parser.add_argument("--burn_in_scan", type = int, nargs = "*", default = [],
                        help = "further burn-ins at which the headline band spectrum is "
                               "recomputed, to check that it has stopped moving (each one "
                               "re-reads every chain)")
    parser.add_argument("--delta_ell", type = float, default = DEFAULT_DELTA_ELL,
                        help = "width of the |L| bands for the spectra and the band report")
    parser.add_argument("--n_blocks", type = int, default = DEFAULT_N_BLOCKS,
                        help = "disjoint post-burn-in blocks whose means are cross multiplied "
                               "for |phi_mean|^2 (even, >= 4); each must be much longer than "
                               "the IAT")
    parser.add_argument("--smooth", action = "store_true",
                        help = "make `nphi` the QE matrix times the band ratio N / N_QE at "
                               "EVERY mode, instead of the per-mode estimate where measured "
                               "(use it when single modes are too noisy)")
    parser.add_argument("--compare", type = str, nargs = "*", default = [],
                        help = "other npz files holding a 2D `nphi` at the SAME box (e.g. "
                               "sampler_noise_estimate.npz), drawn in the spectra plot")
    args = parser.parse_args()

    merged = merge_effective_noise(args.chain_dir, burn_in = args.burn_in,
                                   delta_ell = args.delta_ell, n_blocks = args.n_blocks,
                                   smooth = args.smooth, burn_in_scan = args.burn_in_scan)
    compare = []
    for path in args.compare:
        other = np.load(path, allow_pickle = True)
        if np.asarray(other["nphi"]).shape != merged["nphi"].shape:
            raise ValueError(f"{path} holds an nphi of shape {np.asarray(other['nphi']).shape}"
                             f", not this box's {merged['nphi'].shape}")
        compare.append((os.path.basename(os.path.dirname(os.path.abspath(path))) + "/"
                        + os.path.basename(path), np.asarray(other["nphi"])))
    out_path = write_outputs(merged, args.chain_dir, compare = compare)
    print(f"\nUse it with:\n"
          f"    python -m cmb_lensing.fisher_forecast --nphi_source score "
          f"--phi_noise {out_path} --nside {int(merged['nside'])} "
          f"--theta_pix {float(merged['theta_pix']):g} "
          f"--noise {float(merged['noise_level']):g} ...")


if __name__ == "__main__":
    main()
