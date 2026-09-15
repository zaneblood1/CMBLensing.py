"""One realization of the empirical delensed spectrum: simulate, reconstruct, delens, save.

Spawned once per seed by get_delensed_spectra.sh. Each job runs load_sim at its own seed and
the fiducial cosmology, reconstructs phi with map_joint, inverse-lenses the NOISELESS lensed
field by that estimate, band-averages the result, and divides by CAMB's delensed spectrum at
the same frozen Alens_L to get this realization's estimate of the transfer function R(l).
merge_delensed_spectra.py averages the collected files.

Every job also reports the three self-validation rungs (the measurement normalization, the
forward lensing against CAMB, and the inverse-lensing round trip) so a broken run announces
itself in the slurm log rather than quietly biasing R. See cmb_lensing/delensed_spectrum.py
for what each one checks and why the paired estimator is the one to trust.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import numpy as np

from cmb_lensing.delensed_spectrum import (measure_delensed_spectrum, DEFAULT_DELTA_ELL,
                                           LENSE_STEPS)
from cmb_lensing.fisher_forecast import NPHI_SOURCES, QE_RESPONSE_SOURCES, DEFAULT_QE_RESPONSE
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER

parser = argparse.ArgumentParser()
parser.add_argument("--realization_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--beam_fwhm", type = float, default = 0.0)
parser.add_argument("--delta_ell", type = float, default = DEFAULT_DELTA_ELL)
parser.add_argument("--map_joint_steps", type = int, default = 30)
parser.add_argument("--nphi_source", choices = NPHI_SOURCES, default = "covariance")
parser.add_argument("--qe_response", choices = QE_RESPONSE_SOURCES,
                    default = DEFAULT_QE_RESPONSE)
parser.add_argument("--out_dir", type = str, required = True)
#R is measured AT a cosmology. These shift the fiducial point so that
#compare_transfer_functions.py can test whether R is flat in theta; leave them alone for the
#production run. Each shifted cosmology needs its OWN out_dir - the merge refuses to average
#files taken at different parameter values
parser.add_argument("--shift_param", type = str, default = None, choices = PARAM_ORDER,
                    help = "displace one parameter off GROUND_TRUTH before measuring")
parser.add_argument("--shift_value", type = float, default = 0.0,
                    help = "the absolute displacement applied to --shift_param")
args = parser.parse_args()

param_ground = dict(GROUND_TRUTH)
if args.shift_param is not None:
    param_ground[args.shift_param] = param_ground[args.shift_param] + args.shift_value
    print(f"cosmology shifted: {args.shift_param} "
          f"{GROUND_TRUTH[args.shift_param]:.6g} -> {param_ground[args.shift_param]:.6g}")

os.makedirs(args.out_dir, exist_ok = True)

result = measure_delensed_spectrum(
    args.nside, args.theta_pix, args.noise_level, param_ground, args.map_seed,
    l_knee = args.l_knee, beam_fwhm = args.beam_fwhm, delta_ell = args.delta_ell,
    map_joint_steps = args.map_joint_steps, nphi_source = args.nphi_source,
    qe_response = args.qe_response)

#the index is in the filename and the seed is in the payload; load_transfer_directory rejects
#duplicate seeds, so a mis-set seed_prefix cannot silently double count a realization
out_path = os.path.join(args.out_dir,
                        f"delensed_spectra_{args.realization_index:04d}.npz")
np.savez(out_path,
         realization_index = args.realization_index,
         map_seed = args.map_seed,
         nside = args.nside,
         theta_pix = args.theta_pix,
         noise_level = args.noise_level,
         l_knee = args.l_knee,
         beam_fwhm = args.beam_fwhm,
         delta_ell = args.delta_ell,
         map_joint_steps = args.map_joint_steps,
         nphi_source = args.nphi_source,
         qe_response = args.qe_response,
         lense_steps = LENSE_STEPS,
         params = np.array([param_ground[name] for name in PARAM_ORDER]),
         param_names = np.array(PARAM_ORDER),
         **{key: value for key, value in result.items() if key != "edges"})

print(f"wrote {out_path}")
