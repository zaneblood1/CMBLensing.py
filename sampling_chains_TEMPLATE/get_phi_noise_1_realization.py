"""One realization of the effective phi-noise measurement: simulate, reconstruct, correlate.

Spawned once per seed by get_effective_phi_noise.sh. Each job runs load_sim at its own seed
and the fiducial cosmology, reconstructs phi from the data with map_joint, cross correlates
that estimate against the true phi in |L| annuli, and writes the three band sums. Ratios are
deliberately NOT formed here - merge_phi_noise.py forms them from the averaged sums, because
a mean of per-realization ratios carries a bias that adding realizations does not remove.

Every job also reports the two self-validation rungs (the measurement normalization, and the
correlation coefficient against the analytic quadratic estimator's Wiener weight) so a broken
run announces itself in the slurm log rather than quietly biasing N_eff. See
cmb_lensing/phi_noise.py for the algebra and for what each rung checks.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import numpy as np

from cmb_lensing.phi_noise import measure_phi_noise, DEFAULT_DELTA_ELL
from cmb_lensing.fisher_forecast import QE_RESPONSE_SOURCES, DEFAULT_QE_RESPONSE
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
parser.add_argument("--qe_response", choices = QE_RESPONSE_SOURCES,
                    default = DEFAULT_QE_RESPONSE,
                    help = "which TT spectrum the ANALYTIC reference N^(0) is built from. It "
                           "only sets the rung-1 comparison and the N_eff / N^(0) ratio the "
                           "merge reports - the measurement itself does not use it")
parser.add_argument("--out_dir", type = str, required = True)
#N_eff is measured AT a cosmology and is then held fixed across the forecast's whole
#finite-difference stencil. These shift the fiducial point so that two merged runs can be
#compared to test how much N_eff actually moves with theta; leave them alone for the
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

result = measure_phi_noise(
    args.nside, args.theta_pix, args.noise_level, param_ground, args.map_seed,
    l_knee = args.l_knee, beam_fwhm = args.beam_fwhm, delta_ell = args.delta_ell,
    map_joint_steps = args.map_joint_steps, qe_response = args.qe_response)

#the index is in the filename and the seed is in the payload; load_phi_noise_directory
#rejects duplicate seeds, so a mis-set seed_prefix cannot silently double count a realization
out_path = os.path.join(args.out_dir, f"phi_noise_{args.realization_index:04d}.npz")
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
         qe_response = args.qe_response,
         params = np.array([param_ground[name] for name in PARAM_ORDER]),
         param_names = np.array(PARAM_ORDER),
         **result)

print(f"wrote {out_path}")
