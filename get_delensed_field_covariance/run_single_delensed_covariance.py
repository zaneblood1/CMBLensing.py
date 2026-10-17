"""One realization of the empirical delensed covariance, at every point of the stencil.

Spawned once per seed by get_delensed_covariance.sh. For its seed (one set of common random
numbers) the job runs, at theta_0 and at theta_0 +/- h_i for every --params entry,

    load_sim(theta) -> inverse-lense the NOISELESS lensed field by the Wiener-filtered TRUE
                       phi, W phi with W = C_phi / (C_phi + N_phi) at theta_0
                    -> |F|^2 / nside^2 on the rfft grid

and saves the 1 + 2k per-mode delensed covariances. merge_delensed_covariance.py averages
them locally. See cmb_lensing/delensed_covariance.py for what is measured and why.

The file is written once, when every stencil point is done; a job that dies leaves no file.
"""

import argparse
import os
import numpy as np
from cmb_lensing.delensed_covariance import (measure_delensed_covariance,
                                             realization_file_name, DEFAULT_STEP_SIGMA,
                                             DEFAULT_QE_FILTER, LENSE_STEPS)
from cmb_lensing.fisher_forecast import QE_FILTERS, QE_RESPONSE_SOURCES, DEFAULT_QE_RESPONSE
from cmb_lensing.constants import GROUND_TRUTH, PARAM_ORDER

parser = argparse.ArgumentParser()
parser.add_argument("--realization_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--step_sigma", type = float, default = DEFAULT_STEP_SIGMA)
parser.add_argument("--qe_filter", choices = QE_FILTERS, default = DEFAULT_QE_FILTER)
parser.add_argument("--qe_response", choices = QE_RESPONSE_SOURCES, default = DEFAULT_QE_RESPONSE)
parser.add_argument("--phi_noise", type = str, default = "")
parser.add_argument("--out_dir", type = str, required = True)
parser.add_argument("--params", nargs = "+", required = True, choices = PARAM_ORDER)
args = parser.parse_args()

result = measure_delensed_covariance(
    args.nside, args.theta_pix, args.noise_level, dict(GROUND_TRUTH), args.map_seed,
    args.params, step_sigma = args.step_sigma, l_knee = args.l_knee,
    qe_filter = args.qe_filter, qe_response = args.qe_response,
    phi_noise = args.phi_noise or None)

os.makedirs(args.out_dir, exist_ok = True)
out_path = os.path.join(args.out_dir, realization_file_name(args.realization_index))
np.savez(out_path,
         realization_index = args.realization_index,
         map_seed = args.map_seed,
         nside = args.nside,
         theta_pix = args.theta_pix,
         noise_level = args.noise_level,
         l_knee = args.l_knee,
         lense_steps = LENSE_STEPS,
         step_sigma = args.step_sigma,
         names = np.array(args.params),
         params = np.array([GROUND_TRUTH[name] for name in PARAM_ORDER]),
         param_names = np.array(PARAM_ORDER),
         offsets = result["offsets"],
         steps = result["steps"],
         point_params = result["point_params"],
         qe_filter = result["qe_filter"],
         qe_response = result["qe_response"],
         wiener_nphi_source = result["wiener_nphi_source"],
         wiener_nphi_file = result["wiener_nphi_file"],
         delensed = result["delensed"])
