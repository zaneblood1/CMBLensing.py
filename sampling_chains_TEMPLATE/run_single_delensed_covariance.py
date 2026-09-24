"""One realization of the empirical delensed covariance, at every point of the stencil.

Spawned once per seed by get_delensed_covariance.sh. For its seed (one set of common random
numbers) the job runs, at theta_0 and at theta_0 +/- h_i for every --params entry,

    load_sim(theta) -> map_joint -> inverse-lense the NOISELESS lensed field by phi_hat
                    -> F conj(F) / nside^2 on the rfft grid

and saves the 1 + 2k per-mode covariances (plus the unlensed and lensed ones, which validate
the common random numbers), and the per-mode moments |phi_hat|^2, Re(phi_hat phi*), |phi|^2
the merge turns into the empirical phi noise. merge_delensed_covariance.py averages them locally. See
cmb_lensing/delensed_covariance.py for what is measured and why.

The file is rewritten ATOMICALLY after every stencil point (a hidden temp file, then
os.replace), with `finished = False` until the last one, so a job killed at the wall clock
keeps its finished points and the merge never reads a half-written file. The merge sets
unfinished files aside.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import numpy as np

from cmb_lensing.delensed_covariance import (measure_delensed_covariance,
                                             realization_file_name, DEFAULT_STEP_SIGMA,
                                             RECONSTRUCTIONS, DEFAULT_RECONSTRUCTION,
                                             DEFAULT_CONSTANT_NPHI, FIELD_KINDS,
                                             PHI_MOMENTS, effective_constant_nphi)
from cmb_lensing.delensed_spectrum import LENSE_STEPS
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER

parser = argparse.ArgumentParser()
parser.add_argument("--realization_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--map_joint_steps", type = int, default = 30)
parser.add_argument("--step_sigma", type = float, default = DEFAULT_STEP_SIGMA,
                    help = "the stencil's h_i in units of PARAM_SIGMA")
parser.add_argument("--reconstruction", choices = RECONSTRUCTIONS,
                    default = DEFAULT_RECONSTRUCTION,
                    help = "'fiducial' builds map_joint's operators at theta_0 at every "
                           "stencil point (a frozen estimator); 'shifted' at each point's "
                           "own cosmology")
parser.add_argument("--constant_nphi", type = int, choices = (0, 1),
                    default = int(DEFAULT_CONSTANT_NPHI),
                    help = "--reconstruction shifted only: 1 keeps map_joint's QE norm N_phi "
                           "at theta_0 at every stencil point, 0 rebuilds it per point")
parser.add_argument("--out_dir", type = str, required = True)
parser.add_argument("--params", nargs = "+", required = True, choices = PARAM_ORDER,
                    help = "the parameters to difference")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok = True)
file_name = realization_file_name(args.realization_index)
out_path = os.path.join(args.out_dir, file_name)


def write(result, finished):
    """Write the result file atomically: a hidden temp file, then os.replace over the target."""
    temp_path = os.path.join(args.out_dir, f".{file_name}.tmp")
    with open(temp_path, "wb") as handle:
        np.savez(handle,
                 realization_index = args.realization_index,
                 map_seed = args.map_seed,
                 nside = args.nside,
                 theta_pix = args.theta_pix,
                 noise_level = args.noise_level,
                 l_knee = args.l_knee,
                 map_joint_steps = args.map_joint_steps,
                 lense_steps = LENSE_STEPS,
                 step_sigma = args.step_sigma,
                 reconstruction = args.reconstruction,
                 #what was actually done: "fiducial" freezes N_phi whatever the flag says
                 constant_nphi = effective_constant_nphi(args.reconstruction,
                                                         args.constant_nphi),
                 names = np.array(args.params),
                 params = np.array([GROUND_TRUTH[name] for name in PARAM_ORDER]),
                 param_names = np.array(PARAM_ORDER),
                 offsets = result["offsets"],
                 steps = result["steps"],
                 point_params = result["point_params"],
                 inverse_error = result["inverse_error"],
                 n_done = result["n_done"],
                 finished = finished,
                 **{kind: result[kind] for kind in FIELD_KINDS + PHI_MOMENTS})
    os.replace(temp_path, out_path)


result = measure_delensed_covariance(
    args.nside, args.theta_pix, args.noise_level, dict(GROUND_TRUTH), args.map_seed,
    args.params, step_sigma = args.step_sigma, l_knee = args.l_knee,
    map_joint_steps = args.map_joint_steps, reconstruction = args.reconstruction,
    constant_nphi = bool(args.constant_nphi),
    on_point = lambda partial: write(partial, finished = False))
write(result, finished = True)

print(f"wrote {out_path}")
