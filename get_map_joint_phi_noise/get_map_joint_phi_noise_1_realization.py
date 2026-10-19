"""One realization of the map_joint phi-noise measurement: simulate, reconstruct, store moments.

Spawned once per seed by get_map_joint_phi_noise.sh. Each job runs load_sim at its own seed and
the fiducial cosmology (GROUND_TRUTH), reconstructs phi from the data with map_joint, and writes
the per-mode auto / cross / true-auto moments on the rfft2 grid. Ratios are NOT formed here -
merge_map_joint_phi_noise.py forms N_phi from the realization-averaged moments. See
cmb_lensing/map_joint_phi_noise.py.
"""

import argparse
import os
import numpy as np
from cmb_lensing.map_joint_phi_noise import measure_phi_noise
from cmb_lensing.constants import GROUND_TRUTH, PARAM_ORDER

parser = argparse.ArgumentParser()
parser.add_argument("--realization_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--map_joint_steps", type = int, default = 30)
parser.add_argument("--out_dir", type = str, required = True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok = True)
result = measure_phi_noise(args.nside, args.theta_pix, args.noise_level, args.map_seed,
                           l_knee = args.l_knee, map_joint_steps = args.map_joint_steps)

out_path = os.path.join(args.out_dir, f"map_joint_phi_noise_{args.realization_index:04d}.npz")
np.savez(out_path,
         realization_index = args.realization_index,
         map_seed = args.map_seed,
         nside = args.nside,
         theta_pix = args.theta_pix,
         noise_level = args.noise_level,
         l_knee = args.l_knee,
         map_joint_steps = args.map_joint_steps,
         params = np.array([GROUND_TRUTH[name] for name in PARAM_ORDER]),
         param_names = np.array(PARAM_ORDER),
         **result)
