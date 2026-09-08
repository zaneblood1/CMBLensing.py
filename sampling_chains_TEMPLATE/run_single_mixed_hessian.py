"""One realization of the mixed-coordinate Hessian: simulate, mix, finite-difference, save.

Spawned once per realization by mixed_hessian.sh. Each job draws (f, phi) at the fiducial
cosmology, mixes them ONCE with D and G at that cosmology, then finite-differences
statistics.mixed_logpdf in theta at that fixed mixed pair, and writes -Hessian to a single
small npz. fisher_forecast.py --hessian_dir averages the collected files.

Fanning this out matters here: mixed_logpdf costs TWO lensing solves per evaluation (an
inverse one inside unmix, a forward one inside logpdf), and nothing cancels across the
stencil because D, G and the unmixed fields all move with theta. Sequentially that is
~3800 lensing solves for 100 realizations at 19 stencil points; one job per realization
turns it into 38 apiece.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import numpy as np

from cmb_lensing.fisher_forecast import mixed_hessian_realization
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER

parser = argparse.ArgumentParser()
parser.add_argument("--realization_index", type = int, required = True)
parser.add_argument("--map_seed", type = int, required = True)
parser.add_argument("--nside", type = int, required = True)
parser.add_argument("--theta_pix", type = float, required = True)
parser.add_argument("--noise_level", type = float, required = True)
parser.add_argument("--l_knee", type = float, default = 0.0)
parser.add_argument("--params", nargs = "+", required = True)
parser.add_argument("--out_dir", type = str, required = True)
args = parser.parse_args()

unknown = [name for name in args.params if name not in PARAM_ORDER]
if unknown:
    raise ValueError(f"unknown parameters {unknown}; choose from {PARAM_ORDER}")

os.makedirs(args.out_dir, exist_ok = True)

is_sampled = {name: (name in args.params) for name in PARAM_ORDER}

hessian, names, steps = mixed_hessian_realization(
    args.nside, args.theta_pix, args.noise_level, is_sampled, GROUND_TRUTH,
    args.map_seed, l_knee = args.l_knee)

#the index is in the filename and the seed is in the payload; load_hessian_directory
#rejects duplicate seeds, so a mis-set map_prefix cannot silently double count
out_path = os.path.join(args.out_dir, f"hessian_{args.realization_index:04d}.npz")
np.savez(out_path,
         hessian = hessian,
         names = np.array(names),
         method = "mixed",
         realization_index = args.realization_index,
         map_seed = args.map_seed,
         nside = args.nside,
         theta_pix = args.theta_pix,
         noise_level = args.noise_level,
         l_knee = args.l_knee,
         steps = np.array(steps))

print(f"realization {args.realization_index} (seed {args.map_seed}): diagonal "
      f"{np.array2string(np.diag(hessian), precision = 4)}")
print(f"wrote {out_path}")
