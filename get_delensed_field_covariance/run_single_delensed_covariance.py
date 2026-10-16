"""One realization of the empirical delensed covariance, at every point of the stencil.

Spawned once per seed by get_delensed_covariance.sh. For its seed (one set of common random
numbers) the job runs, at theta_0 and at theta_0 +/- h_i for every --params entry,

    load_sim(theta) -> inverse-lense the NOISELESS lensed field by the Wiener-filtered TRUE
                       phi, W phi with W = C_phi / (C_phi + N_phi) at theta_0
                    -> F conj(F) / nside^2 on the rfft grid
    (map_joint still runs at every point, for the phi moments - unless --phi_noise
     supplies W's N_phi from an earlier run's npz, in which case it is skipped and that
     file's merged phi grids, if it is a delensed_covariance.npz, are copied to
     copied_phi_moments.npz in --out_dir instead)

and saves the 1 + 2k per-mode covariances (plus the unlensed and lensed ones, which validate
the common random numbers), and the per-mode moments |phi_hat|^2, Re(phi_hat phi*), |phi|^2
the merge turns into the empirical phi noise. merge_delensed_covariance.py averages them locally. See
cmb_lensing/delensed_covariance.py for what is measured and why.

The file is rewritten ATOMICALLY after every stencil point (a hidden temp file, then
os.replace), with `finished = False` until the last one, so a job killed at the wall clock
keeps its finished points and the merge never reads a half-written file. The merge sets
unfinished files aside.
"""

import argparse
import os

import numpy as np

from cmb_lensing.delensed_covariance import (measure_delensed_covariance,
                                             copied_phi_grids, write_copied_phi,
                                             realization_file_name, DEFAULT_STEP_SIGMA,
                                             RECONSTRUCTIONS, DEFAULT_RECONSTRUCTION,
                                             DEFAULT_CONSTANT_NPHI, FIELD_KINDS,
                                             PHI_MOMENTS, effective_constant_nphi,
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
parser.add_argument("--qe_filter", choices = QE_FILTERS, default = DEFAULT_QE_FILTER,
                    help = "the TT spectrum in the filter of the QE behind the delensing "
                           "Wiener weight's N_phi: 'lensed' (one pass), 'delensed' (iterated; "
                           "fisher_forecast --nphi_source qe_delensed) or 'unlensed' "
                           "(fisher_forecast --nphi_source qe_unlensed)")
parser.add_argument("--qe_response", choices = QE_RESPONSE_SOURCES,
                    default = DEFAULT_QE_RESPONSE,
                    help = "the TT spectrum weighting the QE response in the delensing Wiener "
                           "weight's N_phi (fisher_forecast --qe_response)")
parser.add_argument("--phi_noise", type = str, default = "",
                    help = "path to an npz holding a per-mode phi noise estimate at this box "
                           "and cosmology (merge_sampler_noise_estimate.py, "
                           "merge_sampler_effective_noise.py or merge_delensed_covariance.py "
                           "with the phi moments): W's N_phi becomes that matrix instead of "
                           "the box QE matrix and map_joint is NOT run (a "
                           "delensed_covariance.npz's merged phi grids are copied to "
                           "--out_dir in place of the phi moments); --qe_filter and "
                           "--qe_response then do not apply. Empty (default): the box QE "
                           "matrix")
parser.add_argument("--phi_noise_key", type = str, default = "",
                    help = "which 2D grid of --phi_noise to use. Empty (default): its `nphi`, "
                           "else its `phi_noise_fid`. Name e.g. nphi_map_joint for the "
                           "map_joint noise of a sampler_noise_estimate.npz, or nphi_smooth "
                           "of a sampler_effective_noise.npz")
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
                 delensing_phi = result["delensing_phi"],
                 qe_filter = result["qe_filter"],
                 qe_response = result["qe_response"],
                 wiener_nphi_source = result["wiener_nphi_source"],
                 wiener_nphi_checksum = result["wiener_nphi_checksum"],
                 wiener_nphi_key = result["wiener_nphi_key"],
                 n_done = result["n_done"],
                 finished = finished,
                 #the phi moments are absent when map_joint was skipped (--phi_noise)
                 **{kind: result[kind] for kind in FIELD_KINDS + PHI_MOMENTS
                    if kind in result})
    os.replace(temp_path, out_path)


result = measure_delensed_covariance(
    args.nside, args.theta_pix, args.noise_level, dict(GROUND_TRUTH), args.map_seed,
    args.params, step_sigma = args.step_sigma, l_knee = args.l_knee,
    map_joint_steps = args.map_joint_steps, reconstruction = args.reconstruction,
    constant_nphi = bool(args.constant_nphi),
    qe_filter = args.qe_filter, qe_response = args.qe_response,
    phi_noise = args.phi_noise or None, phi_noise_key = args.phi_noise_key or None,
    on_point = lambda partial: write(partial, finished = False))
write(result, finished = True)

#map_joint was skipped, so there are no phi moments of this run's own: carry the noise file's
#merged phi grids over (one sidecar for the whole directory; nothing if it has none)
if args.phi_noise:
    copied = copied_phi_grids(args.phi_noise, args.params, args.step_sigma)
    if copied is not None:
        write_copied_phi(args.out_dir, copied)
        print(f"copied the phi grids of {args.phi_noise}"
              + ("" if copied["has_stencil"] else " (centre only: different stencil)"))

print(f"wrote {out_path}")
