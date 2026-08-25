from cmb_lensing.sample_lcdm import *
from cmb_lensing.simulate import *
import argparse

#Parse the arguments of the slurm job
parser = argparse.ArgumentParser()
parser.add_argument("--map_seed", type = int)
parser.add_argument("--chain", type = int)
parser.add_argument("--nside", type = int)
parser.add_argument("--theta_pix", type = float)
parser.add_argument("--noise_level", type = float)
parser.add_argument("--pol", type = str)
parser.add_argument("--num_burn_in_fix_theta", type = int)
parser.add_argument("--iters_per_chain", type = int)
parser.add_argument("--num_burn_in_always_accept", type = int)
parser.add_argument("--ombh2_min", type = float)
parser.add_argument("--omch2_min", type = float)
parser.add_argument("--theta_MC_100_min", type = float)
parser.add_argument("--log_a_min", type = float)
parser.add_argument("--ns_min", type = float)
parser.add_argument("--ombh2_max", type = float)
parser.add_argument("--omch2_max", type = float)
parser.add_argument("--theta_MC_100_max", type = float)
parser.add_argument("--log_a_max", type = float)
parser.add_argument("--ns_max", type = float)
parser.add_argument("--ombh2_init", type = float)
parser.add_argument("--omch2_init", type = float)
parser.add_argument("--theta_MC_100_init", type = float)
parser.add_argument("--log_a_init", type = float)
parser.add_argument("--ns_init", type = float)
parser.add_argument("--ombh2_ground", type = float)
parser.add_argument("--omch2_ground", type = float)
parser.add_argument("--theta_MC_100_ground", type = float)
parser.add_argument("--log_a_ground", type = float)
parser.add_argument("--ns_ground", type = float)
args = parser.parse_args()

#Generate a single realization of the ground truth data set using the chain index as the seed
ground_truth_params = {}
ground_truth_params["ombh2"] = args.ombh2_ground
ground_truth_params["omch2"] = args.omch2_ground
ground_truth_params["cosmomc_theta"] = args.theta_MC_100_ground / 100
ground_truth_params["As"] = jnp.exp(args.log_a_ground) * 1e-10
ground_truth_params["ns"] = args.ns_ground

#Use the same seed depending on which map we are currently on in the loop
data_set = load_sim(args.nside, args.theta_pix, args.pol, args.map_seed, **ground_truth_params,
                    uk_arcmin_t = args.noise_level, nt = 0, r = 0, l_knee = 0)

#initial starting guesses for parameters
param_init = {}
param_init["ombh2"] = args.ombh2_init
param_init["omch2"] = args.omch2_init
param_init["theta_MC_100"] = args.theta_MC_100_init
param_init["logA"] = args.log_a_init
param_init["ns"] = args.ns_init

#allowed hi / lo search range for parameters
param_ranges = {}
param_ranges["ombh2"] = (args.ombh2_min, args.ombh2_max)
param_ranges["omch2"] = (args.omch2_min, args.omch2_max)
param_ranges["theta_MC_100"] = (args.theta_MC_100_min, args.theta_MC_100_max)
param_ranges["logA"] = (args.log_a_min, args.log_a_max)
param_ranges["ns"] = (args.ns_min, args.ns_max)

#NOTE flip these depending on which parameter is being sampled
#Whether or not to sample each parameter
should_sample = {}
should_sample["ombh2"] = False
should_sample["omch2"] = True
should_sample["theta_MC_100"] = True
should_sample["logA"] = True
should_sample["ns"] = False

#Tune these with a single pilot chain to achieve around 44% acceptance per parameter
proposal_sigmas = {}
proposal_sigmas["ombh2"] = 1e-4
proposal_sigmas["omch2"] = 1e-3
proposal_sigmas["theta_MC_100"] = 1e-3
proposal_sigmas["logA"] = 1e-2
proposal_sigmas["ns"] = 1e-2


#Plug the ground truth into sample_joint() to try and learn the LCDM distributions
hpc_path = "/resnick/groups/wugroup/zblood/cmb_lensing/sampling_chains/joint_inference_08_24_26/"
_ = sample_joint(data_set, param_init, proposal_sigmas, param_ranges, should_sample, args.noise_level,
                 iters_per_chain = args.iters_per_chain, num_burn_in_fix_theta = args.num_burn_in_fix_theta,
                 map_idx = args.map_seed, sub_chain_idx = args.chain, seed = None,
                 num_burn_in_always_accept = 0, hpc_path = hpc_path)
