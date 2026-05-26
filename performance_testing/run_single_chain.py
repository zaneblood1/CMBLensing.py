import os
from cmb_lensing.sampling import *
from cmb_lensing.simulate import *
import argparse
import time

#Parse the arguments of the slurm job
parser = argparse.ArgumentParser()
parser.add_argument("--chain", type = int)
parser.add_argument("--master_seed", type = int)
parser.add_argument("--nside", type = int)
parser.add_argument("--theta_pix", type = float)
parser.add_argument("--a_phi_ground", type = float)
parser.add_argument("--a_phi_init", type = float)
parser.add_argument("--noise_level", type = float)
parser.add_argument("--pol", type = str)
parser.add_argument("--num_burn_in_fix_theta", type = int)
parser.add_argument("--iters_per_chain", type = int)
parser.add_argument("--num_burn_in_always_accept", type = int)
args = parser.parse_args()

#First run a simulation given the master_seed to get the same data field across chains
data_set = load_sim(nside = args.nside, 
                    theta_pix = args.theta_pix, 
                    pol = args.pol, 
                    master_seed = args.master_seed, 
                    a_phi = args.a_phi_ground, 
                    uk_arcmin_t = args.noise_level)
data_field = data_set.data

#Plug the ground truth into sample_joint() to try and learn the a_phi distribution
start_time = time.time()
a_phi_distribution = sample_joint(data_field, 
                     num_burn_in_fix_theta = args.num_burn_in_fix_theta, 
                     iters_per_chain = args.iters_per_chain, 
                     num_burn_in_always_accept = args.num_burn_in_always_accept,
                     noise_level = args.noise_level,
                     a_phi_init = args.a_phi_init, seed = args.chain)
end_time = time.time()

#save the a_phi_distribution to the correct data folder
file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/chains/"
os.makedirs(file_path, exist_ok = True)
np.savez(file_path + f"chain_{args.chain}.npz", a_phi_distribution)

#record the time as well
total_time = end_time - start_time
np.savetxt(file_path + f"chain_{args.chain}_time.txt", np.array([total_time]))