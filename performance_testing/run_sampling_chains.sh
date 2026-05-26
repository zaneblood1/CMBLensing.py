#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#parameter values for the ground truth simulated data
master_seed=42
nside=128
theta_pix=2.5
noise_level=5
a_phi_ground=0.75
pol="I"

#MCMC parameters for the experiment
num_chains=30
a_phi_min=0.5
a_phi_max=1.5
a_phi_delta=$(echo "($a_phi_max - $a_phi_min) / $num_chains" | bc -l)
num_burn_in_fix_theta=100 
iters_per_chain=1000
num_burn_in_always_accept=0

#store all these common parameters in an 
#array to pass to our slurm jobs
args=(
      "$master_seed"
      "$nside"
      "$theta_pix"
      "$noise_level"
      "$a_phi_ground"
      "$pol"
      "$num_burn_in_fix_theta"
      "$iters_per_chain"
      "$num_burn_in_always_accept"
)

#spawn off a separate job for each chain
for ((chain=0; chain<=num_chains; chain++)); do
    a_phi_init=$(echo "$a_phi_min + $chain * $a_phi_delta" | bc -l)
    sbatch run_single_chain.sh "$chain" "$a_phi_init" "${args[@]}"
done
