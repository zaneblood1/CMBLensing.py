#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#sytematics
nside=128
theta_pix=2.5
noise_level=5
map_prefix=234567
pol="I"

#parameter values for the ground truth simulated data
ombh2_ground=0.022386
omch2_ground=0.109381
theta_MC_100_ground=1.031732
log_a_ground=3.218387
ns_ground=0.959814

#MCMC parameters for the experiment
num_maps=10 #10 total data map realizations of ground truth
num_chains=5 #5 different MCMC chains per data map

#We sample linearly from -5 sigma to +5 sigma with the ground truth equal to the mean (0 sigma)
ombh2_min=0.020413
ombh2_max=0.024389

omch2_min=0.079704
omch2_max=0.155541

theta_MC_100_min=0.9328
theta_MC_100_max=1.1452

log_a_min=2.661635
log_a_max=3.782861

ns_min=0.867143
ns_max=1.042186

num_burn_in_fix_theta=100 
iters_per_chain=1500
num_burn_in_always_accept=0

#store all these common parameters in an 
#array to pass to our slurm jobs
args=(
      "$nside"
      "$theta_pix"
      "$noise_level"
      "$ombh2_min"
      "$ombh2_ground"
      "$ombh2_max"
      "$omch2_min"
      "$omch2_ground"
      "$omch2_max"
      "$theta_MC_100_min"
      "$theta_MC_100_ground"
      "$theta_MC_100_max"
      "$log_a_min"
      "$log_a_ground"
      "$log_a_max"
      "$ns_min"
      "$ns_ground"
      "$ns_max"
      "$pol"
      "$num_burn_in_fix_theta"
      "$iters_per_chain"
      "$num_burn_in_always_accept"
)

#draw a uniform random value in [min, max] using bash's $RANDOM (0..32767)
rand_uniform() {
    local min=$1
    local max=$2
    echo "$min + ($RANDOM / 32767) * ($max - $min)" | bc -l
}

#Parameters held constant for this experiment
ombh2_init=$ombh2_ground
ns_init=$ns_ground

for ((map=1; map<=num_maps; map++)); do
    map_seed=$(echo "$map_prefix * $map" | bc -l)
    for ((chain=1; chain<=num_chains; chain++)); do
        omch2_init=$(rand_uniform "$omch2_min" "$omch2_max")
        log_a_init=$(rand_uniform "$log_a_min" "$log_a_max")
        theta_MC_100_init=$(rand_uniform "$theta_MC_100_min" "$theta_MC_100_max")
        #random initializations for each parameter for each (map, phi) combination
        sbatch run_single_lcdm_chain.sh "$map_seed" "$chain" "$ombh2_init" "$omch2_init" "$theta_MC_100_init" "$log_a_init" "$ns_init" "${args[@]}"
    done
done
