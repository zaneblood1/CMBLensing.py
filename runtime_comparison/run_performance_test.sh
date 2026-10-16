#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#the total number of seeds (f, phi) pairs from load_sim() and the
#total number of trials per (f, phi, map_size, pol) quadruplet combo
num_seeds=10
num_trials=10
theta_pix=2.5

#loop over polarization keys
for pol_key in "I" "P" "IP"; do
    #loop over map size values
    for map_size in 128 256 512 1024; do
        #loop over number of seeds
        for ((seed=1; seed<=num_seeds; seed++)); do
            #loop over number of trials per (f, phi, map size, pol) tuple
            for ((trial=1; trial<=num_trials; trial++)); do
                #spawn a slurm job from a helper script that takes in (seed_id, map_size, trial_id, pol_key)
                #and make sure this file writes to the proper folders... Do this for both julia and python
                sbatch julia_performance_test.sh $map_size $seed $trial $pol_key $theta_pix 
                sbatch python_performance_test.sh $map_size $seed $trial $pol_key $theta_pix
            done
        done
    done
done