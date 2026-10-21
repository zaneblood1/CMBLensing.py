#!/bin/bash

#Run with "bash run_performance_test.sh" (not sbatch): it only submits the jobs. It moves into
#its own folder and sources the untracked repo-root hpc_local.sh (conda, SBATCH_MAIL_USER,
#CMBLENSING_JL_PATH; copy hpc_local.example.sh to create it), which the jobs inherit. Results
#land in performance_results/ next to the timing scripts
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1
source ../hpc_local.sh || exit 1

#the total number of seeds (f, phi) pairs from load_sim() and the
#total number of trials per (f, phi, map_size, pol) quadruplet combo
num_seeds=10
num_trials=10
theta_pix=2.5

#loop over polarization keys
for pol_key in "I" "P" "IP"; do
    #loop over map size values
    for map_size in 128 256 512 1024; do
        #wall time halves with each halving of the map size: 48 h at 1024, 24 h at 512,
        #12 h at 256, 6 h at 128 (overrides the #SBATCH --time of the job scripts)
        wall_hours=$((48 * map_size / 1024))
        #loop over number of seeds
        for ((seed=1; seed<=num_seeds; seed++)); do
            #loop over number of trials per (f, phi, map size, pol) tuple
            for ((trial=1; trial<=num_trials; trial++)); do
                #spawn a slurm job from a helper script that takes in (seed_id, map_size, trial_id, pol_key)
                #and make sure this file writes to the proper folders... Do this for both julia and python
                sbatch --time=${wall_hours}:00:00 julia_performance_test.sh $map_size $seed $trial $pol_key $theta_pix
                sbatch --time=${wall_hours}:00:00 python_performance_test.sh $map_size $seed $trial $pol_key $theta_pix
            done
        done
    done
done