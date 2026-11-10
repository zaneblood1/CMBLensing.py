#!/bin/bash

#Run with "bash run_performance_test.sh" (not sbatch): it only submits the jobs. It moves into
#its own folder and sources the untracked repo-root hpc_local.sh (conda, SBATCH_MAIL_USER,
#CMBLENSING_JL_PATH; copy hpc_local.example.sh to create it), which the jobs inherit. Results
#land in <out_dir_name>/ next to the timing scripts. Submission is throttled
#(../hpc_submit.sh), so the driver runs until the last job is submitted: start it with
#"nohup bash run_performance_test.sh > submit.log 2>&1 &" or inside tmux
#
#Every setting below is a default; override any of them as --<name>=<value> (quote lists), e.g.
#    bash run_performance_test.sh --map_sizes="128 256" --pol_keys="I" --out_dir_name=performance_results_small
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1

#every --name=value argument becomes the variable name=value
for arg in "$@"; do declare "${arg#--}"; done

source ../hpc_local.sh || exit 1
source ../hpc_submit.sh || exit 1

#the total number of seeds (f, phi) pairs from load_sim() and the
#total number of trials per (f, phi, map_size, pol) quadruplet combo
num_seeds=${num_seeds:-10}
num_trials=${num_trials:-10}
theta_pix=${theta_pix:-2.5}
#temperature noise in uK-arcmin, handed to both load_sims (3 is CMBLensing.jl's default)
noise_level=${noise_level:-3}

#polarization keys and map sizes to time (space separated)
pol_keys=${pol_keys:-"I P IP"}
map_sizes=${map_sizes:-"128 256 512 1024"}

#wall time of the largest map size, in hours; it scales linearly with the map size (48 h at
#1024, 24 h at 512, 12 h at 256, 6 h at 128 by default) and overrides the #SBATCH --time of
#the job scripts
max_wall_hours=${max_wall_hours:-48}

#name of the results folder, created inside this folder
out_dir_name=${out_dir_name:-performance_results}
out_dir="$here/$out_dir_name"
mkdir -p "$out_dir"

#the largest map size, which gets max_wall_hours
largest_map_size=0
for map_size in $map_sizes; do
    if [ "$map_size" -gt "$largest_map_size" ]; then largest_map_size=$map_size; fi
done

#audit trail: the submission date and every setting, rewritten on every run
cat > "$out_dir/submission_settings.txt" <<EOF
submitted $(date "+%Y-%m-%d %H:%M:%S %Z") by $USER on $(hostname)
num_seeds=$num_seeds
num_trials=$num_trials
theta_pix=$theta_pix
noise_level=$noise_level
pol_keys=$pol_keys
map_sizes=$map_sizes
max_wall_hours=$max_wall_hours
out_dir_name=$out_dir_name
EOF

#loop over polarization keys
for pol_key in $pol_keys; do
    #loop over map size values
    for map_size in $map_sizes; do
        #rounded up to whole hours
        wall_hours=$(( (max_wall_hours * map_size + largest_map_size - 1) / largest_map_size ))
        #loop over number of seeds
        for ((seed=1; seed<=num_seeds; seed++)); do
            #loop over number of trials per (f, phi, map size, pol) tuple
            for ((trial=1; trial<=num_trials; trial++)); do
                #spawn a slurm job from a helper script that takes in (seed_id, map_size, trial_id, pol_key)
                #and make sure this file writes to the proper folders... Do this for both julia and python
                for job_script in julia_performance_test.sh python_performance_test.sh; do
                    submit_throttled --time=${wall_hours}:00:00 $job_script $map_size $seed $trial $pol_key \
                        $theta_pix $noise_level "$out_dir"
                done
            done
        done
    done
done
