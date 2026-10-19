#!/bin/bash

#Run with "bash get_map_joint_phi_noise.sh" (not sbatch): it only submits the jobs. It moves
#into its own folder and sources the untracked repo-root hpc_local.sh (conda,
#SBATCH_MAIL_USER; copy hpc_local.example.sh to create it), which the jobs inherit
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1
source ../hpc_local.sh || exit 1

#Driver for the per-mode MAP_JOINT PHI-NOISE measurement. Spawns one job per seed: each simulates
#a realization at the fiducial cosmology (GROUND_TRUTH), reconstructs phi from the data with
#map_joint, and stores the auto / cross / true-auto moments of the estimate and the true phi on
#every rfft2 mode. Averaged over jobs they give, per mode,
#
#    N_phi = C_phi (1 / r^2 - 1),        r^2 = <phi_GT* phi_hat>^2 / (<|phi_hat|^2><|phi_GT|^2>)

#systematics: MATCH sample_lcdm.sh so the measurement is comparable to the chains and to
#the forecasts quoted against them
nside=128
theta_pix=2.5
noise_level=5
l_knee=0

#map_joint iterations for the phi reconstruction. 30 is the map_joint default.
map_joint_steps=30

#one slurm job per realization
num_realizations=100
seed_prefix=234567

#output folder shared by every job; the merge points here
out_dir="$here/map_joint_phi_noise_output"
mkdir -p "$out_dir"

for ((m=0; m<num_realizations; m++)); do
    map_seed=$((seed_prefix + m))
    sbatch get_map_joint_phi_noise_1_realization.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$map_joint_steps" "$out_dir"
done
