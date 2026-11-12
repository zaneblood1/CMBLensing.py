#!/bin/bash

#Run with "bash get_map_joint_phi_noise.sh" (not sbatch): it only submits the jobs. It moves
#into its own folder and sources the untracked repo-root hpc_local.sh (conda,
#SBATCH_MAIL_USER; copy hpc_local.example.sh to create it), which the jobs inherit. Submission is
#throttled (../../hpc_submit.sh), so the driver runs until the last job is submitted: start it
#with "nohup bash get_map_joint_phi_noise.sh > submit.log 2>&1 &" or inside tmux
#
#Every setting below is a default; override any of them as --<name>=<value>, e.g.
#    bash get_map_joint_phi_noise.sh --noise_level=10 --out_dir_name=map_joint_phi_noise_output_10uk
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1

#every --name=value argument becomes the variable name=value
for arg in "$@"; do declare "${arg#--}"; done

source ../../hpc_local.sh || exit 1
source ../../hpc_submit.sh || exit 1

#Driver for the per-mode MAP_JOINT PHI-NOISE measurement. Spawns one job per seed: each simulates
#a realization at the fiducial cosmology (GROUND_TRUTH), reconstructs phi from the data with
#map_joint, and stores the auto / cross / true-auto moments of the estimate and the true phi on
#every rfft2 mode. Averaged over jobs they give, per mode,
#
#    N_phi = C_phi (1 / r^2 - 1),        r^2 = <phi_GT* phi_hat>^2 / (<|phi_hat|^2><|phi_GT|^2>)

#systematics: MATCH sample_lcdm.sh so the measurement is comparable to the chains and to
#the forecasts quoted against them
nside=${nside:-128}
theta_pix=${theta_pix:-2.5}
noise_level=${noise_level:-5}
l_knee=${l_knee:-0}

#the data's polarization: I (temperature), P (polarization) or IP (both)
pol=${pol:-I}

#map_joint iterations for the phi reconstruction. 30 is the map_joint default.
map_joint_steps=${map_joint_steps:-30}

#one slurm job per realization
num_realizations=${num_realizations:-100}
seed_prefix=${seed_prefix:-234567}

#name of the output folder shared by every job, created inside this folder; the merge points here
out_dir_name=${out_dir_name:-map_joint_phi_noise_output}
out_dir="$here/$out_dir_name"
mkdir -p "$out_dir"

#audit trail: the submission date and every setting, rewritten on every run
cat > "$out_dir/submission_settings.txt" <<EOF
submitted $(date "+%Y-%m-%d %H:%M:%S %Z") by $USER on $(hostname)
nside=$nside
theta_pix=$theta_pix
noise_level=$noise_level
l_knee=$l_knee
pol=$pol
map_joint_steps=$map_joint_steps
num_realizations=$num_realizations
seed_prefix=$seed_prefix
out_dir_name=$out_dir_name
EOF

for ((m=0; m<num_realizations; m++)); do
    map_seed=$((seed_prefix + m))
    submit_throttled get_map_joint_phi_noise_1_realization.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$map_joint_steps" "$out_dir" "$pol"
done
