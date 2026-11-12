#!/bin/bash

#Run with "bash get_delensed_field_covariance.sh" (not sbatch): it only submits the jobs. It moves
#into its own folder and sources the untracked repo-root hpc_local.sh (conda,
#SBATCH_MAIL_USER; copy hpc_local.example.sh to create it), which the jobs inherit. Submission is
#throttled (../hpc_submit.sh), so the driver runs until the last job is submitted: start it
#with "nohup bash get_delensed_field_covariance.sh > submit.log 2>&1 &" or inside tmux
#
#Every setting below is a default; override any of them as --<name>=<value>, e.g.
#    bash get_delensed_field_covariance.sh --noise_level=10 --qe_filter=lensed --pol=IP \
#        --derivative_params="omch2 logA" --out_dir_name=delensed_covariance_output_10uk
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1

#every --name=value argument becomes the variable name=value
for arg in "$@"; do declare "${arg#--}"; done

source ../hpc_local.sh || exit 1
source ../hpc_submit.sh || exit 1

#Driver for the EMPIRICAL delensed covariance stencil. Spawns one job per seed: each runs
#load_sim at theta_0 and at theta_0 +/- h_i for every parameter in derivative_params, ALL at
#its one seed (common random numbers - across the stencil the fields differ only through the
#cosmology), inverse-lenses the noiseless lensed field by the Wiener-filtered TRUE phi,
#with W = C_phi / (C_phi + N_phi) held at theta_0, and stores the per-mode covariance of
#every component of the delensed field on the rfft grid, at each of the 1 + 2k stencil points.

#systematics: MATCH sample_lcdm.sh so the measurement is comparable to the chains and to
#the forecasts quoted against them
nside=${nside:-128}
theta_pix=${theta_pix:-2.5}
noise_level=${noise_level:-5}
l_knee=${l_knee:-0}

#the data's polarization: I (temperature; stores TT), P (polarization; EE, BB) or IP (both;
#TT, TE, EE, BB). The forecast must be run with the same --pol
pol=${pol:-I}

#the parameters to difference (PARAM_ORDER names, space separated). The forecast must be run
#with --params drawn from this list
derivative_params=${derivative_params:-"omch2 theta_MC_100 logA"}

#the finite difference step size, in units of PARAM_SIGMA (the Python default too,
#delensed_covariance.DEFAULT_STEP_SIGMA). 0.05 is accurate for the configuration tested
#(nside 128, theta_pix 2.5, noise 2.5 / 5 / 10). For any other nside, theta_pix or noise level
#check it by hand: rerun with the step halved and confirm the forecast does not change
#substantially
step_sigma=${step_sigma:-0.05}

# qe_filter: the spectra in the filter of the box QE N0 behind the Wiener weight W (the TT
# estimator for pol I, the EB estimator for P and IP; the response is always unlensed)
# - "unlensed" uses the fully unlensed spectra
# - "lensed" uses the fully lensed spectra
qe_filter=${qe_filter:-unlensed}

#phi_noise: path ON THE HPC to an npz whose `nphi` is a per-mode rfft2 phi noise matrix
#(empty: W from the box QE above)
phi_noise=${phi_noise:-}

#one slurm job per realization
num_realizations=${num_realizations:-100}
seed_prefix=${seed_prefix:-234567}

#name of the output folder shared by every job, created inside this folder; scp it to the
#local machine for the merge
out_dir_name=${out_dir_name:-delensed_covariance_output}
out_dir="$here/$out_dir_name"

mkdir -p "$out_dir"

#audit trail: the submission date and every setting, rewritten on every run
cat > "$out_dir/submission_settings.txt" <<EOF
submitted $(date "+%Y-%m-%d %H:%M:%S %Z") by $USER on $(hostname)
nside=$nside
theta_pix=$theta_pix
noise_level=$noise_level
l_knee=$l_knee
derivative_params=$derivative_params
step_sigma=$step_sigma
pol=$pol
qe_filter=$qe_filter
phi_noise=$phi_noise
num_realizations=$num_realizations
seed_prefix=$seed_prefix
out_dir_name=$out_dir_name
EOF

#every job uses a distinct seed; load_covariance_directory refuses duplicates, so a loop
#change that repeats one fails the merge loudly instead of double counting
for ((m=0; m<num_realizations; m++)); do
    map_seed=$((seed_prefix + m))
    submit_throttled run_single_delensed_field_covariance.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$step_sigma" "$qe_filter" "$pol" "$phi_noise" \
        "$out_dir" \
        $derivative_params
done
