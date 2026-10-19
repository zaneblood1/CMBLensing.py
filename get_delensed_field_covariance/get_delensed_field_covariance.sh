#!/bin/bash

#Run with "bash get_delensed_covariance.sh" (not sbatch): it only submits the jobs. It moves
#into its own folder and sources the untracked repo-root hpc_local.sh (conda,
#SBATCH_MAIL_USER; copy hpc_local.example.sh to create it), which the jobs inherit
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1
source ../hpc_local.sh || exit 1

#Driver for the EMPIRICAL delensed covariance stencil. Spawns one job per seed: each runs
#load_sim at theta_0 and at theta_0 +/- h_i for every parameter in derivative_params, ALL at
#its one seed (common random numbers - across the stencil the fields differ only through the
#cosmology), inverse-lenses the noiseless lensed field by the Wiener-filtered TRUE phi,
#with W = C_phi / (C_phi + N_phi) held at theta_0, and stores |rfft2|^2 / nside^2 on
#the rfft grid: num_realizations x (1 + 2k) covariance matrices in total.

#systematics: MATCH sample_lcdm.sh so the measurement is comparable to the chains and to
#the forecasts quoted against them
nside=128
theta_pix=2.5
noise_level=5
l_knee=0

#the parameters to difference (PARAM_ORDER names, space separated). The forecast must be run
#with --params drawn from this list
derivative_params="omch2 theta_MC_100 logA"

#the finite difference step size, in units of PARAM_SIGMA. Good to check that halving this term
#results in no changes to forecasting. step_sigma = 0.05 has been fine for this configuration in the past
step_sigma=0.05

# qe_filter: the TT spectrum in the box QE's filter
# - "delensed" performs iterative delensing via CAMB
# - "unlensed" uses the fully unlensed temperature spectrum
# - "lensed" uses the fully lensed temperature spectrum
qe_filter=unlensed

# qe_response: which TT spectrum weights the QE response - "unlensed" (Hu & Okamoto) or
# "gradient" (CAMB's lensed T-grad-T, Lewis, Challinor & Hanson 2011)
qe_response=unlensed

#phi_noise: path ON THE HPC to an npz whose `nphi` is a per-mode rfft2 phi noise matrix
phi_noise=""

#one slurm job per realization
num_realizations=100
seed_prefix=234567

#output folder shared by every job; scp this to the local machine for the merge
out_dir="$here/delensed_covariance_output"

mkdir -p "$out_dir"
#every job uses a distinct seed; load_covariance_directory refuses duplicates, so a loop
#change that repeats one fails the merge loudly instead of double counting
for ((m=0; m<num_realizations; m++)); do
    map_seed=$((seed_prefix + m))
    sbatch run_single_delensed_field_covariance.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$step_sigma" "$qe_filter" "$qe_response" "$phi_noise" \
        "$out_dir" \
        $derivative_params
done
