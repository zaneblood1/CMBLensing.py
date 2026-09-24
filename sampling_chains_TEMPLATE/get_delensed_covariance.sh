#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the absolute path on your HPC for the out_dir variable...

#Driver for the EMPIRICAL delensed covariance stencil. Spawns one job per seed: each runs
#load_sim at theta_0 and at theta_0 +/- h_i for every parameter in derivative_params, ALL at
#its one seed (common random numbers - across the stencil the fields differ only through the
#cosmology), reconstructs phi with map_joint at each point, inverse-lenses the noiseless
#lensed field by that estimate, and stores |rfft2|^2 / nside^2 on the rfft grid:
#num_realizations x (1 + 2k) covariance matrices in total. The same jobs also store, per
#rfft mode and stencil point, |phi_hat|^2, Re(phi_hat phi*) and |phi|^2, from which the merge
#builds the per-mode empirical phi noise (fisher_forecast --empirical_phi_noise).
#
#WHY THIS EXISTS: get_delensed_spectra.sh only corrects CAMB's delensed spectrum by a
#transfer function R(l) (and optionally dR/dtheta); the forecast's delensed block and its
#theta dependence are still CAMB's. This replaces the delensed block outright, value AND
#finite difference, with what this codebase's lense_flow and map_joint produce - see
#cmb_lensing/delensed_covariance.py.
#
#Once every job has finished, scp out_dir to the local machine and run:
#    python merge_delensed_covariance.py --covariance_dir <local copy of out_dir>
#    python -m cmb_lensing.fisher_forecast --spectra delensed --params <same params> \
#        --nside ... --theta_pix ... --noise ... \
#        --delensed_covariance <local copy of out_dir>/delensed_covariance.npz

#systematics: MATCH sample_lcdm.sh so the measurement is comparable to the chains and to
#the forecasts quoted against them
nside=128
theta_pix=2.5
noise_level=5
l_knee=0

#map_joint iterations for the phi reconstruction. 30 is the map_joint default
map_joint_steps=30

#the parameters to difference (PARAM_ORDER names, space separated). The forecast must be run
#with --params drawn from this list
derivative_params="omch2 theta_MC_100 logA"

#the stencil's h_i, in units of PARAM_SIGMA. Wider than the forecast's 0.05 sigma on purpose:
#each point runs its own map_joint, and a narrow step would difference its convergence
#noise. The merge prints the second-order term per band as the linearity check. The forecast
#takes its steps from the merged file, for the phi block too
step_sigma=0.5

#which cosmology map_joint's C_f / C_phi / D / QE norm are built at for the displaced points:
#"fiducial" holds them at theta_0 (a frozen estimator, the convention every other delensing
#derivative in the codebase uses); "shifted" rebuilds them at each point's own cosmology
reconstruction=fiducial

#reconstruction=shifted only ("fiducial" freezes N_phi with everything else): 1 keeps
#map_joint's QE norm N_phi at theta_0 while C_f / C_phi / D follow each point's cosmology - the
#same frozen-N_phi convention as fisher_forecast's default phi block; 0 rebuilds it per point
#(what every shifted run before this flag did). N_phi only preconditions map_joint's phi step,
#so it moves phi_hat through incomplete convergence, not through the MAP optimum. Run the
#forecast with --vary_nphi when this is 0 so both blocks share one convention
constant_nphi=1

#one slurm job per realization
num_realizations=100
seed_prefix=246813

#output folder shared by every job; scp this to the local machine for the merge
out_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/delensed_covariance_output"

mkdir -p "$out_dir"
#every job uses a distinct seed; load_covariance_directory refuses duplicates, so a loop
#change that repeats one fails the merge loudly instead of double counting
for ((m=0; m<num_realizations; m++)); do
    map_seed=$((seed_prefix + m))
    sbatch run_single_delensed_covariance.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$map_joint_steps" "$step_sigma" "$reconstruction" \
        "$constant_nphi" "$out_dir" $derivative_params
done
