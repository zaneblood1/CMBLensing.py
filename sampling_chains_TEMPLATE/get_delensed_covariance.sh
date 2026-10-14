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
#cosmology), inverse-lenses the noiseless lensed field by the Wiener-filtered TRUE phi,
#W phi with W = C_phi / (C_phi + N_phi) held at theta_0 (CAMB C_phi, the box's physical QE N0),
#and stores |rfft2|^2 / nside^2 on the rfft grid (map_joint still runs at each point, for the
#phi moments below):
#num_realizations x (1 + 2k) covariance matrices in total. The same jobs also store, per
#rfft mode and stencil point, |phi_hat|^2, Re(phi_hat phi*) and |phi|^2, from which the merge
#builds the per-mode empirical phi noise (fisher_forecast --empirical_phi_noise).
#
#WHY THIS EXISTS: get_delensed_spectra.sh only corrects CAMB's delensed spectrum by a
#transfer function R(l) (and optionally dR/dtheta); the forecast's delensed block and its
#theta dependence are still CAMB's. This replaces the delensed block outright, value AND
#finite difference, with what this codebase's lense_flow produces - see
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

#the N_phi inside the delensing Wiener weight W = C_phi / (C_phi + N_phi) (built once at
#theta_0, applied to the true phi at every stencil point). Neither touches map_joint's own QE
#norm. Run the forecast with the matching --iterative_delens / --qe_response so its phi block
#sits on the same N_phi (it warns otherwise); the merge refuses to mix settings.
#iterative_delens: 1 iterates N_phi against the delensed filter exactly as
#fisher_forecast.iterative_delensing does (box QE matrix; the filter's TT goes from lensed to
#CAMB's delensed TT at the converged Alens_L), 0 the one-shot lensed-filter N_phi
iterative_delens=0
#qe_response: which TT spectrum weights the QE response - "unlensed" (Hu & Okamoto) or
#"gradient" (CAMB's lensed T-grad-T, Lewis, Challinor & Hanson 2011)
qe_response=unlensed
#phi_noise: path ON THE HPC to an npz holding a per-mode phi noise estimate measured by an
#EARLIER run at THIS box (nside / theta_pix / noise_level / l_knee) and cosmology. When set,
#W's N_phi is that matrix instead of the box QE matrix and the jobs do NOT run map_joint at
#all: map_joint_steps / reconstruction / constant_nphi then do nothing. Accepted files:
#  - merge_sampler_noise_estimate.py sampler_noise_estimate.npz: the score bound 1/<F_phi>
#    (`nphi`); with phi_noise_key=nphi_map_joint, map_joint's fixed-phi noise instead
#  - merge_sampler_effective_noise.py sampler_effective_noise*.npz: sample_joint's effective
#    noise (`nphi`; phi_noise_key=nphi_smooth for the band-smoothed one)
#  - merge_delensed_covariance.py delensed_covariance.npz whose jobs stored the phi moments:
#    map_joint's N_eff (`phi_noise_fid`; unmeasured modes get W = 0)
#In place of its own phi moments a run given a delensed_covariance.npz COPIES that file's
#merged map_joint phi grids (copied_phi_moments.npz in out_dir, passed through by the merge),
#so --empirical_phi_noise still works; the +/- grids (--vary_nphi, --empirical_phi_block) are
#copied only if that file's stencil has every derivative_params entry at this step_sigma. The
#other two kinds of file have no moments to copy. Requires iterative_delens=0 (refused
#otherwise); qe_response then does not apply. The job files record what the matrix is, its
#key and a checksum; the merge refuses to mix, and the forecast warns unless it runs
#--nphi_source score --phi_noise <same npz> on the same matrix (add --map_joint_phi_noise
#for phi_noise_key=nphi_map_joint). Leave empty for the QE matrix
phi_noise=""
#phi_noise_key: which 2D grid of the phi_noise file to use. Empty takes the file's default
#(`nphi`, else `phi_noise_fid`)
phi_noise_key=""

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
        "$constant_nphi" "$iterative_delens" "$qe_response" "$phi_noise" "$phi_noise_key" \
        "$out_dir" \
        $derivative_params
done
