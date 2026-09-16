#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the absolute path on your HPC for the out_dir variable...

#Driver for the EFFECTIVE PHI-NOISE measurement. Spawns one job per seed: each simulates a
#realization at the fiducial cosmology, reconstructs phi from the data with map_joint, and
#cross correlates that estimate against the true phi in |L| annuli. The average over jobs
#gives the correlation coefficient r^2, and with it
#
#    N_L^eff = C_L^phiphi (1 / r_L^2 - 1)        <=>        r_L^2 = C_L / (C_L + N_L^eff)
#
#WHY THIS EXISTS: every Fisher forecast in this package writes the lensing block as
#C_L^phiphi + N_L and takes N_L from a quadratic estimator, but this codebase reconstructs
#phi with map_joint's MAP estimate, not with a QE. N_L^eff is what that reconstruction
#actually achieves, and it feeds BOTH the phi block and - through
#Alens_L = N_L / (C_L + N_L) - the delensed temperature block. See cmb_lensing/phi_noise.py.
#
#Collect the results with:
#    python merge_phi_noise.py --noise_dir "$out_dir"
#and use them with:
#    python -m cmb_lensing.fisher_forecast --spectra delensed --nphi_source measured \
#        --phi_noise "$out_dir/effective_phi_noise.npz"
#
#N_eff is measured at ONE cosmology and then held fixed across the forecast's whole
#finite-difference stencil. To check how much it really moves with theta, run this script a
#second time with shift_param / shift_value set and a different out_dir, then compare the two
#merged files.

#systematics: MATCH sample_lcdm.sh so the measurement is comparable to the chains and to
#the forecasts quoted against them
nside=128
theta_pix=2.5
noise_level=5
l_knee=0

#width of the |L| annuli N_eff is reported in. N_eff is smooth in L, so these exist to beat
#down Monte Carlo noise rather than to resolve structure - widen them if N_eff is still noisy
#after num_realizations jobs, rather than adding more jobs
delta_ell=100

#map_joint iterations for the phi reconstruction. 30 is the map_joint default. N_eff is a
#property of the reconstruction AS RUN, so an under-converged map_joint inflates it - check
#that N_eff has stopped moving between 30 and 60 steps before trusting a production run
map_joint_steps=30

#which TT spectrum the ANALYTIC reference N^(0) is built from. It sets only the comparison
#the merge reports (N_eff / N^(0) and the rung-1 check), not the measurement itself
qe_response=unlensed

#one slurm job per realization
num_realizations=100
seed_prefix=246813

#leave shift_param empty for the production run. To measure N_eff at a DISPLACED cosmology,
#set shift_param to one of PARAM_ORDER and shift_value to an absolute displacement - make it
#several sigma, far wider than fisher_forecast's 0.05 sigma finite-difference step, so that a
#flat result there certainly implies flatness over the step. Give the shifted run its own
#out_dir: the merge refuses to average across cosmologies
shift_param=""
shift_value=0

#output folder shared by every job; the merge and --phi_noise point here
out_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/phi_noise_output"

mkdir -p "$out_dir"

#each job gets a distinct seed. load_phi_noise_directory refuses duplicate seeds, so if this
#loop is ever changed in a way that repeats one, the merge fails loudly instead of double
#counting a realization
for ((m=0; m<num_realizations; m++)); do
    map_seed=$((seed_prefix + m))
    sbatch get_phi_noise_1_realization.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$delta_ell" "$map_joint_steps" "$qe_response" \
        "$out_dir" "$shift_param" "$shift_value"
done
