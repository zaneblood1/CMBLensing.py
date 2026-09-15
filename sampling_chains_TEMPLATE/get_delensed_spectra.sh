#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the absolute path on your HPC for the out_dir variable...

#Driver for the EMPIRICAL delensed-spectrum measurement. Spawns one job per seed: each
#simulates a realization at the fiducial cosmology, reconstructs phi with map_joint,
#inverse-lenses the noiseless lensed field by that estimate with the codebase's own
#lense_flow, and divides the band-averaged result by CAMB's delensed spectrum at the same
#frozen Alens_L. The average over jobs is the transfer function R(l).
#
#WHY THIS EXISTS: fisher_forecast.py --spectra delensed takes its delensed TT spectrum from
#CAMB's get_partially_lensed_cls - a full-sky correlation-function lensing calculation with
#C_L^phiphi scaled by Alens_L = N_L / (C_L + N_L). This codebase instead lenses by
#integrating the LenseFlow ODE on a periodic flat-sky box, and reconstructs phi with a MAP
#estimate rather than a Wiener-filtered quadratic estimator. Nothing has ever checked that
#those two agree. R(l) measures the disagreement, and fisher_forecast's --transfer_function
#applies it so the forecast keeps CAMB's (noiseless) theta dependence while carrying the
#box's actual amplitude - see cmb_lensing/delensed_spectrum.py.
#
#Collect the results with:
#    python merge_delensed_spectra.py --spectra_dir "$out_dir"
#and test that R does not move with cosmology (which is what licenses using one R across the
#whole finite-difference stencil) by running this script a second time with shift_param /
#shift_value set and a different out_dir, then:
#    python compare_transfer_functions.py --reference <dir_a> --shifted <dir_b>

#systematics: MATCH sample_lcdm.sh so the measurement is comparable to the chains and to
#the forecasts quoted against them
nside=128
theta_pix=2.5
noise_level=5
l_knee=0

#band width for the |l| annuli R is reported in. R is smooth, so these exist to beat down
#Monte Carlo noise rather than to resolve structure - widen them if R is still noisy after
#num_realizations jobs, rather than adding more jobs
delta_ell=100

#map_joint iterations for the phi reconstruction. 30 is the map_joint default
map_joint_steps=30

#must MATCH the fisher_forecast run this R will be applied to: both feed Alens_L, which sets
#the CAMB reference spectrum in the denominator of R
nphi_source=covariance
qe_response=unlensed

#one slurm job per realization
num_realizations=100
seed_prefix=246813

#leave shift_param empty for the production run. To measure R at a DISPLACED cosmology for
#the flatness test, set shift_param to one of PARAM_ORDER and shift_value to an absolute
#displacement - make it several sigma, far wider than fisher_forecast's 0.05 sigma
#finite-difference step, so that a flat result there certainly implies flatness over the
#step. Give the shifted run its own out_dir: the merge refuses to average across cosmologies
shift_param=""
shift_value=0

#output folder shared by every job; the merge and --transfer_function point here
out_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/delensed_spectra_output"

mkdir -p "$out_dir"

#each job gets a distinct seed. load_transfer_directory refuses duplicate seeds, so if this
#loop is ever changed in a way that repeats one, the merge fails loudly instead of double
#counting a realization
for ((m=0; m<num_realizations; m++)); do
    map_seed=$((seed_prefix + m))
    sbatch get_single_delensed_spectra.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$delta_ell" "$map_joint_steps" "$nphi_source" \
        "$qe_response" "$out_dir" "$shift_param" "$shift_value"
done
