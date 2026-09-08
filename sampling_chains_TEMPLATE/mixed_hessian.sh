#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the absolute path on your HPC for the out_dir variable...

#Driver for the MIXED-coordinate Hessian forecast. Spawns one job per realization: each
#draws (f, phi) at the fiducial cosmology, mixes them once with D and G there, and
#finite-differences statistics.mixed_logpdf in theta at that fixed mixed pair.
#
#Why this is fanned out and the other forecasts are not: mixed_logpdf costs TWO lensing
#solves per evaluation (an inverse one inside unmix, a forward one inside logpdf), and
#nothing cancels across the stencil because D, G and the unmixed fields all move with
#theta - so there is no fast mode of the kind fisher_forecast's unmixed logpdf path has.
#Sequentially that is ~3800 lensing solves; one job per realization makes it 38 apiece.
#MEASURED: one job is 1m41s at nside 128 / 2.5' / 3 params, so the whole 100-job forecast
#finishes in minutes of wall clock once the jobs are scheduled, against a few hours if run
#sequentially in fisher_forecast.forecast_from_mixed_logpdf.
#
#Collect the results with:
#    python -m cmb_lensing.fisher_forecast --hessian_dir "$out_dir"

#systematics: MATCH sample_lcdm.sh so the forecast is comparable to the chains
nside=128
theta_pix=2.5
noise_level=5
l_knee=0
map_prefix=987654

#the parameters to forecast. must match the was_sampled entries in chain_analysis.py
params="omch2 theta_MC_100 logA"

#one slurm job per realization
num_realizations=100

#output folder shared by every job; --hessian_dir points here
out_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/mixed_hessian_output"

mkdir -p "$out_dir"

#each job gets a distinct seed. load_hessian_directory refuses duplicate seeds, so if this
#loop is ever changed in a way that repeats one, the averaging step fails loudly instead
#of double counting a realization
for ((m=0; m<num_realizations; m++)); do
    map_seed=$((map_prefix + m))
    sbatch run_single_mixed_hessian.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$out_dir" $params
done
