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
#theta - so there is no fast mode of the kind fisher_forecast_from_logpdf has.
#Sequentially that is ~3800 lensing solves; one job per realization makes it 38 apiece.
#MEASURED: one job is 1m41s at nside 128 / 2.5' / 3 params, so the whole 100-job forecast
#finishes in minutes of wall clock once the jobs are scheduled, against a few hours if run
#sequentially in fisher_forecast_from_mixed_logpdf.forecast_from_mixed_logpdf.
#
#Collect the results with:
#    python -m cmb_lensing.fisher_forecast_from_mixed_logpdf --hessian_dir "$out_dir"

#systematics: MATCH sample_lcdm.sh so the forecast is comparable to the chains
nside=128
theta_pix=2.5
noise_level=5
l_knee=0
map_prefix=987654

#the parameters to forecast. must match the was_sampled entries in chain_analysis.py
params="omch2 theta_MC_100 logA"

#one slurm job per realization (per sub-chain of each realization when louis=1)
num_realizations=100

#1 = Louis's observed information (the MARGINAL Fisher of p(d | theta)): each job samples
#(f, phi) | d at theta_0 with the sampler's Gibbs sweep, runs the stencil at every sweep
#and the merge subtracts the posterior covariance of the score. 0 = the plain complete-data
#Hessian. Louis jobs take hours, not minutes - (1 + louis_draws) stencils plus louis_draws
#Gibbs sweeps - so they get louis_time instead of the 20 minutes in
#run_single_mixed_hessian.sh, and should write to their own out_dir.
#louis_chains independent sub-chains run per data map (same map_seed, different MCMC
#randomness), ONE SLURM JOB EACH, so a louis run is num_realizations * louis_chains jobs.
#That divides the wall clock per realization by louis_chains and lets the merge measure the
#Gelman-Rubin R-hat of every score component and Hessian entry. Ignored when louis=0 (the
#plain Hessian has no MCMC).
#louis_draws is the RAW length of each sub-chain, burn-in INCLUDED: every sweep from the
#first is differentiated and saved. There is deliberately no burn-in here - it is chosen at
#merge time (--louis_burn, LOUIS_BURN_IN in chain_analysis.py) after looking at the traces,
#so a burn-in that turns out too short costs a re-merge, not a re-run. The merge then thins
#each sub-chain by its max IAT over scores and Hessian entries and pools the sub-chains;
#raise louis_draws or louis_chains if it warns that a realization prunes to too few samples
#louis_mem overrides run_single_mixed_hessian.sh's 2G: a louis job also carries the sampler
#stack (sample_lcdm, map_joint, the HMC), MEASURED at 2.5 GB peak RSS on nside 64. The
#(f, phi) draws themselves are NOT what costs memory - each is differentiated and dropped
#as the chain produces it, so louis_draws does not move this number
louis=0
louis_draws=150
louis_chains=4
louis_time="06:00:00"
louis_mem="4G"

#output folder shared by every job; --hessian_dir points here
out_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/mixed_hessian_output"

mkdir -p "$out_dir"

#each realization gets a distinct seed and each of its sub-chains a distinct index.
#load_hessian_directory refuses duplicate seeds (duplicate (seed, sub-chain) pairs for louis
#files), so if this loop is ever changed in a way that repeats one, the averaging step fails
#loudly instead of double counting
time_args=()
num_chains=1
if [ "$louis" = "1" ]; then
    time_args=(--time="$louis_time" --mem-per-cpu="$louis_mem")
    num_chains=$louis_chains
fi
for ((m=0; m<num_realizations; m++)); do
    map_seed=$((map_prefix + m))
    for ((c=0; c<num_chains; c++)); do
        sbatch "${time_args[@]}" run_single_mixed_hessian.sh "$m" "$map_seed" "$nside" \
            "$theta_pix" "$noise_level" "$l_knee" "$out_dir" "$louis" "$louis_draws" \
            "$c" $params
    done
done
