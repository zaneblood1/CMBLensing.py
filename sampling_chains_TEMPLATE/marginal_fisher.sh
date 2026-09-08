#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the absolute path on your HPC for the out_dir variable...

#Driver for the MARGINAL Fisher forecast (cmb_lensing/marginal_fisher.py). Unlike
#sample_lcdm.sh this spawns ONE job per data map, not one per (map, chain): theta is held
#completely fixed, so there is no theta chain to run several of - each job draws
#(f, phi) ~ p(f, phi | d, theta_0) and writes that map's complete-data scores.
#
#These jobs are CHEAPER per sweep than sample_lcdm.sh's because the theta Metropolis step
#- the dominant cost there - is absent entirely.

#systematics: MUST MATCH sample_lcdm.sh for the forecast to be comparable to the chains
nside=128
theta_pix=2.5
noise_level=5
l_knee=0
map_prefix=234567

#the parameters to forecast. must match the was_sampled entries in chain_analysis.py
params="omch2 theta_MC_100 logA"

#Number of data realizations. Louis's identity gives the OBSERVED information for one map,
#so averaging over maps turns it into a forecast; the score-covariance cross-check is a
#covariance ACROSS maps and needs strictly more than n_params of them to be non-singular.
num_maps=50

#Retained draws per map, and the sweeps discarded first. The Louis subtraction is only as
#good as the EFFECTIVE sample size behind Cov(score): the relative error on that term goes
#as sqrt(2 / N_eff), and N_eff = n_draws / tau with tau the integrated autocorrelation time
#the job prints. At nside 64 tau came out ~2, so 2000 draws buys N_eff ~ 1000 and a ~4%
#error on Cov(score). Check the printed tau before trusting a shorter run.
n_draws=2000
burn_in=300

#output folder shared by every job
out_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/marginal_fisher_output"

mkdir -p "$out_dir"

#spawn one job per data map
for ((m=0; m<num_maps; m++)); do
    map_seed=$((map_prefix + m))
    sbatch run_single_marginal_fisher.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$n_draws" "$burn_in" "$out_dir" $params
done
