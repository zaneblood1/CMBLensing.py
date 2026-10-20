#!/bin/bash

#Run with "bash get_sampler_phi_noise.sh" (not sbatch): it only submits the jobs. It
#moves into its own folder and sources the untracked repo-root hpc_local.sh (conda,
#SBATCH_MAIL_USER; copy hpc_local.example.sh to create it), which the jobs inherit
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1
source ../hpc_local.sh || exit 1

#Driver for the SAMPLER EFFECTIVE NOISE: the phi reconstruction noise the sampler's own
#posterior mean reaches. Spawns one job per rng seed. Each job builds the deterministic
#(f, phi, d) realization load_sim gives at its seed, then runs ONE chain of sample_joint's
#Gibbs sweep with theta HELD at the ground truth - (f, phi) are sampled, theta is not -
#started at phi = 0, and stores EVERY phi sample (no burn-in cut, no thinning).
#
#The analysis burns each chain in, averages it into the posterior mean phi_mean, and cross
#correlates that with the realization's true phi. Per mode, with the moments summed over the
#realizations before the ratio is taken:
#
#    r_L^2 = <phi_mean phi_true*>^2 / (<|phi_mean|^2> <|phi_true|^2>)
#    N_L   = C_L^phiphi (1 / r_L^2 - 1)
#
#with a delete-one-realization jackknife error. See cmb_lensing/sampler_phi_noise.py.
#
#The jobs can be analysed WHILE THEY RUN (each flushes its samples every 10 sweeps and the
#merge reads only the sweeps a job reports as complete), on the HPC, from this folder (set
#burn_in in it first):
# sbatch merge_sampler_phi_noise.sh

#systematics: MATCH sample_lcdm.sh so the noise describes the box the chains run on
nside=128
theta_pix=2.5
noise_level=5
l_knee=0

#Gibbs sweeps per chain
num_sweeps=6000

#one slurm job (one data map, one chain) per seed. The moments are summed over the maps per
#mode, so the per-mode noise of N falls as 1/sqrt(num_maps)
num_maps=100
seed_prefix=234567

#wall time and memory per job. A job killed at the wall clock keeps every flushed sweep and
#is analysed as a shorter chain. 
chain_time="18:00:00"
chain_mem="16G"

#output folder shared by every job.
out_dir="$here/sampler_phi_noise_output"
mkdir -p "$out_dir"

#every job uses a distinct seed; the analysis refuses duplicates, so a loop change that
#repeats one fails loudly instead of double counting a realization
for ((m=0; m<num_maps; m++)); do
    map_seed=$((seed_prefix + m))
    sbatch --time="$chain_time" --mem-per-cpu="$chain_mem" \
        get_sampler_phi_noise_1_chain.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$num_sweeps" "$out_dir"
done
