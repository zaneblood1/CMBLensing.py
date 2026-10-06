#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the absolute path on your HPC for the out_dir variable...

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
#with a delete-one-realization jackknife error. See cmb_lensing/sampler_effective_noise.py.
#
#The jobs can be analysed WHILE THEY RUN (each flushes its samples every 10 sweeps and the
#analysis reads only the sweeps a job reports as complete), on the HPC:
#    python merge_sampler_effective_noise.py --chain_dir <out_dir> --burn_in 200
#and the merged file plugs into the forecast as an N_phi matrix:
#    python -m cmb_lensing.fisher_forecast --nphi_source score \
#        --phi_noise <out_dir>/sampler_effective_noise.npz ...

#systematics: MATCH sample_lcdm.sh so the noise describes the box the chains run on
nside=128
theta_pix=2.5
noise_level=5
l_knee=0

#Gibbs sweeps per chain, INCLUDING whatever the analysis later cuts as burn-in. The chain
#mean's Monte Carlo error falls as IAT / sweeps per mode, and the "naive" estimate's bias as
#(1 + N / C) IAT / sweeps, so the high-L modes (N >> C) are what set this.
#MEASURED 1.35 s per sweep at nside 64 / 5' (8 local cores) after ~25 s of compilation
num_sweeps=3000

#one slurm job (one data map, one chain) per seed. The moments are summed over the maps per
#mode, so the per-mode noise of N falls as 1/sqrt(num_maps)
num_maps=50
seed_prefix=345678

#wall time and memory per job. A job killed at the wall clock keeps every flushed sweep and
#is analysed as a shorter chain. MEASURED 1.35 GB peak RSS at nside 64
chain_time="12:00:00"
chain_mem="4G"

#output folder shared by every job. Each chain stores num_sweeps complex64 rfft grids:
#66 kB per sweep at nside 128, so 200 MB per chain at 3000 sweeps and 10 GB for 50 maps
out_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/sampler_effective_noise_output"

mkdir -p "$out_dir"
#every job uses a distinct seed; the analysis refuses duplicates, so a loop change that
#repeats one fails loudly instead of double counting a realization
for ((m=0; m<num_maps; m++)); do
    map_seed=$((seed_prefix + m))
    sbatch --time="$chain_time" --mem-per-cpu="$chain_mem" \
        get_sampler_effective_noise_1_chain.sh "$m" "$map_seed" "$nside" "$theta_pix" \
        "$noise_level" "$l_knee" "$num_sweeps" "$out_dir"
done
