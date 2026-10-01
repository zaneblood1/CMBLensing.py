#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the absolute path on your HPC for the out_dir variable...

#Driver for the SAMPLER NOISE ESTIMATE: the Fisher-information bound on the phi
#reconstruction noise the sampler's posterior can reach. Spawns one job per phi
#realization: each takes load_sim's phi at its seed, draws num_draws fresh (f, n) data sets
#AT THAT phi, and accumulates the per-mode score
#
#    score = grad_phi logpdf(f_WF(d, phi), phi)        (f_WF: the Wiener-filtered field)
#
#whose variance over the draws is the Fisher information F_phi of p(d | phi). The merge
#averages F over the phi realizations and inverts it per mode:
#
#    N_bound(k) = 1 / <F_phi(k)>_phi   <=   N_sampler(k)        (van Trees inequality)
#
#WHY THIS EXISTS: the forecasts' phi block is C_phi + N_phi with N_phi from a quadratic
#estimator, but sample_joint samples the exact posterior, which can beat the QE. This
#measures how much without running the sampler. See cmb_lensing/sampler_noise_estimate.py.
#
#Once every job has finished, scp out_dir to the local machine and run:
#    python merge_sampler_noise_estimate.py --noise_dir <local copy of out_dir> \
#        --map_joint_noise <a merge_delensed_covariance.py or merge_phi_noise.py npz at this box>
#    python -m cmb_lensing.fisher_forecast --nphi_source score \
#        --phi_noise <local copy of out_dir>/sampler_noise_estimate.npz ...

#systematics: MATCH sample_lcdm.sh so the bound describes the box the chains run on
nside=128
theta_pix=2.5
noise_level=5
l_knee=0

#data draws per phi realization. The per-mode noise of N_bound is ~1/sqrt(total draws):
#num_phi_realizations * num_draws = 1000 gives ~3% per mode, and band averages far less.
#MEASURED ~1 s per draw at nside 128 / 2.5' (8 local cores) plus ~20 s of start-up
num_draws=50

#the Wiener filter's CG tolerance. map_joint's 1e-1 puts a 4e-3 rms error on the score;
#1e-8 puts 1e-6 for ~20% more time
tol=1e-8

#one slurm job per phi realization. The phi-to-phi spread of N_bound per band was 1-3% at
#nside 64, so beyond ~20 phi realizations add draws rather than realizations
num_phi_realizations=20
seed_prefix=369258

#1 adds one CALIBRATION job at phi = 0, where the bound must reproduce the quadratic
#estimator's N0 with the unlensed filter and response exactly; the merge reports the ratio
#per band. Strongly recommended for any new box
calibration=1

#output folder shared by every job; scp this to the local machine for the merge
out_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/sampler_noise_estimate_output"

mkdir -p "$out_dir"
#every job uses a distinct seed; load_score_directory refuses duplicates, so a loop change
#that repeats one fails the merge loudly instead of double counting a phi
for ((m=0; m<num_phi_realizations; m++)); do
    map_seed=$((seed_prefix + m))
    sbatch get_sampler_noise_estimate_1_phi_realization.sh "$m" "$map_seed" "$nside" \
        "$theta_pix" "$noise_level" "$l_knee" "$num_draws" "$tol" 0 "$out_dir"
done
if [ "$calibration" -eq 1 ]; then
    sbatch get_sampler_noise_estimate_1_phi_realization.sh 0 "$((seed_prefix - 1))" \
        "$nside" "$theta_pix" "$noise_level" "$l_knee" "$num_draws" "$tol" 1 "$out_dir"
fi
