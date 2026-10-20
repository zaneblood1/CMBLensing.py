#!/bin/bash

#SBATCH --time=01:00:00 #ESTIMATED (not measured): one read of every chain
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G #ESTIMATED: one chain's post-burn-in samples at a time
#SBATCH --mail-type=END,FAIL #mail updates when the merge ends or fails (to SBATCH_MAIL_USER)
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for numpy
#SBATCH --job-name=merge_sampler_phi_noise

#Runs merge_sampler_phi_noise.py on a compute node. Submit from THIS folder, once the
#chains exist (they may still be running):
#    sbatch merge_sampler_phi_noise.sh
#Mail only reaches you if SBATCH_MAIL_USER is set in the shell you submit from (put it in
#~/.bashrc, or run "source ../hpc_local.sh" first). Writes sampler_phi_noise.npz into chain_dir.

#the out_dir get_sampler_phi_noise.sh wrote its chains to
chain_dir="$SLURM_SUBMIT_DIR/sampler_phi_noise_output"

#sweeps cut from the start of every chain. This should be determined empirically
#by plotting e.g. the logpdf values of a small sample_lcdm.py run and noting at what
#step the values plateau... This indicates the sampler has started polling from the typical set
burn_in=300

#conda and the other machine-specific setup
source "$SLURM_SUBMIT_DIR/../hpc_local.sh" || exit 1

#-u: unbuffered, so the output reaches the slurm log while the job is still running
python3 -u "$SLURM_SUBMIT_DIR/merge_sampler_phi_noise.py" \
    --chain_dir "$chain_dir" \
    --burn_in "$burn_in"
