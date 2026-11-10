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
#~/.bashrc, or run "source ../../hpc_local.sh" first). Writes sampler_phi_noise.npz into chain_dir.
#
#Every setting below is a default; override any of them as --<name>=<value> after the script
#name (sbatch passes those to the script), e.g.
#    sbatch merge_sampler_phi_noise.sh --burn_in=600 --out_dir_name=sampler_phi_noise_output_10uk

#every --name=value argument becomes the variable name=value
for arg in "$@"; do declare "${arg#--}"; done

#name of the folder get_sampler_phi_noise.sh wrote its chains to (its out_dir_name)
out_dir_name=${out_dir_name:-sampler_phi_noise_output}
chain_dir="$SLURM_SUBMIT_DIR/$out_dir_name"

#sweeps cut from the start of every chain. Choose it from a pilot chain (e.g. where the logpdf
#of a small sample_lcdm.py run plateaus), then confirm it with plot_sampler_convergence.py on
#the merged file: the mean R-hat across each map's chains should be ~1 in every L band
burn_in=${burn_in:-300}

#conda and the other machine-specific setup
source "$SLURM_SUBMIT_DIR/../../hpc_local.sh" || exit 1

#-u: unbuffered, so the output reaches the slurm log while the job is still running
python3 -u "$SLURM_SUBMIT_DIR/merge_sampler_phi_noise.py" \
    --chain_dir "$chain_dir" \
    --burn_in "$burn_in" || exit 1

#add this merge to the driver's audit trail (only reached if the merge succeeded); the last
#line describes the current sampler_phi_noise.npz
echo "merged $(date "+%Y-%m-%d %H:%M:%S %Z") by $USER: burn_in=$burn_in" >> "$chain_dir/submission_settings.txt"
