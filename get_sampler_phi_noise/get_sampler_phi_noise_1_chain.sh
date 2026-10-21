#!/bin/bash

#SBATCH --time=12:00:00 #the driver overrides this on the sbatch command line (chain_time).
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G 
#SBATCH --mail-type=FAIL #mail updates on failure only (to SBATCH_MAIL_USER, see hpc_local.sh)
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for the JAX code

#conda and the other machine-specific setup; SLURM_SUBMIT_DIR is this folder, since the driver
#submits from it
source "$SLURM_SUBMIT_DIR/../hpc_local.sh" || exit 1

#call the python script for a single chain (one data map)
python3 "$SLURM_SUBMIT_DIR/get_sampler_phi_noise_1_chain.py" \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --num_sweeps "$7" \
    --out_dir "$8" \
    --hmc_step_size "$9" \
    --hmc_num_steps "${10}"
