#!/bin/bash

#SBATCH --time=12:00:00 #the driver overrides this on the sbatch command line (chain_time).
#The job flushes its samples every 10 sweeps, so one killed at the wall clock is still
#analysed with the sweeps it finished
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=4G #MEASURED 1.35 GB peak RSS at nside 64; the driver overrides this (chain_mem)
#SBATCH --mail-type=FAIL #mail updates on failure only (to SBATCH_MAIL_USER, see hpc_local.sh)
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for the JAX code

#conda and the other machine-specific setup; SLURM_SUBMIT_DIR is this folder, since the driver
#submits from it
source "$SLURM_SUBMIT_DIR/../hpc_local.sh" || exit 1

#call the python script for a single chain (one data map)
python3 "$SLURM_SUBMIT_DIR/get_sampler_effective_noise_1_chain.py" \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --num_sweeps "$7" \
    --out_dir "$8"
