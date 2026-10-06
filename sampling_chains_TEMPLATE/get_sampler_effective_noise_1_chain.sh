#!/bin/bash

#SBATCH --time=12:00:00 #the driver overrides this on the sbatch command line (chain_time).
#The job flushes its samples every 10 sweeps, so one killed at the wall clock is still
#analysed with the sweeps it finished
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=4G #MEASURED 1.35 GB peak RSS at nside 64; the driver overrides this (chain_mem)
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for the JAX code

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the ABSOLUTE_PATH_TO place holders with the actual paths on your HPC

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv

#call the python script for a single chain (one data map)
python3 ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/get_sampler_effective_noise_1_chain.py \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --num_sweeps "$7" \
    --out_dir "$8"
