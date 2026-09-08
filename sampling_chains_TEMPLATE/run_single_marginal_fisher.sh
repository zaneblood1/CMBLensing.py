#!/bin/bash

#SBATCH --time=08:00:00 #one map's field chain; scale with n_draws and nside
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=8G   #memory per CPU core
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=8 #XLA on CPU; the lensing RK4 is the cost, and it does thread

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the ABSOLUTE_PATH_TO place holders with the actual paths on your HPC

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv
#call the python script for a single data realization
python3 ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/run_single_marginal_fisher.py \
    --map_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --n_draws "$7" \
    --burn_in "$8" \
    --out_dir "$9" \
    --params "${@:10}"
