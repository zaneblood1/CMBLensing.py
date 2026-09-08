#!/bin/bash

#SBATCH --time=00:20:00 #MEASURED 1m41s for one realization at nside 128 / 2.5' /
#3 params (19 stencil points x 2 lensing solves); this is a ~12x margin for a
#slower node. Keep it short - 100 short jobs backfill far better than 100 long ones
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=2G   #memory per CPU core; MEASURED peak RSS 1.78 GB per job
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for the JAX code

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the ABSOLUTE_PATH_TO place holders with the actual paths on your HPC

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv
#call the python script for a single realization
python3 ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/run_single_mixed_hessian.py \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --out_dir "$7" \
    --params "${@:8}"
