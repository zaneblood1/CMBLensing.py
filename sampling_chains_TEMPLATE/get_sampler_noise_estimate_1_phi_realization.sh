#!/bin/bash

#SBATCH --time=00:30:00 #MEASURED ~1 s per draw at nside 128 / 2.5' plus ~20 s start-up, so
#50 draws is a few minutes; a wide margin for a slower node. The job checkpoints every 5
#draws, so one killed at the wall clock still merges with the draws it finished
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=2G #MEASURED 1.2 GB peak RSS at nside 128; raise for nside 256 and above
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for the JAX code

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the ABSOLUTE_PATH_TO place holders with the actual paths on your HPC

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv

#call the python script for a single phi realization (calibration = 1: phi = 0 instead)
python3 ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/get_sampler_noise_estimate_1_phi_realization.py \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --num_draws "$7" \
    --tol "$8" \
    --calibration "$9" \
    --out_dir "${10}"
