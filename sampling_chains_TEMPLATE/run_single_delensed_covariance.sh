#!/bin/bash

#SBATCH --time=02:00:00 #1 + 2k stencil points, each a load_sim + map_joint + one inverse
#lensing solve. get_single_delensed_spectra MEASURED ~25s per point at nside 64 / 5' and
#map_joint scales roughly with the pixel count, so nside 128 with k = 3 (7 points) is on the
#order of 10-20 minutes. The file is checkpointed after every point, so a job that does hit
#the wall clock keeps its finished points (the merge sets it aside as unfinished)
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=2G #memory per CPU core; raise for nside 256 and above
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for the JAX code

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the ABSOLUTE_PATH_TO place holders with the actual paths on your HPC

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv

#call the python script for a single realization; every argument from the 11th on is a
#parameter to difference
python3 ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/run_single_delensed_covariance.py \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --map_joint_steps "$7" \
    --step_sigma "$8" \
    --reconstruction "$9" \
    --out_dir "${10}" \
    --params "${@:11}"
