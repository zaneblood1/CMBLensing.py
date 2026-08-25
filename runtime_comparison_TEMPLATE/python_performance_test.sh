#!/bin/bash

#SBATCH --time=48:00:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G   #memory per CPU core
#SBATCH --mail-user=<USER>@<INSTITUTION>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for JAX and Julia code

#NOTE this is only a template. You must replace the file path place holders with the actual
#location of the code on your own institution's HPC...

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv
#Prevent Julia precompilation lock contention across concurrent SLURM jobs
export JULIA_PKG_PRECOMPILE_AUTO=0
#call the python performance timing script for a specific (f, phi) combo
#and a specific trial number, map_size, and polarity key
python3 ABSOLUTE_PATH_TO/cmb_lensing/runtime_comparison/python_performance_test.py --map_size $1 --seed $2 --trial $3 --pol $4 --theta_pix $5