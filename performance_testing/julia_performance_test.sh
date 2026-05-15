#!/bin/bash

#SBATCH --time=48:00:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G   #memory per CPU core
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for JAX and Julia code

#activate your own specific conda
source /resnick/groups/wugroup/zblood/miniconda3/etc/profile.d/conda.sh
conda activate myenv
#IMPORTANT!! Set the following environment variable to 0 such that calling Pkg.activate("/path/to/Project.toml")
#will only give julia instructions on how to execute code and will NOT cause concurrent writes to ~/.julia/compiled
#which will lead to lock contention...
export JULIA_PKG_PRECOMPILE_AUTO=0

#call the julia performance timing script for a specific (f, phi) combo
#and a specific trial number, map_size, and polarity key
julia /resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/julia_performance_test.jl --map_size $1 --seed $2 --trial $3 --pol $4 --theta_pix $5