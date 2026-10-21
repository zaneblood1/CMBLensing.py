#!/bin/bash

#SBATCH --time=48:00:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G   #memory per CPU core
#SBATCH --mail-type=FAIL #mail updates on failure only (to SBATCH_MAIL_USER, see hpc_local.sh)
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for JAX and Julia code

#conda, CMBLENSING_JL_PATH and the other machine-specific setup; SLURM_SUBMIT_DIR is this
#folder, since the driver submits from it
source "$SLURM_SUBMIT_DIR/../hpc_local.sh" || exit 1
#Prevent Julia precompilation lock contention across concurrent SLURM jobs
export JULIA_PKG_PRECOMPILE_AUTO=0
#call the python performance timing script for a specific (f, phi) combo
#and a specific trial number, map_size, and polarity key
python3 "$SLURM_SUBMIT_DIR/python_performance_test.py" \
    --map_size $1 \
    --seed $2 \
    --trial $3 \
    --pol $4 \
    --theta_pix $5