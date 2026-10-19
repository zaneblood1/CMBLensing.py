#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G #memory per CPU core; raise for nside 256 and above
#SBATCH --mail-type=FAIL #mail updates on failure only (to SBATCH_MAIL_USER, see hpc_local.sh)
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for the JAX code

#conda and the other machine-specific setup; SLURM_SUBMIT_DIR is this folder, since the driver submits from it
source "$SLURM_SUBMIT_DIR/../hpc_local.sh" || exit 1

#call the python script for a single realization; 
python3 "$SLURM_SUBMIT_DIR/run_single_delensed_field_covariance.py" \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --step_sigma "$7" \
    --qe_filter "$8" \
    --qe_response "$9" \
    --phi_noise "${10}" \
    --out_dir "${11}" \
    --params "${@:12}"
