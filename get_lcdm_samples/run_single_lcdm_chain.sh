#!/bin/bash

#SBATCH --time=02:00:00 #run job for max X amount of time...
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G   #memory per CPU core
#SBATCH --mail-type=FAIL #mail updates on failure only (to SBATCH_MAIL_USER, see hpc_local.sh)
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for JAX and Julia code

#conda and the other machine-specific setup; SLURM_SUBMIT_DIR is this folder, since the driver
#submits from it
source "$SLURM_SUBMIT_DIR/../hpc_local.sh" || exit 1
#call the python sampling script
export JULIA_PKG_PRECOMPILE_AUTO=0
python3 "$SLURM_SUBMIT_DIR/run_single_lcdm_chain.py" \
    --map_seed "$1" \
    --chain "$2" \
    --ombh2_init "$3" \
    --omch2_init "$4" \
    --theta_MC_100_init "$5" \
    --log_a_init "$6" \
    --ns_init "$7" \
    --nside "$8" \
    --theta_pix "$9" \
    --noise_level "${10}" \
    --ombh2_min "${11}" \
    --ombh2_ground "${12}" \
    --ombh2_max "${13}" \
    --omch2_min "${14}" \
    --omch2_ground "${15}" \
    --omch2_max "${16}" \
    --theta_MC_100_min "${17}" \
    --theta_MC_100_ground "${18}" \
    --theta_MC_100_max "${19}" \
    --log_a_min "${20}" \
    --log_a_ground "${21}" \
    --log_a_max "${22}" \
    --ns_min "${23}" \
    --ns_ground "${24}" \
    --ns_max "${25}" \
    --pol "${26}" \
    --num_burn_in_fix_theta "${27}" \
    --iters_per_chain "${28}" \
    --num_burn_in_always_accept "${29}"