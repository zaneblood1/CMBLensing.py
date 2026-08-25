#!/bin/bash

#SBATCH --time=02:00:00 #run job for max X amount of time...
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G   #memory per CPU core
#SBATCH --mail-user=<USER>@<INSTITUTION>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for JAX and Julia code

#NOTE: This submission script is only a template. You must at the replace the 
#ABSOLUTE_PATH_TO place holders with the actual file path on your HPC

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv
#call the python sampling script
export JULIA_PKG_PRECOMPILE_AUTO=0
python3 ABSOLUTE_PATH_TO/sampling_chains/run_single_lcdm_chain.py \
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