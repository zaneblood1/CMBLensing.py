#!/bin/bash

#SBATCH --time=7-00:00:00 #run job for max 7 days
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G   #memory per CPU core
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for JAX and Julia code

#activate your own specific conda
source /resnick/groups/wugroup/zblood/miniconda3/etc/profile.d/conda.sh
conda activate myenv
#call the python sampling script
python3 /resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/run_single_chain.py \
    --chain "$1" \
    --a_phi_init "$2" \
    --master_seed "$3" \
    --nside "$4" \
    --theta_pix "$5" \
    --noise_level "$6" \
    --a_phi_ground "$7" \
    --pol "$8" \
    --num_burn_in_fix_theta "$9" \
    --iters_per_chain "${10}" \
    --num_burn_in_always_accept "${11}"