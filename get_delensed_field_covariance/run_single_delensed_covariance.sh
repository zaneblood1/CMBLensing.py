#!/bin/bash

#SBATCH --time=00:10:00 
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=16G #memory per CPU core; raise for nside 256 and above
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for the JAX code

#activate your own specific conda
source /resnick/groups/wugroup/zblood/miniconda3/etc/profile.d/conda.sh
conda activate myenv

#call the python script for a single realization; every argument from the 16th on is a
#parameter to difference
python3 /resnick/groups/wugroup/zblood/cmb_lensing/get_delensed_field_covariance/run_single_delensed_covariance.py \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --map_joint_steps "$7" \
    --step_sigma "$8" \
    --reconstruction "$9" \
    --constant_nphi "${10}" \
    --qe_filter "${11}" \
    --qe_response "${12}" \
    --phi_noise "${13}" \
    --phi_noise_key "${14}" \
    --out_dir "${15}" \
    --params "${@:16}"
