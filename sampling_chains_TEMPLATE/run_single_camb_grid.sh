#!/bin/bash

#SBATCH --time=00:30:00 #81 CAMB calls at ~3.6 s each is ~5 min; raise to 04:00:00 if accuracy_boost=2
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=4G   #memory per CPU core
#SBATCH --mail-user=<USER>@<INSTITUTION>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only
#SBATCH --cpus-per-task=1 #parallelism is across jobs, so keep CAMB single threaded

#NOTE: This submission script is only a template. You must at the replace the 
#ABSOLUTE_PATH_TO place holders with the actual file path on your HPC

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv
#one thread per job so 875 concurrent jobs do not each spawn a full OpenMP pool
export OMP_NUM_THREADS=1
#call the python grid script
python3 ABSOLUTE_PATH_TO/sampling_chains/run_single_camb_grid.py \
    --log_a_index "$1" \
    --log_a "$2" \
    --ns_index "$3" \
    --ns "$4" \
    --ombh2_index "$5" \
    --ombh2 "$6" \
    --omch2_index "$7" \
    --omch2 "$8" \
    --h0_min "$9" \
    --h0_max "${10}" \
    --h0_nodes "${11}" \
    --lmax "${12}" \
    --tau "${13}" \
    --mnu "${14}" \
    --k_pivot "${15}" \
    --alens "${16}" \
    --accuracy_boost "${17}" \
    --l_sample_boost "${18}" \
    --l_accuracy_boost "${19}" \
    --out_dir "${20}"
