#!/bin/bash

#SBATCH --time=00:20:00 #MEASURED ~25s for one realization at nside 64 / 5' (load_sim +
#map_joint + three lensing solves + the band averaging), and map_joint scales roughly with
#the pixel count, so nside 128 is a few minutes. This is a wide margin for a slower node.
#Keep it short - 100 short jobs backfill far better than 100 long ones
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

#the cosmology shift is optional and is empty for the production run. It must be omitted
#entirely rather than passed as "", since --shift_param is checked against PARAM_ORDER.
#${14} says whether ${13} is an absolute displacement ("value") or one in PARAM_SIGMA units
#("sigma"), and ${15} which cosmology the reconstruction is built at ("fiducial"/"shifted")
shift_args=()
if [ -n "${12}" ]; then
    shift_args=(--shift_param "${12}" "--shift_${14:-value}" "${13}"
                --reconstruction "${15:-fiducial}")
fi

#call the python script for a single realization
python3 ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/get_single_delensed_spectra.py \
    --realization_index "$1" \
    --map_seed "$2" \
    --nside "$3" \
    --theta_pix "$4" \
    --noise_level "$5" \
    --l_knee "$6" \
    --delta_ell "$7" \
    --map_joint_steps "$8" \
    --nphi_source "$9" \
    --qe_response "${10}" \
    --out_dir "${11}" \
    "${shift_args[@]}"
