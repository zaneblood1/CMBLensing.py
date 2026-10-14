#!/bin/bash

#SBATCH --time=04:00:00 #ESTIMATED (not measured) 20-40 minutes per burn-in at nside 128 /
#6000 sweeps / 100 maps; every burn_in_scan value is one more full pass over the chains
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mem-per-cpu=4G #ESTIMATED ~3 GB peak (one chain's samples + the IAT FFTs)
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=END,FAIL #mail updates when the analysis ends or fails
#SBATCH --cpus-per-task=4 #This is the flag that actually increases CPUs for numpy / CAMB
#SBATCH --job-name=merge_sampler_effective_noise

#NOTE: This submission script is only a template. You must at the very least replace the
#email address and the ABSOLUTE_PATH_TO place holders with the actual paths on your HPC

#Runs merge_sampler_effective_noise.py on a compute node. Submit from the folder holding
#the python script, once the chains exist (they may still be running):
#    sbatch merge_sampler_effective_noise.sh
#The report is printed to slurm-<jobid>.out as it goes; every burn-in's npz and png files
#are written into chain_dir as soon as that burn-in completes.

#the out_dir get_sampler_effective_noise.sh wrote its chains to
chain_dir="ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/sampler_effective_noise_output"

#sweeps cut from the start of every chain: writes sampler_effective_noise.npz (+ plots)
burn_in=200

#further burn-ins (space separated, may be empty), analysed one after the other AFTER
#burn_in and written as sampler_effective_noise_burn_in_<b>.npz (+ plots)
burn_in_scan="100 400 800 1600"

#width of the |L| bands in the report and the spectra plot
delta_ell=100

#activate your own specific conda
source ABSOLUTE_PATH_TO/miniconda3/etc/profile.d/conda.sh
conda activate myenv

#-u: unbuffered, so the report reaches the slurm log while the job is still running
python3 -u ABSOLUTE_PATH_TO/cmb_lensing/sampling_chains/merge_sampler_effective_noise.py \
    --chain_dir "$chain_dir" \
    --burn_in "$burn_in" \
    --burn_in_scan $burn_in_scan \
    --delta_ell "$delta_ell"
