#Machine-specific setup for the HPC scripts. Copy this file to hpc_local.sh at the repo root
#(hpc_local.sh is gitignored) and fill in your own values:
#    cp hpc_local.example.sh hpc_local.sh
#
#Every driver (bash get_*.sh, bash camb_grid.sh, ...) and every job script it submits
#sources hpc_local.sh, so nothing personal lives in the tracked scripts.

#the conda install and environment the jobs run in
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate myenv

#where Slurm mails job failures. sbatch reads SBATCH_MAIL_USER from the environment, so every
#job a driver submits picks it up. A job submitted by hand (sbatch merge_*.sh) only sees it if
#it is in your login shell too: put this line in ~/.bashrc as well, or source this file first
export SBATCH_MAIL_USER=you@example.edu

#the CMBLensing.jl checkout, used only by runtime_comparison/
export CMBLENSING_JL_PATH=/path/to/CMBLensing.jl
