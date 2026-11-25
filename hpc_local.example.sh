#Machine-specific setup for the HPC scripts. Copy this file to hpc_local.sh at the repo root
#(hpc_local.sh is gitignored) and fill in your own values:
#    cp hpc_local.example.sh hpc_local.sh
#
#Every driver (bash get_*.sh, bash camb_grid.sh, ...) and every job script it submits
#sources hpc_local.sh, so nothing personal lives in the tracked scripts.

#the conda install and environment the jobs run in
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate myenv

#Alternative method to activate the conda environment using modules and explicit PATH setting
# module load Anaconda3/2022.10
# source activate myenv
# export PATH=/scratch/user/<user>/.conda/envs/myenv/bin:$PATH

#optional: nodes no job may run on, comma separated. Every driver passes it to sbatch as
#--exclude (hpc_submit.sh); sbatch reads no environment variable for this. A job submitted by
#hand needs the option itself: sbatch --exclude=ac006 merge_sampler_phi_noise.sh
export EXCLUDE_NODES=ac006

#where Slurm mails job failures. sbatch reads SBATCH_MAIL_USER from the environment, so every
#job a driver submits picks it up. A job submitted by hand (sbatch merge_*.sh) only sees it if
#it is in your login shell too: put this line in ~/.bashrc as well, or source this file first
export SBATCH_MAIL_USER=you@example.edu

#optional: the submission throttle of hpc_submit.sh (defaults 85 jobs, 60 s). Set
#MAX_QUEUED_JOBS a few jobs below the cluster's cap on queued + running jobs
export MAX_QUEUED_JOBS=89
export SUBMIT_POLL_SECONDS=600

#Shortcuts for the login shell. To have them at every login, add to ~/.bashrc:
#    source /path/to/CMBLensing.py/hpc_local.sh
#
#run_driver <driver.sh> [--<name>=<value> ...]: start a driver under nohup, so it keeps
#submitting after you log out. Its output goes to submit_<out_dir_name>.log in the driver's
#folder (submit_<driver name>.log without an --out_dir_name). After a login node reboot, give
#the same command again: the jobs already submitted are skipped (hpc_submit.sh's submit_once)
#    run_driver get_lcdm_samples/sample_lcdm.sh --noise_level=5 --out_dir_name=lcdm_5uk
run_driver() {
    local driver="$1"
    shift
    local run_name
    run_name=$(basename "$driver" .sh)
    local argument
    for argument in "$@"; do
        if [[ "$argument" == --out_dir_name=* ]]; then
            run_name="${argument#--out_dir_name=}"
        fi
    done
    local log
    log="$(dirname "$driver")/submit_$run_name.log"
    nohup bash "$driver" "$@" > "$log" 2>&1 < /dev/null &
    echo "driver started (process $!); follow it with: tail -f $log"
}

#kill_driver <text>: stop every running driver of yours whose command line holds the text (a
#driver's file name stops all its runs, an out_dir_name stops that run). Without an argument
#it only lists the running drivers. The jobs already submitted are NOT cancelled (scancel)
#    kill_driver lcdm_5uk
kill_driver() {
    if [ -z "$1" ]; then
        pgrep -u "$USER" -af "^bash .*\.sh"
        return
    fi
    #list what is about to be stopped, then stop it
    pgrep -u "$USER" -af "^bash .*$1" && pkill -u "$USER" -f "^bash .*$1"
}

#the CMBLensing.jl checkout, used only by runtime_comparison/
export CMBLENSING_JL_PATH=/path/to/CMBLensing.jl
