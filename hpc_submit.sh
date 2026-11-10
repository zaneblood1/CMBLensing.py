#Throttled job submission, sourced by every HPC driver (after hpc_local.sh, so hpc_local.sh can
#override the two settings below). Some clusters cancel any submission beyond a
#cap on queued jobs instead of leaving it pending, so submit_throttled waits until the user
#has fewer than MAX_QUEUED_JOBS jobs queued or running before each sbatch.
#
#A driver using it blocks until its last job is submitted, which can take hours. Keep it alive
#after logging out by running it in the background via
#
#    nohup bash <driver>.sh > submit.log 2>&1 &
#
#(nohup ignores the hangup signal sent at logout; output goes to submit.log), or inside a tmux
#session (detach with Ctrl-b d, reattach later with "tmux attach" on the same login node).

#queued + running jobs allowed at once, kept below the cluster's hard cap
MAX_QUEUED_JOBS=${MAX_QUEUED_JOBS:-85}
#seconds between queue checks (and between retries of a rejected sbatch)
SUBMIT_POLL_SECONDS=${SUBMIT_POLL_SECONDS:-60}

#sbatch with the same arguments, once fewer than MAX_QUEUED_JOBS jobs are queued
submit_throttled() {
    #-r lists every job array task separately, so arrays count task by task
    while [ "$(squeue -u "$USER" -h -r | wc -l)" -ge "$MAX_QUEUED_JOBS" ]; do
        sleep "$SUBMIT_POLL_SECONDS"
    done
    #a rejected submission is retried, so no job is silently dropped
    until sbatch "$@"; do
        echo "sbatch failed; retrying in $SUBMIT_POLL_SECONDS s" >&2
        sleep "$SUBMIT_POLL_SECONDS"
    done
}
