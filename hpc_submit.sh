#Throttled job submission, sourced by every HPC driver (after hpc_local.sh, so hpc_local.sh can
#override the three settings below). Some clusters cancel any submission beyond a
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
#Neither survives a reboot of the login node. Every driver submits through submit_once, so it
#can be rerun with the same arguments after any interruption: it submits only the jobs still
#missing. hpc_local.example.sh has run_driver / kill_driver shortcuts for the nohup line.

#queued + running jobs allowed at once, kept below the cluster's hard cap
MAX_QUEUED_JOBS=${MAX_QUEUED_JOBS:-85}
#seconds between queue checks (and between retries of a rejected sbatch)
SUBMIT_POLL_SECONDS=${SUBMIT_POLL_SECONDS:-60}

#nodes no job may run on, comma separated (e.g. "ac006" or "ac006,ac012"); empty for none. It
#is passed as sbatch --exclude: sbatch has no environment variable for it
EXCLUDE_NODES=${EXCLUDE_NODES:-}

#sbatch with the same arguments, once fewer than MAX_QUEUED_JOBS jobs are queued
submit_throttled() {
    local exclude_option=()
    if [ -n "$EXCLUDE_NODES" ]; then
        exclude_option=(--exclude="$EXCLUDE_NODES")
    fi
    #-r lists every job array task separately, so arrays count task by task
    while [ "$(squeue -u "$USER" -h -r | wc -l)" -ge "$MAX_QUEUED_JOBS" ]; do
        sleep "$SUBMIT_POLL_SECONDS"
    done
    #a rejected submission is retried, so no job is silently dropped
    until sbatch "${exclude_option[@]}" "$@"; do
        echo "sbatch failed; retrying in $SUBMIT_POLL_SECONDS s" >&2
        sleep "$SUBMIT_POLL_SECONDS"
    done
}

#submit_throttled, unless the job was already submitted by an earlier run of the driver: the
#first argument is a marker file, created when the job is handed to sbatch. Rerunning a driver
#that was interrupted then submits only the jobs still missing. Delete a job's marker to have
#it submitted again
submit_once() {
    local marker="$1"
    shift
    if [ -e "$marker" ]; then
        return
    fi
    mkdir -p "$(dirname "$marker")"
    touch "$marker"
    submit_throttled "$@"
}
