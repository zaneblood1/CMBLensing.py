# HPC hosts

Shared by `/refresh-hpc` and `/run-pipeline`. One row per cluster the user can reach. To add a cluster, add a row (and name it in the two skills' `argument-hint`).

| hpc | ssh alias (in `~/.ssh/config`) | repo directory on the cluster |
|---|---|---|
| resnick | `hpc` | not recorded yet |
| aces | `login.aces.hprc.tamu.edu` | not recorded yet |

Notes

- **Repo directory.** Write it with the cluster's own variables (`$HOME/CMBLensing.py`, `$SCRATCH/CMBLensing.py`), never an absolute path holding a user name: this file is tracked. When a row says "not recorded yet", find the checkout with

  ```bash
  ssh <share> <alias> 'bash -lc "find \$HOME \$SCRATCH -maxdepth 3 -type d -name CMBLensing.py 2>/dev/null"'
  ```

  One hit: record it in the table and carry on. No hit or several: ask the user where the install is (or should go), then record the answer.
- **Login.** Assume EVERY cluster asks for two-factor authentication (a push the user approves on their phone) on every new connection. So:
  - Before each ssh call, tell the user in one line that a push is coming (which cluster, what the call does), then make the call and simply WAIT for them to approve it: run it with the Bash tool's longest timeout (`timeout: 600000`) and never with `-o BatchMode=yes`, which would fail instead of waiting. Do not ask them to log in by hand first.
  - Make every ssh call with the connection-sharing options below, so one approval is reused by the later calls of the same skill run (10 minutes after the last use) instead of a push per call:

    ```bash
    mkdir -p ~/.ssh/sockets
    ssh -o ControlMaster=auto -o ControlPath=~/.ssh/sockets/%r@%h:%p -o ControlPersist=10m <alias> ...
    ```

    The skills write this as `ssh <share> <alias>`.
  - Still bundle the remote commands of one step into ONE ssh call.
  - A call that times out or is denied was most likely a missed push: say so and make the same call again once, unless it was the call that deletes a checkout or launches a driver, which is first checked on the cluster (see the skill) before any retry.
- **Login shell.** Run remote commands through `bash -l` (`ssh <share> <alias> 'bash -l -s' <<'EOF' ... EOF`) so `sbatch`, `squeue`, `module` and `$SCRATCH` are defined.
