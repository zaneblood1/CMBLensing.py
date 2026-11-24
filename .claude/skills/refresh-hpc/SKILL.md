---
name: refresh-hpc
description: Delete the CMBLensing.py install on one of the user's HPC clusters (resnick or aces) and reinstall the current version from GitHub, keeping the cluster's hpc_local.sh and the CAMB grid. Use when the user asks to refresh, update or reinstall the repository on a cluster.
argument-hint: "<resnick|aces>"
disable-model-invocation: true
---

# Refresh the CMBLensing.py install on an HPC

Replace the checkout on the cluster named in the arguments with a fresh clone of `origin/master`, and reinstall it into the cluster's environment.

Arguments: `$ARGUMENTS`

The first word is the cluster. It must be a row of [hosts.md](hosts.md) (currently `resnick` or `aces`); if it is missing or anything else, ask which one and stop. Read `hosts.md` now: it gives the ssh alias, the repo directory on the cluster (and how to find and record it when it is not recorded yet) and how to run remote commands. Below, `<alias>` and `<repo>` are those two values and `<share>` is its connection-sharing options. Every ssh call waits for the user's two-factor approval, as that file describes.

Invoking this skill is the user's go-ahead to delete `<repo>` and everything git tracks in it. It is NOT a go-ahead to delete results: step 2 decides what else is in there.

## 1. What will be installed (local)

```bash
git fetch origin
git status -sb
git rev-parse origin/master
```

The cluster gets `origin/master` from GitHub, not the local working tree. If there are local commits not pushed or uncommitted changes to tracked files, tell the user in one line that the cluster will not have them, and carry on (ask first only if they said the point of the refresh is a change that turns out not to be pushed). Keep the `origin/master` hash for step 4.

## 2. Look at the current install (one ssh call, read only)

```bash
ssh <share> <alias> 'bash -l -s' <<'EOF'
repo=<repo>
ls -d "$repo" && grep -m1 '^name' "$repo/pyproject.toml"
cd "$repo" || exit 0
git log --oneline -1
git status --short
#everything git does not track, with sizes: the things a delete would lose for good
git status --short --ignored | grep -E '^(\?\?|!!)' | grep -vE '__pycache__|\.egg-info|\.pytest_cache' | cut -c4- | xargs -r du -sh
#jobs and drivers that live in this checkout
squeue -u "$USER" -h -o '%i %T %j %Z' | grep -F "$repo"
pgrep -u "$USER" -af 'bash .*\.sh' | grep -v pgrep
EOF
```

- **No checkout there** (first install): skip to step 3's clone. There is no `hpc_local.sh` to keep, so after the clone tell the user to create it (`cp hpc_local.example.sh hpc_local.sh` and fill it in, see `docs/getting-started/installation.md`); the `pip install` needs its environment, so stop there and say what is left.
- **`pyproject.toml` is not `cmb_lensing`'s**: the directory is not this repository. Stop and ask.
- **Queued or running jobs whose work directory is inside `<repo>`, or a driver still submitting**: stop. Job scripts read their `.py` and `hpc_local.sh` through `$SLURM_SUBMIT_DIR` and write their output inside the checkout, so deleting it breaks them. Report the jobs and let the user decide (wait, or cancel them themselves).
- **Untracked or ignored content.** Two things are always carried over to the new checkout: `hpc_local.sh` and `camb_splines/` (the ~8 GB grid). Anything else (`lcdm_sampler_data/`, `*_output*/`, `performance_results*/`, `multi_param_CAMB_grid/`, logs, edited tracked files) is the user's data: list it with its sizes and ask what to do with each (carry over to the same place, move somewhere outside the checkout, or delete). Do not delete any of it without an answer in this conversation. `slurm-*.out`, `submit*.log`, `__pycache__` and `*.egg-info` need no question.

## 3. Reinstall (one ssh call)

Fill in `keep` with `hpc_local.sh`, `camb_splines` and whatever step 2's answers said to carry over (paths relative to the repo root).

```bash
ssh <share> <alias> 'bash -l -s' <<'EOF'
set -e
repo=<repo>
stash="${repo}.keep_$(date +%Y%m%d_%H%M%S)"
keep="hpc_local.sh camb_splines"

#set the kept paths aside, next to the checkout (same file system, so a rename, not a copy)
mkdir "$stash"
for path in $keep; do
    if [ -e "$repo/$path" ]; then
        mkdir -p "$stash/$(dirname "$path")"
        mv "$repo/$path" "$stash/$path"
    fi
done

#the delete, guarded: only a directory named CMBLensing.py that holds this package
[ "$(basename "$repo")" = "CMBLensing.py" ] && grep -q 'cmb_lensing' "$repo/pyproject.toml"
rm -rf "$repo"

git clone https://github.com/zaneblood1/CMBLensing.py.git "$repo"

#put the kept paths back and drop the (now empty) stash
for path in $keep; do
    if [ -e "$stash/$path" ]; then
        mkdir -p "$repo/$(dirname "$path")"
        mv "$stash/$path" "$repo/$path"
    fi
done
find "$stash" -type d -empty -delete

cd "$repo"
git rev-parse HEAD
source hpc_local.sh
pip install -e .
python -c "import cmb_lensing, os; print(os.path.dirname(cmb_lensing.__file__))"
ls -la camb_splines/camb_grid_spline.npz 2>/dev/null || echo "NO CAMB GRID"
EOF
```

If the call fails partway, the kept paths are in `<repo>.keep_<timestamp>` on the cluster: say so, with the full path, and put them back before anything else. Never rerun the block blindly after a failure; look at where it stopped first.

## 4. Check and report

- `git rev-parse HEAD` on the cluster equals the `origin/master` hash of step 1.
- The `import cmb_lensing` line printed a path inside `<repo>`.
- `NO CAMB GRID`: `sample_lcdm` chains need it. Offer to fetch it on the login node (`cmb-lensing-download-splines`, ~8 GB, run under `nohup`); do not start it unasked.

Report in a few lines: the cluster, the commit now installed (hash and subject), what was carried over, what was deleted or moved on the user's instruction, and anything left to do (missing `hpc_local.sh`, missing grid, unpushed local work).
