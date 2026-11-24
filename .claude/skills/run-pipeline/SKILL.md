---
name: run-pipeline
description: Submit one of this repository's HPC pipelines (sample_lcdm, sampler_noise, map_noise, delense_covariance, runtime_comparison or camb_grid) on one of the user's clusters (resnick or aces), with any of the driver's settings overridden. Use when the user asks to run, launch or submit a pipeline on a cluster.
argument-hint: "<resnick|aces> <sample_lcdm|sampler_noise|map_noise|delense_covariance|runtime_comparison|camb_grid> [--<setting>=<value> ...]"
disable-model-invocation: true
---

# Run a pipeline on an HPC

Start one pipeline's driver on a cluster. The driver submits the Slurm jobs; this skill only launches it and reports where the output will land.

Arguments: `$ARGUMENTS`

1. **The cluster**: a row of [../refresh-hpc/hosts.md](../refresh-hpc/hosts.md) (currently `resnick` or `aces`). Read that file now: it gives the ssh alias, the repo directory on the cluster (and how to find and record it when it is not recorded yet) and how to run remote commands. Below, `<alias>` and `<repo>` are those two values and `<share>` is its connection-sharing options. Every ssh call waits for the user's two-factor approval, as that file describes.
2. **The pipeline**: one of the table below.
3. **The settings**: everything after it.

If the cluster or the pipeline is missing or not in the lists, ask and stop.

## Pipelines

| pipeline | driver (relative to the repo root) | jobs it submits | output folder |
|---|---|---|---|
| `sample_lcdm` | `get_lcdm_samples/sample_lcdm.sh` | `num_maps` x `num_chains` | `get_lcdm_samples/lcdm_sampler_data/<out_dir_name>/` |
| `sampler_noise` | `measure_phi_noise/get_sampler_phi_noise/get_sampler_phi_noise.sh` | `num_maps` x `num_chains` | `measure_phi_noise/get_sampler_phi_noise/<out_dir_name>/` |
| `map_noise` | `measure_phi_noise/get_map_joint_phi_noise/get_map_joint_phi_noise.sh` | `num_realizations` | `measure_phi_noise/get_map_joint_phi_noise/<out_dir_name>/` |
| `delense_covariance` | `get_delensed_field_covariance/get_delensed_field_covariance.sh` | `num_realizations` | `get_delensed_field_covariance/<out_dir_name>/` |
| `runtime_comparison` | `runtime_comparison/run_performance_test.sh` | per (`pol_keys`, `map_sizes`) | `runtime_comparison/<out_dir_name>/` |
| `camb_grid` | `camb_grid_generation/camb_grid.sh` | `log_a_nodes` x `ns_nodes` x `ombh2_nodes` x `omch2_nodes` (875 by default), one H0 sweep each | `camb_grid_generation/<out_dir_name>/` |

Every driver writes its audit trail, `<output folder>/submission_settings.txt`, before it submits its first job: the submission date, user and host, then every setting as `name=value` (`sample_lcdm` also appends each chain's starting point as it is submitted).

## 1. Settings

A driver's settings are exactly its `name=${name:-default}` lines, each overridable as `--<name>=<value>`. Read them from the driver in the local checkout; never list them from memory, they change:

```bash
grep -nE '^[A-Za-z_0-9]+=\$\{' <driver>
```

Turn what the user wrote into `--<name>=<value>` arguments:

- Accept loose forms (`noise=10`, `--noise 10`, "at 10 uK with 20 maps") and map each to the driver's own name (`noise` → `noise_level`, `maps` → `num_maps` or `num_realizations`, `logA` settings → `log_a_*`, ...).
- A name that matches no setting of this driver (for example `lmax`, which only `camb_grid` has) must not be passed: the driver would `declare` it and silently ignore it. Tell the user it is not a setting of this pipeline, show the list, and ask.
- A setting the user did not give keeps the driver's default. Do not pass defaults explicitly.
- A value with spaces is one argument: `--derivative_params="omch2 logA"`, `--map_sizes="128 256"`, `--chain_seeds="11 22"`.
- Paths (`phi_noise` of `delense_covariance`) are paths ON THE CLUSTER.
- If no `out_dir_name` was given and the run is not at the driver's defaults, pick one that says what differs, in the style of the existing data (`joint_inference_<YYYY_MM_DD>_10uk`, `sampler_phi_noise_output_10uk`, `delensed_covariance_output_qe_filter_lensed_10uk`), and say so. `sample_lcdm`'s default already carries the submission date. For `camb_grid`, a grid that differs from the default one always gets its own `out_dir_name` (e.g. `multi_param_CAMB_grid_lmax_5000`), and `lmax` other than `constants.DEFAULT_MAX_ELL` gives a grid the sampler cannot use as it is: say so before submitting.

Ask only when a setting is ambiguous or unknown. Otherwise go on.

## 2. Pre-flight on the cluster (one ssh call, read only)

```bash
git fetch origin && git rev-parse origin/master      #local, for the comparison below
ssh <share> <alias> 'bash -l -s' <<'EOF'
cd <repo> || exit 1
git rev-parse HEAD; git status --short
ls hpc_local.sh
grep -nE '^[A-Za-z_0-9]+=\$\{' <driver>
ls -d <output folder> 2>/dev/null && ls <output folder> | head
squeue -u "$USER" -h -r | wc -l
pgrep -u "$USER" -af 'bash .*\.sh' | grep -v pgrep
#sample_lcdm only
ls -la camb_splines/camb_grid_spline.npz
#runtime_comparison only
grep CMBLENSING_JL_PATH hpc_local.sh
EOF
```

Stop and tell the user, rather than submitting, when:

- there is no checkout or no `hpc_local.sh` (point to `/refresh-hpc <hpc>` and `hpc_local.example.sh`);
- the cluster's `HEAD` is not `origin/master`, or its tracked files are edited: say which commit it is on and ask whether to run `/refresh-hpc <hpc>` first or submit as it is. The settings that count are the ones in the CLUSTER's driver (the `grep` above): if a requested setting is missing there, it cannot be passed;
- the output folder already exists and is not empty: new jobs would write into the old run and `submission_settings.txt` would be overwritten. Ask for another `out_dir_name` or an explicit go-ahead;
- `sample_lcdm` without the CAMB grid (fetch it on the login node with `cmb-lensing-download-splines`), or `runtime_comparison` without `CMBLENSING_JL_PATH`;
- the same driver is already running.

A full queue is not a reason to stop: the driver throttles itself (`hpc_submit.sh`, `MAX_QUEUED_JOBS`) and simply takes longer. Mention the count.

## 3. Submit

The driver blocks until its last job is submitted, which can take hours, so it runs under `nohup` and the ssh call returns at once. The quoted heredoc keeps the arguments' own quoting intact.

```bash
ssh <share> <alias> 'bash -l -s' <<'EOF'
cd <repo>/<driver folder> || exit 1
nohup bash <driver file> <--name=value ...> > submit_<out_dir_name>.log 2>&1 < /dev/null &
echo "driver pid $!"
#the audit trail is written before the first sbatch; give the driver time to reach it
for attempt in $(seq 30); do [ -s <output folder>/submission_settings.txt ] && break; sleep 2; done
sleep 20
echo "=== submit log"; cat submit_<out_dir_name>.log
echo "=== audit trail: $(realpath <output folder>/submission_settings.txt)"
cat <output folder>/submission_settings.txt
echo "=== queued or running jobs: $(squeue -u "$USER" -h -r | wc -l)"
EOF
```

Submit ONCE. If the ssh call errors or times out, check the cluster (`pgrep`, the log, `squeue`) before any retry: the driver may be running, and a second launch duplicates every job.

No audit trail after that wait means the driver died before submitting anything (usually `hpc_local.sh`): report the log instead. Otherwise check that the log shows `Submitted batch job` lines and no error (an `sbatch failed; retrying` line means the queue cap was hit and the driver is waiting, which is fine).

## 4. Report, from the audit trail

The report is what the cluster's `submission_settings.txt` says, read in step 3's call, not what this skill meant to pass: that file is the record of what the driver actually ran with.

- **Header**: the audit trail's first line (submitted when, by whom, on which host), the pipeline, the commit of the cluster's checkout and the exact command that was launched.
- **Settings**: every `name=value` line of the audit trail, as a table, with the ones that differ from the driver's defaults marked. If a value in it is not the one the user asked for, say so first, above everything else: the override did not take.
- **Outputs**: the output folder (the audit trail's `out_dir_name`, as the full path on the cluster), the audit trail's own full path, the submit log and the driver's pid; the number of jobs the settings imply and how many are queued so far. For `sample_lcdm`, the chain starting points listed so far (the driver is still appending them while it submits).
- **Following it**: `squeue -u $USER` and `tail` of the submit log on the cluster.
- **Next step** once the jobs finish, from `CLAUDE.md`'s Commands section: `python -m cmb_lensing.chain_analysis <chain_dir> ...` (`sample_lcdm`), `sbatch merge_sampler_phi_noise.sh --burn_in=... --out_dir_name=...` (`sampler_noise`), `merge_map_joint_phi_noise.py --noise_dir ...` (`map_noise`), `merge_delensed_field_covariance.py --covariance_dir ...` (`delense_covariance`), `performance_analysis.py` (`runtime_comparison`), `python merge_camb_grid.py --grid_dir <out_dir_name>` then `python tests/test_camb_grid.py --grid <...>/camb_grid_spline.npz -n 20` (`camb_grid`).

Do not wait for the jobs, poll the queue or download results unless asked.
