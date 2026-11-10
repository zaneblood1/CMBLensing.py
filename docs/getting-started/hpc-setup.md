# HPC Setup

All cluster scripts are Slurm scripts. Before running any of them, create `hpc_local.sh` at the repository root as described in [Installation](installation.md#hpc-setup-hpc_localsh).

## How the drivers work

Each pipeline folder has a **driver** that only submits jobs, and is run with `bash`, not `sbatch`:

```bash
bash get_lcdm_samples/sample_lcdm.sh
```

A driver can be run from any directory. It `cd`s into its own folder, sources the repo-root `hpc_local.sh` and `hpc_submit.sh` (`../`, or `../../` from inside `measure_phi_noise/`) and submits its job script once per job. Job scripts find their Python file and `hpc_local.sh` through `$SLURM_SUBMIT_DIR`, and write their output inside the driver's folder, so the checkout can live anywhere.

## Submission throttle

Some clusters (ACES among them) cancel every submission beyond a cap on queued jobs instead of leaving it pending. Every driver therefore submits through `submit_throttled` (repo-root `hpc_submit.sh`): before each `sbatch` it waits until you have fewer than `MAX_QUEUED_JOBS` jobs queued or running (default 85), checking every `SUBMIT_POLL_SECONDS` (default 60), and it retries a rejected `sbatch`. Override either value in `hpc_local.sh`.

A driver now runs until its last job is submitted, which can take hours. Keep it alive after you log out:

```bash
nohup bash get_lcdm_samples/sample_lcdm.sh > submit.log 2>&1 &
tail -f submit.log
```

`nohup` ignores the hangup signal sent at logout, `> submit.log 2>&1` sends output and errors to the file, and `&` runs it in the background. Alternatively run the driver inside `tmux`, detach with `Ctrl-b d`, and reattach later with `tmux attach` (on the same login node).

The physical configuration (`nside`, `theta_pix`, `noise_level`, `l_knee`, ...) is set by editing variables at the top of each driver. Keep them **the same across drivers** for one experiment, so the chains, the phi-noise measurements and the forecasts all describe the same box.

| Driver | Output folder |
|--------|--------|
| `camb_grid_generation/camb_grid.sh` | `camb_grid_generation/multi_param_CAMB_grid/` |
| `get_lcdm_samples/sample_lcdm.sh` | `get_lcdm_samples/lcdm_sampler_data/<experiment>/` |
| `measure_phi_noise/get_map_joint_phi_noise/get_map_joint_phi_noise.sh` | `measure_phi_noise/get_map_joint_phi_noise/map_joint_phi_noise_output/` |
| `measure_phi_noise/get_sampler_phi_noise/get_sampler_phi_noise.sh` | `measure_phi_noise/get_sampler_phi_noise/sampler_phi_noise_output/` |
| `get_delensed_field_covariance/get_delensed_field_covariance.sh` | `get_delensed_field_covariance/delensed_covariance_output/` |
| `runtime_comparison/run_performance_test.sh` | `runtime_comparison/performance_results/` |

Every driver's settings are defaults that can be overridden on the command line as `--<name>=<value>` (quote a list, e.g. `--map_sizes="128 256"`); the folder name is `--out_dir_name`. Each run writes the submission date and every setting to `submission_settings.txt` in its output folder, replacing any earlier copy.
