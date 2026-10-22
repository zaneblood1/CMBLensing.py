# HPC Setup

All cluster scripts are Slurm scripts. Before running any of them, create `hpc_local.sh` at the repository root as described in [Installation](installation.md#hpc-setup-hpc_localsh).

## How the drivers work

Each pipeline folder has a **driver** that only submits jobs, and is run with `bash`, not `sbatch`:

```bash
bash get_lcdm_samples/sample_lcdm.sh
```

A driver can be run from any directory. It `cd`s into its own folder, sources `../hpc_local.sh` and `sbatch`es its job script once per job. Job scripts find their Python file and `hpc_local.sh` through `$SLURM_SUBMIT_DIR`, and write their output inside the driver's folder, so the checkout can live anywhere.

The physical configuration (`nside`, `theta_pix`, `noise_level`, `l_knee`, ...) is set by editing variables at the top of each driver. Keep them **the same across drivers** for one experiment, so the chains, the phi-noise measurements and the forecasts all describe the same box.

| Driver | Output folder |
|--------|--------|
| `camb_grid_generation/camb_grid.sh` | `camb_grid_generation/multi_param_CAMB_grid/` |
| `get_lcdm_samples/sample_lcdm.sh` | `get_lcdm_samples/lcdm_sampler_data/<experiment>/` |
| `get_map_joint_phi_noise/get_map_joint_phi_noise.sh` | `get_map_joint_phi_noise/map_joint_phi_noise_output/` |
| `get_sampler_phi_noise/get_sampler_phi_noise.sh` | `get_sampler_phi_noise/sampler_phi_noise_output/` |
| `get_delensed_field_covariance/get_delensed_field_covariance.sh` | `get_delensed_field_covariance/delensed_covariance_output/` |
| `runtime_comparison/run_performance_test.sh` | `runtime_comparison/performance_results/` |
