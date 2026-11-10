# LCDM Sampling on the HPC


```bash
bash get_lcdm_samples/sample_lcdm.sh
```

submits `run_single_lcdm_chain.sh` once per (data map, chain). Specify `num_maps` data realizations and `num_chains` chains per map in the master slurm submission script `sample_lcdm.sh`. Averaging over maps removes the cosmic variance of any one map, and running several chains per map checks convergence.

## What to set in `sample_lcdm.sh`

| Variables | Meaning |
|---|---|
| `nside`, `theta_pix`, `noise_level`, `pol` | basic configurations like pixel resolution, map size, etc, |
| `*_ground` | the ground-truth cosmology the data are simulated at |
| `*_min`, `*_max` | the sampling range for each parameter (by default ±5 sigma) |
| `num_maps`, `num_chains`, `map_prefix` | how many realizations and chains; map seeds are `map_prefix * map` |
| `sample_*` | `true` / `false`: whether each parameter is sampled (default omch2, theta_MC_100, logA). A sampled parameter starts each chain at a uniform random point in its `[*_min, *_max]` box, an unsampled one is held at `*_ground` |
| `*_proposal_sigma` | the Metropolis proposal width of each parameter (`ombh2`, `omch2`, `theta_MC_100`, `log_a`, `ns`) |
| `num_burn_in_fix_theta`, `iters_per_chain`, `num_burn_in_always_accept`, `hmc_num_steps`, `hmc_step_size` | sampler settings |
| `out_dir_name` | the run's folder inside `lcdm_sampler_data/` (default `joint_inference_<submission date>`) |

Every variable has a default in the script and can be overridden on the command line as `--<name>=<value>`, e.g.

```bash
bash get_lcdm_samples/sample_lcdm.sh --noise_level=10 --omch2_proposal_sigma=2e-3 --out_dir_name=run_10uk
```

A misspelled name is ignored and the default used. Tune the proposal widths first with a [pilot chain](../concepts/sample-joint.md#tuning).

## Output

Chains are written as `*_history.txt` files (one line per iteration, every parameter) under `get_lcdm_samples/lcdm_sampler_data/<out_dir_name>/`. The same folder holds `submission_settings.txt`: the submission date, every setting, and each chain's random starting point (rewritten on every run, so use a new `out_dir_name` when you change settings). Copy that folder to your machine and run [the chain analysis](chain-analysis.md) script.
