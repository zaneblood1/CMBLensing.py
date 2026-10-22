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
| `num_burn_in_fix_theta`, `iters_per_chain`, `hmc_num_steps`, `hmc_step_size` | sampler settings |

Which parameters are sampled, and the proposal widths, are set in `run_single_lcdm_chain.py`. Tune them first with a [pilot chain](../concepts/sample-joint.md#tuning).

## Output

Chains are written as `*_history.txt` files (one line per iteration, every parameter) under `get_lcdm_samples/lcdm_sampler_data/<experiment>/`. Copy that folder to your machine and run [the chain analysis](chain-analysis.md) script.
