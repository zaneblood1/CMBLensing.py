# LCDM Sampling on the HPC

A single chain on a single data map says little on its own: the map carries cosmic variance, and one chain cannot show whether the sampler has converged. This pipeline therefore runs the LCDM sampler on many simulated data maps, with several chains per map. Averaging over maps removes the cosmic variance of any one map, and comparing the chains of one map checks convergence.

By default it simulates 10 data maps at the ground-truth cosmology and runs 5 chains on each, as 50 separate Slurm jobs. Each map is 128 x 128 pixels of temperature data at 2.5 arcminutes per pixel with 5 uK-arcmin of noise, and each chain runs for 3000 iterations and samples three parameters (`omch2`, `theta_MC_100` and `logA`).

Before you start, set up the cluster as described in [Installation](../getting-started/installation.md#hpc-setup), and tune the proposal widths on a single chain ([Running a Single LCDM Chain](../sample_joint_tutorial.ipynb#tuning)).

## Submitting the chains

The driver script submits one job per chain. Run it from the repository root on a login node:

```bash
bash get_lcdm_samples/sample_lcdm.sh
```

The driver waits whenever too many of your jobs are queued, so it can keep running for a long time. To let it continue after you log out, start it this way instead:

```bash
nohup bash get_lcdm_samples/sample_lcdm.sh > submit.log 2>&1 &
```

See the [submission throttle](../getting-started/installation.md#submission-throttle) for details.

## Changing the settings

Every setting has a default, listed at the top of the driver script. To change one, pass it as a flag of the form `--<name>=<value>`. For example, this runs at a higher noise level with a wider proposal for one parameter, and writes to its own folder:

```bash
bash get_lcdm_samples/sample_lcdm.sh --noise_level=10 --omch2_proposal_sigma=2e-3 --out_dir_name=run_10uk
```

Check the spelling of each flag: a misspelled name is not an error. It is ignored and the default is used.

The settings are:

| Setting | Default | Meaning |
|---|---|---|
| `nside`, `theta_pix` | 128, 2.5 | pixels per map side and arcminutes per pixel |
| `noise_level` | 5 | white noise level in uK-arcmin |
| `pol` | `I` | the data the chains see: `I` (temperature), `P` (polarization) or `IP` (both) |
| `num_maps`, `num_chains` | 10, 5 | the number of data maps and of chains per map |
| `map_prefix` | 234567 | sets the map seeds: map number `m` uses the seed `map_prefix * m` |
| `sample_ombh2`, `sample_omch2`, `sample_theta_MC_100`, `sample_log_a`, `sample_ns` | `omch2`, `theta_MC_100` and `logA` are `true` | `true` or `false`: whether each parameter is sampled |
| `<param>_ground` | the fiducial cosmology | the ground-truth value the data maps are simulated at |
| `<param>_min`, `<param>_max` | about ±5 sigma around the ground truth | the range each parameter is allowed to explore |
| `<param>_proposal_sigma` | 1e-4 (`ombh2`), 1e-3 (`omch2`, `theta_MC_100`), 1e-2 (`log_a`, `ns`) | the width of each parameter's Metropolis proposal |
| `iters_per_chain` | 3000 | iterations per chain |
| `num_burn_in_fix_theta` | 100 | initial iterations in which the LCDM parameters are held fixed |
| `num_burn_in_always_accept` | 0 | initial iterations in which every lensing potential step is accepted |
| `hmc_num_steps`, `hmc_step_size` | 10, 0.05 | the number and size of the steps in the lensing potential update |
| `out_dir_name` | `joint_inference_<submission date>` | the name of the run's output folder |

In the names above, `<param>` is one of `ombh2`, `omch2`, `theta_MC_100`, `log_a` and `ns`.

A sampled parameter starts each chain at a random point in its range. A parameter that is not sampled is held at its ground-truth value.

## Output

The run's folder is created inside the driver's own folder:

```
get_lcdm_samples/lcdm_sampler_data/<out_dir_name>/
```

It holds, for every chain, one history file per sampled parameter, with one sample per line:

```
<param>_map_<map seed>_chain_<chain index>_history.txt
```

The same folder holds a file named `submission_settings.txt`, which records the submission date, every setting and the random starting point of each chain. It is rewritten by every run, so use a new `out_dir_name` whenever you change the settings.

## Next step

Copy the run's folder to your machine and analyze it as shown in [Analyzing Chain Data](../chain_analysis_tutorial.ipynb).
