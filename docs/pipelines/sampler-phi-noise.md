# Measuring Noise in the Sampler

The sampler does not return one estimate of the lensing potential but many samples of it, whose average is the posterior mean. This pipeline measures the noise of that posterior mean for every Fourier mode of the map. The result can be used as the lensing noise of the [Fisher forecast](../api/fisher-forecast.md), or to build the Wiener filter of the [delensed covariance](delensed-covariance.md).

## How the noise is measured

Each job simulates one data map and runs a chain on it with the LCDM parameters held at their ground-truth values, so that only the fields and the lensing potential are sampled. Several chains are run per map. The first starts at a lensing potential of zero and the others at random draws, so the chains start far apart.

The merge step removes the burn-in from every chain and averages the remaining samples of each map into its posterior mean \(\bar\phi\). It then averages two quantities over the maps, and takes their ratio:

\[
r^2 = \frac{\langle \mathrm{Re}(\bar\phi\, \phi_{\rm true}^*) \rangle}{\langle |\phi_{\rm true}|^2 \rangle}.
\]

This is how strongly the posterior mean responds to the true lensing potential. It is converted into a noise level with the theory spectrum \(C_\phi\) of the lensing potential:

\[
N_\phi = C_\phi \left( \frac{1}{r^2} - 1 \right).
\]

This differs from the formula for the [MAP estimates](map-joint-phi-noise.md), which also uses the power of the estimate, \(\langle |\hat\phi|^2 \rangle\). For an exact posterior mean the two formulas agree. For a finite chain, the power of the posterior mean is inflated by the chain's own sampling error, and the formula here avoids it.

## Running it

By default the pipeline simulates 100 temperature maps of 128 x 128 pixels at 2.5 arcminutes per pixel with 5 uK-arcmin of noise, and runs 2 chains of 6000 iterations on each, one Slurm job per chain. The settings are changed with flags, as for every driver (see [Installation](../getting-started/installation.md#how-the-drivers-work)).

**Step 1.** On the cluster, submit the chains from the repository root. The driver keeps running until every job is submitted (see the [submission throttle](../getting-started/installation.md#submission-throttle)).

```bash
nohup bash measure_phi_noise/get_sampler_phi_noise/get_sampler_phi_noise.sh > submit.log 2>&1 &
```

If the driver is interrupted before its last job is submitted (for example by a reboot of the login node), run the same command again with the same flags: chains already submitted are skipped, as for every driver (see the [submission throttle](../getting-started/installation.md#submission-throttle)).

**Step 2.** Merge the chains, giving the number of iterations to remove from the start of each chain as burn-in. The merge is itself a Slurm job, submitted from the driver's folder. It can be run while chains are still going: it uses the iterations finished so far.

```bash
cd measure_phi_noise/get_sampler_phi_noise
sbatch merge_sampler_phi_noise.sh --burn_in=600
```

The merged file is written into the output folder:

```
sampler_phi_noise_output/sampler_phi_noise.npz
```

**Step 3.** Check that the burn-in was long enough.

Point the entry of `convergence_files` in the `__main__` block of `measure_phi_noise/phi_noise_analysis.py` at the merged file, under its noise level, then run the script. It takes no arguments:

```bash
python ../phi_noise_analysis.py
```

Next to the phi noise spectra, this writes `sampler_rhat_<noise>uk.png` into `measure_phi_noise/phi_noise_plots/noise_level_<noise>uk/`, a plot that compares the chains of each map with one another, as a function of scale. It shows the Gelman-Rubin statistic, which is close to 1 when the chains agree. Every scale should be below the reference line at 1.01. A scale well above it means the burn-in or the chains were too short there.

## What the merged file holds

| Key | Content |
|---|---|
| `nphi` | the noise of every mode. It is NaN where it could not be measured, and 0 at the zero mode |
| `nphi_error` | the jackknife error on the noise |
| `nphi_leave_one_out` | the noise recomputed with each map left out, one array per map |
| `rhat_real`, `rhat_imaginary` | the Gelman-Rubin statistic of every mode's real and imaginary part, one array per map |
| `auto_per_realization`, `cross_per_realization`, `true_auto_per_realization` | the power of the posterior mean, its cross power with the truth and the power of the truth, one array per map |
| `burn_in`, `n_chains`, `n_used` | the burn-in removed, the chains per map and the iterations kept per chain |

It also records the seeds, the map size, the resolution, the noise settings, the data type and the cosmology of the run.

## Using the result

Pass the merged file to the Fisher forecast as its lensing noise, together with a delensed covariance:

```bash
python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance <delensed_covariance.npz> --phi_noise sampler_phi_noise.npz
```

The forecast must use the same data type (`--pol`) as the run that produced the file. The same file can be given to the delensed covariance pipeline (see [Computing Delensed Covariance Matrices](delensed-covariance.md)).

## Plotting the phi noise spectra

One script plots this noise next to the noise of the MAP estimates, as a function of scale:

```bash
python measure_phi_noise/phi_noise_analysis.py
```

The paths of the two merged files and the map settings are set at the bottom of the script. The plots compare both noise levels with the theory spectrum of the lensing potential and with the quadratic-estimate noise, and are written into this folder:

```
measure_phi_noise/phi_noise_plots/noise_level_<noise>uk/
```
