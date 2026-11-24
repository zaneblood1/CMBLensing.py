# Measuring Noise in the MAP Estimates

The MAP estimate of the lensing potential is never exact: it differs from the true lensing potential by a reconstruction noise. This pipeline measures that noise for every Fourier mode of the map, at the ground-truth cosmology. The result can be used as the lensing noise of the [Fisher forecast](../api/fisher-forecast.md), or to build the Wiener filter of the [delensed covariance](delensed-covariance.md).

## How the noise is measured

Each job simulates one data map, runs the MAP estimate on it, and compares the estimated lensing potential \(\hat\phi\) with the true one \(\phi_{\rm true}\). For every Fourier mode it stores three numbers:

\[
|\hat\phi|^2, \qquad \mathrm{Re}(\hat\phi\, \phi_{\rm true}^*), \qquad |\phi_{\rm true}|^2 .
\]

These are the power of the estimate, the cross power of the estimate with the truth, and the power of the truth.

The merge step averages each of the three over all of the simulated maps. It then forms the squared correlation between the estimate and the truth,

\[
r^2 = \frac{\langle \mathrm{Re}(\hat\phi\, \phi_{\rm true}^*) \rangle^2}{\langle |\hat\phi|^2 \rangle \langle |\phi_{\rm true}|^2 \rangle},
\]

and converts it into a noise level, using the theory spectrum \(C_\phi\) of the lensing potential:

\[
N_\phi = C_\phi \left( \frac{1}{r^2} - 1 \right).
\]

A mode that is reconstructed perfectly has \(r^2 = 1\) and no noise. A mode that carries no information has \(r^2\) near 0 and a very large noise. The averages are taken before the ratio, because a ratio of averages is far less noisy than an average of ratios.

The error on the noise is a jackknife: the noise is recomputed with each simulated map left out in turn, and the scatter of those results gives the error.

## Running it

By default the pipeline simulates 100 temperature maps of 128 x 128 pixels at 2.5 arcminutes per pixel with 5 uK-arcmin of noise, and submits one Slurm job per map. The settings are changed with flags, as for every driver (see [Installation](../getting-started/installation.md#how-the-drivers-work)).

**Step 1.** On the cluster, submit the jobs from the repository root.

```bash
bash measure_phi_noise/get_map_joint_phi_noise/get_map_joint_phi_noise.sh
```

**Step 2.** When the jobs have finished, copy the output folder to your machine. It is created inside the driver's own folder:

```
measure_phi_noise/get_map_joint_phi_noise/map_joint_phi_noise_output/
```

**Step 3.** Merge the job files into a single file.

```bash
python measure_phi_noise/get_map_joint_phi_noise/merge_map_joint_phi_noise.py --noise_dir <path to map_joint_phi_noise_output>
```

The merged file is written into that folder:

```
map_joint_phi_noise.npz
```

## What the merged file holds

Every array is on the same Fourier grid as the maps, with the shape `(nside, nside // 2 + 1)`.

| Key | Content |
|---|---|
| `nphi` | the noise of every mode. It is NaN where the average cross power is zero or negative, and 0 at the zero mode |
| `nphi_error` | the jackknife error on the noise |
| `nphi_leave_one_out` | the noise recomputed with each map left out, one array per map |
| `auto_per_realization`, `cross_per_realization`, `true_auto_per_realization` | the three stored numbers for every map, one array per map. The noise and its error can be rebuilt from them without the job files |
| `nside`, `theta_pix`, `noise_level`, `l_knee`, `pol`, `param_names`, `params` | the map size, resolution, noise settings, data type and cosmology of the run |

## Using the result

Pass the merged file to the Fisher forecast as its lensing noise, together with a delensed covariance:

```bash
python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance <delensed_covariance.npz> --phi_noise map_joint_phi_noise.npz
```

The forecast must use the same data type (`--pol`) as the run that produced the file.

The same file can be given to the delensed covariance pipeline, which then builds its Wiener filter from this noise (see [Computing Delensed Covariance Matrices](delensed-covariance.md)).

To plot the noise as a function of scale, next to the sampler's noise, see [Measuring Noise in the Sampler](sampler-phi-noise.md#plotting-the-phi-noise-spectra).
