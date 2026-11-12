# MAP Joint Phi Noise

Measures the per-mode reconstruction noise of `map_joint` at `GROUND_TRUTH`. Each job simulates one realization with `load_sim`, reconstructs phi with `map_joint`, and stores per rfft2 mode the moments of the estimate \(\hat\phi\) and the true phi:

\[
A = |\hat\phi|^2, \qquad B = \mathrm{Re}\, \hat\phi\, \phi^*, \qquad D = |\phi|^2 .
\]

The merge averages them over realizations **before** taking the ratio

\[
r^2 = \frac{\langle B \rangle^2}{\langle A \rangle \langle D \rangle}, \qquad N_\phi = C_\phi \left( \frac{1}{r^2} - 1 \right),
\]

with a delete-one-realization jackknife error.

## Running it

```bash
bash measure_phi_noise/get_map_joint_phi_noise/get_map_joint_phi_noise.sh          #one job per seed
#copy map_joint_phi_noise_output/ to your machine, then
python measure_phi_noise/get_map_joint_phi_noise/merge_map_joint_phi_noise.py --noise_dir <map_joint_phi_noise_output>
```

Settings at the top of the driver: `nside`, `theta_pix`, `noise_level`, `l_knee`, `pol` (`I`, `P` or `IP`: the data map_joint reconstructs phi from; default `I`), `map_joint_steps`, `num_realizations`, `seed_prefix`, `out_dir_name` (the output folder's name inside `get_map_joint_phi_noise/`, default `map_joint_phi_noise_output`). Each is a default that can be overridden on the command line as `--<name>=<value>`, e.g.

```bash
bash measure_phi_noise/get_map_joint_phi_noise/get_map_joint_phi_noise.sh --noise_level=10 --out_dir_name=map_joint_phi_noise_output_10uk
```

A misspelled name is ignored and the default used. Each run writes the submission date and every setting to `submission_settings.txt` in the output folder (replacing any earlier copy); use a new `out_dir_name` when you change settings.

## Output: `map_joint_phi_noise.npz`

| Key | Content |
|---|---|
| `nphi` | per-mode phi noise on the rfft2 grid; NaN where the mean cross is <= 0, 0 at the origin |
| `nphi_error` | jackknife error |
| `nphi_leave_one_out` | the N leave-one-out `nphi` the jackknife comes from, shape `(n_realizations, nside, nside // 2 + 1)` |
| box / cosmology keys | `nside`, `theta_pix`, `noise_level`, `l_knee`, `pol`, `param_names`, `params` |

Can be used in forecasting via `fisher_forecast --delensed_covariance <npz> --phi_noise map_joint_phi_noise.npz` (with the matching `--pol`) or as the noise source for [the delensed covariance](delensed-covariance.md). Its spectra are plotted by `measure_phi_noise/phi_noise_analysis.py` (see [Sampler Phi Noise](sampler-phi-noise.md#plotting-the-phi-noise-spectra)).
