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
bash get_map_joint_phi_noise/get_map_joint_phi_noise.sh          #one job per seed
#copy map_joint_phi_noise_output/ to your machine, then
python get_map_joint_phi_noise/merge_map_joint_phi_noise.py --noise_dir <map_joint_phi_noise_output>
```

Settings at the top of the driver: `nside`, `theta_pix`, `noise_level`, `l_knee`, `map_joint_steps`, `num_realizations`, `seed_prefix`.

## Output: `map_joint_phi_noise.npz`

| Key | Content |
|---|---|
| `nphi` | per-mode phi noise on the rfft2 grid; NaN where the mean cross is <= 0, 0 at the origin |
| `nphi_error` | jackknife error |
| box / cosmology keys | `nside`, `theta_pix`, `noise_level`, `l_knee`, `param_names`, `params` |

Can be used in forecasting via `fisher_forecast --nphi_source map_joint --phi_noise map_joint_phi_noise.npz` or as the noise source for [the delensed covariance](delensed-covariance.md).
