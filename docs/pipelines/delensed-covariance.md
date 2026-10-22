# Delensed Covariance

Measures the **empirical** per-mode covariance of the delensed temperature field, and its finite difference in theta, to replace the CAMB delensed T block of [the Fisher forecast](../fisher-forecast.md).

Each job uses one seed and runs at every stencil point \(\theta_0\) and \(\theta_0 \pm h_i\):

1. `load_sim(theta)` (common random numbers: at one seed the fields differ only through the cosmology),
2. inverse-lens the **noiseless** lensed field by the Wiener-filtered true phi \(W\phi\), with \(W = C_\phi / (C_\phi + N_\phi)\) built once at \(\theta_0\),
3. store \(|F|^2 / n_\text{side}^2\) on the rfft grid.

The merge is the realization mean at each stencil point.

## Choosing N_phi for the Wiener Filter

| Driver Setting | What it Controls |
|---|---|
| `phi_noise=""`, `qe_filter=lensed` or `delensed` or `unlensed`, `qe_response=unlensed` or `gradient` | the box quadratic estimator |
| `phi_noise=<path>` | the `nphi` of a merged [map_joint](map-joint-phi-noise.md) or [sampler](sampler-phi-noise.md) phi-noise file |

Match it to the forecast it will feed: nothing checks that the file's W was built from the same N_phi as the forecast's phi block.

## Running it

```bash
bash get_delensed_field_covariance/get_delensed_field_covariance.sh
#copy delensed_covariance_output/ to your machine, then
python get_delensed_field_covariance/merge_delensed_field_covariance.py --covariance_dir <delensed_covariance_output>
```

Other settings in the driver: `derivative_params` (the forecast's `--params` must be drawn from this list), `step_sigma` (0.05, in units of `PARAM_SIGMA`), `num_realizations`, `seed_prefix`. A 3-point job at nside 128 takes ~20 s.

## Output: `delensed_covariance.npz`

`delensed_fid`, `delensed_plus`, `delensed_minus` and metadata.
