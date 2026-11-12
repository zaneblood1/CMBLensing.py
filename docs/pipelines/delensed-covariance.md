# Delensed Covariance

Measures the **empirical** per-mode covariance of the delensed field, and its finite difference in theta. It is the field block of [the Fisher forecast](../fisher-forecast.md), which cannot run without it.

Each job uses one seed and runs at every stencil point \(\theta_0\) and \(\theta_0 \pm h_i\):

1. `load_sim(theta)` at the chosen `pol` (common random numbers: at one seed the fields differ only through the cosmology),
2. inverse-lens the **noiseless** lensed field by the Wiener-filtered true phi \(W\phi\), with \(W = C_\phi / (C_\phi + N_\phi)\) built once at \(\theta_0\),
3. store the per-mode covariance \(\mathrm{Re}(X Y^*) / n_\text{side}^2\) of every component on the rfft grid.

| `pol` | Stored components |
|---|---|
| `I` | TT |
| `P` | EE, BB |
| `IP` | TT, TE, EE, BB |

The merge is the realization mean at each stencil point.

## Choosing N_phi for the Wiener Filter

| Driver Setting | What it Controls |
|---|---|
| `phi_noise=""`, `qe_filter=unlensed` (default) or `lensed` | the box quadratic-estimate \(N^{(0)}\) (the TT estimator for `I`, the EB estimator for `P` and `IP`), with the unlensed or lensed spectra in its filter; the response is always unlensed |
| `phi_noise=<path>` | the `nphi` of a merged [map_joint](map-joint-phi-noise.md) or [sampler](sampler-phi-noise.md) phi-noise file |

Match it to the forecast it will feed: nothing checks that the file's W was built from the same N_phi as the forecast's phi block.

## Running it

```bash
bash get_delensed_field_covariance/get_delensed_field_covariance.sh
#copy delensed_covariance_output/ to your machine, then
python get_delensed_field_covariance/merge_delensed_field_covariance.py --covariance_dir <delensed_covariance_output>
```

Other settings in the driver: `pol` (`I`, `P` or `IP`; the forecast must use the same), `derivative_params` (the forecast's `--params` must be drawn from this list), `step_sigma` (0.05, in units of `PARAM_SIGMA`; accurate at nside 128 / 2.5' / 2.5 to 10 uK-arcmin, so for another box or noise level rerun with the step halved and check that the forecast does not change substantially), `num_realizations`, `seed_prefix`, `out_dir_name` (the output folder's name inside `get_delensed_field_covariance/`). A 3-point temperature job at nside 128 takes ~20 s.

Every setting has a default in the driver and can be overridden on the command line as `--<name>=<value>`, e.g.

```bash
bash get_delensed_field_covariance/get_delensed_field_covariance.sh --noise_level=10 --pol=IP \
    --derivative_params="omch2 logA" --out_dir_name=delensed_covariance_output_IP_10uk
```

A misspelled name is ignored and the default used. Each run writes the submission date and every setting to `submission_settings.txt` in the output folder (replacing any earlier copy); use a new `out_dir_name` when you change settings.

## Output: `delensed_covariance.npz`

| Key | Content |
|---|---|
| `pol`, `components` | the data's polarization and the stored components, in order |
| `delensed_fid` | the delensed covariance at \(\theta_0\), shape `(n_components, nside, nside // 2 + 1)` |
| `delensed_plus`, `delensed_minus` | the same at \(\theta_0 \pm h_i\), shape `(n_params, n_components, nside, nside // 2 + 1)` |
| `names`, `steps` | the differentiated parameters (in the package's standard order) and their steps \(h_i\) |
| metadata | the box, the stencil points, the seeds and how W was built |

Merged temperature files written before 2026-11-11 have no `components` key and no component axis; the forecast still reads them.
