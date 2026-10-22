# Simulating Data

`cmb_lensing.simulate.load_sim` is the main entry point. It

1. runs CAMB for the requested cosmology,
2. builds the field, phi and noise covariances in Fourier space,
3. draws the unlensed field, phi and the noise,
4. lenses the field with `primal_lense_flow`,
5. returns a `DataSetT` / `DataSetEB` / `DataSetTEB` holding the fields, the operators and the simulated data.

```python
from cmb_lensing.simulate import load_sim
data_set = load_sim(nside = 128,         #pixels per side
                    theta_pix = 2.5,     #arcmin per pixel
                    pol = "IP",          #"I", "P" or "IP"
                    master_seed = 67,
                    uk_arcmin_t = 5,     #white noise level in T (P is sqrt(2) higher)
                    l_knee = 0)          #0 for pure white noise
```

Useful attributes of the returned data set: `unlensed_field`, `lensed_field`, `phi`, `data`, `field_covariance`, `phi_covariance`, `noise_covariance`, `mask`, `beam`, `mixing_d`, `mixing_g`, `quadratic_estimate`.

## Choices to know about

- **Noise sprectra** are computed via `noise_cls(...)` with a 1/f term. The defaults are `l_knee = 100` and `alpha_knee = 3`. The default beam FWHM is 0.
- **Cosmology** is given in CAMB's names (`cosmomc_theta`, `As`, `ombh2`, `omch2`, `ns`, ...). The sampler's names (`theta_MC_100`, `logA`) are converted by `to_camb_naming_conv` (which edits its dict **in place**) or `fisher_forecast.load_sim_cosmology`.
- **CAMB runs inside `jax.pure_callback`**, so `load_sim` is traceable. A CAMB failure gives NaN spectra and a non-finite logpdf, never an exception.
