# Chain Analysis

```bash
python get_lcdm_samples/chain_analysis.py
```

turns the chains into parameter distributions and compares them with every available [Fisher forecast](../fisher-forecast.md).

## What it computes

- **The chains' covariance**: one `np.cov` per map (its sub-chains concatenated), averaged over maps, with a delete-one-map jackknife error.
- **Per-parameter plots**, each chain thinned by its integrated autocorrelation time.
- **Forecasts** for each N_phi source, with the CAMB delensed T block and, where a file is given, the empirical one. Forecasts carry no error bars.
- **Phi-noise and delensed-spectra plots** at the chains' noise level.

## Inputs

The measured inputs are arguments of `main`, set in `__main__` for the run's noise level:

```python
forecast_phi_noise = {"map_joint": "<merged map_joint_phi_noise.npz>",
                      "sample_joint": "<merged sampler_phi_noise.npz>"}
forecast_delensed_covariance = {"qe_delensed": "<merged delensed_covariance.npz>"}
```

A file-based N_phi source without an entry is skipped.

## Output Folder

`get_lcdm_samples/lcdm_chain_plots/noise_level_<noise>uk/`
