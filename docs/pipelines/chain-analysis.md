# Chain Analysis

```bash
python get_lcdm_samples/chain_analysis.py
```

turns the chains into parameter distributions and compares them with one [Fisher forecast](../fisher-forecast.md).

## What it computes

- **The chains' covariance**: one `np.cov` per map (its sub-chains concatenated), averaged over maps, with a delete-one-map jackknife error.
- **Per-parameter plots**, each chain thinned by its integrated autocorrelation time.
- **The product distribution** (the renormalized product of the per-map KDEs) and `z_scores.png`. The table is written in standard notation: `0.1092(1)` means 0.1092 ± 0.0001. The product mean, product std and z-score, (truth − product mean) / product std, carry Monte Carlo errors from a bootstrap. Each of the `N_BOOTSTRAP` replicates resamples every map's thinned chain with replacement, keeping its length, then rebuilds the product and forms its z-score. The average single-map std carries the std of the per-map widths / sqrt(number of maps).
- **One forecast**, compared with the chains. Forecasts carry no error bars.

The phi-noise spectra plots are made separately by `measure_phi_noise/phi_noise_analysis.py` (see [Sampler Phi Noise](sampler-phi-noise.md)).

## Inputs

The forecast's inputs are optional arguments of `main`, set in `__main__` for the run's box and noise level:

```python
main(..., t_block = "unlensed",            #"unlensed" | "lensed" | "delensed" | "measured"
          delensed_covariance = None,      #merged delensed_covariance.npz, for "measured"
          phi_noise_source = "qe",         #"qe" | "measured"
          phi_noise = None,                #merged map_joint_phi_noise.npz / sampler_phi_noise.npz, for "measured"
          qe_filter = "unlensed",          #"unlensed" | "delensed", for "qe"
          qe_response = "unlensed")        #"unlensed" | "gradient", for "qe"
```

`t_block` sets the temperature block: CAMB's unlensed or lensed C_TT, CAMB's analytic delensing at the forecast's N_phi, or a measured delensed covariance. `phi_noise_source` sets the phi block's N_phi: the box QE N0 or a measured phi-noise matrix. A `"measured"` option needs its file. Build a measured delensed covariance's W from the same N_phi as the forecast.

## Output Folder

`get_lcdm_samples/chain_analysis_output/`:

- `<param>/`, one folder per sampled parameter
- `joint_information/`: `z_scores.png`, `triangle_plot.png`, `covariance_matrix_{empirical,fisher}.png`, `correlation_matrix_{empirical,fisher}.png`, `covariance_ratio_fisher_over_empirical.png`, `correlation_ratio_fisher_over_empirical.png`
