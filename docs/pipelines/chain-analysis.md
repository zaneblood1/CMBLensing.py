# Chain Analysis

```bash
python -m cmb_lensing.chain_analysis <chain_dir> --nside 128 --theta_pix 2.5 --noise 2.5
```

turns a directory of [LCDM chains](lcdm-chains.md) into MCMC plots and, on request, compares them with one [Fisher forecast](../fisher-forecast.md). The [tutorial notebook](../chain_analysis_tutorial.ipynb) walks through it on data in the repository, and the Python entry points are in the [API reference](../api/chain-analysis.md).

`<chain_dir>` holds the `<param>_map_<seed>_chain_<index>_history.txt` files that `sample_joint` writes. The sampled parameters, the maps and their sub-chains are read off the file names. The analysis does not depend on which data the chains saw, so it works for temperature (`--pol I`), polarization (`P`) and joint (`IP`) runs; `--pol` labels the plots and tells the forecast which data to model.

## What it computes

- **Per-parameter diagnostics**: the Gelman-Rubin R-hat across each map's sub-chains, the Metropolis acceptance rate, autocorrelations and integrated autocorrelation times.
- **Per-parameter plots**, each sub-chain thinned by its integrated autocorrelation time: traces, per-map marginals and the histogram of per-map means.
- **The product distribution** (the renormalized product of the per-map KDEs) and `z_scores.png`. The table is written in standard notation: `0.1092(1)` means 0.1092 ± 0.0001. The product mean, product std and z-score, (truth − product mean) / product std, carry Monte Carlo errors from a bootstrap. Each of the `N_BOOTSTRAP` replicates resamples every map's thinned chain with replacement, keeping its length, then rebuilds the product and forms its z-score. The average single-map std carries the std of the per-map widths / sqrt(number of maps).
- **The chains' covariance**: one `np.cov` per map (its unthinned sub-chains concatenated), averaged over maps, with a delete-one-map jackknife error.
- **Optionally one forecast** (`--run_fisher`), compared with the chains. Forecasts carry no error bars.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--nside`, `--theta_pix`, `--noise` | 128, 2.5, 2.5 | the box the chains ran on (pixels per side, arcmin, uK-arcmin); they label the triangle plot and set the forecast's box |
| `--pol` | `I` | `I`, `P` or `IP`: the data the chains saw |
| `--burn_in` | 1000 | sweeps removed from the start of every sub-chain |
| `--no_thin` | off | skip the thinning before the per-parameter plots |
| `--out_dir` | `<chain_dir>/chain_analysis_output` | where the plots go |
| `--run_fisher` | off | also run the forecast and compare it with the chains |
| `--delensed_covariance` | none | a merged [`delensed_covariance.npz`](delensed-covariance.md); required by `--run_fisher` |
| `--qe_filter` | `unlensed` | `unlensed` or `lensed`: the spectra in the filter of the forecast's QE N0 |
| `--phi_noise` | none | a merged [`map_joint_phi_noise.npz`](map-joint-phi-noise.md) or [`sampler_phi_noise.npz`](sampler-phi-noise.md); its `nphi` replaces the QE N0 |
| `--no_bb` | off | for `P` / `IP` chains: leave the delensed BB block out of the forecast |
| `--camb_lmax` | 4000 | how far the forecast runs CAMB |

## The forecast

The forecast does not run unless `--run_fisher` is given. It is [the Fisher forecast](../fisher-forecast.md) for the chains' `--pol`, with white noise and no beam.

- **Field blocks:** always the empirical delensed covariance of `--delensed_covariance`, measured at the chains' box, noise level and `--pol`. Without the file `--run_fisher` is an error; there is no analytic delensing fallback.
- **Phi noise:** by default the box quadratic-estimate N0, built like `load_sim`'s (without its `NPHI_FAC`; the TT estimator for `I`, the EB estimator for `P` and `IP`), with the unlensed spectra in both the response and the filter. `--qe_filter lensed` puts the lensed spectra in the filter; the response is always unlensed. `--phi_noise` replaces the N0 with a measured per-mode matrix.
- **BB:** for `P` and `IP` the delensed BB block is counted unless `--no_bb` is given.

Build the delensed covariance's Wiener filter from the same N_phi the forecast uses.

```bash
python -m cmb_lensing.chain_analysis <chain_dir> --noise 2.5 --run_fisher \
    --delensed_covariance <delensed_covariance.npz>                          #unlensed QE N0
python -m cmb_lensing.chain_analysis <chain_dir> --noise 2.5 --run_fisher \
    --delensed_covariance <delensed_covariance.npz> --qe_filter lensed       #lensed filter
python -m cmb_lensing.chain_analysis <chain_dir> --noise 2.5 --run_fisher \
    --delensed_covariance <delensed_covariance.npz> --phi_noise <map_joint_phi_noise.npz>
```

## Output Folder

`<chain_dir>/chain_analysis_output/` (or `--out_dir`), overwritten by every run:

- `<param>/`, one folder per sampled parameter
- `joint_information/`: `z_scores.png`, `triangle_plot.png`, `covariance_matrix_empirical.png`, `correlation_matrix_empirical.png`; with `--run_fisher` also `covariance_matrix_fisher.png`, `correlation_matrix_fisher.png`, `covariance_ratio_fisher_over_empirical.png` and `correlation_ratio_fisher_over_empirical.png`

The phi-noise spectra plots are made separately by `measure_phi_noise/phi_noise_analysis.py` (see [Sampler Phi Noise](sampler-phi-noise.md)).
