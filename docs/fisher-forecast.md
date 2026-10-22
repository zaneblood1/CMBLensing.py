# Fisher Forecast

A Gaussian Fisher forecast of the sampler's posterior widths, from two covariance blocks on the sampler's own rfft grid:

| Block | Covariance |
|---|---|
| delensed T | \(C_{TT}^\text{delensed} + C_n\) (\(C_n\) analytic, theta independent) |
| phi | \(C_\phi + N_\phi\) (\(N_\phi\) frozen at \(\theta_0\)) |

\[
F_{ij} = \frac{1}{2} \sum_\text{blocks} \sum_k w_k \,
\frac{\partial \ln C_k}{\partial \theta_i} \frac{\partial \ln C_k}{\partial \theta_j},
\]

with \(w_k\) the rfft mode weights (2 per bulk mode, 1 on the self-conjugate columns) and central finite differences of step \(h = 0.05\,\sigma\) from direct CAMB calls.

## One N_phi drives both blocks

N_phi is the phi block's noise and also sets the residual lensing in the delensed T through \(A_L = N_L / (C_L + N_L)\). `--nphi_source` picks it:

| `--nphi_source` | N_phi | Role |
|---|---|---|
| `qe_unlensed` | box QE with the unlensed C_TT in its filter | **upper** bound on the information |
| `qe_delensed` (default) | box QE with the filter iterated to the delensed C_TT | **lower** bound |
| `map_joint` | `nphi` of [`map_joint_phi_noise.npz`](pipelines/map-joint-phi-noise.md) (`--phi_noise`) | comparison |
| `sample_joint` | `nphi` of [`sampler_phi_noise.npz`](pipelines/sampler-phi-noise.md) (`--phi_noise`) | comparison |


## Running it

```bash
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source qe_unlensed                       #upper bound
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source qe_delensed --qe_response gradient #lower bound
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source map_joint --phi_noise map_joint_phi_noise.npz
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source sample_joint --phi_noise sampler_phi_noise.npz
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source qe_unlensed --delensed_covariance delensed_covariance.npz
```

Other flags: `--nside`, `--theta_pix`, `--l_knee`, `--beam_fwhm`, `--params` (default `logA omch2 theta_MC_100`), `--camb_lmax`, `--stability` (also computes at step 2h).

By default the T block is CAMB's partially delensed spectrum at the frozen \(A_L\). `--delensed_covariance` swaps in the [empirical delensed covariance](pipelines/delensed-covariance.md).

Outputs go to `cmb_lensing/fisher_output/fisher*_from_blocks_<nphi>_<camb|empirical>_<noise>_uk.*`, with parameters in the order `omch2, ombh2, ns, theta_MC_100, logA`.