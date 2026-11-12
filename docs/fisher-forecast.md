# Fisher Forecast

A Gaussian Fisher forecast of the sampler's posterior widths, for temperature (`I`), polarization (`P`) or joint (`IP`) data, from per-mode covariance blocks on the sampler's own rfft grid:

\[
F_{ij} = \frac{1}{2} \sum_\text{blocks} \sum_k w_k \,
\mathrm{Tr}\!\left[ C_k^{-1} \frac{\partial C_k}{\partial \theta_i}\, C_k^{-1} \frac{\partial C_k}{\partial \theta_j} \right],
\]

with \(w_k\) the rfft mode weights (2 per bulk mode, 1 on the self-conjugate columns). For a single-number block the trace is \(\partial_i \ln C_k \, \partial_j \ln C_k\).

## The blocks

| `--pol` | Field blocks | Phi block |
|---|---|---|
| `I` | \(C_{TT} + N_{TT}\) | \(C_\phi + N_\phi\) |
| `P` | \(C_{EE} + N_{EE}\); \(C_{BB} + N_{BB}\) | same |
| `IP` | \(\begin{pmatrix} C_{TT} + N_{TT} & C_{TE} \\ C_{TE} & C_{EE} + N_{EE} \end{pmatrix}\); \(C_{BB} + N_{BB}\) | same |

- **The field blocks are always measured.** Their signal at \(\theta_0\) and \(\theta_0 \pm h_i\) is the [empirical delensed covariance](pipelines/delensed-covariance.md) of `--delensed_covariance`, measured at the same box, noise level and `--pol`. The finite-difference steps \(h_i\) come from that file. Without the file the forecast is an error: there is no analytic delensing.
- **The noise** is analytic and independent of the cosmology, with \(N_{EE} = N_{BB} = 2 N_{TT}\).
- **The phi block** uses \(C_\phi(\theta)\) from direct CAMB calls and an \(N_\phi\) held at \(\theta_0\).

## Choosing N_phi

| Option | N_phi |
|---|---|
| default | the quadratic-estimate \(N^{(0)}\) of the box, built like `load_sim`'s (the TT estimator for `I`, the EB estimator for `P` and `IP`), with the unlensed spectra in both its response and its filter |
| `--qe_filter lensed` | the same, with the lensed spectra in the filter; the response stays unlensed |
| `--phi_noise <npz>` | the `nphi` of a merged [`map_joint_phi_noise.npz`](pipelines/map-joint-phi-noise.md) or [`sampler_phi_noise.npz`](pipelines/sampler-phi-noise.md); unmeasured (NaN) modes carry no phi information |

Build the delensed covariance's Wiener filter from the same N_phi the forecast uses; nothing checks it.

## Running it

```bash
python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance delensed_covariance.npz
python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance delensed_covariance.npz --qe_filter lensed
python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance delensed_covariance.npz --phi_noise map_joint_phi_noise.npz
python -m cmb_lensing.fisher_forecast --noise 5 --pol IP --delensed_covariance delensed_covariance_IP.npz
python -m cmb_lensing.fisher_forecast --noise 5 --pol IP --delensed_covariance delensed_covariance_IP.npz --no_bb
```

Other flags: `--nside`, `--theta_pix`, `--l_knee`, `--beam_fwhm`, `--params` (default `logA omch2 theta_MC_100`), `--camb_lmax` (default 4000, the support of the simulations).

Outputs go to `cmb_lensing/fisher_output/` as `fisher_matrix`, `covariance_matrix`, `correlation_matrix` (PNG) and `fisher` (npz), each tagged `_<pol>_<nphi>_<noise>_uk`, e.g. `fisher_I_qe_unlensed_5_uk.npz`. Parameters are in the package's standard order: `theta_MC_100, logA, ns, ombh2, omch2`.

## What the forecast leaves out

- The blocks are added as independent. The residual lensing left in the delensed field tracks \(C_\phi(\theta)\), so the field blocks count some of the lensing information the phi block already has.
- The delensed BB is pure residual lensing at \(r = 0\). Its modes are neither Gaussian nor independent, so counting them as such is optimistic; `--no_bb` leaves the BB block out and shows how much of a `P` or `IP` forecast rests on it.
- The correlations between the fields and phi are ignored.
- For `IP` the \(N^{(0)}\) is the EB estimator's, as in the sampler; a combined temperature and polarization estimator would be lower.
- Parameters not in `--params` are held fixed, not marginalized over.

The analytic-delensing options of the earlier forecast (CAMB's partially delensed spectrum, the iterated QE filter, the gradient response) were removed on 2026-11-11; the last version with them is commit `b50d4cc`.
