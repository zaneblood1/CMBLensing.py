# CMBLensing.py

CMBLensing.py is the JAX port of the Julia package [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl). It

- **simulates** flat-sky CMB temperature and polarization fields, lensing potentials and noisy lensed data maps (`load_sim`),
- **lenses**, inverse-lenses and adjoint-lenses fields with the LenseFlow ODE (`lense_flow`),
- **reconstructs** the unlensed field and the lensing potential by joint MAP optimization (`map_joint`),
- **samples** the LCDM parameters jointly with the field and the lensing potential by Gibbs sampling (`sample_joint`), extending the algorithm developed by [Millea et al. 2020](https://arxiv.org/abs/2002.00965),
- **forecasts** the sampler's posterior widths with a Gaussian Fisher forecast (`fisher_forecast`).

All array operations run in JAX (float64), so the code runs on CPU and GPU and is differentiable where it needs to be.

## Where to start

| I want to... | Go to |
|---|---|
| install the package | [Installation](getting-started/installation.md) |
| simulate a map and run `map_joint` | [MAP Estimation Tutorial](map_joint_tutorial.ipynb) |
| learn more about sampling the LCDM parameters | [Sampling Tutorial](sample_joint_tutorial.ipynb), [LCDM Sampling](concepts/sample-joint.md) |
| run a full experiment on a cluster | [HPC Setup](getting-started/hpc-setup.md) |
| compare sampler data to the Fisher forecast | [Fisher Forecast](fisher-forecast.md), [Chain Analysis](pipelines/chain-analysis.md) |
| change the code | [Testing](testing.md) |

## Repository layout

```
cmb_lensing/                    the Python package
camb_splines/                   1D CAMB spline caches (tracked) + the 5D grid (download)
camb_grid_generation/           scripts that build the 1D caches and the 5D grid
get_lcdm_samples/               LCDM chains on the HPC + chain_analysis.py
get_map_joint_phi_noise/        map_joint's per-mode N_phi
get_sampler_phi_noise/          the sampler's per-mode N_phi
get_delensed_field_covariance/  the empirical delensed T covariance stencil
runtime_comparison/             map_joint timing, Python vs Julia
tests/                          A/B tests against CMBLensing.jl + the CAMB grid test
docs/                           this site (Markdown pages + tutorial notebooks)
paper/                          the paper source
```
