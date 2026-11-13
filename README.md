# CMBLensing.py

CMBLensing.py is a [JAX](https://github.com/jax-ml/jax) port of the Julia package [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl). It simulates gravitationally lensed CMB temperature and polarization maps on a flat sky, recovers the unlensed field and the lensing potential from a lensed data map, and jointly samples the five LCDM parameters from that map. All array operations run through JAX, so the code is GPU compatible and automatically differentiable.

Full documentation, tutorials and the API reference: <https://zaneblooddev.com/CMBLensing.py/>

## What it does

- **Simulate** (`load_sim`): runs CAMB for a chosen cosmology, draws the unlensed field(s), the lensing potential and the noise, and lenses the field with the `lense_flow` algorithm. Temperature only (`"I"`), polarization only (`"P"`) and joint (`"IP"`) data sets are supported.
- **Lense** (`lense_flow`): quickly lense, inverse lense or adjoint lense a field by a given lensing potential.
- **Estimate** (`map_joint`): the joint maximum a posteriori estimate of the unlensed field `f` and the lensing potential `phi`, found from gradients of the logpdf of an `(f, phi)` pair given the observed data.
- **Sample** (`sample_joint` in `sample_lcdm.py`): re-purposes the Gibbs sampler of CMBLensing.jl (see [Millea et al. 2020](https://arxiv.org/abs/2002.00965)) to draw joint samples of `f`, `phi` and the five LCDM parameters (`theta_MC_100`, `logA`, `ns`, `ombh2`, `omch2`). Given a synthetic data map generated at known ground truth values and a search range for each parameter, the chains give posterior distributions whose means are the inferred parameter values and whose widths are their uncertainties.
- **Forecast** (`fisher_forecast.py`): a Gaussian Fisher forecast to compare against the sampler's posterior widths.

## Relationship to CMBLensing.jl

This code base was diffed against commit [`8e75a7c`](https://github.com/marius311/CMBLensing.jl/commit/8e75a7c56e37d337256c5646305630816e1ee196) of CMBLensing.jl, and the Julia comparison tests in `tests/` check `load_sim`, the lensing, the logpdf, its gradients, the Wiener filter and `map_joint` against that version.

The two code bases will not necessarily be kept in sync past this commit. If you need a specific enhancement from a newer version of the Julia code, please request it by opening an [issue](https://github.com/zaneblood1/CMBLensing.py/issues).

**Performance.** For maps at or below 256 x 256 pixels, `map_joint` takes the same order of magnitude of time in Python as in Julia. For larger maps the Python code is slightly slower; closing that gap is an active area of investigation.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/zaneblood1/CMBLensing.py.git
cd CMBLensing.py
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"    #pytest + jupyter
pip install -e ".[docs]"   #MkDocs, to build the documentation locally
```

`sample_joint` reads its theory spectra from a 5D CAMB spline (~7.9 GB), which is downloaded automatically from [Hugging Face](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data) into `camb_splines/` the first time it is needed. To fetch it ahead of time (e.g. on a cluster login node), run `cmb-lensing-download-splines`.

## Data

All of the repository's large data is stored on the Hugging Face dataset [zaneblood1/cmb_lensing_data](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data), since it is too large for GitHub: the 5D CAMB grid, the LCDM chains, the `map_joint` and sampler lensing reconstruction noise, the delensed covariances used by the Fisher forecast, and the Python vs Julia `map_joint` comparison outputs (~23 GB in total). Only the CAMB grid is downloaded automatically. To fetch one of the other folders, e.g. the chains:

```bash
hf download zaneblood1/cmb_lensing_data --repo-type dataset \
    --include "joint_inference_data/*" --local-dir cmb_lensing_data
```

The dataset card and the [data documentation](https://zaneblooddev.com/CMBLensing.py/getting-started/data/) describe every folder and how the scripts here read it.

## Quick start

```python
from cmb_lensing.simulate import load_sim
from cmb_lensing.map_joint import map_joint

#simulate a 64 x 64 pixel temperature map at 5 arcminutes per pixel
data_set = load_sim(nside = 64, theta_pix = 5, pol = "I", master_seed = 1)

#recover the unlensed field and the lensing potential
f_map, phi_map = map_joint(data_set)
```

To run a single pilot LCDM sampling chain on one data map:

```bash
python -m cmb_lensing.sample_lcdm
```

A larger experiment, averaging over many data map realizations to reduce cosmic variance, can be run with the slurm scripts in `get_lcdm_samples/`.

## Repository layout

| Path | Contents |
|---|---|
| `cmb_lensing/` | the Python package |
| `docs/` | the documentation pages and the `map_joint` / `sample_joint` / chain analysis tutorial notebooks |
| `tests/` | comparison tests against CMBLensing.jl and the CAMB grid test |
| `get_lcdm_samples/` | slurm scripts for LCDM chains (analysed by `python -m cmb_lensing.chain_analysis`) |
| `measure_phi_noise/` | the lensing reconstruction noise of `map_joint` and of the sampler |
| `get_delensed_field_covariance/` | the empirical delensed covariance used by the forecast |
| `camb_grid_generation/` | scripts that build the CAMB splines |
| `runtime_comparison/` | `map_joint` timings, Python vs Julia |
| `experimental_results/`, `paper/` | results and the accompanying paper |

## Testing

A pytest suite calls all the 1-to-1 methods that can be compared between Julia and Python versions and checks that for the same inputs the two implementations produce outputs that are identical to within numerical precision. The full suite of tests can be run by executing `pytest` in the repo root.

Most tests compare against ground truth generated by CMBLensing.jl. Regenerating that data (`pytest --generate`) needs [`juliacall`](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) and a CMBLensing.jl checkout at the commit above; see the [testing documentation](https://zaneblooddev.com/CMBLensing.py/testing/).

## Contributing, citing and license

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). To cite this software, see [CITATION.cff](CITATION.cff). Released under the [MIT license](LICENSE).
