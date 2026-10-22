# Testing

Most tests are A/B comparisons against CMBLensing.jl: for the same inputs, the Python and Julia outputs should agree to numerical precision. The Julia outputs are stored in `tests/ground_truth_data/*.npz`.

| Test | Covers |
|---|---|
| `test_covariance_matrices.py`, `test_simulated_cls.py` | `load_sim`'s spectra and covariances |
| `test_lensing.py` | `lense_flow` |
| `test_logpdf.py`, `test_gradients.py` | the logpdf and its gradients |
| `test_wiener_filter.py`, `test_map_joint.py` | the Wiener filter and `map_joint` |
| `test_camb_grid.py` | the 5D CAMB grid vs direct CAMB (skipped without the grid) |

The sampler, the forecast, the 1D spline caches and the measurement modules are not tested.

## Running

```bash
pytest                          #all tests
pytest tests/test_lensing.py    #one file
```

Many tests write comparison plots to `tests/test_generated_figures/` rather than asserting. Open `tests/index.html` with a local server (e.g. VS Code Live Server) and look at the diff plots: a test can pass while a plot shows a problem.

## Regenerating the Julia data

Needs [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) and a local CMBLensing.jl checkout:

```bash
export CMBLENSING_JL_PATH=/path/to/CMBLensing.jl
python tests/generate_julia_data/generate_all.py              #all of it
python tests/generate_julia_data/generate_lensing.py          #one test's data
pytest --generate                                             #regenerate, then test
```

`conftest.py` maps each `test_*` module to its `generate_*` module for `--generate`.
