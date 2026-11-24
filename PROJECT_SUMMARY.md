# Project summary

A plain-language description of what this repository does and how its parts fit together. It is meant to be read top to bottom by a person who is new to the project. Install steps and a quick start are in `README.md`, tutorials and command references are on the [documentation site](https://zaneblooddev.com/CMBLensing.py/), the change history is in `CHANGELOG.md`, and the detailed notes for AI coding agents are in `CLAUDE.md` / `AGENTS.md`. The `/update-docs` skill (`.claude/skills/update-docs/SKILL.md`) keeps all of these, including this file, in step with the code.

## What the project is

CMBLensing.py is a [JAX](https://github.com/jax-ml/jax) port of the Julia package [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl). It works on small, flat, periodic patches of sky and does four things:

1. **Simulates** a lensed map of the cosmic microwave background (CMB): an unlensed temperature and / or polarization field `f`, a lensing potential `phi`, the lensed field, and instrument noise.
2. **Reconstructs** `f` and `phi` from the noisy lensed map, as the single most probable pair (`map_joint`).
3. **Samples** the full posterior of `f`, `phi` and the five LCDM cosmological parameters given that map (`sample_joint`). This is the part the project is centred on and the part that goes beyond the Julia package's own sampler.
4. **Forecasts** how tightly the map should constrain those parameters (a Fisher forecast), so the sampler's posterior widths have something to be compared against.

The rest of the repository exists to run those four things at scale on a cluster, to measure the inputs the forecast needs, and to check the port against the Julia original.

## Words used throughout

- **Box.** A simulated patch is `nside` x `nside` pixels of `theta_pix` arcminutes each. The standard configuration is nside 128 at 2.5 arcminutes.
- **`pol`.** Which data the map holds: `"I"` (temperature only), `"P"` (polarization only, E and B modes) or `"IP"` (both). Each has its own field type (`FlatS0`, `FlatS2`, `FlatS02`), covariance type (`DiagonalScalar`, `DiagonalEB`, `BlockTEB`) and data set type (`DataSetT`, `DataSetEB`, `DataSetTEB`).
- **theta.** The five sampled parameters, always in the order of `constants.PARAM_ORDER`: `theta_MC_100` (100 x the angular size of the sound horizon), `logA` (the log of the primordial amplitude), `ns` (the spectral tilt), `ombh2` and `omch2` (the baryon and cold dark matter densities). The optical depth and the neutrino mass are fixed, and there are no tensor modes (r = 0).
- **Ground truth.** Every simulated data map, forecast and measurement is built at one fiducial cosmology, `constants.GROUND_TRUTH`. `constants.PARAM_BOUNDS` is the box the sampler is allowed to search (roughly five sigma either side) and `constants.PARAM_SIGMA` a rough one-sigma scale per parameter.
- **N_phi.** The noise of a lensing reconstruction: how far a recovered `phi` is from the true one, per Fourier mode.

Fields are stored as real FFTs (`rfft2`, shape `(nside, nside//2 + 1)`), and everything is double precision.

## The pipeline, step by step

### Simulating a data map (`simulate.load_sim`)

`load_sim(nside, theta_pix, pol, master_seed, ...)` runs [CAMB](https://camb.info) for the requested cosmology, turns the resulting power spectra into covariance matrices on the Fourier grid, draws the unlensed field(s), the lensing potential and the noise from them, lenses the field, and returns one data set object holding the fields, the covariances, the instrument operators and the simulated data. One seed fixes every random draw, and two calls at one seed but different cosmologies share the same underlying white noise. There is no sky mask and by default no beam. A failed CAMB run gives NaN spectra instead of an error.

### Lensing (`lense_flow.py`)

Lensing moves the CMB pattern along the gradient of `phi`. The code does this with the LenseFlow algorithm: an ordinary differential equation integrated with a fixed number of Runge-Kutta steps, which can be run forwards (lens), backwards (delens) or as its adjoint. The gradient of the lensing with respect to `phi` is not left to automatic differentiation: a hand-written adjoint equation is integrated instead, because that is what reproduces the Julia results.

### The most probable reconstruction (`map_joint.map_joint`)

`map_joint(data_set, num_steps = 30)` returns the pair `(f, phi)` that maximizes the joint posterior at the data set's own cosmology. It alternates two steps: a Wiener filter that gives the best `f` for the current `phi`, and a line search that improves `phi` along the gradient of the log-posterior.

### Sampling the cosmology (`sample_lcdm.sample_joint`)

`sample_joint` draws samples of `(f, phi, theta)` given one data map by Gibbs sampling, updating one block at a time while the others are held fixed. One iteration is:

1. **Draw `f`** exactly from its conditional distribution (a simulated field plus a Wiener-filter correction toward the data).
2. **Change variables** to a "mixed" parametrization of `f` and `phi` in which the three blocks are less correlated, so the chain moves faster.
3. **Update `phi`** with one Hamiltonian Monte Carlo step.
4. **Update each sampled parameter** in turn with a one-dimensional random-walk Metropolis move. Every proposal needs the CMB and lensing power spectra at a new cosmology (next section). Proposals outside the search box are rejected.
5. **Rebuild** the covariance and mixing matrices at the accepted parameters and change variables back.

Which parameters are sampled, their starting values, proposal widths and ranges are passed as dictionaries keyed by parameter name. The function returns the parameter histories, the final `phi` and the final random key, and can append every sample to text files (`<param>_map_<map>_chain_<chain>_history.txt`) so a cluster run can be analysed while it is running or resumed later.

`python -m cmb_lensing.sample_lcdm` runs one pilot chain on a laptop: a temperature map at nside 128, sampling `omch2`, `theta_MC_100` and `logA`.

### Where the sampler's spectra come from (the 5D CAMB grid)

Running CAMB at every Metropolis proposal would be far too slow, so the spectra are precomputed. `camb_grid_generation/` runs CAMB on a regular grid over the five parameters (70,875 runs, split over 875 cluster jobs) and merges the results into one file, `camb_splines/camb_grid_spline.npz` (about 8 GB). `camb_grid_interp.py` evaluates a cubic spline through that grid at any point inside it. The file is not in git: it is downloaded from Hugging Face the first time the sampler needs it (or ahead of time with `cmb-lensing-download-splines`). A chain on temperature data holds about 4.5 GB of it in memory, and a temperature + polarization chain about 9 GB.

### Analysing chains (`chain_analysis.py`)

`python -m cmb_lensing.chain_analysis <chain_dir>` reads a directory of history files, which normally holds several chains for each of several data maps, and writes plots into `<chain_dir>/chain_analysis_output/`: traces, autocorrelation, per-map posteriors, and for each parameter the combined ("product") distribution over maps. A summary table reports, per parameter, the combined mean and width, how many sigma the mean sits from the truth (the z-score), and the Gelman-Rubin convergence statistic. With `--run_fisher` it also overlays the Fisher forecast on a triangle plot and compares the two covariance matrices.

### The Fisher forecast (`fisher_forecast.py`)

`python -m cmb_lensing.fisher_forecast --noise <level> --delensed_covariance <file>` predicts the parameter covariance from how the power spectra of the data change with each parameter. It treats the delensed CMB field and the lensing potential as independent Gaussian fields, each observed with noise. Two of its inputs cannot be computed analytically and are measured from simulations by the pipelines below:

- the power spectrum of the **delensed field** at the fiducial cosmology and at a small step either side of it in each parameter (required), and
- optionally a measured **N_phi**; without one, the analytic noise of the standard quadratic estimator is used.

The forecast is an approximation, not the exact information content of the data: it ignores the correlation between the field and the lensing potential and counts some lensing information twice. It supports all three `pol` settings.

## Measurement pipelines

Each is a Python module in the package, a folder of cluster scripts that runs it once per random seed, and a merge script that combines the per-seed files.

| Quantity | Module | Scripts | Merged output |
|---|---|---|---|
| Delensed field covariance: simulate a map, undo the lensing with a filtered copy of the true `phi`, record the power spectrum; repeated at the fiducial cosmology and at a step either side in each parameter | `delensed_covariance.py` | `get_delensed_field_covariance/` | `delensed_covariance.npz` |
| N_phi of `map_joint`: compare the reconstructed `phi` with the true one over many maps | `map_joint_phi_noise.py` | `measure_phi_noise/get_map_joint_phi_noise/` | `map_joint_phi_noise.npz` |
| N_phi of the sampler: run chains over `f` and `phi` with the cosmology held at the truth, and compare the posterior-mean `phi` with the true one | `sampler_phi_noise.py`, `fixed_theta_chain.py` | `measure_phi_noise/get_sampler_phi_noise/` | `sampler_phi_noise.npz` |

`measure_phi_noise/phi_noise_analysis.py` plots the measured noise spectra against the quadratic estimator's.

## Running on a cluster

The cluster scripts are written for slurm. Each pipeline has a driver (`camb_grid.sh`, `sample_lcdm.sh`, `get_*.sh`, `run_performance_test.sh`) that is run with plain `bash`, submits one job per unit of work, and writes its outputs into a folder next to itself. Every setting of a driver is a line with a default that can be overridden as `--<name>=<value>`, and each run records its settings in `<out_dir>/submission_settings.txt`. Machine-specific settings (the conda environment, an e-mail address, the path of a CMBLensing.jl checkout) live in a gitignored `hpc_local.sh`, copied from `hpc_local.example.sh`. Submissions go through `hpc_submit.sh`, which waits when too many jobs are already queued.

The default chain experiment (`get_lcdm_samples/sample_lcdm.sh`) is 10 data maps with 5 chains each of 3000 iterations, on temperature data at nside 128, 2.5 arcminute pixels and 5 uK-arcmin noise, sampling `omch2`, `theta_MC_100` and `logA` from random starting points.

## Repository layout

| Path | Contents |
|---|---|
| `cmb_lensing/` | the Python package |
| `docs/`, `mkdocs.yml` | the documentation site (MkDocs Material): getting started, three tutorial notebooks, one page per cluster pipeline, and an API reference generated from the docstrings |
| `tests/` | comparison tests against CMBLensing.jl and the CAMB grid test |
| `camb_grid_generation/` | builds the 5D CAMB grid |
| `get_lcdm_samples/` | runs LCDM chains on a cluster |
| `get_delensed_field_covariance/`, `measure_phi_noise/` | the measurement pipelines above |
| `runtime_comparison/` | times `map_joint` in Python against Julia |
| `experimental_results/` | figures and summaries of finished experiments; also the source of the results website |
| `paper/` | the accompanying paper |
| `camb_splines/`, `cmb_lensing_data/` | downloaded data, not in git |

Inside the package:

| Module | Role |
|---|---|
| `constants.py` | parameter order, fiducial values, search box, defaults |
| `fields.py`, `matrix_operators.py`, `dataset.py` | the field, covariance and data set types |
| `util.py` | FFT helpers, derivatives, grids, the Runge-Kutta solver |
| `simulate.py` | CAMB, covariance matrices, random fields, `load_sim` |
| `lense_flow.py` | lensing and its adjoint |
| `statistics.py`, `gradients.py`, `mixing.py`, `wiener_filter.py` | the log-posterior, its gradients, the mixed parametrization, the Wiener filter |
| `map_joint.py` | the most probable reconstruction |
| `sample_lcdm.py` | the LCDM sampler |
| `camb_grid_interp.py`, `camb_splines.py` | the 5D grid spline and its download |
| `chain_analysis.py`, `mcmc_diagnostics.py` | chain plots and convergence statistics |
| `fisher_forecast.py`, `camb_spectra.py`, `matrix_plots.py` | the forecast and its helpers |
| `delensed_covariance.py`, `map_joint_phi_noise.py`, `sampler_phi_noise.py`, `fixed_theta_chain.py` | the measurement pipelines |

## Data

Large data is stored on the Hugging Face dataset [`zaneblood1/cmb_lensing_data`](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data), not in git: the CAMB grid, the chains, the delensed covariances, the phi noise measurements and the Python against Julia timing results. Only the grid is downloaded automatically; the other folders are fetched by hand into `cmb_lensing_data/` when a script needs them (`docs/getting-started/data.md`).

Finished results are shown on a static [results website](https://zaneblood1.github.io/CMBLensing.py/results/). It is built from `experimental_results/<section>/<dataset>/`: every folder of figures there becomes one selectable data set of its section (LCDM sampling, the Python against Julia comparison, the noise spectra).

## Tests and what is validated

- The test suite (`pytest`) compares the Python output with stored CMBLensing.jl results for `load_sim`, the lensing, the log-posterior, its gradients, the Wiener filter and `map_joint`. Many tests write comparison plots to `tests/test_generated_figures/` instead of asserting, so the plots (viewed through `tests/index.html`) are part of the check. `pytest --generate` regenerates the Julia reference data and needs Julia 1.11, `juliacall` and a CMBLensing.jl checkout (`docs/getting-started/unit-tests.md`).
- `tests/test_camb_grid.py` checks the grid spline against direct CAMB runs, and is skipped when the grid file is absent.
- **Not covered by any test:** the LCDM sampler, the Fisher forecast and the three measurement pipelines. The sampler is checked by experiment instead: over many simulated maps, the z-scores of the recovered parameters should scatter about zero with unit width.
- **Polarization** (`"P"`, `"IP"`) is supported by `load_sim`, `map_joint`, the sampler, the forecast and the measurement pipelines, but every published chain experiment is on temperature data. The polarization forecast has only been run at nside 64 with 4 realizations, and no polarization chains exist to compare it with.
- Seeds do not reproduce data made before 2026-10-15 (simulated maps) or 2026-11-10 (fixed-cosmology chains), when the handling of random keys changed.

## Conventions worth knowing

- The code style (spaces around `=`, comments without a leading space after `#`, named constants instead of magic numbers, library functions over hand-rolled ones, little defensive checking) is described in `CONTRIBUTING.md` and `CLAUDE.md`.
- There is no linter or formatter. The only continuous integration job builds and deploys the documentation site on every push to `master`.
- `CODE_REVIEW_FINDINGS.md` is a local, untracked list of open review findings; it is maintained by hand and is not part of the documentation the `/update-docs` skill touches.
