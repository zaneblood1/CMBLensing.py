# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). No version has been tagged yet: everything below is the development history of the upcoming `0.1.0`, newest first, with the date each change landed.

## [Unreleased]

### Added

- **2026-11-12** All large data (chain directories, delensed field covariances, map_joint and sampler phi noise, runtime comparison results) is stored next to the 5D CAMB grid on the Hugging Face dataset `zaneblood1/cmb_lensing_data`.
- **2026-11-11** Polarization support (`pol = "I"`, `"P"`, `"IP"`) in the Fisher forecast, the empirical delensed covariance stencil and the fixed-theta sampler phi noise chains.
- **2026-11-11** `python -m cmb_lensing.chain_analysis <chain_dir>` command line interface; `chain_analysis.py` now lives in the package.
- **2026-11-11** `cmb_lensing/camb_spectra.py` (direct CAMB spectra, instrument matrices, quadratic-estimate noise) and `cmb_lensing/matrix_plots.py` (annotated heatmaps), shared by the forecast and the measurement modules.
- **2026-11-09** `hpc_submit.sh` with `submit_throttled`: every HPC driver caps its queued jobs and retries a rejected `sbatch`.
- **2026-11-09** Every HPC driver setting is a `--<name>=<value>` override, and each run writes its settings to `<out_dir>/submission_settings.txt`.
- **2026-11-09** Several chains per map in the sampler phi noise measurement, with per-mode Gelman-Rubin R-hat and `plot_sampler_convergence.py`.
- **2026-11-09** `cmb_lensing/mcmc_diagnostics.py` (Gelman-Rubin R-hat, autocorrelation, integrated autocorrelation time).
- **2026-10-28** The 5D CAMB grid is downloaded from Hugging Face on first use (`cmb_lensing/camb_splines.py`, `cmb-lensing-download-splines`).
- **2026-10-28** `CONTRIBUTING.md` and the mkdocstrings API reference (`docs/api/`).
- **2026-10-21** MkDocs Material documentation site, deployed by `.github/workflows/docs.yml`, and the `sample_joint` tutorial notebook.
- **2026-10-19** `tests/test_camb_grid.py`: the 5D grid spline checked against direct CAMB, including the theta_MC_100 to H0 inversion.
- **2026-10-16** `hpc_local.example.sh`: personal cluster settings live in a gitignored `hpc_local.sh`, so the tracked HPC scripts hold nothing personal and run from any checkout location.
- **2026-10-06** Sampler effective phi noise measurement from fixed-theta chains (`sampler_phi_noise.py`, `fixed_theta_chain.py`).
- **2026-09-24** Empirical delensed field covariance stencil (`delensed_covariance.py`), the source of the Fisher forecast's field blocks and their finite differences.
- **2026-09-16** Per-mode map_joint phi noise measurement (now `map_joint_phi_noise.py`).
- **2026-09-10** T-phi cross spectrum (`cls["TP"]`) in `load_sim`'s spectra.
- **2026-09-03** `experimental_results/` and the JOSS paper (`paper/`).
- **2026-08-27** Fisher forecast (`fisher_forecast.py`) and triangle plots comparing chains with the forecast.
- **2026-08-24** 5D CAMB grid spline (`camb_grid_interp.py`, `camb_grid_generation/`) as the sampler's Cl source.
- **2026-06-17** LCDM sampler (`sample_lcdm.py`): Gibbs sampling of the fields, the lensing potential and the LCDM parameters.
- **2026-05-20** Joint sampling of the fields and the lensing potential (HMC step on the mixed phi).
- **2026-05-05** juliacall-based tests against CMBLensing.jl ground truth, and HPC scripts timing `map_joint` against Julia (`runtime_comparison/`).
- **2026-05-01** Initial JAX port of CMBLensing.jl: `load_sim`, LenseFlow lensing with a hand-written adjoint, the Wiener filter and `map_joint`.

### Changed

- **2026-11-11** The Fisher forecast always takes its field blocks from a measured delensed covariance (`--delensed_covariance` is required) and contracts per-mode covariance blocks, so one code path serves temperature, polarization and joint data.
- **2026-11-11** One parameter ordering, `constants.PARAM_ORDER`, in every matrix, figure and output file.
- **2026-11-11** `chain_analysis.py` is narrowed to MCMC diagnostics plus the comparison with one forecast (off by default, `--run_fisher`); it reads map seeds and sub-chain indices off the history file names.
- **2026-11-10** Fixed-theta chains derive their key from `chain_key(map_seed, chain_seed)`; chains written before this date are not reproducible under the current scheme.
- **2026-11-09** The sampler phi noise uses the response estimator, which a finite chain's Monte Carlo error does not bias.
- **2026-11-03** `get_map_joint_phi_noise/`, `get_sampler_phi_noise/` and `phi_noise_analysis.py` moved into `measure_phi_noise/`.
- **2026-10-28** Test file paths work on Windows.
- **2026-10-20** Simplified the sampler phi noise merge and the `get_lcdm_samples/` code path.
- **2026-10-19** Simplified the 5D grid interpolation around one function, `spline_coefficients`, evaluated with `scipy.interpolate.NdBSpline` (same file format, outputs unchanged to ~1e-13 relative). The map_joint phi noise is measured per Fourier mode.
- **2026-10-15** `load_sim` draws every random field from a subkey split off one root key in a fixed order, so two cosmologies at one seed share their white noise. A seed no longer reproduces realizations made before this date.
- **2026-10-15** The `get_*` folders, `runtime_comparison/` and `camb_splines/` moved to the repository root.
- **2026-10-14** `fisher_forecast.py` rewritten after the sibling forecast modules were deleted; chain outputs reorganized into per-task subdirectories.
- **2026-08-24** `sample_lcdm.py` is the sole sampler, and all spectra come from CAMB.

### Removed

- **2026-11-11** Rotated-coordinate theta sampling (`reparameterize_lcdm`), the 1D CAMB splines, and the forecast's analytic CAMB delensing, iterated quadratic-estimate filter, gradient response and `nphi_source` options. Commit `b50d4cc` is the last one containing them.
- **2026-10-28** `camb_splines/` from git (the grid is downloaded instead).
- **2026-10-19** The low-memory grid evaluator, the checks in `merge_camb_grid.py`, and the band averaging and validation options of the map_joint phi noise measurement.
- **2026-10-15** The mixed-logpdf Hessian forecast and the Louis information forecast.
- **2026-10-14** The sibling Fisher forecast modules, `lensing_covariance.py`, the delensing transfer function (`delensed_spectrum.py`), the score-variance phi noise bound (`sampler_noise_estimate.py`) and both `*_TEMPLATE` directories.
- **2026-08-24** The CAMB emulator, `sampling_ar.py` (the A_phi / r sampler), `sample_lcdm_legacy.py` and every experimental theta sampler (grid inverse-CDF, PCA, joint Metropolis-Hastings, GHMC / NUTS); only per-parameter random-walk Metropolis remains.

### Fixed

- **2026-10-29** Fourier weight discrepancy in the band calculations of `chain_analysis.py`.
- **2026-08-26** Lensing `num_steps` bug.
- **2026-06-29** Mixed phi gradient: the anti-Hermitian content on the two self-conjugate rfft columns is kept by carrying phi in Fourier space through the whole gradient path, bringing the agreement with CMBLensing.jl from 2e-2 to 2e-4.
- Random fields are drawn from real white noise, so the two self-conjugate rfft columns get full rather than half power.
