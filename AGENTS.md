# AGENTS.md

This file provides guidance to Codex (claude.ai/code) when working with code in this repository.

## Project Overview

JAX port of CMBLensing.jl — simulates gravitational lensing of CMB temperature and polarization fields, recovers the lensing potential via gradient-based MAP optimization (`map_joint`), and jointly samples the five LCDM cosmological parameters from a lensed data map via Gibbs sampling (`sample_joint` in `sample_lcdm.py`). The sampler's posterior widths are compared against a Gaussian Fisher forecast (`fisher_forecast.py`) built from a measured delensed covariance, for temperature, polarization or joint data. All array operations use JAX for GPU/TPU support and automatic differentiation.

History that explains the current shape of the tree:
- **2026-08-24 refactor:** the CAMB emulator, `sampling_ar.py`, `sample_lcdm_legacy.py` and every experimental theta sampler were removed; `sample_lcdm.py` is the sole sampler; spline caches lived in `cmb_lensing/camb_splines/` (moved to the repo-root `camb_splines/` on 2026-10-15). See "Removed features".
- **2026-10-14 cleanup:** the user deleted the sibling forecast modules (`fisher_forecast_from_cls.py`, `_full_sky.py`, `_from_logpdf.py`, `_from_1st_principles.py`), `lensing_covariance.py`, `delensed_spectrum.py` (transfer function), `sampler_noise_estimate.py` (score bound), `compare_grid_derivatives.py`, `plot_nphi_sources.py`, and both `*_TEMPLATE` directories. `fisher_forecast.py` was then rewritten around the two-bound report, and `sampling_chains/` was reorganized into per-task subdirectories.
- **2026-11-11:** `chain_analysis.py` moved into the package with a CLI; `fisher_forecast.py` narrowed to the measured-delensing forecast (CAMB delensing removed, see "Removed features") and extended, with its three input pipelines, to pol "I" / "P" / "IP"; its shared helpers moved to `camb_spectra.py` / `matrix_plots.py`; one parameter ordering (`PARAM_ORDER`) everywhere. The 1D CAMB spline source was removed the same day; the 5D grid is the sampler's only Cl source.

## Commands

```bash
# Install (editable)
pip install -e .

# Tests (the user runs these manually; do not run the whole suite after every change)
pytest
pytest tests/test_lensing.py

# Regenerate Julia ground-truth data before testing (requires juliacall + CMBLensing.jl;
# first export CMBLENSING_JL_PATH=/path/to/CMBLensing.jl)
pytest --generate
python tests/generate_julia_data/generate_all.py

# A single pilot LCDM sampling chain locally (nside 128 / 2.5', T-only, 2.5 uK; omch2 / theta_MC_100 / logA)
python -m cmb_lensing.sample_lcdm

# 5D CAMB grid on the HPC (camb_grid_generation/)
bash camb_grid.sh                                          # 875 slurm jobs, one H0 sweep each; every setting (lmax, tau, mnu, k_pivot,
                                                           # alens, *_boost, out_dir_name, <axis>_min/_max, <axis>_nodes) is a
                                                           # name=${name:-default} line, overridable as --<name>=<value>; each run
                                                           # (re)writes date + all settings to <out_dir>/submission_settings.txt
python merge_camb_grid.py --grid_dir multi_param_CAMB_grid  # -> camb_grid_spline.npz
python tests/test_camb_grid.py --grid <...>/camb_grid_spline.npz -n 20   # spline vs direct CAMB

# HPC setup (since 2026-10-16): nothing personal is in the tracked scripts. On the cluster,
#   cp hpc_local.example.sh hpc_local.sh    # gitignored; conda activate, SBATCH_MAIL_USER, CMBLENSING_JL_PATH
# Drivers (camb_grid.sh, sample_lcdm.sh, get_*.sh, run_performance_test.sh) are plain bash, run from
# any directory: each cd's into its own folder, sources the repo-root hpc_local.sh and hpc_submit.sh
# (../, or ../../ under measure_phi_noise/) and submits its job script by a relative name through
# submit_throttled (hpc_submit.sh, since 2026-11-09): before each sbatch it waits while
# MAX_QUEUED_JOBS (85) of your jobs are queued (ACES cancels submissions past ~90) and retries a
# rejected sbatch. A driver therefore blocks until its last job is submitted: run it as
#   nohup bash <driver>.sh > submit.log 2>&1 &      (or inside tmux)
# Job scripts find their .py and the repo-root hpc_local.sh through $SLURM_SUBMIT_DIR
# (= the driver's folder) and write their out_dir inside that folder, so the checkout can live anywhere.

# LCDM chains on the HPC (get_lcdm_samples/; chains -> get_lcdm_samples/lcdm_sampler_data/<experiment>/)
bash sample_lcdm.sh              # spawns run_single_lcdm_chain.sh per (map, chain): 10 maps x 5 chains. Settings
                                 # (incl. *_proposal_sigma, out_dir_name) are name=${name:-default} lines, overridable
                                 # as --<name>=<value>; (re)writes <out_dir>/submission_settings.txt with each chain's start
python -m cmb_lensing.chain_analysis <chain_dir> --nside 128 --theta_pix 2.5 --noise 2.5 --burn_in 1000   # MCMC plots -> <chain_dir>/chain_analysis_output/ (run from the repo root)
python -m cmb_lensing.chain_analysis <chain_dir> --noise 2.5 --run_fisher --delensed_covariance <delensed_covariance.npz>   # + triangle plot vs one forecast
                                 # [--qe_filter lensed | --phi_noise <map_joint_phi_noise.npz / sampler_phi_noise.npz>] [--pol I|P|IP] [--no_bb] [--no_thin] [--out_dir ...]

# Fisher forecast (writes cmb_lensing/fisher_output/{fisher_matrix,covariance_matrix,correlation_matrix}_<pol>_<nphi>_<noise>_uk.png
# and fisher_<pol>_<nphi>_<noise>_uk.npz). --delensed_covariance is REQUIRED: there is no analytic delensing
python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance <delensed_covariance.npz>                     # QE N0, unlensed filter
python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance <delensed_covariance.npz> --qe_filter lensed  # QE N0, lensed filter
python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance <delensed_covariance.npz> --phi_noise <map_joint_phi_noise.npz | sampler_phi_noise.npz>
python -m cmb_lensing.fisher_forecast --noise 5 --pol IP --delensed_covariance <IP delensed_covariance.npz> [--no_bb]   # --pol I|P|IP

# Empirical delensed covariance stencil (get_delensed_field_covariance/)
bash get_delensed_field_covariance.sh                          # one job per seed; settings are name=${name:-default} lines,
                                                               # overridable as --<name>=<value> (e.g. --pol=IP, --qe_filter=lensed,
                                                               # --out_dir_name=...); (re)writes <out_dir>/submission_settings.txt
python merge_delensed_field_covariance.py --covariance_dir <scp'd out_dir>   # -> delensed_covariance.npz

# Sampler effective phi noise (measure_phi_noise/get_sampler_phi_noise/)
bash get_sampler_phi_noise.sh                                  # num_maps x num_chains fixed-theta chains (one job each); settings
                                                               # are name=${name:-default} lines, overridable as --<name>=<value>
                                                               # (e.g. --out_dir_name=...); (re)writes <out_dir>/submission_settings.txt
sbatch merge_sampler_phi_noise.sh --burn_in=600               # a job itself, submit from its folder; burn_in (default 300) / out_dir_name are
                                                               # --<name>=<value> overrides; -> sampler_phi_noise.npz ("nphi"); works on running jobs;
                                                               # a successful merge appends "merged <date> by <user>: burn_in=..." to submission_settings.txt
python plot_sampler_convergence.py --merged <sampler_phi_noise.npz>   # band-averaged R-hat across chains -> sampler_rhat_<noise>uk.png
python -m cmb_lensing.sampler_phi_noise --nside 64 --theta_pix 5 --n_maps 4 --n_chains 2 --n_sweeps 400 --burn_in 100 --out_dir <dir>   # local

# Per-mode map_joint N_phi at GROUND_TRUTH (measure_phi_noise/get_map_joint_phi_noise/)
bash get_map_joint_phi_noise.sh                                # one job per seed; settings are name=${name:-default} lines,
                                                               # overridable as --<name>=<value> (e.g. --out_dir_name=...);
                                                               # (re)writes <out_dir>/submission_settings.txt
python merge_map_joint_phi_noise.py --noise_dir <scp'd out_dir>  # -> map_joint_phi_noise.npz ("nphi")

# Phi noise spectra plots (measure_phi_noise/; box, noise and merged-file paths set in __main__)
python measure_phi_noise/phi_noise_analysis.py   # -> measure_phi_noise/phi_noise_plots/noise_level_<noise>uk/
                                                 #    phi_noise_spectra_<noise>uk.png, phi_noise_spectra_over_qe_<noise>uk.png

# Docs site (MkDocs Material; pages in docs/, nav in mkdocs.yml; pip install -e ".[docs]")
mkdocs serve                 # local preview at http://127.0.0.1:8000
mkdocs build --strict        # what to check before pushing; notebooks are rendered, never executed
```

No linter or formatter is configured. The only CI is `.github/workflows/docs.yml`, which runs `mkdocs gh-deploy` on every push to master. MkDocs is pinned `< 2` (2.0 drops the plugin system). Docs pages are a user-facing digest of this file; keep them in step when a pipeline's commands or outputs change. `docs/api/` (added 2026-10-28) is generated by mkdocstrings from the Google-style docstrings of `load_sim`, `map_joint`, `sample_joint`, `fisher_forecast.forecast`, `chain_analysis` (`main`, `load_chains`, `get_avg_covariance_matrix`, `get_forecasted_covariance`) and the field / operator / data set classes; update those docstrings when their arguments change. mkdocstrings reads the source statically, so the docs CI does not install the package.

## Repository Layout

```
cmb_lensing/                  the Python package (constants.py holds PARAM_ORDER / GROUND_TRUTH / PARAM_BOUNDS / PARAM_SIGMA / CAMB_LMAX / CAMB_SPLINES_DIR / CAMB_GRID_PATH / DEFAULT_HMC_*)
  fisher_output/              fisher_forecast.py's figures and npz
  sample_lcdm_output/         sample_lcdm's progress plots (must exist before plotting)
camb_splines/                 the ~8 GB camb_grid_spline.npz (5D grid); the whole folder is gitignored (since 2026-10-28)
camb_grid_generation/         slurm / merge scripts for the 5D grid
docs/                         MkDocs pages (getting-started/, concepts/, pipelines/, api/, fisher-forecast.md, testing.md)
                              + map_joint_tutorial.ipynb, sample_joint_tutorial.ipynb, chain_analysis_tutorial.ipynb
experimental_results/         hand-maintained results (README + experimental_details.md per run)
paper/                        paper.md / paper.bib / figures / build_pdf.sh
tests/                        Julia A/B benchmarks + test_camb_grid
get_lcdm_samples/             sample_lcdm.sh, run_single_lcdm_chain.sh/.py (the chains are analysed by cmb_lensing/chain_analysis.py)
  lcdm_sampler_data/            chain *_history.txt directories (gitignored; three older
                                omch2_logA_theta_MC_100_joint_inference_* runs are tracked): sampler_output_2.5uk_11_10_2026/
                                (10 maps x 5 chains) is the current local one. Each is a chain_analysis `chain_dir`
    <run>/chain_analysis_output/  chain_analysis's default output (overwritten by every run):
      <param>/                    one folder per sampled parameter, all lowercase
      joint_information/          z_scores.png (per sampled parameter, standard notation value(err): product
                                  mean / std / z-score with bootstrap MC errors, N_BOOTSTRAP = 200, z formed per
                                  replicate; avg single-map std, err = std of widths / sqrt(N_maps)),
                                  triangle_plot.png (chains blue; forecast red with --run_fisher),
                                  {covariance,correlation}_matrix_empirical.png; with --run_fisher also
                                  {covariance,correlation}_matrix_fisher.png and
                                  {covariance,correlation}_ratio_fisher_over_empirical.png.
                                  Chain errors are always the delete-one-map jackknife
  chain_analysis_output/, lcdm_chain_plots/   outputs of the pre-2026-11-11 script; no longer written
```

**`cmb_lensing/chain_analysis.py`** (moved into the package from `get_lcdm_samples/` and given a CLI on 2026-11-11; `python -m cmb_lensing.chain_analysis <chain_dir> ...`, `main(chain_dir, ...)` returns a dict of results). Its job: chain history files (+ optionally a merged delensed covariance) → MCMC plots and the comparison with ONE forecast. Anything else belongs in a debug script.
- **Loading.** `load_chains(chain_dir, burn_in) -> (chains, map_seeds)` reads every sampled parameter (there is no option to analyse a subset), map seeds and sub-chain indices off the file names (`HISTORY_FILE_PATTERN`; no `num_maps` / `num_chains` / `map_pre_factor` / `was_sampled`). A sub-chain is kept when every parameter has its file; its files are cut to their common length BEFORE burn-in removal and concatenation, so rows line up across parameters. `chains[param][map][sub_chain]`.
- **Chain side** is independent of the data's polarization (`--pol I|P|IP` only labels the triangle plot): R-hat per map (printed, no pruning), acceptance rate, ACF / IAT, thinning by IAT (`--no_thin`) for the per-parameter plots, the product distribution with bootstrap errors, and the covariance = one `np.cov` per map (unthinned sub-chains concatenated) averaged over maps, jackknifed over maps (needs at least two maps).
- **Forecast** is OFF by default (`--run_fisher` / `run_fisher = True`). `get_forecasted_covariance(nside, theta_pix, noise, param_names, delensed_covariance, pol, qe_filter, phi_noise, include_bb, ...)` is a thin wrapper of `fisher_forecast.forecast` at the chains' `pol` (white noise, no beam) and returns F^-1 in PARAM_ORDER, the order `load_chains` always returns its parameters in (no permutation anywhere). A missing `delensed_covariance` → `ValueError` / parser error. `--qe_filter {unlensed,lensed}`, `--phi_noise`, `--no_bb` (`include_bb`), `--camb_lmax` (default `CAMB_LMAX`) pass straight through. Never with error bars. The ratio plots colour each entry about 1 (`CenteredNorm(vcenter = 1)`).
- The analytic-delensing comparison (CAMB delensed T block, iterated QE filter, gradient response) no longer exists anywhere; see "Removed features", 2026-11-11, for what it was and the commit that still has it.
- `docs/chain_analysis_tutorial.ipynb` runs it on `experimental_results/lcdm_sampling/logA_omch2_theta_MC_100_2dot5uk_white_noise/data` (tracked); its forecast cells need the gitignored local delensed covariance / phi noise files. Executed locally; re-execute it when outputs change.

The phi-noise spectra plots live in `measure_phi_noise/phi_noise_analysis.py` (2026-11-03), which holds `band_index` / `band_median` / `band_median_jackknife` / `qe_noise_matrices`. Only current-format files are supported (the legacy `sampler_effective_noise.npz` / `phi_noise_fid` paths are gone).

```
get_delensed_field_covariance/ get_delensed_field_covariance.sh, run_single_delensed_field_covariance.sh/.py, merge_delensed_field_covariance.py
  delensed_field_covariance_data/  merged delensed_covariance.npz runs, nside 128 / 2.5', see below
measure_phi_noise/            phi_noise_analysis.py (phi noise spectra plots -> phi_noise_plots/noise_level_<noise>uk/); moved here 2026-11-03
  get_sampler_phi_noise/        get_sampler_phi_noise.sh, *_1_chain.sh/.py, merge_sampler_phi_noise.sh/.py,
                                plot_sampler_convergence.py
    sampler_phi_noise_data_new/   sampler_phi_noise_output_{2.5,5,10}uk/sampler_phi_noise.npz (pre-2026-11-09 format: one chain per
                                  map, 100 maps, burn_in 600, correlation estimator, no R-hat)
    sampler_phi_noise_data_old/   the same folders holding pre-2026-10-20 sampler_effective_noise.npz (burn_in 1000 at 2.5 uK);
                                  convert_old_sampler_phi_noise.py rebuilds a current-format sampler_phi_noise.npz next to each
                                  from its moment_auto_naive / moment_cross_true / moment_true_auto (no Hermitian projection)
  get_map_joint_phi_noise/      get_map_joint_phi_noise.sh, get_map_joint_phi_noise_1_realization.sh/.py, merge_map_joint_phi_noise.py
    map_joint_phi_noise/          map_joint_phi_noise_output_{2.5,5,10}uk/map_joint_phi_noise.npz (100 realizations)
runtime_comparison/           slurm scripts timing map_joint Python vs Julia, performance_analysis.py. run_performance_test.sh's settings
                              (num_seeds, num_trials, theta_pix, noise_level [uK-arcmin, default 3 = CMBLensing.jl's load_sim default,
                              passed to both load_sims], pol_keys, map_sizes, max_wall_hours [the largest map size's; scales linearly],
                              out_dir_name) are --<name>=<value> overrides; (re)writes <out_dir>/submission_settings.txt
hpc_local.example.sh          template for the gitignored hpc_local.sh every HPC script sources
hpc_submit.sh                 submit_throttled, the queue-capped sbatch every driver uses (MAX_QUEUED_JOBS / SUBMIT_POLL_SECONDS)
README.md, CONTRIBUTING.md, CITATION.cff, .gitattributes (`*.sh text eol=lf`, so a Windows checkout's scripts still run under bash)
```

The `get_*` folders and `runtime_comparison/` were moved to the repo root on 2026-10-15 and are no longer gitignored; on 2026-11-03 `get_map_joint_phi_noise/` and `get_sampler_phi_noise/` moved into `measure_phi_noise/`.

**`delensed_field_covariance_data/` naming.** Flat, one folder per (W, noise level), `delensed_covariance_output_<W>_<noise>uk/delensed_covariance.npz`, noise in {2.5, 5, 10}, all current format, 100 realizations at nside 128 / 2.5' (the old `lower_qe_bound/`, `upper_qe_bound/`, `empirical_*` folders are gone):
- `qe_filter_<unlensed|lensed|delensed>_response_unlensed` — W from the box QE (`wiener_nphi_source = "qe"`). `qe_filter_unlensed` matches the forecast's default N_phi; `qe_filter_delensed` (the iterated filter, with the unlensed response) can no longer be regenerated. Only the three `qe_filter_unlensed_response_unlensed_{2.5,5,10}uk` folders are on this machine (2026-11-11). All are temperature files in the pre-2026-11-11 format (no component axis), which the forecast still reads.
- `map_joint` — W from the merged `map_joint_phi_noise.npz` (`wiener_nphi_source = "file"`).
- `sampler_phi_noise` — W from the merged `sampler_phi_noise.npz` (`wiener_nphi_source = "file"`).

**`camb_splines/` is not in git (2026-10-28).** The grid lives on the Hugging Face dataset `zaneblood1/cmb_lensing_data` (`constants.CAMB_SPLINES_REPO`, file `CAMB_GRID_FILE` at the dataset root; the dataset also still holds the five unused 1D caches). The dataset was `zaneblood1/5D_LCDM_CAMB_Grid_Spline` until 2026-11-12 (the old id redirects) and now holds all of the repository's large data next to the grid: `joint_inference_data/` (chain_dirs), `delensed_field_covariance_data/`, `map_joint_phi_noise_data/`, `sampler_phi_noise_data/` (incl. `*_pre_refactor` legacy folders) and `map_joint_comparison/performance_results/` (the former Google Drive folder). Only the grid is auto-downloaded; `docs/getting-started/data.md` and the dataset card document the rest. `cmb_lensing/camb_splines.camb_spline_file(filename)` returns `camb_splines/<filename>`, first downloading it with `hf_hub_download(..., local_dir = CAMB_SPLINES_DIR)` if it is missing (no revision pin, by user decision: the dataset is treated as fixed). It is called by `sample_lcdm.get_camb_grid_predictors` / `camb_grid_bb_is_zero`. A file already present is never re-downloaded, so a locally rebuilt grid at that path wins. `cmb-lensing-download-splines` (`[project.scripts]`) pre-fetches it, e.g. on a cluster login node. `tests/test_camb_grid.py` keeps its skip-if-missing and never downloads. Never force-add the grid: it was once committed by accident and pushed a pack past GitHub's 2 GiB limit.

## Architecture

### Computational Pipeline

`simulate.load_sim(nside, theta_pix, pol, master_seed, ...)` is the main entry point. It runs CAMB (through `jax.pure_callback`, so a CAMB failure yields NaN spectra → non-finite logpdf, never an exception), builds covariance matrices and random fields in Fourier space, lenses them via `primal_lense_flow`, and returns a `DataSetT` / `DataSetEB` / `DataSetTEB` (`dataset.py`) holding the fields, operators, and simulated data. Notable current choices, all made for LCDM inference:
- **Mask is forced to all ones** (`jnp.ones_like(get_mask(...))`) — there is no sky mask in any data set. This also applies to the Julia-comparison tests.
- **Noise:** `noise_cls(..., l_knee, alpha_knee = 3)` includes the 1/f term `1 + (l_knee/ell)**3`; `load_sim`'s default is `l_knee = 100` (matches Julia), while every sampler entry point (`sample_lcdm.__main__`, `run_single_lcdm_chain.py`, `add_starting_matrices_to_args`) passes `l_knee = 0` for pure white noise. Beam FWHM defaults to 0.
- **Cls are CAMB-only.** The `use_emulator_cls` flag and the `cambemul` import are gone.
- **The cls dict carries the T-phi cross spectrum** (added 2026-09-10): `_run_camb` keeps the PP and PT columns of `get_lens_potential_cls(CMB_unit = "muK")` (PP is dimensionless either way; "muK" puts PT in the TT units), the callback returns a `(lmax_prime, 2)` array, and `_extract_all_cls` adds `cls["TP"]` via `dl2cl(..., is_tphi = True)` = D · 2π / (L(L+1))^{3/2}. Note `cls["phi"]` still uses the ℓ⁴ convention, so `TP/√(TT·PP)` differs from CAMB's correlation coefficient by (1 + 1/ℓ). TP changes sign near ℓ ~ 1100 (1e-16 level), so it must not go through the log-interpolating `covar_matrix_from_cls`. Nothing consumes it since the forecast's "lensed" mode was removed (2026-10-14). The 5D grid does not have it. `EXPECTED_CL_KEYS` includes `"TP"`, so a `precomputed_cls` dict must too.
- Tensor covariances are added back in (`cf = cf_scalar + cf_tensor`), but every sampler path runs at `r = 0`.
- `get_d_tt_matrix(cf, cn)` (2 args), `get_d_teb_matrix`, `get_d_eb_matrix` build the D mixing matrix; `get_g_matrix(cphi, nphi, a_phi_fid, a_phi)` / `get_g_matrix_lcdm(cphi_fid, cphi_curr, nphi, cn_tt)` build G (`cn_tt` is unused there). `lmax_prime = min(lmax, DEFAULT_MAX_ELL)` with `DEFAULT_MAX_ELL = 4000` in `constants.py` (renamed from `EMULATOR_MAX_ELL`).

`map_joint(data_set, num_steps = 30, constant_step = False)` recovers f and phi by alternating Wiener-filter steps (estimate f) with a BFGS line search on phi along the preconditioned mixed gradient (`mixed_grad_phi_logpdf`, hessian = `pinv(Cphi^-1 + Nphi^-1)`). G is set to the identity there (MAP is invariant to it).

`sample_joint(...)` (in `sample_lcdm.py`) draws samples from the full joint posterior P(f, phi, theta | d) via Gibbs sampling, where theta are the five LCDM parameters. This is the code path the project is centred on.

> **DEFAULT FILE FOR CONVERSATIONS:** when the user refers to "the sampler", "sample_joint", "gibbs_sample_theta", theta/parameter sampling, etc., they mean **`cmb_lensing/sample_lcdm.py`**. `sample_lcdm_legacy.py` and `sampling_ar.py` no longer exist.

### The LCDM sampler (`sample_lcdm.py`)

**Parameters.** `PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]` (tau is pinned at 0.05 by prior, mnu at 0.06). `current_params` is a `(5,)` jnp array in that order; user-facing dicts (`param_init`, `proposal_sigmas`, `param_ranges`, `should_sample`) are keyed by name. Single sources of truth in `cmb_lensing/constants.py` (every module and script imports them from `constants`): `PARAM_ORDER` / `PARAM_INDEX`, `GROUND_TRUTH` (fiducial values), `PARAM_BOUNDS` (±5-sigma search box), `PARAM_SIGMA` (normalizes the progress plot and sets the forecast's / stencils' steps), `CAMB_LMAX`, `CAMB_SPLINES_DIR`, `CAMB_GRID_PATH`. Note the Julia tests each define their own module-level `GROUND_TRUTH` (a data path) after their wildcard imports, which shadows the dict inside those files only. `to_camb_naming_conv` converts `theta_MC_100 -> cosmomc_theta = theta/100` and `logA -> As = exp(logA)*1e-10` **in place** (it deletes the old keys from the dict it is given).

**Cl source.** One: the 5D CAMB grid spline (the 1D spline source and the `USE_CAMB_GRID` / `USE_CAMB_SPLINE` module switches were removed on 2026-11-11). `get_camb_grid_predictors()` lazily loads the grid once per process from `constants.CAMB_GRID_PATH = camb_splines/camb_grid_spline.npz` (repo root, downloaded on first use) and returns a `GridPredictors` namedtuple (see the grid section). `sample_joint` checks that `param_init` is inside the grid (finite TT) before doing anything heavy. A run therefore always needs the ~7.9 GB file (~4.5 GB of RAM for "I").

Every predictor has the signature `f(params_batch: (M, 5)) -> (M, n_ell)` (the emulator-era `emu_params` first argument is gone). Predictors are passed as **static jit arguments** to `_recompute_cosmo_matrices*`, so they must be the same cached function objects every call (fresh closures retrace and recompile the lensing graph).

**Gibbs structure of `sample_joint`** (mirrors CMBLensing.jl's `sample_joint`), per iteration:
1. `gibbs_sample_f` — draw f from its conditional: simulate a fresh field and noise realization from the current covariances, lens the field, and add a Wiener-filter correction driven by `data - new_data`. Branches on the covariance type: `DiagonalScalar` (independent draw), `BlockTEB` (`_field_matrices_from_teb_covar`: E ~ N(0, C_EE), T = (C_TE/C_EE)E + N(0, C_TT − C_TE²/C_EE), identical to `load_sim`'s correlated T/E draw; a zero TE block reduces it to independent draws so the same helper serves the noise), `DiagonalEB` (`_field_matrices_from_eb_covar`).
2. `mix` (`mixing.py`) — `f° = L(phi) D f`, `phi° = G phi`.
3. `gibbs_sample_phi` — one HMC step (`hmc_step` → `symplectic_integrate`; `sample_joint`'s `hmc_num_steps` / `hmc_step_size` default to `constants.DEFAULT_HMC_NUM_STEPS = 10` / `DEFAULT_HMC_STEP_SIZE = 0.05`; Julia uses N = 25, ε = 0.01) on the mixed phi with mass matrix `pinv(G)² (pinv(Cphi) + pinv(Nphi))` (`get_mass_matrix`). The momentum is drawn from the **phi** template `x`, never from `mixed_field` (an S02 template would corrupt it). Accept if `log(u) < ΔH`; always accepted for the first `num_burn_in_always_accept` iterations.
4. Theta step (skipped while `iter < num_burn_in_fix_theta`): for each parameter in `proposal_sigmas` **dict order** with `should_sample[name] = True`, `gibbs_sample_theta` → `metropolis_sample_theta`: a symmetric Gaussian random-walk Metropolis-within-Gibbs move of scale `proposal_sigmas[name]`, `metropolis_num_steps` per sweep. Proposals outside `param_ranges[name]` are rejected before the interpolator is touched (exact MH, symmetric proposal); NaN logpdfs (out of grid / unreachable theta) reject via `(x < nan) == False`. The conditional is the **mixed** logpdf — `make_eval_logpdf_batch` rebuilds Cphi, Cf (TT / TT+TE+EE+BB / EE+BB), G and D at every candidate theta and evaluates `mixed_logpdf` per row under `jax.vmap`. Tune sigmas toward ~44 % acceptance per parameter (phi HMC toward ~65 %). This is the only theta sampler (the rotated-coordinate option was removed on 2026-11-11, see "Removed features").
5. `update_args_after_sample` → `get_new_cosmo_matrices` — recompute Cphi, Cf, D, G (and optionally the QE norm) at the newly accepted theta.
6. `unmix` — back to `(f, phi)` with the **updated** D/G.

`args` is a plain dict carrying operators (`noise_covariance`, `mask`, `beam`, `field_covariance`, `phi_covariance`, `mixing_d`, `mixing_g`, `quadratic_estimate`) plus metadata (`nside`, `pix_width`, `ell_grid`, `ells`, `lmax`, `lmax_prime`, `cphi_fid`). `add_starting_matrices_to_args` initialises them at `param_init`: it runs a **throwaway `load_sim` at `param_init`** (random seed, `l_knee = 0`) purely to get the lensed covariances for the initial QE norm, then computes Cf from the predictors (`get_new_cf_matrix`) and D/G/Cphi via `get_new_cosmo_matrices`. `set_initial_ds_conditions` swaps those into the data set so `map_joint` (for `phi_init = "MAP"`) sees the initial cosmology rather than ground truth. `cphi_fid` — the ground-truth phi covariance from the data set — is what `get_g_matrix_lcdm` normalises G against.

**refresh_qe.** The jitted kernels `_recompute_cosmo_matrices[_teb|_eb](..., refresh_qe = False)` can rebuild the quadratic-estimate norm from the grid's lensed Cls at the new theta (scalar QE for "I", polar QE for "IP"/"P"), so G and the phi mass matrix would track theta. **`sample_joint` does not currently expose this** — `update_args_after_sample` is called without `refresh_qe`, so the QE norm stays frozen at the `param_init` cosmology for the whole chain.

**Polarization plumbing.** The jitted theta kernels take raw arrays, not structs: `field_matrix_stack` gives `(3, nside, nside//2+1)` T/E/B stacks for "IP", `(2, ...)` E/B for "P", the bare matrix for "I"; `op_matrix_stack` gives `(4, ...)` TT/TE/EE/BB or `(2, ...)` EE/BB operator stacks; `op_shared_matrix` returns the single mask/beam block. `make_eval_logpdf_batch(..., pol, bb_is_zero)` rebuilds `FlatS0/FlatS2/FlatS02` and `DiagonalScalar/DiagonalEB/BlockTEB` per row. `sample_joint` infers `pol` from the data set type. `_replace_teb_blocks` / `_replace_eb_blocks` write recomputed blocks back into the operators (ET mirrors TE).

**Physics decisions carried over from the pre-refactor sampler** (still in force):
- **TE is carried as the correlation ratio** `te_rho = Cl_TE/√(Cl_TT·Cl_EE)` (linearly interpolated, `te_covar_from_rho`), so `Cl_TE = ρ√(TT·EE)` from the same TT/EE the model uses everywhere else keeps the 2×2 T/E block positive semi-definite by construction. The D matrix carries the full coupled T/E block (`get_d_teb_matrix`: Schur inverse + `block_matrix_sqrt`, `d_et` used for both off-diagonals — empirically matches Julia).
- **Unlensed BB = 0 at r = 0.** Identically-zero spectra must bypass the log-interpolating `covar_matrix_from_cls` (log(0) → NaN): eager code uses `_covar_or_zeros`, jitted code the static `bb_is_zero` flag. The D-mixing B block is the **identity** where `cf_bb = 0` (the naive formula gives `d_bb = 0` and zeroes the mixed B sector).
- Never pin f/phi at fixed values while still re-mixing them with theta-dependent D/G every sweep: that makes the theta chain a fixed-point iteration on the mixed conditional (measured attractor at +1 training sigma in theta_MC_100 for "IP"), not MCMC. `fixed_fields = True` (hold f / phi at the data set's true fields, re-mixed every sweep) is exactly that mode: a debugging aid for the theta step only, never a posterior. With genuine f/phi resampling (the default) the mixed-coordinate theta step is exact.

**Logging / outputs.** `advanced_logging` dict keys: `phi_acceptance`, `lcdm_acceptance` (print running acceptance rates), `plot_log_pdf`, `plot_lcdm_sigmas` (overwrite `cmb_lensing/sample_lcdm_output/logpdf_progress.png` and `lcdm_progress.png` every iteration; the directory is untracked and is not created by the code, so it must exist first). `matplotlib.use("Agg")` is set at the very top of the module — TkAgg aborted long chains from JAX worker threads; keep it first. With `hpc_path` set, every iteration appends the latest value of each **sampled** parameter to `{hpc_path}{theta}_map_{map_idx}_chain_{sub_chain_idx}_history.txt`; `chain_analysis.py` reads those. `sample_joint` returns `(param_vals, phi, sub_key)` (dict of per-parameter lists, the final phi, the final key). `resume_chain_dict` (one entry per parameter name + `"phi_map"` + `"sub_key"`, e.g. `np.load` of the `map_<seed>_chain_<i>_data_dict.npz` that `run_single_lcdm_chain.py` writes) restarts a chain, overriding `param_init` / `phi_init` (and the key unless `seed` is given). `phi_init`: `"MAP"` (default, `map_joint`), `"RNG"` (a draw from the inverse mass matrix), anything else zero.

**`__main__`** builds an nside 128, theta_pix 2.5, T-only, 2.5 µK-arcmin, `l_knee = 0` data set at `GROUND_TRUTH` (`master_seed = 469134`), samples omch2 / theta_MC_100 / logA starting at box edges (omch2 and theta at the top, logA at the bottom), `phi_init = "ZEROES"`, `num_burn_in_fix_theta = 100`, 10 000 iterations, `seed = 67`. The HPC path (`get_lcdm_samples/sample_lcdm.sh` → `run_single_lcdm_chain.sh/.py`): nside 128 / 2.5' / 5 uK, 10 maps (seed `234567 * map`) x 5 chains, omch2 / theta_MC_100 / logA from uniform random starts inside the box (ombh2, ns at the truth), `num_burn_in_fix_theta = 100`, 3000 iterations, HMC 10 / 0.05, proposal widths 1e-4 / 1e-3 / 1e-3 / 1e-2 / 1e-2 (ombh2 / omch2 / theta_MC_100 / logA / ns; set in `sample_lcdm.sh` as `*_proposal_sigma`, passed through `run_single_lcdm_chain.sh` as args 32-36), `phi_init = "ZEROES"`, `seed = None`; histories go to `lcdm_sampler_data/<out_dir_name>/` (default `joint_inference_<YYYY_MM_DD>`, the SUBMISSION date, chosen by the driver and passed as `--out_dir`, arg 37) next to the script, plus a resume npz per chain. Which parameters are sampled is `sample_<ombh2|omch2|theta_MC_100|log_a|ns>=true|false` in the driver (args 38-42 → `should_sample`); a sampled parameter starts each chain uniformly at random in its box, an unsampled one at its `*_ground`, and every chain's start is listed in `submission_settings.txt`.

### 5D CAMB Grid Spline (`camb_grid_interp.py`)

Real CAMB spectra precomputed on a full tensor-product grid over `(H0, logA, ns, ombh2, omch2)` (tau 0.05, mnu 0.06, r = 0, tensors off) and evaluated as the tensor-product cubic spline through lnCl (raw values for `te_rho`). Out-of-box queries return NaN → rejected proposal. Simplified 2026-10-19 (same file format, outputs identical to ~1e-13 relative on the real grid):

- **The maths is one function**, `camb_grid_interp.spline_coefficients(axes, values)`: a tensor-product spline is separable, so its coefficients come from a not-a-knot `make_interp_spline` (= CubicSpline's default) along each axis in turn; it returns the coefficients + per-axis knots that `scipy.interpolate.NdBSpline` evaluates. `merge_camb_grid.py` calls it per spectrum (in ell chunks, `--ell_chunk`), `CambGrid` calls it on the small theta grid. `evaluate_spline` / `screen` add the `BOUNDARY_TOL` clip (the sampler's scans end exactly on the box edges, and the theta → H0 round trip lands a few ulp outside) and NaN rows out of box. Not `RegularGridInterpolator(method = "cubic")` (re-solves every construction, ~10 min per spectrum, iterative solver ~4e-4 off) nor `ndimage.spline_filter` (wrong boundary conditions, ~2e-4 off).
- **Grid file:** `camb_splines/camb_grid_spline.npz` (~7.9 GB, gitignored; `constants.CAMB_GRID_PATH`): `coeff_<spectrum>` float32 tables, `axis_*`, `knots_*`, `ells`, raw `theta_grid(H0, ombh2, omch2)`, `bb_is_zero`, CAMB settings. Built on the HPC by `camb_grid_generation/camb_grid.sh` → `run_single_camb_grid.sh/.py` (one slurm job per (logA, ns, ombh2, omch2) node doing the 81-point H0 sweep; 81×5×5×5×7 = 70 875 CAMB calls in 875 jobs; numpy + camb only, no JAX) → `merge_camb_grid.py` (no checks; one spectrum at a time, appended to the npz with `zip_write_array` so the tables are never all in memory) → `python tests/test_camb_grid.py --grid ... -n 20` (random interior points vs direct CAMB).
- **`CambGrid`** opens the npz once (`load_camb_grid` singleton per path) and builds each spectrum's `NdBSpline` **lazily** on first use via `CambGrid.spline(name)` (~1.1 GB read, ~2.1 GB resident since NdBSpline casts to float64; only the spline is cached, in `spline_cache`). The sampler loads tt + pp for "I" (~4.5 GB) and tt / ee / te_rho / pp for "IP" (~9.1 GB); the lensed tables load only under `refresh_qe`, which `sample_joint` never sets. Unlensed BB returns zeros (NaN rows out of box).
- **`load_camb_grid_predictors(path)`** → `GridPredictors(tt, ee, bb, pp, tt_lensed, ee_lensed, bb_lensed, te_rho)`, each `f(params_batch) -> (M, n_ell)` via `jax.pure_callback(vmap_method = "sequential")`, cached per path. `LINEAR_SPECTRA = {"te_rho"}` skip the exp.
- **H0 vs theta_MC_100.** The grid is laid out in H0 because a rectangular theta box has CAMB-unsolvable corners. Every node records theta; the merge keeps the logA = ns = node-0 slice as `theta_grid(H0, ombh2, omch2)`. `CambGrid` puts the same tensor-product spline through it, and `h0_from_theta` evaluates theta(H0) on a 2000-point H0 line at the query's (ombh2, omch2) and inverts it with `np.interp`. The sampler works in theta_MC_100 throughout.
- (Measured 2026-09-22: the grid's dC/dtheta moves the forecast's marginalized sigmas by <= 0.26% vs direct CAMB at nside 128 / 2.5' / 5 uK, so grid interpolation does not explain chain-vs-forecast gaps.)
- `tests/test_camb_grid.py` (2026-10-19) checks the whole path (theta → H0 included) against direct CAMB at 5 seeded points; skipped without the grid; ~35 s, ~4.2 GB peak. Measured max |Δ lnCl|: tt 1.3e-3, ee 4.7e-3 (an EE trough near ℓ 470), tt_lensed 1.1e-3, ee_lensed 4.3e-3, bb_lensed 3e-4, pp 5e-5; |Δρ| te_rho 2.4e-3; medians 2e-5 to 2e-4.

### The Fisher forecast (`fisher_forecast.py`; measured delensing only + polarization since 2026-11-11)

One forecast, for data of polarization `pol` = "I", "P" or "IP": the central finite difference of per-mode covariance BLOCKS on the sampler's own rfft grid (`gen_ell_grid`), contracted as `F_ij = 1/2 sum_blocks sum_k w_k Tr[C_k^-1 d_i C_k C_k^-1 d_j C_k]` with `w_k = get_fourier_weights` (2 per bulk entry, 1 on the self-conjugate columns, summing to nside^2 - the weights `statistics.logpdf` is built from; for a 1x1 block the trace is `dln C/dtheta_i dln C/dtheta_j`; derivation in the module docstring):

```
pol "I"    [TT + N_TT]
pol "P"    [EE + N_EE], [BB + N_BB]
pol "IP"   [[TT + N_TT, TE], [TE, EE + N_EE]], [BB + N_BB]
all        [C_phi + N_phi]                          (N_phi frozen at theta_0)
```

`forecast(nside, theta_pix, noise_level, is_sampled, param_ground, delensed_covariance, pol = "I", qe_filter = "unlensed", phi_noise = None, include_bb = True, l_knee = 0, beam_fwhm = 0, l_cutoff, camb_lmax = CAMB_LMAX) -> (fisher, names)`.

- **The field blocks are ALWAYS measured**: `delensed_covariance` (a merged delensed_covariance.npz at this box, noise level and pol) gives their signal at theta_0 and theta_0 +/- h_i AND the steps h_i; the analytic noise (`N_EE = N_BB = 2 N_TT`, `POLARIZATION_NOISE_FACTOR`) is added in `field_blocks`. `delensed_covariance = None` raises `ValueError` (CLI: parser error). There is no CAMB delensing, no `t_block`, no `FD_STEP_FRAC`, no `--stability`.
- **N_phi** (`phi_noise_matrix`), built once at theta_0: the box QE N0 = `camb_spectra.qe_noise_matrix(cls, ..., pol, qe_filter)` - `load_sim`'s estimator without `NPHI_FAC` (`scalar_quadratic_estimate` for "I", `polar_quadratic_estimate`, the EB estimator, for "P" AND "IP"), response ALWAYS from the unlensed spectra, filter from the unlensed (default) or lensed (`qe_filter = "lensed"`) spectra; or with `phi_noise` the `"nphi"` of a merged map_joint_phi_noise.npz / sampler_phi_noise.npz (NaN modes carry no phi information). No `nphi_source` strings, no `qe_response`.
- **`include_bb`** (`--no_bb`): whether the delensed BB block is counted for "P" / "IP". At r = 0 it is pure residual lensing, neither Gaussian nor independent between modes, so counting it is optimistic.
- **`fisher_from_blocks(blocks_fid, blocks_plus, blocks_minus, steps, nside)`**: each block is an `(n, n, nside, nside//2+1)` numpy array (n = 1 or 2); per-mode `np.linalg.inv` + `einsum` trace. A mode whose centre block is not finite and positive definite carries no information (the zeroed origin, NaN N_phi modes). Verified 2026-11-11: a 2x2 block with TE = 0 equals the two scalar blocks to 2e-16, and a correlated 2x2 block equals a dense per-mode loop to 6e-16.
- **One parameter order.** Names, matrices, figures and the npz are in `constants.PARAM_ORDER` (`OUTPUT_PARAM_ORDER` is gone). The forecast looks the sampled names up BY NAME in the file's `names` (the file may hold more parameters; files merged before 2026-11-11 are in driver order).
- **`camb_lmax`** defaults to `CAMB_LMAX` (4000), the support of the simulations behind the delensed covariance. `camb_lmax_for_grid` is gone.
- **CLI**: `--delensed_covariance` (required), `--pol`, `--qe_filter {unlensed,lensed}`, `--phi_noise`, `--no_bb`, `--nside --theta_pix --noise --l_knee --beam_fwhm --params --camb_lmax`. Outputs in `cmb_lensing/fisher_output/` tagged `forecast_file_suffix(pol, nphi_label, noise_level)` = `_<pol>_<qe_<filter>|measured_nphi>_<noise:g>_uk`. The npz stores the run config.
- **`NPHI_FAC = 2`** is CMBLensing.jl's preconditioning heuristic; `qe_noise_matrix` does NOT divide by it (physical N^(0)). The box QE norm is the square-domain, periodically wrapped Hu & Okamoto N0 to 0.5%; its annulus scatter (1.6x at L = 540, 5.7x at L = 2090 on nside 64 / 5') is real anisotropy.
- **Verified 2026-11-11** (nside 128 / 2.5' / 2.5 uK, omch2 / theta_MC_100 / logA, T-only): the refactored forecast reproduces the pre-refactor Fisher matrices under all three N_phi options to 8e-17 relative. Sigmas (theta_MC_100 / logA / omch2): QE unlensed filter 0.002088 / 0.018878 / 0.001728; QE lensed filter 0.002164 / 0.018899 / 0.002021; map_joint N_phi 0.002177 / 0.018902 / 0.002068 (all with the `qe_filter_unlensed_response_unlensed` delensed covariance). The polarization path has only been exercised at nside 64 with 4 realizations; no P / IP chains exist to compare with.

What the forecast ignores: the blocks are added as independent (the residual lensing in the delensed field tracks C_phi(theta), so the field blocks count some of the phi block's lensing information), the field-phi correlations, and for "IP" a combined T + EB estimator (the N0 is the EB one, as in the sampler). None of this is the marginal Fisher of p(d | theta) that the sampler targets. Two facts from the deleted marginal-Fisher work worth keeping: **never `jax.grad` w.r.t. theta** - `phi_dot_wrapper`'s custom VJP (`statistics.py`) differentiates `dot(phi, Cphi)` rather than `dot(phi, Cphi^-1 phi)`, so any autodiff through `phi_covariance` is silently wrong; and the mixed change of variables has a theta-dependent Jacobian (`mixed_logpdf` subtracts `logdet(G) + logdet(D)`), so a mixed Hessian is not comparable to an unmixed one.

**Shared helpers moved out of the forecast (2026-11-11):**
- `cmb_lensing/camb_spectra.py`: `camb_cls_at_params` (direct CAMB through `simulate.camb_parameters`, memoized in `_CAMB_CACHE`; RAISES on failure), `load_sim_cosmology`, `covar_from_cl` (a C_l on the rfft grid via `_covar_or_zeros`, so the zero unlensed BB is safe), `instrument_matrices` (temperature noise, mask, beam), `qe_noise_matrix`, and the constants `QE_FILTERS = ("unlensed", "lensed")`, `DEFAULT_QE_FILTER`, `POLS`, `DEFAULT_POL`. Imported by fisher_forecast, delensed_covariance, map_joint_phi_noise, sampler_phi_noise, chain_analysis, phi_noise_analysis and the sample_joint tutorial.
- `cmb_lensing/matrix_plots.py`: `annotated_heatmap`, `plot_annotated_matrix` (fisher_forecast, chain_analysis).

### The empirical delensed covariance stencil (`delensed_covariance.py`, added 2026-09-24; polarization 2026-11-11)

The forecast's field blocks, value AND finite difference. Per seed (common random numbers) and per stencil point `{theta_0, theta_0 +/- h_i}` a job runs `load_sim(theta, pol)` → inverse-lenses the NOISELESS lensed field by the Wiener-filtered TRUE phi `W phi` (`W = C_phi / (C_phi + N_phi)` per mode, built once at theta_0; `delens` converts E/B to Q/U maps around `lense_flow` as `mixing.unmix` does) → stores `Re(X conj(Y)) / nside^2` on the rfft grid (covar_matrix_from_cls units, origin zeroed) for every component of `POL_COMPONENTS[pol]`: `TT` ("I"), `EE, BB` ("P"), `TT, TE, EE, BB` ("IP"). The merge is the realization mean at each stencil point.

- **W's N_phi** (recorded in every job file; the merge refuses to mix): the box QE N0 for that pol (`wiener_nphi_source = "qe"`, `qe_filter` "unlensed" (default) or "lensed"; no "delensed" filter and no `qe_response` since 2026-11-11), or a stored matrix (`phi_noise` path; `wiener_nphi_source = "file"`; `load_stored_nphi` reads `"nphi"`, NaN / negative modes → W = 0, and refuses another box).
- **Files.** Job: `delensed` of shape `(1 + 2k, n_components, nside, nside//2+1)` + `pol`, `components`, `names` (sorted into PARAM_ORDER by `stencil_points`), `steps`, `offsets`, `point_params`. Merged `delensed_covariance.npz`: `delensed_fid` `(n_components, ...)`, `delensed_plus` / `delensed_minus` `(k, n_components, ...)`, `components`, `pol`, `names`, `steps` (the jobs' stored steps, no longer recomputed from PARAM_SIGMA). `load_merged_covariance(path)` is what the forecast reads; a merged file WITHOUT `components` is a pre-2026-11-11 temperature file with no component axis and is still read. Job files written before 2026-11-11 (no `pol`) cannot be merged.
- Verified 2026-11-11: a pol "I" job at nside 64 with a fixed seed is bit-identical to the pre-refactor job for both QE filters.
- Driver: `get_delensed_field_covariance/get_delensed_field_covariance.sh` → `run_single_delensed_field_covariance.sh/.py`, merged by `merge_delensed_field_covariance.py` (job args: index, seed, nside, theta_pix, noise, l_knee, step_sigma, qe_filter, pol, phi_noise, out_dir, params from 12). `band_edges` / `DEFAULT_DELTA_ELL` live here (used by `phi_noise_analysis.py`, `plot_sampler_convergence.py`); `check_box` and `PARAM_RTOL` moved here from fisher_forecast. MEASURED ~21 s / 1.3 GB for a 3-point T job at nside 128 locally.

### The sampler phi noise (`sampler_phi_noise.py`, added 2026-10-06 as `sampler_effective_noise.py`, renamed 2026-10-15 and 2026-10-20; merge simplified 2026-10-20)

Several chains per (f, phi, d) realization (`num_chains` in the driver, default 2, any number works; since 2026-11-09) sample (f, phi) | d with theta HELD at GROUND_TRUTH, and the pooled posterior-mean phi is compared with the true phi. The chain is `fixed_theta_chain.posterior_mixed_draws` (`sample_joint`'s sweep minus the theta step; operators from `fixed_theta_chain.mixed_theta_matrices(data_set)`: the ground-truth data set's own C_f / C_phi / D and G rebuilt from load_sim's QE norm, for any pol (`run_job(..., pol = "I")`, driver `pol`; verified 2026-11-11 bit-identical to the old T-only construction), key `chain_key(map_seed, chain_seed)`), unmixing phi° by `pinv(G)`. `chain_seed` is `run_job(..., chain_seed = None)`'s argument (driver `chain_seeds="s0 s1 ..."`, one per chain index, reused for every map) or, when unset, `np.random.randint(0, MAX_CHAIN_SEED = 2**31)` like `sample_lcdm.sample_joint`; it is stored in the sidecar (`chain_seed`) and rerunning with it reproduces the chain exactly (verified 2026-11-10). The chain INDEX sets only the file name and the start: chain 0 at phi = 0, chain k > 0 at `rfft2(field_from_covar_single_key(nside, C_phi, fold_in(chain_root_key, PRIOR_START_STREAM)))` (over-dispersed starts). Two chains of one map with the same seed would be identical. No thinning, no split chains.

**The merge** (`merge_phi_noise(directory, burn_in)`; ONE user-set burn-in, in `merge_sampler_phi_noise.sh`) groups the chains by `map_seed`, trims a map's chains to the shortest post-burn-in length, and pools them: phi_mean = mean of the chain means, each projected onto its Hermitian part `rfft2(irfft2(.))`. Per mode the moments |phi_mean|^2, Re(phi_mean phi_true*), |phi_true|^2 (`map_joint_phi_noise.cross_power_grid`) are averaged over realizations BEFORE the ratio. **`nphi` uses the RESPONSE estimator** (`map_joint_phi_noise.noise_from_response`, `noise_with_jackknife(..., estimator = "response")`): r^2 = <Re(phi_mean phi_true*)> / <|phi_true|^2>, N = C_phi (1/r^2 - 1). It equals the correlation estimator (`noise_from_correlation`, r^2 = <Re(phi_mean phi_true*)>^2 / (<|phi_mean|^2><|phi_true|^2>), what map_joint uses) for the EXACT posterior mean, because <Re(posterior_mean phi_true*)> = <|posterior_mean|^2>, but never uses |phi_mean|^2, which a finite chain's Monte Carlo error inflates (the correlation estimator then reads that error as noise, biased high by ~(1 + N/C)/n_eff, worst at high L). It is not valid for a MAP estimate, so map_joint keeps the correlation estimator. A chain still climbing from phi = 0 shrinks phi_mean, which the response estimator reads as extra noise, hence the R-hat check. `sampler_phi_noise.npz` holds `nphi` (NaN where the mean cross is <= 0, 0 at the origin), `nphi_error`, `nphi_leave_one_out`, `rhat_real` / `rhat_imaginary` ((n_realizations, nside, nside//2+1) Gelman-Rubin R-hat across the map's chains of each mode's real / imaginary part, from per-chain means and variances via `mcmc_diagnostics.gelman_rubin_from_moments`; NaN at the origin and with one chain), `cphi`, the mean moments, `burn_in`, `n_chains`, `n_used` (post-burn-in sweeps per chain), seeds and the box / cosmology keys. `plot_sampler_convergence.py` averages R-hat over the real and imaginary parts of each |L| band's modes per map (`delensed_covariance.band_edges`) and plots the mean over maps with +/- 1 std over maps, reference line 1.01. **Self-conjugate columns are measured, not QE-filled** (the ratio of expectations is as valid there as in the bulk); the Hermitian projection removes the anti-Hermitian content the Fourier-space phi HMC can put on exactly those columns (see "The Phi Gradient in Fourier Space"), and is the identity elsewhere.

**Storage is a live-readable memmap:** `sampler_phi_NNNN_cK_samples.npy` (realization NNNN, chain K; (n_sweeps, nside, nside//2+1) complex64), preallocated (`rsync --sparse` to copy a partial run), flushed every 10 sweeps, then the sidecar `sampler_phi_NNNN_cK.npz` (`n_done`, `finished`, `chain_index`, ...) is rewritten atomically, so the merge runs on RUNNING jobs (a chain not yet past the burn-in is left out). MEASURED nside 64 / 5': ~1.0-1.35 s per sweep after ~25 s compile, 1.35 GB RSS. The merge reads every chain once (`merge_sampler_phi_noise.sh`). Directories written before 2026-11-09 (`sampler_phi_NNNN.npz`, one chain per map) still merge, with NaN R-hat.

### Other measurement modules

- **`map_joint_phi_noise.py`** (renamed from `phi_noise.py` on 2026-10-15; per-mode since 2026-10-19) — map_joint's per-MODE N_phi = C_phi (1/r^2 - 1) on the rfft2 grid (`measure_phi_noise/get_map_joint_phi_noise/`), always at GROUND_TRUTH. Each job (one seed: load_sim → map_joint) stores the per-mode moments `auto` = |phi_hat|^2, `cross` = Re phi_hat phi*, `true_auto` = |phi|^2 (/ nside^2, covar_matrix_from_cls units) and `cphi`. The merge averages them over realizations BEFORE forming r^2 = <B>^2/(<A><D>), and writes `map_joint_phi_noise.npz` with `nphi` (NaN where the mean cross is <= 0, 0 at the origin), a delete-one jackknife `nphi_error`, the N leave-one-out N_phi `nphi_leave_one_out` (read by chain_analysis.py for the jackknife of the band median), the mean moments and the box / cosmology keys, so it works directly as the forecast's `--phi_noise` and as delensed_covariance's `phi_noise`. No band averaging, no QE reference, no validation rungs, no cosmology shift, no `qe_response` / `beam_fwhm` (all removed 2026-10-19). The only place map_joint's reconstruction error is measured (the delensed-covariance path no longer runs map_joint).
- **`fixed_theta_chain.py`** (renamed from `fisher_forecast_from_mixed_logpdf.py` on 2026-10-15, when its mixed-Hessian and Louis forecast were deleted) — only what `sampler_phi_noise.run_job` needs: `mixed_theta_matrices(data_set)` (the data set's own C_f / C_phi / D, and G = `get_g_matrix_lcdm(C_phi, QE)`; any pol), `posterior_mixed_draws` (the fixed-theta Gibbs sweep, streamed to an `on_draw` callback) and `chain_key(map_seed, chain_seed) = fold_in(fold_in(PRNGKey(map_seed), CHAIN_STREAM), chain_seed)` (since 2026-11-10; every chain is its own branch and nothing draws from the stream itself). `posterior_mixed_draws(..., phi_start = None)` starts at phi = 0 unless given a start field. CHAIN_STREAM must not be 0 or 1: load_sim's keys all descend from `split(PRNGKey(map_seed))` = `fold_in(PRNGKey(map_seed), 0 | 1)` under partitionable threefry (`split(k, n)[i] == fold_in(k, i)`, VERIFIED 2026-11-10 on JAX 0.12), so a chain rooted at `PRNGKey(map_seed)` itself would reuse them. For the same reason never root a chain AT a key another chain's `split` descends from: the 2026-11-09 scheme (chain 0 at the stream, chain k at `fold_in(stream, k)`) gave chain 1 chain 0's sweep-1 f-draw key; removed, and pre-2026-11-10 chains are not reproducible under the current `chain_key`.
- **`mcmc_diagnostics.py`** (2026-11-09) — `gelman_rubin(chains)` / `gelman_rubin_from_moments(chain_means, chain_variances, n_samples)` (R-hat across chains, no splitting, elementwise over trailing axes), `autocorrelation`, `integrated_autocorrelation_time` (Sokal window). Moved out of `chain_analysis.py`, which imports them.

### Polarization Modes

Three configurations control field dimensions and covariance structure, in `load_sim`, `map_joint` and `sample_joint` alike:
- **I** (intensity only): `FlatS0` fields, `DiagonalScalar` operators, `DataSetT`
- **P** (polarization only): `FlatS2` fields (E/B), `DiagonalEB` operators, `DataSetEB` (phi is still a `FlatS0`). At r = 0 this is effectively E-only information with no TE lever — expect much wider theta posteriors than "IP" (ombh2 especially weak).
- **IP** (both): `FlatS02` fields, `BlockTEB` operators (TT/TE/ET/EE/BB, mask and beam with zero TE blocks), `DataSetTEB`. EE's acoustic structure plus the physical TE correlation break the ns–ombh2 degeneracy T-only leaves (marginal corr ~ −0.76), at a considerably higher cost per sweep.

### Removed features (do not assume they exist)

- **2026-11-11 (rotated-coordinate theta sampling, `reparameterize_lcdm`).** Last commit with it: **`b50d4cc`** (`git show b50d4cc:cmb_lensing/sample_lcdm.py`: `prepare_rotation`, `rotation_from_covariance`, `metropolis_sample_rotated_theta`, `gibbs_sample_rotated_theta`, and the wiring in `sample_joint` / `__main__`). Removed with them: `sample_joint`'s `reparameterize_lcdm`, `rotation_matrix`, `rotated_proposal_sigmas` arguments and `fisher_forecast.forecast_npz_path`. How it worked, should it be rebuilt:
  - **The rotation.** `rotation_from_covariance(covariance, names, sampled_names, sigma_scale = 2.4)` took an estimate of the posterior covariance (a Fisher forecast's `fisher*.npz` `"covariance"` / `"names"`, or a pilot chain's), restricted it to the sampled parameters as the CONDITIONAL covariance `inv(precision[sub, sub])` (unsampled parameters are frozen by the chain, not marginalized, so the plain sub-block would be too wide and differently oriented), eigendecomposed its CORRELATION matrix (the parameters span orders of magnitude, so the covariance itself is too ill-conditioned), and returned `R = eigenvectors.T / sigmas` with rows ordered widest direction first, so `theta' = R @ theta` is whitened and decorrelated, plus proposal widths `sigma_scale * sqrt(eigenvalue)` keyed `rot_0`, `rot_1`, ... (2.4 sigma is the optimal 1D random-walk scale, ~44 % acceptance). `R` is `(N, N)` over the N sampled parameters in `PARAM_ORDER`; `prepare_rotation` checked that shape and returned `pinv(R)`.
  - **The step.** Per sweep, one 1D symmetric Gaussian random-walk Metropolis move per rotated axis: perturb component i of `theta'` by `sigma_i * normal`, map back with `theta = pinv(R) @ theta'`, REJECT if any original parameter leaves its `param_ranges` box (the flat prior is on the original parameters, checked before the interpolator is touched), else accept if `log(u) < mixed_logpdf(prop) - mixed_logpdf(current)` (NaN rejects). Symmetric proposal and a linear map, so no Jacobian. After an accept `theta'` was recomputed as `R @ theta` to keep the pair exact. The logpdf batch was `make_eval_logpdf_batch` evaluated on `current_params` with the sampled entries replaced (`params_batch.at[:, sampled_idx].set(...)`).
  - **The wiring.** A rotated move changes EVERY sampled parameter, so after each axis all of them were written into `current_params` (the next axis conditions on it) and after the sweep all were appended to `param_vals`; the per-parameter loop was skipped; `update_args_after_sample` ran once per sweep as now; acceptance lists were kept per `rot_i`. It needs a Cl source that can move several parameters at once (the 5D grid).
  - **When rebuilding:** the forecast now writes its covariance in `PARAM_ORDER` (no reordering needed) under `fisher_forecast.forecast_file_suffix(pol, nphi_label, noise)`, and at `b50d4cc` `__main__` looked for a file name the forecast never wrote; `chain_analysis.metropolis_acceptance_rate` counts changed values per sweep, which is not the Metropolis rate under rotated moves.

- **2026-11-11 (the 1D CAMB splines).** Last commit with them: **`b50d4cc`** (`git show b50d4cc:camb_grid_generation/precompute_camb_1d.py`, and `sample_lcdm.py` at that commit for the loader). Removed: `camb_grid_generation/precompute_camb_1d.py` (a 50-node sweep of one parameter over `PARAM_BOUNDS` with the other four at `GROUND_TRUTH`, cubic splines in lnCl for TT and PP only, written to `camb_splines/camb_<param>_grid.npz`), and in `sample_lcdm.py` `USE_CAMB_SPLINE`, `USE_CAMB_GRID`, `get_camb_spline_predictors`, `_load_precompute_camb_1d` (import by file path), `wrap_temp_predictors` (a `GridPredictors` with `None` for every non-TT/PP spectrum) and every branch on the switches; `constants.CAMB_SPLINE_FILES` (now `CAMB_GRID_FILE`). It served one sampled parameter, temperature only. The five cache files are still in the local `camb_splines/` and on the Hugging Face dataset, unused. Two facts from it worth keeping: the shipped theta_MC_100 cache lost 8 of 50 nodes and ended at 1.114375 (+3.1 sigma) because theta_MC_100 >~ 1.117 needs H0 > 100, outside CAMB's default `theta_H0_range` (`_run_camb` now passes `DEFAULT_THETA_H0_RANGE = [10, 150]`); and Cls must live on ells 2..`DEFAULT_MAX_ELL`-1 (`CAMB_LMAX`), the exact support of `load_sim`'s CAMB path, so model and data share the same log-log extrapolation anchor (a one-multipole mismatch measurably biased the ombh2 conditional), which still applies to the 5D grid. `camb_grid.sh`'s node counts per axis were chosen from the spline error measured on those 1D sweeps.

- **2026-11-11 (CAMB delensing and the forecast's option matrix).** The last commit with all of it is **`b50d4cc`** (on `origin/master`): `git show b50d4cc:cmb_lensing/fisher_forecast.py` (the 1030-line module) and `git show b50d4cc:get_lcdm_samples/chain_analysis.py`. Removed from `fisher_forecast.py`:
  - **Analytic delensing:** `delensed_cls_at_params(params, alens)` = CAMB's `results.get_partially_lensed_cls(alens, lensing_method = 3)` (the non-perturbative correlation-function lensing with C_L^phiphi scaled per L by `Alens_L`; NOT an interpolation between unlensed and lensed), `delensing_alens` (`Alens_L = N_L / (C_L + N_L)`), `radial_cl_profile` (the azimuthal, geometric-mean annulus average of a per-mode N_phi that gives N_L), `frozen_reconstruction`, `delensing_efficiency`, `camb_results_at_params`, and `t_block = "unlensed" | "lensed" | "delensed"` (`T_BLOCKS`, `temperature_cl_at_params`). The old default forecast was `forecast(..., nphi_source = "qe_delensed", t_block = "delensed")`.
  - **Iterated QE filter:** `iterative_delensing` (N_L → Alens_L → partially lensed C_TT → filter, 5-6 iterations to 1e-6), `qe_reconstruction`, the `qe_delensed` source (the LOWER information bound; `qe_unlensed` was the UPPER one) and `delensed_covariance.py`'s `qe_filter = "delensed"`.
  - **Gradient response:** `gradient_cls_at_params` / `delensed_gradient_cls_at_params` (CAMB `get_lensed_gradient_cls`, the lensed T-grad-T of Lewis, Challinor & Hanson 2011), `--qe_response gradient`, `cls_with_qe_response`, `qe_response_cl`. Measured at nside 64 / 5' / 5 uK: it raised N_phi 0.4-2.6% and moved the sigmas < 0.35%.
  - **Other:** the `nphi_source` strings (`NPHI_SOURCES`), `OUTPUT_PARAM_ORDER`, `FD_STEP_FRAC` / `step_fracs` / `--stability` / `step_stability`, `camb_lmax_for_grid` (the grid's corner, 6110 at 2.5'; power-law continuation past CAMB's 3999 underestimates the damping tail, 2.5x at the corner), `prior_covariances`, `same_cosmology`, `interpolate_spectrum`, `load_phi_noise_matrix`, `sampled_names_and_steps`; `delensed_covariance.py`'s `band_average`, `leave_one_out_grids`, `forecast_grids`; `phi_noise_analysis.py`'s delensed-filter QE curve; `chain_analysis.py`'s `t_block` / `phi_noise_source` / `qe_response` options and its R-hat map pruning.
  - **Numbers measured with the removed code** (2026-10-14, nside 128 / 2.5' / 5 uK, omch2 / theta_MC_100 / logA, CAMB T block, `camb_lmax = 4000`), s(omch2) / s(theta) / s(logA) / r(om, th): qe_unlensed 0.0028 / 0.0033 / 0.0219 / +0.19; qe_delensed + gradient 0.0030 / 0.0034 / 0.0221 / +0.23; map_joint 0.0030 / 0.0033 / 0.0219 / +0.24; sample_joint 0.0030 / 0.0033 / 0.0218 / +0.23. With the empirical T block: qe_delensed / gradient 0.0028 / 0.0033 / 0.0219, r +0.27. The bracket was an ordering of the marginalized sigmas, not of the matrices (the delensed T block LOSES information as N_phi drops).

- **2026-10-19 (5D grid simplification):** `camb_grid_interp._TensorSpline`, `value_and_grads`, `CambGrid.cl_local` / `coeff_memmap` / `_local_support` (few-point low-memory evaluator), `has_spectrum`, the missing-spectrum predictors, and every check in `merge_camb_grid.py`; `camb_grid_generation/validate_camb_grid.py` (now `tests/test_camb_grid.py`), `tests/test_native_5d_interp.py`, `tests/handrolled_bspline_reference.py`, `tests/benchmark_native_5d_interp.py`.

- **2026-10-15:** the mixed-logpdf Hessian forecast and Louis information (`forecast_from_mixed_logpdf`, `mixed_hessian_realization`, `louis_*`, `stencil_offsets`, `hessian_from_stencil`, the `--louis` CLI, the hessian-directory loaders) and `chain_analysis.py`'s Louis / realization-averaged-forecast plumbing (`LOUIS_*`, `louis_file_label`, `leave_one_out_covariances`, the ceiling / lensed styles and bracket). The module survives as `fixed_theta_chain.py`.

- **CAMB emulator** (`cambemul`, `camb_emulator/*.npz`, `load_sim(use_emulator_cls = ...)`) — removed because emulator-predicted Cls gave theta conditionals inconsistent with real CAMB spectra.
- **`sampling_ar.py`** (A_phi / r sampler port of `sampling.jl`, `grid_and_sample`, `loess`-based inverse-CDF theta step, `AR_KEYS`) and **`sample_lcdm_legacy.py`**. `util.loess` and `rk4_solve` remain in `util.py`.
- **Theta samplers:** grid inverse-CDF, PCA eigen-direction Metropolis (`use_pca`/`save_pca`/`theta_pca.npz`), plain joint MH (`use_joint_mh`), GHMC/NUTS with finite-difference gradients (`use_ghmc_theta`), the Cholesky whitening, `THETA_PRIORS`, `fixed_field_theta`, over-relaxation. Only per-parameter random-walk Metropolis survives. One hard-won fact worth keeping: **`jax.grad(mixed_logpdf)` w.r.t. theta through the G mixing matrix was measured ~3000× wrong with a flipped sign** (cotangent path through `unmix` and the hand-written lensing adjoint; root cause never found), so any future gradient-based theta update needs finite differences or a fixed adjoint.
- **Differentiable grid predictors** (`load_camb_grid_predictors_jax`, `load_camb_grid_predictors_grad`) and `tests/test_camb_grid_interp.py`.
- `constants.py`: `AR_KEYS`, `EMULATOR_MAX_ELL` (→ `DEFAULT_MAX_ELL`), the diffrax PID constants (and the `diffrax` dependency). `map_joint.py`: the unused jitted `line_search`.
- `docs/tutorial.ipynb` / `docs/debug.ipynb` → replaced by `docs/map_joint_tutorial.ipynb`. All the `cmb_lensing/*.png` chain diagnostics and the `scratch_*.log` files were deleted from git.
- **2026-10-14:** the forecast siblings (`cls` with `lensing_covariance.py` and `tests/test_lensing_covariance.py`, `full_sky`, `logpdf`, `1st_principles` with the Hu & Okamoto `qe_noise_spectrum`), the transfer function (`delensed_spectrum.py`, `--transfer_function`), the score bound (`sampler_noise_estimate.py`, `--nphi_source score`, `--map_joint_phi_noise`), the forecast's `--spectra lensed|unlensed|ceiling` modes, `--nphi_source covariance|hu_okamoto|measured`, `--vary_nphi`, `--iterative_delens`, `--empirical_phi_noise|_block`, `--freeze_phi_noise`, `--radial_mean arithmetic`, the grid Cl source (`grid_cls_at_params`), the marginal Fisher (removed 2026-09-10), and the `sampling_chains_TEMPLATE/` / `runtime_comparison_TEMPLATE/` directories.

### The Phi Gradient in Fourier Space (Reproducing Julia's Anti-Hermitian Nyquist Content)

This section documents a subtle but important correctness issue in the **mixed phi gradient** (`mixed_grad_phi_logpdf` in `gradients.py`, which drives the HMC `symplectic_integrate` step above) and the rework that fixed it. Read this before touching `gradients.py`, `lense_flow.py`'s gradient path, `statistics.py:logpdf`, or `fields.py:undo_inner_product`. It is easy to "simplify" this code back into the broken state because the broken version *looks* more natural.

#### The symptom

When reproducing the Julia chain, the mixed phi gradient (`gradient.scalar_matrix`) agreed with the Julia `mixed_phi_gradient_*` dumps everywhere **except** the first column (`[:, 0]`, kx = 0) and the last column (`[:, -1]`, kx = Nyquist) of the rfft array. There the fractional difference was ~2% (`2.07e-2` overall, dominated by those two columns). The error had a tell-tale symmetry: along each of those columns, the **real part of the error was odd** and the **imaginary part was even** under row-reversal (`n -> (N-n) mod N`) — i.e. the error was *anti-Hermitian*, the opposite symmetry of the (Hermitian) gradient itself. Compounded over the leapfrog steps, this shifted the sampled phi enough to matter; substituting the Julia gradients into the integrator dropped the phi disagreement by orders of magnitude, confirming the gradient — not the integrator — was the culprit.

#### The root cause: invisible imaginary degrees of freedom that the lensing actually uses

Fields are stored as a real FFT (`rfft2`) half-plane of shape `(N, N//2+1)`. The last axis (the rfft axis, length `N//2+1`) is the "half" axis. Two of its columns are **self-conjugate**: column `0` (kx = 0) and column `-1` (kx = Nyquist), because `-0 ≡ 0` and `-Nyquist ≡ Nyquist (mod N)`. For these two columns, the reality of the underlying map forces Hermitian symmetry *within the column* along the full (row) axis, which means the imaginary parts of the DC-row and Nyquist-row entries are structurally redundant.

Concretely, the key fact (1-D intuition, length-4 real signal): its rfft has coefficients `F[0]` (DC), `F[1]`, `F[2]` (Nyquist). For a **real** signal `F[0]` and `F[2]` are purely real. Crucially, **`irfft` ignores the imaginary parts of `F[0]` and `F[2]`** — feeding `irfft` an array with `F[2] = a + ib` produces the exact same real signal as `b = 0`. Those imaginary parts are *invisible* to `irfft`.

But the lensing does **not** use phi directly — it only uses its derivatives `∇phi`, and in Fourier `∇ = i·ℓ`. Multiplying by `i·ℓ` *mixes real and imaginary parts*:

```
i·ℓ_nyq · (a + ib)  =  i·ℓ_nyq·a  −  ℓ_nyq·b
```

The `−ℓ_nyq·b` term is a **real** contribution to the derivative. So `∇phi` genuinely depends on the otherwise-invisible imaginary part `b`. Therefore the log-likelihood depends on `b`, and `∂(logpdf)/∂b ≠ 0`. That nonzero derivative — living in the imaginary DC/Nyquist degrees of freedom of the two self-conjugate columns — **is** the "anti-Hermitian content" Julia's gradient carries. It is genuine, not a numerical artifact.

This is also why the bug was confined to exactly two columns. For any *other* column `kx`, the conjugate partner `-kx` lives in the dropped half of the rfft, so `Im(phi[:, kx])` is just an ordinary free complex DOF that ordinary autodiff already handles correctly. Only kx = 0 and kx = Nyquist are their own conjugate partner, so only there does an "invisible" (to `irfft`) imaginary DOF exist. A per-mode comparison against Julia confirmed every other mode matched at ratio `1.0` exactly — the discrepancy was surgically localized.

#### Why the old Python code lost it — in two independent places

The pre-fix code destroyed `b` twice, and **both** had to be repaired (fixing only one is insufficient):

1. **Forward pass.** `statistics.py:logpdf` did `phi = map(phi)` (i.e. `irfft2`) *before* calling the lensing. That zeroed `b` immediately — the likelihood never depended on it, so `∂(logpdf)/∂b = 0` by construction. No amount of clever adjoint work can recover a derivative that is structurally zero in the forward model.

2. **Adjoint pass (the `delta_phi` ODE).** The hand-written lensing adjoint (`get_lensing_operator_gradients` → `get_delta_phi_tqu_roc` in `lense_flow.py`) built `d_delta_phi_dt` with `get_primal_derivatives`, which ends each derivative with an `irfft2`. So the accumulated `delta_phi` came out as a **real-space** array; its `rfft2` has zero imaginary DC/Nyquist, discarding `b`'s gradient again. (This is the part most people guess at — and it is real — but it is only half the story; see #1.)

Equivalently: `irfft2` is a projection onto "real-field land," where `b` does not exist. Any time phi (or its gradient) passes through `irfft2`, `b` is annihilated. Reproducing Julia requires keeping phi in **Fourier** throughout the entire gradient path so that `i·ℓ` can act on `b` before any `irfft2`, and so the adjoint accumulates `b`'s gradient without an intervening `irfft2`.

#### The fix (all changes preserve the forward lensed-field value bit-for-bit)

The forward lensed value is unchanged because, for a Hermitian phi, computing derivatives directly from the Fourier array gives the identical real-space derivatives as `irfft2(phi)` then re-`rfft2` — the round trip is a no-op on the Hermitian part. Only the *gradient* changes.

- **`util.py`** — `get_k_meshgrid` (shared `(KX, KY)` builder, with the negative-Nyquist `ky` convention — see below), `get_primal_derivatives_from_fourier(phi_fourier, pix_width)` (derivatives computed directly from an rfft2 array — same values as `get_primal_derivatives(irfft2(...))` but keeps `i·ℓ` in the autodiff graph), and `get_primal_derivatives_to_fourier(field, pix_width)` (applies the derivative operators but returns the result *in* rfft2 space, with no final `irfft2`).
- **`lense_flow.py`** — `get_lensing_operator_gradients`, `lensing_gradients_integration_step`, and `get_delta_phi_tqu_roc` are **basis-aware**. They branch on whether `phi.scalar_matrix` is square (MAP, real-space) or rectangular (FOURIER) via `shape[0] != shape[1]`, threaded through the RK4 loop as a static Python bool `phi_fourier`. When `phi_fourier` is true: phi-derivatives come from `get_primal_derivatives_from_fourier`; `delta_phi` is initialized as a complex `(N, N//2+1)` array and **accumulated in Fourier space**; and the final divergence/laplacian operators that build `d_delta_phi_dt` use `get_primal_derivatives_to_fourier` (no closing `irfft2`). This mirrors CMBLensing.jl's `negδvelocityᴴ` (`lenseflow.jl`), which accumulates `δϕ` via `-∇'·Ð(...)` in Fourier. `delta_phi` never feeds back into another rate of change, so it can live at a different shape/dtype than the (square, real) `t/q/u/delta_t/...` state inside the RK4 tuple. The two inner integration functions are intentionally **not** `@jax.jit`'d (they run inside the already-jitted `get_lensing_operator_gradients`, and `phi_fourier` must stay a Python bool to select dtype/shape).
- **`lense_flow.py:primal_lense_flow`** — also branches on phi shape so the forward lensing accepts a Fourier phi (same value, different autodiff graph). Forward-only callers (`mixing.py`, `simulate.py`, `gibbs_sample_f`, `gradf_logpdf`) still pass `map(phi)` (square, MAP) and therefore hit the unchanged real-space path — no behavior change for them.
- **`statistics.py:logpdf`** — no longer does `phi = map(phi)`; it passes the **Fourier** phi straight into `lense_flow_wrapper`. (The `phi_dot_wrapper` prior term already used the Fourier phi, so it is unaffected.)
- **`gradients.py:mixing_jacobian_phi_component`** — differentiates with respect to the **Fourier** phi (`jax.vjp(unmix_partial, phi)`, returning the cotangent directly with no `fourier(differential)` re-wrap), AND uses `lense_flow_wrapper` (the custom analytic-adjoint VJP) instead of plain `lense_flow`. See the critical subtlety below.
- **`fields.py:undo_inner_product`** — a rectangular (Fourier) gradient is passed through unchanged. The original real-space body — `irfft2(conj(rfft2(m)/fourier_weights) * nside**2)` — composed with the implicit `irfft2`-Gram / `map` VJP that followed it, and that composite cancels to the identity on the bulk for a gradient that is already in Fourier; applying the real-space body to a Fourier array would instead corrupt it and re-symmetrize away the anti-Hermitian content.

#### The critical subtlety: analytic adjoint vs. autodiff-through-the-solver

There are two terms in the mixed phi gradient: the **data term** (flows through `logpdf`'s `lense_flow_wrapper`, which has the hand-written analytic adjoint) and the **f-prior chain-rule term** (`mixing_jacobian_phi_component`). The f-prior term originally used **plain autodiff through the RK4 ODE** (`jax.vjp` of the un-wrapped `lense_flow`). Julia uses an **analytic continuous-adjoint ODE** (`negδvelocityᴴ`), and autodiff-through-the-discretized-solver differs from that analytic adjoint by the ODE discretization error — with only ~7–10 RK4 steps this is ~15%, which completely swamps the ~2e-4 agreement we are chasing and corrupts the *bulk* (Hermitian) modes, not just the two columns. The fix routes the chain-rule term through `lense_flow_wrapper` too, so **both** terms use the same analytic adjoint Julia uses. This is why, after the fix, the bulk stays exact (ratio 1.0) *and* the anti-Hermitian content appears: do not "simplify" `mixing_jacobian_phi_component` back to plain `lense_flow`/autodiff.

#### The Nyquist sign convention (`get_primal_derivatives`)

`get_k_meshgrid` sets the half-axis Nyquist wavenumber **negative** (`ky = ky.at[-1].set(-1*ky[-1])`), matching CMBLensing.jl's `ifftshift(-N÷2:(N-1)÷2)` construction (numpy's `rfftfreq` would make it positive). On its own this sign is nearly invisible to the *forward* derivatives (the Nyquist column's anti-Hermitian content is discarded by `irfft2` either way). But it sets the **sign** of the `i·ℓ` adjoint's anti-Hermitian content on the Nyquist row, so once that content is preserved (above) the sign must match Julia. Keep it.

#### Result and how to re-validate

After the rework the mixed phi gradient matches the Julia `mixed_phi_gradient_*` dumps to ~`2.2e-4` across the entire leapfrog trajectory (down from `2.07e-2`), the forward field value is unchanged (`F` fractional difference `~4.8e-6`), and all gradient/logpdf/lensing/map_joint/wiener regression tests pass. To re-validate per-mode against Julia at a *known* input (not just the trajectory dumps), reconstruct the exact dataset in Julia from the operator dumps and call `gradient(ϕ° -> logpdf(Mixed(ds); f°, ϕ°), ϕ°)` — covariances/mask/beam/mixing must be wrapped as **real-valued** `FieldOp`s, `G` is not dumped (rebuild it from `Nphi`/`Cphi0` exactly as `get_g_matrix` does), and the dumps are stored transposed relative to Julia's `(Ny÷2+1, Nx)` layout. Julia and JAX share FFTW conventions (unnormalized forward `rfft`, `1/N²` inverse), so no extra FFT normalization factor is needed.

### Key Design Patterns

**Flax pytree dataclasses**: Fields (`FlatS0`, `FlatS2`, `FlatS02`) and matrix operators (`DiagonalScalar`, `DiagonalEB`, `BlockTEB`) use `@flax.struct.dataclass` so they work as JAX pytree leaves — passable through `jit`, `grad`, `vmap`. Arithmetic is overloaded element-wise per matrix name; `BlockTEB.__mul__`/`__rmul__` implement the non-commutative 2×2 T/E block product; `pinv` / `log_det` are `singledispatch`ed per operator type (`BlockTEB` via Schur complements in `util.py`).

**Custom VJP on lense_flow**: `lense_flow_wrapper` has a hand-written backward pass (`lense_flow_backwards`) that integrates the adjoint ODE in reverse, rather than relying on JAX's default autodiff through the ODE solver. This analytic adjoint (not autodiff-through-the-solver) is required for the phi gradient to match Julia — and its `delta_phi` accumulation is basis-aware (real-space vs Fourier). See *The Phi Gradient in Fourier Space* above. The ODE solver is a constant-step RK4 (`util.rk4_solve`, port of CMBLensing.jl's `OutOfPlaceRK4Solver`), `n = 12` steps by default (`n = 10` in `load_sim`, `gibbs_sample_f`, `mixing_jacobian_phi_component`).

**Basis and parametrization switching**: Fields carry a `basis` (MAP = real space, FOURIER) and are implicitly in a parametrization (T, QU, or EB). Conversion helpers `map()`, `fourier()`, `qu2eb()`, `eb2qu()` are used extensively — gradient code manually converts between representations before and after lensing. Note that `fields.map` shadows the Python builtin inside every wildcard-importing module. `fields.map` / `fields.fourier` have **no guardrails** — they apply `irfft2` / `rfft2` unconditionally, so `fourier` on an already-FOURIER field raises "only real valued inputs supported for rfft"; check `field.basis` first where fields arrive from mixed sources (`delensed_covariance._to_fourier` / `_to_map`, `sampler_phi_noise._to_fourier`).

**RNG keys** (reworked 2026-10-15): `load_sim` starts from one root key `PRNGKey(master_seed)` and every random field splits its own subkey off it in a fixed order — phi, then the unlensed field(s) (T, or T_indep / E / B), then the noise — via `field_from_covar(nside, covar, rng_key) -> (field, rng_key)`. Because the order and the white noise do not depend on the cosmology, two calls at one seed share every white-noise draw (the common random numbers `delensed_covariance.py` relies on; verified exact to 4e-14). Do not go back to pre-splitting a batch of keys. Chains that reuse a map seed must root their keys off `fold_in(PRNGKey(seed), CHAIN_STREAM)` (`fixed_theta_chain.chain_key`), never `PRNGKey(seed)` itself.

**Random fields**: `field_from_covar_single_key` draws `irfft2(rfft2(white) * sqrt(C))` from REAL white noise so the two self-conjugate rfft columns get full (not half) power — the earlier per-mode complex draw halved their variance and biased every sampled field, noise realization and HMC momentum.

**CAMB through `pure_callback`**: `_camb_via_callback` wraps `_run_camb` so `load_sim` stays traceable; failures return NaN arrays. The 5D grid predictors use the same pattern. The forecast instead calls CAMB directly (`fisher_forecast.camb_results_at_params`, via `simulate.camb_parameters`) and RAISES on failure.

**Wildcard imports throughout**: Modules use `from cmb_lensing.util import *`, `from cmb_lensing.lense_flow import *`, etc.

**Atomic checkpoints**: the long HPC jobs write their npz through a hidden temp file + `os.replace` with a `finished` flag, and their merges set unfinished files aside. The short delensed-covariance jobs just write once at the end. Every merge refuses mixed configurations and duplicate seeds.

### Module Dependency Graph

```
simulate.py ─────► util.py (FFT, derivatives, coordinate grids, RK4, loess)
  │                lense_flow.py ──► fields.py (FlatS0/S2/S02 dataclasses)
  │                dataset.py       constants.py
  │                statistics.py

map_joint.py ─────► gradients.py ──► lense_flow.py
                    wiener_filter.py
                    statistics.py
                    mixing.py ──► mix/unmix into mixed parametrization

sample_lcdm.py ──► simulate.py (load_sim, covariance/D/G/QE builders)
                   map_joint.py, wiener_filter.py, mixing.py
                   gradients.py ──► mixed_grad_phi_logpdf (phi HMC)
                   statistics.py ──► mixed_logpdf (theta Metropolis)
                   constants.py ──► PARAM_ORDER / GROUND_TRUTH / PARAM_BOUNDS / PARAM_SIGMA / CAMB_GRID_PATH
                   camb_grid_interp.py ──► 5D grid predictors (default Cl source)

camb_spectra.py ──► simulate.py (camb_parameters, _extract_all_cls, noise / beam / mask, the two QE norms)
  ▲                 direct CAMB cls, covar_from_cl, instrument_matrices, qe_noise_matrix
  ├── delensed_covariance.py ──► load_sim, lense_flow (delens), fields (eb2qu / qu2eb)
  ├── map_joint_phi_noise.py ──► map_joint
  ├── sampler_phi_noise.py ──► fixed_theta_chain (posterior_mixed_draws, mixed_theta_matrices, chain_key),
  │                            map_joint_phi_noise (cross_power_grid, noise_with_jackknife, stored_pol),
  │                            mcmc_diagnostics (gelman_rubin_from_moments)
  └── fisher_forecast.py ──► delensed_covariance (load_merged_covariance), matrix_plots

fixed_theta_chain.py ──► mixing, simulate (get_g_matrix_lcdm), sample_lcdm (gibbs_sample_f / gibbs_sample_phi, imported at call time)
matrix_plots.py ──► matplotlib only

camb_grid_generation/{run_single_camb_grid, merge_camb_grid}.py ──► the 5D grid (checked by tests/test_camb_grid.py)
get_lcdm_samples/run_single_lcdm_chain.py ──► sample_lcdm.sample_joint
cmb_lensing/chain_analysis.py ──► fisher_forecast.forecast, matrix_plots, camb_spectra (constants), mcmc_diagnostics (reads get_lcdm_samples/lcdm_sampler_data/<run>/)
get_delensed_field_covariance/*.py ──► delensed_covariance
measure_phi_noise/phi_noise_analysis.py ──► camb_spectra (QE noise), fisher_forecast (fourier_weights), delensed_covariance (band_edges)
measure_phi_noise/get_sampler_phi_noise/*.py ──► sampler_phi_noise
measure_phi_noise/get_map_joint_phi_noise/*.py ──► map_joint_phi_noise
runtime_comparison/python_performance_test.py ──► map_joint (Julia data via juliacall)
```

`cmb_lensing/__init__.py` imports the core modules only — not `sample_lcdm`, `camb_grid_interp` or any forecast / measurement module.

### Testing Approach

Most tests are validation benchmarks comparing Python output against Julia (CMBLensing.jl) ground truth stored in `tests/ground_truth_data/*.npz`. They cover `load_sim`, lensing, logpdf, gradients, the Wiener filter and `map_joint`; **nothing tests the sampler**. Exceptions to the Julia pattern: `tests/test_camb_grid.py` checks the real 5D grid against direct CAMB (skipped when the grid file is absent). The forecast, `delensed_covariance` and `sampler_phi_noise` are untested (open review findings are listed in `CODE_REVIEW_FINDINGS.md` at the repo root). Many Julia tests produce comparison plots in `tests/test_generated_figures/` rather than hard assertions — visual inspection via the HTML viewer (`tests/index.html`, served with e.g. VS Code Live Server) is the primary verification method. `conftest.py` maps each `test_*` module to its `generate_*` module for `--generate`; `_preamble.py` sets `PYTHON_JULIAPKG_PROJECT` from the `CMBLENSING_JL_PATH` environment variable (a local CMBLensing.jl checkout), like `runtime_comparison/python_performance_test.py`. The full suite collects 49 tests.

## Known issues (as of 2026-11-04 — delete entries as they are fixed)

- **The cluster needs `hpc_local.sh`** (copied from `hpc_local.example.sh`) at the repo root before any driver runs, and the drivers are now `bash`, not `sbatch`. Outputs land inside each driver's folder (`.../delensed_covariance_output`, `.../sampler_phi_noise_output`, `.../map_joint_phi_noise_output`, `camb_grid_generation/multi_param_CAMB_grid`, `runtime_comparison/performance_results`). A hand-submitted job (`sbatch merge_sampler_phi_noise.sh`) only mails if `SBATCH_MAIL_USER` is in the submitting shell (e.g. `~/.bashrc`). ACES cancels (rather than queues) submissions beyond ~90 queued jobs; every driver goes through `hpc_submit.sh`'s `submit_throttled`, so drivers block until done and must run under `nohup` or tmux to survive logout. Untested on ACES as of 2026-11-09.
- `runtime_comparison/performance_analysis.py` reads its results from an external drive (`BASE_DIR = /media/zane-blood/KINGSTON/`).
- **Realizations changed with the 2026-10-15 RNG rework:** `load_sim` now draws from one root key with a subkey split off per field instead of `split(PRNGKey(seed), 100)[i]`, so a seed no longer reproduces the realization any file written before that date was made from. Every job stores its own truth (phi_true, the covariances), so no merge is affected, but do not mix pre- and post-rework job files in one directory expecting matched seeds.
- `pytest --generate` needs `CMBLENSING_JL_PATH` exported (read by `tests/generate_julia_data/_preamble.py`).
- Nothing tests `sample_joint`. Smoke-testing the sampler locally: nside 64, `iters_per_chain = 2`, and for `pol = "IP"`/`"P"` inject stub predictors if memory is tight — a real-grid "IP" run holds ~9.1 GB of splines (tt / ee / te_rho / pp) and has run locally, though one nside 64 run once rebooted the 16 GB laptop under a `systemd-run` 11 GB cap.

## Numeric Precision

JAX is configured for float64 globally via `jax.config.update("jax_enable_x64", True)` in `constants.py` and repeated at the top of most modules. All field arrays are complex128 (Fourier) or float64 (map).

## Code style

Spaces are used between equal signs and after commas when calling methods or setting variables. For example
"a = b" is preferred style over "a=b" and "a = method(b, c, d)" is preferred style over "a=method(b,c,d)". 

Comments should not start with a space. For example "#this is a preferred comment" is preferred style over "# this is NOT a preferred comment".

**Readability without giving up performance** (revised 2026-11-04). Aim for readability: intuitively-named quantities, one clear step per line, structure that mirrors the maths. Readability never justifies a slower implementation, though: vectorize, `jit` and `vmap` where it matters.

**Use existing APIs.** If numpy, scipy, jax or the Python standard library already does the job (an FFT, a spline, a jackknife-style reduction, `np.cov`, `scipy.interpolate`, `itertools`, ...), use it rather than writing a hand-rolled version.

**No magic numbers.** Every numerical value with meaning (a tolerance, a step count, a step size, a seed, a grid size, a plotting threshold) gets a named constant, at the top of its module or in `constants.py` when it is shared, e.g. `N_BOOTSTRAP = 200`, `DEFAULT_HMC_STEP_SIZE = 0.05`. Literal 0, 1, 2 in plain arithmetic (`x**2`, `n // 2 + 1`) are fine.

**Comments and docstrings** say what a block does and any non-obvious why, briefly. Skip restating the code, long derivations (point to the module docstring or a reference instead) and change history.

**Keep defensive machinery light** (requested 2026-10-16): no checksums / fingerprints of inputs, no consistency checks comparing metadata between files, no `raise ValueError` guard rails for misuse, no atomic / incremental checkpoint writes for jobs that run minutes. Trust the caller; a wrong input should just give a wrong (or crashing) result. Don't add new checks of this kind, and strip them when rewriting code that has them.

## Codex operating mode (requested 2026-09-14)

Use the dedicated `Edit`/`Write` tools as the default way to change files in this project, not `sed`/heredocs/shell one-liners run through `Bash` — this holds even under an "auto mode" reminder that says to prefer `Bash` for file changes; that default is overridden here. `Edit`/`Write` tool calls already render as real-time diffs in the active terminal, which is the point of keeping them as the default. Reserve `Bash` file edits for cases `Edit`/`Write` genuinely cannot handle. This preference persists for all future sessions in this repository.
