# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX port of CMBLensing.jl — simulates gravitational lensing of CMB temperature and polarization fields, recovers the lensing potential via gradient-based MAP optimization (`map_joint`), and jointly samples the five LCDM cosmological parameters from a lensed data map via Gibbs sampling (`sample_joint` in `sample_lcdm.py`). The sampler's posterior widths are compared against a Gaussian Fisher forecast (`fisher_forecast.py`), reported as an upper / lower bound pair. All array operations use JAX for GPU/TPU support and automatic differentiation.

History that explains the current shape of the tree:
- **2026-08-24 refactor:** the CAMB emulator, `sampling_ar.py`, `sample_lcdm_legacy.py` and every experimental theta sampler were removed; `sample_lcdm.py` is the sole sampler; spline caches lived in `cmb_lensing/camb_splines/` (moved to the repo-root `camb_splines/` on 2026-10-15). See "Removed features".
- **2026-10-14 cleanup:** the user deleted the sibling forecast modules (`fisher_forecast_from_cls.py`, `_full_sky.py`, `_from_logpdf.py`, `_from_1st_principles.py`), `lensing_covariance.py`, `delensed_spectrum.py` (transfer function), `sampler_noise_estimate.py` (score bound), `compare_grid_derivatives.py`, `plot_nphi_sources.py`, and both `*_TEMPLATE` directories. `fisher_forecast.py` was then rewritten around the two-bound report (see "The Fisher forecast"), and `sampling_chains/` was reorganized into per-task subdirectories.

## Commands

```bash
# Install (editable)
pip install -e .

# Tests (the user runs these manually; do not run the whole suite after every change)
pytest
pytest tests/test_lensing.py
pytest tests/test_fisher_forecast.py            # ~1 min, runs CAMB

# Regenerate Julia ground-truth data before testing (requires juliacall + CMBLensing.jl;
# first set PYTHON_JULIAPKG_PROJECT in tests/generate_julia_data/_preamble.py)
pytest --generate
python tests/generate_julia_data/generate_all.py

# A single pilot LCDM sampling chain locally (nside 64, T-only, theta_MC_100 only by default)
python -m cmb_lensing.sample_lcdm

# 1D CAMB spline caches (writes camb_splines/camb_<param>_grid.npz at the repo root)
python camb_grid_generation/precompute_camb_1d.py <theta_MC_100|logA|ns|ombh2|omch2|all> [--test]

# 5D CAMB grid on the HPC (camb_grid_generation/)
bash camb_grid.sh                                          # 875 slurm jobs, one H0 sweep each
python merge_camb_grid.py --grid_dir multi_param_CAMB_grid  # -> camb_grid_spline.npz
python validate_camb_grid.py --grid <...>/camb_grid_spline.npz -n 20

# HPC setup (since 2026-10-16): nothing personal is in the tracked scripts. On the cluster,
#   cp hpc_local.example.sh hpc_local.sh    # gitignored; conda activate, SBATCH_MAIL_USER, CMBLENSING_JL_PATH
# Drivers (camb_grid.sh, sample_lcdm.sh, get_*.sh, run_performance_test.sh) are plain bash, run from
# any directory: each cd's into its own folder, sources ../hpc_local.sh and sbatch-es its job script
# by a relative name. Job scripts find their .py and ../hpc_local.sh through $SLURM_SUBMIT_DIR
# (= the driver's folder) and write their out_dir inside that folder, so the checkout can live anywhere.

# LCDM chains on the HPC (get_lcdm_samples/; chains -> get_lcdm_samples/lcdm_sampler_data/<experiment>/)
bash sample_lcdm.sh              # spawns run_single_lcdm_chain.sh per (map, chain)
python chain_analysis.py         # chains -> distributions, triangle plot vs the forecasts, and the
                                 # phi-noise / delensed-spectra plots at the chains' noise level

# Fisher forecast (writes cmb_lensing/fisher_output/fisher*_from_blocks_<nphi>_<camb|empirical>_<noise>_uk.*)
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source qe_unlensed                       # UPPER bound
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source qe_delensed --qe_response gradient # LOWER bound
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source map_joint --phi_noise <delensed_covariance.npz>
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source sample_joint --phi_noise <sampler_effective_noise.npz>
python -m cmb_lensing.fisher_forecast --noise 5 --nphi_source qe_unlensed --delensed_covariance <delensed_covariance.npz>
python -m cmb_lensing.fisher_forecast --stability ...          # also at 2h; not with --delensed_covariance

# Empirical delensed covariance stencil (get_delensed_field_covariance/)
bash get_delensed_covariance.sh                                # one job per seed; set qe_filter / phi_noise in it
python merge_delensed_covariance.py --covariance_dir <scp'd out_dir>   # -> delensed_covariance.npz

# Sampler effective phi noise (get_sampler_phi_noise/)
bash get_sampler_effective_noise.sh                            # one fixed-theta chain per seed
sbatch merge_sampler_effective_noise.sh                        # a job itself, submit from its folder; -> sampler_effective_noise.npz; works on running jobs
python -m cmb_lensing.sampler_effective_phi_noise --nside 64 --theta_pix 5 --n_maps 4 --n_sweeps 400 --burn_in 100 --out_dir <dir>   # local

# Band-averaged map_joint N_eff (diagnostic only; get_map_joint_phi_noise/)
bash get_effective_phi_noise.sh
python merge_phi_noise.py --noise_dir <scp'd out_dir>
```

No linter or formatter is configured. No CI pipeline exists.

## Repository Layout

```
cmb_lensing/                  the Python package (constants.py holds PARAM_ORDER / GROUND_TRUTH / PARAM_BOUNDS / PARAM_SIGMA / CAMB_LMAX / CAMB_SPLINES_DIR / CAMB_GRID_PATH)
  fisher_output/              fisher_forecast.py's figures and npz
  sample_lcdm_output/         sample_lcdm's progress plots (must exist before plotting)
camb_splines/                 five ~3 MB camb_<param>_grid.npz (1D caches) + the gitignored ~8 GB camb_grid_spline.npz (5D grid)
camb_grid_generation/         precompute_camb_1d.py (the 1D caches) + slurm / merge / validate scripts for the 5D grid
docs/                         map_joint_tutorial.ipynb, sample_joint_tutorial.ipynb
experimental_results/         hand-maintained results (README + experimental_details.md per run)
paper/                        paper.md / paper.bib / figures / build_pdf.sh
tests/                        Julia A/B benchmarks + test_native_5d_interp + test_fisher_forecast
get_lcdm_samples/             sample_lcdm.sh, run_single_lcdm_chain.sh/.py, chain_analysis.py
  lcdm_sampler_data/            chain *_history.txt directories
  lcdm_chain_plots/noise_level_<tag>/   chain_analysis.py output (tag e.g. 2dot5uk), all lowercase:
    <param>/                    one folder per sampled parameter
    spectra_plots/              phi_noise_vs_qe_per_mode_<tag>.png, delensed_spectra_comparison_<tag>.png
    fisher/                     covariance_matrix_empirical.png / correlation_matrix_empirical.png (the chains)
      <nphi_source>/<t_block>/  one per forecast: nphi_source in qe_unlensed / qe_delensed / map_joint /
                                sample_joint, t_block camb_delensed (always) or empirically_delensed (when
                                FORECAST_DELENSED_COVARIANCE has a file at this noise level); each holds
                                triangle_plot_<label>.png (chains blue, forecast red), covariance_matrix_<label>.png,
                                correlation_matrix_<label>.png, covariance_ratio_empirical_over_<label>.png,
                                label = <nphi_source>_<t_block>. FORECAST_PHI_NOISE / FORECAST_DELENSED_COVARIANCE
                                paths carry {tag}. Chain errors are always the delete-one-map jackknife
get_delensed_field_covariance/ get_delensed_covariance.sh, run_single_delensed_covariance.sh/.py, merge_delensed_covariance.py
  delensed_field_covariance_data/  merged delensed_covariance.npz runs, nside 128 / 2.5', see below
get_sampler_phi_noise/        get_sampler_effective_noise.sh, *_1_chain.sh/.py, merge_sampler_effective_noise.sh/.py
  sampler_phi_noise_data/       sampler_effective_noise_output_mean_chains_{2dot5,5,10}uk/
get_map_joint_phi_noise/      get_effective_phi_noise.sh, get_phi_noise_1_realization.sh/.py, merge_phi_noise.py
runtime_comparison/           slurm scripts timing map_joint Python vs Julia
hpc_local.example.sh          template for the gitignored hpc_local.sh every HPC script sources
```

`camb_grid_generation/precompute_camb_1d.py` is NOT part of the package: `sample_lcdm` imports it by file path, and only on the `USE_CAMB_SPLINE` path (`_load_precompute_camb_1d`). The `get_*` folders and `runtime_comparison/` were moved to the repo root on 2026-10-15 and are no longer gitignored (they are currently untracked).

**`delensed_field_covariance_data/` naming.** The `*_qe_bound` folder names refer to bounds on **N_phi**, the opposite way round from the forecast's information bounds:
- `lower_qe_bound/*_nphi_unlensedx2` — W built from the QE with the unlensed C_TT in the filter and the unlensed response, i.e. the forecast's `qe_unlensed` (UPPER information bound). Made with a since-removed local code hack, so their merged npz recorded `iterative_delens = True`; on 2026-10-14 `qe_filter = "unlensed"` was added to the three merged files. The 5 uK folder also contains a stray nested, unpatched copy of the 2.5 uK run.
- `upper_qe_bound/*_iterative_delens_qe_gradient` — the iterated delensed filter with the gradient response, i.e. `qe_delensed --qe_response gradient` (LOWER information bound).
- `empirical_map_joint_noise/` — W from the `lower_qe_bound` files' `phi_noise_fid` (map_joint's N_eff); merged files only, with `copied_phi_moments.npz`.
- `empirical_sampler_noise/` — W from `sampler_effective_noise.npz`'s `nphi`; the per-seed job files are present, so `forecast_jackknife` works on them.

`camb_splines/camb_grid_spline.npz` (~8 GB) is gitignored — it was once committed by accident and pushed a pack past GitHub's 2 GiB limit; never force-add it.

## Architecture

### Computational Pipeline

`simulate.load_sim(nside, theta_pix, pol, master_seed, ...)` is the main entry point. It runs CAMB (through `jax.pure_callback`, so a CAMB failure yields NaN spectra → non-finite logpdf, never an exception), builds covariance matrices and random fields in Fourier space, lenses them via `primal_lense_flow`, and returns a `DataSetT` / `DataSetEB` / `DataSetTEB` (`dataset.py`) holding the fields, operators, and simulated data. Notable current choices, all made for LCDM inference:
- **Mask is forced to all ones** (`jnp.ones_like(get_mask(...))`) — there is no sky mask in any data set. This also applies to the Julia-comparison tests.
- **Noise:** `noise_cls(..., l_knee, alpha_knee = 3)` includes the 1/f term `1 + (l_knee/ell)**3`; `load_sim`'s default is `l_knee = 100` (matches Julia), while every sampler entry point (`sample_lcdm.__main__`, `run_single_lcdm_chain.py`, `add_starting_matrices_to_args`) passes `l_knee = 0` for pure white noise. Beam FWHM defaults to 0.
- **Cls are CAMB-only.** The `use_emulator_cls` flag and the `cambemul` import are gone.
- **The cls dict carries the T-phi cross spectrum** (added 2026-09-10): `_run_camb` keeps the PP and PT columns of `get_lens_potential_cls(CMB_unit = "muK")` (PP is dimensionless either way; "muK" puts PT in the TT units), the callback returns a `(lmax_prime, 2)` array, and `_extract_all_cls` adds `cls["TP"]` via `dl2cl(..., is_tphi = True)` = D · 2π / (L(L+1))^{3/2}. Note `cls["phi"]` still uses the ℓ⁴ convention, so `TP/√(TT·PP)` differs from CAMB's correlation coefficient by (1 + 1/ℓ). TP changes sign near ℓ ~ 1100 (1e-16 level), so it must not go through the log-interpolating `covar_matrix_from_cls`. Nothing consumes it since the forecast's "lensed" mode was removed (2026-10-14). The 5D grid and the 1D spline caches do not have it. `EXPECTED_CL_KEYS` includes `"TP"`, so a `precomputed_cls` dict must too.
- Tensor covariances are added back in (`cf = cf_scalar + cf_tensor`), but every sampler path runs at `r = 0`.
- `get_d_tt_matrix(cf, cn)` (2 args), `get_d_teb_matrix`, `get_d_eb_matrix` build the D mixing matrix; `get_g_matrix(cphi, nphi, a_phi_fid, a_phi)` / `get_g_matrix_lcdm(cphi_fid, cphi_curr, nphi, cn_tt)` build G (`cn_tt` is unused there). `lmax_prime = min(lmax, DEFAULT_MAX_ELL)` with `DEFAULT_MAX_ELL = 4000` in `constants.py` (renamed from `EMULATOR_MAX_ELL`).

`map_joint(data_set, num_steps = 30, constant_step = False)` recovers f and phi by alternating Wiener-filter steps (estimate f) with a BFGS line search on phi along the preconditioned mixed gradient (`mixed_grad_phi_logpdf`, hessian = `pinv(Cphi^-1 + Nphi^-1)`). G is set to the identity there (MAP is invariant to it).

`sample_joint(...)` (in `sample_lcdm.py`) draws samples from the full joint posterior P(f, phi, theta | d) via Gibbs sampling, where theta are the five LCDM parameters. This is the code path the project is centred on.

> **DEFAULT FILE FOR CONVERSATIONS:** when the user refers to "the sampler", "sample_joint", "gibbs_sample_theta", theta/parameter sampling, etc., they mean **`cmb_lensing/sample_lcdm.py`**. `sample_lcdm_legacy.py` and `sampling_ar.py` no longer exist.

### The LCDM sampler (`sample_lcdm.py`)

**Parameters.** `PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]` (tau is pinned at 0.05 by prior, mnu at 0.06). `current_params` is a `(5,)` jnp array in that order; user-facing dicts (`param_init`, `proposal_sigmas`, `param_ranges`, `should_sample`) are keyed by name. Single sources of truth in `cmb_lensing/constants.py` (moved there from `precompute_camb_1d.py` on 2026-10-15; every module and script imports them from `constants`): `PARAM_ORDER` / `PARAM_INDEX`, `GROUND_TRUTH` (fiducial values), `PARAM_BOUNDS` (±5-sigma search box), `PARAM_SIGMA` (normalizes the progress plot and sets the forecast's / stencils' steps), `CAMB_LMAX`, `CAMB_SPLINES_DIR`, `CAMB_GRID_PATH`. Note the Julia tests each define their own module-level `GROUND_TRUTH` (a data path) after their wildcard imports, which shadows the dict inside those files only. `to_camb_naming_conv` converts `theta_MC_100 -> cosmomc_theta = theta/100` and `logA -> As = exp(logA)*1e-10` **in place** (it deletes the old keys from the dict it is given).

**Cl source.** Module-level switches, mutually exclusive (a `ValueError` fires at import if both are True):
- `USE_CAMB_GRID = True` (**default**) — `get_camb_grid_predictors()` lazily loads the 5D grid once per process from `constants.CAMB_GRID_PATH = camb_splines/camb_grid_spline.npz` (repo root) and returns a `GridPredictors` namedtuple (see the grid section). Required for `pol = "IP"` / `"P"`. `sample_joint` checks that `param_init` is inside the grid (finite TT) before doing anything heavy.
- `USE_CAMB_SPLINE = True` — 1D cached-CAMB splines, loaded through `get_camb_spline_predictors` → `_load_precompute_camb_1d` (imports `camb_grid_generation/precompute_camb_1d.py` by path); exactly one `should_sample` entry may be True. `wrap_temp_predictors` lifts the `(predict_tt, predict_pp)` pair into a `GridPredictors` with `None` for every other spectrum (T-only).

Every predictor (grid and 1D) has the signature `f(params_batch: (M, 5)) -> (M, n_ell)` (the emulator-era `emu_params` first argument is gone). Predictors are passed as **static jit arguments** to `_recompute_cosmo_matrices*`, so they must be the same cached function objects every call (fresh closures retrace and recompile the lensing graph).

**Gibbs structure of `sample_joint`** (mirrors CMBLensing.jl's `sample_joint`), per iteration:
1. `gibbs_sample_f` — draw f from its conditional: simulate a fresh field and noise realization from the current covariances, lens the field, and add a Wiener-filter correction driven by `data - new_data`. Branches on the covariance type: `DiagonalScalar` (independent draw), `BlockTEB` (`_field_matrices_from_teb_covar`: E ~ N(0, C_EE), T = (C_TE/C_EE)E + N(0, C_TT − C_TE²/C_EE), identical to `load_sim`'s correlated T/E draw; a zero TE block reduces it to independent draws so the same helper serves the noise), `DiagonalEB` (`_field_matrices_from_eb_covar`).
2. `mix` (`mixing.py`) — `f° = L(phi) D f`, `phi° = G phi`.
3. `gibbs_sample_phi` — one HMC step (`hmc_step` → `symplectic_integrate`, defaults `num_steps = 5, step_size = 0.1`; Julia uses N = 25, ε = 0.01) on the mixed phi with mass matrix `pinv(G)² (pinv(Cphi) + pinv(Nphi))` (`get_mass_matrix`). The momentum is drawn from the **phi** template `x`, never from `mixed_field` (an S02 template would corrupt it). Accept if `log(u) < ΔH`; always accepted for the first `num_burn_in_always_accept` iterations.
4. Theta step (skipped while `iter < num_burn_in_fix_theta`): for each parameter in `proposal_sigmas` **dict order** with `should_sample[name] = True`, `gibbs_sample_theta` → `metropolis_sample_theta`: a symmetric Gaussian random-walk Metropolis-within-Gibbs move of scale `proposal_sigmas[name]`, `metropolis_num_steps` per sweep. Proposals outside `param_ranges[name]` are rejected before the interpolator is touched (exact MH, symmetric proposal); NaN logpdfs (out of grid / unreachable theta) reject via `(x < nan) == False`. The conditional is the **mixed** logpdf — `make_eval_logpdf_batch` rebuilds Cphi, Cf (TT / TT+TE+EE+BB / EE+BB), G and D at every candidate theta and evaluates `mixed_logpdf` per row under `jax.vmap`. Tune sigmas toward ~44 % acceptance per parameter (phi HMC toward ~65 %).
5. `update_args_after_sample` → `get_new_cosmo_matrices` — recompute Cphi, Cf, D, G (and optionally the QE norm) at the newly accepted theta.
6. `unmix` — back to `(f, phi)` with the **updated** D/G.

`args` is a plain dict carrying operators (`noise_covariance`, `mask`, `beam`, `field_covariance`, `phi_covariance`, `mixing_d`, `mixing_g`, `quadratic_estimate`) plus metadata (`nside`, `pix_width`, `ell_grid`, `ells`, `lmax`, `lmax_prime`, `cphi_fid`). `add_starting_matrices_to_args` initialises them at `param_init`: it runs a **throwaway `load_sim` at `param_init`** (random seed, `l_knee = 0`) purely to get the lensed covariances for the initial QE norm, then computes Cf from the predictors (`get_new_cf_matrix`) and D/G/Cphi via `get_new_cosmo_matrices`. `set_initial_ds_conditions` swaps those into the data set so `map_joint` (for `phi_init = "MAP"`) sees the initial cosmology rather than ground truth. `cphi_fid` — the ground-truth phi covariance from the data set — is what `get_g_matrix_lcdm` normalises G against.

**refresh_qe.** The jitted kernels `_recompute_cosmo_matrices[_teb|_eb](..., refresh_qe = False)` can rebuild the quadratic-estimate norm from the grid's lensed Cls at the new theta (scalar QE for "I", polar QE for "IP"/"P"), so G and the phi mass matrix would track theta. **`sample_joint` does not currently expose this** — `update_args_after_sample` is called without `refresh_qe`, so the QE norm stays frozen at the `param_init` cosmology for the whole chain.

**Polarization plumbing.** The jitted theta kernels take raw arrays, not structs: `field_matrix_stack` gives `(3, nside, nside//2+1)` T/E/B stacks for "IP", `(2, ...)` E/B for "P", the bare matrix for "I"; `op_matrix_stack` gives `(4, ...)` TT/TE/EE/BB or `(2, ...)` EE/BB operator stacks; `op_shared_matrix` returns the single mask/beam block. `make_eval_logpdf_batch(..., pol, bb_is_zero)` rebuilds `FlatS0/FlatS2/FlatS02` and `DiagonalScalar/DiagonalEB/BlockTEB` per row. `sample_joint` infers `pol` from the data set type. `_replace_teb_blocks` / `_replace_eb_blocks` write recomputed blocks back into the operators (ET mirrors TE).

**Physics decisions carried over from the pre-refactor sampler** (still in force):
- **TE is carried as the correlation ratio** `te_rho = Cl_TE/√(Cl_TT·Cl_EE)` (linearly interpolated, `te_covar_from_rho`), so `Cl_TE = ρ√(TT·EE)` from the same TT/EE the model uses everywhere else keeps the 2×2 T/E block positive semi-definite by construction. The D matrix carries the full coupled T/E block (`get_d_teb_matrix`: Schur inverse + `block_matrix_sqrt`, `d_et` used for both off-diagonals — empirically matches Julia).
- **Unlensed BB = 0 at r = 0.** Identically-zero spectra must bypass the log-interpolating `covar_matrix_from_cls` (log(0) → NaN): eager code uses `_covar_or_zeros`, jitted code the static `bb_is_zero` flag. The D-mixing B block is the **identity** where `cf_bb = 0` (the naive formula gives `d_bb = 0` and zeroes the mixed B sector).
- Never pin f/phi at fixed values while still re-mixing them with theta-dependent D/G every sweep: that makes the theta chain a fixed-point iteration on the mixed conditional (measured attractor at +1 training sigma in theta_MC_100 for "IP"), not MCMC. With genuine f/phi resampling (the only mode the current code has) the mixed-coordinate theta step is exact.

**Logging / outputs.** `advanced_logging` dict keys: `phi_acceptance`, `lcdm_acceptance` (print running acceptance rates), `plot_log_pdf`, `plot_lcdm_sigmas` (overwrite `cmb_lensing/sample_lcdm_output/logpdf_progress.png` and `lcdm_progress.png` every iteration; the directory is untracked and is not created by the code, so it must exist first). `matplotlib.use("Agg")` is set at the very top of the module — TkAgg aborted long chains from JAX worker threads; keep it first. With `hpc_path` set, every iteration appends the latest value of **every** parameter (sampled or not) to `{hpc_path}{theta}_map_{map_idx}_chain_{sub_chain_idx}_history.txt`; `chain_analysis.py` reads those. `sample_joint` returns `param_vals` (dict of per-parameter lists).

**`__main__`** builds an nside 64, theta_pix 5, T-only, 5 µK-arcmin, `l_knee = 0` data set at `GROUND_TRUTH`, samples only `theta_MC_100` starting from the lower edge of its box, `phi_init = "ZEROES"`, `num_burn_in_fix_theta = 0`, `seed = 67`. The HPC script (`get_lcdm_samples/run_single_lcdm_chain.py`, writing to `lcdm_sampler_data/joint_inference_08_24_26/` next to itself) samples omch2 / theta_MC_100 / logA from random starts inside the box, `num_burn_in_fix_theta = 100`, 1500 iterations, default `phi_init` (= `"MAP"`).

### Cached-CAMB 1D Splines (`camb_grid_generation/precompute_camb_1d.py`)

`python camb_grid_generation/precompute_camb_1d.py <param|all> [--test]` sweeps one parameter over `PARAM_BOUNDS` (50 nodes, `GRID_SIZE` / `PARAM_GRIDS` stay local to it) with the other four pinned at `GROUND_TRUTH`, writing `camb_splines/camb_<param>_grid.npz` at the repo root (`default_cache_path` = `constants.CAMB_SPLINES_DIR`). `load_camb_spline_predictors(param_name)` returns `(predict_tt, predict_pp)` — cubic splines in lnCl through the CAMB nodes, jit-safe via `jax.pure_callback`, raising if any non-swept parameter moves off its cached value, NaN beyond the cached grid. Cls live on ells 2..`DEFAULT_MAX_ELL`-1 (`CAMB_LMAX`) — the exact support of `load_sim`'s CAMB path, so model and data share the same log-log extrapolation anchor (a one-multipole mismatch measurably biased the ombh2 conditional). All five caches were generated with CAMB 2.0.0 (July 2026); regenerate if CAMB is upgraded. theta_MC_100 caveats: theta_MC_100 ≳ 1.117 needs H0 > 100, which is outside CAMB's *default* `theta_H0_range`, so the shipped cache lost 8 nodes and ends at 1.114375 (+3.1σ) instead of the box top 1.1452 — `_run_camb` now passes `DEFAULT_THETA_H0_RANGE = [10, 150]` (`constants.py`), so re-running the sweep recovers all 50 nodes. Until it is regenerated, spline-path proposals above 1.114375 get a NaN logpdf and auto-reject, which matters most for `run_single_lcdm_chain.py`'s random-in-box starts. Its TT spline error (~5e-3 lnCl leave-half-out) is the largest — densify first if its conditional looks suspicious.

### 5D CAMB Grid Spline (`camb_grid_interp.py`)

Real CAMB spectra precomputed on a full tensor-product grid over `(H0, logA, ns, ombh2, omch2)` (tau 0.05, mnu 0.06, r = 0, tensors off) and evaluated as a tensor-product cubic B-spline in lnCl over coefficients prefiltered at merge time with not-a-knot ends. Out-of-box queries return NaN → rejected proposal.

- **Grid file:** `camb_splines/camb_grid_spline.npz` (~7.9 GB, gitignored; `constants.CAMB_GRID_PATH`). Built on the HPC by `camb_grid_generation/camb_grid.sh` → `run_single_camb_grid.sh/.py` (one slurm job per (logA, ns, ombh2, omch2) node doing the 81-point H0 sweep; 81×5×5×5×7 = 70 875 CAMB calls in 875 jobs; numpy + camb only, no JAX) → `merge_camb_grid.py` (streaming, one spectrum at a time, incremental zip writes; asserts no holes, uniform axes, theta independent of logA/ns, |te_rho| < 1) → `validate_camb_grid.py` (random interior points vs direct CAMB; expect ≲ CAMB's own ~1e-3 lnCl floor).
- **Evaluation is `scipy.interpolate.NdBSpline`** (`_TensorSpline`), swapped in 2026-09-07 for ~150 lines of hand-rolled B-spline machinery (`_axis_stencil`, `_axis_stencil_derivs`, `_eval_bspline`, `_eval_bspline_grads`). This works with **no regeneration of the grid file**: `merge_camb_grid.py` already writes exactly NdBSpline's input (coefficients + knots with `len(t) == n + k + 1` per axis). Agreement with the old evaluator is ~4e-14 relative on the real grid, at ~3× the speed. Two things `_TensorSpline` still owns: the `BOUNDARY_TOL` clamp (bare `NdBSpline(extrapolate = False)` NaNs the box edges the sampler's conditional scans land on) and NaN-not-raise for genuine out-of-box rows. Cost: NdBSpline casts coefficients to float64, so a spectrum costs ~2.1 GB resident instead of ~1.1 GB — `CambGrid.spline` therefore caches only the spline, never the float32 array. `RegularGridInterpolator(method = "cubic")` is **not** an equivalent shortcut despite building an NdBSpline internally since scipy 1.12: it re-solves for coefficients with an iterative Krylov solver (`gcrotmk`), ~4e-4 away in lnCl from `make_interp_spline`'s direct banded solve, at ~10 min per spectrum. The old implementation is frozen in `tests/handrolled_bspline_reference.py` and `tests/test_native_5d_interp.py` checks production against it (`tests/benchmark_native_5d_interp.py` times them).
- **`CambGrid`** opens the npz once (`load_camb_grid` singleton per path) and builds `_TensorSpline`s **lazily per spectrum** via `CambGrid.spline(name)` (~1.1 GB read, ~2.1 GB resident, six of them) — T-only runs never touch the polarization/lensed tables. `bb_is_zero` flag: unlensed BB is identically zero on the r = 0 grid, so the `bb` predictor returns exact zeros (NaN out of box). A pre-polarization grid file stays loadable and raises a `KeyError` with a regeneration pointer only when a missing spectrum is requested.
- **`load_camb_grid_predictors(path)`** → `GridPredictors(tt, ee, bb, pp, tt_lensed, ee_lensed, bb_lensed, te_rho)`, each `f(params_batch) -> (M, n_ell)` via `jax.pure_callback(vmap_method = "sequential")`, cached per path. `LINEAR_SPECTRA = {"te_rho"}` are splined raw (no exp). This is the **only** predictor loader now — the pure-JAX (`_jax`) and custom-VJP (`_grad`) differentiable variants were removed with the gradient-based theta samplers. (`CambGrid.spline_param_grads` no longer exists; `_TensorSpline.value_and_grads` gives per-GRID-axis partials at fixed H0.)
- **`CambGrid.cl_local(spectrum, params_batch, grid_axis = None)`** (added 2026-09-22) evaluates a FEW points at a few MB: `coeff_memmap` memory-maps the ZIP_STORED `.npy` member and `_local_support` slices each point's 4^5 spline support (a cubic B-spline on one knot span needs only 4 coefficients + 8 knots per axis), so it never builds the ~2.1 GB NdBSpline. Bit-identical to `cl` (values, NaN pattern, box edges, and `grid_axis` analytic partials vs `value_and_grads`, checked on a synthetic grid). Use it for stencils / diagnostics on the 16 GB laptop; the sampler still uses `cl`. (Measured 2026-09-22: the grid's dC/dtheta moves the forecast's marginalized sigmas by <= 0.26% vs direct CAMB at nside 128 / 2.5' / 5 uK, so grid interpolation does not explain chain-vs-forecast gaps.)
- **H0 vs theta_MC_100.** The grid is laid out in H0 because a rectangular theta box has CAMB-unsolvable corners. Every node records theta; the merge collapses it to a 3D `theta_grid(H0, ombh2, omch2)`; `CambGrid.h0_from_theta` densifies theta(H0) to 2000 points, splines it over (ombh2, omch2), and inverts by interpolation. The sampler works in theta_MC_100 throughout.
- `tests/test_native_5d_interp.py` covers the evaluator; nothing covers the merge or the theta→H0 inversion end to end.

### The Fisher forecast (`fisher_forecast.py`, rewritten 2026-10-14)

One forecast: the central finite difference of two covariance blocks on the sampler's own rfft grid (`gen_ell_grid`), contracted as `F_ij = 1/2 sum_blocks sum_k w_k dln(C_k)/dtheta_i dln(C_k)/dtheta_j` with `w_k = get_fourier_weights` (2 per bulk entry, 1 on the self-conjugate columns, summing to nside^2 — the weights `statistics.logpdf` is built from; derivation in the module docstring):

```
f_delensed   C_TT^delensed + C_n     (C_n analytic, theta-independent)
phi          C_phi + N_phi           (N_phi frozen at theta_0)
```

**One N_phi drives both blocks**: it is the phi block's noise and, through `Alens_L = N_L / (C_L + N_L)` with `N_L` its azimuthal average (`radial_cl_profile`, geometric annulus mean), the residual lensing in the delensed T. `--nphi_source` (`NPHI_SOURCES`):

| source | N_phi | role |
|---|---|---|
| `qe_unlensed` | box QE matrix (`qe_noise_matrix` = `scalar_quadratic_estimate` without `NPHI_FAC`) with the **unlensed** C_TT in its filter (cfl_tt → cf_tt) | **UPPER bound** on the information |
| `qe_delensed` (default) | same QE with the filter **iterated** from the lensed to the partially delensed C_TT (`iterative_delensing`, 5-6 iterations to 1e-6) | **LOWER bound** |
| `map_joint` | `phi_noise_fid` of a merged `delensed_covariance.npz` with phi moments (`--phi_noise`); NaN (unmeasured) modes carry no phi information | comparison |
| `sample_joint` | `nphi` of a `sampler_effective_noise.npz` (`--phi_noise`) | comparison |

`--qe_response` (`unlensed` default / `gradient` = CAMB's lensed T-grad-T, iterated with the filter under `qe_delensed`) applies to the two `qe_` sources; `chain_analysis.py` uses `gradient` for `qe_delensed` and `unlensed` for `qe_unlensed`, matching the data folders. `load_phi_noise_matrix` classifies a noise file (`nphi_cross_raw` present → `sample_joint`; `phi_noise_fid` / `nphi_map_joint` → `map_joint`; else `score`, a legacy sampler_noise_estimate bound the forecast no longer accepts) and refuses another box or cosmology.

**Temperature block.** Default: CAMB's `get_partially_lensed_cls` at the frozen `Alens_L`, re-run at every stencil point (`delensed_cls_at_params`; the non-perturbative correlation-function lensing, NOT an interpolation between unlensed and lensed). `--delensed_covariance <npz>` swaps in the EMPIRICAL per-mode delensed covariance (centre and +/- points; C_n still analytic) and its steps h_i for both blocks. Nothing checks that the file's W was built from the same N_phi as the forecast's phi block (`check_wiener_matches` / `nphi_checksum` removed 2026-10-16); matching them is the user's job. `forecast_jackknife(..., delensed_covariance, **stencil_kwargs)` re-forms the f block from every delete-one realization mean (`delensed_covariance.leave_one_out_grids`; needs the per-seed job files) and returns `(fisher, names, leave_one_out)`. Nothing calls it any more: since 2026-10-15 `chain_analysis.py` draws every forecast WITHOUT error bars (only the measured chains carry them).

**Measured 2026-10-14**, nside 128 / 2.5' / 5 uK, omch2 / theta_MC_100 / logA, `camb_lmax = 4000`:

| N_phi (CAMB T) | s(omch2) | s(theta) | s(logA) | r(om, th) |
|---|---|---|---|---|
| qe_unlensed | 0.0028 | 0.0033 | 0.0219 | +0.19 |
| qe_delensed, gradient | 0.0030 | 0.0034 | 0.0221 | +0.23 |
| map_joint | 0.0030 | 0.0033 | 0.0219 | +0.24 |
| sample_joint | 0.0030 | 0.0033 | 0.0218 | +0.23 |

With the empirical T block: qe_delensed/gradient 0.0028 / 0.0033 / 0.0219, r +0.27; sample_joint 0.0028 / 0.0033 / 0.0218, r +0.27 (jackknife error on the sigmas ~4e-6).

**The bracket is an ordering of the marginalized sigmas, not of the matrices.** The phi block's `F(qe_unlensed) - F(qe_delensed)` is PSD, but the delensed T block LOSES information as N_phi drops (less residual lensing → less of the lensing-derived omch2 / logA information it double counts with the phi block), so the total difference has a small negative eigenvalue almost along logA (-12 vs a largest eigenvalue of 2.25e4 at the box above). `tests/test_fisher_forecast.py::test_bounds_are_ordered` pins exactly this.

Conventions and facts that still apply:
- **Frozen reconstruction.** N_phi and Alens_L are built once at theta_0 (`frozen_reconstruction`); `--vary_nphi` is gone. N_phi is a property of the estimator, and the empirical sources have no theta dependence to difference.
- **Steps** `h = FD_STEP_FRAC * PARAM_SIGMA` (0.05 sigma), direct CAMB (never the grid spline), `2 n + 1` runs memoized in `_CAMB_CACHE` / `_CAMB_RESULTS_CACHE` / `_GRADIENT_CACHE`. `--stability` recomputes at 2h.
- **`camb_lmax`** defaults to `camb_lmax_for_grid` (the grid's corner, 6110 at 2.5'): power-law continuation past CAMB's 3999 underestimates the damping tail (2.5x at the corner, 19% on sigma(theta) once). `chain_analysis.py` pins `FORECAST_CAMB_LMAX = 4000`. delensed_covariance.py builds W at CAMB_LMAX.
- **`NPHI_FAC = 2`** is CMBLensing.jl's preconditioning heuristic; `qe_noise_matrix` does NOT divide by it (physical N^(0)). The box QE norm is the square-domain, periodically wrapped Hu & Okamoto N0 to 0.5%; its annulus scatter (1.6x at L = 540, 5.7x at L = 2090 on nside 64 / 5') is real anisotropy.
- **`gradient` response:** raises N_phi 0.4-2.6% and moves the sigmas < 0.35% at nside 64 / 5' / 5 uK.
- Outputs are in `OUTPUT_PARAM_ORDER` (`omch2, ombh2, ns, theta_MC_100, logA`), chain_analysis's order. The npz stores the run config, steps, and the N_phi description (source, filter, response, file).
- Kept for other modules: `prior_covariances`, `load_sim_cosmology`, `sampled_names_and_steps`, `covariance_from_fisher`, `report_sigmas`, `save_outputs`, `add_box_arguments`, `sampled_from_args`, `box_subtitle`, `run_config(args, names, steps = None, **extra)` (mixed_logpdf), `qe_noise_matrix` (sampler_effective_phi_noise, plots), `_instrument_matrices` / `qe_response_cl` / `cls_with_qe_response` (map_joint_phi_noise.py), `iterative_delensing` (plots).

None of these is the marginal Fisher of p(d | theta) that the sampler targets. Two facts from the deleted marginal-Fisher work worth keeping: **never `jax.grad` w.r.t. theta** - `phi_dot_wrapper`'s custom VJP (`statistics.py`) differentiates `dot(phi, Cphi)` rather than `dot(phi, Cphi^-1 phi)`, so any autodiff through `phi_covariance` is silently wrong; and the mixed change of variables has a theta-dependent Jacobian (`mixed_logpdf` subtracts `logdet(G) + logdet(D)`), so a mixed Hessian is not comparable to an unmixed one.

### The empirical delensed covariance stencil (`delensed_covariance.py`, added 2026-09-24)

Replaces the forecast's CAMB delensed T block, value AND finite difference. Per seed (common random numbers) and per stencil point `{theta_0, theta_0 +/- h_i}` a job runs `load_sim(theta)` → inverse-lenses the NOISELESS lensed field by the Wiener-filtered TRUE phi `W phi` (`W = C_phi / (C_phi + N_phi)` per mode, built once at theta_0) → stores `|F|^2 / nside^2` on the rfft grid (covar_matrix_from_cls units, origin zeroed). Only the delensed field is stored. The merge is just the realization mean at each stencil point (`delensed_fid` / `_plus` / `_minus` + metadata): no band diagnostics, no plot, no common-random-number or round-trip check (all removed 2026-10-16; the CRN check measured ~5e-13 when it existed).

**This path does nothing else (2026-10-16):** no map_joint, no phi moments, no `copied_phi_moments.npz`, no `reconstruction` / `constant_nphi` (N_phi is always constant in theta). map_joint's reconstruction noise is `get_map_joint_phi_noise/`'s job. A merged file written from now on has no `phi_noise_fid`, so it can no longer be `--phi_noise` for `--nphi_source map_joint`; the older merged files in `delensed_field_covariance_data/` still can.

**W's N_phi** (recorded in every job file; the merge refuses to mix):
- box QE (`wiener_nphi_source = "qe"`): `qe_filter` (`QE_FILTERS`: `lensed` one-shot, the default; `delensed` iterated; `unlensed`) + `qe_response`, built by `fisher_forecast.qe_reconstruction`.
- a stored matrix (`phi_noise` path, job arg 10; `wiener_nphi_source = "file"`): `load_stored_nphi` reads the npz's `"nphi"` key (a per-mode `(nside, nside//2+1)` rfft2 matrix; NaN / negative modes → W = 0) and refuses another box or cosmology, so the file must carry `nside`, `theta_pix`, `noise_level`, `l_knee`, `param_names`, `params`. The job records the path (`wiener_nphi_file`), no checksum.

No backward compatibility (since 2026-10-16): the merge reads `qe_filter`, `qe_response` and `wiener_nphi_source` directly, so job files written before 2026-10-16 cannot be merged; all job and merged files are to be regenerated. Jobs write their npz once, at the end (no checkpoints since 2026-10-16: a job takes minutes; a job that dies leaves no file). The band helpers (`band_edges`, `band_average`, `DEFAULT_DELTA_ELL`) live here only for `chain_analysis.py`'s delensed-spectra plot, which draws `delensed_fid` band-averaged directly against CAMB (unpaired since 2026-10-16, so it carries the realizations' cosmic variance). Driver: `get_delensed_field_covariance/get_delensed_covariance.sh` (job args: index, seed, nside, theta_pix, noise, l_knee, step_sigma, qe_filter, qe_response, phi_noise, out_dir, params from 12). MEASURED ~21 s / 1.3 GB for a 3-point job at nside 128 locally (with map_joint; less now).

### The sampler effective phi noise (`sampler_effective_phi_noise.py`, added 2026-10-06 as `sampler_effective_noise.py`, renamed 2026-10-15)

One chain per (f, phi, d) realization samples (f, phi) | d with theta HELD at GROUND_TRUTH, and the posterior-mean phi is cross correlated with the true phi. The chain is `fixed_theta_chain.posterior_mixed_draws` (`sample_joint`'s sweep minus the theta step; operators from `fixed_theta_chain.mixed_theta_matrices` at the ground truth with load_sim's QE norm, key `chain_key(map_seed)`, start phi = 0), unmixing phi° by `pinv(G)`. No sub-chains, no R-hat (user decision), no thinning. Per mode, with the moments SUMMED OVER REALIZATIONS before the ratio (one mode of one realization has r = cos(phase difference), and C tan^2 of it has infinite mean): `r^2 = <B>^2 / (<A><D>)`, `N = C_phi (1/r^2 - 1)`, delete-one-realization jackknife. Four estimates that must agree (`ESTIMATES`): `cross` (headline; A = cross power between the means of 4 disjoint post-burn-in blocks, free of the Var_post / n_eff bias), `naive` (biased HIGH by ~(1 + N/C) / n_eff), `b_only` (r^2 = <B>/<D>), `variance` (r^2 = 1 - <V>/C, fragile where N >> C). `--burn_in_scan` repeats the whole analysis per burn-in (`_burn_in_<b>` suffix). Self-conjugate columns / origin / unmeasured modes are filled as N_QE x the band ratio; `--smooth` fills every mode that way (`nphi_smooth`). Helpers it used from the deleted `sampler_noise_estimate.py` (`used_modes`, `band_index`, `band_harmonic`, `qe_load_sim_matrix`, `save_atomic`, `simulate_base`) are now defined in it. **Storage is a live-readable memmap:** `sampler_phi_NNNN_samples.npy` (n_sweeps, nside, nside//2+1) complex64, preallocated (`rsync --sparse` to copy a partial run), flushed every 10 sweeps, then the sidecar `sampler_phi_NNNN.npz` (`n_done`, `finished`, ...) is rewritten atomically, so the analysis runs on RUNNING jobs. MEASURED nside 64 / 5': ~1.0-1.35 s per sweep after ~25 s compile, 1.35 GB RSS. The merge is too heavy for a login node (ESTIMATED 20-40 min and ~3 GB per burn-in at nside 128 / 6000 sweeps / 100 maps): `merge_sampler_effective_noise.sh`.

### Other measurement modules

- **`map_joint_phi_noise.py`** (renamed from `phi_noise.py` on 2026-10-15) — map_joint's band-averaged N_L^eff from |L|-annulus moments (`get_map_joint_phi_noise/`). Since 2026-10-16 the only place map_joint's reconstruction error is measured (the delensed-covariance path no longer runs map_joint); the forecast's map_joint source still reads the per-MODE `phi_noise_fid` of the existing merged delensed_covariance.npz files until this path writes a per-mode `nphi`.
- **`fixed_theta_chain.py`** (renamed from `fisher_forecast_from_mixed_logpdf.py` on 2026-10-15, when its mixed-Hessian and Louis forecast were deleted) — only what `sampler_effective_phi_noise.run_job` needs: `mixed_theta_matrices` (C_f / C_phi / D / G at one cosmology, built like the sampler's), `posterior_mixed_draws` (the fixed-theta Gibbs sweep, streamed to an `on_draw` callback) and `chain_key(map_seed, chain_index = 0) = fold_in(fold_in(PRNGKey(map_seed), CHAIN_STREAM), chain_index)`. CHAIN_STREAM must not be 0 or 1: load_sim's keys all descend from `split(PRNGKey(map_seed))` = `fold_in(PRNGKey(map_seed), 0 | 1)` under partitionable threefry, so a chain rooted at `PRNGKey(map_seed)` itself would reuse them. `chain_key(map_seed)` gives exactly the key the old `chain_key(map_seed, 0)` did.

### Polarization Modes

Three configurations control field dimensions and covariance structure, in `load_sim`, `map_joint` and `sample_joint` alike:
- **I** (intensity only): `FlatS0` fields, `DiagonalScalar` operators, `DataSetT`
- **P** (polarization only): `FlatS2` fields (E/B), `DiagonalEB` operators, `DataSetEB` (phi is still a `FlatS0`). At r = 0 this is effectively E-only information with no TE lever — expect much wider theta posteriors than "IP" (ombh2 especially weak).
- **IP** (both): `FlatS02` fields, `BlockTEB` operators (TT/TE/ET/EE/BB, mask and beam with zero TE blocks), `DataSetTEB`. EE's acoustic structure plus the physical TE correlation break the ns–ombh2 degeneracy T-only leaves (marginal corr ~ −0.76), at a considerably higher cost per sweep.

### Removed features (do not assume they exist)

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

**Basis and parametrization switching**: Fields carry a `basis` (MAP = real space, FOURIER) and are implicitly in a parametrization (T, QU, or EB). Conversion helpers `map()`, `fourier()`, `qu2eb()`, `eb2qu()` are used extensively — gradient code manually converts between representations before and after lensing. Note that `fields.map` shadows the Python builtin inside every wildcard-importing module. `fields.map` / `fields.fourier` have **no guardrails** — they apply `irfft2` / `rfft2` unconditionally, so `fourier` on an already-FOURIER field raises "only real valued inputs supported for rfft"; check `field.basis` first where fields arrive from mixed sources (`delensed_covariance._to_fourier` / `_to_map`, `sampler_effective_phi_noise._to_fourier`).

**RNG keys** (reworked 2026-10-15): `load_sim` starts from one root key `PRNGKey(master_seed)` and every random field splits its own subkey off it in a fixed order — phi, then the unlensed field(s) (T, or T_indep / E / B), then the noise — via `field_from_covar(nside, covar, rng_key) -> (field, rng_key)`. Because the order and the white noise do not depend on the cosmology, two calls at one seed share every white-noise draw (the common random numbers `delensed_covariance.py` relies on; verified exact to 4e-14). Do not go back to pre-splitting a batch of keys. Chains that reuse a map seed must root their keys off `fold_in(PRNGKey(seed), CHAIN_STREAM)` (`fixed_theta_chain.chain_key`), never `PRNGKey(seed)` itself.

**Random fields**: `field_from_covar_single_key` draws `irfft2(rfft2(white) * sqrt(C))` from REAL white noise so the two self-conjugate rfft columns get full (not half) power — the earlier per-mode complex draw halved their variance and biased every sampled field, noise realization and HMC momentum.

**CAMB through `pure_callback`**: `_camb_via_callback` wraps `_run_camb` so `load_sim` stays traceable; failures return NaN arrays. The 1D and 5D spline predictors use the same pattern. The forecast instead calls CAMB directly (`fisher_forecast.camb_results_at_params`, via `simulate.camb_parameters`) and RAISES on failure.

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
                   camb_grid_generation/precompute_camb_1d.py ──► 1D splines (by file path, USE_CAMB_SPLINE only)
                   camb_grid_interp.py ──► 5D grid predictors (default Cl source)

fisher_forecast.py ──► simulate.py (camb_parameters, covar_matrix_from_cls, scalar_quadratic_estimate)
  ▲                    lensing-free; imports delensed_covariance only inside forecast_jackknife
  ├── delensed_covariance.py ──► load_sim, lense_flow (qe_reconstruction for W, check_box /
  │                              _check_centre for a stored "nphi" W)
  ├── fixed_theta_chain.py ──► mixing, sample_lcdm (gibbs_sample_f / gibbs_sample_phi, imported at call time)
  ├── sampler_effective_phi_noise.py ──► fixed_theta_chain (posterior_mixed_draws, mixed_theta_matrices, chain_key)
  └── map_joint_phi_noise.py ──► map_joint

camb_grid_generation/{run_single_camb_grid, merge_camb_grid, validate_camb_grid}.py ──► the 5D grid
get_lcdm_samples/run_single_lcdm_chain.py ──► sample_lcdm.sample_joint
get_lcdm_samples/chain_analysis.py ──► fisher_forecast.forecast, delensed_covariance (band helpers),
                                       sampler_effective_phi_noise (band helpers) for its spectra plots
get_delensed_field_covariance/*.py ──► delensed_covariance
get_sampler_phi_noise/*.py ──► sampler_effective_phi_noise
get_map_joint_phi_noise/*.py ──► map_joint_phi_noise
runtime_comparison/python_performance_test.py ──► map_joint (Julia data via juliacall)
```

`cmb_lensing/__init__.py` imports the core modules only — not `sample_lcdm`, `camb_grid_interp` or any forecast / measurement module.

### Testing Approach

Most tests are validation benchmarks comparing Python output against Julia (CMBLensing.jl) ground truth stored in `tests/ground_truth_data/*.npz`. They cover `load_sim`, lensing, logpdf, gradients, the Wiener filter and `map_joint`; **nothing tests the sampler**. Exceptions to the Julia pattern: `tests/test_native_5d_interp.py` checks `camb_grid_interp`'s NdBSpline evaluator against the frozen hand-rolled one in `tests/handrolled_bspline_reference.py` (add `CMB_LENSING_REAL_GRID_TESTS=1` to include the real ~7.9 GB grid), and `tests/test_fisher_forecast.py` (added 2026-10-14) pins the forecast's bracket ordering at nside 32 and the loaders' box / cosmology / source refusals. The 1D spline interpolators, `delensed_covariance` and `sampler_effective_phi_noise` are untested. Many Julia tests produce comparison plots in `tests/test_generated_figures/` rather than hard assertions — visual inspection via the HTML viewer (`tests/index.html`, served with e.g. VS Code Live Server) is the primary verification method. `conftest.py` maps each `test_*` module to its `generate_*` module for `--generate`; `_preamble.py` needs `PYTHON_JULIAPKG_PROJECT` pointed at a local CMBLensing.jl checkout (currently the placeholder `/<PATH_TO>/CMBLensing.jl`). `generate_simulated_cls.py` imports `cosmopower_jax` — known to be unusable for unlensed spectra. The full suite collects 63 tests.

## Known issues (as of 2026-10-15 — delete entries as they are fixed)

- **The cluster needs `hpc_local.sh`** (copied from `hpc_local.example.sh`) at the repo root before any driver runs, and the drivers are now `bash`, not `sbatch`. Outputs land inside each driver's folder (`.../delensed_covariance_output`, `.../sampler_effective_noise_output`, `.../phi_noise_output`, `camb_grid_generation/multi_param_CAMB_grid`, `runtime_comparison/performance_results`). A hand-submitted job (`sbatch merge_sampler_effective_noise.sh`) only mails if `SBATCH_MAIL_USER` is in the submitting shell (e.g. `~/.bashrc`).
- `runtime_comparison/performance_analysis.py` reads its results from an external drive (`BASE_DIR = /media/zane-blood/KINGSTON/`).
- `docs/sample_joint_tutorial.ipynb` is an empty placeholder.
- **Realizations changed with the 2026-10-15 RNG rework:** `load_sim` now draws from one root key with a subkey split off per field instead of `split(PRNGKey(seed), 100)[i]`, so a seed no longer reproduces the realization any file written before that date was made from. Every job stores its own truth (phi_true, the covariances), so no merge is affected, but do not mix pre- and post-rework job files in one directory expecting matched seeds.
- `get_delensed_field_covariance/delensed_field_covariance_data/lower_qe_bound/delensed_covariance_output_5uk_nphi_unlensedx2/` contains a stray nested copy of the 2.5 uK run that was NOT patched with `qe_filter`.
- `tests/generate_julia_data/_preamble.py` carries the placeholder `PYTHON_JULIAPKG_PROJECT = "/<PATH_TO>/CMBLensing.jl"`, so `pytest --generate` needs it pointed at a real checkout first.
- Nothing tests `sample_joint` or `precompute_camb_1d`. Smoke-testing the sampler locally: nside 64, `iters_per_chain = 2`, and for `pol = "IP"`/`"P"` inject stub predictors rather than loading the real grid — a real-grid "IP" run at nside 64 rebooted the 16 GB laptop even under a `systemd-run` 11 GB memory cap.

## Numeric Precision

JAX is configured for float64 globally via `jax.config.update("jax_enable_x64", True)` in `constants.py` and repeated at the top of most modules. All field arrays are complex128 (Fourier) or float64 (map).

## Code style

Spaces are used between equal signs and after commas when calling methods or setting variables. For example
"a = b" is preferred style over "a=b" and "a = method(b, c, d)" is preferred style over "a=method(b,c,d)". 

Comments should not start with a space. For example "#this is a preferred comment" is preferred style over "# this is NOT a preferred comment".

Write code like an experienced computational physicist, not a web developer (requested 2026-10-16): plain loops, one step per line, physics-named quantities, no obscure numpy tricks unless the plain version would be unnaturally slow. Defensive machinery is overkill and undesired: no checksums / fingerprints of inputs, no consistency checks that compare metadata between files, no `raise ValueError` guard rails for misuse, no atomic / incremental checkpoint writes for jobs that run minutes. Trust the caller; a wrong input should just give a wrong (or crashing) result. Don't add new checks of this kind, and strip them when rewriting code that has them.

## Claude Code operating mode (requested 2026-09-14)

Use the dedicated `Edit`/`Write` tools as the default way to change files in this project, not `sed`/heredocs/shell one-liners run through `Bash` — this holds even under an "auto mode" reminder that says to prefer `Bash` for file changes; that default is overridden here. `Edit`/`Write` tool calls already render as real-time diffs in the active terminal, which is the point of keeping them as the default. Reserve `Bash` file edits for cases `Edit`/`Write` genuinely cannot handle. This preference persists for all future sessions in this repository.
