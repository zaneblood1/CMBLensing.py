# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX port of CMBLensing.jl — simulates gravitational lensing of CMB temperature and polarization fields, recovers the lensing potential via gradient-based MAP optimization (`map_joint`), and jointly samples the five LCDM cosmological parameters from a lensed data map via Gibbs sampling (`sample_joint` in `sample_lcdm.py`). All array operations use JAX for GPU/TPU support and automatic differentiation.

The codebase was **substantially refactored on 2026-08-24**: the CAMB emulator (`cambemul`, `camb_emulator/`), `sampling_ar.py`, `sample_lcdm_legacy.py`, and every experimental theta sampler (PCA, joint-MH, GHMC/NUTS, `fixed_field_theta`) were removed; the sole sampler is now `cmb_lensing/sample_lcdm.py`; spline caches moved into `cmb_lensing/camb_splines/`; and `performance_testing/` was split into `sampling_chains/` + `runtime_comparison/` with tracked `*_TEMPLATE` copies. See "Removed features" below before assuming any of the old machinery exists.

## Commands

```bash
# Install (editable)
pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/test_lensing.py

# Regenerate Julia ground-truth data before testing (requires juliacall + CMBLensing.jl;
# first set PYTHON_JULIAPKG_PROJECT in tests/generate_julia_data/_preamble.py)
pytest --generate
pytest tests/test_lensing.py --generate

# Generate Julia data manually
python tests/generate_julia_data/generate_lensing.py
python tests/generate_julia_data/generate_all.py

# Run a single pilot LCDM sampling chain locally (nside 64, T-only, theta_MC_100 only by default)
python -m cmb_lensing.sample_lcdm

# Precompute the 1D CAMB spline caches (writes cmb_lensing/camb_splines/camb_<param>_grid.npz)
python -m cmb_lensing.precompute_camb_1d <theta_MC_100|logA|ns|ombh2|omch2|all> [--test]

# Build the 5D CAMB grid on an HPC (from a filled-in copy of sampling_chains_TEMPLATE/)
sbatch camb_grid.sh                                        # 875 slurm jobs, one H0 sweep each
python merge_camb_grid.py --grid_dir multi_param_CAMB_grid  # -> camb_grid_spline.npz
python validate_camb_grid.py --grid <...>/camb_grid_spline.npz -n 20

# Launch a multi-map / multi-chain LCDM experiment on an HPC
sbatch sample_lcdm.sh            # spawns run_single_lcdm_chain.sh per (map, chain)
python chain_analysis.py         # post-process the *_history.txt chains into distributions

# Fisher forecasts (one module per method; all write to cmb_lensing/fisher_output/ under their own suffix)
python -m cmb_lensing.fisher_forecast --spectra ceiling                 # "blocks": the covariance-block trace formula
python -m cmb_lensing.fisher_forecast_from_cls --spectra lensed         # "cls": bandpower Fisher + lensing NG covariance
python -m cmb_lensing.fisher_forecast_full_sky                          # "full_sky": (2l+1)/2 f_sky sum over the GRID's ells
python -m cmb_lensing.fisher_forecast_full_sky --ell_source camb        # over CAMB's integer ells instead
python -m cmb_lensing.fisher_forecast_full_sky --delta_ell 50 --ell_max 2160
python -m cmb_lensing.fisher_forecast_from_1st_principles --ell_max 3000 --delta_ell 20   # "1st_principles": 1D CAMB Cls, analytic QE noise
python -m cmb_lensing.fisher_forecast_from_logpdf --fast --nside 64     # "logpdf": <-d2 logpdf> over prior draws
python -m cmb_lensing.fisher_forecast_from_mixed_logpdf --realizations 2 --nside 64   # "mixed": same for mixed_logpdf
sbatch mixed_hessian.sh                                                 # one job per realization, from a filled-in sampling_chains/
python -m cmb_lensing.fisher_forecast_from_mixed_logpdf --hessian_dir mixed_hessian_output

# Empirical delensed spectrum: measure R(l) = C^delensed[box] / C^delensed[CAMB] (from a filled-in sampling_chains/)
sbatch get_delensed_spectra.sh                                          # 100 jobs, one seed each
python merge_delensed_spectra.py --spectra_dir delensed_spectra_output  # -> transfer_function.npz
python compare_transfer_functions.py --reference <dir_a> --shifted <dir_b>   # is R flat in theta?
python -m cmb_lensing.fisher_forecast --spectra delensed --transfer_function <...>/transfer_function.npz
```

No linter or formatter is configured. No CI pipeline exists.

## Repository Layout

```
cmb_lensing/                  the Python package
  camb_splines/               spline caches: five tracked ~3 MB camb_<param>_grid.npz + the gitignored ~8 GB camb_grid_spline.npz
docs/map_joint_tutorial.ipynb load_sim / lense_flow / logpdf / map_joint walkthrough
sampling_chains_TEMPLATE/     tracked slurm + analysis scripts for LCDM chains and the 5D CAMB grid
runtime_comparison_TEMPLATE/  tracked slurm scripts timing map_joint Python vs Julia
sampling_chains/              gitignored LOCAL copy of the template with real paths/emails filled in
runtime_comparison/           gitignored LOCAL copy of the template with real paths/emails filled in
tests/                        Julia A/B benchmarks (see Testing Approach)
```

**Template convention.** `.gitignore` excludes `sampling_chains/` and `runtime_comparison/`; the `*_TEMPLATE` directories are what ships. The only intended differences between a live dir and its template are placeholders: `ABSOLUTE_PATH_TO/...` for HPC paths, `<USER>@<INSTITUTION>.edu` for `#SBATCH --mail-user`, `FILE_CONTAINING_LEARNED_DATA` in `chain_analysis.py`, and `<ABSOLUTE_PATH_TO>/CMBLensing.jl` in the runtime-comparison scripts. **Code changes must be made in both copies** (the live copy is what actually runs; the template is what gets committed). `run_single_lcdm_chain.py` and `run_single_camb_grid.py` are byte-identical between the two.

`cmb_lensing/camb_splines/camb_grid_spline.npz` (~8 GB) is gitignored — it was once committed by accident and pushed a pack past GitHub's 2 GiB limit; never force-add it.

## Architecture

### Computational Pipeline

`simulate.load_sim(nside, theta_pix, pol, master_seed, ...)` is the main entry point. It runs CAMB (through `jax.pure_callback`, so a CAMB failure yields NaN spectra → non-finite logpdf, never an exception), builds covariance matrices and random fields in Fourier space, lenses them via `primal_lense_flow`, and returns a `DataSetT` / `DataSetEB` / `DataSetTEB` (`dataset.py`) holding the fields, operators, and simulated data. Notable current choices, all made for LCDM inference:
- **Mask is forced to all ones** (`jnp.ones_like(get_mask(...))`) — there is no sky mask in any data set. This also applies to the Julia-comparison tests.
- **Noise:** `noise_cls(..., l_knee, alpha_knee = 3)` includes the 1/f term `1 + (l_knee/ell)**3`; `load_sim`'s default is `l_knee = 100` (matches Julia), while every sampler entry point (`sample_lcdm.__main__`, `run_single_lcdm_chain.py`, `add_starting_matrices_to_args`) passes `l_knee = 0` for pure white noise. Beam FWHM defaults to 0.
- **Cls are CAMB-only.** The `use_emulator_cls` flag and the `cambemul` import are gone.
- **The cls dict carries the T-phi cross spectrum** (added 2026-09-10): `_run_camb` keeps the PP and PT columns of `get_lens_potential_cls(CMB_unit = "muK")` (PP is dimensionless either way; "muK" puts PT in the TT units), the callback returns a `(lmax_prime, 2)` array, and `_extract_all_cls` adds `cls["TP"]` via `dl2cl(..., is_tphi = True)` = D · 2π / (L(L+1))^{3/2}. Note `cls["phi"]` still uses the ℓ⁴ convention, so `TP/√(TT·PP)` differs from CAMB's correlation coefficient by (1 + 1/ℓ). TP changes sign near ℓ ~ 1100 (1e-16 level), so it must not go through the log-interpolating `covar_matrix_from_cls`; `fisher_forecast._covar_linear` interpolates it linearly and the "lensed" `covariance_blocks` carry it as a `"TP"` key. The 5D grid and the 1D spline caches do not have it. `EXPECTED_CL_KEYS` includes `"TP"`, so a `precomputed_cls` dict must too.
- Tensor covariances are added back in (`cf = cf_scalar + cf_tensor`), but every sampler path runs at `r = 0`.
- `get_d_tt_matrix(cf, cn)` (2 args), `get_d_teb_matrix`, `get_d_eb_matrix` build the D mixing matrix; `get_g_matrix(cphi, nphi, a_phi_fid, a_phi)` / `get_g_matrix_lcdm(cphi_fid, cphi_curr, nphi, cn_tt)` build G (`cn_tt` is unused there). `lmax_prime = min(lmax, DEFAULT_MAX_ELL)` with `DEFAULT_MAX_ELL = 4000` in `constants.py` (renamed from `EMULATOR_MAX_ELL`).

`map_joint(data_set, num_steps = 30, constant_step = False)` recovers f and phi by alternating Wiener-filter steps (estimate f) with a BFGS line search on phi along the preconditioned mixed gradient (`mixed_grad_phi_logpdf`, hessian = `pinv(Cphi^-1 + Nphi^-1)`). G is set to the identity there (MAP is invariant to it).

`sample_joint(...)` (in `sample_lcdm.py`) draws samples from the full joint posterior P(f, phi, theta | d) via Gibbs sampling, where theta are the five LCDM parameters. This is the code path the project is centred on.

> **DEFAULT FILE FOR CONVERSATIONS:** when the user refers to "the sampler", "sample_joint", "gibbs_sample_theta", theta/parameter sampling, etc., they mean **`cmb_lensing/sample_lcdm.py`**. `sample_lcdm_legacy.py` and `sampling_ar.py` no longer exist.

### The LCDM sampler (`sample_lcdm.py`)

**Parameters.** `PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]` (tau is pinned at 0.05 by prior, mnu at 0.06). `current_params` is a `(5,)` jnp array in that order; user-facing dicts (`param_init`, `proposal_sigmas`, `param_ranges`, `should_sample`) are keyed by name. Single sources of truth imported from `precompute_camb_1d.py`: `GROUND_TRUTH` (fiducial values), `PARAM_BOUNDS` (±5-sigma search box), `PARAM_SIGMA` (renamed from `TRAINING_SIGMA`; used only to normalize the progress plot). `to_camb_naming_conv` converts `theta_MC_100 -> cosmomc_theta = theta/100` and `logA -> As = exp(logA)*1e-10` **in place** (it deletes the old keys from the dict it is given).

**Cl source.** Module-level switches, mutually exclusive (a `ValueError` fires at import if both are True):
- `USE_CAMB_GRID = True` (**default**) — `get_camb_grid_predictors()` lazily loads the 5D grid once per process from `CAMB_GRID_PATH = cmb_lensing/camb_splines/camb_grid_spline.npz` and returns a `GridPredictors` namedtuple (see the grid section). Required for `pol = "IP"` / `"P"`. `sample_joint` checks that `param_init` is inside the grid (finite TT) before doing anything heavy.
- `USE_CAMB_SPLINE = True` — 1D cached-CAMB splines; exactly one `should_sample` entry may be True. `wrap_temp_predictors` lifts the `(predict_tt, predict_pp)` pair into a `GridPredictors` with `None` for every other spectrum (T-only).

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

**`__main__`** builds an nside 64, theta_pix 5, T-only, 5 µK-arcmin, `l_knee = 0` data set at `GROUND_TRUTH`, samples only `theta_MC_100` starting from the lower edge of its box, `phi_init = "ZEROES"`, `num_burn_in_fix_theta = 0`, `seed = 67`. The HPC template (`run_single_lcdm_chain.py`) samples omch2 / theta_MC_100 / logA from random starts inside the box, `num_burn_in_fix_theta = 100`, 1500 iterations, default `phi_init` (= `"MAP"`).

### Cached-CAMB 1D Splines (`precompute_camb_1d.py`)

`python -m cmb_lensing.precompute_camb_1d <param|all> [--test]` sweeps one parameter over `PARAM_BOUNDS` (50 nodes) with the other four pinned at `GROUND_TRUTH`, writing `cmb_lensing/camb_splines/camb_<param>_grid.npz` (`default_cache_path`). `load_camb_spline_predictors(param_name)` returns `(predict_tt, predict_pp)` — cubic splines in lnCl through the CAMB nodes, jit-safe via `jax.pure_callback`, raising if any non-swept parameter moves off its cached value, NaN beyond the cached grid. Cls live on ells 2..`DEFAULT_MAX_ELL`-1 (`CAMB_LMAX`) — the exact support of `load_sim`'s CAMB path, so model and data share the same log-log extrapolation anchor (a one-multipole mismatch measurably biased the ombh2 conditional). All five caches were generated with CAMB 2.0.0 (July 2026); regenerate if CAMB is upgraded. theta_MC_100 caveats: theta_MC_100 ≳ 1.117 needs H0 > 100, which is outside CAMB's *default* `theta_H0_range`, so the shipped cache lost 8 nodes and ends at 1.114375 (+3.1σ) instead of the box top 1.1452 — `_run_camb` now passes `DEFAULT_THETA_H0_RANGE = [10, 150]` (`constants.py`), so re-running the sweep recovers all 50 nodes. Until it is regenerated, spline-path proposals above 1.114375 get a NaN logpdf and auto-reject, which matters most for `run_single_lcdm_chain.py`'s random-in-box starts. Its TT spline error (~5e-3 lnCl leave-half-out) is the largest — densify first if its conditional looks suspicious.

### 5D CAMB Grid Spline (`camb_grid_interp.py`)

Real CAMB spectra precomputed on a full tensor-product grid over `(H0, logA, ns, ombh2, omch2)` (tau 0.05, mnu 0.06, r = 0, tensors off) and evaluated as a tensor-product cubic B-spline in lnCl over coefficients prefiltered at merge time with not-a-knot ends. Out-of-box queries return NaN → rejected proposal.

- **Grid file:** `cmb_lensing/camb_splines/camb_grid_spline.npz` (~7.9 GB, gitignored). Built on the HPC by `sampling_chains_TEMPLATE/camb_grid.sh` → `run_single_camb_grid.sh/.py` (one slurm job per (logA, ns, ombh2, omch2) node doing the 81-point H0 sweep; 81×5×5×5×7 = 70 875 CAMB calls in 875 jobs; numpy + camb only, no JAX) → `merge_camb_grid.py` (streaming, one spectrum at a time, incremental zip writes; asserts no holes, uniform axes, theta independent of logA/ns, |te_rho| < 1) → `validate_camb_grid.py` (random interior points vs direct CAMB; expect ≲ CAMB's own ~1e-3 lnCl floor).
- **Evaluation is `scipy.interpolate.NdBSpline`** (`_TensorSpline`), swapped in 2026-09-07 for ~150 lines of hand-rolled B-spline machinery (`_axis_stencil`, `_axis_stencil_derivs`, `_eval_bspline`, `_eval_bspline_grads`). This works with **no regeneration of the grid file**: `merge_camb_grid.py` already writes exactly NdBSpline's input (coefficients + knots with `len(t) == n + k + 1` per axis). Agreement with the old evaluator is ~4e-14 relative on the real grid, at ~3× the speed. Two things `_TensorSpline` still owns: the `BOUNDARY_TOL` clamp (bare `NdBSpline(extrapolate = False)` NaNs the box edges the sampler's conditional scans land on) and NaN-not-raise for genuine out-of-box rows. Cost: NdBSpline casts coefficients to float64, so a spectrum costs ~2.1 GB resident instead of ~1.1 GB — `CambGrid.spline` therefore caches only the spline, never the float32 array. `RegularGridInterpolator(method = "cubic")` is **not** an equivalent shortcut despite building an NdBSpline internally since scipy 1.12: it re-solves for coefficients with an iterative Krylov solver (`gcrotmk`), ~4e-4 away in lnCl from `make_interp_spline`'s direct banded solve, at ~10 min per spectrum. The old implementation is frozen in `tests/handrolled_bspline_reference.py` and `tests/test_native_5d_interp.py` checks production against it (`tests/benchmark_native_5d_interp.py` times them).
- **`CambGrid`** opens the npz once (`load_camb_grid` singleton per path) and builds `_TensorSpline`s **lazily per spectrum** via `CambGrid.spline(name)` (~1.1 GB read, ~2.1 GB resident, six of them) — T-only runs never touch the polarization/lensed tables. `bb_is_zero` flag: unlensed BB is identically zero on the r = 0 grid, so the `bb` predictor returns exact zeros (NaN out of box). A pre-polarization grid file stays loadable and raises a `KeyError` with a regeneration pointer only when a missing spectrum is requested.
- **`load_camb_grid_predictors(path)`** → `GridPredictors(tt, ee, bb, pp, tt_lensed, ee_lensed, bb_lensed, te_rho)`, each `f(params_batch) -> (M, n_ell)` via `jax.pure_callback(vmap_method = "sequential")`, cached per path. `LINEAR_SPECTRA = {"te_rho"}` are splined raw (no exp). This is the **only** predictor loader now — the pure-JAX (`_jax`) and custom-VJP (`_grad`) differentiable variants were removed with the gradient-based theta samplers. `CambGrid.spline_param_grads` (analytic ∂lnCl/∂params through the theta→H0 inversion) survives unused.
- **H0 vs theta_MC_100.** The grid is laid out in H0 because a rectangular theta box has CAMB-unsolvable corners. Every node records theta; the merge collapses it to a 3D `theta_grid(H0, ombh2, omch2)`; `CambGrid.h0_from_theta` densifies theta(H0) to 2000 points, splines it over (ombh2, omch2), and inverts by interpolation. The sampler works in theta_MC_100 throughout.
- `tests/test_camb_grid_interp.py` (the tensor-product reference check) was deleted in the 2026-08-24 refactor; `tests/test_native_5d_interp.py` (added 2026-09-07) now covers the evaluator, though nothing still covers the merge or the theta→H0 inversion end to end.

### Fisher forecasts (`fisher_forecast*.py`)

Six Gaussian / Hessian estimates of the parameter information, **one module per method** since
the 2026-09-10 split (the sixth, `1st_principles`, was added 2026-09-11). All write to `cmb_lensing/fisher_output/` under a per-method filename suffix
(each module's `METHOD` / `SUFFIX` / `LABEL` constants) so they never overwrite each other, and each
has its own `python -m` CLI (see Commands). The five agree bit-for-bit with the pre-split
single-file implementation.

| module | method | what it contracts |
|---|---|---|
| `fisher_forecast.py` | `blocks` | `F = 1/2 sum_k w_k dlnC/di dlnC/dj` on the rfft grid (the original; what `chain_analysis.py` imports) |
| `fisher_forecast_from_cls.py` | `cls` | bandpower Fisher `dmu^T Cov(mu_hat)^-1 dmu` with the lensing non-Gaussian covariance |
| `fisher_forecast_full_sky.py` | `full_sky` | `sum_l (2l+1)/2 f_sky Tr[...]` over a 1D ell axis (grid or CAMB ells) |
| `fisher_forecast_from_1st_principles.py` | `1st_principles` | `sum_l (2l+1)/2 f_sky dC (C+N)^-2 dC` over TT and phiphi on CAMB's integer ells, with N_phi from the Hu & Okamoto N^(0) integral (`qe_noise_spectrum`) rather than the box matrix |
| `fisher_forecast_from_logpdf.py` | `logpdf` | `<-d2 statistics.logpdf>` over prior draws of (d, f, phi) |
| `fisher_forecast_from_mixed_logpdf.py` | `mixed` | `<-d2 statistics.mixed_logpdf>` at a fixed mixed pair; fanned out by `mixed_hessian.sh` |

**`fisher_forecast.py` is the base module** and the only one the others import from: it owns the
memoized direct-CAMB runs (`camb_cls_at_params`, `_CAMB_CACHE` - shared across modules within a
process, so running two methods costs one set of `2 * n_sampled + 1` CAMB calls), the
theta-independent instrument matrices, `qe_noise_matrix`, the delensing residual, `covariance_blocks`
and `covariance_stencil` (the shared block stencil `cls` and the grid-source `full_sky` both read),
the finite-difference helpers (`sampled_names_and_steps`, `stencil_offsets`, `hessian_from_stencil`,
`prior_covariances`, `load_sim_cosmology`), `covariance_from_fisher`, `step_stability` (takes a
`forecast_fn(step_fracs)` closure), the plotting (`annotated_heatmap`, `plot_annotated_matrix`, ...),
`write_outputs` / `save_outputs`, and the shared argparse pieces (`add_box_arguments`,
`add_spectra_arguments`, `sampled_from_args`, `box_subtitle`, `run_config`). `forecast(...)` is the
blocks method itself (no `method` kwarg any more); the siblings expose `forecast_from_cls`,
`forecast_full_sky`, `forecast_from_logpdf`, `forecast_from_mixed_logpdf` with the same leading
arguments. There is no `forecast_all` / `--method both|all` - run the CLIs separately.

**`--qe_response`: the QE response spectrum (added 2026-09-15).** `QE_RESPONSE_SOURCES` picks which
TT spectrum weights the quadratic estimator's response `f(l, l')`, and so sets its normalization
`N^(0)`. It is independent of `--nphi_source` and applies to both of its settings:
- **`unlensed` (DEFAULT, `DEFAULT_QE_RESPONSE`)** `C_l^TT`, Hu & Okamoto's original expression and
  the spectrum the sampler's own QE norm uses. Reproduces every number these modules produced before
  2026-09-15, bit for bit, and costs no extra CAMB work.
- **`gradient`** CAMB's lensed temperature-gradient spectrum `C_l^(T grad T)`
  (`get_lensed_gradient_cls` column 0). The response is a derivative of the LENSED temperature with
  respect to the deflection, so this is the correct weight (Lewis, Challinor & Hanson 2011); it
  resums the N^(2) bias into the normalization.

New in `fisher_forecast.py`: `gradient_cls_at_params` (memoized in its own `_GRADIENT_CACHE`, ~5 s
per cosmology, so it is deliberately NOT folded into `camb_cls_at_params` - only the fiducial point
would need it, and the default never asks for it at all), `cls_with_qe_response(params,
qe_response)` (a no-op on `unlensed`; on `gradient` it adds a `"gradient_TT"` key - what
`covariance_stencil`, `fisher_forecast_full_sky` and `fisher_forecast_from_cls` build their
`cls_fid` from), `qe_response_cl(cls, qe_response)` (the accessor, which RAISES rather than falling
back to `scalar_TT` when `gradient` is asked for without the key, mirroring the `"delensed_TT"`
guard) and `add_qe_response_argument` (its own argparse helper, since `1st_principles` has no
spectra-mode knobs; `add_spectra_arguments` calls it so blocks / cls / full_sky get it for free).
`qe_noise_spectrum`'s first spectrum argument is renamed `cl_tt_unlensed -> cl_tt_response`.
The kwarg threads through `qe_noise_matrix` / `qe_noise_cl` / `qe_noise_grid` /
`iterative_delensing` / `frozen_reconstruction` / `covariance_stencil` and every `forecast*`
entry point, and is recorded in each `fisher<suffix>.npz` by `run_config`.

**This is isolated to the forecasts:** `simulate.scalar_quadratic_estimate` is untouched and
`load_sim` / `map_joint` / `sample_lcdm` keep their unlensed response, where the QE norm only
preconditions G and the phi mass matrix. Measured at nside 64 / 5' / 5 uK, T-only, `gradient` vs
`unlensed`: `N_phi` rises 0.4-2.6% (covariance source) and up to 19% at L < 100 (hu_okamoto - low L
is sensitive to the response's SLOPE through the near-cancellation of the two `L . l` terms, and the
unlensed spectrum's acoustic peaks are not smoothed, so its slope is wrong there - a uniform rescale
of the response, by contrast, moves `N_L` by exactly `1/scale^2` at every L). The marginalized
sigmas move < 0.35% (blocks, all spectra modes) and < 0.03% (1st_principles), because `C_phi`
dominates `N_phi` over the modes carrying the Fisher weight on this box; expect more on a noisier or
finer box. Higher-order reconstruction biases are still absent and mostly do not belong: N^(1) is
linear in `C_phi`, so it is a property of the signal, not a per-mode noise, and has no slot in a
`C_phi + N_phi` block.

**`blocks` vs `cls`.** With a covariance diagonal in the modes `cls` IS `blocks` (verified to 1e-17),
and that is what it returns for `ceiling` / `unlensed`. Its purpose is the **lensing-induced
non-Gaussian covariance** (`cmb_lensing/lensing_covariance.py`,
added 2026-09-09) for `lensed` / `delensed`: the realization's phi power moves the TT power at every
multipole through the first-order flat-sky kernel `K(l, L) = dC_lensed(l)/dC_phi(L)` (Hu 2000;
Benoit-Levy, Smith & Hu 2012), giving a TT-TT covariance `2 K diag(C_eff^2) K^T` and, for `lensed`, a
TT-phiphi cross term `2 K diag(C_phi^2)` (Schmittfull+13 / Peloton+17's signal term). `C_eff` is the
full `C_phi` for `lensed` and the per-mode residual `C_phi N_phi/(C_phi + N_phi)` for `delensed`,
whose cross term vanishes (a Wiener residual is uncorrelated with the recovered part). It is applied
matrix-free (three FFT convolutions per kernel apply, circular because the box is periodic) and
inverted by preconditioned CG on the stacked (TT, phiphi) vector of `independent_modes` (one entry
per conjugate pair - the rfft half plane holds the two self-conjugate columns' pairs twice, which a
covariance cannot carry); the kernel is symmetrized under joint negation of its wavevectors because
the periodic grid's Nyquist sign breaks that symmetry. `--cls_gaussian_only` switches it off. Every
verbose run prints the kernel's first-order correction against CAMB's `C_lensed - C_unlensed` per
annulus (0.93-1.00 below ell 1000 at nside 128; the ~1000-1500 annulus is where that correction
crosses zero). `tests/test_lensing_covariance.py` pins the operator against a brute-force kernel
matrix on nside 8/10. **Measured effect at nside 128 / 2.5' / 5 uK, T-only: about 1%** - per-mode
TT-TT correlations reach 0.045 at ell 2500 and TT-phiphi 0.12 against the lowest phi mode, which is
the literature's size once bandpower binning is undone, so on this box it does not move the
`lensed` forecast (r(omch2, theta) +0.085 -> +0.060). `--spectra` picks `lensed` / `unlensed` /
`ceiling` / `delensed`; parameters come out in `OUTPUT_PARAM_ORDER` (`omch2, ombh2, ns,
theta_MC_100, logA`), matching `chain_analysis.py`'s `ground_truth_values` order rather than
`PARAM_ORDER`.

**`--spectra delensed` (reworked 2026-09-11).** The f block is now `C_TT^delensed + C_n` with
`C_TT^delensed` from CAMB's own `get_partially_lensed_cls` - the full non-perturbative
correlation-function lensing run with `C_L^phiphi` scaled per multipole by
`Alens_L = N_L / (C_L^phiphi + N_L)`, the residual fraction a Wiener-filtered reconstruction leaves
(NOT a linear interpolation between unlensed and lensed, which it differs from at the percent
level). `--nphi_source` picks the `N_L` behind `Alens_L`: `covariance` (default) azimuthally
averages the box's `qe_noise_matrix` (`_radial_cl_profile`, now in the base module), `hu_okamoto`
uses the analytic `qe_noise_spectrum` (also moved into the base module; the `1st_principles` module
imports it). Since 2026-09-14 the flag also picks the phi block's 2D `N_phi` in EVERY spectra mode
(`qe_noise_grid`): the box matrix, or the Hu & Okamoto spectrum put on the rfft grid by
`covar_matrix_from_cls` like `C_phi` (isotropic, unlike the box matrix). `qe_noise_matrix` no longer
divides by `NPHI_FAC` (that factor is a sampler preconditioning choice that cancels inside G).
**Also since 2026-09-14, `hu_okamoto` is capped at the box, not at `DEFAULT_MAX_ELL`:** `qe_noise_cl`
cuts the integral's multipole axis at `grid_max_ell(ell_grid)` = the rfft grid's corner mode
`sqrt(2) pi / pix_width` (3055 at 5', 6109 at 2.5', independent of nside - nside sets the
fundamental, not the corner), so both legs `l` and `|L - l|` stay inside the range the box measures
and the two sources now differ only in the SHAPE of the domain (isotropic annulus vs the periodic
square), not in how far it reaches. The returned axis is still CAMB's 2..`CAMB_LMAX`-1, log-log
continued past the cap with the same warning `_radial_cl_profile` prints for the covariance source.
At 2.5' the corner clears CAMB's range and the result is bit-for-bit unchanged; at 5' the cap raises
`N_phi` by 1.6x at L = 100, 2.5x at L = 1000 and 10x at L = 3000, dropping the mean delensing
efficiency from 0.99 to 0.68 - so any pre-2026-09-14 `hu_okamoto` number at 5' or coarser is stale.
`fisher_forecast_from_1st_principles.py` is deliberately NOT capped: it calls `qe_noise_spectrum`
directly on its own `--ell_min` / `--ell_max` axis, whose whole point is to count modes past CAMB's
range out to the Nyquist and the corner. `Alens_L` is frozen at the fiducial
point and `delensed_cls_at_params(params, alens)` re-delenses at every stencil point, which needs
CAMB's results object: `camb_results_at_params` memoizes `(pars, results)` (built by
`simulate.camb_parameters`, split out of `_run_camb`) and `camb_cls_at_params` now reads its spectra
from that object instead of the pure_callback path (bit-for-bit identical). `--iterative_delens`
iterates `Alens_L` against `N_L` (five iterations to 1e-6 at nside 64) and recomputes the box matrix
with the delensed filter. `delensing_residual` and the scalar-alpha residual are gone; the block key
is `f_delensed` (also in `fisher_forecast_from_cls._CLS_BLOCK_KEYS` and the full-sky `cl_blocks`, which
now adds `C_n` like the others). Caveat carried over from "lensed": the residual lensing tracks
`C_phi(theta)`, so the f block gains omch2 information the phi block counts again - delensed lands
BELOW ceiling in omch2 (0.0042 vs 0.0069 at nside 64 / 2.5' / 5 uK) and above it in theta / logA.
NOTE the working tree's `covariance_blocks` had lost its `"ceiling"` branch (returned `None`, the
CLI default) while the `"lensed"` branch was being reworked to carry `scalar_TT` + a `TP` pseudo-block;
`"ceiling"` now shares the `"unlensed"` branch so the default runs, but the two are identical until
that edit is finished, and the `TP` block treated as an auto-spectrum makes "lensed" implausibly tight.

### The empirical delensed spectrum (`delensed_spectrum.py`, added 2026-09-15)

`--spectra delensed` takes its delensed TT spectrum from CAMB (`get_partially_lensed_cls`, full-sky
correlation-function lensing with `C_L^phiphi` scaled by `Alens_L`). That is **not** how this
codebase lenses (`primal_lense_flow`: a LenseFlow ODE on a periodic flat-sky box, 10 RK4 steps) or
reconstructs (`map_joint`'s MAP estimate, not a Wiener-filtered QE with residual `N/(C+N)`). The new
`cmb_lensing/delensed_spectrum.py` measures the disagreement as a **transfer function**

```
R(l) = C_l^delensed [box, lense_flow + map_joint] / C_l^delensed [CAMB, same frozen Alens_L]
```

at the fiducial cosmology only, and `fisher_forecast --transfer_function <npz>` rescales CAMB's
delensed spectrum by it **at every stencil point**, so the contracted derivative is
`R * dC_CAMB/dtheta`. Measuring R at one cosmology is the whole point: finite-differencing an
independent Monte Carlo at each stencil point divides the MC noise by `2h` with `h = 0.05 sigma`.

- **Key numbers, measured 2026-09-15 at nside 64 / 5':** `C_l = |rfft2(f)|^2 * pix_width^2 / nside^2`
  (verified to 1e-5 on 400 draws — `grid_power_spectrum`); the inverse-lensing round trip
  `L(-phi)L(phi)f = f` is exact to **1.4e-8** (LenseFlow integrates its ODE backwards, so delensing
  is not a Taylor remapping).
- **Two R estimators.** `paired` (the default) divides by the SAME realization's unlensed spectrum
  before bringing CAMB in, cancelling the cosmic variance; `naive` does not. On one realization the
  paired estimator sat in 0.98-1.03 over the first six bands where the naive one spanned 0.79-1.16 —
  that gap is the whole reason the measurement is affordable. They must agree in the MEAN; the merge
  prints their difference, and a real disagreement invalidates the run.
- **Three self-validation rungs** reported by every job: rung 0 (measured unlensed vs the input
  `C_f` — pins the normalization), rung 1 (measured lensing vs CAMB's — validates the chain), rung 2
  (the inverse round trip). A broken run announces itself in the slurm log.
- **`fields.map` / `fields.fourier` have no guardrails** — they apply `irfft2`/`rfft2`
  unconditionally rather than checking `field.basis`, so calling `fourier` on an already-FOURIER
  field raises "only real valued inputs supported for rfft". `_to_fourier` / `_to_map` in
  `delensed_spectrum.py` check the basis; do the same anywhere fields arrive from mixed sources.
- **R is applied by linear interpolation and HELD CONSTANT outside its measured bands** — it is a
  ratio near one, so `covar_matrix_from_cls`'s log-log continuation would be meaningless.
  `check_transfer_function` refuses an R measured on a different box (nside / theta_pix / noise /
  l_knee), and `covariance_stencil` raises if `--transfer_function` is passed with any `--spectra`
  but `delensed`.
- **Flatness in theta is an ASSUMPTION and must be tested.** `compare_transfer_functions.py` takes
  two merged runs at cosmologies separated by many sigma and reports their ratio, a chi-squared, and
  — the number that decides it — the drift **rescaled to one finite-difference step**, which is the
  systematic the assumption puts into `dC^delensed/dtheta`. If R does move, the fallback is a
  per-stencil-point MC with common random numbers (same seeds at theta, theta±h) so the MC noise
  cancels in the difference instead of being amplified.
- **The comparison's error mode matters more than anything else in that script.**
  `get_delensed_spectra.sh` uses the same `seed_prefix` for every run, so the two cosmologies are
  measured on the SAME realizations (common random numbers) — which is the right way to run it, but
  it makes the two jackknife errors strongly CORRELATED. `ratio_and_error` detects shared seeds and
  jackknifes the **per-realization ratio** instead of adding the two errors in quadrature.
  Measured 2026-09-15 on a 4-realization pair: the correct paired error is ~30x tighter, which flips
  the raw test from "chi^2/band 0.03, flat" to "chi^2/band 56.7, moves at 8.7 sigma". Propagating
  independently is not conservative here — it hides real motion.
- **Detectable and important are different questions, and the script reports them separately.** In
  that same pair, R genuinely moves (1.5% at l ~ 2850 over a 0.59 sigma shift in omch2) but rescaled
  to ONE finite-difference step the drift is 1.3e-3 worst / 3.6e-4 mean — negligible. The VERDICT is
  therefore decided by the rescaled drift against `--tolerance` (default 1e-2), with the chi-squared
  reported only as "is the motion detectable at all". With enough realizations the ratio will always
  become distinguishable from flat; that is not a reason to abandon the construction.
- **Dropping the flatness assumption (added 2026-09-18): measured dR/dtheta.** Set
  `derivative_params` (and `derivative_sigma`, default 0.5 sigma) in `get_delensed_spectra.sh`: it
  runs `$out_dir/reference` plus `$out_dir/<param>_plus|_minus` on the SAME seeds, then
  `merge_delensed_spectra.py --spectra_dir <out>/reference --shifted_dirs <out>/*_plus <out>/*_minus`
  (`delensed_spectrum.merge_transfer_derivatives`) differences R PER REALIZATION and jackknifes it,
  writing `transfer_derivative` (+ error, schemes, deltas, second difference) into the same
  `transfer_function.npz`. `fisher_forecast.apply_transfer_function(..., params)` then evaluates
  `R_0 + sum_i (theta_i - theta_0,i) dR/dtheta_i` at every stencil point (`transfer_at_params`);
  `check_transfer_derivatives` requires R_0's cosmology == the forecast's fiducial point and warns
  about sampled parameters with no derivative (held flat). **Convention:** shifted runs now default
  to `--reconstruction fiducial` (`freeze_reconstruction=1`): data at the shifted theta, but
  map_joint's C_f / C_phi / D / QE norm and the CAMB reference's Alens_L at GROUND_TRUTH
  (`measure_delensed_spectrum(reconstruction_params = ...)`), matching the forecast's frozen
  Alens_L; the derivative merge refuses runs whose reconstruction tracked theta, which is what
  every pre-2026-09-18 shifted run did (files without `reconstruction_params` are read that way).
  Measured on one nside 64 / 5' seed: per-band R scatter 12%, per-band R(+0.5 sigma omch2) - R(0)
  median 0.15% — the common random numbers cancel ~80x.

Scripts (in BOTH `sampling_chains/` and `sampling_chains_TEMPLATE/`, `.py` byte-identical):
`get_delensed_spectra.sh` (one slurm job per seed, 100 by default) -> `get_single_delensed_spectra.sh`
-> `get_single_delensed_spectra.py` -> `merge_delensed_spectra.py --spectra_dir` (writes
`transfer_function.npz` + `.png`) -> `compare_transfer_functions.py --reference --shifted`. The
collection mirrors `load_hessian_directory`: it refuses mixed configurations, duplicate seeds, and —
unique to this one — files taken at **different cosmologies**, since averaging those would destroy
the very theta dependence the flatness test looks for. `--shift_param` / `--shift_value` on the job
script displace the fiducial point for that test; each shifted run needs its own `out_dir`.
MEASURED ~40 s per realization at nside 64 / 5' (load_sim + `map_joint` + 3 lensing solves), so the
100-job set is minutes of wall clock on the cluster and well under an hour locally at that box.

**`fisher_forecast_full_sky.py` is the textbook forecast formula**,
`F_ij = sum_l (2l+1)/2 f_sky Tr[C_l^-1 dC_l/di C_l^-1 dC_l/dj]`. Same likelihood and same
two powers of `C^-1` as `blocks`; only the mode counting changes - `blocks` weights each
rfft entry by its real DOF `w_k` (summing to `nside^2`), this weights each multipole by the
`2l+1` modes a sky of `f_sky = (nside*pix_width)^2/(4 pi)` would carry (`sky_fraction`; the
mask is all ones, so the patch IS the box). Suffix `_full_sky`.

`--ell_source` picks where the multipoles and spectra come from:
- **`grid` (DEFAULT)** the DISTINCT `|l|` the rfft grid carries (`grid_ell_axis`), with the
  grid's own irregular spacing as the per-multipole width, and the blocks read off
  `covariance_stencil` — so the Cls are *exactly the ones the sampler sees* (`covar_matrix_from_cls`'s
  log-log interpolation onto `ell_grid`, same mask/beam/`C_n`/`N_phi`/delensing residual).
  All three methods then contract ONE identical set of matrices and only the mode counting
  separates them; it costs no extra CAMB calls at all. The one approximation is that a 1D
  ell sum cannot carry an anisotropic covariance: everything `covar_matrix_from_cls` builds
  is isotropic to 3e-8, but `N_phi` (FFT convolutions on a SQUARE box) scatters up to 78%
  within one `|l|`. Measured cost at nside 64 / 5', with mode counting held fixed: f block
  exact to 5e-14, phi block 1.8%, Fisher diagonal 1.4%/0.02%/0.03%, marginalized sigmas
  0.7%/0.007%/0.02%. Every run prints the per-block scatter.
- **`camb`** the raw CAMB spectra on their native integer multipoles (2..`CAMB_LMAX`-1) —
  free of the box's resolution, so this is the one to quote against the literature, but it
  is not the model the sampler runs on. Here `N_phi` must cross bases, so
  `_radial_cl_profile` azimuthally averages it and log-log EXTRAPOLATES past the grid's
  largest mode, with a printed warning.

Other knobs: `--ell_min` / `--ell_max` (default = the chosen source's own edges) and
`--delta_ell` (extra band binning on top of the source's spacing; it pools `C` and `dC`
before contracting, so it LOSES information as `delta_ell` grows past the acoustic width).

**`fisher_forecast_from_1st_principles.py`** (added 2026-09-11) is the same `(2l+1)/2 f_sky`
sum written for two diagonal blocks, TT (`--spectra lensed|unlensed`) and phiphi, on integer
multipoles with `--ell_min` / `--ell_max` / `--delta_ell`, and NO rfft grid anywhere: the box only
sets `f_sky`. The default range is CAMB's 2..3999, but `--ell_max` may exceed it: the sampler's
grid reaches the Nyquist `pi / pix_width` (4320 at 2.5') and `sqrt(2)` times that in the corners,
filled by `covar_matrix_from_cls`'s log-log extrapolation, and `interpolate_spectrum` applies the
same extrapolation so the forecast can count those modes. It matters: at nside 128 / 2.5' /
2.5 uK the omch2 sigma is 0.00355 to 3999, 0.00290 to the Nyquist, 0.00183 to the corner (a disk
to the corner overcounts the square, a disk to the Nyquist undercounts it by pi/4). Its point is `qe_noise_spectrum`, the flat-sky Hu & Okamoto (2002) TT
`N^(0)` as a polar `(|l|, angle)` quadrature over the analysis's own multipole range, so `N_phi` is
a 1D isotropic function with no matrix and no azimuthal decoding. Verified 2026-09-11 that
`scalar_quadratic_estimate` (before `/ NPHI_FAC`) IS that integral restricted to the box's square
of modes with the FFT's periodic wrap of `L - l` (0.1-0.5% at every L), so the `_radial_cl_profile`
"scatter" was the genuine square-box anisotropy (1.6x at L = 540, 5.7x at L = 2090 on nside 64 /
5'), and the periodic wrap lowers the box's N_phi 2-6x below the un-wrapped square domain.
`NPHI_FAC = 2` mirrors CMBLensing.jl's `load_sim(Nϕ_fac = 2)` preconditioning heuristic (Julia's
own docstring: `Nϕ == AL` is the analytic N0) - harmless for G / the HMC mass matrix, but
`qe_noise_matrix` divides by it too, so the `blocks` / `cls` / `full_sky` phi blocks carry HALF the
physical reconstruction noise. `1st_principles` returns the undivided N0. The map noise is
`noise_cls`'s beam-deconvolved `N_l` against the bare `C_l` (consistent at any beam, unlike the
siblings' `B^2 C + N/B^2`, which only agree at beam 0). Extra outputs: `qe_noise_1st_principles.png`
and the spectra / noises / per-block Fisher matrices inside `fisher_1st_principles.npz`.

Do not expect `full_sky` to equal `blocks` even on identical blocks: `nside^2` DOF fill a
SQUARE in `(lx, ly)` while `sum_l (2l+1) f_sky` counts a DISK. Over the grid's full range
that disk holds 1.595x the square's modes (`sqrt` = 1.263, against measured
`sqrt(F_full_sky/F_blocks)` = 1.10/1.42/1.27); truncated at the box's Nyquist it holds
`pi/4` (measured 0.7861 vs 0.7854, `sqrt(F)` ratios 0.96/0.79/0.88).

**The two realization methods** finite-difference the codebase's own log-densities on
`stencil_offsets`' 19-point (for three parameters) central-difference set at a FIXED realization
drawn from the prior with `load_sim`, and average over realizations. `logpdf` differentiates
`statistics.logpdf` with (d, f, phi) held fixed and theta moving only through the bare `C_f` /
`C_phi` (`prior_covariances`); since `log p(d | f, phi)` is theta-independent its `--fast` mode
evaluates only the two prior terms (no lensing solve, one shared CAMB run; 64x measured) and
self-verifies against the full logpdf on the first realization. It converges to Louis's FIRST term
only - the complete-data information - NOT the marginal Fisher of `p(d | theta)`; on prior draws
the omitted Cov term equals it exactly. `mixed` does the same for `statistics.mixed_logpdf` with
(f°, phi°) mixed once at the fiducial `D` / `G` and the QE norm frozen (mirroring the sampler);
every stencil point costs an inverse plus a forward lensing solve and nothing cancels, so it is
fanned out one slurm job per realization by `sampling_chains_TEMPLATE/mixed_hessian.sh` ->
`run_single_mixed_hessian.py` (`mixed_hessian_realization`, one `hessian_<i>.npz` each, ~1m41s at
nside 128) and collected with `--hessian_dir` - or, from Python, `forecast_from_mixed_logpdf(...,
hessian_dir = ...)`, which since 2026-09-11 averages the cached `hessian_*.npz` instead of running
realizations and raises if the files' box / noise / l_knee / parameter set differ from the
arguments (`hessian_directory_config` reads them; `load_hessian_directory` refuses mixed
configurations and duplicate seeds). The CLI takes its box from the files in that mode; pass
`--hessian_dir ""` to compute sequentially. Louis's two-term split is NOT parametrization invariant, so the
`mixed` and `logpdf` Hessians are different decompositions, not estimates of one number.

**None of these is what the sampler targets.** Measured against the 50-map chains at nside 128 /
2.5' / 5 uK, the posterior sits BETWEEN ceiling and lensed in all three sigmas AND all three
correlations. For `omch2` - `theta_MC_100` the bracket straddles zero (ceiling -0.132, chains
+0.377, lensed +0.552), so the two bounds disagree on the sign of a real degeneracy - do not read
the sign of any ceiling entry with |r| < ~0.2. The Louis-identity marginal Fisher
(`marginal_fisher.py`, its `sampling_chains_TEMPLATE/*marginal*` scripts and
`tests/test_marginal_fisher.py`) was **removed on 2026-09-10**: it agreed with the chains within one
jackknife sigma but was 28-37% noisy per entry, and served as a validation of the chains rather
than as a forecast. Two facts from it worth keeping: **never `jax.grad` w.r.t. theta** -
`phi_dot_wrapper`'s custom VJP (`statistics.py`) differentiates `dot(phi, Cphi)` rather than
`dot(phi, Cphi^-1 phi)`, so any autodiff through `phi_covariance` is silently wrong (latent today,
since the only autodiff call is w.r.t. `phi`); and the mixed change of variables has a
theta-dependent Jacobian (`mixed_logpdf` subtracts `logdet(G) + logdet(D)`), which is why the
`mixed` Hessian is not comparable to the unmixed one.

### Polarization Modes

Three configurations control field dimensions and covariance structure, in `load_sim`, `map_joint` and `sample_joint` alike:
- **I** (intensity only): `FlatS0` fields, `DiagonalScalar` operators, `DataSetT`
- **P** (polarization only): `FlatS2` fields (E/B), `DiagonalEB` operators, `DataSetEB` (phi is still a `FlatS0`). At r = 0 this is effectively E-only information with no TE lever — expect much wider theta posteriors than "IP" (ombh2 especially weak).
- **IP** (both): `FlatS02` fields, `BlockTEB` operators (TT/TE/ET/EE/BB, mask and beam with zero TE blocks), `DataSetTEB`. EE's acoustic structure plus the physical TE correlation break the ns–ombh2 degeneracy T-only leaves (marginal corr ~ −0.76), at a considerably higher cost per sweep.

### Removed features (do not assume they exist)

- **CAMB emulator** (`cambemul`, `camb_emulator/*.npz`, `load_sim(use_emulator_cls = ...)`) — removed because emulator-predicted Cls gave theta conditionals inconsistent with real CAMB spectra.
- **`sampling_ar.py`** (A_phi / r sampler port of `sampling.jl`, `grid_and_sample`, `loess`-based inverse-CDF theta step, `AR_KEYS`) and **`sample_lcdm_legacy.py`**. `util.loess` and `rk4_solve` remain in `util.py`.
- **Theta samplers:** grid inverse-CDF, PCA eigen-direction Metropolis (`use_pca`/`save_pca`/`theta_pca.npz`), plain joint MH (`use_joint_mh`), GHMC/NUTS with finite-difference gradients (`use_ghmc_theta`), the Cholesky whitening, `THETA_PRIORS`, `fixed_field_theta`, over-relaxation. Only per-parameter random-walk Metropolis survives. One hard-won fact worth keeping: **`jax.grad(mixed_logpdf)` w.r.t. theta through the G mixing matrix was measured ~3000× wrong with a flipped sign** (cotangent path through `unmix` and the hand-written lensing adjoint; root cause never found), so any future gradient-based theta update needs finite differences or a fixed adjoint.
- **Differentiable grid predictors** (`load_camb_grid_predictors_jax`, `load_camb_grid_predictors_grad`) and `tests/test_camb_grid_interp.py`.
- `constants.py`: `AR_KEYS`, `EMULATOR_MAX_ELL` (→ `DEFAULT_MAX_ELL`), the diffrax PID constants (and the `diffrax` dependency). `map_joint.py`: the unused jitted `line_search`.
- `docs/tutorial.ipynb` / `docs/debug.ipynb` → replaced by `docs/map_joint_tutorial.ipynb`. All the `cmb_lensing/*.png` chain diagnostics and the `scratch_*.log` files were deleted from git.

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

**Basis and parametrization switching**: Fields carry a `basis` (MAP = real space, FOURIER) and are implicitly in a parametrization (T, QU, or EB). Conversion helpers `map()`, `fourier()`, `qu2eb()`, `eb2qu()` are used extensively — gradient code manually converts between representations before and after lensing. Note that `fields.map` shadows the Python builtin inside every wildcard-importing module.

**Random fields**: `field_from_covar_single_key` draws `irfft2(rfft2(white) * sqrt(C))` from REAL white noise so the two self-conjugate rfft columns get full (not half) power — the earlier per-mode complex draw halved their variance and biased every sampled field, noise realization and HMC momentum.

**CAMB through `pure_callback`**: `_camb_via_callback` wraps `_run_camb` so `load_sim` stays traceable; failures return NaN arrays. The 1D and 5D spline predictors use the same pattern.

**Wildcard imports throughout**: Modules use `from cmb_lensing.util import *`, `from cmb_lensing.lense_flow import *`, etc.

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
                   precompute_camb_1d.py ──► GROUND_TRUTH / PARAM_BOUNDS / PARAM_SIGMA, 1D splines
                   camb_grid_interp.py ──► 5D grid predictors (default Cl source)

sampling_chains_TEMPLATE/run_single_lcdm_chain.py ──► sample_lcdm.sample_joint
sampling_chains_TEMPLATE/{run_single_camb_grid, merge_camb_grid, validate_camb_grid}.py ──► the 5D grid
sampling_chains_TEMPLATE/run_single_mixed_hessian.py ──► fisher_forecast_from_mixed_logpdf.mixed_hessian_realization
sampling_chains_TEMPLATE/get_single_delensed_spectra.py ──► delensed_spectrum.measure_delensed_spectrum
sampling_chains_TEMPLATE/{merge_delensed_spectra, compare_transfer_functions}.py ──► delensed_spectrum.merge_transfer_function

delensed_spectrum.py ──► simulate.load_sim, map_joint, lense_flow (INVERSE_LENSE), fisher_forecast
                         (delensed_cls_at_params / delensing_alens / qe_noise_cl)
  ▲ measures R(l); fisher_forecast.load_transfer_function + apply_transfer_function consume it
sampling_chains_TEMPLATE/chain_analysis.py ──► fisher_forecast.forecast (the "blocks" bound in the triangle plot)
runtime_comparison_TEMPLATE/python_performance_test.py ──► map_joint (Julia data via juliacall)

fisher_forecast.py ("blocks") ──► simulate.py (CAMB, covar_matrix_from_cls, scalar_quadratic_estimate)
  ▲                                lensing-free: the base every other forecast imports from
  ├── fisher_forecast_from_cls.py ──► lensing_covariance.py (non-Gaussian bandpower covariance)
  ├── fisher_forecast_full_sky.py
  ├── fisher_forecast_from_logpdf.py ──► statistics.logpdf, simulate.load_sim
  └── fisher_forecast_from_mixed_logpdf.py ──► statistics.mixed_logpdf, mixing.mix, load_sim
```

`cmb_lensing/__init__.py` imports the core modules only — not `sample_lcdm`, `camb_grid_interp` or `precompute_camb_1d`.

### Testing Approach

Tests are validation benchmarks comparing Python output against Julia (CMBLensing.jl) ground truth stored in `tests/ground_truth_data/*.npz` (regenerated with the refactor — every npz changed). They cover `load_sim`, lensing, logpdf, gradients, the Wiener filter and `map_joint`; **nothing tests the sampler**. `tests/test_native_5d_interp.py` is the one exception to the Julia-ground-truth pattern: it checks `camb_grid_interp`'s NdBSpline evaluator against the frozen hand-rolled one in `tests/handrolled_bspline_reference.py` (add `CMB_LENSING_REAL_GRID_TESTS=1` to include the real ~7.9 GB grid). The 1D spline interpolators are still untested. Many tests produce comparison plots in `tests/test_generated_figures/` rather than hard assertions — visual inspection via the HTML viewer (`tests/index.html`, served with e.g. VS Code Live Server) is the primary verification method. `conftest.py` maps each `test_*` module to its `generate_*` module for `--generate`; `_preamble.py` needs `PYTHON_JULIAPKG_PROJECT` pointed at a local CMBLensing.jl checkout (currently the placeholder `/<PATH_TO>/CMBLensing.jl`). `generate_simulated_cls.py` imports `cosmopower_jax` — known to be unusable for unlensed spectra.

## Known issues (as of 2026-08-24 — delete entries as they are fixed)

- `sampling_chains_TEMPLATE/run_single_lcdm_chain.py` hard-codes a real cluster `hpc_path` (`/resnick/groups/wugroup/...`) instead of an `ABSOLUTE_PATH_TO` placeholder, and `sampling_chains_TEMPLATE/chain_analysis.py`'s unused A_phi map-correlation helpers still read from `performance_testing/chain_maps/`.
- `tests/generate_julia_data/_preamble.py` carries the placeholder `PYTHON_JULIAPKG_PROJECT = "/<PATH_TO>/CMBLensing.jl"`, so `pytest --generate` needs it pointed at a real checkout first.
- Nothing tests `sample_joint` or `precompute_camb_1d` (`camb_grid_interp`'s evaluator is covered by `tests/test_native_5d_interp.py`; the rest of the module is not). Smoke-testing the sampler locally: nside 64, `iters_per_chain = 2`, and for `pol = "IP"`/`"P"` inject stub predictors rather than loading the real grid — a real-grid "IP" run at nside 64 rebooted the 16 GB laptop even under a `systemd-run` 11 GB memory cap.

## Numeric Precision

JAX is configured for float64 globally via `jax.config.update("jax_enable_x64", True)` in `constants.py` and repeated at the top of most modules. All field arrays are complex128 (Fourier) or float64 (map).

## Code style

Spaces are used between equal signs and after commas when calling methods or setting variables. For example
"a = b" is preferred style over "a=b" and "a = method(b, c, d)" is preferred style over "a=method(b,c,d)". 

Comments should not start with a space. For example "#this is a preferred comment" is preferred style over "# this is NOT a preferred comment".

## Claude Code operating mode (requested 2026-09-14)

Use the dedicated `Edit`/`Write` tools as the default way to change files in this project, not `sed`/heredocs/shell one-liners run through `Bash` — this holds even under an "auto mode" reminder that says to prefer `Bash` for file changes; that default is overridden here. `Edit`/`Write` tool calls already render as real-time diffs in the active terminal, which is the point of keeping them as the default. Reserve `Bash` file edits for cases `Edit`/`Write` genuinely cannot handle. This preference persists for all future sessions in this repository.
