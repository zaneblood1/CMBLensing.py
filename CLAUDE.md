# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX port of CMBLensing.jl — simulates gravitational lensing of CMB temperature and polarization fields, then recovers the lensing potential via gradient-based MAP optimization. All array operations use JAX for GPU/TPU support and automatic differentiation.

## Commands

```bash
# Install (editable)
pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/test_lensing.py

# Regenerate Julia ground-truth data before testing (requires juliacall + CMBLensing.jl)
pytest --generate
pytest tests/test_lensing.py --generate

# Generate Julia data manually
python tests/generate_julia_data/generate_lensing.py
python tests/generate_julia_data/generate_all.py
```

No linter or formatter is configured. No CI pipeline exists.

## Architecture

### Computational Pipeline

`simulate.load_sim()` is the main entry point. It uses CAMB to produce power spectra, builds covariance matrices and random fields in Fourier space, then lenses them via `lense_flow()`. The result is a `DataSet` (T, EB, or TEB) containing the fields, operators, and simulated data.

`map_joint()` recovers f and phi from data by alternating Wiener filter steps (estimate f) with gradient descent on phi, using `gradf_logpdf` and `grad_phi_logpdf`.

`sample_joint()` (in `sampling_ar.py`) goes one step further than MAP: instead of a single point estimate, it draws samples from the full joint posterior P(f, phi, theta | d) via Gibbs sampling, where theta are cosmological parameters. This is the code path that matters going forward as the project shifts toward cosmological parameter inference.

> **DEFAULT FILE FOR CONVERSATIONS:** unless stated otherwise, when the user refers to "the sampler", "sample_joint", "gibbs_sample_theta", theta/parameter sampling, etc. in conversation, they mean **`sample_lcdm_legacy.py`** — the live, actively-edited LCDM cosmological-parameter sampler (samples `theta_MC_100`, `logA`, `ns`, `ombh2`, `omch2` via the 5D CAMB grid spline (`USE_CAMB_GRID`, current default), cached-CAMB 1D splines (`USE_CAMB_SPLINE`), or optionally a CAMB emulator — see the two sections below). Do NOT assume `sampling_ar.py` or the unused `sample_lcdm.py`. Edit `sample_lcdm_legacy.py`.

### Cached-CAMB 1D Splines (`precompute_camb_1d.py`)

**Why this exists.** The CAMB emulator (cambemul) produced **unphysical 1D theta conditionals that did not line up with CAMB** — sampling against emulator-predicted Cls gave conditionals inconsistent with what real CAMB spectra imply. The fix is to cut the emulator out of the theta step entirely: precompute real CAMB TT/PP Cls on a fixed 1D grid per parameter and evaluate them via cubic splines (in lnCl) at sample time. `USE_CAMB_SPLINE = True` in `sample_lcdm_legacy.py` swaps these in for every Cl prediction (theta conditionals and post-sample covariance recomputes); the data map must then also be CAMB-generated (`load_sim(..., use_emulator_cls = False)`) so model and data share the same Cl source.

**Mechanics.** `python -m cmb_lensing.precompute_camb_1d <param|all> [--test]` sweeps one parameter over its ±5-training-sigma range (50 nodes) with the other four pinned at ground truth, writing `cmb_lensing/camb_<param>_grid.npz`. `load_camb_spline_predictors(param_name)` returns jit-safe (`jax.pure_callback`) drop-in replacements for the emulator's `predict_tt`/`predict_pp` and raises if any non-swept parameter moves off its cached fixed value. Because the caches are 1D, `sample_joint` requires **exactly one** `should_sample` parameter when `USE_CAMB_SPLINE` is on, and `__main__` forces non-sampled `param_init` entries to ground truth. Single sources of truth in this module (imported by the sampler): `GROUND_TRUTH` (fiducial values), `PARAM_BOUNDS` (grid/search endpoints), `TRAINING_SIGMA` (progress-plot normalization). Cls live on ells 2..EMULATOR_MAX_ELL-1 — the exact support of `load_sim`'s CAMB data-map path, so model and data share the same log-log extrapolation anchor (a one-multipole mismatch measurably biased the ombh2 conditional).

**Provenance.** All five caches were generated with **CAMB 2.0.0** (July 2026, on this machine). Regenerate them if CAMB is upgraded — spectra differences move the conditionals directly.

**theta_MC_100 caveats.** (1) CAMB has **no H0 < 100 solution for theta_MC_100 ≳ 1.117** with the other parameters at ground truth (`CAMBParamRangeError`; last solvable node H0 = 98.2), so that cache holds **42/50 nodes ending at 1.114375**. `run_sweep` skips unsolvable nodes and the predictors return **NaN Cls beyond the cached grid**, which flows to a non-finite logpdf and a rejected proposal — byte-for-byte what the direct-CAMB path does (its callback returns NaNs on CAMB errors). (2) Spline fidelity: leave-half-out max |Δ lnCl| is tiny for most caches (TT ~1e-9 for ns/logA, ~2e-4 for omch2; PP ≤ ~2e-4 everywhere) but **theta_MC_100 TT is ~5e-3** because Cl^TT varies rapidly with theta (acoustic peaks shift in ell). The check doubles node spacing, so true full-grid error is roughly 16× smaller (~3e-4), but if the theta_MC_100 conditional looks suspicious, densifying that grid is the first knob to turn.

### 5D CAMB Grid Spline (`camb_grid_interp.py`) — multi-parameter Cl interpolation

**Why this exists.** The 1D caches above pin four parameters at ground truth, so only one parameter can be sampled at a time. The 5D grid removes that restriction: real CAMB TT/PP Cls are precomputed on a full tensor-product grid over all five LCDM parameters (tau pinned at 0.05) and evaluated via a tensor-product cubic B-spline (in lnCl), so **all five parameters can be sampled simultaneously**. `USE_CAMB_GRID = True` in `sample_lcdm_legacy.py` (the **current default**) swaps `load_camb_grid_predictors(CAMB_GRID_PATH)` in for every Cl prediction. Mutually exclusive with `USE_CAMB_SPLINE`; like it, the data map must be CAMB-generated (`load_sim(..., use_emulator_cls = False)`).

**Mechanics.** The grid file (`performance_testing/sampling_chains/multi_param_CAMB_grid/camb_grid_spline.npz`, ~GB-scale, opened once per process via a lazy singleton — fresh predictor closures would retrace the jitted `_recompute_cosmo_matrices`) is produced by `performance_testing/sampling_chains/merge_camb_grid.py` from per-node CAMB runs (`run_single_camb_grid.py`, driven by `camb_grid.sh`). `load_camb_grid_predictors` returns a **`GridPredictors` namedtuple** of jit-safe (`jax.pure_callback`) predictors — `tt, ee, bb, pp, tt_lensed, ee_lensed, bb_lensed, te_rho`, all with the emulator's predictor signature `f(emu_params, params_batch) -> (M, n_ell)`. Spline evaluation is **hand-rolled** (`_eval_bspline`: local 4-points-per-axis separable contraction over precomputed B-spline coefficients, not-a-knot ends) because scipy's `RegularGridInterpolator(method = "cubic")` rebuilds splines per call (minutes per evaluation); it is exact against a tensor-product reference in `tests/test_camb_grid_interp.py`. Out-of-box queries return NaN → non-finite logpdf → rejected proposal, matching the direct-CAMB and 1D-cache paths.

**Spectra carried by the grid.** Every node records the full unlensed-scalar AND lensed (`total`) spectra: unlensed TT/EE feed the T+P field covariances, PP the phi covariance, and lensed TT/EE/BB the theta-dependent quadratic-estimate recompute (`refresh_qe`, below). **Unlensed scalar BB is identically zero** on the r = 0 / tensors-off grids, so it cannot go through the lnCl spline — the merge verifies that and records a `bb_is_zero` flag; the `bb` predictor then returns exact zeros (NaN out of box, like every spectrum). **TE crosses zero** so it cannot go through the lnCl spline: the merge splines it LINEARLY as the correlation ratio `te_rho = Cl_TE/√(Cl_TT·Cl_EE)` (bounded in (−1, 1), smooth in every parameter; the merge asserts |ρ| < 1 grid-wide). The sampler reconstructs `Cl_TE = ρ·√(TT·EE)` from its own splined TT/EE (`te_covar_from_rho`: linear interpolation of ρ onto the 2D ell grid, flat beyond the last ell), which keeps the 2×2 T/E block **positive semi-definite by construction** — an independently splined TE could overshoot √(TT·EE) and silently break `invert_block_matrix`/`block_matrix_logdet`. Lensed TE is stored raw in the slabs only. Coefficient tables (~1.1 GB each, six of them) load **lazily per spectrum** on first use, so T-only runs never pay for the polarization/lensed tables. A pre-polarization grid file (only `coeff_tt`/`coeff_pp`) stays loadable and raises with a regeneration pointer only when a missing spectrum is actually requested — the production grid file predates this extension and **must be regenerated (all 875 jobs) before pol = "IP" or refresh_qe can run**; do not mix old and new slabs in one out_dir.

**H0 vs theta_MC_100.** The grid axes are `(H0, logA, ns, ombh2, omch2)` — laid out in **H0, not theta_MC_100** — because a rectangular theta box has corners CAMB cannot solve. theta is a background quantity (depends only on H0/ombh2/omch2 with mnu and tau pinned), so every node records theta for free; `merge_camb_grid.py` collapses that to a 3D theta table, and `CambGrid.h0_from_theta` inverts the (densified, monotone) theta(H0) curve at query time. The sampler keeps working in theta_MC_100 throughout and `PARAM_ORDER` is unchanged.

**Current usage — fixed-field theta sampling.** `sample_lcdm_legacy.py`'s chain loop is presently running **theta-only** sampling against this grid: the f- and phi-sampling steps are commented out and every iteration mixes the ground-truth fields (`mix(f_ground, phi_ground, ...)`), i.e. all 5 LCDM parameters are sampled (initialized at ground truth) with f and phi held at ground truth. The theta step uses per-parameter Metropolis (`theta_sampler = "metropolis"`) with hand-tuned per-parameter proposal sigmas and a shuffled parameter order each sweep.

> **`fixed_field_theta = True` is REQUIRED whenever the fields are pinned** (current `__main__` default). With fields pinned, re-mixing them with D/G(theta_current) after every accepted theta makes the chain a stochastic fixed-point iteration on the MIXED conditional — not MCMC on p(theta | f, phi, d). Measured drift map (nside 64): truth is marginally unstable for pol = "IP" (theta_c = +0.5 training sigma → conditional argmax +0.8σ) with a stable attractor at ≈ **+1.0 training sigma in theta_MC_100 — where four different data seeds all parked with artificially tight width**; the T-only map is contractive (slope ≈ 0.4), which is why "I" chains never showed it. The coupled T/E D block (TE inclusion) is what destabilized the truth fixed point; the grid spline, theta→H0 inversion, and te_rho path were all verified accurate (|ΔlnCl| ≤ 4e-3 vs CAMB, theta round-trip 2e-7). The flag pins `mixing_d`/`mixing_g` to the identity at setup, stops `update_args_after_sample` from touching them, and makes every theta path (`make_eval_logpdf_batch(..., fixed_field = True)`) use identity per-row G/D — so `unmix` returns the pinned fields for every candidate theta, logdet(G) = logdet(D) = 0, and the chain samples the UNMIXED conditional p(theta | f_ground, phi_ground, d), whose peak was verified to sit at ground truth and to be exactly independent of theta_current. Set it **False** when f/phi sampling is re-enabled: full Gibbs with theta-dependent mixing is exact when the fields are genuinely resampled each sweep (the production configuration, matching CMBLensing.jl).

**Polarization (T + P) theta sampling — pol = "IP" (current `__main__` default).** `sample_joint` infers the mode from the dataset type (`DataSetTEB` → "IP", `DataSetT` → "I"; requires `USE_CAMB_GRID` for "IP"). The point: **EE's acoustic structure and the physical TE cross-correlation break the ns–ombh2 degeneracy** the T-only map leaves (marginal corr ~ −0.76), so the BBN-style ombh2 entry in `THETA_PRIORS` is now **off by default** (kept commented for A/B against the old T-only-with-prior chains). Modeling, consistent between `load_sim` (whose "P"/"IP" dataset paths are re-enabled, CAMB-Cl-only — `use_emulator_cls = True` raises for pol ≠ "I") and the theta step: **T and E are drawn CORRELATED** to the physical CAMB TE spectrum — E ~ N(0, C_EE), then T = (C_TE/C_EE)·E + an independent piece with variance C_TT − C_TE²/C_EE (≥ 0 by the |ρ| ≤ 1 construction), giving exactly the [[TT, TE], [TE, EE]] block the likelihood assumes. TE covariances everywhere come from `te_covar_from_rho` (see the grid section); the **D matrix carries the full coupled T/E block** (`get_d_teb_matrix`: Schur inverse + `block_matrix_sqrt`, same `d_et` off-diagonal choice as `get_d_matrix` that empirically matches Julia). **Unlensed BB = 0** at r = 0, whose D block is the **identity** where `cf_bb = 0` (the naive formula's pinv-style reciprocal would give `d_bb = 0` and zero out the mixed B sector, while `primal_log_det`/`reciprocal_matrix` already treat the zero-covariance B prior pinv-style); identically-zero spectra (BB, the TE noise) are routed around the log-interpolating covariance builder via `_covar_or_zeros`. Plumbing: the jitted theta kernels take **stacked matrices** — `(3, nside, nside//2+1)` T/E/B fields via `field_matrix_stack`, `(4, ...)` TT/TE/EE/BB operator blocks via `op_matrix_stack`; `make_eval_logpdf_batch(..., pol, bb_is_zero)` rebuilds `FlatS02`/`BlockTEB` structs and evaluates the TEB `mixed_logpdf` per row. All theta samplers (grid/Metropolis/PCA/joint-MH/GHMC) work in both modes. The **f- and phi-sampling steps also support "IP"** (they were T-only until 2026-08): `gibbs_sample_f` branches on the covariance type — `_field_matrices_from_teb_covar` draws the field simulation with the same correlated-T/E construction as `load_sim` (and serves the noise simulation too, where the zero TE block reduces it to independent draws), the loop passes a `FlatS02`-shaped `field_zeroes` template (never the phi-shaped `zeroes`), and `hmc_step` builds the phi momentum from the **phi** template `x`, not `mixed_field` (an S02 template would corrupt the momentum). Wiener filter / gradients / unmix were already operator-generic. Smoke-checked end-to-end at nside 64: 2 full Gibbs sweeps, phi HMC |ΔH| ~ 1e-3 at 100% acceptance, theta stepping normally.

**Polarization-only theta sampling — pol = "P" (added 2026-08-20).** Pass `load_sim(..., pol = "P", ...)`'s `DataSetEB` to `sample_joint` and it runs the E + B-only analysis (`FlatS2` fields, `DiagonalEB` operators; requires `USE_CAMB_GRID` like "IP"). It is purely additive to the "I"/"IP" paths — every `pol`-branching site gained an `elif pol == "P"`: `field_matrix_stack` gives a `(2, ...)` E/B stack, `op_matrix_stack` a `(2, ...)` EE/BB stack, `op_shared_matrix` returns `matrix_EE`; `make_eval_logpdf_batch` rebuilds `FlatS2`/`DiagonalEB` structs and evaluates **only the `ee` (and, off r = 0 grids, `bb`) predictors — no TT, no `te_rho`, no TE block**; `_recompute_cosmo_matrices_eb` is the sibling of the `_teb` kernel (polar QE for `refresh_qe`), `get_new_cf_matrix` returns the `(cf_ee, cf_bb)` pair, `_replace_eb_blocks` mirrors `_replace_teb_blocks` in `add_starting_matrices_to_args` / `update_args_after_sample` / the `fixed_field_theta` identity pin, and `gibbs_sample_f` has a `DiagonalEB` branch (`_field_matrices_from_eb_covar`: independent E/B draws, zero B field at r = 0). The D matrix is `simulate.get_d_eb_matrix` — `get_d_teb_matrix` restricted to its E/B sector (identical to it with `cf_te = 0` to round-off, including the **identity B block** where `cf_bb = 0`). The `DataSetEB.phi` annotation was corrected from `FlatS2` to `FlatS0` (phi is always scalar; flax never type-checked it). Smoke-checked at nside 64: f draw, phi HMC (|ΔH| ~ 5e-2, accepted), mix/unmix round trip 7e-7, finite theta conditionals, 3 Metropolis sweeps, with "I" and "IP" re-run unchanged. Physics caveat: at r = 0 "P" is effectively E-only information with no TE lever, so expect noticeably wider theta posteriors than "IP" (ombh2 in particular is weak: |Δ logpdf| ~ 0.4 at ±1 training σ on the nside-64 probe).

**refresh_qe — theta-tracking QE norm / phi mass matrix.** `sample_joint(..., refresh_qe)` (default: on whenever `USE_CAMB_GRID`) rebuilds the quadratic-estimate norm from the grid's **interpolated lensed Cls** at the newly accepted theta inside `_recompute_cosmo_matrices[_teb|_eb]` (scalar QE from lensed TT for "I", polar QE from unlensed+lensed EE/BB for "IP" and "P"), so `args["quadratic_estimate"]`, the G matrix and the phi HMC mass matrix `pinv(G)² (pinv(Cphi) + pinv(Nphi))` track theta continuously instead of freezing at the initial cosmology. The QE is **fixed within each theta update** (it enters `eval_logpdf_batch` as a batch-constant, so both sides of every accept/reject share one mixing — exact MH) and refreshed **between** sweeps in `update_args_after_sample`. Because it now changes per sweep, the GHMC kernel takes the QE matrix as an **argument** rather than closing over it — a closed-over copy would go stale inside the jitted trace.

**PCA reparametrization of the theta step.** The 5 LCDM parameters are strongly degenerate, so axis-aligned Metropolis mixes slowly along the posterior ridges. `sample_joint(..., use_pca, save_pca, pca_path, pca_proposal_sigma, pca_burn_in)` implements an eigenbasis sampler: `save_pca = True` writes the chain's empirical 5×5 covariance, eigendecomposed, to `THETA_PCA_PATH` (`performance_testing/sampling_chains/theta_pca.npz`) at the end of the run (`compute_and_save_pca`, dropping `pca_burn_in` leading samples); `use_pca = True` loads it (`load_pca_directions`, rows = eigenvectors scaled by sqrt(eigenvalue) to one conditional sigma) and replaces the per-parameter loop with one Metropolis update along each scaled eigen-direction per sweep (`metropolis_sample_direction` — moves the **full 5-vector**, single global `pca_proposal_sigma ≈ 2.4` in sigma units instead of five hand-tuned sigmas). Workflow: pilot run with `use_pca = False, save_pca = True`, then flip `use_pca = True` (keeping `save_pca = True` refreshes the transform from the better-mixed chain). Both flags require all five `should_sample` True, and `use_pca` requires `USE_CAMB_GRID` (eigen-moves change all parameters at once; guards fail fast at the top of `sample_joint` before any heavy loading). Out-of-box proposals are pre-rejected against the `param_ranges` box (exact MH, symmetric proposal) with the grid's NaN → rejected-step backstop covering H0-layout unreachability. Supporting refactor: `gibbs_sample_theta` now builds `eval_logpdf_batch(params_batch)` (arbitrary `(M, 5)` batches) with `eval_logpdf_grid` as a thin axis-aligned wrapper, and takes optional `direction`/`direction_idx`/`pca_accept_history`/`bounds_lo`/`bounds_hi`/`joint_sigmas`/`joint_accept_history` kwargs — with `direction` or `joint_sigmas` set it returns the updated full parameter vector, not a scalar.

**Plain joint MH baseline.** `sample_joint(..., use_joint_mh = True, joint_proposal_sigmas = {param: sigma})` (mutually exclusive with `use_pca`; requires `USE_CAMB_GRID`) runs `metropolis_sample_joint`: the sampled parameters are kicked at once with independent per-parameter Gaussian sigmas, one accept/reject on the joint logpdf change. Unlike `use_pca` it works on **any subset** of parameters — `should_sample = False` entries get proposal sigma 0, so they receive no kick and stay frozen at `param_init` (sigmas are only required for sampled parameters; the chain then samples the conditional posterior given the frozen values). Under strong correlations the sigmas must shrink toward the **conditional** widths `1/sqrt(diag(C⁻¹))` (times 2.38/√d) to be accepted, so mixing along degeneracy ridges stays slow — this is the simple baseline the PCA eigen-direction sampler improves on. Starting sigmas in `__main__` were derived from the pilot covariance's conditional widths; tune toward ~23% acceptance.

### Differentiable Grid Predictors and Gradient-Based Theta Sampling (GHMC)

**Three predictor variants in `camb_grid_interp.py`.** The callback predictors (`load_camb_grid_predictors`) are opaque to autodiff — `jax.grad` through `pure_callback` raises. Two differentiable variants exist:
- `load_camb_grid_predictors_jax` — pure-JAX re-evaluation of the same spline coefficients (`_basis_weights_jax` = de Boor's BasisFuns unrolled at degree 3, `_eval_bspline_jax` = one `dynamic_slice` of the local 4⁵ block + separable contraction; theta → H0 inversion via `jnp.interp` over the densified curve). Exact against the numpy path to round-off, but **jit embeds the closed-over jnp coefficient tables (~2.3 GB) as constants in EVERY executable that traces the predictors** — observed as the "large amount of constants were captured during lowering" warning followed by an OOM kill (exit 137) on the 16 GB machine. Kept for tests and future GPU/big-RAM use.
- `load_camb_grid_predictors_grad` — **what the samplers actually use.** Numpy callback forward pass (identical to the plain callback path, shares the same `CambGrid` singleton — zero extra memory) plus an analytic backward pass glued with `jax.custom_vjp`: `_axis_stencil_derivs` (degree-lowering identity B'ᵢ,₃ = 3(Bᵢ,₂/Δ₁ − Bᵢ₊₁,₂/Δ₂)) → `_eval_bspline_grads` → `CambGrid.spline_param_grads`, which chains the splined quantity’s gradient (lnCl, or the raw te_rho ratio) through the theta → H0 inversion with the implicit function theorem (∂H0/∂theta = 1/(∂theta/∂H0), ∂H0/∂ombh2 = −(∂theta/∂ombh2)/(∂theta/∂H0)). Matches `jax.grad` of the pure-JAX path to ~1e-10 (`test_grad_predictors_match_jax_predictors`). Out-of-box queries: NaN values, **zero finite gradients** (sanitize-then-mask; a bare `jnp.where(valid, val, nan)` would poison upstream gradients via 0·NaN in the cotangent).

**Shared logpdf factory.** `make_eval_logpdf_batch(...)` (module level in `sample_lcdm_legacy.py`) builds the batched mixed-logpdf evaluator that was previously inlined in `gibbs_sample_theta`; every theta path uses it — grid/Metropolis/PCA/joint-MH via `gibbs_sample_theta`, and the GHMC logdensity in `sample_joint` (which `jax.grad`s through it, so it must be handed the `_grad` predictors there).

**Whitened coordinates = the educated-guess tuning (no warmup phase).** GHMC runs in theta_sampled = T @ u, with `load_theta_whitening` building T from the **Cholesky factor of the pilot covariance** (`theta_pca.npz`) restricted to the sampled subset — capturing the logA–ns (−0.9) and ombh2–omch2 (+0.86) correlations so the posterior of u is ≈ N(0, I) and the **identity mass matrix is ideal from the first step**. Fallback without a pilot: T = diag(`TRAINING_SIGMA`). The flat prior is uniform so the constant linear transform needs no Jacobian term. Raw-scale sampling with identity mass crashed the step size to ~3e-8 (scales span ~1e-4..1e-1) and maxed every trajectory's doublings. The correlations MUST live in the position transform, not the metric: blackjax `ghmc`'s docstring claims a `(d, d)` `momentum_inverse_scale` is used as a dense inverse mass matrix, but in practice it diverges 100% of the time (measured on an ideal Gaussian) — only 1-D per-dimension scales work. This whitening replaces `blackjax.window_adaptation`, whose cost scales with map size (hours at nside 256; a 20-step warmup at nside 64 already took ~1 h).

**The logdensity gradient is FINITE DIFFERENCES, not autodiff.** Plain `jax.grad` of `mixed_logpdf` is **wrong with respect to the G mixing matrix** — measured ~3000× too large with a flipped sign (per-input check: autodiff −5.9e5 vs FD +2.0e2 at nside 64, ground truth), which made every gradient-based trajectory divergent (and, before its removal, drove NUTS window adaptation to step sizes ~3e-8). The G cotangent flows through `unmix` and the hand-written lensing adjoint, a path the validated phi gradient in `gradients.py` deliberately hand-assembles rather than trusting naive autodiff; root cause in the `lense_flow_wrapper` custom-VJP composition is still open. The grid-spline predictors' own VJP is exact (verified against FD on the production grid, ratio 1.000 on all axes for TT and PP) — the break is strictly downstream. So the GHMC logdensity carries a `jax.custom_vjp` whose gradient is central differences in the whitened coordinates (`theta_fd_step = 1e-3`): one **batched** `eval_logpdf_batch` call of 2d+1 rows serves value + gradient together, the acceptance test still uses the exact logpdf value (FD error only perturbs trajectories, never the invariant distribution), and theta is only 4–5 dimensional so the cost is a handful of logpdf evals. (A selectable `theta_grad_method = "autodiff"` logdensity was removed along with NUTS; if the lensing-adjoint cotangent is ever fixed, an autodiff logdensity would have to be re-added.)

**`use_ghmc_theta`** (the production gradient-based configuration) — generalized HMC: ONE leapfrog step per sweep with persistent momentum and slice variables carried across sweeps (mirroring `gibbs_sample_phi_ghmc`), so the trajectory is effectively continued across the Gibbs cycle at ~2 gradient evals per sweep. (A `use_nuts` blackjax-NUTS path used to sit beside it; it was **removed** — between the broken autodiff above and blackjax NUTS never functioning on this problem, GHMC with the FD gradient is the sole surviving gradient-based theta update.) The carried state's logdensity/gradient are recomputed inside the kernel each sweep because the conditional's mixing matrices move after every theta update; the momentum stays valid because its metric (the mass matrix) is fixed. Step size tunes online (Robbins–Monro toward `ghmc_theta_target_accept = 0.95` over the first `ghmc_theta_adapt_sweeps` sweeps, then frozen — GHMC needs high acceptance: rejections flip the persistent momentum). Smoke-validated at nside 64: 10 sweeps + full setup in ~26 s, 100% acceptance, zero divergences, all sampled parameters moving, frozen parameters exactly fixed. The flag is mutually exclusive with `use_pca`/`use_joint_mh`, requires `USE_CAMB_GRID`, and works on any `should_sample` subset. The kernel is **jitted once** with the per-sweep mixed matrices as arguments — fresh closures would recompile the lensing graph every sweep. Non-finite logpdf (out-of-box / unreachable theta) maps to −inf → blackjax divergence, preserving the rejected-proposal semantics. blackjax 1.6.2 quirks: `ghmc`'s 1-D `momentum_inverse_scale` is squared elementwise into the inverse mass matrix, and its advertised dense `(d, d)` form is broken (see above).

### Cosmological Parameter Sampling (`sampling_ar.py`)

This module is a direct port of `CMBLensing.jl/src/sampling.jl` — when in doubt about intended behavior, read the Julia source, which is the ground-truth reference. ("AR" = the two parameters originally sampled: **A**_phi and tensor-to-scalar ratio **r**.)

**What is being sampled.** Two scalar cosmological parameters, each rescaling a covariance:
- **A_phi** — amplitude of the lensing-potential power. Scales the phi covariance: `phi_covariance = A_phi * unscaled_phi_covariance`. Fiducial value is `a_phi_fid`.
- **r** — tensor-to-scalar ratio. Scales the tensor contribution to the field covariance: `field_covariance = scalar_field_covariance + (r/r_fid) * tensor_field_covariance`. Fiducial value is `r_fid` (`DataSet.fid_r`).

**Gibbs structure of `sample_joint()`.** Each chain iteration cycles through (mirrors the `gibbs_samplers` list in the Julia `sample_joint`):
1. **Sample f** (`gibbs_sample_f`) — a Wiener-filter draw of the unlensed field given phi and the current covariances. Equivalent to Julia `gibbs_sample_f!` / `sample_f`.
2. **Mix** (`mix` in `mixing.py`) — transform `(f, phi) -> (f°, phi°)` into the *mixed* parametrization. Julia `gibbs_mix!`.
3. **Sample phi** (`gibbs_sample_phi`) — one HMC step on the mixed phi°. Julia `gibbs_sample_ϕ!`.
4. **Sample theta** (`gibbs_sample_theta`) — grid-and-sample each cosmological parameter from its 1D conditional. Julia `gibbs_sample_slice_θ!`. Skipped for the first `num_burn_in_fix_theta` iterations.
5. **Recompute** the G/D mixing matrices and covariances at the newly sampled theta.
6. **Unmix** (`unmix`) back to `(f, phi)`. Julia `gibbs_unmix!`.

**The mixed parametrization.** Sampling phi directly is poorly conditioned, so fields are reparametrized as `f° = L(phi) * D * f` and `phi° = G * phi` (see `mix`/`unmix` in `mixing.py`, matching Julia `mix`/`unmix` in `dataset.jl`). `L(phi)` is the lensing operator (lense flow), and:
- **D** (`mixing_d`) — whitens the field; built by `get_d_tt_matrix` (T-only) / `get_d_matrix` (full TEB) in `simulate.py`. Depends on **r**.
- **G** (`mixing_g`) — whitens phi relative to the quadratic-estimate noise; built by `get_g_matrix` in `simulate.py`. Depends on **A_phi**.

Because the transformation has a non-trivial Jacobian, `mixed_logpdf` (`statistics.py`) evaluates the posterior by calling `unmix`, then `logpdf`, then subtracting `logdet(G) + logdet(D)`. This exactly mirrors the Julia `logpdf(Mixed(ds); ...)` in `dataset.jl`. When sampling a parameter, the corresponding mixing matrix and covariance must be rebuilt *inside* the per-value `logpdf_partial` closure (see `gibbs_sample_a_phi` / `gibbs_sample_r`) because changing theta changes G/D/covariances.

**HMC for phi** (`hmc_step` + `symplectic_integrate`). A leapfrog/symplectic integrator on the Hamiltonian `H = logpdf(x) - p·(M⁻¹p)/2`, with mass matrix `M = pinv(G)² * (pinv(Cphi) + pinv(Nphi))` (`get_mass_matrix`, = Julia `mass_matrix_ϕ`). Default integration is `num_steps=30, step_size=0.01` on the Python side vs. `N=25, ϵ=0.01` in Julia (`symp_kwargs`) — a known divergence to keep in mind. Metropolis acceptance compares `log(u) < ΔH`.

**`grid_and_sample` — the theta conditional sampler.** Given log-pdf values on a grid of theta values, it draws one sample via inverse-CDF sampling. There are two implementations kept side by side so they can be swapped and compared:
- `grid_and_sample` — the faithful port of Julia `grid_and_sample` (`sampling.jl`). LOESS-smooths the log pdf (`loess` in `util.py`, a port of Julia's Loess.jl), then uses `scipy.integrate.quad` (≈ Julia `quadgk`) to build the normalized CDF and `scipy.optimize.brentq` (≈ Julia `find_zero` with `Roots.Brent()`) for the inverse-CDF root solve. The whole numpy/scipy body runs inside a single `jax.pure_callback` so the function stays JIT-compatible.
- `grid_and_sample_cumsum` — a faster, fully-JAX approximation that replaces adaptive quadrature with a cumulative-trapezoid CDF and `jnp.interp` for the inverse CDF. Use this to A/B against the scipy version.

**Per-parameter samplers.** `gibbs_sample_a_phi` and `gibbs_sample_r` each build a `logpdf_partial(theta)` closure (rebuilding G or D and the relevant covariance for that theta), `jax.vmap` it across the parameter's grid `theta_range`, then call `grid_and_sample`. `gibbs_sample_theta` dispatches to the right one via `AR_KEYS` (`constants.py`).

> **NOTE — current debug state:** `sampling_ar.py` is presently wired to *reproduce a specific Julia chain* for validation, not to run standalone. It `precision_load`s covariances, masks, fields, momentum kicks (`p_matrix`), and RNG draws from hardcoded `/home/zane-blood/Desktop/julia_chain_debug/*.npz|.txt` paths written by the `#ZXB_DEBUG` blocks in the Julia `sampling.jl`. The real RNG / simulation code paths are commented out alongside these. Before using this for production sampling, restore the commented-out `jax.random`-based draws and the `field_from_covar_single_key` / new-simulation logic. `sample_lcdm.py` is a separate future-work sampler and can be ignored for now.

### The Phi Gradient in Fourier Space (Reproducing Julia's Anti-Hermitian Nyquist Content)

This section documents a subtle but important correctness issue in the **mixed phi gradient** (`mixed_grad_phi_logpdf` in `gradients.py`, which drives the HMC `symplectic_integrate` step above) and the rework that fixed it. Read this before touching `gradients.py`, `lense_flow.py`'s gradient path, `statistics.py:logpdf`, or `fields.py:undo_inner_product`. It is easy to "simplify" this code back into the broken state because the broken version *looks* more natural.

#### The symptom

When reproducing the Julia chain, the mixed phi gradient (`gradient.scalar_matrix`) agreed with the Julia `mixed_phi_gradient_*` dumps everywhere **except** the first column (`[:, 0]`, kx = 0) and the last column (`[:, -1]`, kx = Nyquist) of the rfft array. There the fractional difference was ~2% (`2.07e-2` overall, dominated by those two columns). The error had a tell-tale symmetry: along each of those columns, the **real part of the error was odd** and the **imaginary part was even** under row-reversal (`n -> (N-n) mod N`) — i.e. the error was *anti-Hermitian*, the opposite symmetry of the (Hermitian) gradient itself. Compounded over the 30 leapfrog steps, this shifted the sampled phi enough to matter; substituting the Julia gradients into the integrator dropped the phi disagreement by orders of magnitude, confirming the gradient — not the integrator — was the culprit.

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

- **`util.py`** — added `get_k_meshgrid` (shared `(KX, KY)` builder, with the negative-Nyquist `ky` convention — see below), `get_primal_derivatives_from_fourier(phi_fourier, pix_width)` (derivatives computed directly from an rfft2 array — same values as `get_primal_derivatives(irfft2(...))` but keeps `i·ℓ` in the autodiff graph), and `get_primal_derivatives_to_fourier(field, pix_width)` (applies the derivative operators but returns the result *in* rfft2 space, with no final `irfft2`).
- **`lense_flow.py`** — `get_lensing_operator_gradients`, `lensing_gradients_integration_step`, and `get_delta_phi_tqu_roc` are now **basis-aware**. They branch on whether `phi.scalar_matrix` is square (MAP, real-space) or rectangular (FOURIER) via `shape[0] != shape[1]`, threaded through the RK4 loop as a static Python bool `phi_fourier`. When `phi_fourier` is true: phi-derivatives come from `get_primal_derivatives_from_fourier`; `delta_phi` is initialized as a complex `(N, N//2+1)` array and **accumulated in Fourier space**; and the final divergence/laplacian operators that build `d_delta_phi_dt` use `get_primal_derivatives_to_fourier` (no closing `irfft2`). This mirrors CMBLensing.jl's `negδvelocityᴴ` (`lenseflow.jl`), which accumulates `δϕ` via `-∇'·Ð(...)` in Fourier. `delta_phi` never feeds back into another rate of change, so it can live at a different shape/dtype than the (square, real) `t/q/u/delta_t/...` state inside the RK4 tuple. The two inner integration functions are intentionally **not** `@jax.jit`'d (they run inside the already-jitted `get_lensing_operator_gradients`, and `phi_fourier` must stay a Python bool to select dtype/shape).
- **`lense_flow.py:primal_lense_flow`** — also branches on phi shape so the forward lensing accepts a Fourier phi (same value, different autodiff graph). Forward-only callers (`mixing.py`, `simulate.py`, `gibbs_sample_f`, `gradf_logpdf`) still pass `map(phi)` (square, MAP) and therefore hit the unchanged real-space path — no behavior change for them.
- **`statistics.py:logpdf`** — no longer does `phi = map(phi)`; it passes the **Fourier** phi straight into `lense_flow_wrapper`. (The `phi_dot_wrapper` prior term already used the Fourier phi, so it is unaffected.)
- **`gradients.py:mixing_jacobian_phi_component`** — now differentiates with respect to the **Fourier** phi (`jax.vjp(unmix_partial, phi)`, returning the cotangent directly with no `fourier(differential)` re-wrap), AND uses `lense_flow_wrapper` (the custom analytic-adjoint VJP) instead of plain `lense_flow`. See the critical subtlety below.
- **`fields.py:undo_inner_product`** — a rectangular (Fourier) gradient is now passed through unchanged. The original real-space body — `irfft2(conj(rfft2(m)/fourier_weights) * nside**2)` — composed with the implicit `irfft2`-Gram / `map` VJP that followed it, and that composite cancels to the identity on the bulk for a gradient that is already in Fourier; applying the real-space body to a Fourier array would instead corrupt it and re-symmetrize away the anti-Hermitian content.

#### The critical subtlety: analytic adjoint vs. autodiff-through-the-solver

There are two terms in the mixed phi gradient: the **data term** (flows through `logpdf`'s `lense_flow_wrapper`, which has the hand-written analytic adjoint) and the **f-prior chain-rule term** (`mixing_jacobian_phi_component`). The f-prior term originally used **plain autodiff through the RK4 ODE** (`jax.vjp` of the un-wrapped `lense_flow`). Julia uses an **analytic continuous-adjoint ODE** (`negδvelocityᴴ`), and autodiff-through-the-discretized-solver differs from that analytic adjoint by the ODE discretization error — with only ~7–10 RK4 steps this is ~15%, which completely swamps the ~2e-4 agreement we are chasing and corrupts the *bulk* (Hermitian) modes, not just the two columns. The fix routes the chain-rule term through `lense_flow_wrapper` too, so **both** terms use the same analytic adjoint Julia uses. This is why, after the fix, the bulk stays exact (ratio 1.0) *and* the anti-Hermitian content appears: do not "simplify" `mixing_jacobian_phi_component` back to plain `lense_flow`/autodiff.

#### The Nyquist sign convention (`get_primal_derivatives`)

`get_k_meshgrid` sets the half-axis Nyquist wavenumber **negative** (`ky = ky.at[-1].set(-1*ky[-1])`), matching CMBLensing.jl's `ifftshift(-N÷2:(N-1)÷2)` construction (numpy's `rfftfreq` would make it positive). On its own this sign is nearly invisible to the *forward* derivatives (the Nyquist column's anti-Hermitian content is discarded by `irfft2` either way). But it sets the **sign** of the `i·ℓ` adjoint's anti-Hermitian content on the Nyquist row, so once that content is preserved (above) the sign must match Julia. Keep it.

#### Result and how to re-validate

After the rework the mixed phi gradient matches the Julia `mixed_phi_gradient_*` dumps to ~`2.2e-4` across the entire 30-step leapfrog trajectory (down from `2.07e-2`), the forward field value is unchanged (`F` fractional difference `~4.8e-6`), and all gradient/logpdf/lensing/map_joint/wiener regression tests pass. To re-validate per-mode against Julia at a *known* input (not just the trajectory dumps), reconstruct the exact dataset in Julia from the operator dumps and call `gradient(ϕ° -> logpdf(Mixed(ds); f°, ϕ°), ϕ°)` — covariances/mask/beam/mixing must be wrapped as **real-valued** `FieldOp`s, `G` is not dumped (rebuild it from `Nphi`/`Cphi0` exactly as `get_g_matrix` does), and the dumps are stored transposed relative to Julia's `(Ny÷2+1, Nx)` layout. Julia and JAX share FFTW conventions (unnormalized forward `rfft`, `1/N²` inverse), so no extra FFT normalization factor is needed.

### Key Design Patterns

**Flax pytree dataclasses**: Fields (`FlatS0`, `FlatS2`, `FlatS02`) and matrix operators (`DiagOp`, `BlockDiagOp`) use `@flax.struct.dataclass` so they work as JAX pytree leaves — passable through `jit`, `grad`, `vmap`.

**Custom VJP on lense_flow**: `lense_flow_wrapper` has a hand-written backward pass (`lense_flow_backwards`) that integrates the adjoint ODE in reverse, rather than relying on JAX's default autodiff through the ODE solver. This analytic adjoint (not autodiff-through-the-solver) is required for the phi gradient to match Julia — and its `delta_phi` accumulation is basis-aware (real-space vs Fourier). See *The Phi Gradient in Fourier Space* above.

**Basis and parametrization switching**: Fields carry a `basis` (MAP = real space, FOURIER) and are implicitly in a parametrization (T, QU, or EB). Conversion helpers `map()`, `fourier()`, `qu2eb()`, `eb2qu()` are used extensively — gradient code manually converts between representations before and after lensing.

**Wildcard imports throughout**: Modules use `from cmb_lensing.util import *`, `from cmb_lensing.lense_flow import *`, etc.

### Module Dependency Graph

```
simulate.py ─────► util.py (FFT, derivatives, coordinate grids)
  │                lense_flow.py ──► fields.py (FlatS0/S2/S02 dataclasses)
  │                dataset.py       constants.py
  │                statistics.py

map_joint.py ─────► gradients.py ──► lense_flow.py
                    wiener_filter.py
                    statistics.py

sampling_ar.py ──► map_joint.py, wiener_filter.py  (Gibbs sampling of f, phi, theta)
                   mixing.py ──► mix/unmix into mixed parametrization
                   gradients.py ──► mixed_grad_phi_logpdf (HMC)
                   statistics.py ──► mixed_logpdf
                   simulate.py ──► get_g_matrix / get_d_tt_matrix (rebuild G, D per theta)
```

`sampling_ar.py` ports `CMBLensing.jl/src/sampling.jl`; `mixing.py` ports the `mix`/`unmix` in `dataset.jl`.

### Polarization Modes

Three polarization configurations control field dimensions and covariance structure:
- **I** (intensity only): scalar `FlatS0` fields
- **P** (polarization only): spin-2 `FlatS2` fields (E/B modes)
- **IP** (both): combined `FlatS02` fields with block-diagonal covariance

### Testing Approach

Tests are validation benchmarks comparing Python output against Julia (CMBLensing.jl) ground truth stored in `tests/ground_truth_data/*.npz`. Many tests produce comparison plots in `tests/test_generated_figures/` rather than hard assertions — visual inspection via the HTML viewer (`tests/index.html`, served with e.g. VS Code Live Server) is the primary verification method.

## Numeric Precision

JAX is configured for float64 globally via `jax.config.update("jax_enable_x64", True)` in `constants.py` and repeated at the top of most modules. All field arrays are complex128 (Fourier) or float64 (map).

## Code style

Spaces are used between equal signs and after commas when calling methods or setting variables. For example
"a = b" is preferred style over "a=b" and "a = method(b, c, d)" is preferred style over "a=method(b,c,d)". 

Comments should not start with a space. For example "#this is a preferred comment" is preferred style over "# this is NOT a preferred comment".
