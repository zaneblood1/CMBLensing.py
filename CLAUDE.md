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

### Key Design Patterns

**Flax pytree dataclasses**: Fields (`FlatS0`, `FlatS2`, `FlatS02`) and matrix operators (`DiagOp`, `BlockDiagOp`) use `@flax.struct.dataclass` so they work as JAX pytree leaves — passable through `jit`, `grad`, `vmap`.

**Custom VJP on lense_flow**: `lense_flow_wrapper` has a hand-written backward pass (`lense_flow_backwards`) that integrates the adjoint ODE in reverse, rather than relying on JAX's default autodiff through the ODE solver.

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
