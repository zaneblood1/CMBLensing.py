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
