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
```

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

Comments should not start with a space. For example "#this is a preffered comment" is preferred style over "# this is NOT a preferred comment".
