# Unit Tests

Most tests compare CMBLensing.py with CMBLensing.jl: given the same inputs, the Python and Julia outputs should agree to numerical precision. The Julia outputs are stored in the repository, in the folder `tests/ground_truth_data/`, so the tests run without Julia installed.

The Julia outputs come from commit [`8e75a7c`](https://github.com/marius311/CMBLensing.jl/commit/8e75a7c56e37d337256c5646305630816e1ee196) of CMBLensing.jl, run with Julia 1.11.

| Test file | What it checks |
|---|---|
| `test_covariance_matrices.py`, `test_simulated_cls.py` | the spectra and covariances of the simulation |
| `test_lensing.py` | the lensing operation |
| `test_logpdf.py`, `test_gradients.py` | the log-probability and its gradients |
| `test_wiener_filter.py`, `test_map_joint.py` | the Wiener filter and the MAP estimate |
| `test_camb_grid.py` | the 5D CAMB grid against direct CAMB calls (skipped when the grid file is absent) |

The LCDM sampler, the Fisher forecast and the noise measurement modules have no tests.

These comparisons are closer to validation benchmarks than to unit tests. They check whole functions against Julia, and many of them produce plots to inspect instead of failing on their own. Real unit tests, which check small pieces of the code in isolation and fail automatically when something breaks, are an area for future development and improvement.

## Running the tests

Run every test from the repository root:

```bash
pytest
```

Or run a single file:

```bash
pytest tests/test_lensing.py
```

Many tests write comparison plots into the folder `tests/test_generated_figures/` instead of asserting a result. A test can therefore pass while a plot shows a problem, so look at the plots. The page `tests/index.html` displays them; open it through a local server such as the VS Code Live Server extension.

## Regenerating the Julia data

This is only needed when the Julia reference data has to change. It requires Julia, a CMBLensing.jl checkout at the commit above, and the Python package juliacall.

**Step 1.** Install juliacall into your Python environment.

```bash
pip install juliacall
```

**Step 2.** Make Julia 1.11 your default version. The commands below use [juliaup](https://github.com/JuliaLang/juliaup).

```bash
juliaup add 1.11
juliaup default 1.11
julia --version
```

The last command should print a 1.11 version, for example:

```
julia version 1.11.9
```

**Step 3.** Clone CMBLensing.jl and check out the commit. A clone always fetches the latest version, so the commit is checked out afterwards.

```bash
git clone https://github.com/marius311/CMBLensing.jl.git
cd CMBLensing.jl
git checkout 8e75a7c56e37d337256c5646305630816e1ee196
```

Git reports a "detached HEAD". This is expected: the checkout now sits on that commit instead of on a branch.

**Step 4.** Start Julia from the terminal.

```bash
julia
```

**Step 5.** In the Julia session, activate the environment of the checkout. 

```julia
using Pkg
Pkg.activate("/path/to/CMBLensing.jl")
```

**Step 6.** In the same session, install the dependencies of the checkout, then the three extra packages the data generators load.

```julia
Pkg.instantiate()
Pkg.add(["PythonCall", "PythonPlot", "NPZ"])
Pkg.precompile()
```

CMBLensing itself is not added: the activated environment is CMBLensing, and Julia refuses to add a package to its own environment.

**Step 7.** Still in the same session, check that the checkout loads, then leave Julia.

```julia
using CMBLensing
exit()
```

**Step 8.** Tell the test scripts where the checkout is. They read the path from an environment variable and pass it to juliacall, which activates the same environment as step 5.

On Linux and macOS, add the following line to the end of the file `~/.bashrc` (or `~/.zshrc` if your shell is zsh), so that every new terminal has the variable set:

```bash
export CMBLENSING_JL_PATH=/path/to/CMBLensing.jl
```

Then load it into the terminal you already have open:

```bash
source ~/.bashrc
```

Running the export line directly in a terminal also works, but only for that terminal: the variable is gone once it is closed.

On Windows, store the variable permanently with the command below, then close and reopen the terminal:

```bat
setx CMBLENSING_JL_PATH "C:\path\to\CMBLensing.jl"
```

If juliaup also has a newer Julia installed, juliacall picks the newest one rather than your default, and the 1.11 environment of steps 5 and 6 then fails to load (`Artifact "LLVMExtra" was not found`). Point juliacall at the 1.11 executable the same way:

```bash
export PYTHON_JULIAPKG_EXE="$(julia +1.11 -e 'print(joinpath(Sys.BINDIR, "julia"))')"
```

**Step 9.** From the root of the CMBLensing.py repository, regenerate all of the Julia data.

```bash
python tests/generate_julia_data/generate_all.py
```

To regenerate the data of a single test instead, run its generator on its own:

```bash
python tests/generate_julia_data/generate_lensing.py
```

To regenerate the data and run the tests in one command:

```bash
pytest --generate
```

## Confirming the checkout works

After regenerating the data, run the MAP estimate comparison:

```bash
pytest tests/test_map_joint.py
```

It writes its plots into three folders, one per kind of data:

```
tests/test_generated_figures/map_joint/intensity_only/
tests/test_generated_figures/map_joint/polarization_only/
tests/test_generated_figures/map_joint/intensity_and_polarization/
```

Each folder holds two sub-folders of plots to inspect:

| Plot | What to look for |
|---|---|
| `phi/julia_mle_phi.png` | Julia's estimate of the lensing potential: a smooth map with visible structure, like the Python estimate next to it |
| `fields/julia_mle_field_*.png` | Julia's estimates of the unlensed fields, again matching the Python estimates |
| `phi/phi_cross_correlation.png`, `fields/*_cross_correlation.png` | the estimates compared with the simulated truth: the "Julia x Sim" curve should be close to 1 on large scales and lie on top of the "Python x Sim" curve |

A Julia result that is NaN or all zeros typically means one of three things:

- the environment of the checkout was not activated (steps 5 and 8),
- the Julia version is not 1.11 (step 2),
- the checkout is not at the commit above (step 3).
