# Contributing to CMBLensing.py

Thanks for your interest in CMBLensing.py. Bug reports, questions, documentation fixes and code contributions are all welcome. This page explains how to get help, how to report a problem, and how to propose a change.

CMBLensing.py is a JAX port of [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl). Background on the package, its pipelines and its tutorials is in the [documentation](https://zaneblooddev.com/CMBLensing.py/).

## Getting help

All support goes through [GitHub Issues](https://github.com/zaneblood1/CMBLensing.py/issues). If you have a question about using the package, open an issue and say so in the title (e.g. "Question: ..."). Please search the existing issues first in case it has already been answered.

## Reporting a bug

Open a [GitHub issue](https://github.com/zaneblood1/CMBLensing.py/issues) that includes:

- what you ran: ideally a short, self-contained script that reproduces the problem
- what you expected to happen and what happened instead, with the full error traceback if there is one
- the run configuration: polarization mode (`"I"`, `"P"` or `"IP"`), `nside`, `theta_pix`, noise level, and which Cl source the sampler used (the 5D CAMB grid or the 1D splines)
- your environment: operating system, Python version, the versions of `jax`, `jaxlib` and `camb`, and whether you ran on CPU or GPU

For numerical problems (a result that looks wrong rather than a crash), a plot or the numbers you compared against are very helpful.

## Suggesting a feature

Open an issue describing what you would like to do and why. For larger changes, please open the issue before writing code so we can agree on the approach first.

## Setting up a development environment

```bash
git clone https://github.com/zaneblood1/CMBLensing.py.git
cd CMBLensing.py
pip install -e ".[dev,docs]"
```

The LCDM sampler needs precomputed CAMB spectra, which are hosted on the Hugging Face dataset [`zaneblood1/5D_LCDM_CAMB_Grid_Spline`](https://huggingface.co/datasets/zaneblood1/5D_LCDM_CAMB_Grid_Spline). They are downloaded into `camb_splines/` automatically the first time they are needed, or ahead of time with

```bash
cmb-lensing-download-splines
```

The 5D grid is about 8 GB on disk, and a temperature + polarization (`"IP"`) chain holds about 9 GB of it in memory. The `camb_splines/` folder is gitignored: **never commit or force-add these files.**

## Running the tests

```bash
pytest                          #all tests
pytest tests/test_lensing.py    #one file
```

Most tests compare the Python output against CMBLensing.jl results stored in `tests/ground_truth_data/`. Many of them write comparison plots to `tests/test_generated_figures/` instead of asserting, so please also look at the plots (open `tests/index.html` through a local web server, e.g. VS Code Live Server): a test can pass while a plot shows a problem. `tests/test_camb_grid.py` is skipped if the 5D grid has not been downloaded.

Regenerating the Julia reference data requires [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) and a local CMBLensing.jl checkout, whose location is read from the `CMBLENSING_JL_PATH` environment variable.

On Linux (and macOS), set it with `export`. To make it permanent, add the line to your `~/.bashrc` (or `~/.zshrc`):

```bash
export CMBLENSING_JL_PATH=/path/to/CMBLensing.jl
pytest --generate
```

On Windows, `export` does not exist. Use `setx` instead, which stores the variable permanently for your user account. The syntax is `setx <NAME> "<value>"`:

```bat
setx CMBLENSING_JL_PATH "C:\path\to\CMBLensing.jl"
```

`setx` only affects terminals opened afterwards, so close and reopen your terminal before running `pytest --generate`.

See the [testing page](https://zaneblooddev.com/CMBLensing.py/testing/) of the documentation for details. The LCDM sampler, the Fisher forecast and the measurement modules are not yet covered by tests, and contributions there are especially welcome.

## Submitting a pull request

1. Fork the repository and create a branch off `master`.
2. Keep each pull request focused on one change.
3. Run the tests that cover the code you touched, and look at their comparison plots.
4. If your change affects numerical results, say how you checked it (e.g. against CMBLensing.jl or direct CAMB) in the pull request description.
5. If you change a pipeline's commands or outputs, update the matching page in `docs/` too.
6. Open the pull request against `master` and describe what it changes and why.

## Code style

There is no automatic formatter, so please follow these conventions:

- Put spaces around `=` and after commas, including keyword arguments: `a = f(b, c, key = 1)`, not `a=f(b,c,key=1)`.
- Comments start without a space after the `#`: `#like this`, not `# like this`.
- Prefer plain, readable code: simple loops, one step per line, and variable names that follow the physics (`cphi`, `nphi`, `cl_tt`, ...).
- Keep it lean: avoid adding defensive input checks or extra machinery that the computation does not need.

## Documentation

The documentation site is built with MkDocs Material. Pages live in `docs/` and the navigation is in `mkdocs.yml`.

```bash
mkdocs serve            #local preview at http://127.0.0.1:8000
mkdocs build --strict   #run this before pushing
```

The site is redeployed automatically on every push to `master`.

## HPC scripts

The cluster scripts read personal settings (conda environment, e-mail address, paths) from a gitignored `hpc_local.sh`. Copy `hpc_local.example.sh` to `hpc_local.sh` and edit that copy. Please keep personal settings out of tracked files.

## License

CMBLensing.py is released under the [MIT License](LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.
