# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It holds only what an agent needs on every task: the commands, where things are, the rules that must not be broken and the working conventions. The exposition (how each module works, why, the numbers measured, the tricks found, the development history, what was removed) is in **`PROJECT_SUMMARY.md`**. Read the section of it named in a "→" pointer below before changing the code that pointer is attached to. New narrative, history and measurements go there, not here.

## Project Overview

JAX port of CMBLensing.jl. It simulates gravitational lensing of CMB temperature and polarization fields (`simulate.load_sim`), recovers the lensing potential by gradient-based MAP optimization (`map_joint`), jointly samples f, phi and the five LCDM parameters from a lensed data map by Gibbs sampling (`sample_joint` in `sample_lcdm.py`), and compares the sampler's posterior widths with a Gaussian Fisher forecast (`fisher_forecast.py`) built from a measured delensed covariance, for `pol` = "I", "P" or "IP".

> **DEFAULT FILE FOR CONVERSATIONS:** when the user refers to "the sampler", "sample_joint", "gibbs_sample_theta", theta/parameter sampling, etc., they mean **`cmb_lensing/sample_lcdm.py`**. It is the only sampler; `sample_lcdm_legacy.py` and `sampling_ar.py` no longer exist.

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
bash camb_grid.sh                                          # 875 slurm jobs, one H0 sweep each
python merge_camb_grid.py --grid_dir multi_param_CAMB_grid  # -> camb_grid_spline.npz
python tests/test_camb_grid.py --grid <...>/camb_grid_spline.npz -n 20   # spline vs direct CAMB

# LCDM chains on the HPC (get_lcdm_samples/; chains -> get_lcdm_samples/lcdm_sampler_data/<experiment>/)
bash sample_lcdm.sh              # spawns run_single_lcdm_chain.sh per (map, chain): 10 maps x 5 chains
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
bash get_delensed_field_covariance.sh                          # one job per seed (e.g. --pol=IP, --qe_filter=lensed)
python merge_delensed_field_covariance.py --covariance_dir <scp'd out_dir>   # -> delensed_covariance.npz; add --legacy_format
                                                               # for job files written before 2026-11-11 (no pol key)

# Sampler effective phi noise (measure_phi_noise/get_sampler_phi_noise/)
bash get_sampler_phi_noise.sh                                  # num_maps x num_chains fixed-theta chains (one job each)
sbatch merge_sampler_phi_noise.sh --burn_in=600               # a job itself, submit from its folder; burn_in (default 300) / out_dir_name
                                                               # overrides; -> sampler_phi_noise.npz ("nphi"); works on running jobs
python ../phi_noise_analysis.py                                # no arguments: also the R-hat plot of the files in its __main__ `convergence_files`
python -m cmb_lensing.sampler_phi_noise --nside 64 --theta_pix 5 --n_maps 4 --n_chains 2 --n_sweeps 400 --burn_in 100 --out_dir <dir>   # local

# Per-mode map_joint N_phi at GROUND_TRUTH (measure_phi_noise/get_map_joint_phi_noise/)
bash get_map_joint_phi_noise.sh                                # one job per seed
python merge_map_joint_phi_noise.py --noise_dir <scp'd out_dir>  # -> map_joint_phi_noise.npz ("nphi")

# Phi noise spectra plots (box, noise and merged-file paths set in __main__)
python measure_phi_noise/phi_noise_analysis.py   # -> measure_phi_noise/phi_noise_plots/noise_level_<noise>uk/

# Docs site (MkDocs Material; pages in docs/, nav in mkdocs.yml; pip install -e ".[docs]")
mkdocs serve                 # local preview at http://127.0.0.1:8000
mkdocs build --strict        # what to check before pushing; notebooks are rendered, never executed

# Results website: add a data set by writing a chain_analysis output into experimental_results/lcdm_sampling/
# (the folder name is the data set toggle's label), then push
python -m cmb_lensing.chain_analysis <chain_dir> --noise 2.5 --out_dir experimental_results/lcdm_sampling/<dataset>
```

**HPC drivers** (`camb_grid.sh`, `sample_lcdm.sh`, `get_*.sh`, `run_performance_test.sh`):
- They are plain `bash`, not `sbatch`, and can be run from any directory. Every setting is a `name=${name:-default}` line in the driver, overridable as `--<name>=<value>`; read the driver for the current names and defaults, never quote one from memory. Each run (re)writes `<out_dir>/submission_settings.txt`.
- The cluster needs a gitignored `hpc_local.sh` at the repo root (`cp hpc_local.example.sh hpc_local.sh`: conda activate, `SBATCH_MAIL_USER`, `CMBLENSING_JL_PATH`). Nothing personal goes into a tracked script.
- Every driver submits through `hpc_submit.sh`'s `submit_throttled` (waits while `MAX_QUEUED_JOBS` = 85 jobs are queued, since ACES cancels submissions past ~90; `EXCLUDE_NODES` → `sbatch --exclude`), so it blocks until its last job is submitted: run it as `nohup bash <driver>.sh > submit.log 2>&1 &` or inside tmux. Neither survives a login node reboot.
- A driver can be rerun with the same arguments after an interruption: each job goes through `submit_once`, which skips a job with a marker in `<out_dir>/submitted/` (also one that timed out or failed; delete the marker to resubmit it). A new driver must do the same, with a marker name unique per job. Runs started before 2026-11-25 have no markers. → "The cluster scripts in detail"
- `hpc_local.example.sh` defines `run_driver <driver.sh> [--<name>=<value> ...]` (the `nohup` line, log in the driver's folder as `submit_<out_dir_name>.log`) and `kill_driver <text>` (stops matching drivers, not their jobs). They exist on a cluster only once copied into its `hpc_local.sh`.
- Outputs land inside the driver's own folder. Job scripts find their `.py` and `hpc_local.sh` through `$SLURM_SUBMIT_DIR`.

There is no linter or formatter. The only CI is `.github/workflows/docs.yml`, which runs `mkdocs gh-deploy` on every push to master. MkDocs is pinned `< 2`.

## Repository Layout

```
cmb_lensing/                  the Python package
  constants.py                  PARAM_ORDER / PARAM_INDEX / GROUND_TRUTH / PARAM_BOUNDS / PARAM_SIGMA / CAMB_LMAX / DEFAULT_MAX_ELL /
                                CAMB_SPLINES_DIR / CAMB_GRID_PATH / DEFAULT_HMC_*: the single source of truth, import from here
  simulate.py, lense_flow.py, fields.py, matrix_operators.py, dataset.py, util.py, statistics.py, gradients.py, mixing.py, wiener_filter.py
  map_joint.py                  the MAP estimate
  sample_lcdm.py                the LCDM sampler (sample_joint)
  camb_grid_interp.py, camb_splines.py   the 5D grid spline and its Hugging Face download
  chain_analysis.py, mcmc_diagnostics.py chain plots (CLI), R-hat / autocorrelation
  fisher_forecast.py, camb_spectra.py, matrix_plots.py   the forecast and its shared helpers
  delensed_covariance.py, map_joint_phi_noise.py, sampler_phi_noise.py, fixed_theta_chain.py   the measurement pipelines
  fisher_output/                fisher_forecast.py's figures and npz
  sample_lcdm_output/           sample_lcdm's progress plots (untracked; must exist before plotting)
camb_splines/                 the ~8 GB camb_grid_spline.npz; gitignored, NEVER force-add it
cmb_lensing_data/             hand-downloaded Hugging Face data (gitignored); nothing else is kept locally
camb_grid_generation/         slurm / merge scripts for the 5D grid
get_lcdm_samples/             sample_lcdm.sh, run_single_lcdm_chain.sh/.py; chains -> lcdm_sampler_data/<run>/ (gitignored)
get_delensed_field_covariance/ driver, job script, merge_delensed_field_covariance.py
measure_phi_noise/            phi_noise_analysis.py, get_sampler_phi_noise/, get_map_joint_phi_noise/
runtime_comparison/           slurm scripts timing map_joint Python vs Julia, performance_analysis.py
hpc_local.example.sh, hpc_submit.sh
docs/                         MkDocs pages: getting-started/ (installation.md, data.md, developing.md), concepts/basic-concepts.md,
                              pipelines/, api/, three tutorial notebooks, results/ (the static results website)
experimental_results/         <section>/<dataset>/ drop folders of the results website + build_results_site.py (MkDocs hook)
paper/                        paper.md / paper.bib / figures / build_pdf.sh
tests/                        Julia A/B benchmarks + test_camb_grid
PROJECT_SUMMARY.md            the narrative: how the code works, tricks, development history
README.md, CONTRIBUTING.md, CHANGELOG.md, CITATION.cff
CODE_REVIEW_FINDINGS.md       local, untracked list of open review findings (maintained by hand)
.claude/skills/               the project's slash commands (local only: .claude/ is gitignored since 2026-11-24)
```

**Skills** (`.claude/skills/<name>/SKILL.md`):
- `/update-docs`: brings CHANGELOG, PROJECT_SUMMARY, CLAUDE, AGENTS, CONTRIBUTING, README and the docs site in line with the code. It never edits `CODE_REVIEW_FINDINGS.md`.
- The four HPC skills were added 2026-11-24 and have NOT yet been run on a cluster. They assume two-factor authentication: they log in once per run through `refresh-hpc/hpc_login.sh` (opens a terminal window for the password, returns once the shared ssh connection is up) and reuse that connection. `refresh-hpc/hosts.md` lists the clusters (ssh alias, repo directory).
  - `/refresh-hpc <resnick|aces>`: deletes the checkout on the cluster and re-clones origin/master + `pip install -e .`, carrying over `hpc_local.sh` and `camb_splines/`; stops and asks when jobs run from the checkout or it holds other untracked output.
  - `/run-pipeline <cluster> <sample_lcdm|sampler_noise|map_noise|delense_covariance|runtime_comparison|camb_grid> [--<setting>=<value> ...]`: launches that driver under nohup; reports the cluster's `submission_settings.txt`.
  - `/merge-pipeline <cluster> <pipeline> --out_dir_name=<run folder> [...]`: runs the merge on the cluster and copies ONLY the merged file + `submission_settings.txt` to `cmb_lensing_data/<dataset folder>/<out_dir_name>/`, sha256-checked (for `sample_lcdm` it only fetches the `*_history.txt` files). Never copies job files.
  - `/report-pipeline-status <cluster> [<pipeline> ...|all] [--out_dir_name=<run folder>]`: read-only progress of runs at most 7 days old (`pipeline_status.sh` runs on the cluster through ssh's stdin).

## What to know before touching each part

Each entry is the minimum; the "→" names the `PROJECT_SUMMARY.md` section with the full explanation.

**Everywhere**
- One parameter order: `PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]`. `current_params` is a `(5,)` array in that order; user-facing dicts are keyed by name. Names, matrices, figures and npz files are all in this order.
- Fields are `rfft2` half-planes of shape `(nside, nside//2+1)`; fields and operators are `@flax.struct.dataclass` pytrees. Modules use wildcard imports, and `fields.map` shadows the builtin `map` inside them. `fields.map` / `fields.fourier` have no guardrails: check `field.basis` before converting a field of unknown basis.
- `to_camb_naming_conv` converts in place and deletes the old keys of the dict it is given.
- The Julia tests each define a module-level `GROUND_TRUTH` (a data path) that shadows the constants dict inside those files only.
- → "Key Design Patterns", "Module Dependency Graph"

**`load_sim` / simulation**
- The mask is all ones. `load_sim` defaults to `l_knee = 100` (matches Julia); every sampler entry point passes `l_knee = 0`.
- CAMB runs through `jax.pure_callback`: a failure gives NaN spectra, never an exception. `camb_spectra.camb_cls_at_params` (the forecast's path) RAISES instead.
- RNG: one root key `PRNGKey(master_seed)`, one subkey split off per field in a fixed order (phi, field(s), noise). Do not go back to pre-splitting a batch of keys: the common random numbers of `delensed_covariance.py` depend on it. Chains that reuse a map seed root their keys with `fixed_theta_chain.chain_key`, never at `PRNGKey(seed)`.
- Random fields are drawn from REAL white noise (`field_from_covar_single_key`); a per-mode complex draw halves the power of the two self-conjugate columns.
- A `precomputed_cls` dict must include `"TP"` (`EXPECTED_CL_KEYS`). `cls["TP"]` changes sign, so it must not go through the log-interpolating `covar_matrix_from_cls`.
- → "Computational Pipeline", "Key Design Patterns"

**The phi gradient (`gradients.py`, `lense_flow.py`'s gradient path, `statistics.py:logpdf`, `fields.py:undo_inner_product`)**
- Do NOT "simplify" this code. phi stays in FOURIER through the whole gradient path (no `map(phi)` in `logpdf`, `delta_phi` accumulated in Fourier), `mixing_jacobian_phi_component` goes through `lense_flow_wrapper` (the analytic adjoint, not autodiff through the RK4 solver), `get_k_meshgrid` keeps the NEGATIVE Nyquist wavenumber, and the two inner integration functions stay un-jitted with `phi_fourier` a Python bool. Undoing any of these breaks the 2e-4 agreement with Julia.
- → "The Phi Gradient in Fourier Space" (read it first)

**The sampler (`sample_lcdm.py`)**
- `matplotlib.use("Agg")` stays the first thing in the module.
- Predictors have the signature `f(params_batch: (M, 5)) -> (M, n_ell)` and are STATIC jit arguments: pass the same cached function objects every call (a fresh closure recompiles the lensing graph).
- The only Cl source is the 5D grid (`get_camb_grid_predictors()`, `constants.CAMB_GRID_PATH`); a run needs the ~7.9 GB file (~4.5 GB of RAM for "I", ~9.1 GB for "IP").
- The only theta sampler is per-parameter random-walk Metropolis on the MIXED logpdf, in `proposal_sigmas` dict order. The HMC momentum is drawn from the phi template, never from `mixed_field`.
- TE is carried as the correlation ratio `te_rho`. Unlensed BB is identically zero: it must bypass log interpolation (`_covar_or_zeros`, the static `bb_is_zero` flag) and the D-mixing B block is the identity there.
- Never `jax.grad` with respect to theta (silently wrong, two independent causes).
- `fixed_fields = True` is a debugging aid for the theta step, never a posterior.
- `sample_joint` does not expose `refresh_qe`: the QE norm stays frozen at `param_init`.
- Nothing tests the sampler. Smoke test locally at nside 64 with `iters_per_chain = 2`; for "IP" / "P" inject stub predictors if memory is tight (a real-grid "IP" run once rebooted the 16 GB laptop).
- → "The LCDM sampler", "5D CAMB Grid Spline", "Polarization Modes"

**The 5D grid (`camb_grid_interp.py`, `camb_grid_generation/`)**
- The maths is one function, `spline_coefficients`; the grid is laid out in H0 and `h0_from_theta` inverts theta(H0). Out-of-box queries return NaN (a rejected proposal).
- `camb_splines.camb_spline_file` downloads a missing file and never re-downloads a present one, so a locally rebuilt grid wins. `tests/test_camb_grid.py` never downloads.
- → "5D CAMB Grid Spline", "The data on Hugging Face"

**The forecast and the measurement modules**
- `fisher_forecast.forecast` needs a merged `delensed_covariance.npz` (`None` raises); there is no CAMB delensing, no `t_block`, no `nphi_source` strings. N_phi is the box QE N0 (`qe_filter` "unlensed" / "lensed") or a merged phi noise file's `"nphi"`.
- `qe_noise_matrix` does NOT divide by `NPHI_FAC`.
- The sampler phi noise merge uses the RESPONSE estimator, map_joint the CORRELATION estimator; do not swap them. `phi_noise_analysis.py` rebuilds N_phi from the `*_per_realization` keys, so a merged file without them cannot be plotted.
- `fixed_theta_chain.CHAIN_STREAM` must not be 0 or 1.
- `chain_analysis.py` has one job (chain history files → MCMC plots and the comparison with ONE forecast); anything else belongs in a debug script. Its forecast is off by default (`--run_fisher`).
- → "The Fisher forecast", "The empirical delensed covariance stencil", "The sampler phi noise", "Other measurement modules", "Chain analysis", "The chain analysis outputs"

**Removed code.** A long list of features no longer exists (the CAMB emulator, every theta sampler but Metropolis, the rotated-coordinate sampler, the 1D CAMB splines, CAMB delensing and the forecast's option matrix, the sibling forecast modules, the Louis / mixed-Hessian forecast, the KDE product). Do not assume a function exists because an old file, notebook or data set mentions it: grep first. → "Removed features" records what each was, how it worked and the last commit that has it (mostly `b50d4cc`).

**Data.** Large data is on the Hugging Face dataset `zaneblood1/cmb_lensing_data`; only the grid is auto-downloaded. Fetch the rest into `cmb_lensing_data/` with `hf download zaneblood1/cmb_lensing_data --repo-type dataset --include "<folder>/*" --local-dir cmb_lensing_data`. Older data does not always line up with the current code (seeds before 2026-10-15 / 2026-11-10 do not reproduce, pre-2026-11-11 delensed covariance job files need `--legacy_format`, some sampler phi noise files lack the moment keys). → "The data on Hugging Face", "Local data and old data against current code"

**Tests.** Julia validation benchmarks against `tests/ground_truth_data/*.npz` (49 tests collected); each file lists its measured differences (`MEASURED_*`) and fails past `regression_thresholds.max_allowed(measured)`. When a difference changes on purpose, update the measured value. The sampler, the forecast, `delensed_covariance` and `sampler_phi_noise` are untested. → "Testing Approach"

## Documentation upkeep

- `PROJECT_SUMMARY.md` gets the narrative: how a module works, why a choice was made, measurements (dated, absolute), tricks, and when something is deleted what it was, how it worked and the last commit that has it. This file gets a line only when an agent needs it on every task (a command, a path, a rule), with a "→" pointer.
- `AGENTS.md` is a copy of this file except for its first heading and the agent named in its opening sentence; `diff AGENTS.md CLAUDE.md` must show those two lines only.
- The docs pages are a user-facing digest; keep them in step when a pipeline's commands or outputs change. `docs/api/` is generated by mkdocstrings from the Google-style docstrings of `load_sim`, `map_joint`, `sample_joint`, `fisher_forecast.forecast`, `chain_analysis` (`main`, `load_chains`, `get_avg_covariance_matrix`, `get_forecasted_covariance`) and the field / operator / data set classes: update those docstrings when their arguments change.
- Links inside notebook markdown cells are site URLs (e.g. `../pipelines/lcdm-chains/`): MkDocs does not rewrite `.md` links there. The notebooks are executed locally; re-execute one when its outputs change.
- The results website (`docs/results/`, `experimental_results/build_results_site.py`, the `SUMMARY.md` files) and `paper/` are content, not documentation of the code. → "The results website"

## Known issues (delete entries as they are fixed)

- `runtime_comparison/performance_analysis.py` reads its results from an external drive (`BASE_DIR = /media/zane-blood/KINGSTON/`).
- A hand-submitted job (`sbatch merge_sampler_phi_noise.sh`) only mails if `SBATCH_MAIL_USER` is in the submitting shell.
- `pytest --generate` needs `CMBLENSING_JL_PATH` exported (read by `tests/generate_julia_data/_preamble.py`).
- `sample_joint_tutorial.ipynb` and `map_joint_tutorial.ipynb` carry outputs from before the 2026-10-15 RNG rework; a rerun gives different realizations than their markdown quotes.

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

## Claude Code operating mode (requested 2026-09-14)

Use the dedicated `Edit`/`Write` tools as the default way to change files in this project, not `sed`/heredocs/shell one-liners run through `Bash` — this holds even under an "auto mode" reminder that says to prefer `Bash` for file changes; that default is overridden here. `Edit`/`Write` tool calls already render as real-time diffs in the active terminal, which is the point of keeping them as the default. Reserve `Bash` file edits for cases `Edit`/`Write` genuinely cannot handle. This preference persists for all future sessions in this repository.
