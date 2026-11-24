# Installation

Clone the repository and install it in editable mode from the repository root (the folder holding `pyproject.toml`):

```bash
git clone https://github.com/zaneblood1/CMBLensing.py.git
cd CMBLensing.py
pip install -e .
```

This pulls in JAX (pinned to `jax == jaxlib == 0.9.2`), Flax, CAMB, NumPy, SciPy and Matplotlib. Python 3.10 or newer is required. JAX is configured for float64 globally on import.

Optional extras:

```bash
pip install -e ".[dev]"    #pytest + jupyter
pip install -e ".[docs]"   #MkDocs, to build this site locally
```

## Check the install

```python
from cmb_lensing.simulate import load_sim
data_set = load_sim(nside = 64, theta_pix = 5, pol = "I", master_seed = 1)
print(f"Pixel resolution = {data_set.theta_pix} arcminutes per pixel")
```

If you wish to run the unit tests or actively contribute to the development of the repository it will also be necessary to have [`juliacall`](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) installed and a [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl) checkout - see [Unit Tests](unit-tests.md) for more information.

## Downloading the CAMB grid

`sample_joint` needs the theory power spectra at every proposed LCDM parameter combination. It gets them from a precomputed grid of real CAMB spectra: a 5D tensor-product cubic spline over `(H0, logA, ns, ombh2, omch2)`, stored in `camb_splines/camb_grid_spline.npz` (~7.9 GB). It serves any subset of the five parameters and every polarization mode.

This file is too large to store on GitHub. It is hosted on the Hugging Face dataset [zaneblood1/cmb_lensing_data](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data) and is **downloaded automatically into `camb_splines/` the first time the code needs it** (`cmb_lensing/camb_splines.py`). No manual setup is needed.

To fetch it ahead of time, run the following command in a bash terminal from the repo root:

```bash
cmb-lensing-download-splines
```

On a cluster, run this once on a login node: compute nodes often have no internet access. To use your own grid instead, rebuild it on the cluster ([Building the CAMB Grid](../pipelines/camb-grid.md)), copy it back to your local machine and place it at `camb_splines/camb_grid_spline.npz`.

The same Hugging Face dataset holds the chains, phi noise, delensed covariance and `map_joint` comparison outputs. None of those are downloaded automatically; see [Data Availability](data.md).

### Validating the grid

```bash
python tests/test_camb_grid.py --grid camb_splines/camb_grid_spline.npz -n 20
```

compares the spline against direct CAMB at random interior points. Typical max |Δ ln Cl| is ~1e-3 (TT) to ~5e-3 (EE near an acoustic trough).

## HPC setup

All cluster scripts are Slurm scripts, used for e.g. sampling LCDM parameters en masse or generating data for Fisher forecasting. Before running any of them, make a copy of the `hpc_local.example.sh` script at the repository root and fill it in with your own cluster configuration. The copy is gitignored to avoid sharing personal email accounts, etc.

```bash
cp hpc_local.example.sh hpc_local.sh
```

then edit `hpc_local.sh`:

```bash
#the conda install and environment the jobs run in
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate myenv

#where Slurm mails job failures
export SBATCH_MAIL_USER=you@example.edu

#the CMBLensing.jl checkout, used only by runtime_comparison/
export CMBLENSING_JL_PATH=/path/to/CMBLensing.jl
```

Every driver and every job script it submits sources this file.

### How the drivers work

Each pipeline folder has a **driver** that only submits jobs, and is run with `bash`, not `sbatch`:

```bash
bash get_lcdm_samples/sample_lcdm.sh
```

A driver can be run from any directory. It `cd`s into its own folder, sources the repo-root `hpc_local.sh` and `hpc_submit.sh` and submits its job script once per job. The physical configuration (`nside`, `theta_pix`, `noise_level`, `l_knee`, ...) is set by passing flags to each driver, e.g. `bash get_lcdm_samples/sample_lcdm.sh --nside=128 --noise_level=5`; a setting that is not passed keeps the default listed at the top of the driver. Keep them **the same across drivers** for one experiment, so the chains, the phi-noise measurements and the forecasts all describe the same box.

| Driver | What it does |
|--------|--------|
| `camb_grid_generation/camb_grid.sh` | Computes the CAMB spectra on the 5D parameter grid that `sample_joint` interpolates. Only needed to rebuild the grid instead of downloading it ([Building the CAMB Grid](../pipelines/camb-grid.md)). |
| `get_lcdm_samples/sample_lcdm.sh` | Runs the LCDM sampler on many simulated data maps, with several chains per map. These chains are the main result ([LCDM Sampling on the HPC](../pipelines/lcdm-chains.md)). |
| `measure_phi_noise/get_map_joint_phi_noise/get_map_joint_phi_noise.sh` | Runs `map_joint` on many realizations to measure the noise of its lensing potential estimate. Used as the forecast's phi noise ([Measuring Noise in the MAP Estimates](../pipelines/map-joint-phi-noise.md)). |
| `measure_phi_noise/get_sampler_phi_noise/get_sampler_phi_noise.sh` | Runs chains with the LCDM parameters held fixed to measure the noise of the sampler's lensing potential estimate. Also usable as the forecast's phi noise ([Measuring Noise in the Sampler](../pipelines/sampler-phi-noise.md)). |
| `get_delensed_field_covariance/get_delensed_field_covariance.sh` | Measures the covariance of the delensed field and how it changes with each parameter. The Fisher forecast cannot run without it ([Computing Delensed Covariance Matrices](../pipelines/delensed-covariance.md)). |
| `runtime_comparison/run_performance_test.sh` | Times `map_joint` in Python against CMBLensing.jl over a range of map sizes. |


### Submission throttle

Some clusters cancel every submission beyond a cap on queued jobs instead of leaving it pending. Every driver therefore submits through `submit_throttled` (repo-root `hpc_submit.sh`): before each `sbatch` it waits until you have fewer than `MAX_QUEUED_JOBS` jobs queued or running (default 85), checking every `SUBMIT_POLL_SECONDS` (default 60), and it retries a rejected `sbatch`. Override either value in `hpc_local.sh`.

A driver therefore runs until its last job is submitted, which can take hours. Keep it alive after you log out:

```bash
nohup bash get_lcdm_samples/sample_lcdm.sh > submit.log 2>&1 &
tail -f submit.log
```

`nohup` ignores the hangup signal sent at logout, `> submit.log 2>&1` sends output and errors to the file, and `&` runs it in the background. Alternatively run the driver inside `tmux`, detach with `Ctrl-b d`, and reattach later with `tmux attach` (on the same login node).

## Building these docs locally

```bash
mkdocs serve      #live-reloading preview at http://127.0.0.1:8000
mkdocs build      #static site in site/
```
