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

If you wish to run the unit tests or actively contribute to the development of the repository it will also be necessary to have [`juliacall`](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) installed and a [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl) checkout - see [Testing](../testing.md) for more information.

## Downloading the 5D CAMB grid

`sample_joint` evaluates its theory spectra by default through a 5D CAMB spline, `camb_splines/camb_grid_spline.npz` (~7.9 GB). This file and the five small 1D caches (`camb_splines/camb_<param>_grid.npz`) are too large to store on GitHub. They are hosted on the Hugging Face dataset [zaneblood1/5D_LCDM_CAMB_Grid_Spline](https://huggingface.co/datasets/zaneblood1/5D_LCDM_CAMB_Grid_Spline) and are **downloaded automatically into `camb_splines/` the first time the code needs them** (`cmb_lensing/camb_splines.py`). No manual step is needed.

To fetch all six files ahead of time, run

```bash
cmb-lensing-download-splines
```

On a cluster, run this once on a login node: compute nodes often have no internet access. You can also rebuild the grid yourself ([Building the CAMB Grid](../pipelines/camb-grid.md)) and place it at `camb_splines/camb_grid_spline.npz`. See [The CAMB Grid](camb-grid.md) for more information on how the two splines are used.

## HPC setup (`hpc_local.sh`)

If you wish to run any of the HPC slurm scripts for e.g. sampling LCDM parameters en masse or generating data for Fisher forecasting, you will need to make a copy of the ``hpc_local.example.sh `` script and fill it in with your own personal cluster configurations. The example script is gitignored to avoid sharing personal email accounts, etc.

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

Every driver and every job script it submits sources this file. See [HPC setup](hpc-setup.md) for how the drivers work.

## Building these docs locally

```bash
mkdocs serve      #live-reloading preview at http://127.0.0.1:8000
mkdocs build      #static site in site/
```
