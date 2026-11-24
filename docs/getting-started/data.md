# Data Availability

The repository's large data is too big for GitHub. All of it is stored on the Hugging Face dataset [zaneblood1/cmb_lensing_data](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data) (~24 GB). The dataset card there describes every file format.

| Dataset path | Size | Contents | Written by |
|---|---|---|---|
| `camb_grid_spline.npz` | 7.9 GB | the 5D CAMB grid ([Installation](installation.md#downloading-the-camb-grid)) | `camb_grid_generation/` |
| `joint_inference_data/` | 659 MB | LCDM chains, one folder per run | `get_lcdm_samples/` ([LCDM Sampling on the HPC](../pipelines/lcdm-chains.md)) |
| `delensed_field_covariance_data/` | 713 MB | empirical delensed covariances | `get_delensed_field_covariance/` ([Computing Delensed Covariance Matrices](../pipelines/delensed-covariance.md)) |
| `map_joint_phi_noise_data/` | 162 MB | `map_joint`'s per-mode N_phi | `measure_phi_noise/get_map_joint_phi_noise/` ([Measuring Noise in the MAP Estimates](../pipelines/map-joint-phi-noise.md)) |
| `sampler_phi_noise_data/` | 1.0 GB | the sampler's per-mode N_phi | `measure_phi_noise/get_sampler_phi_noise/` ([Measuring Noise in the Sampler](../pipelines/sampler-phi-noise.md)) |
| `map_joint_comparison/` | 13 GB | `map_joint` timings and reconstructions, Python vs Julia | `runtime_comparison/` |

## Downloading

Only the CAMB grid is downloaded automatically ([Installation](installation.md#downloading-the-camb-grid)). Fetch anything else with the `hf` command that `huggingface_hub` installs, one folder at a time:

```bash
hf download zaneblood1/cmb_lensing_data --repo-type dataset \
    --include "joint_inference_data/*" --local-dir cmb_lensing_data
```

Drop the `--include` to download everything. `cmb_lensing_data/` at the repository root is gitignored, and it is where `measure_phi_noise/phi_noise_analysis.py` looks for its input files. The [Analyzing Chain Data](../chain_analysis_tutorial.ipynb) tutorial downloads the files it needs into that folder by itself.

## Browsing the results

Results from various trials of the code can be found  [here](../results/index.html).