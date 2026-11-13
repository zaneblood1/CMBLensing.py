# Data

The repository's large data is too big for GitHub. All of it is stored on the Hugging Face dataset [zaneblood1/cmb_lensing_data](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data) (~23 GB). The dataset card there describes every file format.

| Dataset path | Size | Contents | Written by |
|---|---|---|---|
| `camb_grid_spline.npz` | 7.9 GB | the 5D CAMB grid ([The CAMB Grid](camb-grid.md)) | `camb_grid_generation/` |
| `joint_inference_data/` | 512 MB | LCDM chains, one folder per run | `get_lcdm_samples/` ([LCDM Sampling on the HPC](../pipelines/lcdm-chains.md)) |
| `delensed_field_covariance_data/` | 683 MB | empirical delensed covariances | `get_delensed_field_covariance/` ([Delensed Covariances](../pipelines/delensed-covariance.md)) |
| `map_joint_phi_noise_data/` | 98 MB | `map_joint`'s per-mode N_phi | `measure_phi_noise/get_map_joint_phi_noise/` ([MAP Joint Phi Noise](../pipelines/map-joint-phi-noise.md)) |
| `sampler_phi_noise_data/` | 787 MB | the sampler's per-mode N_phi | `measure_phi_noise/get_sampler_phi_noise/` ([Sampler Phi Noise](../pipelines/sampler-phi-noise.md)) |
| `map_joint_comparison/` | 13 GB | `map_joint` timings and reconstructions, Python vs Julia | `runtime_comparison/` |

## Downloading

Only the CAMB grid is downloaded automatically ([Installation](installation.md#downloading-the-5d-camb-grid)). Fetch anything else with the `hf` command that `huggingface_hub` installs, one folder at a time:

```bash
hf download zaneblood1/cmb_lensing_data --repo-type dataset \
    --include "joint_inference_data/*" --local-dir cmb_lensing_data
```

Drop the `--include` to download everything.

## Using it

Nothing in the package looks for these folders at a fixed location: every script takes the path of the file or folder it reads.

```bash
#chain analysis of one run, with the forecast
python -m cmb_lensing.chain_analysis cmb_lensing_data/joint_inference_data/joint_inference_2026_10_29_2.5uk \
    --nside 128 --theta_pix 2.5 --noise 2.5 --run_fisher \
    --delensed_covariance cmb_lensing_data/delensed_field_covariance_data/delensed_covariance_output_map_joint_2.5uk/delensed_covariance.npz \
    --phi_noise cmb_lensing_data/map_joint_phi_noise_data/map_joint_phi_noise_output_2.5uk/map_joint_phi_noise.npz

#the forecast on its own
python -m cmb_lensing.fisher_forecast --noise 5 \
    --delensed_covariance cmb_lensing_data/delensed_field_covariance_data/delensed_covariance_output_map_joint_5uk/delensed_covariance.npz
```

See [Chain Analysis](../pipelines/chain-analysis.md) and [Fisher Forecast](../fisher-forecast.md) for the options. The `map_joint` comparison figures are remade by `runtime_comparison/performance_analysis.py`, with the `BASE_DIR` / `RESULTS_DIR` at its top pointed at the downloaded `map_joint_comparison/performance_results` folder.
