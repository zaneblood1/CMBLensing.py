LCDM chains of `sample_joint` on simulated lensed temperature maps: the summer 2026 run at 5 uK-arcmin, on Hugging Face as [`omch2_logA_theta_MC_100_joint_inference_08_27_26`](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data/tree/main/joint_inference_data/omch2_logA_theta_MC_100_joint_inference_08_27_26).

| Setting | Value |
| --- | --- |
| Data | Temperature only (`I`) |
| Map size | 128 x 128 pixels at 2.5 arcmin per pixel |
| White noise level | 5 uK-arcmin |
| Sampled parameters | `theta_MC_100`, `logA`, `omch2` |
| Maps | 10 (the first 10 of the run's 50) |
| Chains per map | 5 |
| Sweeps per chain | 6000 |
| Burn-in removed | 600 sweeps per chain |
| Product distribution | Gaussian: the product of one normal per map (its mean and std) |
| Product errors | Bootstrap over each map's thinned samples (Monte Carlo error of the chains) |
| Fisher forecast | Field blocks from the measured delensed covariance (Wiener filter from the QE N0 with unlensed filter and unlensed response); phi noise the QE N0 with unlensed filter and unlensed response |

| Parameter | Ground truth | Mean R-hat over maps | Std of R-hat over maps | Acceptance rate |
| --- | --- | --- | --- | --- |
| `theta_MC_100` | 1.031732 | 1.0105 | 0.0060 | 0.43 |
| `logA` | 3.218387 | 1.0039 | 0.0029 | 0.57 |
| `omch2` | 0.109381 | 1.0247 | 0.0297 | 0.78 |

Held at the ground truth: `ns` = 0.959814, `ombh2` = 0.022386.

**Notes written when the chains were run**

- 128 x 128 sized temperature only maps
- theta_pix = 2.5
- (f, phi) both sampled, phi using HMC with step_size = 0.05, num_steps = 10
- working in mixed parametrization withOUT perfect G using cphi_fid
- phi_init = ZEROES (MAP seems to sometimes go NaN for certain LCDM parameter combos)
- 3 sequential 1-D MH steps for omch2, logA, theta_MC_100 with theta_MC_100_step = 3e-3, logA_step = 2e-2, omch2_step = 8e-4
- Starting out at random initial points in the 3-D param space... (raised lower bound on omch2 from ~ 0.079 to 0.085)
- No 1/f noise OR beam, only white noise at a noise-level of 5 uk-arcmin.
- No mask
- Using CAMB multi-dimensional interpolator instead of the emulator
- map_prefix = 234567
- num_burn_in_fix_theta = 100
- 6000 iterations per sub-chain, 5 sub-chains per map, 50 maps
- deterministic order of theta parameter sampling
