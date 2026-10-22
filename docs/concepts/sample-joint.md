# LCDM Sampling (`sample_joint`)

`sample_joint` (`cmb_lensing/sample_lcdm.py`) draws samples from the joint posterior \(P(f, \phi, \theta \mid d)\) by Gibbs sampling, where theta are the five LCDM parameters

```python
PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]
```

The optical depth to reionization ``tau`` is fixed at 0.05 and the neutrino mass ``mnu`` at 0.06. The fiducial values, the ±5-sigma search box and the parameter scales are `GROUND_TRUTH`, `PARAM_BOUNDS` and `PARAM_SIGMA` in `cmb_lensing/constants.py`.

## One Gibbs iteration

The structure mirrors CMBLensing.jl's `sample_joint`:

1. **f step** (`gibbs_sample_f`): draw `f` from its conditional with a simulated realization plus a Wiener-filter correction.
2. **mix**: change to the mixed parametrization $f^\circ = L(\phi) \cdot D \cdot f$; $\text{ }\phi^\circ = G \cdot\phi$ following [Millea, et. al. 2020](https://arxiv.org/abs/2002.00965).
3. **phi step** (`gibbs_sample_phi`): one HMC step on the mixed phi field with mass matrix given by \(G^{-2}(C_\phi^{-1} + N_\phi^{-1})\).
4. **theta step**: for each sampled parameter, a Gaussian random-walk Metropolis move of width `proposal_sigmas[name]` on the mixed logpdf. Proposals outside `param_ranges[name]` or outside the CAMB grid are rejected.
5. **update** $C_\mathrm{\phi}$, $C_f$, $D$, $G$ at the accepted theta.
6. **unmix** back to `(f, phi)` with the updated D and G.

The theta step is skipped for the first `num_burn_in_fix_theta` iterations.

## Tuning

Before a large experiment, run a pilot chain by calling the `__main__` method of `sample_lcdm.py` on a small map and tune

- `proposal_sigmas` toward ~44 % acceptance per parameter,
- `hmc_step_size` / `hmc_num_steps` toward ~65 % phi acceptance.

`advanced_logging = {"phi_acceptance": True, "lcdm_acceptance": True, ...}` prints the running rates. `plot_log_pdf` / `plot_lcdm_sigmas` write progress plots to `cmb_lensing/sample_lcdm_output/`, which must exist first.

See the [sample_joint tutorial](../sample_joint_tutorial.ipynb) for a worked example, and [LCDM Sampling on the HPC](../pipelines/lcdm-chains.md) for running many chains in parallel.

## Outputs

`sample_joint` returns `param_vals`, a dict of per-parameter lists. With `hpc_path` set, every iteration appends the latest value of every parameter to `{hpc_path}{theta}_map_{map_idx}_chain_{sub_chain_idx}_history.txt`, which `chain_analysis.py` reads and analyzes to form distributions out of the sampled data.

