# Sampler Phi Noise

Measures the phi reconstruction noise reached by the sampler's own posterior mean. One chain per realization samples `(f, phi) | d` with theta **held** at `GROUND_TRUTH` (`sample_joint`'s Gibbs sweep minus the theta step, `fixed_theta_chain.posterior_mixed_draws`), started at phi = 0. Every phi sample is stored.

The merge burns each chain in, averages it into the posterior mean \(\bar\phi\), and forms the same per-mode estimator as the [map_joint phi noise](map-joint-phi-noise.md) with \(\bar\phi\) in place of \(\hat\phi\).

## Running it

```bash
bash get_sampler_phi_noise/get_sampler_phi_noise.sh    #one chain per seed
cd get_sampler_phi_noise
sbatch merge_sampler_phi_noise.sh                      #set burn_in in it first
```

The merge works on **running** jobs: each chain flushes its samples every 10 sweeps to a memory-mapped `sampler_phi_NNNN_samples.npy`, and the merge reads only the sweeps its sidecar `sampler_phi_NNNN.npz` reports as done. A chain not yet past the burn-in is left out. Use `rsync --sparse` to copy a partial run.

Locally, for a small test:

```bash
python -m cmb_lensing.sampler_phi_noise --nside 64 --theta_pix 5 --n_maps 4 --n_sweeps 400 --burn_in 100 --out_dir <dir>
```

## Output: `sampler_phi_noise.npz`

`nphi` (NaN where unmeasured, 0 at the origin), `nphi_error`, `cphi`, the mean moments, `burn_in`, `n_used`, the seeds and the box / cosmology keys. Can be used for forecasting via `fisher_forecast --nphi_source sample_joint --phi_noise sampler_phi_noise.npz` or as the `phi_noise` key in [the delensed covariance](delensed-covariance.md) slurm script.
