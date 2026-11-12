# Sampler Phi Noise

Measures the phi reconstruction noise reached by the sampler's own posterior mean. Several chains per realization sample `(f, phi) | d` with theta **held** at `GROUND_TRUTH` (`sample_joint`'s Gibbs sweep minus the theta step, `fixed_theta_chain.posterior_mixed_draws`). Chain 0 of each realization starts at phi = 0 and every other chain at its own draw from the \(C^{\phi\phi}\) prior, so the chains start far apart. Every phi sample is stored.

The merge burns every chain in and pools each realization's chains into its posterior mean \(\bar\phi\). Per mode, with the averages taken over realizations before the ratio,

\[
r^2 = \frac{\langle \mathrm{Re}(\bar\phi\,\phi_{\rm true}^*)\rangle}{\langle|\phi_{\rm true}|^2\rangle},
\qquad N = C^{\phi\phi}\left(\frac{1}{r^2} - 1\right).
\]

This is the posterior mean's response to the truth. For the exact posterior mean it equals the squared correlation \(\langle \mathrm{Re}(\bar\phi\,\phi_{\rm true}^*)\rangle^2 / (\langle|\bar\phi|^2\rangle\langle|\phi_{\rm true}|^2\rangle)\) used for the [map_joint phi noise](map-joint-phi-noise.md), but it never uses \(\langle|\bar\phi|^2\rangle\), which the Monte Carlo error of a finite chain inflates. It is not valid for a MAP estimate, so map_joint keeps the correlation estimator.

## Running it

```bash
nohup bash measure_phi_noise/get_sampler_phi_noise/get_sampler_phi_noise.sh > submit.log 2>&1 &   #num_maps x num_chains jobs
cd measure_phi_noise/get_sampler_phi_noise
sbatch merge_sampler_phi_noise.sh --burn_in=600                          #default burn_in 300
python plot_sampler_convergence.py --merged sampler_phi_noise_output/sampler_phi_noise.npz
```

The driver submits through the [submission throttle](../getting-started/hpc-setup.md#submission-throttle), so it keeps running until every job is submitted.

Its settings (`nside`, `theta_pix`, `noise_level`, `l_knee`, `pol` (`I`, `P` or `IP`: the data the chains sample from; default `I`), `hmc_step_size`, `hmc_num_steps`, `num_sweeps`, `num_maps`, `seed_prefix`, `num_chains`, `chain_time`, `chain_mem`, and `out_dir_name`, the output folder's name inside `get_sampler_phi_noise/`, default `sampler_phi_noise_output`) are defaults that can be overridden on the command line as `--<name>=<value>`, e.g.

```bash
bash measure_phi_noise/get_sampler_phi_noise/get_sampler_phi_noise.sh --noise_level=10 --num_chains=4 --out_dir_name=sampler_phi_noise_output_10uk
```

A misspelled name is ignored and the default used. Each run writes the submission date and every setting to `submission_settings.txt` in the output folder (replacing any earlier copy); use a new `out_dir_name` when you change settings.

Each chain's random numbers come from its **chain seed**. `--chain_seeds="11 22"` gives chain 0 of every map the seed 11, chain 1 the seed 22, and so on; a chain with no seed listed (the default, `chain_seeds` empty) draws a random one, as `sample_joint` does. Either way the seed is saved as `chain_seed` in the chain's sidecar `sampler_phi_NNNN_cK.npz`, and rerunning with it reproduces the chain exactly. Give the chains of one map different seeds: two chains with the same seed would be identical.

The merge takes the same kind of overrides after the script name, `burn_in` (default 300) and `out_dir_name` (the driver's, default `sampler_phi_noise_output`):

```bash
sbatch merge_sampler_phi_noise.sh --burn_in=600 --out_dir_name=sampler_phi_noise_output_10uk
```

Each successful merge appends a line `merged <date> by <user>: burn_in=<value>` to that folder's `submission_settings.txt`, so the last such line describes the current `sampler_phi_noise.npz`.

The merge works on **running** jobs: each chain flushes its samples every 10 sweeps to a memory-mapped `sampler_phi_NNNN_cK_samples.npy` (realization NNNN, chain K), and the merge reads only the sweeps its sidecar `sampler_phi_NNNN_cK.npz` reports as done. A chain not yet past the burn-in is left out, and a realization's chains are trimmed to the shortest. Use `rsync --sparse` to copy a partial run.

Locally, for a small test:

```bash
python -m cmb_lensing.sampler_phi_noise --nside 64 --theta_pix 5 --n_maps 4 --n_chains 2 --n_sweeps 400 --burn_in 100 --out_dir <dir>
```

## Checking the burn-in

`plot_sampler_convergence.py` writes `sampler_rhat_<noise>uk.png` next to the merged file. For each map it computes the Gelman-Rubin \(\hat R\) across that map's chains for the real and imaginary part of every Fourier mode, averages it over each \(|L|\) band, and plots the mean over maps with \(\pm 1\) standard deviation over maps as error bars. Every band near 1 (below the 1.01 reference line) means the chains agree after the burn-in; a band well above 1 means the burn-in or the chain length was too short there.

## Output: `sampler_phi_noise.npz`

`nphi` (NaN where unmeasured, 0 at the origin), `nphi_error`, the N leave-one-out `nphi_leave_one_out`, `rhat_real` / `rhat_imaginary` (per map and mode), `cphi`, the mean moments, `burn_in`, `n_chains`, `n_used`, the seeds and the box / cosmology keys (including `pol`). Can be used for forecasting via `fisher_forecast --delensed_covariance <npz> --phi_noise sampler_phi_noise.npz` (with the matching `--pol`) or as the `phi_noise` key in [the delensed covariance](delensed-covariance.md) slurm script.

## Plotting the phi noise spectra

```bash
python measure_phi_noise/phi_noise_analysis.py
```

draws the merged `map_joint_phi_noise.npz` and `sampler_phi_noise.npz` (paths and box set in its `__main__`) against CAMB's \(C_L^{\phi\phi}\) and the box QE noise, into `measure_phi_noise/phi_noise_plots/noise_level_<noise>uk/`: `phi_noise_spectra_<noise>uk.png` (band medians) and `phi_noise_spectra_over_qe_<noise>uk.png` (band median of the ratio to the lensed-filter QE, with a jackknife band).
