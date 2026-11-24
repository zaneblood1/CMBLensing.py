Effective lensing reconstruction noise of `map_joint` and of the Gibbs sampler (theta held at the ground truth), against the quadratic-estimate N0, at 10 uK-arcmin.

| Setting | Value |
| --- | --- |
| Data | Temperature only (`I`) |
| Map size | 128 x 128 pixels at 2.5 arcmin per pixel |
| White noise level | 10 uK-arcmin, no 1/f noise, no beam |
| Noise definition | N = C_phi (1 / r^2 - 1) per mode, the moments averaged over realizations before the ratio |
| Joint MAP | 100 realizations; r^2 from the correlation of the estimate with the true phi, <Re(phi_est phi_true*)>^2 / (<\|phi_est\|^2> <\|phi_true\|^2>) |
| Gibbs sampler | 100 realizations, one chain per map, burn-in 600 sweeps; r^2 from the response of the posterior mean to the true phi, <Re(phi_mean phi_true*)> / <\|phi_true\|^2> |
| Curves | Band medians over \|L\| bands; the shaded bands are a delete-one-realization jackknife |
| Source files | `map_joint_phi_noise_data/map_joint_phi_noise_output_10uk/map_joint_phi_noise.npz` and `sampler_phi_noise_data/sampler_phi_noise_output_10uk_pre_refactor/sampler_phi_noise_response.npz` on Hugging Face |

**Notes**

- The sampler run has one chain per map, so there is no R-hat check of its convergence; a chain that has not converged reads as extra noise in the response estimator.
- The sampler's posterior mean was not projected onto its Hermitian part in this run (only the kx = 0 and kx = Nyquist columns are affected).
