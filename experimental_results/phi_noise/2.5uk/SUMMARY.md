Lensing reconstruction noise of `map_joint` and of the Gibbs sampler (cosmological parameters held at the ground truth), compared with the quadratic-estimate noise N<sup>(0)</sup>, at 2.5 uK-arcmin.

| Setting | Value |
| --- | --- |
| Data | Temperature only (`I`) |
| Map size | 128 x 128 pixels at 2.5 arcmin per pixel |
| White noise level | 2.5 uK-arcmin, no 1/f noise, no beam |
| Joint MAP | 100 realizations |
| Gibbs sampler | 100 realizations, one chain per map, burn-in 1000 sweeps |
| Curves | Medians over bands of the multipole L; the shaded bands are a delete-one-realization jackknife |
| Source files | `map_joint_phi_noise_data/map_joint_phi_noise_output_2.5uk/map_joint_phi_noise.npz` and `sampler_phi_noise_data/sampler_phi_noise_output_2.5uk_pre_refactor/sampler_phi_noise_response.npz` on Hugging Face |

**How the noise is measured**

Each reconstruction is compared with the true lensing potential &phi;<sub>true</sub> of its simulation, one Fourier mode at a time:

N<sub>&phi;</sub> = C<sub>&phi;</sub> (1 / r<sup>2</sup> &minus; 1)

Here C<sub>&phi;</sub> is the power spectrum of the true lensing potential, and r<sup>2</sup> is the fraction of the true potential that the reconstruction recovers: r<sup>2</sup> = 1 gives no noise, and r<sup>2</sup> near 0 gives a noise far above the signal. The angle brackets below are averages over the realizations, taken before the ratio is formed.

- **Joint MAP.** r<sup>2</sup> is the squared correlation of the MAP estimate &phi;<sub>MAP</sub> with the true potential:

    r<sup>2</sup> = &lang;Re(&phi;<sub>MAP</sub> &phi;<sub>true</sub><sup>&lowast;</sup>)&rang;<sup>2</sup> / ( &lang;|&phi;<sub>MAP</sub>|<sup>2</sup>&rang; &lang;|&phi;<sub>true</sub>|<sup>2</sup>&rang; )

- **Gibbs sampler.** r<sup>2</sup> is the response of the posterior mean &phi;<sub>mean</sub> (the average of a chain's samples) to the true potential:

    r<sup>2</sup> = &lang;Re(&phi;<sub>mean</sub> &phi;<sub>true</sub><sup>&lowast;</sup>)&rang; / &lang;|&phi;<sub>true</sub>|<sup>2</sup>&rang;

    The response is used here instead of the correlation because a chain is finite: its mean is the exact posterior mean plus a Monte Carlo error &epsilon;, &phi;<sub>mean</sub> = &phi;<sub>exact</sub> + &epsilon;. The correlation divides by &lang;|&phi;<sub>mean</sub>|<sup>2</sup>&rang; = &lang;|&phi;<sub>exact</sub>|<sup>2</sup>&rang; + &lang;|&epsilon;|<sup>2</sup>&rang;, and the &lang;|&epsilon;|<sup>2</sup>&rang; term is never zero. It adds spurious power to the sampler mean, which lowers r<sup>2</sup> and inflates the noise. The response never uses |&phi;<sub>mean</sub>|<sup>2</sup>, and &epsilon; is not correlated with the true potential, so it averages out of &lang;Re(&phi;<sub>mean</sub> &phi;<sub>true</sub><sup>&lowast;</sup>)&rang;. For the exact posterior mean the two definitions of r<sup>2</sup> agree.

**Notes**

- The sampler run has one chain per map, so its convergence cannot be checked with the Gelman-Rubin statistic. A chain that has not converged shows up as extra noise.
- The sampler's posterior mean was not projected onto its Hermitian part in this run. Only the modes with k<sub>x</sub> = 0 and k<sub>x</sub> at the Nyquist frequency are affected.
