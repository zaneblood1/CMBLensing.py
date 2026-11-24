Phi noise spectra plots for the results visualizer, one folder per data set (a sampler phi noise run paired with a map_joint phi noise run at one noise level):

```
phi_noise/<dataset>/           phi_noise_spectra_<noise>uk.png
                              phi_noise_spectra_over_qe_<noise>uk.png
                              sampler_rhat_<noise>uk.png          (optional)
                              SUMMARY.md                          (optional)
```

The first two are written by `measure_phi_noise/phi_noise_analysis.py`, the third by `measure_phi_noise/get_sampler_phi_noise/plot_sampler_convergence.py`. See `../README.md`.
