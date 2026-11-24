Run time and accuracy of `map_joint` in CMBLensing.py against CMBLensing.jl on the same simulated data.

| Quantity | Value |
| --- | --- |
| Data | Temperature plus polarization (`IP`) |
| Map sizes | 128, 256, 512 and 1024 pixels per side |
| Resolution | 2.5 arcminutes per pixel |
| White noise level | 3 uK-arcmin in T, sqrt(2) x 3 uK-arcmin in E and B |
| 1/f noise | l_knee = 100, alpha_knee = 3 |
| Beam | None (FWHM = 0) |
| Mask | No pixel space sky mask. Fourier space low-pass held at unity through l = 2950, then a cosine ramp down to zero at l = 3000 |
| `num_steps` | 30 |
| Data maps per map size | 10 |
| Trials per data map | 10 |
| Cached timing measurements per map size, per language | 100 |
