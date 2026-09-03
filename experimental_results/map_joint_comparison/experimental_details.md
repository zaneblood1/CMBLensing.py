| Quantity | Value |
| --- | --- |
| Data | Temperature plus polarization (```IP```) |
| Map sizes | $128^2$, $256^2$, $512^2$, $1024^2$ pixels |
| Resolution | $2.5$ arcminutes per pixel |
| White noise level | $3$ $\mu K$ arcminutes in $T$, $\sqrt{2} \times 3$ $\mu K$ arcminutes in $E$ and $B$ |
| $1/f$ noise | $\ell$-knee $= 100$, $\alpha$-knee $= 3$ |
| Beam | None (FWHM $= 0$) |
| Mask | No pixel space sky mask. Fourier space low-pass held at unity through $\ell = 2950$, then a cosine ramp down to zero at $\ell = 3000$ |
| ```num_steps``` | $30$ |
| Data maps per map size | $10$ |
| Trials per data map | $10$ |
| Cached timing measurements per map size, per language | $100$ |