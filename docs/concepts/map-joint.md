# MAP Estimation

`map_joint(data_set, num_steps = 30, constant_step = False)` (`cmb_lensing/map_joint.py`) finds the joint maximum a posteriori `(f, phi)` given the data, following [Millea et al. 2017](https://arxiv.org/abs/1708.06753). The objective is

\[
-2 \log P(f, \phi \mid d) = \frac{(d - M B L(\phi) f)^2}{C_n} + \frac{f^2}{C_f} + \frac{\phi^2}{C_\phi} + \text{const.}
\]

Each step alternates

1. a **Wiener filter** for f at the current phi, and
2. a **BFGS line search** on phi along the preconditioned mixed gradient (`mixed_grad_phi_logpdf`), with Hessian approximation `pinv(Cphi^-1 + Nphi^-1)`.

The G mixing matrix is set to the identity (the MAP is invariant to it).

```python
from cmb_lensing.simulate import load_sim
from cmb_lensing.map_joint import map_joint

data_set = load_sim(nside = 128, theta_pix = 2.5, pol = "I", master_seed = 67, uk_arcmin_t = 5)
field_predict, phi_predict = map_joint(data_set, num_steps = 30)
```

A full worked example, with plots of the lensing, inverse lensing, logpdf and the cross spectra of the reconstruction against the truth, can be found in the [map_joint tutorial](../map_joint_tutorial.ipynb) notebook.
