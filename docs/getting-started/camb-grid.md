# The CAMB Grid

`sample_joint` needs the theory power spectra at every proposed LCDM parameter combination. It gets them from a precomputed grid of real CAMB spectra: a 5D tensor-product cubic spline over `(H0, logA, ns, ombh2, omch2)`, stored in `camb_splines/camb_grid_spline.npz` (~7.9 GB). It serves any subset of the five parameters and every polarization mode.

The file is fetched automatically from the [Hugging Face dataset](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data) the first time it is needed (or ahead of time with `cmb-lensing-download-splines`, see [Installation](installation.md#downloading-the-5d-camb-grid)). To use your own grid instead, rebuild it on the cluster using the slurm scripts in ``/camb_grid_generation``, SCP it back to your local machine and place it at `camb_splines/camb_grid_spline.npz`.

## Memory

The grid is opened once per process and each spectrum's spline is built lazily on first use (~2.1 GB resident per spectrum). Temperature-only sampling loads TT + PP (~4.5 GB); `pol = "IP"` loads TT / EE / TE / PP (~9.1 GB).

## Validating the grid

```bash
python tests/test_camb_grid.py --grid camb_splines/camb_grid_spline.npz -n 20
```

compares the spline against direct CAMB at random interior points. Typical max |Δ ln Cl| is ~1e-3 (TT) to ~5e-3 (EE, near an acoustic trough).
