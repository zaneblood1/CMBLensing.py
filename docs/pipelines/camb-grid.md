# Building the CAMB Grid

Most users should [download the grid](../getting-started/installation.md#downloading-the-5d-camb-grid). Rebuild it when CAMB is upgraded or the grid's box changes.

## The 5D grid

The 5D spline is used to interpolate real CAMB spectra on a tensor-product grid over `(H0, logA, ns, ombh2, omch2)` with `(tau = 0.05, mnu = 0.06, r = 0)`, evaluated in log spectra space (`cmb_lensing/camb_grid_interp.py`). The grid is laid out in `H0` because a rectangular box in `theta_MC_100` has corners CAMB cannot solve; the sampler works in `theta_MC_100` and converts back and forth through a stored `theta(H0, ombh2, omch2)` table.

```bash
cd camb_grid_generation
bash camb_grid.sh                                            #875 jobs, one 81-point H0 sweep each
python merge_camb_grid.py --grid_dir multi_param_CAMB_grid   #-> multi_param_CAMB_grid/camb_grid_spline.npz
python ../tests/test_camb_grid.py --grid multi_param_CAMB_grid/camb_grid_spline.npz -n 20
```

Regenerating the grid on an cluster will result in `81 x 5 x 5 x 5 x 7 = 70_875` CAMB calls in total. The jobs need only numpy and CAMB. The merge splines one spectrum at a time, in ell chunks, so the tables are never all in memory. Move the result to `camb_splines/camb_grid_spline.npz`.

## The 1D caches

```bash
python camb_grid_generation/precompute_camb_1d.py <theta_MC_100|logA|ns|ombh2|omch2|all> [--test]
```

If you only wish to sample one parameter at a time, you can just create a single 1-D spline. Precomputed versions of these splines are kept in the repository and tracked with git. The above code sweeps one parameter over `PARAM_BOUNDS` (50 nodes) with the others held at `GROUND_TRUTH` and writes `camb_splines/camb_<param>_grid.npz`. The shipped caches were made with CAMB 2.0.0.

