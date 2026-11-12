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

Every grid setting has a default in `camb_grid.sh` and can be overridden on the command line as `--<name>=<value>`: the CAMB settings (`lmax`, `tau`, `mnu`, `k_pivot`, `alens`, `accuracy_boost`, `l_sample_boost`, `l_accuracy_boost`), the output folder's name inside `camb_grid_generation/` (`out_dir_name`), each axis range (`h0_min` / `h0_max`, `log_a_*`, `ns_*`, `ombh2_*`, `omch2_*`) and the nodes per axis (`h0_nodes`, `log_a_nodes`, `ns_nodes`, `ombh2_nodes`, `omch2_nodes`). For example

```bash
bash camb_grid.sh --lmax=5000 --h0_nodes=101 --out_dir_name=grid_lmax_5000
```

A misspelled name is not an error: it is simply ignored and the default is used. Every run writes the submission date and all setting values to `submission_settings.txt` in the output folder (replacing any earlier copy), so check that file to see what a grid was built with. Use a new `out_dir_name` when you change settings.

Point `merge_camb_grid.py --grid_dir` at the folder you named.

Regenerating the grid on an cluster will result in `81 x 5 x 5 x 5 x 7 = 70_875` CAMB calls in total. The jobs need only numpy and CAMB. The merge splines one spectrum at a time, in ell chunks, so the tables are never all in memory. Move the result to `camb_splines/camb_grid_spline.npz` (a file already there is used as is, never re-downloaded). If the new grid should replace the shared one, upload it to the [Hugging Face dataset](https://huggingface.co/datasets/zaneblood1/5D_LCDM_CAMB_Grid_Spline) and tell everyone to delete their local copy.
