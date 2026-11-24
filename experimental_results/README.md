The data and figures in these folders are maintained by hand, and the figures are what the [results visualizer](https://zaneblood1.github.io/CMBLensing.py/results/) shows.

## Layout

```
experimental_results/<section>/<dataset>/      e.g. lcdm_sampling/2.5uk_summer_2026/
    *.png, <group>/*.png      the figures the visualizer shows; a sub-folder is one headed group
    SUMMARY.md                optional: shown above the figures (how the run was generated)
    data/                     optional: what the figures were made from
```

- `<section>` is one of the things the page can show: `lcdm_sampling` (LCDM Sampler), `map_joint_comparison` (Python v.s. Julia Runtime Comparison) or `phi_noise` (Noise Spectra). A new folder next to them becomes a new one.
- `<dataset>` is one option of that section's data set toggle. The folder name is the label on the toggle, so name it whatever you like.

A `SUMMARY.md` describes the data set to a reader of the page (what was run, the settings that matter, anything to know before reading the figures). It is ordinary Markdown, tables included; the existing ones are templates to copy. An empty or missing one shows nothing.

A few figures are kept here but left off the page (`HIDDEN_FIGURE_PATTERNS` in `build_results_site.py`): the covariance / correlation matrices and their ratios (only the triangle plot shows), the histograms of per-map means / modes, the single-component cross correlations (only the `_all_` panel figures show), the run time tables and the uncached run time plot, and the Julia vs Python prediction difference. A parameter's autocorrelation and integrated autocorrelation time plots share one row.

## Adding a data set

Take the folder of figures, rename it, drop it into its section, commit and push to master. The docs workflow rebuilds the site, and `build_results_site.py` (a MkDocs hook, see `mkdocs.yml`) copies the figures into it and writes the manifest the page reads. Nothing else needs editing. `mkdocs serve` previews it locally at http://127.0.0.1:8000/results/.

```bash
#LCDM sampler: the chain_analysis_output folder chain_analysis.py writes is a data set as it is
cp -r <chain_dir>/chain_analysis_output experimental_results/lcdm_sampling/<any name>

#noise spectra: the plots of measure_phi_noise/phi_noise_analysis.py (and, if wanted, the
#sampler_rhat_<noise>uk.png of plot_sampler_convergence.py)
cp -r measure_phi_noise/phi_noise_plots/noise_level_2.5uk experimental_results/phi_noise/<any name>

#runtime comparison: the folder runtime_comparison/performance_analysis.py writes
cp -r runtime_comparison/performance_analysis experimental_results/map_joint_comparison/<any name>
```

## Sections

`map_joint_comparison/` holds plots and tables comparing the run time and accuracy of the `map_joint` algorithm between the CMBLensing.jl and CMBLensing.py implementations, along with the .txt and .npz files needed to regenerate these figures.

`lcdm_sampling/` holds the learned distributions, autocorrelation plots, data, etc. from runs of the `sample_joint` method in `sample_lcdm.py`.

`phi_noise/` holds the lensing reconstruction noise spectra of `map_joint` and of the sampler against the quadratic estimate.
