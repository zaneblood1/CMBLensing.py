"""One-glance convergence check of the sampler phi noise chains.

Reads a merged sampler_phi_noise.npz (merge_sampler_phi_noise.py) and plots, per |L| band, the
Gelman-Rubin R-hat across each data map's chains: for every map the R-hat of the real and
imaginary parts of the band's Fourier modes are averaged, and the plot shows the mean of that
over maps with +/- 1 standard deviation over maps as error bars. Every band near 1 means the
chains started at phi = 0 and at prior draws agree after the burn-in; a band well above 1 means
the burn-in (or the chain length) was too short there.

    python plot_sampler_convergence.py [--merged PATH_TO/sampler_phi_noise.npz]

Writes sampler_rhat_<noise>uk.png next to the merged file.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.delensed_covariance import band_edges

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MERGED = os.path.join(HERE, "sampler_phi_noise_output", "sampler_phi_noise.npz")

#R-hat below this counts as converged (the usual modern threshold)
R_HAT_REFERENCE = 1.01
FIGURE_SIZE = (10, 6)
DPI = 150
TITLE_FONTSIZE = 13
LABEL_FONTSIZE = 14

def band_mean_rhat(rhat_real, rhat_imaginary, ell_grid, edges):
    """(n_maps, n_bands): each map's R-hat averaged over the real and imaginary parts of the
    band's modes (NaN entries, e.g. the origin, left out)."""
    index = np.digitize(ell_grid, edges) - 1
    n_bands = len(edges) - 1
    values = np.full((len(rhat_real), n_bands), np.nan)
    for band in range(n_bands):
        inside = index == band
        if inside.any():
            both_parts = np.concatenate([rhat_real[:, inside], rhat_imaginary[:, inside]], axis = 1)
            with np.errstate(invalid = "ignore"):
                values[:, band] = np.nanmean(both_parts, axis = 1)
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type = str, default = DEFAULT_MERGED,
                        help = "the merged sampler_phi_noise.npz")
    args = parser.parse_args()

    merged = np.load(args.merged)
    nside = int(merged["nside"])
    theta_pix = float(merged["theta_pix"])
    noise = float(merged["noise_level"])
    ell_grid = np.asarray(gen_ell_grid(nside, theta_pix)[0])
    edges = band_edges(ell_grid)
    centres = 0.5 * (edges[1:] + edges[:-1])

    per_map = band_mean_rhat(np.asarray(merged["rhat_real"]), np.asarray(merged["rhat_imaginary"]),
                             ell_grid, edges)
    with np.errstate(invalid = "ignore"):
        mean = np.nanmean(per_map, axis = 0)
        spread = np.nanstd(per_map, axis = 0)
    good = np.isfinite(mean)

    figure, axis = plt.subplots(figsize = FIGURE_SIZE)
    axis.errorbar(centres[good], mean[good], yerr = spread[good], fmt = "o", markersize = 4,
                  capsize = 3, color = "C0", label = r"mean over maps $\pm$ 1 std over maps")
    axis.axhline(R_HAT_REFERENCE, color = "C3", linestyle = "--", linewidth = 1,
                 label = f"$\\hat R$ = {R_HAT_REFERENCE:g}")
    axis.axhline(1.0, color = "0.5", linewidth = 0.8)
    axis.set_xscale("log")
    axis.set_xlabel(r"$L$ (band centre)", fontsize = LABEL_FONTSIZE)
    axis.set_ylabel(r"Band-averaged Gelman-Rubin $\hat R$ across chains", fontsize = LABEL_FONTSIZE)
    axis.set_title(f"Sampler phi chains | {int(merged['n_realizations'])} maps, "
                   f"{int(np.min(merged['n_chains']))}-{int(np.max(merged['n_chains']))} chains "
                   f"per map, burn-in {int(merged['burn_in'])} | nside {nside}, "
                   f"{theta_pix:g}', {noise:g} $\\mu$K-arcmin", fontsize = TITLE_FONTSIZE)
    axis.legend()
    figure.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.merged)),
                            f"sampler_rhat_{noise:g}uk.png")
    figure.savefig(out_path, dpi = DPI)
    plt.close(figure)
    print(f"wrote {out_path}")

    worst = np.nanargmax(np.where(good, mean, -np.inf))
    print(f"worst band: L = {centres[worst]:.0f}, mean R-hat {mean[worst]:.4f} "
          f"(std {spread[worst]:.4f}); {int(np.sum(mean[good] > R_HAT_REFERENCE))} of "
          f"{int(np.sum(good))} bands have mean R-hat above {R_HAT_REFERENCE:g}")

if __name__ == "__main__":
    main()
