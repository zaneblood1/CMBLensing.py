"""Average the per-realization delensed-spectrum files into one transfer function R(l).

Run after get_delensed_spectra.sh has finished, pointing --spectra_dir at the out_dir that
script wrote to:

    python merge_delensed_spectra.py --spectra_dir ABSOLUTE_PATH_TO/delensed_spectra_output

Writes transfer_function.npz into that directory (plus transfer_function.png) - the file
fisher_forecast.py's --transfer_function reads. Refuses to average files that disagree on the
box, the noise, the binning, the reconstruction settings, or the cosmology, and refuses
duplicate seeds; see cmb_lensing/delensed_spectrum.py for why each of those would be wrong.

The default estimator is the cosmic-variance-cancelled "paired" one. Both are computed, and
the run prints the fractional difference between them: they estimate the same quantity and
differ only in variance, so a real disagreement means the pairing is doing something other
than cancelling and the result should not be used.

With --shifted_dirs (the out_dirs of get_delensed_spectra.sh's derivative_params mode) it also
measures dR/dtheta_i from the common-random-number runs and stores it in the same npz, so the
forecast no longer has to assume R is flat in theta:

    python merge_delensed_spectra.py --spectra_dir <out_dir>/reference \\
        --shifted_dirs <out_dir>/*_plus <out_dir>/*_minus

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cmb_lensing.delensed_spectrum import (merge_transfer_function, merge_transfer_derivatives,
                                           MERGED_NAME)
from cmb_lensing.precompute_camb_1d import PARAM_ORDER, PARAM_SIGMA


def plot_transfer_function(merged, path):
    """R(l) with its jackknife band, over the two validation rungs that share its axis."""
    ells = merged["band_ells"]
    figure, axes = plt.subplots(2, 1, figsize = (9, 8), sharex = True,
                                gridspec_kw = dict(height_ratios = [2, 1]))

    axes[0].axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    axes[0].errorbar(ells, merged["transfer"], yerr = merged["transfer_error"],
                     marker = "o", markersize = 4, capsize = 3, color = "C0",
                     label = f"R(l), {merged['estimator']} estimator")
    axes[0].plot(ells, merged["transfer_naive"], marker = ".", linestyle = ":",
                 color = "C3", alpha = 0.7, label = "naive estimator (cross-check)")
    axes[0].set_ylabel(r"$R(\ell) = C_\ell^{\rm delensed,\ box} / C_\ell^{\rm delensed,\ CAMB}$")
    axes[0].legend(fontsize = 9)
    axes[0].set_title(f"Empirical delensing transfer function  |  "
                      f"nside {merged['nside']}, {merged['theta_pix']:g}', "
                      f"{merged['noise_level']:g} uK-arcmin  |  "
                      f"{merged['n_realizations']} realizations", fontsize = 10)

    axes[1].axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    axes[1].errorbar(ells, merged["rung_0"], yerr = merged["rung_0_error"],
                     marker = "s", markersize = 3, capsize = 2, color = "C2",
                     label = "rung 0: measured unlensed / input $C_f$")
    axes[1].errorbar(ells, merged["rung_1"], yerr = merged["rung_1_error"],
                     marker = "^", markersize = 3, capsize = 2, color = "C1",
                     label = "rung 1: measured lensing / CAMB lensing")
    axes[1].set_xlabel(r"$\ell$")
    axes[1].set_ylabel("validation")
    axes[1].legend(fontsize = 9)

    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def plot_transfer_derivatives(merged, path):
    """sigma_i dR/dtheta_i per parameter - how far R moves over one sigma - with its error."""
    ells = merged["band_ells"]
    figure, axis = plt.subplots(figsize = (9, 5))
    axis.axhline(0.0, color = "0.5", linestyle = "--", linewidth = 1)
    for i, name in enumerate(merged["derivative_names"]):
        sigma = PARAM_SIGMA[str(name)]
        axis.errorbar(ells, merged["transfer_derivative"][i] * sigma,
                      yerr = merged["transfer_derivative_error"][i] * sigma,
                      marker = "o", markersize = 3, capsize = 2, color = f"C{i}",
                      label = f"{name} ({merged['derivative_schemes'][i]}, "
                              f"{int(merged['derivative_n_realizations'][i])} realizations)")
    axis.set_xlabel(r"$\ell$")
    axis.set_ylabel(r"$\sigma_i\ \partial R(\ell) / \partial \theta_i$")
    axis.set_title("Empirical delensing transfer function: drift per sigma  |  "
                   f"nside {merged['nside']}, {merged['theta_pix']:g}', "
                   f"{merged['noise_level']:g} uK-arcmin", fontsize = 10)
    axis.legend(fontsize = 9)
    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description = "Average per-realization delensed spectra into a transfer function")
    parser.add_argument("--spectra_dir", type = str, required = True,
                        help = "the out_dir get_delensed_spectra.sh wrote its jobs to")
    parser.add_argument("--estimator", choices = ("paired", "naive"), default = "paired",
                        help = "which R estimator the npz advertises as `transfer`. "
                               "'paired' (default) divides by the same realization's "
                               "unlensed spectrum first, cancelling the cosmic variance")
    parser.add_argument("--out_name", type = str, default = MERGED_NAME)
    parser.add_argument("--shifted_dirs", type = str, nargs = "+", default = None,
                        help = "out_dirs of runs with ONE parameter displaced, on the same "
                               "seeds as --spectra_dir and with the reconstruction frozen at "
                               "its cosmology (get_delensed_spectra.sh's derivative_params "
                               "mode). Adds dR/dtheta to the npz, so fisher_forecast "
                               "evaluates R at every stencil point instead of holding it flat")
    args = parser.parse_args()

    merged = merge_transfer_function(args.spectra_dir, estimator = args.estimator)
    if args.shifted_dirs:
        merged.update(merge_transfer_derivatives(args.spectra_dir, args.shifted_dirs,
                                                 estimator = args.estimator))

    out_path = os.path.join(args.spectra_dir, args.out_name)
    np.savez(out_path, **merged)
    plot_path = os.path.splitext(out_path)[0] + ".png"
    plot_transfer_function(merged, plot_path)
    if args.shifted_dirs:
        derivative_path = os.path.splitext(out_path)[0] + "_derivatives.png"
        plot_transfer_derivatives(merged, derivative_path)
        print(f"wrote {derivative_path}")

    cosmology = ", ".join(f"{name} {value:.6g}"
                          for name, value in zip(PARAM_ORDER, merged["params"]))
    print(f"\nmeasured at: {cosmology}")
    print(f"wrote {out_path}")
    print(f"wrote {plot_path}")
    print(f"\nUse it with:\n"
          f"    python -m cmb_lensing.fisher_forecast --spectra delensed "
          f"--transfer_function {out_path}")


if __name__ == "__main__":
    main()
