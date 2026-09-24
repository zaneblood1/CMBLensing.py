"""Average the per-realization delensed covariances into the 1 + 2k stencil matrices.

Run locally after get_delensed_covariance.sh has finished on the HPC and its out_dir has been
scp'd over:

    python merge_delensed_covariance.py --covariance_dir <local copy of out_dir>

Writes delensed_covariance.npz into that directory (the realization mean of the delensed,
unlensed and lensed per-mode covariances at theta_0 and theta_0 +/- h_i, the per-mode empirical
phi noise when the jobs stored the phi moments, plus band-averaged diagnostics) and
delensed_covariance.png (+ delensed_covariance_phi_noise.png). The npz is what fisher_forecast.py's
--delensed_covariance reads. Refuses files that disagree on the box, the stencil or the
reconstruction, refuses duplicate seeds, and sets aside unfinished (checkpointed) files.

The printed report opens with the common-random-number check - the empirical UNLENSED
dlnC/dtheta must equal CAMB's per mode to machine precision - and then, per parameter and
band, the delensed log-derivative next to the unlensed and lensed ones, empirical and CAMB.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cmb_lensing.delensed_covariance import merge_delensed_covariance, MERGED_NAME
from cmb_lensing.delensed_spectrum import DEFAULT_DELTA_ELL
from cmb_lensing.precompute_camb_1d import PARAM_SIGMA


def plot_delensed_covariance(merged, path):
    """Top: residual lensing D(l) at the centre. Below: sigma * dlnC/dtheta per parameter."""
    ells = merged["band_ells"]
    names = [str(name) for name in merged["names"]]
    figure, axes = plt.subplots(1 + len(names), 1, figsize = (9, 3 + 3 * len(names)),
                                sharex = True)

    axes[0].axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    axes[0].plot(ells, merged["band_c_delensed"] / merged["band_c_unlensed"], marker = "o",
                 markersize = 3, color = "C0", label = "empirical delensed / unlensed")
    axes[0].plot(ells, merged["band_c_lensed"] / merged["band_c_unlensed"], marker = ".",
                 linestyle = ":", color = "C3", label = "empirical lensed / unlensed")
    axes[0].plot(ells, merged["band_c_camb_lensed"] / merged["band_c_camb_unlensed"],
                 color = "0.3", linewidth = 1, label = "CAMB lensed / unlensed")
    axes[0].set_ylabel("C / C_unlensed")
    axes[0].legend(fontsize = 8)
    axes[0].set_title(f"Empirical delensed covariance  |  nside {merged['nside']}, "
                      f"{merged['theta_pix']:g}', {merged['noise_level']:g} uK-arcmin  |  "
                      f"{merged['n_realizations']} realizations, h = "
                      f"{merged['step_sigma']:g} sigma, {merged['reconstruction']} "
                      f"reconstruction", fontsize = 10)

    for i, name in enumerate(names):
        axis = axes[1 + i]
        sigma = PARAM_SIGMA[name]
        axis.axhline(0.0, color = "0.5", linestyle = "--", linewidth = 1)
        axis.errorbar(ells, merged["band_dlnc_delensed"][i] * sigma,
                      yerr = merged["band_dlnc_delensed_error"][i] * sigma, marker = "o",
                      markersize = 3, capsize = 2, color = "C0", label = "delensed (empirical)")
        axis.plot(ells, merged["band_dlnc_camb_unlensed"][i] * sigma, color = "C2",
                  label = "unlensed (CAMB)")
        axis.plot(ells, merged["band_dlnc_unlensed"][i] * sigma, color = "C2", marker = "x",
                  linestyle = "none", markersize = 4, label = "unlensed (empirical)")
        axis.plot(ells, merged["band_dlnc_camb_lensed"][i] * sigma, color = "C3",
                  label = "lensed (CAMB)")
        axis.plot(ells, merged["band_dlnc_lensed"][i] * sigma, color = "C3", marker = "+",
                  linestyle = "none", markersize = 5, label = "lensed (empirical)")
        axis.set_ylabel(rf"$\sigma\ \partial \ln C / \partial$ {name}", fontsize = 9)
        axis.legend(fontsize = 7, ncol = 2)
    axes[-1].set_xlabel(r"$\ell$")

    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def plot_phi_noise(merged, path):
    """Top: band N_eff against the QE N^(0) and C_phi, plus a per-mode map of N_eff / N^(0)
    (where map_joint beats the quadratic estimator, mode by mode). Below: sigma * dlnN/dtheta."""
    ells = merged["band_ells"]
    names = [str(name) for name in merged["names"]]
    figure = plt.figure(figsize = (13, 4 + 3 * len(names)))
    grid = figure.add_gridspec(1 + len(names), 2, width_ratios = [2, 1])

    axis = figure.add_subplot(grid[0, 0])
    axis.errorbar(ells, merged["band_phi_noise"], yerr = merged["band_phi_noise_error"],
                  marker = "o", markersize = 3, capsize = 2, color = "C0",
                  label = "N_eff (map_joint, empirical)")
    axis.plot(ells, merged["band_phi_noise_qe"], color = "C1", label = "N^(0) (box QE)")
    axis.plot(ells, merged["band_c_camb_phi"], color = "0.3", label = "C_phi (CAMB)")
    axis.set_yscale("log")
    axis.set_ylabel("grid units")
    axis.legend(fontsize = 8)
    smoothing = merged["phi_smooth_delta_ell"]
    axis.set_title(f"Empirical phi noise  |  nside {merged['nside']}, "
                   f"{merged['theta_pix']:g}', {merged['noise_level']:g} uK-arcmin  |  "
                   f"{merged['n_realizations']} realizations"
                   + (f", moments smoothed over {smoothing:g}" if smoothing > 0 else ""),
                   fontsize = 10)

    #per-mode N_eff / N^(0), fftshifted along the full axis so the origin sits mid-left
    image_axis = figure.add_subplot(grid[0, 1])
    noise = np.asarray(merged["phi_noise_fid"], dtype = float)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        ratio = np.log10(noise / np.asarray(merged["phi_noise_qe_fid"]))
    image = image_axis.imshow(np.fft.fftshift(ratio, axes = 0), aspect = "auto",
                              cmap = "RdBu_r", vmin = -1.5, vmax = 1.5)
    image_axis.set_title("log10 N_eff / N^(0) per mode;\nwhite = unmeasured", fontsize = 9)
    image_axis.set_xlabel("kx index")
    image_axis.set_ylabel("ky index (shifted)")
    figure.colorbar(image, ax = image_axis, fraction = 0.046)

    for i, name in enumerate(names):
        axis = figure.add_subplot(grid[1 + i, :])
        sigma = PARAM_SIGMA[name]
        axis.axhline(0.0, color = "0.5", linestyle = "--", linewidth = 1)
        axis.errorbar(ells, merged["band_dlnn_phi"][i] * sigma,
                      yerr = merged["band_dlnn_phi_error"][i] * sigma, marker = "o",
                      markersize = 3, capsize = 2, color = "C0")
        axis.set_ylabel(rf"$\sigma\ \partial \ln N_{{eff}} / \partial$ {name}", fontsize = 9)
    figure.axes[-1].set_xlabel(r"$L$")

    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description = "Average per-realization delensed covariances into the stencil")
    parser.add_argument("--covariance_dir", type = str,
                        default = "/home/zane-blood/Desktop/cmb_lensing/sampling_chains/"
                                  "delensed_covariance_output/",
                        help = "the (scp'd) out_dir get_delensed_covariance.sh wrote to")
    parser.add_argument("--delta_ell", type = float, default = DEFAULT_DELTA_ELL,
                        help = "band width of the printed / plotted diagnostics only; the "
                               "stored matrices are per mode")
    parser.add_argument("--smooth_delta_ell", type = float, default = 0.0,
                        help = "average the phi moments in |L| annuli of this width before "
                               "forming the per-mode phi noise (isotropic, quieter). 0 "
                               "(default) keeps every mode, i.e. the box's anisotropy")
    parser.add_argument("--out_name", type = str, default = MERGED_NAME)
    args = parser.parse_args()

    merged = merge_delensed_covariance(args.covariance_dir, delta_ell = args.delta_ell,
                                       smooth_delta_ell = args.smooth_delta_ell)

    out_path = os.path.join(args.covariance_dir, args.out_name)
    np.savez(out_path, **merged)
    plot_path = os.path.splitext(out_path)[0] + ".png"
    plot_delensed_covariance(merged, plot_path)

    print(f"\nwrote {out_path}")
    print(f"wrote {plot_path}")
    if merged["has_phi_moments"]:
        phi_path = os.path.splitext(out_path)[0] + "_phi_noise.png"
        plot_phi_noise(merged, phi_path)
        print(f"wrote {phi_path}")
    print(f"\nUse it with:\n"
          f"    python -m cmb_lensing.fisher_forecast --spectra delensed "
          f"--nside {merged['nside']} --theta_pix {merged['theta_pix']:g} "
          f"--noise {merged['noise_level']:g} --l_knee {merged['l_knee']:g} "
          f"--params {' '.join(str(name) for name in merged['names'])} "
          f"--delensed_covariance {out_path}"
          + (" [--empirical_phi_noise [--vary_nphi]]" if merged["has_phi_moments"] else ""))


if __name__ == "__main__":
    main()
