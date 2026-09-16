"""Average the per-realization files into one effective noise spectrum N_L^eff.

Run after get_effective_phi_noise.sh has finished, pointing --noise_dir at the out_dir that
script wrote to:

    python merge_phi_noise.py --noise_dir ABSOLUTE_PATH_TO/phi_noise_output

Writes effective_phi_noise.npz into that directory (plus effective_phi_noise.png) - the file
fisher_forecast.py's --phi_noise reads. Refuses to average files that disagree on the box, the
noise, the binning, the reconstruction settings, or the cosmology, and refuses duplicate
seeds; see cmb_lensing/phi_noise.py for why each of those would be wrong.

The headline number is the Fisher-weighted mean of N_eff / N^(0). It is 1 if map_joint
reconstructs exactly as well as a quadratic estimator, below 1 if it does better. That ratio
is what decides whether a forecast built on the QE's N_phi can be reconciled with the chains
by any rescaling of N_phi at all.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cmb_lensing.phi_noise import merge_phi_noise, MERGED_NAME
from cmb_lensing.precompute_camb_1d import PARAM_ORDER


def plot_phi_noise(merged, path):
    """N_eff against C_phi and the analytic N^(0), over the diagnostics that share its axis."""
    ells = merged["band_ells"]
    figure, axes = plt.subplots(3, 1, figsize = (9, 11), sharex = True,
                                gridspec_kw = dict(height_ratios = [2, 1, 1]))

    axes[0].loglog(ells, merged["cphi_band"], color = "0.4", linestyle = "-",
                   label = r"$C_L^{\phi\phi}$")
    axes[0].loglog(ells, merged["qe_noise_band"], color = "C3", linestyle = "--",
                   label = r"analytic QE $N_L^{(0)}$")
    axes[0].errorbar(ells, merged["n_eff"], yerr = merged["n_eff_error"], marker = "o",
                     markersize = 4, capsize = 3, color = "C0", linestyle = "none",
                     label = rf"measured $N_L^{{\rm eff}}$ "
                             rf"({int(merged['n_realizations'])} realizations)")
    axes[0].set_ylabel("power")
    axes[0].legend(fontsize = 9)
    axes[0].set_title(f"Effective lensing reconstruction noise  |  "
                      f"nside {merged['nside']}, {merged['theta_pix']:g}', "
                      f"{merged['noise_level']:g} uK-arcmin  |  "
                      f"map_joint {int(merged['map_joint_steps'])} steps", fontsize = 10)

    #the ratio the whole measurement exists to report
    ratio = merged["n_eff"] / merged["qe_noise_band"]
    ratio_error = merged["n_eff_error"] / merged["qe_noise_band"]
    axes[1].axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    axes[1].errorbar(ells, ratio, yerr = ratio_error, marker = "o", markersize = 4,
                     capsize = 3, color = "C0")
    axes[1].set_xscale("log")
    axes[1].set_ylabel(r"$N_L^{\rm eff} / N_L^{(0)}$")
    axes[1].set_ylim(0, max(2.5, float(np.nanpercentile(ratio, 95))))

    axes[2].axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    axes[2].errorbar(ells, merged["response"], yerr = merged["response_error"],
                     marker = "s", markersize = 3, capsize = 2, color = "C2",
                     label = r"response $\rho_L$")
    axes[2].errorbar(ells, merged["shrinkage"], yerr = merged["shrinkage_error"],
                     marker = "^", markersize = 3, capsize = 2, color = "C1",
                     label = r"shrinkage $\epsilon_L$ (1 iff optimally scaled)")
    axes[2].plot(ells, merged["rung_0"], marker = ".", linestyle = ":", color = "C4",
                 label = "rung 0: measured $|\\phi|^2$ / $C_\\phi$")
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"$L$")
    axes[2].set_ylabel("diagnostics")
    axes[2].legend(fontsize = 9)

    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description = "Average per-realization phi cross correlations into N_L^eff")
    parser.add_argument("--noise_dir", type = str, required = True,
                        help = "the out_dir get_effective_phi_noise.sh wrote its jobs to")
    parser.add_argument("--out_name", type = str, default = MERGED_NAME)
    args = parser.parse_args()

    merged = merge_phi_noise(args.noise_dir)

    out_path = os.path.join(args.noise_dir, args.out_name)
    np.savez(out_path, **merged)
    plot_path = os.path.splitext(out_path)[0] + ".png"
    plot_phi_noise(merged, plot_path)

    cosmology = ", ".join(f"{name} {value:.6g}"
                          for name, value in zip(PARAM_ORDER, merged["params"]))
    print(f"\nmeasured at: {cosmology}")
    print(f"wrote {out_path}")
    print(f"wrote {plot_path}")
    print(f"\nUse it with:\n"
          f"    python -m cmb_lensing.fisher_forecast --spectra delensed "
          f"--nphi_source measured --phi_noise {out_path}")


if __name__ == "__main__":
    main()
