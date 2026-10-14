"""Per-mode phi noise spectra against the box QE matrix, one png per noise level.

The QE matrix is NOT read from any npz: it is rebuilt here from direct CAMB spectra with
simulate.scalar_quadratic_estimate (fisher_forecast.qe_noise_matrix) - unlensed C_TT response,
the fully lensed C_TT in the filter's sigma = C_TT^lensed + C_n, no iterative delensing, no
NPHI_FAC. CAMB's C_phi is rebuilt the same way.

Series, all per rfft mode:
    map_joint N_eff    C_phi (1 / r^2 - 1), r the cross correlation of phi_MAP with the true phi
                       (sampler_noise_estimate.npz: phi_map_correlation_noise). A file merged
                       before that key existed only holds Var(phi_MAP | phi) / rho^2
                       (phi_map_noise), which is drawn instead and labelled as such.
    sampler bound      1 / <F_phi>, variance of the score  (sampler_noise_estimate.npz: nphi_raw)
    b_only             C_phi (1 / r^2 - 1), r^2 = <phi_bar phi*> / <|phi|^2>
                       (sampler_effective_noise.npz: nphi_b_only_raw; only where the directory exists)

Top panel: information-weighted band means with CAMB's C_L^phiphi. Bottom panel: every mode's
N_k / N_QE,k as a faint point, with the band median on top.

    python plot_phi_noise_vs_qe.py [--out_dir <dir>] [--burn_in <b>]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmb_lensing.simulate import gen_ell_grid
from cmb_lensing.sample_lcdm import PARAM_ORDER
from cmb_lensing.fisher_forecast import (camb_cls_at_params, qe_noise_matrix, prior_covariances,
                                         iterative_delensing)
from cmb_lensing.sampler_noise_estimate import used_modes, band_index, band_harmonic

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE_LEVELS = (("2dot5uk", "2.5"), ("5uk", "5"), ("10uk", "10"))


def band_median(matrix, index, n_band):
    """Median over each band's finite modes (NaN for an empty band)."""
    values = np.full(n_band, np.nan)
    for b in range(n_band):
        inside = matrix[(index == b) & np.isfinite(matrix)]
        if inside.size:
            values[b] = np.median(inside)
    return values


def qe_matrix_from_first_principles(params, nside, theta_pix, noise_level, l_knee, beam_fwhm):
    """(N_phi^QE, C_phi) on the rfft grid from direct CAMB: the box QE normalization with the
    unlensed response and the lensed spectrum in the filter, undivided by NPHI_FAC."""
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = camb_cls_at_params(params)
    nphi = qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                           10_000, filter_tt = cls["total_TT"], qe_response = "unlensed")
    cphi = prior_covariances(params, nside, pix_width, ell_grid)[1]
    return np.asarray(nphi), np.asarray(cphi.scalar_matrix if hasattr(cphi, "scalar_matrix")
                                        else cphi)


def qe_variants(params, nside, theta_pix, noise_level, l_knee, beam_fwhm):
    """[(label, color, N_phi matrix)] of the other box QE matrices drawn azimuthally averaged:
    the iterated-delensing fixed point (unlensed response, the converged delensed C_TT in the
    filter) and the one-shot matrix with the UNLENSED C_TT in the filter instead of the
    lensed one."""
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = camb_cls_at_params(params)
    #iterative_delensing + qe_noise_matrix directly, NOT frozen_reconstruction: the working
    #tree's frozen_reconstruction builds its final matrix with the unlensed C_TT in the filter
    #rather than the converged delensed one, which would make this curve the next one
    delensed_tt = iterative_delensing(params, cls, nside, pix_width, ell_grid, noise_level,
                                      l_knee, beam_fwhm, 10_000, "covariance", verbose = False,
                                      qe_response = "unlensed")[2]
    iterated = qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                               10_000, filter_tt = delensed_tt, qe_response = "unlensed")
    unlensed = qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                               10_000, filter_tt = cls["scalar_TT"], qe_response = "unlensed")
    return [("QE, iterative delensing (unlensed response)", "C4", np.asarray(iterated)),
            ("QE, unlensed $C^{TT}$ in the filter (unlensed response)", "C5",
             np.asarray(unlensed))]


def plot_noise_level(tag, label, out_dir, burn_in = None):
    score_path = os.path.join(HERE, f"sampler_noise_estimate_output_{tag}_wf_map_joint_data",
                              "sampler_noise_estimate.npz")
    if not os.path.exists(score_path):
        print(f"{label} uK: no {score_path}, skipped")
        return
    score = np.load(score_path)
    nside, theta_pix = int(score["nside"]), float(score["theta_pix"])
    ell_grid = np.asarray(gen_ell_grid(nside, theta_pix)[0])
    use = used_modes(ell_grid)
    edges = np.asarray(score["band_edges"])
    n_band = len(edges) - 1
    index = band_index(ell_grid, edges, use)
    centres = np.asarray(score["band_ells"])
    params = {name: float(score["params"][i]) for i, name in enumerate(PARAM_ORDER)}
    nphi_qe, cphi = qe_matrix_from_first_principles(
        params, nside, theta_pix, float(score["noise_level"]), float(score["l_knee"]),
        float(score["beam_fwhm"]))

    if "phi_map_correlation_noise" in score:
        map_name = "map_joint $N_{\\rm eff}$ = $C_\\phi(1/r^2-1)$"
        map_matrix = np.asarray(score["phi_map_correlation_noise"])
    else:
        print(f"{label} uK: {score_path} has no phi_map_correlation_noise (re-merge the job "
              f"files to get it); drawing Var / rho^2")
        map_name = "map_joint $N_{\\rm eff}$ = Var$(\\phi_{\\rm MAP}|\\phi)/\\rho^2$"
        map_matrix = np.asarray(score["phi_map_noise"])
    series = [(f"{map_name} ({int(score['phi_map_realizations'])} phi)", "C2", "s", map_matrix),
              (f"sampler bound $1/\\langle F_\\phi\\rangle$ ({int(score['n_realizations'])} phi, "
               f"{int(score['n_draws_total'])} draws)", "C0", "o", np.asarray(score["nphi_raw"]))]

    suffix = "" if burn_in is None else f"_burn_in_{burn_in}"
    chain_path = os.path.join(HERE, f"sampler_effective_noise_output_mean_chains_{tag}",
                              f"sampler_effective_noise{suffix}.npz")
    if os.path.exists(chain_path):
        chain = np.load(chain_path)
        for key in ("nside", "theta_pix", "noise_level"):
            if float(chain[key]) != float(score[key]):
                raise ValueError(f"{chain_path}: {key} differs from {score_path}")
        series.append(("b_only $\\langle\\bar\\phi\\,\\phi^*\\rangle/\\langle|\\phi|^2\\rangle$ "
                       f"({int(chain['n_realizations'])} chains, burn-in {int(chain['burn_in'])})",
                       "C1", "^", np.asarray(chain["nphi_b_only_raw"])))
    else:
        print(f"{label} uK: no {chain_path}, b_only left out")

    figure, (top, bottom) = plt.subplots(2, 1, figsize = (10, 10), sharex = True,
                                         gridspec_kw = dict(height_ratios = [3, 2]))
    order = np.argsort(ell_grid[use])
    top.loglog(ell_grid[use][order], cphi[use][order], color = "0.35", linewidth = 1.5,
               label = r"CAMB $C_L^{\phi\phi}$")
    band_qe = band_harmonic(np.where(use, nphi_qe, np.nan), index, n_band)[0]
    keep = np.isfinite(band_qe)
    top.loglog(centres[keep], band_qe[keep], color = "C3", linestyle = "--", linewidth = 1.5,
               label = "QE (box matrix: unlensed response, lensed filter, no NPHI_FAC)")
    bottom.axhline(1.0, color = "C3", linestyle = "--", linewidth = 1)
    for name, color, matrix in qe_variants(params, nside, theta_pix, float(score["noise_level"]),
                                           float(score["l_knee"]), float(score["beam_fwhm"])):
        band = band_harmonic(np.where(use, matrix, np.nan), index, n_band)[0]
        good = keep & np.isfinite(band)
        top.loglog(centres[good], band[good], color = color, linestyle = "-.", linewidth = 1.3,
                   label = name)
        bottom.plot(centres[good], band[good] / band_qe[good], color = color, linestyle = "-.",
                    linewidth = 1.3, label = f"{name} / QE (ratio of band means)")
        print(f"{label} uK  {name[:40]:40s} band ratio to QE: min {np.min(band[good] / band_qe[good]):.3f} "
              f"median {np.median(band[good] / band_qe[good]):.3f} max {np.max(band[good] / band_qe[good]):.3f}")

    for name, color, marker, matrix in series:
        measured = use & np.isfinite(matrix) & (matrix > 0)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            ratio = np.where(measured, matrix / nphi_qe, np.nan)
        band = band_harmonic(np.where(measured, matrix, np.nan), index, n_band)[0]
        good = np.isfinite(band)
        top.loglog(centres[good], band[good], color = color, marker = marker, markersize = 3,
                   linewidth = 1, label = name)
        bottom.plot(ell_grid[measured], ratio[measured], linestyle = "none", marker = ".",
                    markersize = 1.5, color = color, alpha = 0.12, rasterized = True)
        median = band_median(ratio, index, n_band)
        good = np.isfinite(median)
        bottom.plot(centres[good], median[good], color = color, marker = marker, markersize = 3,
                    linewidth = 1.2,
                    label = f"{name.split(' (')[0].split(' = ')[0]} / QE (band median)")
        print(f"{label} uK  {name.split(' (')[0][:24]:24s} modes {int(measured.sum()):5d}  "
              f"median N/N_QE {np.nanmedian(ratio):.3f}")

    top.set_ylabel(r"$N_L$ (covar units; information-weighted band mean of the per-mode matrix)")
    top.legend(fontsize = 8)
    bottom.set_xscale("log")
    bottom.set_ylim(0.0, 2.0)
    bottom.set_xlabel(r"$L$")
    bottom.set_ylabel(r"per-mode $N_k / N_k^{\rm QE}$")
    bottom.legend(fontsize = 7, loc = "upper left", ncol = 2)
    figure.suptitle(f"phi noise per mode relative to the QE | nside {nside}, {theta_pix:g}', "
                    f"{label} uK-arcmin", fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"phi_noise_vs_qe_per_mode_{tag}.png")
    figure.savefig(path, dpi = 150)
    plt.close(figure)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    parser.add_argument("--out_dir", default = HERE)
    parser.add_argument("--burn_in", type = int, default = None,
                        help = "read sampler_effective_noise_burn_in_<b>.npz instead of the "
                               "un-suffixed merge")
    args = parser.parse_args()
    for tag, label in NOISE_LEVELS:
        plot_noise_level(tag, label, args.out_dir, args.burn_in)


if __name__ == "__main__":
    main()
