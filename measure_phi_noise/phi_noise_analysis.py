import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.constants import GROUND_TRUTH, DEFAULT_ELL_CUTOFF
from cmb_lensing.fisher_forecast import (fourier_weights, camb_cls_at_params, iterative_delensing,
                                         qe_noise_matrix, prior_covariances)
from cmb_lensing.delensed_covariance import band_edges

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_ROOT = os.path.join(HERE, "phi_noise_plots")

#(colour, marker) of each measured N_phi source's series
SOURCE_STYLE = {"map_joint": ("C2", "^"),
                "sample_joint": ("C9", "D")}
#the measured N_phi sources' names in the plots
SOURCE_NAMES = {"map_joint": "Joint MAP",
                "sample_joint": "Gibbs Sampler"}
#font sizes
SPECTRA_TITLE_FONTSIZE = 15
SPECTRA_LABEL_FONTSIZE = 15
SPECTRA_LEGEND_FONTSIZE = 11

def plot_dir(noise):
    """phi_noise_plots/noise_level_<noise>uk/ (created), next to this script."""
    path = os.path.join(PLOT_ROOT, f"noise_level_{noise}uk")
    os.makedirs(path, exist_ok = True)
    return path

def band_index(ell_grid, edges, use):
    """The |L| band of every rfft-grid mode, -1 for a mode outside the bands or not in `use`."""
    ells = np.asarray(ell_grid)
    index = np.digitize(ells, edges) - 1
    return np.where(use & (index >= 0) & (index < len(edges) - 1), index, -1)

def band_median(matrix, index, n_band):
    """Median over each band's finite modes (NaN for an empty band), each rfft entry counted
    w_k times (2 in the bulk, 1 on the self-conjugate columns), i.e. the median over the
    full-plane modes."""
    dof = np.asarray(fourier_weights(matrix.shape[0])).astype(int)
    values = np.full(n_band, np.nan)
    for b in range(n_band):
        inside = (index == b) & np.isfinite(matrix)
        if inside.any():
            values[b] = np.median(np.repeat(matrix[inside], dof[inside]))
    return values

def band_median_jackknife(nphi_leave_one_out, nphi_qe, use, index, n_band):
    """The delete-one-realization jackknife error of the band median of N_k / N_QE,k:

        sigma^2_b = (M - 1) / M * sum_j (median_b^(j) - mean_j median_b^(j))^2

    with median_b^(j) the band median of the N_phi that leaves realization j out. A band left
    with no modes in any leave-one-out N_phi comes back NaN. MEASURED on Gaussian simulations
    at nside 128 / 2.5', 100 realizations: this is 2.3-2.9x the true scatter of the band median
    (a median is not a smooth statistic), so it is a deliberate over-estimate."""
    num_realizations = len(nphi_leave_one_out)
    medians = []
    for nphi in nphi_leave_one_out:
        measured = use & np.isfinite(nphi) & (nphi > 0)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            ratio = np.where(measured, nphi / nphi_qe, np.nan)
        medians.append(band_median(ratio, index, n_band))
    medians = np.array(medians)
    spread = medians - np.mean(medians, axis = 0)
    return np.sqrt((num_realizations - 1) / num_realizations * np.sum(spread**2, axis = 0))

def qe_noise_matrices(params, nside, theta_pix, noise_level, l_knee, beam_fwhm):
    """(C_phi, [(label, color, linestyle, N_phi^QE matrix)]) on the rfft grid from direct CAMB,
    all with the unlensed response and undivided by NPHI_FAC: the box QE with the lensed C_TT
    in its filter (the reference every ratio is taken against, first), the iterated-delensing
    fixed point (fisher_forecast's qe_delensed filter) and the unlensed C_TT in the filter
    (qe_unlensed)."""
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = camb_cls_at_params(params)
    delensed_tt = iterative_delensing(params, cls, nside, pix_width, ell_grid, noise_level,
                                      l_knee, beam_fwhm, DEFAULT_ELL_CUTOFF, verbose = False,
                                      qe_response = "unlensed")[2]
    matrices = []
    for label, color, linestyle, filter_tt in (
            ("QE: lensed filter, unlensed response", "C3", "--", cls["total_TT"]),
            ("QE: delensed filter, unlensed response", "C4", "-.", delensed_tt),
            ("QE: unlensed filter, unlensed response", "C5", "-.", cls["scalar_TT"])):
        matrices.append((label, color, linestyle, np.asarray(qe_noise_matrix(
            cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, DEFAULT_ELL_CUTOFF,
            filter_tt = filter_tt, qe_response = "unlensed"))))
    cphi = np.asarray(prior_covariances(params, nside, pix_width, ell_grid)[1])
    return cphi, matrices

def plot_phi_noise_vs_qe(nside, theta_pix, noise, phi_noise_files, out_dir):
    """Per-mode phi noise spectra against the box QE matrix at this noise level:
    phi_noise_spectra_<noise>uk.png and phi_noise_spectra_over_qe_<noise>uk.png in out_dir.

    Draws the `nphi` (C_phi (1/r^2 - 1), r^2 = <phi_bar phi*>^2 / (<|phi_bar|^2><|phi|^2>))
    of every merged map_joint_phi_noise.npz / sampler_phi_noise.npz in `phi_noise_files`,
    on the box at GROUND_TRUTH (white noise, no beam). phi_noise_spectra: band medians, with
    CAMB's C_L^phiphi and the QE matrices of qe_noise_matrices. phi_noise_spectra_over_qe: the
    band median of each series' per-mode ratio to the lensed-filter QE, the measured series
    shaded +/- 1 jack-knife (band_median_jackknife, from the files' `nphi_leave_one_out`).
    """
    ell_grid = np.asarray(gen_ell_grid(nside, theta_pix)[0])
    use = ell_grid > 0
    edges = band_edges(ell_grid)
    n_band = len(edges) - 1
    index = band_index(ell_grid, edges, use)
    centres = 0.5 * (edges[1:] + edges[:-1])
    cphi, qe_matrices = qe_noise_matrices(GROUND_TRUTH, nside, theta_pix, noise, 0.0, 0.0)
    nphi_qe = qe_matrices[0][3]
    series = []
    for nphi_source, path in phi_noise_files.items():
        merged = np.load(path)
        colour, marker = SOURCE_STYLE[nphi_source]
        short_name = f"{SOURCE_NAMES[nphi_source]} Effective $N_L^{{\\phi\\phi}}$"
        name = (f"{short_name} = $C_L^{{\\phi\\phi}}(1/r_L^2 - 1)$ "
                f"({int(merged['n_realizations'])} realizations)")
        series.append((name, short_name, colour, marker, np.asarray(merged["nphi"]),
                       np.asarray(merged["nphi_leave_one_out"])))

    top_figure, top = plt.subplots(figsize = (10, 6.5))
    bottom_figure, bottom = plt.subplots(figsize = (10, 5))
    order = np.argsort(ell_grid[use])
    top.loglog(ell_grid[use][order], cphi[use][order], color = "0.35", linewidth = 1.5,
               label = r"CAMB $C_L^{\phi\phi}$")
    bottom.axhline(1.0, color = "C3", linestyle = "--", linewidth = 1)
    for position, (name, color, linestyle, matrix) in enumerate(qe_matrices):
        measured = use & np.isfinite(matrix) & (matrix > 0)
        band = band_median(np.where(measured, matrix, np.nan), index, n_band)
        good = np.isfinite(band)
        top.loglog(centres[good], band[good], color = color, linestyle = linestyle,
                   linewidth = 1.3 if position else 1.5, label = name)
        if position:
            with np.errstate(divide = "ignore", invalid = "ignore"):
                ratio = np.where(measured, matrix / nphi_qe, np.nan)
            median = band_median(ratio, index, n_band)
            good = np.isfinite(median)
            bottom.plot(centres[good], median[good], color = color, linestyle = linestyle,
                        linewidth = 1.3, label = name)

    for name, short_name, color, marker, matrix, leave_one_out in series:
        measured = use & np.isfinite(matrix) & (matrix > 0)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            ratio = np.where(measured, matrix / nphi_qe, np.nan)
        band = band_median(np.where(measured, matrix, np.nan), index, n_band)
        good = np.isfinite(band)
        top.loglog(centres[good], band[good], color = color, marker = marker, markersize = 3,
                   linewidth = 1, label = name)
        median = band_median(ratio, index, n_band)
        median_error = band_median_jackknife(leave_one_out, nphi_qe, use, index, n_band)
        good = np.isfinite(median) & np.isfinite(median_error)
        bottom.fill_between(centres[good], median[good] - median_error[good],
                            median[good] + median_error[good], color = color, alpha = 0.25,
                            linewidth = 0)
        bottom.plot(centres[good], median[good], color = color, marker = marker, markersize = 3,
                    linewidth = 1.2, label = f"{short_name} $\\pm$ 1 jack-knife")

    title = (f"Phi Noise Spectra Comparison | nside = {nside}, "
             f"$\\theta_{{\\rm pix}}$ = {theta_pix:g}', "
             f"noise level = {noise:g} $\\mu$K-arcmin")

    top.set_xlabel(r"$L$", fontsize = SPECTRA_LABEL_FONTSIZE)
    top.set_ylabel(r"$N_L^{\phi\phi}$ Estimates (Band Median)", fontsize = SPECTRA_LABEL_FONTSIZE)
    top.legend(fontsize = SPECTRA_LEGEND_FONTSIZE)
    top.set_title(title, fontsize = SPECTRA_TITLE_FONTSIZE)
    top_figure.tight_layout()
    path = os.path.join(out_dir, f"phi_noise_spectra_{noise}uk.png")
    top_figure.savefig(path, dpi = 150)
    plt.close(top_figure)
    print(f"wrote {path}")

    bottom.set_xscale("log")
    bottom.set_ylim(0.0, 2.0)
    bottom.set_xlabel(r"$L$", fontsize = SPECTRA_LABEL_FONTSIZE)
    bottom.set_ylabel("Band median of ratio relative to QE\n (lensed filter, unlensed response)",
                      fontsize = 12)
    bottom.legend(fontsize = SPECTRA_LEGEND_FONTSIZE, loc = "upper left")
    bottom.set_title(title, fontsize = SPECTRA_TITLE_FONTSIZE)
    bottom_figure.tight_layout()
    path = os.path.join(out_dir, f"phi_noise_spectra_over_qe_{noise}uk.png")
    bottom_figure.savefig(path, dpi = 150)
    plt.close(bottom_figure)
    print(f"wrote {path}")

if __name__ == "__main__":

    #the box and noise level the merged phi noise files were measured at
    nside = 128
    theta_pix = 2.5
    noise = 10

    #merged map_joint_phi_noise.npz / sampler_phi_noise.npz files at THIS box and noise level.
    #Edit to wherever the merged files are; drop an entry to skip it
    phi_noise_files = {
        "map_joint": os.path.join(HERE, "get_map_joint_phi_noise", "map_joint_phi_noise",
                                  f"map_joint_phi_noise_output_{noise}uk", "map_joint_phi_noise.npz"),
        "sample_joint": os.path.join(HERE, "get_sampler_phi_noise", "sampler_phi_noise_data",
                                     f"sampler_phi_noise_output_{noise}uk", "sampler_phi_noise.npz"),
    }

    plot_phi_noise_vs_qe(nside, theta_pix, noise, phi_noise_files, plot_dir(noise))
