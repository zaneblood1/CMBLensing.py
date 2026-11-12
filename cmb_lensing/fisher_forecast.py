"""Gaussian Fisher forecast for the LCDM parameters on sample_lcdm.py's flat-sky box.

A central finite difference of per-mode covariance blocks on the SAME rfft grid the sampler
runs on (gen_ell_grid at the given nside / theta_pix), contracted with the zero-mean Gaussian
Fisher formula for a covariance that is block diagonal in Fourier space,

    F_ij = 1/2 * sum_blocks sum_k w_k * Tr[ C_k^-1 dC_k/dtheta_i  C_k^-1 dC_k/dtheta_j ]

with w_k = util.get_fourier_weights, the number of independent real degrees of freedom of
each rfft entry (2 in the bulk, 1 on the self-conjugate columns; they sum to nside**2 and are
the weights statistics.logpdf is built from). For a 1 x 1 block the trace is
dln(C_k)/dtheta_i * dln(C_k)/dtheta_j.

THE BLOCKS (per rfft mode k), by the data's polarization `pol`:

    "I"    [TT + N_TT]                                  the delensed field + instrument noise
    "P"    [EE + N_EE], [BB + N_BB]
    "IP"   [[TT + N_TT, TE], [TE, EE + N_EE]], [BB + N_BB]
    all    [C_phi + N_phi]                              the lensing potential measured to
                                                        within the reconstruction noise

THE FIELD BLOCKS ARE ALWAYS MEASURED. Their signal, at theta_0 and at theta_0 +/- h_i, is the
empirical per-mode covariance of the box's own lense_flow delensing, read from a merged
delensed_covariance.npz (cmb_lensing/delensed_covariance.py); the analytic noise is added
here. The steps h_i are the file's. There is no CAMB delensing: without the file the forecast
raises. The delensed BB is pure residual lensing at r = 0, so treating its modes as
independent Gaussian information is optimistic; `include_bb = False` leaves that block out.

THE PHI BLOCK. C_phi(theta) comes from direct CAMB at the stencil points. N_phi is built
once at theta_0 and held there (it is a property of the estimator, like the instrument
noise), from one of

    the box quadratic-estimate N^(0) (camb_spectra.qe_noise_matrix: load_sim's estimator
    without NPHI_FAC), with the unlensed spectra in its response and the unlensed (default)
    or lensed (`qe_filter`) spectra in its filter
    the `nphi` matrix of a merged map_joint_phi_noise.npz / sampler_phi_noise.npz
    (`phi_noise`); NaN (unmeasured) modes carry no phi information

Build the delensed covariance's Wiener filter from the same N_phi the forecast uses.

The blocks are added as independent, and the field-phi correlations are ignored. The residual
lensing left in the delensed field tracks C_phi(theta), so the field blocks count some of the
lensing information the phi block already has. None of this is the marginal Fisher of
p(d | theta) that sample_joint targets; the chains are the measurement of that.

Parameters with is_sampled[name] = False are held FIXED, not marginalized over. Every matrix
is laid out in constants.PARAM_ORDER.

Usage:
    python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance <delensed_covariance.npz>
    python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance <npz> --qe_filter lensed
    python -m cmb_lensing.fisher_forecast --noise 5 --delensed_covariance <npz> --phi_noise <map_joint_phi_noise.npz>
    python -m cmb_lensing.fisher_forecast --noise 5 --pol IP --delensed_covariance <npz> [--no_bb]

Writes into cmb_lensing/fisher_output/, tagged with forecast_file_suffix (pol, the N_phi and
the noise level, e.g. "_I_qe_unlensed_5_uk"):
    fisher_matrix<suffix>.png       F_ij
    covariance_matrix<suffix>.png   F^-1 and the marginalized sigmas
    correlation_matrix<suffix>.png  F^-1 normalized to unit diagonal
    fisher<suffix>.npz              all three plus the run configuration
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.constants import (PARAM_ORDER, GROUND_TRUTH, PARAM_SIGMA, CAMB_LMAX,
                                   DEFAULT_ELL_CUTOFF)
from cmb_lensing.camb_spectra import (camb_cls_at_params, covar_from_cl, instrument_matrices,
                                      qe_noise_matrix, QE_FILTERS, DEFAULT_QE_FILTER, POLS,
                                      DEFAULT_POL)
from cmb_lensing.delensed_covariance import load_merged_covariance
from cmb_lensing.matrix_plots import annotated_heatmap, plot_annotated_matrix, FIGURE_DPI

#the key a merged phi noise file holds its per-mode rfft2 noise matrix under
STORED_NPHI_KEY = "nphi"
#the file-name label of a forecast whose N_phi was read from a file
MEASURED_NPHI_LABEL = "measured_nphi"
#a Fisher matrix with a condition number above this is reported as near-degenerate
MAX_CONDITION_NUMBER = 1e12
#the E and B instrument noise over the temperature noise (noise_cls)
POLARIZATION_NOISE_FACTOR = 2


def output_dir():
    #a directory next to the package modules, as sample_lcdm.py's sample_lcdm_output
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisher_output")


def nphi_label(qe_filter = DEFAULT_QE_FILTER, phi_noise = None):
    """The N_phi's name in file names: "qe_<filter>", or MEASURED_NPHI_LABEL for a file."""
    return f"qe_{qe_filter}" if phi_noise is None else MEASURED_NPHI_LABEL


def forecast_file_suffix(pol, nphi_label, noise_level):
    """The tag of one forecast's output files, e.g. "_I_qe_unlensed_5_uk"."""
    return f"_{pol}_{nphi_label}_{noise_level:g}_uk"


# ── Covariance blocks ─────────────────────────────────────────────────────

def fourier_weights(nside):
    """w_k, the independent real DOF per rfft entry, on the (nside, nside//2+1) grid
    (get_fourier_weights indexes the half axis, i.e. the columns)."""
    return np.broadcast_to(np.real(np.asarray(get_fourier_weights((nside, nside // 2 + 1)))),
                           (nside, nside // 2 + 1))


def phi_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, l_cutoff,
                     pol = DEFAULT_POL, qe_filter = DEFAULT_QE_FILTER, phi_noise = None):
    """N_phi on the rfft grid: the box QE N^(0) with `qe_filter`, or the "nphi" matrix of
    the npz at `phi_noise` (NaN where unmeasured)."""
    if phi_noise is not None:
        return np.asarray(np.load(phi_noise)[STORED_NPHI_KEY], dtype = float)
    return np.asarray(qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                                      beam_fwhm, l_cutoff, pol = pol, qe_filter = qe_filter))


def scalar_block(matrix):
    """A per-mode scalar as a 1 x 1 block, (1, 1, nside, nside//2+1)."""
    return np.asarray(matrix)[None, None]


def field_blocks(delensed, noise, pol = DEFAULT_POL, include_bb = True):
    """The field covariance blocks at one stencil point, each (n, n, nside, nside//2+1).

    `delensed` maps a component ("TT", "TE", "EE", "BB") to its measured per-mode delensed
    covariance; `noise` is the temperature noise matrix (E and B carry twice it). See the
    module docstring for the blocks of each `pol`.
    """
    polarization_noise = POLARIZATION_NOISE_FACTOR * noise
    blocks = []
    if pol == "I":
        blocks.append(scalar_block(delensed["TT"] + noise))
    if pol == "P":
        blocks.append(scalar_block(delensed["EE"] + polarization_noise))
    if pol == "IP":
        blocks.append(np.array([[delensed["TT"] + noise, delensed["TE"]],
                                [delensed["TE"], delensed["EE"] + polarization_noise]]))
    if pol != "I" and include_bb:
        blocks.append(scalar_block(delensed["BB"] + polarization_noise))
    return blocks


def covariance_stencil(nside, theta_pix, noise_level, is_sampled, param_ground,
                       delensed_covariance, pol = DEFAULT_POL, qe_filter = DEFAULT_QE_FILTER,
                       phi_noise = None, include_bb = True, l_knee = 0, beam_fwhm = 0,
                       l_cutoff = DEFAULT_ELL_CUTOFF, camb_lmax = CAMB_LMAX):
    """The covariance blocks the forecast differences. Arguments as forecast().

    Returns (names, steps, blocks_fid, blocks_plus, blocks_minus): the sampled names in
    PARAM_ORDER, the file's steps h_i for them, the list of blocks at theta_0 (the field
    blocks, then the phi block) and that list at theta_0 +/- h_i for every sampled parameter.
    """
    if delensed_covariance is None:
        raise ValueError("the forecast needs delensed_covariance, a merged "
                         "delensed_covariance.npz: there is no analytic delensing fallback")
    measured = load_merged_covariance(delensed_covariance)
    names = [name for name in PARAM_ORDER if is_sampled.get(name, False)]
    #the file's own parameter list may be longer and in another order
    file_index = [measured["names"].index(name) for name in names]
    steps = [float(measured["steps"][index]) for index in file_index]

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls_fid = camb_cls_at_params(param_ground, camb_lmax = camb_lmax)
    noise = np.asarray(instrument_matrices(nside, pix_width, ell_grid, noise_level, l_knee,
                                           beam_fwhm, l_cutoff)[0])
    #N_phi is built once, at theta_0
    nphi = phi_noise_matrix(cls_fid, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                            l_cutoff, pol = pol, qe_filter = qe_filter, phi_noise = phi_noise)

    def blocks_at(params, delensed_stack):
        delensed = dict(zip(measured["components"], delensed_stack))
        cphi = np.asarray(covar_from_cl(nside, pix_width, ell_grid,
                                        camb_cls_at_params(params, camb_lmax = camb_lmax)["phi"]))
        return field_blocks(delensed, noise, pol, include_bb) + [scalar_block(cphi + nphi)]

    blocks_fid = blocks_at(param_ground, measured["delensed_fid"])
    blocks_plus = []
    blocks_minus = []
    for name, step, index in zip(names, steps, file_index):
        up = dict(param_ground)
        up[name] = param_ground[name] + step
        down = dict(param_ground)
        down[name] = param_ground[name] - step
        blocks_plus.append(blocks_at(up, measured["delensed_plus"][index]))
        blocks_minus.append(blocks_at(down, measured["delensed_minus"][index]))
    return names, steps, blocks_fid, blocks_plus, blocks_minus


def fisher_from_blocks(blocks_fid, blocks_plus, blocks_minus, steps, nside):
    """F_ij = 1/2 sum_blocks sum_k w_k Tr[C^-1 dC/dtheta_i C^-1 dC/dtheta_j].

    blocks_fid is a list of (n, n, nside, nside//2+1) blocks; blocks_plus[i] / blocks_minus[i]
    the same list at theta_0 +/- steps[i]. A mode whose centre block is not finite and
    positive definite carries no information: the zeroed origin, and a measured N_phi's
    unmeasured (NaN) modes.
    """
    weights = fourier_weights(nside)
    n_param = len(steps)
    fisher = np.zeros((n_param, n_param))
    for block_index, block_fid in enumerate(blocks_fid):
        #(nside, nside//2+1, n, n): one small matrix per mode
        centre = np.moveaxis(block_fid, (0, 1), (-2, -1))
        size = centre.shape[-1]
        with np.errstate(invalid = "ignore"):
            good = np.all(np.isfinite(centre), axis = (-2, -1)) & (centre[..., 0, 0] > 0)
            good &= np.linalg.det(np.where(good[..., None, None], centre, np.eye(size))) > 0
        inverse = np.linalg.inv(np.where(good[..., None, None], centre, np.eye(size)))

        #C^-1 dC/dtheta_i per mode, zero where the mode carries no information
        normalized_derivatives = []
        for i in range(n_param):
            difference = np.moveaxis(blocks_plus[i][block_index] - blocks_minus[i][block_index],
                                     (0, 1), (-2, -1))
            derivative = np.where(good[..., None, None], difference / (2 * steps[i]), 0.0)
            normalized_derivatives.append(inverse @ derivative)
        for i in range(n_param):
            for j in range(n_param):
                trace = np.einsum("...ab,...ba->...", normalized_derivatives[i],
                                  normalized_derivatives[j])
                fisher[i, j] += 0.5 * np.sum(weights * trace)
    return fisher


def forecast(nside, theta_pix, noise_level, is_sampled, param_ground, delensed_covariance,
             pol = DEFAULT_POL, qe_filter = DEFAULT_QE_FILTER, phi_noise = None,
             include_bb = True, l_knee = 0, beam_fwhm = 0, l_cutoff = DEFAULT_ELL_CUTOFF,
             camb_lmax = CAMB_LMAX):
    """Gaussian Fisher matrix for the sampled LCDM parameters on an nside x nside box.

    Args:
        nside, theta_pix, noise_level: the box (pixels per side, arcmin, uK-arcmin in
                      temperature; the polarization noise is sqrt(2) higher)
        is_sampled:   {param_name: bool}; False parameters are held FIXED
        param_ground: {param_name: float} fiducial point, all five
        delensed_covariance: a merged delensed_covariance.npz measured at this box, noise
                      level and `pol` (required): the field blocks' delensed signal at
                      theta_0 and theta_0 +/- h, and the steps h
        pol:          "I" (temperature), "P" (polarization) or "IP" (both): the data
        qe_filter:    "unlensed" (default) or "lensed": the spectra in the filter of the
                      quadratic-estimate N0 used as the phi noise. Its response always uses
                      the unlensed spectra. The estimator is load_sim's: TT for "I", EB for
                      "P" and "IP"
        phi_noise:    a merged map_joint_phi_noise.npz / sampler_phi_noise.npz whose `nphi`
                      replaces the QE N0
        include_bb:   for "P" / "IP", whether the delensed BB block is counted
        l_knee, beam_fwhm: 1/f knee (0 = white, as every sampler entry point) and beam FWHM
        camb_lmax:    how far CAMB is run (default CAMB_LMAX, the simulations' support)

    Returns:
        (fisher, names) - the n_sampled x n_sampled matrix and the sampled names, both in
        constants.PARAM_ORDER.

    Raises:
        ValueError: if delensed_covariance is None.
    """
    names, steps, blocks_fid, blocks_plus, blocks_minus = covariance_stencil(
        nside, theta_pix, noise_level, is_sampled, param_ground, delensed_covariance,
        pol = pol, qe_filter = qe_filter, phi_noise = phi_noise, include_bb = include_bb,
        l_knee = l_knee, beam_fwhm = beam_fwhm, l_cutoff = l_cutoff, camb_lmax = camb_lmax)
    return fisher_from_blocks(blocks_fid, blocks_plus, blocks_minus, steps, nside), names


# ── Post-processing ───────────────────────────────────────────────────────

def covariance_from_fisher(fisher, names):
    """F^-1, with a warning when F is close to singular."""
    eigenvalues = np.linalg.eigvalsh(fisher)
    condition = eigenvalues[-1] / eigenvalues[0]
    if condition > MAX_CONDITION_NUMBER or eigenvalues[0] <= 0:
        print(f"WARNING: Fisher eigenvalues {eigenvalues} (condition number {condition:.3g}): "
              f"at least one direction over {names} is near-degenerate, so the inverse is "
              f"unreliable")
    return np.linalg.inv(fisher)


def correlation_from_covariance(covariance):
    """A covariance normalized to unit diagonal."""
    sigmas = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigmas, sigmas)


def report_sigmas(covariance, names, label):
    """Print the marginalized 1-sigma errors."""
    sigmas = np.sqrt(np.diag(covariance))
    print(f"\nForecast 1-sigma errors [{label}]:")
    print(f"  {'parameter':<14s} {'sigma':>12s} {'sigma/PARAM_SIGMA':>20s}")
    for name, sigma in zip(names, sigmas):
        print(f"  {name:<14s} {sigma:>12.4g} {sigma / PARAM_SIGMA[name]:>20.3g}")


def plot_fisher_matrix(fisher, names, path, subtitle = ""):
    figure, axis = plt.subplots(figsize = (7.5, 6.5))
    annotated_heatmap(axis, fisher, names, f"Fisher matrix $F_{{ij}}$\n{subtitle}",
                      r"$F_{ij}\,/\,\sqrt{F_{ii}F_{jj}}$")
    figure.tight_layout()
    figure.savefig(path, dpi = FIGURE_DPI)
    plt.close(figure)


def plot_covariance_matrix(covariance, names, path, subtitle = ""):
    """The covariance heat map next to the marginalized sigmas in units of PARAM_SIGMA."""
    sigmas = np.sqrt(np.diag(covariance))
    ratios = np.array([sigma / PARAM_SIGMA[name] for name, sigma in zip(names, sigmas)])

    figure, (left, right) = plt.subplots(
        1, 2, figsize = (12.5, 6.0), gridspec_kw = {"width_ratios": [1.35, 1]})
    annotated_heatmap(left, covariance, names, f"Covariance $F^{{-1}}$\n{subtitle}",
                      "correlation coefficient")

    positions = np.arange(len(names))
    right.barh(positions, ratios, color = "#4C72B0")
    right.axvline(1.0, color = "k", linestyle = "--", linewidth = 1)
    right.set_yticks(positions)
    right.set_yticklabels(names)
    right.invert_yaxis()
    right.set_xscale("log")
    #the default log locator crowds this narrow range with overlapping minor labels
    right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    right.xaxis.set_minor_formatter(ticker.NullFormatter())
    right.set_xlim(0.5 * float(np.min(ratios)), 3.0 * float(np.max(ratios)))
    right.set_xlabel(r"forecast $\sigma$ / PARAM_SIGMA")
    right.set_title("Marginalized errors\n(dashed = PARAM_SIGMA; < 1 means informative)")
    for position, ratio, sigma in zip(positions, ratios, sigmas):
        right.text(ratio, position, f"  {sigma:.3g}", va = "center", fontsize = 8)

    figure.tight_layout()
    figure.savefig(path, dpi = FIGURE_DPI)
    plt.close(figure)


def save_outputs(fisher, covariance, names, suffix, subtitle, config):
    """The three figures and the npz of one run into output_dir(), tagged with `suffix`.
    `config` is stored alongside the matrices so a saved forecast can be traced to its run."""
    directory = output_dir()
    os.makedirs(directory, exist_ok = True)
    correlation = correlation_from_covariance(covariance)
    plot_fisher_matrix(fisher, names, os.path.join(directory, f"fisher_matrix{suffix}.png"),
                       subtitle)
    plot_covariance_matrix(covariance, names,
                           os.path.join(directory, f"covariance_matrix{suffix}.png"), subtitle)
    plot_annotated_matrix(correlation, names, f"Forecast Correlation Matrix\n{subtitle}",
                          os.path.join(directory, f"correlation_matrix{suffix}.png"),
                          norm = matplotlib.colors.Normalize(vmin = -1, vmax = 1))
    np.savez(os.path.join(directory, f"fisher{suffix}.npz"),
             fisher = fisher, covariance = covariance, correlation = correlation,
             names = np.array(names), sigmas = np.sqrt(np.diag(covariance)), **config)
    print(f"\nWrote fisher_matrix / covariance_matrix / correlation_matrix{suffix}.png and "
          f"fisher{suffix}.npz to {directory}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description = "Gaussian Fisher forecast for the LCDM parameters on the sample_lcdm.py "
                      "flat-sky box: measured delensed field blocks + a phi block")
    parser.add_argument("--delensed_covariance", type = str, default = None,
                        help = "a merged delensed_covariance.npz at this box, noise level "
                               "and pol (required): the field blocks and the steps")
    parser.add_argument("--nside", type = int, default = 128)
    parser.add_argument("--theta_pix", type = float, default = 2.5,
                        help = "pixel width in arcmin")
    parser.add_argument("--noise", type = float, default = 2.5,
                        help = "white temperature noise level in uK-arcmin")
    parser.add_argument("--pol", choices = POLS, default = DEFAULT_POL,
                        help = "the data: I (temperature), P (polarization) or IP (both)")
    parser.add_argument("--qe_filter", choices = QE_FILTERS, default = DEFAULT_QE_FILTER,
                        help = "the spectra in the filter of the QE N0 used as the phi noise "
                               "(its response always uses the unlensed spectra)")
    parser.add_argument("--phi_noise", type = str, default = None,
                        help = "a merged map_joint_phi_noise.npz / sampler_phi_noise.npz "
                               "whose `nphi` replaces the QE N0")
    parser.add_argument("--no_bb", action = "store_true",
                        help = "for P / IP: leave the delensed BB block out")
    parser.add_argument("--l_knee", type = float, default = 0.0,
                        help = "1/f knee; 0 (default) matches every sampler entry point")
    parser.add_argument("--beam_fwhm", type = float, default = 0.0)
    parser.add_argument("--params", nargs = "*", default = ["logA", "omch2", "theta_MC_100"],
                        choices = PARAM_ORDER, help = "the sampled parameters")
    parser.add_argument("--camb_lmax", type = int, default = CAMB_LMAX,
                        help = "how far to run CAMB")
    args = parser.parse_args()
    if args.delensed_covariance is None:
        parser.error("--delensed_covariance is required")

    is_sampled = {name: name in args.params for name in PARAM_ORDER}
    include_bb = not args.no_bb
    fisher, names = forecast(args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
                             args.delensed_covariance, pol = args.pol,
                             qe_filter = args.qe_filter, phi_noise = args.phi_noise,
                             include_bb = include_bb, l_knee = args.l_knee,
                             beam_fwhm = args.beam_fwhm, camb_lmax = args.camb_lmax)
    covariance = covariance_from_fisher(fisher, names)
    label = nphi_label(args.qe_filter, args.phi_noise)
    title = f"pol {args.pol}, N_phi {label}"
    report_sigmas(covariance, names, title)

    subtitle = (f"{title}  |  nside {args.nside}, {args.theta_pix:g}', "
                f"{args.noise:g} uK-arcmin")
    config = dict(nside = args.nside, theta_pix = args.theta_pix, noise_level = args.noise,
                  l_knee = args.l_knee, beam_fwhm = args.beam_fwhm, pol = args.pol,
                  include_bb = include_bb, qe_filter = args.qe_filter,
                  phi_noise = args.phi_noise or "",
                  delensed_covariance = args.delensed_covariance, camb_lmax = args.camb_lmax)
    save_outputs(fisher, covariance, names,
                 forecast_file_suffix(args.pol, label, args.noise), subtitle, config)


if __name__ == "__main__":
    main()
