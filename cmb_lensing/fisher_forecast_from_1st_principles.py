"""A first-principles Fisher forecast, "1st_principles":

    F_ij = sum_l (2l+1)/2 f_sky  dC_l/dtheta_i (C_l + N_l)^-2 dC_l/dtheta_j

summed over TWO independent blocks - the temperature field and the lensing potential:

    field      C_l = C_l^TT (lensed or unlensed CAMB spectrum),  N_l = the map noise
    potential  C_L = C_L^phiphi,                                  N_L = the quadratic-estimator
                                                                        reconstruction noise

Everything is a 1D function of the multipole, computed directly from CAMB spectra and the
experiment's numbers. Nothing here touches the rfft grid: the box enters only through
f_sky = (nside * pix_width)^2 / (4 pi), the sky area whose modes the (2l + 1) sum counts.

The one input the other forecasts could only get from the sampler's 2D machinery is the
reconstruction noise. scalar_quadratic_estimate builds it with FFT convolutions on the
square periodic box, and bringing that matrix onto a 1D ell axis meant azimuthally averaging
away its (genuine) square-box anisotropy. Here it is the flat-sky N^(0) of Hu & Okamoto
(2002) evaluated as a 2D quadrature in polar coordinates - fisher_forecast.qe_noise_spectrum,
which the delensed spectra mode of the block forecasts can also use (nphi_source =
"hu_okamoto") - so it is isotropic by construction and needs no matrix and no decoding. Its
response is the lensed temperature-gradient spectrum rather than Hu & Okamoto's unlensed one,
the same choice every N_phi in this package's forecasts makes; see
fisher_forecast.gradient_cls_at_params.

Usage:
    python -m cmb_lensing.fisher_forecast_from_1st_principles
    python -m cmb_lensing.fisher_forecast_from_1st_principles --spectra unlensed
    python -m cmb_lensing.fisher_forecast_from_1st_principles --ell_min 30 --ell_max 3000 --delta_ell 20
    python -m cmb_lensing.fisher_forecast_from_1st_principles --ell_max 4320   #to the box's Nyquist, extrapolated like the sampler

Writes fisher_matrix_1st_principles.png, covariance_matrix_1st_principles.png,
correlation_matrix_1st_principles.png, qe_noise_1st_principles.png and
fisher_1st_principles.npz into cmb_lensing/fisher_output/.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.simulate import noise_cls
from cmb_lensing.constants import DEFAULT_MAX_ELL
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, CAMB_LMAX
from cmb_lensing.fisher_forecast import (camb_cls_at_params, cls_with_qe_response,
                                         qe_response_cl, DEFAULT_QE_RESPONSE,
                                         add_qe_response_argument,
                                         sampled_names_and_steps,
                                         covariance_from_fisher, step_stability,
                                         report_sigmas, save_outputs, output_dir,
                                         add_box_arguments, sampled_from_args, run_config,
                                         interpolate_spectrum, qe_noise_spectrum)
from cmb_lensing.fisher_forecast_full_sky import sky_fraction


METHOD = "1st_principles"
SUFFIX = "_1st_principles"
LABEL = (r"1st principles:  $F_{ij} = \sum_\ell \frac{2\ell+1}{2} f_{sky}\,"
         r"\partial_i C_\ell\,(C_\ell + N_\ell)^{-2}\,\partial_j C_\ell$  over TT and $\phi\phi$")

#which CAMB temperature spectrum plays C_l^TT in the field block
FIELD_SPECTRA = ("lensed", "unlensed")

#CAMB's multipole support: camb_cls_at_params puts every spectrum on 2 .. CAMB_LMAX - 1.
#ell_max may exceed CAMB_ELL_MAX, see multipole_axis
CAMB_ELL_MIN = 2
CAMB_ELL_MAX = CAMB_LMAX - 1

# ── Spectra on a 1D multipole axis ────────────────────────────────────────

def camb_ell_axis(cls):
    """The integer multipoles every CAMB spectrum in `cls` lives on."""
    return np.arange(CAMB_ELL_MIN, CAMB_ELL_MIN + cls["total_TT"].shape[0], dtype = np.float64)


def map_noise_spectrum(ells, noise_level, l_knee, beam_fwhm):
    """N_l^TT: the beam-DECONVOLVED map noise, so the signal it pairs with is the bare C_l.

    noise_cls already divides the white + 1/f noise by the beam transfer function B_l^2,
    which is the convention in which the observed spectrum is C_l + N_l with C_l the sky
    spectrum itself. That is the convention this whole module uses.
    """
    lmax_prime = min(DEFAULT_MAX_ELL, CAMB_LMAX)
    n_tt, _, _, _ = noise_cls(lmax_prime, noise_level, beam_fwhm = beam_fwhm, l_knee = l_knee)
    noise_ells = np.arange(CAMB_ELL_MIN, lmax_prime, dtype = np.float64)
    return interpolate_spectrum(ells, noise_ells, np.asarray(n_tt))


# ── The blocks and their finite-difference derivatives ────────────────────

def signal_spectra(params, ells, spectra):
    """{"TT": C_l^TT, "PP": C_L^phiphi} from one (memoized) CAMB run at `params`."""
    cls = camb_cls_at_params(params)
    axis = camb_ell_axis(cls)
    tt = cls["total_TT"] if spectra == "lensed" else cls["scalar_TT"]
    return {"TT": interpolate_spectrum(ells, axis, np.asarray(tt)),
            "PP": interpolate_spectrum(ells, axis, np.asarray(cls["phi"]))}


def noise_spectra(ells, spectra, param_ground, noise_level, l_knee, beam_fwhm,
                  qe_response = DEFAULT_QE_RESPONSE):
    """{"TT": N_l^TT, "PP": N_L^phiphi} at the FIDUCIAL cosmology.

    Both noises are properties of the experiment and the estimator rather than of the model
    being constrained, so they are evaluated once at param_ground and held fixed across the
    finite-difference stencil - letting them move would credit the forecast with
    dN/dtheta information no real analysis uses. The reconstruction noise always uses the
    lensed spectrum as the observed one, whichever spectrum the field block contracts, and
    whichever spectrum `qe_response` selects as its response (fisher_forecast.qe_response_cl,
    default "unlensed" as in Hu & Okamoto's original expression).

    Note the response is interpolated the same way every other spectrum here is: `ells` may
    run past CAMB's 2..CAMB_LMAX-1 support (that is the point of this module), so it is
    log-log continued out to ell_max exactly as the TT and phiphi spectra are - and with the
    same caveat that the continuation is an extrapolation, not a measurement.
    """
    cls = cls_with_qe_response(param_ground, qe_response)
    axis = camb_ell_axis(cls)
    noise_tt = map_noise_spectrum(ells, noise_level, l_knee, beam_fwhm)
    noise_pp = qe_noise_spectrum(ells,
                                 interpolate_spectrum(ells, axis,
                                                      np.asarray(qe_response_cl(cls, qe_response))),
                                 interpolate_spectrum(ells, axis, np.asarray(cls["total_TT"])),
                                 noise_tt)
    return {"TT": noise_tt, "PP": noise_pp}


def spectrum_derivatives(names, steps, param_ground, ells, spectra):
    """[{"TT": dC_l/dtheta, "PP": dC_L/dtheta} for each sampled parameter], by central
    differences of the CAMB spectra - 2 * len(names) CAMB calls, memoized."""
    derivatives = []
    for name, step in zip(names, steps):
        up, down = dict(param_ground), dict(param_ground)
        up[name] = param_ground[name] + step
        down[name] = param_ground[name] - step
        plus, minus = signal_spectra(up, ells, spectra), signal_spectra(down, ells, spectra)
        derivatives.append({block: (plus[block] - minus[block]) / (2 * step) for block in plus})
    return derivatives


# ── Binning and the Fisher sum ────────────────────────────────────────────

def band_binner(ells, f_sky, delta_ell):
    """(mode_weights, bin_fn): the (2l + 1)/2 f_sky mode counting, optionally in bands.

    With delta_ell = None every multipole is its own band, mode_weights are simply
    (2l + 1)/2 f_sky and bin_fn is the identity.

    With a band width, the multipoles in each band are pooled: a band's weight is the sum of
    its (2l + 1)/2 f_sky - its share of the sky's modes - and bin_fn replaces any spectrum by
    its (2l + 1)-weighted mean across the band, the bandpower a real analysis would form.
    Pooling C and dC before contracting them is exact only where the spectra are linear
    across a band, so the forecast loosens as delta_ell grows past the acoustic width
    (~50 in TT): binning answers "how much do the features I resolve carry", it is not a
    speed-up.
    """
    weights = 0.5 * f_sky * (2 * ells + 1)
    if delta_ell is None:
        return weights, (lambda spectrum: spectrum)

    if delta_ell < 1:
        raise ValueError(f"delta_ell must be at least 1 multipole, got {delta_ell}")
    band = ((ells - ells[0]) // delta_ell).astype(int)
    band_weights = np.bincount(band, weights = weights)

    def bin_fn(spectrum):
        return np.bincount(band, weights = weights * spectrum) / band_weights

    return band_weights, bin_fn


def fisher_from_spectra(signal, noise, derivatives, mode_weights, bin_fn):
    """F_ij = sum_blocks sum_l w_l dC_l/di (C_l + N_l)^-2 dC_l/dj, with w_l the mode weights.

    Returns (fisher, per_block): the total and a {block: fisher} breakdown, so the field's
    and the potential's shares of the information can be read separately.
    """
    n_param = len(derivatives)
    per_block = {}
    for block in signal:
        inverse_variance = 1.0 / bin_fn(signal[block] + noise[block])**2
        binned = [bin_fn(derivative[block]) for derivative in derivatives]
        fisher = np.empty((n_param, n_param))
        for i in range(n_param):
            for j in range(n_param):
                fisher[i, j] = np.sum(mode_weights * binned[i] * inverse_variance * binned[j])
        per_block[block] = fisher
    return sum(per_block.values()), per_block


# ── Entry points ──────────────────────────────────────────────────────────

def multipole_axis(ell_min, ell_max, verbose = True):
    """The integer multipoles ell_min .. ell_max.

    The default range is CAMB's own support, 2 .. CAMB_LMAX - 1, where every spectrum is a
    pure CAMB value. ell_max MAY go past it: the sampler's rfft grid reaches the box's
    Nyquist multipole pi / pix_width on the axes and sqrt(2) times that in the corners
    (4320 and 6110 at 2.5'), and covar_matrix_from_cls fills those modes for both the
    simulated data and the model by log-log extrapolating CAMB's last two multipoles. Set
    ell_max to the Nyquist (or the corner) to count that information the way the sampler
    sees it - interpolate_spectrum applies the same extrapolation here - bearing in mind
    that C_l, dC_l/dtheta and the reconstruction noise are all power-law continuations
    there, not physics.
    """
    ell_min = CAMB_ELL_MIN if ell_min is None else int(ell_min)
    ell_max = CAMB_ELL_MAX if ell_max is None else int(ell_max)
    if ell_min < CAMB_ELL_MIN:
        raise ValueError(f"ell_min must be at least {CAMB_ELL_MIN}, got {ell_min}")
    if ell_max <= ell_min:
        raise ValueError(f"ell_max ({ell_max}) must exceed ell_min ({ell_min})")
    if ell_max > CAMB_ELL_MAX and verbose:
        print(f"  NOTE: ell {CAMB_ELL_MAX + 1}..{ell_max} lies past CAMB's last multipole; "
              f"every spectrum there is the log-log extrapolation covar_matrix_from_cls "
              f"gives the sampler, not a CAMB value")
    return np.arange(ell_min, ell_max + 1, dtype = np.float64)


def forecast_from_1st_principles(nside, theta_pix, noise_level, is_sampled, param_ground,
                                 spectra = "lensed", step_fracs = None, l_knee = 0,
                                 beam_fwhm = 0, ell_min = None, ell_max = None,
                                 delta_ell = None, verbose = True,
                                 qe_response = DEFAULT_QE_RESPONSE):
    """The first-principles Fisher matrix for the sampled LCDM parameters.

    Arguments:
        nside, theta_pix:  the box, used ONLY for f_sky
        noise_level:       white map noise in uK-arcmin (l_knee, beam_fwhm as in noise_cls)
        is_sampled:        {param: bool}, which of PARAM_ORDER to forecast
        param_ground:      the fiducial cosmology, all five parameters
        spectra:           "lensed" or "unlensed" - which CAMB TT spectrum is the field
        step_fracs:        finite-difference steps as fractions of PARAM_SIGMA
        ell_min, ell_max:  the multipole range of BOTH blocks (default: all of CAMB's;
                           ell_max past CAMB_ELL_MAX extrapolates like the sampler does)
        delta_ell:         band width for binning (default: every multipole separately)
        qe_response:       which TT spectrum weights the quadratic estimator's response,
                           "unlensed" (default) or "gradient" - see QE_RESPONSE_SOURCES

    Returns (fisher, names, details) where details holds the axis, the spectra, the noises,
    the derivatives and the per-block Fisher matrices for plotting or inspection.
    """
    if spectra not in FIELD_SPECTRA:
        raise ValueError(f"spectra must be one of {FIELD_SPECTRA}, got {spectra!r}")

    names, steps, fracs = sampled_names_and_steps(is_sampled, param_ground, step_fracs)
    ells = multipole_axis(ell_min, ell_max, verbose)
    f_sky = sky_fraction(nside, theta_pix)
    mode_weights, bin_fn = band_binner(ells, f_sky, delta_ell)

    if verbose:
        print(f"First-principles Fisher forecast [{spectra}]: f_sky {f_sky:.4g} "
              f"(nside {nside}, theta_pix {theta_pix}'), {noise_level} uK-arcmin, "
              f"l_knee {l_knee}, beam {beam_fwhm}'")
        print(f"  summing ell {ells[0]:.0f}..{ells[-1]:.0f} in {len(mode_weights)} band(s) of "
              f"width {delta_ell if delta_ell is not None else 1}, "
              f"{2 * float(np.sum(mode_weights)):.0f} modes in all")
        print(f"  sampled: {names}, {2 * len(names) + 1} CAMB calls")

    signal = signal_spectra(param_ground, ells, spectra)
    noise = noise_spectra(ells, spectra, param_ground, noise_level, l_knee, beam_fwhm,
                          qe_response = qe_response)
    derivatives = spectrum_derivatives(names, steps, param_ground, ells, spectra)
    fisher, per_block = fisher_from_spectra(signal, noise, derivatives, mode_weights, bin_fn)

    if verbose:
        for name in names:
            print(f"    {name}: h = {steps[names.index(name)]:.6g} ({fracs[name]:g} sigma)")
        share = {block: np.sqrt(np.diag(matrix) / np.diag(fisher)) for block, matrix in per_block.items()}
        print("  sqrt(F_block / F_total) per parameter: "
              + ", ".join(f"{block} {np.array2string(value, precision = 2)}"
                          for block, value in share.items()))

    details = dict(ells = ells, signal = signal, noise = noise, derivatives = derivatives,
                   per_block = per_block, f_sky = f_sky)
    return fisher, names, details


def plot_qe_noise(ells, signal, noise, path, subtitle):
    """C_L^phiphi against the first-principles N_L^phiphi, in the L^4 C_L / 2 pi convention."""
    scale = ells**4 / (2 * np.pi)
    figure, axis = plt.subplots(figsize = (7, 4.5))
    axis.loglog(ells, scale * signal["PP"], label = r"$C_L^{\phi\phi}$")
    axis.loglog(ells, scale * noise["PP"], label = r"$N_L^{\phi\phi}$ (TT quadratic estimator)")
    axis.set_xlabel(r"$L$")
    axis.set_ylabel(r"$L^4 C_L / 2\pi$")
    axis.set_title(f"Lensing potential signal and reconstruction noise\n{subtitle}", fontsize = 9)
    axis.legend()
    axis.grid(alpha = 0.3)
    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description = "First-principles (2l+1)/2 f_sky Fisher forecast for the LCDM "
                      "parameters from CAMB spectra, over the TT field and the lensing "
                      "potential, with the quadratic-estimator noise computed analytically")
    add_box_arguments(parser)
    parser.add_argument("--spectra", choices = FIELD_SPECTRA, default = "lensed",
                        help = "which CAMB TT spectrum is the field (default lensed)")
    parser.add_argument("--ell_min", type = int, default = None,
                        help = f"first multipole of both blocks (default {CAMB_ELL_MIN})")
    parser.add_argument("--ell_max", type = int, default = None,
                        help = f"last multipole of both blocks (default {CAMB_ELL_MAX}, "
                               f"CAMB's last). Past it the spectra are log-log extrapolated "
                               f"exactly as the sampler's covar_matrix_from_cls does, so "
                               f"the box's Nyquist pi / pix_width (4320 at 2.5') counts "
                               f"the high-ell modes the sampler sees")
    parser.add_argument("--delta_ell", type = int, default = None,
                        help = "bin the spectra into bands of this width before contracting "
                               "(default: every multipole separately)")
    add_qe_response_argument(parser)
    parser.add_argument("--stability", action = "store_true",
                        help = "also recompute at 2x the step size and report the drift")
    args = parser.parse_args()
    is_sampled = sampled_from_args(parser, args)

    def run(step_fracs = None, verbose = True):
        return forecast_from_1st_principles(
            args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
            spectra = args.spectra, step_fracs = step_fracs, l_knee = args.l_knee,
            beam_fwhm = args.beam_fwhm, ell_min = args.ell_min, ell_max = args.ell_max,
            delta_ell = args.delta_ell, verbose = verbose,
            qe_response = args.qe_response)

    fisher, names, details = run()
    covariance = covariance_from_fisher(fisher, names)
    report_sigmas(covariance, names, args.spectra, METHOD)

    if args.stability:
        step_stability(lambda fracs: run(step_fracs = fracs, verbose = False)[:2], METHOD)

    ells = details["ells"]
    subtitle = (f"{args.spectra} TT + phiphi  |  f_sky {details['f_sky']:.3g}, "
                f"{args.noise:g} uK-arcmin  |  ell {ells[0]:.0f}..{ells[-1]:.0f}"
                + (f", bands of {args.delta_ell}" if args.delta_ell is not None
                   else ", every multipole"))
    #np.savez cannot store None, so an unset delta_ell is recorded as NaN
    config = run_config(args.spectra, args, names,
                        ell_min = ells[0], ell_max = ells[-1],
                        delta_ell = args.delta_ell if args.delta_ell is not None else np.nan,
                        f_sky = details["f_sky"], ells = ells,
                        cl_tt = details["signal"]["TT"], nl_tt = details["noise"]["TT"],
                        cl_pp = details["signal"]["PP"], nl_pp = details["noise"]["PP"],
                        fisher_tt = details["per_block"]["TT"],
                        fisher_pp = details["per_block"]["PP"])
    save_outputs(fisher, covariance, names, METHOD, SUFFIX, LABEL, subtitle, config)

    qe_path = os.path.join(output_dir(), f"qe_noise{SUFFIX}.png")
    plot_qe_noise(ells, details["signal"], details["noise"], qe_path, subtitle)
    print(f"Wrote qe_noise{SUFFIX}.png to {output_dir()}")


if __name__ == "__main__":
    main()
