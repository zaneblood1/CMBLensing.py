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
away its (genuine) square-box anisotropy. By default this module instead uses the flat-sky
N^(0) of Hu & Okamoto (2002) evaluated as a 2D quadrature in polar coordinates -
fisher_forecast.qe_noise_spectrum, which the block forecasts can also use (nphi_source =
"hu_okamoto") - so it is isotropic by construction and needs no matrix and no decoding.
"--nphi_source covariance" takes the box's matrix and azimuthally averages it after all,
which is the apples-to-apples comparison against the block forecasts at the cost of putting
a grid back into a module built not to have one; see NPHI_SOURCES. Its
response is built from whichever spectrum `--qe_response` selects, defaulting like the rest
of this package to Hu & Okamoto's own unlensed C_l^TT; pass "gradient" for the lensed
temperature-gradient spectrum, which is the correct weight (Lewis, Challinor & Hanson 2011).
See fisher_forecast.QE_RESPONSE_SOURCES.

The same N_L also drives the "delensed" field spectrum, so one reconstruction sets both the
potential block's noise floor and how much lensing is removed from the temperature.

Multipoles above CAMB's shipped 2..3999 are covered by RUNNING CAMB further
(camb_lmax_for_axis), not by extrapolating - see that function for the factor 2.5 this
corrects at the 2.5' corner.

Usage:
    python -m cmb_lensing.fisher_forecast_from_1st_principles
    python -m cmb_lensing.fisher_forecast_from_1st_principles --spectra unlensed
    python -m cmb_lensing.fisher_forecast_from_1st_principles --spectra delensed
    python -m cmb_lensing.fisher_forecast_from_1st_principles --spectra delensed --iterative_delens
    python -m cmb_lensing.fisher_forecast_from_1st_principles --nphi_source covariance   #the box's N_phi, azimuthally averaged
    python -m cmb_lensing.fisher_forecast_from_1st_principles --nphi_source measured --phi_noise <effective_phi_noise.npz>
    python -m cmb_lensing.fisher_forecast_from_1st_principles --nphi_source measured --phi_noise <...> --measured_extend hold
    python -m cmb_lensing.fisher_forecast_from_1st_principles --ell_min 30 --ell_max 3000 --delta_ell 20
    python -m cmb_lensing.fisher_forecast_from_1st_principles --ell_max 4320   #to the box's Nyquist

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

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.simulate import noise_cls
from cmb_lensing.constants import DEFAULT_MAX_ELL
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, CAMB_LMAX, PARAM_ORDER
from cmb_lensing.fisher_forecast import (camb_cls_at_params, cls_with_qe_response,
                                         grid_cls_at_params, CL_SOURCES,
                                         DEFAULT_CL_SOURCE,
                                         delensed_cls_at_params,
                                         qe_response_cl, DEFAULT_QE_RESPONSE,
                                         add_qe_response_argument,
                                         sampled_names_and_steps,
                                         covariance_from_fisher, step_stability,
                                         report_sigmas, save_outputs, output_dir,
                                         add_box_arguments, sampled_from_args, run_config,
                                         interpolate_spectrum, qe_noise_spectrum,
                                         qe_noise_matrix, _radial_cl_profile, grid_max_ell,
                                         RADIAL_MEAN_TYPES, DEFAULT_RADIAL_MEAN,
                                         add_radial_mean_argument,
                                         load_phi_noise, check_phi_noise,
                                         measured_phi_noise_cl, add_phi_noise_argument,
                                         MEASURED_EXTEND_MODES)
from cmb_lensing.fisher_forecast_full_sky import sky_fraction
from cmb_lensing.delensed_spectrum import (measured_delensing_ratio, ratio_at_params,
                                           has_ratio_derivatives)


METHOD = "1st_principles"
SUFFIX = "_1st_principles"
LABEL = (r"1st principles:  $F_{ij} = \sum_\ell \frac{2\ell+1}{2} f_{sky}\,"
         r"\partial_i C_\ell\,(C_\ell + N_\ell)^{-2}\,\partial_j C_\ell$  over TT and $\phi\phi$")

#which CAMB temperature spectrum plays C_l^TT in the field block:
#  "lensed"    C_l^TT lensed - what the instrument measures, the default
#  "unlensed"  C_l^TT unlensed - the perfect-delensing marker. Not reachable by any
#              estimator, so quote it as a marker and not as a bound
#  "delensed"  CAMB's get_partially_lensed_cls with C_L^phiphi scaled per multipole by
#              Alens_L = N_L / (C_L^phiphi + N_L), the residual fraction a Wiener-filtered
#              reconstruction leaves - the honest point between the two. N_L comes from
#              whichever NPHI_SOURCES entry is selected, evaluated on this analysis's own
#              multipole range (delensing_fraction), so ONE reconstruction sets both the
#              phi block's noise floor and how much lensing comes out of the temperature.
#              Carries the same double counting "lensed" has, scaled down by Alens_L: the
#              residual lensing left in the map tracks C_phi(theta), so the field block
#              picks up lensing information on omch2 that the potential block counts again
FIELD_SPECTRA = ("lensed", "unlensed", "delensed")

#where the lensing reconstruction noise N_L^phiphi comes from. The names and meanings match
#fisher_forecast.NPHI_SOURCES exactly:
#  "hu_okamoto"  (DEFAULT) the analytic flat-sky N^(0) of Hu & Okamoto (2002), evaluated by
#                qe_noise_spectrum as a polar quadrature over THIS analysis's own multipole
#                range. Isotropic by construction, no matrix, no box - which is the premise
#                of this whole module, and the reason it can count multipoles past anything
#                a grid carries
#  "covariance"  the box's own N_phi matrix (qe_noise_matrix, the same
#                simulate.scalar_quadratic_estimate the sampler's G is built from),
#                azimuthally averaged back onto the 1D axis by _radial_cl_profile. This is
#                what the block forecasts contract, so it is the apples-to-apples
#                comparison against them
#The two differ for a real reason, and it is not numerical error: the box's estimator
#integrates over a periodic SQUARE of modes rather than an isotropic annulus, which makes
#the QE norm vary by 1.6x around an annulus at L = 540 and 5.7x at L = 2090 (nside 64 / 5'),
#and the FFT's periodic wrap of L - l lowers it a further 2-6x below the un-wrapped square.
#Verified 2026-09-11 that restricting qe_noise_spectrum to the box's square with that wrap
#reproduces scalar_quadratic_estimate to 0.1-0.5%, so the gap is geometry, not a bug.
#IMPORTANT: "covariance" breaks this module's "the box only sets f_sky" premise - nside and
#theta_pix then also set the grid the estimator lives on, and _radial_cl_profile log-log
#EXTRAPOLATES both below the grid's fundamental and above its corner mode. The low-l end
#matters most (it is silent, and it is where C_phi and the delensing live); forecast prints
#how much of the axis is extrapolated at each end
#  "measured"    the EMPIRICAL N_L^eff written by merge_phi_noise.py: the noise map_joint's
#                MAP reconstruction actually achieves, from the cross correlation of the
#                estimate with the true phi over many realizations (cmb_lensing/phi_noise.py).
#                Neither analytic source describes a MAP estimator at all - they both
#                describe a quadratic one - so this is the only N_L that matches the
#                reconstruction this codebase performs. Needs --phi_noise, and the npz must
#                have been measured on THIS box (check_phi_noise refuses otherwise). It is a
#                measurement, so --qe_response and --iterative_delens do not apply to it:
#                there is no response to re-weight and no filter to iterate
NPHI_SOURCES = ("hu_okamoto", "covariance", "measured")
DEFAULT_NPHI_SOURCE = "hu_okamoto"

#how the measured N_L^eff is continued outside the |L| bands it was measured in - the grid's
#fundamental to its corner - when this module's axis reaches past them at either end.
#fisher_forecast.MEASURED_EXTEND_MODES has the full argument; in short, "extrapolate"
#continues the measured power law (what every other spectrum here gets) and "hold" clamps to
#the end values. THE DEFAULT HERE IS "extrapolate", which differs from the block forecasts'
#"hold" - see the warning forecast_from_1st_principles prints, since N_eff is steep and the
#lever arm from the lowest band down to L = 2 is long
DEFAULT_MEASURED_EXTEND = "extrapolate"

#where the delensed temperature spectrum comes from, for --spectra delensed:
#  "camb"      (DEFAULT) CAMB's get_partially_lensed_cls with C_L^phiphi scaled by
#              Alens_L = N_L / (C_L + N_L). CAMB supplies the theta dependence of the
#              delensed spectrum, so the stencil differences a real lensing calculation at
#              every point - but the lensing is CAMB's full-sky correlation-function method
#              and the reconstruction is an idealized Wiener filter, neither of which is what
#              this codebase does
#  "measured"  the EMPIRICAL ratio D(l) = <C_l^delensed / C_l^unlensed> measured on the box
#              by delensed_spectrum.measure_delensed_spectrum - lense_flow forward, map_joint
#              to reconstruct, lense_flow backward to delens - divided per realization so the
#              cosmic variance cancels. The delensed spectrum is then
#              D(l) * C_l^unlensed_CAMB(theta). NO Alens_L and no CAMB lensing calculation
#              enter at all; N_L keeps its OTHER job as the phi block's noise
#              (C_L^phiphi + N_L) and plays no part in the delensing
#CAVEAT, and it is the important one: D is measured at ONE cosmology and held fixed, so the
#contracted derivative is D * dC^unlensed/dtheta and the whole residual-lensing correction is
#frozen in theta. That is a STRONGER assumption than the block forecasts' --transfer_function
#makes: R there is a ratio between two DELENSED spectra and sits near one, while D carries
#the entire residual lensing, which tracks C_phi(theta). Nothing in this module tests it -
#measure two directories at separated cosmologies and compare, as compare_transfer_functions.py
#does for R.
#WHAT THAT COSTS, measured at nside 128 / 2.5' / 5 uK: this mode comes out almost exactly
#equal to --spectra unlensed (omch2 0.004909 vs 0.004909, theta 0.003415 vs 0.003397, logA
#0.02287 vs 0.02282). That is algebra, not coincidence. With C = D * C^unlensed and D frozen,
#dC = D * dC^unlensed, so the Fisher integrand D^2 (dC_u)^2 / (D C_u + N)^2 loses D entirely
#wherever the noise is negligible - the delensing cancels between the derivative and the
#variance, and only the noise weighting remembers it. The CAMB route keeps a real theta
#dependence in the residual lensing and lands well away from unlensed (omch2 0.004107), which
#is the double counting documented in FIELD_SPECTRA. So use this mode to ask "what does the
#BOX's delensed spectrum look like", not as a tighter delensed forecast: as specified it is
#close to the perfect-delensing marker. Measuring dD/dtheta would be what makes it a forecast
DELENSED_SOURCES = ("camb", "measured")
DEFAULT_DELENSED_SOURCE = "camb"

#the isotropic high-l cutoff qe_noise_matrix applies; fisher_forecast.forecast's own default
DEFAULT_L_CUTOFF = 10_000

#the fixed-point iteration between the delensing fraction and the reconstruction noise
#(--iterative_delens), mirroring fisher_forecast.iterative_delensing
DELENS_MAX_ITERATIONS = 25
DELENS_TOLERANCE = 1e-6

#CAMB's multipole support: camb_cls_at_params puts every spectrum on 2 .. CAMB_LMAX - 1.
#ell_max may exceed CAMB_ELL_MAX, see multipole_axis
CAMB_ELL_MIN = 2
CAMB_ELL_MAX = CAMB_LMAX - 1

# ── Spectra on a 1D multipole axis ────────────────────────────────────────

def camb_ell_axis(cls):
    """The integer multipoles every CAMB spectrum in `cls` lives on."""
    return np.arange(CAMB_ELL_MIN, CAMB_ELL_MIN + cls["total_TT"].shape[0], dtype = np.float64)


def camb_lmax_for_axis(ells, camb_lmax = None):
    """How far CAMB is run so that no multipole in the sum is a power-law continuation.

    This module exists partly to count the modes ABOVE CAMB's shipped 2..3999 - the box's
    Nyquist is 4320 at 2.5' and its corner 6109 - and it used to reach them by log-log
    extrapolating CAMB's last two multipoles, the same continuation covar_matrix_from_cls
    gives the sampler. That continuation is BAD: the damping tail falls faster than any power
    law. Measured at the 2.5' corner, the extrapolated lensed TT is 0.40x CAMB's own value
    and dC_l/d(omch2) is 0.39x, which moved the marginalized sigmas by -10.5% (omch2),
    +23.2% (theta_MC_100) and +12.7% (logA) - so any pre-2026-09-21 number from this module
    at an --ell_max past 3999 is stale. fisher_forecast.camb_lmax_for_grid made exactly this
    fix for the block forecasts; this is its 1D counterpart.

    Running CAMB further is the whole remedy and it is cheap, so the default simply covers
    the axis. Floored at CAMB_LMAX so the default range is bit-for-bit what it always was.
    Pass an explicit value to pin it - CAMB_LMAX reproduces the extrapolated numbers.
    """
    if camb_lmax is not None:
        return int(camb_lmax)
    return max(CAMB_LMAX, int(ells[-1]) + 1)


def map_noise_spectrum(ells, noise_level, l_knee, beam_fwhm, camb_lmax = None):
    """N_l^TT: the beam-DECONVOLVED map noise, so the signal it pairs with is the bare C_l.

    noise_cls already divides the white + 1/f noise by the beam transfer function B_l^2,
    which is the convention in which the observed spectrum is C_l + N_l with C_l the sky
    spectrum itself. That is the convention this whole module uses.

    Evaluated out to the same multipole the signal spectra are, so the noise is analytic
    everywhere the sum runs rather than log-log continued. It matters only at a nonzero beam,
    where B_l^-2 grows faster than a power law exactly as the damping tail falls faster than
    one; at the default beam of 0 the white noise is flat and the two agree.
    """
    lmax_prime = min(DEFAULT_MAX_ELL, CAMB_LMAX) if camb_lmax is None else int(camb_lmax)
    n_tt, _, _, _ = noise_cls(lmax_prime, noise_level, beam_fwhm = beam_fwhm, l_knee = l_knee)
    noise_ells = np.arange(CAMB_ELL_MIN, lmax_prime, dtype = np.float64)
    return interpolate_spectrum(ells, noise_ells, np.asarray(n_tt))


# ── The blocks and their finite-difference derivatives ────────────────────

def delensing_fraction(ells, cls, nphi_pp):
    """Alens_L = N_L / (C_L^phiphi + N_L) on this analysis's range, 1 outside it.

    Zero-based in L, CAMB's convention for get_partially_lensed_cls: Alens_L = 1 leaves a
    multipole fully lensed, Alens_L = 0 delenses it perfectly.

    fisher_forecast.delensing_alens does the same arithmetic for the block forecasts, but it
    assumes N_L already lives on CAMB's own axis. Here N_L is defined on `ells`, the
    analysis's multipole range, which may start above 2 and end past CAMB's last multipole -
    so the two have to be reconciled explicitly. Outside [ells[0], ells[-1]] there is no
    reconstruction AT ALL (qe_noise_spectrum builds the estimator from exactly those
    multipoles), so those L stay fully lensed rather than being extrapolated: a delensing
    fraction is not a spectrum and has no power-law continuation.
    """
    #sized from C_phi's OWN axis rather than camb_ell_axis (which reads total_TT's length):
    #the two match today, but fisher_forecast.covariance_blocks keeps them separate and
    #delensing_alens sizes from C_phi, so a future CAMB lens margin cannot silently
    #misalign this by a multipole
    cphi = np.asarray(cls["phi"])
    phi_axis = np.arange(CAMB_ELL_MIN, CAMB_ELL_MIN + len(cphi), dtype = np.float64)
    alens = np.ones(2 + len(cphi))
    inside = (phi_axis >= ells[0]) & (phi_axis <= ells[-1])
    nphi_on_axis = interpolate_spectrum(phi_axis[inside], ells, nphi_pp)
    alens[2:][inside] = nphi_on_axis / (cphi[inside] + nphi_on_axis)
    return alens


def check_delensing_ratio_derivatives(measured, param_ground, names, directory = ""):
    """The linear model D_0 + (theta - theta_0) dD is only meaningful about its own point.

    Mirrors fisher_forecast.check_transfer_derivatives: D_0's cosmology must BE the
    forecast's fiducial point, and any sampled parameter without a measured derivative is
    held flat - which, for D, means that parameter's delensed forecast reduces to the
    unlensed one, so it is worth saying out loud rather than leaving implicit.
    """
    if not has_ratio_derivatives(measured):
        print(f"  WARNING: no dD/dtheta in {directory}, so D is held flat across the "
              f"stencil. The delensing then CANCELS out of the Fisher integrand wherever "
              f"the noise is small and this mode reduces to --spectra unlensed; pass "
              f"--delensed_shifted_dirs to measure it.")
        return

    reference = measured["params"]
    off = {name: (float(reference[name]), float(param_ground[name])) for name in PARAM_ORDER
           if not np.isclose(float(reference[name]), float(param_ground[name]),
                             rtol = 1e-12, atol = 0)}
    if off:
        raise ValueError(
            f"D in {directory} was measured at a cosmology that differs from this "
            f"forecast's fiducial point in {sorted(off)} (measured vs forecast: {off}). "
            f"D_0 + (theta - theta_0) dD/dtheta is an expansion ABOUT the point D was "
            f"measured at, so the stencil has to be centred there too.")

    missing = [name for name in names
               if name not in {str(value) for value in measured["derivative_names"]}]
    if missing:
        print(f"  WARNING: sampled {missing} have no measured dD/dtheta, so D is held flat "
              f"in those directions and their delensed forecast reduces to the unlensed "
              f"one. Measure them with --delensed_shifted_dirs.")


def apply_delensing_ratio(ells, measured, params = None):
    """D(l) on `ells` at the cosmology `params`: LOG-log interpolation, HELD outside.

    `params` is the stencil point the spectrum belongs to. When the measurement carries
    dD/dtheta (measured_delensing_ratio's `shifted_dirs`) D is evaluated there through the
    linear model in ratio_at_params, so the stencil's derivative picks up the second term of
    the product rule, C_u dD/dtheta, alongside D dC_u/dtheta. Without derivatives D_0 is
    applied everywhere and the first term is all there is - see DELENSED_SOURCES for why
    that collapses onto the unlensed forecast.

    Note this differs from fisher_forecast.apply_transfer_function, which interpolates R(l)
    linearly. R is a ratio between two DELENSED spectra and sits within a few percent of one,
    so a straight line between bands is right for it. D is a ratio to the UNLENSED spectrum,
    and lensing fills in the damping tail: measured at nside 128 / 2.5' / 5 uK, D runs from
    0.90 up to 5.5e3 across its bands, because C^unlensed collapses at high l while the
    delensed residual does not. Interpolating a quantity that spans three and a half decades
    linearly between bands 100 multipoles wide would cut the chord across an exponential.
    D is strictly positive, so log-log is both safe and the same rule every positive spectrum
    in this module gets.

    Below l ~ 4000 D is in fact near one and the two rules agree, so this matters only for an
    --ell_max reaching into the damping tail - which is exactly where this module is designed
    to go. Outside the measured bands the value is HELD (np.interp's own clamping): there is
    no measurement to continue from, and holding keeps the product continuous.
    """
    band_ells = np.asarray(measured["band_ells"])
    ratio = ratio_at_params(measured, params)
    #the linear model can in principle push a band non-positive far from theta_0; log-log
    #needs it positive, and a non-positive delensed spectrum is meaningless anyway
    if np.any(ratio <= 0):
        raise ValueError(
            f"the linear model D_0 + (theta - theta_0) dD/dtheta gave a non-positive D in "
            f"{int(np.sum(ratio <= 0))} band(s) at {params}. The displacement is too large "
            f"for the expansion - measure dD/dtheta over a narrower Delta, or check the "
            f"second-difference report from merge_delensing_ratio_derivatives.")
    return np.exp(np.interp(np.log(np.asarray(ells)), np.log(band_ells), np.log(ratio)))


def check_delensing_ratio(measured, nside, theta_pix, noise_level, l_knee, directory = ""):
    """Refuse a D(l) measured on a different box than the forecast runs on.

    D absorbs the box's resolution, its periodic lensing and map_joint's own reconstruction
    quality, so it only means anything at the configuration it was measured at. Mirrors
    fisher_forecast.check_phi_noise and check_transfer_function.
    """
    measured_box = dict(nside = int(measured["nside"]),
                        theta_pix = float(measured["theta_pix"]),
                        noise_level = float(measured["noise_level"]),
                        l_knee = float(measured["l_knee"]))
    wanted = dict(nside = int(nside), theta_pix = float(theta_pix),
                  noise_level = float(noise_level), l_knee = float(l_knee))
    if measured_box != wanted:
        raise ValueError(
            f"the measured delensing ratio in {directory} was taken at {measured_box} but "
            f"this forecast runs at {wanted}. D absorbs the box's resolution and its "
            f"reconstruction quality, so it cannot be carried across - re-run "
            f"get_delensed_spectra.sh at this box.")


def model_cls_at_params(params, cl_source = DEFAULT_CL_SOURCE, camb_lmax = None):
    """The model spectra at `params` from the selected fisher_forecast.CL_SOURCES entry:
    a direct CAMB run, or the 5D grid spline the sampler evaluates (which carries only
    scalar_TT, total_TT and phi, on CAMB's shipped 2..CAMB_LMAX-1 whatever `camb_lmax`)."""
    if cl_source == "grid":
        return grid_cls_at_params(params)
    return camb_cls_at_params(params, camb_lmax = camb_lmax)


def signal_spectra(params, ells, spectra, alens = None, camb_lmax = None,
                   delensing_ratio = None, cl_source = DEFAULT_CL_SOURCE):
    """{"TT": C_l^TT, "PP": C_L^phiphi} from one (memoized) CAMB run at `params`, or with
    cl_source = "grid" from the 5D CAMB grid spline (see model_cls_at_params).

    For spectra = "delensed" the temperature spectrum is CAMB's partially lensed one at the
    frozen per-L `alens` (delensed_cls_at_params - the full non-perturbative
    correlation-function lensing run with the potential scaled, NOT an interpolation between
    unlensed and lensed). `alens` is evaluated once at the fiducial cosmology and re-applied
    at every stencil point, so the derivative this feeds the sum is d/dtheta of CAMB's
    partially lensed spectrum at a FIXED delensing fraction - the same convention
    noise_spectra uses for both noises, and the same one fisher_forecast's constant_nphi
    holds by default.
    """
    if spectra == "delensed" and delensing_ratio is not None:
        #the empirical route: the box's own measured delensing applied to CAMB's UNLENSED
        #spectrum. No Alens_L, no CAMB lensing calculation - see DELENSED_SOURCES
        cls = model_cls_at_params(params, cl_source, camb_lmax = camb_lmax)
        axis = camb_ell_axis(cls)
        #D is evaluated AT `params`, so the finite difference carries both terms of
        #d(D C_u)/dtheta = D dC_u/dtheta + C_u dD/dtheta. Without measured derivatives
        #ratio_at_params returns D_0 and only the first term survives
        unlensed = interpolate_spectrum(ells, axis, np.asarray(cls["scalar_TT"]))
        return {"TT": unlensed * apply_delensing_ratio(ells, delensing_ratio,
                                                       params = params),
                "PP": interpolate_spectrum(ells, axis, np.asarray(cls["phi"]))}

    if spectra == "delensed":
        if alens is None:
            raise ValueError("spectra = 'delensed' needs either the per-L delensing "
                             "fraction (delensed_source = 'camb') or a measured D(l) "
                             "(delensed_source = 'measured'); build one with "
                             "reconstruction(), as forecast_from_1st_principles does")
        cls = delensed_cls_at_params(params, alens, camb_lmax = camb_lmax)
        tt = cls["delensed_TT"]
    else:
        cls = model_cls_at_params(params, cl_source, camb_lmax = camb_lmax)
        tt = cls["total_TT"] if spectra == "lensed" else cls["scalar_TT"]
    axis = camb_ell_axis(cls)
    return {"TT": interpolate_spectrum(ells, axis, np.asarray(tt)),
            "PP": interpolate_spectrum(ells, axis, np.asarray(cls["phi"]))}


def reconstruction(ells, spectra, param_ground, noise_level, l_knee, beam_fwhm,
                   nside, theta_pix, qe_response = DEFAULT_QE_RESPONSE,
                   nphi_source = DEFAULT_NPHI_SOURCE,
                   radial_mean = DEFAULT_RADIAL_MEAN, l_cutoff = DEFAULT_L_CUTOFF,
                   measured_phi_noise = None,
                   measured_extend = DEFAULT_MEASURED_EXTEND,
                   delensed_source = DEFAULT_DELENSED_SOURCE,
                   iterative_delens = False, camb_lmax = None, verbose = True):
    """({"TT": N_l^TT, "PP": N_L^phiphi}, Alens_L) at the FIDUCIAL cosmology.

    Both noises are properties of the experiment and the estimator rather than of the model
    being constrained, so they are evaluated once at param_ground and held fixed across the
    finite-difference stencil - letting them move would credit the forecast with
    dN/dtheta information a real analysis removes with a realization-dependent N0. The same
    reasoning freezes `Alens_L`, which is why it is returned from here alongside the noises
    rather than rebuilt per stencil point; fisher_forecast.DEFAULT_CONSTANT_NPHI is the long
    form of the argument (and its --vary_nphi escape hatch is NOT plumbed through here).

    The reconstruction noise uses the lensed spectrum as the observed one, whichever spectrum
    the field block contracts, and whichever spectrum `qe_response` selects as its response
    (fisher_forecast.qe_response_cl, default "unlensed" as in Hu & Okamoto's original
    expression). `Alens_L` is None unless spectra = "delensed".

    `nphi_source` picks where N_L^phiphi comes from - the analytic quadrature or the box's
    own matrix azimuthally averaged; see NPHI_SOURCES, and note that "covariance" makes
    nside / theta_pix matter for more than f_sky. `radial_mean` and `l_cutoff` only apply to
    that source. Whichever is chosen sets BOTH the potential block's noise floor and, for
    "delensed", the delensing fraction, so the two can never disagree about the same
    estimator.

    With `iterative_delens` the delensing fraction and the estimator's filter are iterated to
    a fixed point: delensing quiets the map, a quieter map lowers N_L, a lower N_L delenses
    better. The first pass reproduces the one-shot answer exactly, so this can only improve
    on it. Only the FILTER moves - the response stays the fiducial `qe_response` spectrum,
    since it is a property of how the estimator is built rather than of the map it is
    applied to.

    Note the response is interpolated the same way every other spectrum here is: `ells` may
    run past CAMB's 2..CAMB_LMAX-1 support (that is the point of this module), so it is
    log-log continued out to ell_max exactly as the TT and phiphi spectra are - and with the
    same caveat that the continuation is an extrapolation, not a measurement.
    """
    if nphi_source not in NPHI_SOURCES:
        raise ValueError(f"nphi_source must be one of {NPHI_SOURCES}, got {nphi_source!r}")
    if nphi_source == "measured" and measured_phi_noise is None:
        raise ValueError(
            "nphi_source = 'measured' needs the merge_phi_noise.py npz. Pass it as "
            "--phi_noise <path>, or measured_phi_noise = load_phi_noise(path) when calling "
            "this module directly.")
    if nphi_source != "measured" and measured_phi_noise is not None:
        raise ValueError(
            f"a measured N_L^eff was given but nphi_source is {nphi_source!r}, so it would "
            f"be parsed and then ignored. Pass --nphi_source measured to use it.")
    #the iteration exists to guess what a MAP reconstruction would achieve by quieting a
    #quadratic estimator's filter; a measured N_eff already IS what the MAP achieved, so
    #iterating it would re-apply the correction on top of the measurement. Same refusal
    #fisher_forecast.frozen_reconstruction makes
    if iterative_delens and nphi_source == "measured":
        raise ValueError(
            "iterative delensing cannot be combined with nphi_source = 'measured'. The "
            "iteration quiets a quadratic estimator's filter to approximate a MAP "
            "reconstruction; N_L^eff is already the MAP's own noise, so there is nothing "
            "to iterate and the correction would be double counted.")

    cls = cls_with_qe_response(param_ground, qe_response, camb_lmax = camb_lmax)
    axis = camb_ell_axis(cls)
    noise_tt = map_noise_spectrum(ells, noise_level, l_knee, beam_fwhm,
                                  camb_lmax = camb_lmax)
    response = interpolate_spectrum(ells, axis,
                                    np.asarray(qe_response_cl(cls, qe_response)))

    if nphi_source == "measured" and verbose:
        measured_phi_noise_cl(measured_phi_noise, ells, verbose = True,
                              extend = measured_extend)
        #N_eff is steep, so a power-law continuation from the lowest band down to L = 2 has
        #a long lever arm and can move Alens_L by orders of magnitude. Say how far it runs
        bands = np.asarray(measured_phi_noise["band_ells"])
        below = int(np.sum(ells < bands[0]))
        if measured_extend == "extrapolate" and below:
            decades = np.log10(bands[0] / max(float(ells[0]), 1.0))
            print(f"    NOTE: {below} multipoles below the lowest band are a log-log "
                  f"CONTINUATION over {decades:.2f} decades of lever arm, fitted to the two "
                  f"end bands. N_eff is steep there and C_phi is largest there, so this "
                  f"drives Alens_L; --measured_extend hold clamps instead")

    ell_grid = pix_width = weights = None
    if nphi_source == "covariance":
        ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
        weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                                   (nside, nside // 2 + 1))
        if verbose:
            on_grid = ell_grid[ell_grid > 0]
            low, high = float(jnp.min(on_grid)), grid_max_ell(ell_grid)
            below = int(np.sum(ells < low))
            above = int(np.sum(ells > high))
            print(f"  N_L from the box's QE matrix, azimuthally averaged ({radial_mean} "
                  f"annulus mean): grid carries l {low:.0f}..{high:.0f}")
            #the low-l end is where C_phi and the delensing live, and _radial_cl_profile
            #extrapolates there SILENTLY - so say it out loud
            if below or above:
                print(f"    {below} multipoles below the grid's fundamental and {above} "
                      f"above its corner are log-log extrapolations of the profile, not "
                      f"box measurements"
                      + (" - the low-l end carries most of C_phi" if below else ""))

    def noise_pp_at(filter_camb):
        """N_L^phiphi on `ells` from the chosen source.

        `filter_camb` is the TT spectrum the estimator's inverse-variance filter sees, on
        CAMB's OWN axis - the lensed spectrum normally, the delensed one while iterating.
        The two sources want it in different bases (the quadrature on `ells`, the matrix on
        the rfft grid via covar_matrix_from_cls), so it arrives unconverted and each branch
        puts it where it needs it.
        """
        if nphi_source == "measured":
            #a measurement, not a recomputation: the filter it was taken with is whatever
            #map_joint ran, so filter_camb has nothing to act on here
            return measured_phi_noise_cl(measured_phi_noise, ells,
                                         extend = measured_extend)
        if nphi_source == "hu_okamoto":
            return qe_noise_spectrum(ells, response,
                                     interpolate_spectrum(ells, axis,
                                                          np.asarray(filter_camb)),
                                     noise_tt)
        matrix = qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                                 beam_fwhm, l_cutoff, filter_tt = filter_camb,
                                 qe_response = qe_response)
        return np.asarray(_radial_cl_profile(matrix, ell_grid, weights, pix_width, ells,
                                             radial_mean = radial_mean))

    noise_pp = noise_pp_at(cls["total_TT"])
    #delensed_source = "measured" carries its own delensing, so there is no Alens_L to build
    #and N_L is left doing nothing but its other job - the phi block's noise term
    if spectra != "delensed" or delensed_source == "measured":
        return {"TT": noise_tt, "PP": noise_pp}, None

    def mean_efficiency(alens_array):
        """The C_phi-weighted delensing actually APPLIED, read off `alens` itself.

        fisher_forecast.delensing_efficiency forms the same weighted mean of C/(C + N) from
        a noise spectrum, but calling it here would need N_L interpolated from `ells` onto
        C_phi's axis - which EXTRAPOLATES wherever the analysis range is narrower than
        CAMB's, and would then report a delensing that delensing_fraction does not apply
        (it holds Alens_L = 1 outside the range). Measured at nside 64 / 5' with
        ell_min = 68, that version claimed 0.989 while Alens_L(50) was 1.00, i.e. no
        delensing at all there. Reading 1 - Alens_L instead makes the printed number
        describe exactly the reconstruction delensed_cls_at_params receives, contributing
        zero where nothing is reconstructed.
        """
        cphi = np.asarray(cls["phi"])
        residual = np.asarray(alens_array[2:])
        return float(np.sum(cphi * (1.0 - residual)) / np.sum(cphi))

    alens = delensing_fraction(ells, cls, noise_pp)

    #C_phi is so steeply red that the multipoles below ell_min carry almost all of it, and
    #delensing_fraction holds Alens_L = 1 there because N_L was never evaluated outside the
    #analysis range. So raising ell_min does not just trim the sum - it switches the
    #delensing off in the C_phi-weighted sense, which the efficiency below reports as ~0.
    #Loud rather than inferred: the forecast still runs, but "delensed" then means
    #"delensed above ell_min only" and is not comparable to an ell_min = 2 run
    missing = float(np.sum(np.asarray(cls["phi"])[:max(0, int(ells[0]) - CAMB_ELL_MIN)]))
    if missing / float(np.sum(np.asarray(cls["phi"]))) > 0.01:
        print(f"  WARNING: ell_min = {ells[0]:.0f} leaves L < {ells[0]:.0f} unreconstructed "
              f"(Alens_L = 1 there), and those multipoles carry "
              f"{100 * missing / float(np.sum(np.asarray(cls['phi']))):.2f}% of C_phi. The "
              f"delensing is effectively switched off; use ell_min = {CAMB_ELL_MIN} for a "
              f"delensed forecast, or restrict the range only in the field block.")

    if iterative_delens:
        if verbose:
            print("  iterating the per-L delensing fraction against N_L:")
        converged = False
        for iteration in range(1, DELENS_MAX_ITERATIONS + 1):
            #on CAMB's axis - noise_pp_at converts it for whichever source is in use
            filtered = delensed_cls_at_params(param_ground, alens,
                                              camb_lmax = camb_lmax)["delensed_TT"]
            noise_pp = noise_pp_at(filtered)
            updated = delensing_fraction(ells, cls, noise_pp)
            shift = float(np.max(np.abs(updated - alens)))
            alens = updated
            if verbose:
                print(f"    iteration {iteration}: mean efficiency "
                      f"{mean_efficiency(alens):.5f} (max Alens shift {shift:.2e})")
            if shift < DELENS_TOLERANCE:
                converged = True
                break
        if not converged and verbose:
            print(f"    WARNING: did not converge in {DELENS_MAX_ITERATIONS} steps "
                  f"(last shift {shift:.2e}); the reported Alens_L is the last iterate")

    if verbose:
        samples = ", ".join(f"L={L} {alens[L]:.2f}" for L in (50, 100, 300, 1000, 2000)
                            if L < len(alens))
        print(f"  delensing: C_phi-weighted mean efficiency "
              f"{mean_efficiency(alens):.3f}; residual fraction Alens_L at {samples}")

    return {"TT": noise_tt, "PP": noise_pp}, alens


def spectrum_derivatives(names, steps, param_ground, ells, spectra, alens = None,
                         camb_lmax = None, delensing_ratio = None,
                         cl_source = DEFAULT_CL_SOURCE):
    """[{"TT": dC_l/dtheta, "PP": dC_L/dtheta} for each sampled parameter], by central
    differences of the CAMB spectra - 2 * len(names) CAMB calls, memoized.

    `alens` is the frozen delensing fraction for spectra = "delensed"; CAMB re-delenses at
    every stencil point with it, so what is differenced is the partially lensed spectrum of
    each cosmology at a fixed reconstruction.
    """
    derivatives = []
    for name, step in zip(names, steps):
        up, down = dict(param_ground), dict(param_ground)
        up[name] = param_ground[name] + step
        down[name] = param_ground[name] - step
        plus = signal_spectra(up, ells, spectra, alens = alens, camb_lmax = camb_lmax,
                              delensing_ratio = delensing_ratio, cl_source = cl_source)
        minus = signal_spectra(down, ells, spectra, alens = alens, camb_lmax = camb_lmax,
                               delensing_ratio = delensing_ratio, cl_source = cl_source)
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

def multipole_axis(ell_min, ell_max, camb_lmax = None, verbose = True):
    """The integer multipoles ell_min .. ell_max.

    The default range is CAMB's own shipped support, 2 .. CAMB_LMAX - 1. ell_max MAY go past
    it, which is a large part of why this module exists: the sampler's rfft grid reaches the
    box's Nyquist multipole pi / pix_width on the axes and sqrt(2) times that in the corners
    (4320 and 6109 at 2.5'), so setting ell_max there counts the modes the box actually
    carries. Since 2026-09-21 those multipoles are covered by RUNNING CAMB that far
    (camb_lmax_for_axis) rather than by log-log extrapolating its last two points, which was
    wrong by a factor 2.5 in C_l at the corner.

    A `camb_lmax` pinned BELOW ell_max puts the remainder back on that extrapolation - the
    continuation covar_matrix_from_cls gives the sampler - and warns, since C_l,
    dC_l/dtheta and the reconstruction noise are then all power-law continuations rather
    than physics.
    """
    ell_min = CAMB_ELL_MIN if ell_min is None else int(ell_min)
    ell_max = CAMB_ELL_MAX if ell_max is None else int(ell_max)
    if ell_min < CAMB_ELL_MIN:
        raise ValueError(f"ell_min must be at least {CAMB_ELL_MIN}, got {ell_min}")
    if ell_max <= ell_min:
        raise ValueError(f"ell_max ({ell_max}) must exceed ell_min ({ell_min})")
    if camb_lmax is not None and ell_max > int(camb_lmax) - 1 and verbose:
        print(f"  WARNING: ell {int(camb_lmax)}..{ell_max} lies past the pinned CAMB lmax; "
              f"every spectrum there is a log-log extrapolation of CAMB's last two "
              f"multipoles, which underestimates the damping tail badly (0.40x at the 2.5' "
              f"corner). Drop --camb_lmax to let CAMB cover the axis")
    return np.arange(ell_min, ell_max + 1, dtype = np.float64)


def forecast_from_1st_principles(nside, theta_pix, noise_level, is_sampled, param_ground,
                                 spectra = "lensed", step_fracs = None, l_knee = 0,
                                 beam_fwhm = 0, ell_min = None, ell_max = None,
                                 delta_ell = None, verbose = True,
                                 qe_response = DEFAULT_QE_RESPONSE,
                                 iterative_delens = False, camb_lmax = None,
                                 nphi_source = DEFAULT_NPHI_SOURCE,
                                 radial_mean = DEFAULT_RADIAL_MEAN,
                                 l_cutoff = DEFAULT_L_CUTOFF,
                                 phi_noise = None,
                                 measured_extend = DEFAULT_MEASURED_EXTEND,
                                 delensed_source = DEFAULT_DELENSED_SOURCE,
                                 delensed_dir = None, delensed_shifted_dirs = None,
                                 cl_source = DEFAULT_CL_SOURCE):
    """The first-principles Fisher matrix for the sampled LCDM parameters.

    Arguments:
        nside, theta_pix:  the box. Used ONLY for f_sky with the default
                           nphi_source = "hu_okamoto"; with "covariance" they also set the
                           rfft grid the quadratic estimator is built on
        noise_level:       white map noise in uK-arcmin (l_knee, beam_fwhm as in noise_cls)
        is_sampled:        {param: bool}, which of PARAM_ORDER to forecast
        param_ground:      the fiducial cosmology, all five parameters
        spectra:           "lensed", "unlensed" or "delensed" - which CAMB TT spectrum is
                           the field block; see FIELD_SPECTRA
        step_fracs:        finite-difference steps as fractions of PARAM_SIGMA
        ell_min, ell_max:  the multipole range of BOTH blocks (default: all of CAMB's
                           shipped range; ell_max past it is covered by running CAMB
                           further, see camb_lmax_for_axis)
        delta_ell:         band width for binning (default: every multipole separately)
        qe_response:       which TT spectrum weights the quadratic estimator's response,
                           "unlensed" (default) or "gradient" - see QE_RESPONSE_SOURCES
        iterative_delens:  spectra = "delensed" only: iterate the per-L delensing fraction
                           against the reconstruction noise to a fixed point rather than
                           taking one quadratic-estimator pass
        camb_lmax:         how far CAMB is run. Defaults to covering the multipole axis, so
                           nothing in the sum is a power-law continuation; pass CAMB_LMAX to
                           reproduce the pre-2026-09-21 extrapolated numbers
        nphi_source:       where N_L^phiphi comes from, "hu_okamoto" (default, the analytic
                           quadrature), "covariance" (the box's QE matrix azimuthally
                           averaged) or "measured" (the empirical N_L^eff of map_joint's own
                           MAP reconstruction, which needs `phi_noise`) - see NPHI_SOURCES
        phi_noise:         path to a merge_phi_noise.py effective_phi_noise.npz. Required by
                           and only valid with nphi_source = "measured", and refused if it
                           was measured on a different box than this forecast runs on
        measured_extend:   "measured" only: how N_L^eff is continued outside the bands it was
                           measured in. "extrapolate" (default here) continues the measured
                           power law; "hold" clamps to the end values, which is what the
                           block forecasts do - see fisher_forecast.MEASURED_EXTEND_MODES
        radial_mean:       "covariance" only: the annulus average inside _radial_cl_profile,
                           "geometric" (default) or "arithmetic"
        l_cutoff:          "covariance" only: the isotropic high-l cutoff qe_noise_matrix
                           applies, as in fisher_forecast.forecast
        cl_source:         where the MODEL spectra (the fiducial C_l and every stencil
                           point) come from: "camb" (default, direct CAMB runs) or "grid"
                           (the 5D grid spline sample_lcdm's theta step uses) - see
                           fisher_forecast.CL_SOURCES. The reconstruction noise and Alens_L
                           stay direct CAMB at the fiducial point either way: they describe
                           the estimator, and the sampler's own QE norm comes from load_sim's
                           direct CAMB call too. "grid" cannot serve the CAMB delensed
                           spectrum (the grid has no partially lensed Cls)

    Returns (fisher, names, details) where details holds the axis, the spectra, the noises,
    the derivatives and the per-block Fisher matrices for plotting or inspection.
    """
    if spectra not in FIELD_SPECTRA:
        raise ValueError(f"spectra must be one of {FIELD_SPECTRA}, got {spectra!r}")
    if iterative_delens and spectra != "delensed":
        raise ValueError(f"iterative delensing only applies to spectra = 'delensed'; got "
                         f"{spectra!r}. The other modes have no delensing step to iterate.")
    if delensed_source not in DELENSED_SOURCES:
        raise ValueError(f"delensed_source must be one of {DELENSED_SOURCES}, got "
                         f"{delensed_source!r}")
    if cl_source not in CL_SOURCES:
        raise ValueError(f"cl_source must be one of {CL_SOURCES}, got {cl_source!r}")
    if cl_source == "grid" and spectra == "delensed" and delensed_source == "camb":
        raise ValueError(
            "cl_source = 'grid' cannot serve --spectra delensed with delensed_source = "
            "'camb': that spectrum is CAMB's get_partially_lensed_cls at a per-L Alens_L, "
            "which the 5D grid does not store. Use delensed_source = 'measured' (which only "
            "needs the grid's unlensed TT and phiphi) or cl_source = 'camb'.")

    #the empirical delensed spectrum: loaded and box-checked before any CAMB work
    delensing_ratio = None
    if delensed_source == "measured":
        if spectra != "delensed":
            raise ValueError(f"delensed_source = 'measured' only applies to "
                             f"spectra = 'delensed'; got {spectra!r}.")
        if delensed_dir is None:
            raise ValueError(
                "delensed_source = 'measured' needs the directory of per-realization "
                "delensed_spectra_*.npz files. Pass it as --delensed_dir <out_dir>, the "
                "directory sampling_chains/get_delensed_spectra.sh writes to.")
        if iterative_delens:
            raise ValueError(
                "iterative delensing cannot be combined with delensed_source = 'measured'. "
                "The iteration refines Alens_L against N_L to guess a delensed spectrum; a "
                "measured D(l) already IS the delensed spectrum this box produces, so there "
                "is nothing to iterate.")
        delensing_ratio = measured_delensing_ratio(
            delensed_dir, shifted_dirs = delensed_shifted_dirs, verbose = verbose)
        check_delensing_ratio(delensing_ratio, nside, theta_pix, noise_level, l_knee,
                              directory = delensed_dir)
    elif delensed_dir is not None:
        raise ValueError(f"delensed_dir was given but delensed_source is "
                         f"{delensed_source!r}, so the measurement would be parsed and then "
                         f"ignored. Pass delensed_source = 'measured' to use it.")

    #loaded and box-checked here rather than inside reconstruction, so a mismatched or
    #missing measurement fails before any CAMB work is done
    measured_phi_noise = None
    if phi_noise is not None:
        if nphi_source != "measured":
            raise ValueError(f"phi_noise was given but nphi_source is {nphi_source!r}, so "
                             f"the measurement would be parsed and then ignored. Pass "
                             f"nphi_source = 'measured' to use it.")
        measured_phi_noise = load_phi_noise(phi_noise)
        check_phi_noise(measured_phi_noise, nside, theta_pix, noise_level, l_knee,
                        path = phi_noise)

    names, steps, fracs = sampled_names_and_steps(is_sampled, param_ground, step_fracs)
    if delensing_ratio is not None:
        check_delensing_ratio_derivatives(delensing_ratio, param_ground, names,
                                          directory = delensed_dir)
    ells = multipole_axis(ell_min, ell_max, camb_lmax, verbose)
    camb_lmax = camb_lmax_for_axis(ells, camb_lmax)
    f_sky = sky_fraction(nside, theta_pix)
    mode_weights, bin_fn = band_binner(ells, f_sky, delta_ell)

    if verbose:
        print(f"First-principles Fisher forecast [{spectra}]: f_sky {f_sky:.4g} "
              f"(nside {nside}, theta_pix {theta_pix}'), {noise_level} uK-arcmin, "
              f"l_knee {l_knee}, beam {beam_fwhm}'")
        print(f"  summing ell {ells[0]:.0f}..{ells[-1]:.0f} in {len(mode_weights)} band(s) of "
              f"width {delta_ell if delta_ell is not None else 1}, "
              f"{2 * float(np.sum(mode_weights)):.0f} modes in all")
        print(f"  CAMB to l = {camb_lmax - 1}"
              + ("" if camb_lmax == CAMB_LMAX else f" (shipped range ends at {CAMB_ELL_MAX})"))
        print(f"  sampled: {names}, {2 * len(names) + 1} "
              + ("CAMB calls" if cl_source == "camb" else
                 "5D CAMB grid evaluations (model spectra; N_L is still direct CAMB)"))
        if cl_source == "grid" and ells[-1] > CAMB_ELL_MAX:
            print(f"  WARNING: the grid stops at l = {CAMB_ELL_MAX}; the model spectra on "
                  f"l {CAMB_ELL_MAX + 1}..{ells[-1]:.0f} are log-log extrapolations of its "
                  f"last two multipoles - what covar_matrix_from_cls gives the sampler, and "
                  f"0.40x too small at the 2.5' corner")

    noise, alens = reconstruction(ells, spectra, param_ground, noise_level, l_knee,
                                  beam_fwhm, nside, theta_pix, qe_response = qe_response,
                                  nphi_source = nphi_source, radial_mean = radial_mean,
                                  l_cutoff = l_cutoff,
                                  measured_phi_noise = measured_phi_noise,
                                  measured_extend = measured_extend,
                                  delensed_source = delensed_source,
                                  iterative_delens = iterative_delens,
                                  camb_lmax = camb_lmax, verbose = verbose)
    signal = signal_spectra(param_ground, ells, spectra, alens = alens,
                            camb_lmax = camb_lmax, delensing_ratio = delensing_ratio,
                            cl_source = cl_source)
    derivatives = spectrum_derivatives(names, steps, param_ground, ells, spectra,
                                       alens = alens, camb_lmax = camb_lmax,
                                       delensing_ratio = delensing_ratio,
                                       cl_source = cl_source)
    fisher, per_block = fisher_from_spectra(signal, noise, derivatives, mode_weights, bin_fn)

    if verbose:
        for name in names:
            print(f"    {name}: h = {steps[names.index(name)]:.6g} ({fracs[name]:g} sigma)")
        share = {block: np.sqrt(np.diag(matrix) / np.diag(fisher)) for block, matrix in per_block.items()}
        print("  sqrt(F_block / F_total) per parameter: "
              + ", ".join(f"{block} {np.array2string(value, precision = 2)}"
                          for block, value in share.items()))

    details = dict(ells = ells, signal = signal, noise = noise, derivatives = derivatives,
                   per_block = per_block, f_sky = f_sky, alens = alens,
                   camb_lmax = camb_lmax)
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
                        help = "which CAMB TT spectrum is the field block. 'lensed' "
                               "(default) is what the instrument measures; 'unlensed' is "
                               "the perfect-delensing marker; 'delensed' is CAMB's "
                               "partially lensed spectrum at Alens_L = N_L / (C_L + N_L), "
                               "with N_L this module's own analytic Hu & Okamoto N^(0)")
    parser.add_argument("--iterative_delens", action = "store_true",
                        help = "--spectra delensed: iterate the per-L delensing fraction "
                               "against the reconstruction noise to a fixed point instead "
                               "of taking one quadratic-estimator pass. Better delensing "
                               "quiets the estimator, which improves delensing; this is "
                               "what iterative lensing reconstruction does")
    parser.add_argument("--camb_lmax", type = int, default = None,
                        help = f"how far CAMB is run (default: enough to cover --ell_max, "
                               f"floored at {CAMB_LMAX}). Pinning it below --ell_max puts "
                               f"the remainder back on a log-log extrapolation of CAMB's "
                               f"last two multipoles, which is 0.40x too small at the 2.5' "
                               f"corner; pass {CAMB_LMAX} to reproduce the pre-2026-09-21 "
                               f"numbers")
    parser.add_argument("--ell_min", type = int, default = None,
                        help = f"first multipole of both blocks (default {CAMB_ELL_MIN})")
    parser.add_argument("--ell_max", type = int, default = None,
                        help = f"last multipole of both blocks (default {CAMB_ELL_MAX}, the "
                               f"end of CAMB's shipped range). It may go past that: CAMB is "
                               f"simply run further to cover it, so the box's Nyquist "
                               f"pi / pix_width (4320 at 2.5') or its corner (6109) counts "
                               f"the high-ell modes the sampler's grid carries")
    parser.add_argument("--delta_ell", type = int, default = None,
                        help = "bin the spectra into bands of this width before contracting "
                               "(default: every multipole separately)")
    add_qe_response_argument(parser)
    parser.add_argument("--nphi_source", choices = NPHI_SOURCES,
                        default = DEFAULT_NPHI_SOURCE,
                        help = "where the lensing reconstruction noise N_L^phiphi comes "
                               "from. 'hu_okamoto' (default) is the analytic flat-sky "
                               "N^(0) quadrature over this analysis's own multipole range "
                               "- isotropic, no box, and able to reach past any grid. "
                               "'covariance' is the box's own N_phi matrix (the same "
                               "scalar_quadratic_estimate the sampler's G uses, and what "
                               "the block forecasts contract) azimuthally averaged onto "
                               "the 1D axis, which is the apples-to-apples comparison "
                               "against them - but it makes --nside / --theta_pix set the "
                               "estimator's grid rather than only f_sky, and the profile "
                               "is extrapolated outside the grid's range. 'measured' is the "
                               "empirical N_L^eff of map_joint's own MAP reconstruction, "
                               "the only source describing the estimator this codebase "
                               "actually runs; it needs --phi_noise")
    add_phi_noise_argument(parser)
    parser.add_argument("--measured_extend", choices = MEASURED_EXTEND_MODES,
                        default = DEFAULT_MEASURED_EXTEND,
                        help = "--nphi_source measured: how N_L^eff is continued outside "
                               "the |L| bands it was measured in (the box's fundamental to "
                               "its corner). 'extrapolate' (the default here) continues the "
                               "measured power law in log-log, the same rule every other "
                               "spectrum in this module gets. 'hold' clamps to the nearest "
                               "measured value, which is what the block forecasts do - "
                               "N_eff is steep, so the continuation below the lowest band "
                               "has a long lever arm and drives Alens_L across exactly the "
                               "multipoles carrying most of C_phi. Inside the bands the two "
                               "are identical")
    parser.add_argument("--delensed_source", choices = DELENSED_SOURCES,
                        default = DEFAULT_DELENSED_SOURCE,
                        help = "--spectra delensed: where the delensed TT spectrum comes "
                               "from. 'camb' (default) is get_partially_lensed_cls at "
                               "Alens_L = N_L / (C_L + N_L). 'measured' uses the EMPIRICAL "
                               "ratio D(l) = <C^delensed / C^unlensed> this box actually "
                               "produces (lense_flow + map_joint + inverse lensing), "
                               "applied to CAMB's unlensed spectrum - no Alens_L and no "
                               "CAMB lensing calculation. N_L then only sets the phi "
                               "block's noise and plays no part in the delensing. Needs "
                               "--delensed_dir. NOTE D is frozen in theta, which is a "
                               "stronger assumption than --transfer_function makes")
    parser.add_argument("--delensed_dir", type = str, default = None,
                        help = "--delensed_source measured: the directory of per-realization "
                               "delensed_spectra_*.npz files written by "
                               "sampling_chains/get_delensed_spectra.sh. Refused if it was "
                               "measured on a different box than this forecast runs on")
    parser.add_argument("--delensed_shifted_dirs", nargs = "*", default = None,
                        help = "--delensed_source measured: directories measuring D(l) at "
                               "cosmologies displaced from --delensed_dir's, ONE parameter "
                               "each, on the SAME seeds (common random numbers) and with "
                               "the reconstruction frozen at the reference cosmology. D is "
                               "then evaluated per stencil point as D_0 + (theta - theta_0) "
                               "dD/dtheta, so the delensed derivative carries both terms of "
                               "d(D C_u)/dtheta = D dC_u/dtheta + C_u dD/dtheta. WITHOUT "
                               "these the second term is zero, D cancels out of the Fisher "
                               "integrand and this mode reduces to --spectra unlensed")
    add_radial_mean_argument(parser)
    parser.add_argument("--cl_source", choices = CL_SOURCES, default = DEFAULT_CL_SOURCE,
                        help = "where the model spectra at the fiducial point and every "
                               "finite-difference stencil point come from. 'camb' "
                               "(default) runs CAMB directly; 'grid' reads the 5D CAMB grid "
                               "spline sample_lcdm's theta step evaluates, so the forecast "
                               "uses the same dC/dtheta as the chains. The reconstruction "
                               "noise stays direct CAMB either way. Memory-mapped: a few "
                               "MB per stencil point, never the ~2 GB tables")
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
            qe_response = args.qe_response, iterative_delens = args.iterative_delens,
            camb_lmax = args.camb_lmax, nphi_source = args.nphi_source,
            radial_mean = args.radial_mean, phi_noise = args.phi_noise,
            measured_extend = args.measured_extend,
            delensed_source = args.delensed_source, delensed_dir = args.delensed_dir,
            delensed_shifted_dirs = args.delensed_shifted_dirs,
            cl_source = args.cl_source)

    fisher, names, details = run()
    covariance = covariance_from_fisher(fisher, names)
    report_sigmas(covariance, names, args.spectra, METHOD)

    if args.stability:
        step_stability(lambda fracs: run(step_fracs = fracs, verbose = False)[:2], METHOD)

    ells = details["ells"]
    subtitle = (f"{args.spectra} TT + phiphi  |  f_sky {details['f_sky']:.3g}, "
                f"{args.noise:g} uK-arcmin  |  ell {ells[0]:.0f}..{ells[-1]:.0f}"
                + (f", bands of {args.delta_ell}" if args.delta_ell is not None
                   else ", every multipole")
                + f"  |  N_L {args.nphi_source}"
                + (" | iterated" if args.iterative_delens else "")
                + (" | Cl from 5D grid" if args.cl_source == "grid" else ""))
    #np.savez cannot store None, so an unset delta_ell is recorded as NaN, and a "delensed"
    #run's per-L delensing fraction is stored while the other modes record an empty array
    config = run_config(args.spectra, args, names,
                        ell_min = ells[0], ell_max = ells[-1],
                        delta_ell = args.delta_ell if args.delta_ell is not None else np.nan,
                        iterative_delens = args.iterative_delens,
                        nphi_source = args.nphi_source,
                        measured_extend = args.measured_extend,
                        delensed_source = args.delensed_source,
                        cl_source = args.cl_source,
                        #np.savez cannot store None
                        delensed_dir = args.delensed_dir or "",
                        alens = (details["alens"] if details["alens"] is not None
                                 else np.zeros(0)),
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
