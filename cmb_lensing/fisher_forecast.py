"""Gaussian Fisher forecast for the LCDM parameters on sample_lcdm.py's flat-sky box.

This answers "what is the best a power-spectrum analysis of this data set could do?" -
the baseline that sample_joint should beat, since the Gibbs sampler works with the full
hierarchical likelihood (phi ~ N(0, Cphi), f ~ N(0, Cf), d = M B L(phi) f + n) and
therefore also sees the lensing-induced non-Gaussianity that a 2-point analysis discards.

This module is the ORIGINAL covariance-block forecast, "blocks", computed on the SAME
flat-sky rfft grid the sampler runs on (gen_ell_grid at the given nside / theta_pix), not
with a full-sky (2l+1) sum, so the box geometry, the finite ell_min = 2*pi/L and the mode
count all match the real run.

    F_ij = 1/2 * sum_k w_k * Tr[C_k^-1 dC_k/dtheta_i C_k^-1 dC_k/dtheta_j]

which for the temperature-only case used here collapses to

    F_ij = 1/2 * sum_k w_k * dln(C_k)/dtheta_i * dln(C_k)/dtheta_j

This is the standard zero-mean Gaussian Fisher F = 1/2 Tr[C^-1 C,i C^-1 C,j] (see e.g.
Tegmark, Taylor & Heavens 1997 for the general form), specialized to a covariance that is
diagonal in the Fourier basis. The sum-over-rfft-entries form above is NOT quoted from
anywhere - the literature writes either the bare trace or the full-sky f_sky (2l+1) / 2
sum; it is a transcription of the same result onto this codebase's array layout, and it
is derived from scratch in the "Derivation" section below. The weights w_k are
util.get_fourier_weights - the number of INDEPENDENT REAL degrees of freedom carried by
each rfft entry (2 for bulk columns, 1 for the two self-conjugate columns kx = 0 and
kx = Nyquist, whose entries are constrained by F[-n] = conj(F[n]) within the column).
They sum to nside**2, the true DOF count of the real map, and they are the same weights
util.primal_dot / util.primal_log_det use to build statistics.logpdf - so this Fisher is
the second moment of the derivative of the very likelihood the codebase evaluates.

THE OTHER FORECASTING METHODS each live in their own module and build on this one - they
import the CAMB cache, the covariance-block stencil, the finite-difference helpers and the
plotting / output machinery from here, and each has its own CLI:

  fisher_forecast_from_cls.py           "cls"       the BANDPOWER Fisher dmu^T Cov(mu_hat)^-1 dmu,
                                                    with the lensing-induced non-Gaussian
                                                    covariance (lensing_covariance.py)
  fisher_forecast_full_sky.py           "full_sky"  the textbook sum_l (2l+1)/2 f_sky Tr[...] over
                                                    a 1D multipole axis (the grid's own distinct
                                                    ells, or CAMB's integer ones)
  fisher_forecast_from_logpdf.py        "logpdf"    < -d2 statistics.logpdf > over prior draws of
                                                    (d, f, phi) - Louis's first term
  fisher_forecast_from_mixed_logpdf.py  "mixed"     the same for statistics.mixed_logpdf at a
                                                    fixed mixed (f, phi) pair, fanned out over
                                                    slurm jobs by sampling_chains/mixed_hessian.sh

They all consume the same 2 * n_sampled + 1 CAMB runs (camb_cls_at_params memoizes them,
so running two methods in one process costs one set of runs) and every one writes to
cmb_lensing/fisher_output/ under its own file suffix, so no method overwrites another's
figures.

Derivation (scalar case, in this codebase's normalization). statistics.logpdf is built
from primal_dot and primal_log_det, i.e. schematically

    log L = -1/2 sum_k w_k [ |x_k|^2 / (nside^2 C_k) + ln C_k ] + const

with E|x_k|^2 = nside^2 C_k (see the field_from_covar_single_key comment). Then
E[d log L / dtheta_i] = 0, and with Var(|x_k|^2) = (nside^2 C_k)^2 * (2 / w_k) - variance
v^2 for a free complex mode, 2 v^2 for a real one - the outer product E[d_i log L d_j log L]
collapses to

    F_ij = 1/4 sum_k w_k^2 (2 / w_k) dln(C_k)/dtheta_i dln(C_k)/dtheta_j
         = 1/2 sum_k w_k dln(C_k)/dtheta_i dln(C_k)/dtheta_j

VALIDATION. Unlensed scalar TT is exactly proportional to As = exp(logA) * 1e-10, so as
the noise goes to zero dln(Cf)/dlogA is identically 1 and the "ceiling" f_unlensed block
ALONE must return exactly 1/2 * sum_k w_k = (nside^2 - 1) / 2 (the [0, 0] origin is
excluded). At nside 64 that is 2047.5; at noise_level 1e-4 the code returns 2047.63. That
single number pins the weights, the factor of 1/2, the origin handling and the finite
difference simultaneously - re-run it after any change to this module. NOTE it is now a
PER-BLOCK check: pass one block at a time to _fisher_from_blocks, because every mode
carries a second block and the f block itself carries C_n at any realistic noise level
(F_logA,logA falls to 2022.3 at 5 uK-arcmin). The phi block has no such closed form -
Cl_phiphi is not exactly proportional to As, and N_phi stays finite as the instrumental
noise vanishes (a cosmic-variance-limited QE is still a noisy QE), so it lands at 85.3
rather than 2047.5.

N_phi is the estimator's N^(0), and --qe_response picks which TT spectrum its RESPONSE is
built from: "unlensed" (the DEFAULT, Hu & Okamoto's original expression and what the sampler
itself uses) or "gradient", the lensed temperature-gradient spectrum C_l^(T grad T) of Lewis,
Challinor & Hanson (2011), which is the correct weight and resums the N^(2) bias into the
normalization. See QE_RESPONSE_SOURCES for the measured difference and gradient_cls_at_params
for the spectrum. Either way this is a forecast-only knob: the sampler's own QE norm
(simulate.py, where it only preconditions G and the phi mass matrix) is untouched, so nothing
here changes load_sim, map_joint or sample_lcdm - see qe_noise_matrix for the two ways this
module's N_phi departs from the sampler's. Higher-order biases beyond N^(2) - N^(1) above
all, which is linear in C_phi and so a property of the signal rather than of the instrument -
are NOT included; on a per-mode phi_hat = phi + n model they have no slot, and they would
only enter a forecast that treated the phi block as a bandpower measurement of C_L^phiphi.

Four spectra modes, bracketing the answer. Every mode carries a lensing block
C_phi + N_phi, with N_phi the temperature quadratic-estimate reconstruction noise from
qe_noise_matrix (frozen at the fiducial cosmology - see covariance_blocks):

  "lensed"   (DEFAULT) C_TT = (mask * beam)^2 * Cl_lensed + N, plus C_PP = Cl_phiphi +
             N_phi. The achievable baseline: the Fisher of a surrogate Gaussian model
             matched to the second moment of the observed map, plus the lensing power
             spectrum measured by a QE. The map-level data is NOT actually a GRF draw of
             Cl_lensed (load_sim draws unlensed f and phi and lenses them), so the TT
             surrogate is wrong above second order; the PP block restores the part of the
             trispectrum a standard lensing-reconstruction analysis recovers, but the two
             blocks are added as if independent, which double counts the lensing signal
             that is already in Cl_lensed. Treat it as a two-point analysis, not a bound.
  "unlensed" C_TT = (mask * beam)^2 * Cl_unlensed + N, plus the same PP block. The
             perfect-delensing heuristic. NOT a Cramer-Rao bound on the real data - no
             estimator acting on d is guaranteed to reach it, and delensing correlates the
             noise (L^-1 n is not isotropic). Quote it as a marker, not a bound.
  "ceiling"  Complete-data Fisher: f and phi measured, each to within its noise. Because
             noise_cls / mask / beam carry no cosmology, log p(d | f, phi) is
             theta-INDEPENDENT, so given the fields the data says nothing about theta and
             all the information sits in the two priors:
                 F = F[Cf_unlensed + C_n] + F[Cphi + N_phi]
             The f block deliberately carries no beam or mask: it is what a measurement of
             the UNLENSED field would carry, not what the instrument returns. With the
             noise terms in it is no longer a strict upper bound on the other two modes -
             at nside 64 / 5 uK-arcmin it now sits only ~1.0-1.2x above "lensed" per
             parameter, where the noiseless version was far looser.
  "delensed" The complete-data f block paying for IMPERFECT delensing: C_TT^delensed + C_n,
             where C_TT^delensed is CAMB's own lensing calculation (get_partially_lensed_cls,
             the non-perturbative correlation-function method) run with the lensing
             potential scaled per multipole by Alens_L = N_L / (C_L^phiphi + N_L) - the
             fraction of the lensing power a Wiener-filtered reconstruction leaves in the
             map. Alens_L = 1 everywhere is the TT block of "lensed", Alens_L = 0 is
             "ceiling"; a real reconstruction sits between, and because Alens_L is
             per-L the acoustic-scale TT multipoles are delensed by exactly the phi modes
             that lens them (the scalar-alpha version this replaced put "delensed" on top
             of "ceiling"). N_L comes from either the box's QE matrix azimuthally averaged
             ("covariance", the default) or the analytic Hu & Okamoto N^(0) quadrature
             ("hu_okamoto"), see frozen_reconstruction; Alens_L is frozen at the fiducial
             cosmology and CAMB re-delenses at every stencil point. Because the residual
             lensing that stays in the map tracks C_phi(theta), the f block picks up
             lensing-derived information on omch2 (it can land BELOW "ceiling" there)
             that the phi block counts again - the same double counting "lensed" has,
             scaled down by Alens_L. Measured at nside 64 / 2.5' / 5 uK-arcmin: omch2
             0.0042 (ceiling 0.0069), theta 0.0062 (0.0052), logA 0.043 (0.027).

WHAT THIS IS NOT. None of these is the marginal Fisher of p(d | theta) = the integral of
p(d, f, phi | theta) over the fields, which is what sample_joint actually targets; the
chains themselves are the measurement of that. Measured against the 50-map chains at
nside 128 / 2.5' / 5 uK, the posterior sits BETWEEN ceiling and lensed in all three
sigmas and all three correlations, and for omch2 - theta_MC_100 the bracket straddles
zero, so do not read the sign of any ceiling entry with |r| < ~0.2.

Finite differences. Cl derivatives are central differences with step
h_p = FD_STEP_FRAC[p] * PARAM_SIGMA[p], and the per-parameter constants in FD_STEP_FRAC
are the tuning knob. The Cls come from DIRECT CAMB calls, never from the 5D grid spline:
the spline's ~1e-3 lnCl interpolation error is a DETERMINISTIC function of theta, so it
does not average away, and differencing it over the grid's coarse node spacing can
produce curvature comparable to the signal. Only 2 * n_sampled + 1 CAMB runs are needed.
Always confirm the answer is step-independent - "--stability" recomputes at 2h and reports
the fractional change in the forecast sigmas.

Parameters with is_sampled[name] = False are held FIXED, not marginalized over; F is
n_sampled x n_sampled.

Usage:
    python -m cmb_lensing.fisher_forecast                       #nside 128, T-only, 2.5 uK-arcmin
    python -m cmb_lensing.fisher_forecast --spectra lensed
    python -m cmb_lensing.fisher_forecast --spectra delensed --iterative_delens
    python -m cmb_lensing.fisher_forecast --nside 64 --noise 5.0 --stability

Writes into cmb_lensing/fisher_output/, tagged with this method's SUFFIX "_from_blocks":
    fisher_matrix_from_blocks.png       F_ij
    covariance_matrix_from_blocks.png   F^-1 and the marginalized sigmas
    correlation_matrix_from_blocks.png  F^-1 normalized to unit diagonal
    fisher_from_blocks.npz              all three plus the run configuration
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import camb
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.simulate import (camb_parameters, _extract_all_cls, dl2cl,
                                  covar_matrix_from_cls, noise_cls, get_beam, get_mask,
                                  scalar_quadratic_estimate, get_g_matrix_lcdm, get_d_tt_matrix)
from cmb_lensing.constants import (DEFAULT_MAX_ELL, DEFAULT_A_LENSE, DEFAULT_K_PIVOT,
                                   DEFAULT_MNU, DEFAULT_TAUREIO, NPHI_FAC)
from cmb_lensing.precompute_camb_1d import (PARAM_ORDER, GROUND_TRUTH, PARAM_SIGMA,
                                            CAMB_LMAX)


#this module's method, the suffix on every file it writes, and the one-line reminder of
#the contraction stamped into every figure title. Each sibling module defines its own three
METHOD = "blocks"
SUFFIX = "_from_blocks"
LABEL = r"blocks:  $F_{ij} = \frac{1}{2}\sum_k w_k\,\partial_i\ln C_k\,\partial_j\ln C_k$"

#finite-difference step for each parameter, as a FRACTION of PARAM_SIGMA. these are the
#tuning knobs: raise them if the forecast is noisy (CAMB's own ~1e-3 lnCl accuracy floor
#leaking into the difference), lower them if it drifts with step size (genuine curvature
#of Cl(theta) breaking the linear approximation). always verify with --stability
FD_STEP_FRAC = {
    "theta_MC_100": 0.05,
    "logA": 0.05,
    "ns": 0.05,
    "ombh2": 0.05,
    "omch2": 0.05,
}

#the order every output of this module is laid out in: the Fisher / covariance rows and
#columns, the heatmap axes, the sigma bar chart, the printed report and fisher.npz. it is
#chain_analysis.py's ground_truth_values order, NOT PARAM_ORDER, so that the matrices
#printed / plotted here can be read side by side with the ones chain_analysis produces
#from the chains (for the usual sampled trio: top / middle / bottom = omch2,
#theta_MC_100, logA). PARAM_ORDER still defines the CAMB-facing naming everywhere else
OUTPUT_PARAM_ORDER = ["omch2", "ombh2", "ns", "theta_MC_100", "logA"]
assert sorted(OUTPUT_PARAM_ORDER) == sorted(PARAM_ORDER)

SPECTRA_MODES = ("lensed", "unlensed", "ceiling", "delensed")

#CAMB is deterministic in the parameters, and the stencil re-queries the fiducial point
#for every parameter, so memoize whole cls dicts across calls (and across the sibling
#modules, which all import this one dict)
_CAMB_CACHE = {}
_CAMB_RESULTS_CACHE = {}
_GRADIENT_CACHE = {}

#where the quadratic-estimator reconstruction noise N_phi comes from - both the 2D matrix
#the phi block C_phi + N_phi carries in EVERY spectra mode (qe_noise_grid) and, for
#spectra = "delensed", the 1D N_L behind the per-L delensing fraction Alens_L (qe_noise_cl):
#  "covariance"  (DEFAULT) qe_noise_matrix - the box's scalar_quadratic_estimate, i.e. the
#                same FFT-convolution QE norm the sampler's G matrix and phi mass matrix are
#                built from (without the sampler's NPHI_FAC preconditioning factor, and with
#                the gradient-spectrum response this module uses everywhere). It is
#                anisotropic on the square box; _radial_cl_profile azimuthally averages it
#                onto CAMB's integer multipoles when a 1D N_L is needed
#  "hu_okamoto"  qe_noise_spectrum - the flat-sky Hu & Okamoto (2002) N^(0) integral in polar
#                coordinates: isotropic, no square domain, no aliasing, but integrated over
#                the SAME multipole range the box carries - qe_noise_cl caps its axis at
#                grid_max_ell, so the two sources differ in the shape of the domain and not
#                in how far it reaches. covar_matrix_from_cls puts it on the rfft grid for
#                the phi block exactly the way C_phi gets there
NPHI_SOURCES = ("covariance", "hu_okamoto")

#which TT spectrum the quadratic estimator's RESPONSE f(l, l') is built from - the `cf_tt`
#argument of simulate.scalar_quadratic_estimate, and the `cl_tt_response` of
#qe_noise_spectrum. Independent of NPHI_SOURCES: it applies to both of them:
#  "unlensed"  (DEFAULT) C_l^TT unlensed, as in Hu & Okamoto's original expression and as
#              the sampler's own QE norm is built (simulate.py / sample_lcdm.py), so this
#              reproduces every number this module produced before 2026-09-15
#  "gradient"  C_l^(T grad T), CAMB's lensed temperature-gradient cross spectrum
#              (gradient_cls_at_params). The response is a derivative of the LENSED
#              temperature with respect to the deflection, so this is the correct weight
#              (Lewis, Challinor & Hanson 2011); it resums the N^(2) bias into the
#              normalization. Costs one extra ~5 s CAMB post-processing call at the
#              fiducial point
#Measured at nside 64 / 5' / 5 uK, T-only: "gradient" raises N_phi by 0.4-2.6% (covariance
#source) and up to 19% at L < 100 (hu_okamoto - low L is sensitive to the response's SLOPE,
#through the near-cancellation of the two L . l terms, and the unlensed spectrum's acoustic
#peaks are not smoothed). The marginalized sigmas move < 0.35% (blocks) and < 0.03%
#(1st_principles), because C_phi dominates N_phi over the modes carrying the Fisher weight
#on this box; expect more on a noisier or finer box where N_phi is not subdominant
QE_RESPONSE_SOURCES = ("unlensed", "gradient")
DEFAULT_QE_RESPONSE = "unlensed"

#resolution of the Hu & Okamoto quadrature (qe_noise_spectrum): the number of multipoles L
#in EACH of its two sets (log-spaced for the power-law rise at low L, linear for the upturn
#near the top of the range) before interpolating onto the full axis, and the number of
#azimuthal angles in the inner integral. Measured at 5 uK-arcmin on 2..3999: doubling num_l
#moves N_L by < 2e-3 below L = 3800 and < 6e-3 in the last hundred multipoles before the
#cutoff, doubling num_angles by < 1.5e-3 anywhere
QE_NUM_L = 80
QE_NUM_ANGLES = 256


def output_dir():
    #mirrors sample_lcdm.py's sample_lcdm_output convention: a directory next to the
    #package modules, so the same relative layout works on the laptop and the cluster
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisher_output")


# ── CAMB ──────────────────────────────────────────────────────────────────

def _camb_key(params):
    return tuple(round(float(params[name]), 12) for name in PARAM_ORDER)


def camb_results_at_params(params):
    """(CAMBparams, CAMBdata) at an arbitrary 5-parameter point, memoized.

    Set up through simulate.camb_parameters, so the run is identical to load_sim's: H0 solved
    from cosmomc_theta, r = 0, mnu = 0.06, tau = 0.05, nt = 0, k_pivot = 0.05, Alens = 1,
    non-linear lensing. The results object is kept (not just its spectra) because
    delensed_cls_at_params needs get_partially_lensed_cls on it.
    """
    key = _camb_key(params)
    if key in _CAMB_RESULTS_CACHE:
        return _CAMB_RESULTS_CACHE[key]

    pars = camb_parameters(None, params["ombh2"], params["omch2"],
                           params["theta_MC_100"] / 100, 0.0, DEFAULT_MNU, DEFAULT_TAUREIO,
                           np.exp(params["logA"]) * 1e-10, 0, params["ns"], CAMB_LMAX,
                           DEFAULT_K_PIVOT, DEFAULT_A_LENSE)
    try:
        results = camb.get_results(pars)
    except Exception as error:
        #load_sim's callback would swallow this and hand the sampler NaNs to reject; here
        #it would silently poison the whole matrix, so fail loudly with the point
        raise RuntimeError(
            f"CAMB failed at {dict((k, float(params[k])) for k in PARAM_ORDER)}: {error}. "
            f"check the stencil point is reachable by CAMB (H0 is solved from "
            f"cosmomc_theta inside DEFAULT_THETA_H0_RANGE), or shrink FD_STEP_FRAC "
            f"for that parameter.") from error

    _CAMB_RESULTS_CACHE[key] = (pars, results)
    return pars, results


def camb_cls_at_params(params):
    """Full cls dict from one CAMB run at an arbitrary 5-parameter point.

    precompute_camb_1d.camb_cls_at only moves a single parameter off GROUND_TRUTH and
    only returns (TT, PP); the Fisher stencil needs an arbitrary point and the lensed
    spectra too. The spectra are exactly what simulate._run_camb extracts (same calls,
    same units, same _extract_all_cls), landing on ells 2..CAMB_LMAX-1 with no
    extrapolation - the same support as load_sim's data-map path.
    """
    key = _camb_key(params)
    if key in _CAMB_CACHE:
        return _CAMB_CACHE[key]

    pars, results = camb_results_at_params(params)
    power_spectra = results.get_cmb_power_spectra(pars, lmax = CAMB_LMAX - 1,
                                                  CMB_unit = "muK")
    lens_potential = results.get_lens_potential_cls(lmax = CAMB_LMAX - 1,
                                                    CMB_unit = "muK")[:, :2]
    cls = _extract_all_cls(jnp.asarray(power_spectra["unlensed_scalar"]),
                           jnp.asarray(power_spectra["tensor"]),
                           jnp.asarray(power_spectra["total"]),
                           jnp.asarray(lens_potential), CAMB_LMAX, CAMB_LMAX)

    if not bool(jnp.all(jnp.isfinite(cls["total_TT"]))):
        raise RuntimeError(
            f"CAMB returned non-finite Cls at {dict((k, float(params[k])) for k in PARAM_ORDER)}. "
            f"check the stencil point is reachable by CAMB (H0 is solved from "
            f"cosmomc_theta inside DEFAULT_THETA_H0_RANGE), or shrink FD_STEP_FRAC "
            f"for that parameter."
        )

    _CAMB_CACHE[key] = cls
    return cls


def gradient_cls_at_params(params):
    """C_l^(T grad T): the lensed TEMPERATURE-GRADIENT cross spectrum, on CAMB's 2..CAMB_LMAX-1.

    This is the spectrum the TT quadratic estimator's RESPONSE should be built from, and the
    only reason this module needs a CAMB product load_sim never asks for. The lensing
    response f(l, l') that normalizes the estimator is a derivative of the lensed temperature
    with respect to the deflection, so the correct weight is the correlation of T with its own
    gradient, NOT the unlensed spectrum: Lewis, Challinor & Hanson (2011, arXiv:1101.2234,
    appendix C) show that using C_l^unlensed there leaves an O(few %) misnormalization whose
    tail is exactly the N^(2) bias, while C_l^(T grad T) resums it. CAMB computes it in the
    same flat-sky approximation this whole module works in (get_lensed_gradient_cls, column 0
    of 8), so no extra approximation is introduced by using it here.

    Accuracy note. CAMB does not extrapolate inside this routine, so the top of the range is
    limited by the run's own max_l (4200 for CAMB_LMAX = 4000, set by CAMB's lens margin).
    Measured against a max_l = 5700 run at GROUND_TRUTH: agreement is 8e-5 at l = 1000, 1.4e-3
    at l = 3000, 3e-3 at l = 3500, degrading to 3% in the last ~100 multipoles. Those
    multipoles carry little of the estimator's weight and the error is far below the
    correction being made, but raise CAMB_LMAX if the QE is ever pushed to the very top.

    Memoized separately from camb_cls_at_params rather than folded into it: the call costs
    ~5 s (its own flat-sky lensing pass), and the reconstruction noise is only ever needed at
    the FIDUCIAL cosmology, so the 2 * n_sampled stencil points must not pay for it.
    """
    key = _camb_key(params)
    if key in _GRADIENT_CACHE:
        return _GRADIENT_CACHE[key]

    _, results = camb_results_at_params(params)
    #_scale_cls leaves this in the l(l+1)/2pi convention with raw_cl = False, i.e. exactly
    #what dl2cl's default (non-phi, non-tphi) branch inverts, and "muK" matches the TT units
    gradient = results.get_lensed_gradient_cls(lmax = CAMB_LMAX - 1, CMB_unit = "muK")
    spectrum = dl2cl(jnp.asarray(gradient[:, 0]), CAMB_LMAX, CAMB_LMAX)

    if not bool(jnp.all(jnp.isfinite(spectrum))) or bool(jnp.any(spectrum <= 0)):
        raise RuntimeError(
            f"CAMB returned a non-positive or non-finite T-grad-T spectrum at "
            f"{dict((k, float(params[k])) for k in PARAM_ORDER)}. The quadratic estimator's "
            f"response needs it positive on the whole axis; check the point is reachable by "
            f"CAMB and that DoLensing is on.")

    _GRADIENT_CACHE[key] = spectrum
    return spectrum


def cls_with_qe_response(params, qe_response = DEFAULT_QE_RESPONSE):
    """camb_cls_at_params, carrying whatever extra spectrum `qe_response` needs.

    "unlensed" (the default) needs nothing beyond scalar_TT, so the dict comes back
    unchanged and no extra CAMB work is done at all; "gradient" attaches "gradient_TT".
    Only the fiducial point is built this way (see gradient_cls_at_params for why the
    gradient spectrum is not folded into camb_cls_at_params), which is enough because the
    reconstruction noise is frozen there across the whole stencil.
    """
    _check_qe_response(qe_response)
    cls = dict(camb_cls_at_params(params))
    if qe_response == "gradient":
        cls["gradient_TT"] = gradient_cls_at_params(params)
    return cls


def _check_qe_response(qe_response):
    if qe_response not in QE_RESPONSE_SOURCES:
        raise ValueError(f"qe_response must be one of {QE_RESPONSE_SOURCES}, got "
                         f"{qe_response!r}")


def qe_response_cl(cls, qe_response = DEFAULT_QE_RESPONSE):
    """The TT spectrum the quadratic estimator's response is built from - see
    QE_RESPONSE_SOURCES.

    The "gradient" branch RAISES rather than falling back to scalar_TT when the key is
    absent, mirroring the "delensed_TT" guard in covariance_blocks: silently reverting to
    the unlensed spectrum is precisely the misnormalization that mode exists to remove.
    """
    _check_qe_response(qe_response)
    if qe_response == "unlensed":
        return cls["scalar_TT"]
    if "gradient_TT" not in cls:
        raise ValueError("qe_response = 'gradient' needs cls['gradient_TT']; build the dict "
                         "with cls_with_qe_response(params, 'gradient') rather than "
                         "camb_cls_at_params (see gradient_cls_at_params)")
    return cls["gradient_TT"]


def load_transfer_function(path):
    """The empirical delensing transfer function R(l) written by merge_delensed_spectra.py.

    R(l) = C_l^delensed measured on THIS box with lense_flow and a map_joint reconstruction,
    divided by CAMB's get_partially_lensed_cls at the same frozen Alens_L - see
    cmb_lensing/delensed_spectrum.py for how it is measured and why it is needed. Returns the
    whole npz as a dict; `band_ells` and `transfer` are what apply_transfer_function uses and
    the rest is the configuration it was measured at.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no transfer function at {path}. Produce one by running "
            f"sampling_chains/get_delensed_spectra.sh and then "
            f"merge_delensed_spectra.py --spectra_dir <its out_dir>.")
    merged = dict(np.load(path, allow_pickle = True))
    missing = [key for key in ("band_ells", "transfer") if key not in merged]
    if missing:
        raise ValueError(f"{path} is missing {missing}; it does not look like a "
                         f"merge_delensed_spectra.py product")
    return merged


def check_transfer_function(merged, nside, theta_pix, noise_level, l_knee, path = ""):
    """Refuse a transfer function measured on a different box than the forecast is running on.

    R absorbs the box's own resolution, periodicity and reconstruction noise, so it is only
    meaningful at the configuration it was measured at - applying an nside 64 / 5' R to an
    nside 128 / 2.5' forecast would import the wrong correction entirely. Mirrors
    load_hessian_directory's refusal to average mixed configurations.
    """
    measured = dict(nside = int(merged["nside"]), theta_pix = float(merged["theta_pix"]),
                    noise_level = float(merged["noise_level"]),
                    l_knee = float(merged["l_knee"]))
    wanted = dict(nside = int(nside), theta_pix = float(theta_pix),
                  noise_level = float(noise_level), l_knee = float(l_knee))
    if measured != wanted:
        raise ValueError(
            f"the transfer function {path} was measured at {measured} but this forecast runs "
            f"at {wanted}. R absorbs the box's resolution and reconstruction noise, so it "
            f"cannot be carried across - re-run get_delensed_spectra.sh at this box.")


def apply_transfer_function(cl, cl_ells, merged, verbose = False):
    """Multiply a delensed spectrum by R(l), interpolated onto its multipole axis.

    R is measured in |l| bands spanning exactly the modes the rfft grid carries (its
    fundamental to its corner mode), while `cl_ells` is CAMB's full 2..CAMB_LMAX-1 axis, so
    the ends have to be handled. R is a RATIO near one, not a power law, so the log-log
    continuation covar_matrix_from_cls uses everywhere else would be meaningless here: it is
    interpolated LINEARLY in l and held at the nearest measured band value outside the
    measured range (np.interp's own clamping). Below the box's fundamental and above its
    corner there is no measurement to extrapolate from and the forecast's own grid carries no
    modes either, so the held value is never contracted - but it is held rather than set to
    one so that nothing discontinuous enters if a caller ever does reach out there.
    """
    band_ells = np.asarray(merged["band_ells"])
    transfer = np.asarray(merged["transfer"])
    factor = np.interp(np.asarray(cl_ells), band_ells, transfer)
    if verbose:
        print(f"  transfer function R(l): {len(band_ells)} bands over "
              f"l = {band_ells[0]:.0f}..{band_ells[-1]:.0f}, R in "
              f"{np.min(transfer):.4f}..{np.max(transfer):.4f} "
              f"({int(merged['n_realizations'])} realizations); held constant outside")
    return jnp.asarray(cl) * jnp.asarray(factor)


def delensed_cls_at_params(params, alens, transfer = None):
    """camb_cls_at_params plus "delensed_TT": CAMB lensing the unlensed spectra at `params`
    with C_L^phiphi scaled by the per-L `alens` (zero-based in L, Alens_L = 1 meaning no
    delensing at that L). CAMB's get_partially_lensed_cls reruns its full non-perturbative
    correlation-function lensing with the scaled potential - NOT a linear interpolation
    between the unlensed and lensed spectra, which it differs from at the percent level -
    so the result is the delensed TT a Wiener-filtered reconstruction with residual
    fraction Alens_L would leave. Cheap (~0.02 s) once the results object is cached.

    Multipoles past the end of `alens` (CAMB lenses with C_L^phiphi up to Params.max_l,
    which exceeds CAMB_LMAX by CAMB's lens margin) carry its last value.

    `transfer` optionally applies the empirical transfer function R(l) (load_transfer_function)
    so that the spectrum becomes the one the BOX's own lense_flow and map_joint produce rather
    than CAMB's. It multiplies at every stencil point, which is the whole construction: R is
    measured once at the fiducial cosmology and CAMB supplies the theta dependence, so the
    derivative this feeds the stencil is R * dC_CAMB/dtheta. Holding R fixed in theta is an
    assumption - sampling_chains/compare_transfer_functions.py is what tests it.
    """
    cls = dict(camb_cls_at_params(params))
    pars, results = camb_results_at_params(params)
    scaling = np.full(pars.max_l + 1, float(alens[-1]))
    scaling[:len(alens)] = alens
    partial = results.get_partially_lensed_cls(scaling, lmax = CAMB_LMAX - 1,
                                               CMB_unit = "muK")
    delensed = dl2cl(jnp.asarray(partial[:, 0]), CAMB_LMAX, CAMB_LMAX)
    if transfer is not None:
        ells = jnp.arange(2, 2 + delensed.shape[0]).astype(jnp.float64)
        delensed = apply_transfer_function(delensed, ells, transfer)
    cls["delensed_TT"] = delensed
    return cls


def load_sim_cosmology(param_ground):
    """param_ground translated into load_sim's keyword names (cosmomc_theta, As).

    The realization-based methods build their data sets with load_sim at the fiducial
    point; this is the one place the theta_MC_100 -> cosmomc_theta / logA -> As renaming
    for that call is spelled out.
    """
    camb_kwargs = dict(param_ground)
    camb_kwargs["cosmomc_theta"] = camb_kwargs.pop("theta_MC_100") / 100
    camb_kwargs["As"] = float(np.exp(camb_kwargs.pop("logA")) * 1e-10)
    return camb_kwargs


# ── Covariance blocks on the flat-sky grid ────────────────────────────────

def _instrument_matrices(nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                         l_cutoff):
    """(noise, mask, beam) on the rfft grid - the theta-INDEPENDENT part of every block."""
    lmax_prime = min(DEFAULT_MAX_ELL, CAMB_LMAX)
    n_tt, _, _, _ = noise_cls(lmax_prime, noise_level, beam_fwhm = beam_fwhm,
                              l_knee = l_knee)
    noise = covar_matrix_from_cls(nside, pix_width, ell_grid,
                                  jnp.arange(2, lmax_prime).astype(jnp.float64), n_tt,
                                  origin_value = 0)
    beam = get_beam(nside, pix_width, ell_grid, lmax_prime, beam_fwhm = beam_fwhm)
    mask = jnp.ones_like(get_mask(l_cutoff, nside, pix_width, ell_grid))
    return noise, mask, beam


def qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                    l_cutoff, filter_tt = None, qe_response = DEFAULT_QE_RESPONSE):
    """N_phi: the scalar quadratic-estimate reconstruction noise at this cosmology.

    Built with the same simulate.scalar_quadratic_estimate the sampler's G matrix uses, and
    the same mask and beam, but differing from sample_lcdm.py's call in TWO deliberate ways:

    1. The RESPONSE spectrum (scalar_quadratic_estimate's `cf_tt`, the `ct` that weights both
       legs of the estimator) is whichever `qe_response` selects - see QE_RESPONSE_SOURCES.
       It defaults to "unlensed", the same spectrum the sampler passes, so the default call
       differs from the sampler's only in point 2; "gradient" is a FORECAST-ONLY choice the
       sampler never takes, since there the QE norm is a preconditioner for G and the phi
       mass matrix, where a few-percent misnormalization is harmless and exact agreement with
       CMBLensing.jl matters more.
    2. No division by NPHI_FAC: that factor is a preconditioning choice inherited from
       CMBLensing.jl's Nphi_fac (it cancels inside G), not part of the estimator's noise, so
       this matrix is the physical N^(0) = A_L the phi block and the delensing fraction need.

    Temperature-only, matching everything else in this module. `filter_tt` replaces the lensed
    TT in the estimator's inverse-variance filter - the total signal power the map actually
    carries. Iterative delensing passes the delensed spectrum here, since a delensed map has
    less variance and so a quieter estimator.
    """
    ells = jnp.arange(2, 2 + cls["total_TT"].shape[0]).astype(jnp.float64)
    noise, mask, beam = _instrument_matrices(nside, pix_width, ell_grid, noise_level,
                                             l_knee, beam_fwhm, l_cutoff)
    filter_tt = cls["total_TT"] if filter_tt is None else filter_tt

    def covar(cl):
        return covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl,
                                     origin_value = 0)

    return scalar_quadratic_estimate(noise, covar(qe_response_cl(cls, qe_response)),
                                     covar(filter_tt),
                                     mask, beam, pix_width) #/ NPHI_FAC


def prior_covariances(params, nside, pix_width, ell_grid):
    """(C_f, C_phi) on the rfft grid at one cosmology - the only theta-dependent inputs
    statistics.logpdf takes. Bare priors: no C_n on the field, no N_phi on the lensing."""
    cls = camb_cls_at_params(params)
    ells = jnp.arange(2, 2 + cls["scalar_TT"].shape[0]).astype(jnp.float64)
    phi_ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)
    cf = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cls["scalar_TT"],
                               origin_value = 0)
    cphi = covar_matrix_from_cls(nside, pix_width, ell_grid, phi_ells, cls["phi"],
                                 origin_value = 0)
    return cf, cphi


def delensing_efficiency(cphi, nphi):
    """alpha = the C_phi-weighted mean Wiener factor C_phi / (C_phi + N_phi).

    How much of the lensing potential the reconstruction actually recovers. alpha -> 1 is a
    perfect reconstruction (delensing removes all the lensing), alpha -> 0 is no lensing
    information at all.

    A SCALAR efficiency is an approximation: properly, which phi modes delens a given TT
    scale is set by the lensing mode-coupling kernel, so alpha should be scale dependent.
    It matters little here because C_phi is so steeply red that low-ell modes dominate any
    weighting - measured 0.937 restricting to l_phi < 200 versus 0.934 over all modes at
    nside 128 / 2.5' / 5 uK. That is a fact about this box, not a general one.
    """
    good = cphi > 0
    rho_squared = jnp.where(good, cphi / (cphi + nphi), 0.0)
    weight = jnp.where(good, cphi, 0.0)
    return float(jnp.sum(weight * rho_squared) / jnp.sum(weight))


def interpolate_spectrum(ells, source_ells, spectrum):
    """A positive spectrum moved onto `ells` by log-log interpolation, extrapolated as a
    power law past either end of its support - the same jnp.interp call
    covar_matrix_from_cls uses to put a CAMB spectrum on the rfft grid."""
    return np.asarray(jnp.exp(jnp.interp(jnp.log(ells), jnp.log(source_ells),
                                         jnp.log(spectrum), left = "extrapolate",
                                         right = "extrapolate")))


def grid_max_ell(ell_grid):
    """The largest multipole the rfft grid carries: the corner mode sqrt(2) pi / pix_width.

    The pixel size alone sets it - nside fixes the fundamental 2 pi / L (the SMALLEST mode
    and the spacing), not the corner - so it is 3055 at 5' and 6109 at 2.5' for every
    nside. Past it the box holds no modes at all, which is what caps the temperature range
    the quadratic estimator can integrate over in qe_noise_cl.
    """
    return float(np.max(np.asarray(ell_grid)))


def qe_noise_spectrum(ells, cl_tt_response, cl_tt_lensed, noise_tt,
                      num_l = QE_NUM_L, num_angles = QE_NUM_ANGLES):
    """N_L^phiphi: the TT quadratic estimator's N^(0) noise, Hu & Okamoto (2002), flat sky.

        N_L = [ int d^2l / (2 pi)^2  f(l, L - l)^2 / (2 Ct_l Ct_|L - l|) ]^-1

        f(l, l') = C_l^grad (L . l) + C_l'^grad (L . l')      the lensing response
        Ct_l     = C_l^lensed + N_l                           the observed spectrum

    `cl_tt_response` is the spectrum the response f is built from. Every caller in this
    package passes the lensed temperature-gradient spectrum C_l^(T grad T)
    (gradient_cls_at_params) rather than the unlensed C_l^TT that appears in Hu & Okamoto's
    original expression: the two differ by the lensing correction to the response, and using
    the gradient spectrum resums the N^(2) bias into the normalization (Lewis, Challinor &
    Hanson 2011). It is a keyword only in the sense that passing the unlensed spectrum still
    reproduces the textbook formula, which is how the pre-2026-09-15 numbers were made.

    `ells` is the multipole axis the temperature analysis uses, and the estimator is built
    from exactly those multipoles: both legs l and L - l must lie within [ells[0], ells[-1]],
    which is what restricting the analysis to that range does to the reconstruction. All
    three input spectra are given on `ells`; `noise_tt` is the beam-deconvolved map noise
    (noise_cls's convention), so `cl_tt_lensed` is the bare sky spectrum.

    The integral is done in polar coordinates (|l|, angle between l and L), so the domain
    is the isotropic annulus the analysis actually covers - not a periodic square - and
    N_L is a function of |L| alone. With L along the x axis,

        |L - l| = sqrt(L^2 + l^2 - 2 L l cos(angle)),   L . l = L l cos(angle),
        L . (L - l) = L^2 - L l cos(angle)

    The |l| integral is a trapezoidal sum over the integer multipoles of `ells`; the angle
    integral a uniform sum over `num_angles` points. It is evaluated at `num_l` log-spaced
    plus `num_l` linearly spaced L across the axis (dense where N_L is a steep power law and
    where it turns up towards the top of the TT range) and log-log interpolated onto every
    multipole of `ells`, since N_L is smooth and the full integer axis would only repeat
    that work.

    Cross-checked against scalar_quadratic_estimate: on the square periodic box the latter
    reproduces this same integral (restricted to the box's square of modes, with the FFT's
    periodic wrap of L - l) to 0.5%, so the two agree in normalization. Note that the
    sampler carries that matrix divided by NPHI_FAC = 2, a preconditioning choice inherited
    from CMBLensing.jl's Nphi_fac; this returns the physical N^(0), undivided.

    Returns N_L on `ells`, in the same units as C_L^phiphi (dimensionless, ell^4 convention).
    """
    ell_min, ell_max = ells[0], ells[-1]
    angles = np.linspace(0.0, 2 * np.pi, num_angles, endpoint = False)
    cos_angle = np.cos(angles)[np.newaxis, :]
    l = ells[:, np.newaxis]

    response_l = cl_tt_response[:, np.newaxis]
    observed_l = (cl_tt_lensed + noise_tt)[:, np.newaxis]

    evaluated_at = np.unique(np.concatenate([np.geomspace(ell_min, ell_max, num_l),
                                             np.linspace(ell_min, ell_max, num_l)]))
    inverse_noise = np.empty(len(evaluated_at))
    for index, big_l in enumerate(evaluated_at):
        l_prime = np.sqrt(big_l**2 + l**2 - 2 * big_l * l * cos_angle)
        inside = (l_prime >= ell_min) & (l_prime <= ell_max)
        l_prime_safe = np.where(inside, l_prime, ell_min)

        response_prime = interpolate_spectrum(l_prime_safe, ells, cl_tt_response)
        observed_prime = interpolate_spectrum(l_prime_safe, ells, cl_tt_lensed + noise_tt)

        big_l_dot_l = big_l * l * cos_angle
        big_l_dot_l_prime = big_l**2 - big_l_dot_l
        response = response_l * big_l_dot_l + response_prime * big_l_dot_l_prime
        integrand = np.where(inside, response**2 / (2 * observed_l * observed_prime), 0.0)

        #d^2l = l dl dangle: trapezoid over l (the integer axis), uniform sum over angle
        angle_integral = np.sum(integrand, axis = 1) * (2 * np.pi / num_angles)
        inverse_noise[index] = np.trapezoid(ells * angle_integral, ells) / (2 * np.pi)**2

    return interpolate_spectrum(ells, evaluated_at, 1.0 / inverse_noise)


def _radial_cl_profile(matrix, ell_grid, weights, pix_width, ell_axis, label = "N_phi"):
    """Azimuthally average an rfft-grid operator back onto a 1D ell axis, in Cl units.

    N_phi from qe_noise_matrix only exists as a 2D array: scalar_quadratic_estimate is
    built from FFT convolutions on the grid. Every input to it is isotropic, so N_phi is a
    function of |l| up to the grid's own anisotropy - which is NOT small: the square rfft
    box's integration domain makes the QE norm vary by 1.6x around an annulus at L = 540
    and 5.7x at L = 2090 (nside 64 / 5'). Averaging over annuli gives the isotropic
    function; qe_noise_spectrum is the box-free alternative.

    Two conventions are undone here. covar_matrix_from_cls divides every spectrum by
    pix_width**2 and covariance_blocks adds nphi straight onto such a matrix, so nphi
    carries the same 1/pix_width**2; multiplying it back gives an honest Cl. The annuli are
    one fundamental mode (2 pi / L) wide - finer bins would leave gaps - and every entry is
    weighted by its real-DOF weight w_k, the same weighting _fisher_from_blocks uses.

    The profile is then log-log interpolated onto `ell_axis` with the SAME "extrapolate at
    both ends" convention covar_matrix_from_cls uses, so the two paths treat the edges of
    the support identically. A warning fires when `ell_axis` reaches past the grid's largest
    mode, where that extrapolation is doing real work: N_phi rises steeply there and a
    power-law continuation of it is a guess, not a measurement.
    """
    ells = np.asarray(ell_grid).ravel()
    values = np.asarray(matrix).ravel() * pix_width**2
    dof = np.asarray(jnp.real(weights)).ravel()

    #the [0, 0] origin carries no ell, and the nan_to_num inside the quadratic estimator
    #leaves exact zeros wherever its norm diverged - neither can enter a log-space average
    usable = (ells > 0) & (values > 0) & np.isfinite(values)
    if not np.any(usable):
        raise RuntimeError(f"{label} has no positive entries on the rfft grid, so it "
                           f"cannot be profiled onto a 1D ell axis")

    fundamental = 2 * np.pi / (np.asarray(ell_grid).shape[0] * pix_width)
    index = np.floor(ells / fundamental).astype(int)
    n_bin = int(index[usable].max()) + 1

    total = np.bincount(index[usable], weights = (dof * values)[usable], minlength = n_bin)
    norm = np.bincount(index[usable], weights = dof[usable], minlength = n_bin)
    centre = np.bincount(index[usable], weights = (dof * ells)[usable], minlength = n_bin)

    filled = norm > 0
    profile_ell = centre[filled] / norm[filled]
    profile = total[filled] / norm[filled]

    #past the grid's largest mode there is no measurement left to interpolate between, so
    #that - not the last annulus's centre - is where the extrapolation starts doing work
    grid_max = float(np.max(ells[usable]))
    if np.max(ell_axis) > grid_max:
        print(f"  WARNING: {label} is log-log extrapolated past l = {grid_max:.0f} (the "
              f"grid's largest mode) out to l = {np.max(ell_axis):.0f}")

    return interpolate_spectrum(ell_axis, profile_ell, profile)


def qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, l_cutoff,
                nphi_source, filter_tt = None, qe_response = DEFAULT_QE_RESPONSE):
    """N_L^phiphi on CAMB's integer multipoles 2..CAMB_LMAX-1, from either NPHI_SOURCES.

    `filter_tt` is the TT spectrum the estimator's filter sees (default: the lensed one);
    iterative delensing passes the delensed spectrum. `qe_response` picks the spectrum the
    estimator's RESPONSE is built from, independently of `nphi_source` and applying to both
    of them (QE_RESPONSE_SOURCES). See NPHI_SOURCES for what the two sources are and how
    they differ.

    BOTH sources are capped at the box: "covariance" because scalar_quadratic_estimate only
    ever sees the rfft grid, "hu_okamoto" because its integration axis is cut at
    grid_max_ell. The returned axis is CAMB's either way, with the same log-log
    continuation past the grid's largest mode and the same warning.
    """
    if nphi_source not in NPHI_SOURCES:
        raise ValueError(f"nphi_source must be one of {NPHI_SOURCES}, got {nphi_source!r}")
    ells = np.arange(2, 2 + cls["total_TT"].shape[0], dtype = np.float64)
    filter_tt = cls["total_TT"] if filter_tt is None else filter_tt

    if nphi_source == "covariance":
        matrix = qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                                 beam_fwhm, l_cutoff, filter_tt = filter_tt,
                                 qe_response = qe_response)
        weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                                   (nside, nside // 2 + 1))
        return _radial_cl_profile(matrix, ell_grid, weights, pix_width, ells,
                                  label = "N_phi (covariance source)")

    lmax_prime = min(DEFAULT_MAX_ELL, CAMB_LMAX)
    n_tt, _, _, _ = noise_cls(lmax_prime, noise_level, beam_fwhm = beam_fwhm,
                              l_knee = l_knee)

    #the estimator can only use temperature modes the box actually holds, so its multipole
    #axis stops at the grid's largest mode rather than at DEFAULT_MAX_ELL: both legs l and
    #|L - l| are then restricted to the range this resolution measures, which is what the
    #covariance source gets for free by living on the grid. The cap bites only when the
    #corner mode falls short of CAMB's range (5' and coarser, where it cuts 3999 -> 3055);
    #at 2.5' the grid reaches 6109 and the full CAMB axis survives untouched
    analysis_ells = ells[ells <= grid_max_ell(ell_grid)]
    if len(analysis_ells) < 2:
        raise ValueError(f"the rfft grid's largest mode l = {grid_max_ell(ell_grid):.0f} "
                         f"leaves fewer than two multipoles for the quadratic estimator - "
                         f"this box is too coarse for a Hu & Okamoto N_phi")

    noise_tt = interpolate_spectrum(analysis_ells,
                                    np.arange(2, lmax_prime, dtype = np.float64),
                                    np.asarray(n_tt))
    #analysis_ells is a PREFIX of ells (both start at 2 and step by one), so the spectra
    #restrict by slicing - no interpolation and no resampling error on the way in
    cut = len(analysis_ells)
    nphi = qe_noise_spectrum(analysis_ells,
                             np.asarray(qe_response_cl(cls, qe_response))[:cut],
                             np.asarray(filter_tt)[:cut], noise_tt)
    if cut == len(ells):
        return nphi

    #the returned axis stays CAMB's, so a capped N_L is continued as a power law past the
    #cap - the same continuation, past the same mode, with the same warning that
    #_radial_cl_profile gives the covariance source
    print(f"  WARNING: N_phi (hu_okamoto source) is log-log extrapolated past l = "
          f"{analysis_ells[-1]:.0f} (the grid's largest mode) out to l = {ells[-1]:.0f}")
    return interpolate_spectrum(ells, analysis_ells, nphi)


def qe_noise_grid(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, l_cutoff,
                  nphi_source, filter_tt = None, qe_response = DEFAULT_QE_RESPONSE):
    """The 2D N_phi the phi block adds to C_phi on the rfft grid, from either NPHI_SOURCES.

    "covariance" is qe_noise_matrix itself. "hu_okamoto" takes the 1D N_L from
    qe_noise_cl and puts it on the grid through covar_matrix_from_cls - the same log-log
    interpolation onto ell_grid, the same 1/pix_width**2 rescale and the same zeroed
    origin C_phi goes through - so the two add on an identical footing and the block is
    a function of |l| alone, which the box matrix is not. `filter_tt` and `qe_response` as in
    qe_noise_cl.
    """
    if nphi_source == "covariance":
        return qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                               beam_fwhm, l_cutoff, filter_tt = filter_tt,
                               qe_response = qe_response)
    nphi_cl = qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                          l_cutoff, nphi_source, filter_tt = filter_tt,
                          qe_response = qe_response)
    ells = jnp.arange(2, 2 + len(nphi_cl)).astype(jnp.float64)
    return covar_matrix_from_cls(nside, pix_width, ell_grid, ells, jnp.asarray(nphi_cl),
                                 origin_value = 0)


def delensing_alens(cls, nphi_cl):
    """Alens_L = N_L / (C_L^phiphi + N_L): the fraction of the lensing power left in the map
    after delensing with the Wiener-filtered reconstruction C / (C + N) phi_hat.

    Zero-based in L (CAMB's convention for get_partially_lensed_cls) with Alens = 1 at the
    monopole and dipole, where C_L^phiphi is zero anyway. Alens_L -> 0 is perfect
    delensing, Alens_L -> 1 no delensing at that L.
    """
    cphi = np.asarray(cls["phi"])
    nphi = np.asarray(nphi_cl)
    alens = np.ones(CAMB_LMAX)
    alens[2:] = nphi / (cphi + nphi)
    return alens


def iterative_delensing(param_ground, cls, nside, pix_width, ell_grid, noise_level,
                        l_knee, beam_fwhm, l_cutoff, nphi_source, max_iterations = 25,
                        tolerance = 1e-6, verbose = True,
                        qe_response = DEFAULT_QE_RESPONSE):
    """Fixed-point iteration between the per-L delensing fraction and the QE noise.

    The one-shot construction is circular in a way it does not admit: it estimates phi with
    a quadratic estimator whose noise is set by the FULLY LENSED map, then uses that phi to
    delens. But once you have delensed, the map carries less lensing power, so the estimator
    filter sees a smaller total variance and its noise drops - which lets you delens better,
    which lowers the noise again. Iterating that to a fixed point is what real iterative
    lensing reconstruction does (Hirata & Seljak 2003; Smith et al. 2012), and it is the
    closest a covariance-only calculation gets to the MAP / Gibbs reconstruction map_joint
    actually performs.

        Alens_0        = 1                          (no delensing: the filter sees the lensed map)
        N_n            = QE noise with the delensed TT_n as the filter variance
        Alens_{n+1}    = N_n / (C_phi + N_n)
        TT_{n+1}       = CAMB partially lensed at Alens_{n+1}   (delensed_cls_at_params)

    The first iteration reproduces the one-shot answer exactly, so this can only improve on
    it. Returns (alens, nphi_cl, delensed_tt, converged, iterations).
    """
    alens = np.ones(CAMB_LMAX)
    filter_tt = cls["total_TT"]
    converged = False
    for iteration in range(1, max_iterations + 1):
        nphi_cl = qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                              beam_fwhm, l_cutoff, nphi_source, filter_tt = filter_tt,
                              qe_response = qe_response)
        updated = delensing_alens(cls, nphi_cl)
        shift = float(np.max(np.abs(updated - alens)))
        alens = updated
        filter_tt = delensed_cls_at_params(param_ground, alens)["delensed_TT"]
        if verbose:
            print(f"    iteration {iteration}: mean efficiency "
                  f"{delensing_efficiency(cls['phi'], nphi_cl):.5f} (max Alens shift "
                  f"{shift:.2e})")
        if shift < tolerance:
            converged = True
            break

    if not converged and verbose:
        print(f"    WARNING: delensing iteration did not converge in {max_iterations} "
              f"steps (last shift {shift:.2e}); the reported Alens_L is the last iterate")

    return alens, nphi_cl, filter_tt, converged, iteration


def _covar_linear(nside, pix_width, ell_grid, ells, cls):
    """covar_matrix_from_cls for a SIGNED spectrum: linear rather than log-log interpolation
    onto the rfft grid, zero beyond the last ell (a sign-changing quantity has no power-law
    continuation), the [0, 0] origin set to zero and the same 1/pix_width**2 rescale."""
    flat = jnp.interp(ell_grid.flatten(), ells, cls, left = 0.0, right = 0.0)
    result = flat.reshape(ell_grid.shape).at[0, 0].set(0.0)
    return result / pix_width**2


def covariance_blocks(cls, spectra, nside, pix_width, ell_grid,
                      noise_level, l_knee, beam_fwhm, l_cutoff, nphi):
    """The theta-dependent covariance block(s) whose Fisher information we are counting.

    Every block is built through the same covar_matrix_from_cls the sampler uses, so the
    signal and noise share a normalization (the 1/pix_width**2 rescale). The trace formula
    is invariant under any theta-independent rescaling of C, so only that relative
    normalization matters.

    `nphi` is the QE reconstruction noise from qe_noise_grid (the box matrix or the
    Hu & Okamoto spectrum on the grid, per nphi_source), evaluated ONCE at the fiducial
    cosmology and passed in frozen. Freezing it is deliberate: N_phi is a property
    of the estimator and the experiment, not of the model being constrained, so only the
    signal should carry theta dependence - the same convention that makes the instrumental
    C_n theta-independent, and the same thing the sampler does (its QE norm stays at the
    param_init cosmology for the whole chain). Letting it move would credit the forecast
    with information from dN_phi/dtheta, which no C_l^phiphi likelihood actually uses.
    """
    ells = jnp.arange(2, 2 + cls["total_TT"].shape[0]).astype(jnp.float64)
    phi_ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)

    def covar(cl, ell_axis):
        return covar_matrix_from_cls(nside, pix_width, ell_grid, ell_axis, cl,
                                     origin_value = 0)

    noise, mask, beam = _instrument_matrices(nside, pix_width, ell_grid, noise_level,
                                             l_knee, beam_fwhm, l_cutoff)

    #C_phi + N_phi: a QE lensing reconstruction, i.e. phi measured to within the
    #quadratic estimator's noise rather than known exactly
    phi_block = covar(cls["phi"], phi_ells) + nphi

    if spectra == "delensed":
        #the complete-data f block, but paying for imperfect delensing: CAMB's own lensing
        #of this cosmology's unlensed spectrum with the potential scaled by the frozen
        #per-L Alens_L (delensed_cls_at_params). Alens_L is evaluated ONCE at the
        #fiducial cosmology, for the same reason nphi is frozen: it is a property of the
        #reconstruction, not of the model being constrained, so letting it move would
        #credit the forecast with dN/dtheta information no real analysis uses
        if "delensed_TT" not in cls:
            raise ValueError("spectra = 'delensed' needs cls['delensed_TT']; build the dict "
                             "with delensed_cls_at_params at the frozen Alens_L, as "
                             "covariance_stencil does")
        return {"f_delensed": covar(cls["delensed_TT"], ells) + noise,
                "phi": phi_block}

    if spectra in ("unlensed", "ceiling"):
        #conditional on (f, phi) the data term carries no theta dependence at all, so the
        #complete-data information is the two priors - but the fields are known only to
        #within the noise, so C_f picks up C_n and C_phi picks up N_phi. No beam or mask
        #on the f block: this is what a measurement of the UNLENSED field would carry,
        #not what the instrument returns
        return {"f_unlensed": covar(cls["scalar_TT"], ells) + noise, 
                "phi": phi_block}

    #load_sim forms data = mask * beam * lensed + noise, so the observed covariance
    #carries (mask * beam)**2 on the signal only
    # source = "total_TT" if spectra == "lensed" else "scalar_TT"
    # return {"TT": (mask * beam)**2 * covar(cls[source], ells) + noise}
    if spectra == "lensed":
        #the CAMB-predicted T-phi cross spectrum (the ISW-lensing correlation), the
        #off-diagonal of the per-mode (T, phi) covariance. It changes sign at high ell
        #(61 negative entries near ell ~1100 at GROUND_TRUTH, at the 1e-16 level), so it
        #cannot go through covar's log-log interpolation; it is interpolated LINEARLY
        #onto the grid with the same origin and 1/pix_width**2 conventions. NOTE every
        #block contraction in this module treats a key as an independent auto-spectrum,
        #which is not what a cross spectrum is - it only means something inside a
        #per-mode (TT, TP; TP, PP) matrix
        return {"TT": covar(cls["scalar_TT"], ells) + noise,
                "TP": _covar_linear(nside, pix_width, ell_grid, ells, cls["TP"]),
                "PP": phi_block}


# ── Finite-difference stencils ────────────────────────────────────────────

def sampled_names_and_steps(is_sampled, param_ground, step_fracs):
    """(names, steps, fracs) - the sampled subset in OUTPUT_PARAM_ORDER and its FD steps.

    Shared by every forecasting module so that no two methods can disagree about which
    parameters are being forecast, in what order, or at what step size.
    """
    missing = [name for name in PARAM_ORDER if name not in param_ground]
    if missing:
        raise ValueError(f"param_ground is missing {missing}; all five of {PARAM_ORDER} "
                         f"are needed because the unsampled ones still fix the cosmology")

    names = [name for name in OUTPUT_PARAM_ORDER if is_sampled.get(name, False)]
    if not names:
        raise ValueError("is_sampled selects no parameters - nothing to forecast")

    fracs = dict(FD_STEP_FRAC)
    if step_fracs is not None:
        fracs.update(step_fracs)
    return names, [fracs[name] * PARAM_SIGMA[name] for name in names], fracs


def stencil_offsets(n_param):
    """Central-difference stencil points as offset tuples in units of h.

    (0,)*n for the centre, +/-1 in one slot for the first derivatives, and the four
    (+/-1, +/-1) corners of each (i, j) pair for the mixed second derivatives. That is
    1 + 2n + 4*n*(n-1)/2 points - 19 for the usual three parameters. This is the stencil
    the realization-based methods (logpdf, mixed) evaluate a log-density on.
    """
    points = [tuple([0] * n_param)]
    for i in range(n_param):
        for sign in (+1, -1):
            offset = [0] * n_param
            offset[i] = sign
            points.append(tuple(offset))
    for i in range(n_param):
        for j in range(i + 1, n_param):
            for sign_i in (+1, -1):
                for sign_j in (+1, -1):
                    offset = [0] * n_param
                    offset[i] = sign_i
                    offset[j] = sign_j
                    points.append(tuple(offset))
    return points


def hessian_from_stencil(values, steps):
    """Second-derivative matrix from log-density values on stencil_offsets' points.

    `values` maps each offset tuple to the scalar evaluated there, `steps` are the h_i:
        H_ii = [l(+i) - 2 l(0) + l(-i)] / h_i^2
        H_ij = [l(+i+j) - l(+i-j) - l(-i+j) + l(-i-j)] / (4 h_i h_j)
    Returns H itself (the caller negates it for a Fisher).
    """
    n_param = len(steps)
    centre = tuple([0] * n_param)

    def unit(i, sign):
        offset = [0] * n_param
        offset[i] = sign
        return tuple(offset)

    def corner(i, j, sign_i, sign_j):
        offset = [0] * n_param
        offset[i] = sign_i
        offset[j] = sign_j
        return tuple(offset)

    hessian = np.zeros((n_param, n_param))
    for i in range(n_param):
        hessian[i, i] = (values[unit(i, +1)] - 2 * values[centre]
                         + values[unit(i, -1)]) / steps[i]**2
    for i in range(n_param):
        for j in range(i + 1, n_param):
            hessian[i, j] = (values[corner(i, j, +1, +1)] - values[corner(i, j, +1, -1)]
                             - values[corner(i, j, -1, +1)] + values[corner(i, j, -1, -1)]
                             ) / (4 * steps[i] * steps[j])
            hessian[j, i] = hessian[i, j]
    return hessian


def frozen_reconstruction(cls_fid, spectra, nside, pix_width, ell_grid, noise_level,
                          l_knee, beam_fwhm, l_cutoff, iterative_delens, verbose,
                          nphi_source = "covariance", param_ground = None,
                          qe_response = DEFAULT_QE_RESPONSE):
    """(N_phi, Alens_L) at the FIDUCIAL cosmology - the reconstruction every stencil freezes.

    Both are properties of the estimator and the experiment rather than of the model being
    constrained, so they are evaluated once and held fixed across the whole
    finite-difference stencil; covariance_blocks explains why letting them move would credit
    the forecast with dN/dtheta information no real analysis uses.

    `N_phi` is the phi block's 2D noise from `nphi_source` (qe_noise_grid: the box's QE
    matrix, or the Hu & Okamoto spectrum interpolated onto the grid). `Alens_L` is None
    unless spectra = "delensed"; then it is delensing_alens of the 1D N_L from the same
    source, one-shot or iterated (iterative_delensing), and `param_ground` is needed so
    delensed_cls_at_params can find CAMB's results object. After an iteration N_phi is
    recomputed with the final delensed spectrum as the filter, so the phi block sees the
    quieter estimator too.
    """
    if iterative_delens and spectra != "delensed":
        raise ValueError(f"iterative delensing only applies to spectra = 'delensed'; got "
                         f"{spectra!r}. The other modes have no delensing step to iterate.")
    if nphi_source not in NPHI_SOURCES:
        raise ValueError(f"nphi_source must be one of {NPHI_SOURCES}, got {nphi_source!r}")
    _check_qe_response(qe_response)

    nphi = qe_noise_grid(cls_fid, nside, pix_width, ell_grid, noise_level, l_knee,
                         beam_fwhm, l_cutoff, nphi_source, qe_response = qe_response)
    if spectra != "delensed":
        return nphi, None
    if param_ground is None:
        raise ValueError("spectra = 'delensed' needs param_ground, so the delensed spectra "
                         "can be built from CAMB's results object at that point")

    if iterative_delens:
        if verbose:
            print(f"  iterating per-L delensing fraction against the {nphi_source} "
                  f"reconstruction noise:")
        alens, nphi_cl, delensed_tt, converged, iterations = iterative_delensing(
            param_ground, cls_fid, nside, pix_width, ell_grid, noise_level, l_knee,
            beam_fwhm, l_cutoff, nphi_source, verbose = verbose,
            qe_response = qe_response)
        #the iteration quiets BOTH the delensing and the phi block's estimator
        nphi = qe_noise_grid(cls_fid, nside, pix_width, ell_grid, noise_level, l_knee,
                             beam_fwhm, l_cutoff, nphi_source, filter_tt = delensed_tt,
                             qe_response = qe_response)
        if verbose:
            print(f"  converged {converged} after {iterations} iterations")
    else:
        nphi_cl = qe_noise_cl(cls_fid, nside, pix_width, ell_grid, noise_level, l_knee,
                              beam_fwhm, l_cutoff, nphi_source, qe_response = qe_response)
        alens = delensing_alens(cls_fid, nphi_cl)

    if verbose:
        alpha = delensing_efficiency(cls_fid["phi"], nphi_cl)
        samples = ", ".join(f"L={L} {alens[L]:.2f}" for L in (50, 100, 300, 1000, 2000))
        print(f"  delensing [{nphi_source} N_L]: C_phi-weighted mean efficiency "
              f"{alpha:.3f}; residual fraction Alens_L at {samples}")
    return nphi, alens


def covariance_stencil(nside, theta_pix, noise_level, is_sampled, param_ground, spectra,
                       step_fracs, l_knee, beam_fwhm, l_cutoff, verbose,
                       iterative_delens = False, nphi_source = "covariance",
                       qe_response = DEFAULT_QE_RESPONSE, transfer_function = None):
    """The CAMB / covariance-block stencil the flat-sky-grid forecasts are built from.

    Returns (names, steps, weights, blocks_fid, blocks_plus, blocks_minus): the sampled
    parameter names, their FD steps, the real-DOF weights w_k, and the block dicts at the
    fiducial point and at theta +/- h for every sampled parameter. "blocks" contracts these
    directly; fisher_forecast_from_cls reads the same stencil so its bandpower Fisher can
    never drift from this one through a differently-built stencil, and
    fisher_forecast_full_sky's default ell_source azimuthally reduces these very blocks.
    """
    if spectra not in SPECTRA_MODES:
        raise ValueError(f"spectra must be one of {SPECTRA_MODES}, got {spectra!r}")

    #R corrects the DELENSED spectrum specifically - there is no such measurement for the
    #other modes, and silently ignoring it would hide a mis-specified run
    transfer = None
    if transfer_function is not None:
        if spectra != "delensed":
            raise ValueError(f"a transfer function only applies to spectra = 'delensed'; got "
                             f"{spectra!r}. R is measured as the ratio of the box's delensed "
                             f"spectrum to CAMB's, so there is nothing for it to correct in "
                             f"the other modes.")
        transfer = load_transfer_function(transfer_function)
        check_transfer_function(transfer, nside, theta_pix, noise_level, l_knee,
                                path = transfer_function)

    names, steps, fracs = sampled_names_and_steps(is_sampled, param_ground, step_fracs)

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    #w_k = independent real DOF per rfft entry; get_fourier_weights indexes the half-axis
    #(columns), so it broadcasts across rows
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))

    #the QE reconstruction noise - and, for "delensed", the delensing efficiency - is
    #evaluated once at the fiducial cosmology and held fixed across the whole
    #finite-difference stencil; see covariance_blocks for why. cls_with_qe_response rather
    #than camb_cls_at_params so that qe_response = "gradient" finds its "gradient_TT" key -
    #it is a no-op on the default "unlensed", and only the fiducial point ever pays for it
    cls_fid = cls_with_qe_response(param_ground, qe_response)
    nphi_fid, alens = frozen_reconstruction(cls_fid, spectra, nside, pix_width, ell_grid,
                                            noise_level, l_knee, beam_fwhm, l_cutoff,
                                            iterative_delens, verbose,
                                            nphi_source = nphi_source,
                                            param_ground = param_ground,
                                            qe_response = qe_response)

    def blocks_at(params):
        #the frozen per-L Alens gives every stencil point its own CAMB-delensed TT, and the
        #frozen R rescales each of them onto the box's own lensing calculation
        cls = (camb_cls_at_params(params) if alens is None
               else delensed_cls_at_params(params, alens, transfer = transfer))
        return covariance_blocks(cls, spectra, nside, pix_width, ell_grid, noise_level,
                                 l_knee, beam_fwhm, l_cutoff, nphi_fid)

    if verbose:
        ells_on_grid = ell_grid[ell_grid > 0]
        print(f"Fisher forecast [{spectra}]: nside {nside}, theta_pix {theta_pix}', "
              f"{noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  box {nside * theta_pix / 60:.2f} deg, ell in "
              f"[{float(jnp.min(ells_on_grid)):.0f}, {float(jnp.max(ells_on_grid)):.0f}], "
              f"{int(jnp.sum(weights))} real DOF")
        print(f"  sampled: {names}")
        if transfer is not None:
            apply_transfer_function(jnp.ones(1), jnp.ones(1), transfer, verbose = True)
        print(f"  running {2 * len(names) + 1} CAMB calls...")

    blocks_fid = blocks_at(param_ground)

    blocks_plus, blocks_minus = [], []
    for name, step in zip(names, steps):
        up = dict(param_ground)
        up[name] = param_ground[name] + step
        down = dict(param_ground)
        down[name] = param_ground[name] - step
        blocks_plus.append(blocks_at(up))
        blocks_minus.append(blocks_at(down))
        if verbose:
            print(f"    {name}: h = {step:.6g} ({fracs[name]:g} sigma)")

    return names, steps, weights, blocks_fid, blocks_plus, blocks_minus


# ── The "blocks" contraction ──────────────────────────────────────────────

def _fisher_from_blocks(blocks_plus, blocks_minus, blocks_fid, steps, weights):
    """F_ij = 1/2 sum_blocks sum_k w_k dln(C)/dtheta_i dln(C)/dtheta_j.

    blocks_plus/minus are lists (one entry per sampled parameter) of block dicts at
    theta +/- h; blocks_fid is the block dict at the fiducial point.
    """
    n_param = len(steps)
    fisher = np.zeros((n_param, n_param))

    for name in blocks_fid:
        c_fid = blocks_fid[name]
        #the [0, 0] origin is set to zero by construction (origin_value = 0) and carries
        #no information; guard the division rather than letting it produce NaN
        good = c_fid > 0
        safe = jnp.where(good, c_fid, 1.0)

        dlog = []
        for i in range(n_param):
            derivative = (blocks_plus[i][name] - blocks_minus[i][name]) / (2 * steps[i])
            dlog.append(jnp.where(good, derivative / safe, 0.0))

        for i in range(n_param):
            for j in range(i, n_param):
                value = 0.5 * float(jnp.sum(weights * dlog[i] * dlog[j]))
                fisher[i, j] += value
                if i != j:
                    fisher[j, i] += value

    return fisher


def forecast(nside, theta_pix, noise_level, is_sampled, param_ground,
             spectra = "lensed", step_fracs = None, l_knee = 0, beam_fwhm = 0,
             l_cutoff = 10_000, verbose = True, iterative_delens = False,
             nphi_source = "covariance", qe_response = DEFAULT_QE_RESPONSE,
             transfer_function = None):
    """Gaussian Fisher matrix for the sampled LCDM parameters on an nside x nside box.

    Args:
        nside:        pixels per side (the sampler's nside)
        theta_pix:    pixel width in arcmin
        noise_level:  white-noise level in uK-arcmin
        is_sampled:   {param_name: bool}. False parameters are held FIXED (not
                      marginalized), matching the sampler's should_sample semantics
        param_ground: {param_name: float} fiducial point. All five must be present,
                      since the four unsampled ones still set the cosmology
        spectra:      "lensed" (default), "unlensed", "ceiling" or "delensed" - see the
                      module docstring for what each one does and does not bound
        step_fracs:   override for FD_STEP_FRAC (fractions of PARAM_SIGMA)
        l_knee:       1/f noise knee. Defaults to 0 (pure white noise), matching every
                      sampler entry point; load_sim's own default of 100 does not apply
        beam_fwhm:    beam FWHM in arcmin, default 0 as in load_sim
        iterative_delens: "delensed" only: iterate the per-L delensing fraction against
                      the reconstruction noise to a fixed point (iterative_delensing)
        nphi_source:  "delensed" only: where the N_L that sets Alens_L comes from,
                      "covariance" (the box's QE matrix, default) or "hu_okamoto" (the
                      analytic N^(0) quadrature) - see NPHI_SOURCES
        qe_response:  which TT spectrum the quadratic estimator's response is built from,
                      "unlensed" (default, as before) or "gradient" - see
                      QE_RESPONSE_SOURCES
        transfer_function: "delensed" only: path to a merge_delensed_spectra.py npz. Rescales
                      CAMB's delensed spectrum at every stencil point by the empirically
                      measured R(l), so the forecast contracts the delensed spectrum THIS
                      codebase's lense_flow and map_joint actually produce rather than
                      CAMB's - see load_transfer_function and cmb_lensing/delensed_spectrum.py

    Returns:
        (fisher, names) - the n_sampled x n_sampled matrix and the parameter names in
        OUTPUT_PARAM_ORDER order (chain_analysis.py's order, not PARAM_ORDER).
    """
    names, steps, weights, fid, plus, minus = covariance_stencil(
        nside, theta_pix, noise_level, is_sampled, param_ground, spectra, step_fracs,
        l_knee, beam_fwhm, l_cutoff, verbose, iterative_delens = iterative_delens,
        nphi_source = nphi_source, qe_response = qe_response,
        transfer_function = transfer_function)
    return _fisher_from_blocks(plus, minus, fid, steps, weights), names


# ── Shared post-processing ────────────────────────────────────────────────

def covariance_from_fisher(fisher, names):
    """Invert the Fisher matrix, failing loudly on a singular / non-PSD result."""
    eigenvalues = np.linalg.eigvalsh(fisher)
    if np.any(eigenvalues <= 0):
        raise RuntimeError(
            f"Fisher matrix is not positive definite (eigenvalues {eigenvalues}). "
            f"Usually a degenerate parameter pair at this box size, or a finite-difference "
            f"step that is too small - try raising FD_STEP_FRAC and re-running --stability."
        )
    condition = eigenvalues[-1] / eigenvalues[0]
    if condition > 1e12:
        print(f"WARNING: Fisher condition number {condition:.3g} - the inverse is "
              f"numerically unreliable, so at least one direction is near-degenerate "
              f"over {names}")
    return np.linalg.inv(fisher)


def correlation_from_covariance(covariance):
    """F^-1 normalized to unit diagonal - the degeneracy structure with the units divided out."""
    sigmas = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigmas, sigmas)


def step_stability(forecast_fn, method = METHOD, factor = 2.0):
    """Recompute a forecast at `factor` x the step size and report the drift.

    A well-converged Fisher is step-independent. Drift means either the step is too large
    (real curvature in Cl(theta)) or too small (CAMB's own accuracy floor amplified by
    1/h). Second derivatives would amplify this by 1/h**2; here the log-derivative is
    first order, so a few percent drift is already worth chasing.

    `forecast_fn(step_fracs) -> (fisher, names)` is the method's forecast with everything
    but the step fractions bound - each module hands in its own closure, so one checker
    serves every contraction. The CAMB runs are memoized, so the doubled-step run is one
    extra set of calls. Returns the per-parameter fractional drift in sigma.
    """
    base = {name: FD_STEP_FRAC[name] for name in PARAM_ORDER}
    doubled = {name: factor * value for name, value in base.items()}

    fisher_a, names = forecast_fn(base)
    fisher_b, _ = forecast_fn(doubled)

    sigma_a = np.sqrt(np.diag(covariance_from_fisher(fisher_a, names)))
    sigma_b = np.sqrt(np.diag(covariance_from_fisher(fisher_b, names)))
    drift = np.abs(sigma_b / sigma_a - 1)

    print(f"\nStep stability [{method}] (h vs {factor:g}h), fractional change in "
          f"forecast sigma:")
    for name, value in zip(names, drift):
        flag = "  <-- unstable" if value > 0.01 else ""
        print(f"  {name:<14s} {value:.2%}{flag}")
    if np.max(drift) > 0.01:
        print("  tune FD_STEP_FRAC until every entry is well under 1%")
    return drift


def report_sigmas(covariance, names, spectra, method):
    """Print the marginalized 1-sigma errors and the strongest degeneracy."""
    sigmas = np.sqrt(np.diag(covariance))
    print(f"\nForecast 1-sigma errors [{spectra}, {method}]:")
    print(f"  {'parameter':<14s} {'sigma':>12s} {'sigma/PARAM_SIGMA':>20s}")
    for i, name in enumerate(names):
        print(f"  {name:<14s} {sigmas[i]:>12.4g} {sigmas[i] / PARAM_SIGMA[name]:>20.3g}")

    if len(names) > 1:
        correlation = covariance / np.outer(sigmas, sigmas)
        pairs = [(abs(correlation[i, j]), correlation[i, j], names[i], names[j])
                 for i in range(len(names)) for j in range(i + 1, len(names))]
        _, value, first, second = max(pairs)
        print(f"  strongest degeneracy: {first} - {second}  (r = {value:+.2f})")


def report_ceiling_ratio(fisher_lensed, fisher_ceiling, method):
    """How far the "ceiling" bound sits above a "lensed" forecast, per parameter.

    The posterior the chains measure sits between the two, so the size of this gap says
    how much room the lensing non-Gaussianity has to move the answer.
    """
    ratio = np.sqrt(np.diag(fisher_ceiling)) / np.sqrt(np.diag(fisher_lensed))
    print(f"\n  [{method}] F_ceiling / F_lensed per parameter (diagonal, sqrt): "
          f"{np.array2string(ratio, precision = 1)}")
    print("  the sampler's posterior sits between these two")


# ── Plotting ──────────────────────────────────────────────────────────────

def annotated_heatmap(axis, matrix, names, title, colorbar_label):
    """Heatmap coloured by the correlation-normalized matrix (bounded, readable) with the
    raw entries annotated - the raw values span many orders of magnitude across
    parameters with wildly different units, so colouring by them directly is useless."""
    diagonal = np.sqrt(np.abs(np.diag(matrix)))
    normalized = matrix / np.outer(diagonal, diagonal)

    image = axis.imshow(normalized, cmap = "RdBu_r", vmin = -1, vmax = 1)
    axis.set_xticks(range(len(names)))
    axis.set_yticks(range(len(names)))
    axis.set_xticklabels(names, rotation = 45, ha = "right")
    axis.set_yticklabels(names)
    axis.set_title(title)

    for i in range(len(names)):
        for j in range(len(names)):
            shade = "white" if abs(normalized[i, j]) > 0.6 else "black"
            axis.text(j, i, f"{matrix[i, j]:.2e}", ha = "center", va = "center",
                      color = shade, fontsize = 7)

    colorbar = plt.colorbar(image, ax = axis, fraction = 0.046, pad = 0.04)
    colorbar.set_label(colorbar_label)
    return image


def plot_fisher_matrix(fisher, names, path, subtitle = ""):
    figure, axis = plt.subplots(figsize = (7.5, 6.5))
    annotated_heatmap(axis, fisher, names,
                       "Fisher matrix $F_{ij}$" + (f"\n{subtitle}" if subtitle else ""),
                       r"$F_{ij}\,/\,\sqrt{F_{ii}F_{jj}}$")
    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def plot_annotated_matrix(matrix, param_names, title, output_path, file_name, cmap = "coolwarm",
                          norm = None, fmt = "{:+.3f}"):
    """Draw a square matrix as a colored grid with the value of every entry printed in its box.

    norm is a matplotlib color normalization (defaults to a symmetric linear scale about zero,
    which suits correlation matrices); fmt is the format string applied to each entry. Text is
    drawn black or white depending on the luminance of the cell behind it so it stays legible
    across the whole colormap.
    """
    matrix = np.asarray(matrix)
    n = len(param_names)
    if norm is None:
        limit = np.max(np.abs(matrix))
        norm = matplotlib.colors.Normalize(vmin = -limit, vmax = limit)
    fig, ax = plt.subplots(figsize = (1.7 * n + 2.5, 1.7 * n + 1.5))
    image = ax.imshow(matrix, cmap = cmap, norm = norm)
    ax.set_xticks(range(n)); ax.set_xticklabels(param_names, rotation = 45, ha = "right")
    ax.set_yticks(range(n)); ax.set_yticklabels(param_names)
    #minor ticks give the white grid lines separating the boxes
    ax.set_xticks(np.arange(n + 1) - 0.5, minor = True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor = True)
    ax.grid(which = "minor", color = "white", linewidth = 2)
    ax.tick_params(which = "minor", length = 0)
    for i in range(n):
        for j in range(n):
            rgba = image.cmap(norm(matrix[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt.format(matrix[i, j]), ha = "center", va = "center",
                    color = "white" if luminance < 0.5 else "black", fontsize = 11)
    fig.colorbar(image, ax = ax, fraction = 0.046, pad = 0.04)
    ax.set_title(title)
    plt.savefig(output_path + file_name, dpi = 150, bbox_inches = "tight")
    plt.close(fig)


def plot_covariance_matrix(covariance, names, path, subtitle = ""):
    sigmas = np.sqrt(np.diag(covariance))
    ratios = np.array([sigmas[i] / PARAM_SIGMA[name] for i, name in enumerate(names)])

    figure, (left, right) = plt.subplots(
        1, 2, figsize = (12.5, 6.0), gridspec_kw = {"width_ratios": [1.35, 1]})

    annotated_heatmap(left, covariance, names,
                       "Covariance $F^{-1}$" + (f"\n{subtitle}" if subtitle else ""),
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
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def write_outputs(fisher, covariance, names, directory, method, suffix, label, subtitle,
                  config):
    """Write the three figures and the npz for one method, tagged with its suffix.

    `method` / `suffix` / `label` are the calling module's METHOD / SUFFIX / LABEL; `config`
    is the run configuration (nside, noise, ...) stored alongside the matrices in the npz
    so a saved forecast can be traced back to the box it was computed on. Returns the file
    names written.
    """
    full_subtitle = f"{subtitle}\n{label}"

    plot_fisher_matrix(fisher, names,
                       os.path.join(directory, f"fisher_matrix{suffix}.png"),
                       full_subtitle)
    plot_covariance_matrix(covariance, names,
                           os.path.join(directory, f"covariance_matrix{suffix}.png"),
                           full_subtitle)
    correlation = correlation_from_covariance(covariance)
    plot_annotated_matrix(correlation, names,
                          f"Forecast Correlation Matrix\n{full_subtitle}",
                          directory, f"/correlation_matrix{suffix}.png",
                          norm = matplotlib.colors.Normalize(vmin = -1, vmax = 1),
                          fmt = "{:+.3f}")
    np.savez(os.path.join(directory, f"fisher{suffix}.npz"),
             fisher = fisher, covariance = covariance, correlation = correlation,
             names = np.array(names), sigmas = np.sqrt(np.diag(covariance)),
             method = method, **config)
    return [f"fisher_matrix{suffix}.png", f"covariance_matrix{suffix}.png",
            f"correlation_matrix{suffix}.png", f"fisher{suffix}.npz"]


def save_outputs(fisher, covariance, names, method, suffix, label, subtitle, config):
    """write_outputs into output_dir() (created if needed) and say what was written."""
    directory = output_dir()
    os.makedirs(directory, exist_ok = True)
    written = write_outputs(fisher, covariance, names, directory, method, suffix, label,
                            subtitle, config)
    print(f"\nWrote {', '.join(written)} to {directory}")
    return written


# ── CLI helpers shared by every forecasting module ────────────────────────

def add_box_arguments(parser):
    """--nside / --theta_pix / --noise / --l_knee / --beam_fwhm / --params: the box and
    the parameter subset, identical across every module's CLI."""
    parser.add_argument("--nside", type = int, default = 128)
    parser.add_argument("--theta_pix", type = float, default = 2.5,
                        help = "pixel width in arcmin")
    parser.add_argument("--noise", type = float, default = 2.5,
                        help = "white noise level in uK-arcmin")
    parser.add_argument("--l_knee", type = float, default = 0.0,
                        help = "1/f knee; 0 (default) matches every sampler entry point")
    parser.add_argument("--beam_fwhm", type = float, default = 0.0)
    parser.add_argument("--params", nargs = "*", default = ["logA", "omch2", "theta_MC_100"],
                        help = f"subset to sample; default all of {PARAM_ORDER}")
    return parser


def add_qe_response_argument(parser):
    """--qe_response: which TT spectrum the quadratic estimator's response is built from.

    Its own helper rather than a line inside add_spectra_arguments because it also applies
    to 1st_principles, which has no spectra-mode knobs at all. add_spectra_arguments calls
    this, so the covariance-stencil CLIs pick it up without asking twice.
    """
    parser.add_argument("--qe_response", choices = QE_RESPONSE_SOURCES,
                        default = DEFAULT_QE_RESPONSE,
                        help = "which TT spectrum weights the quadratic estimator's "
                               "response f(l, l') - and so sets its normalization N^(0). "
                               "'unlensed' (default) is C_l^TT, Hu & Okamoto's original "
                               "expression and the spectrum the sampler's own QE norm uses; "
                               "'gradient' is CAMB's lensed temperature-gradient spectrum "
                               "C_l^(T grad T), the correct response weight (Lewis, "
                               "Challinor & Hanson 2011), which resums the N^(2) bias into "
                               "the normalization at the cost of one extra CAMB call. "
                               "Independent of --nphi_source and applies to both of its "
                               "settings")
    return parser


def add_spectra_arguments(parser):
    """--spectra / --iterative_delens / --nphi_source / --qe_response / --stability: the
    knobs of the covariance-stencil methods (blocks, cls, full_sky). The realization methods
    draw from the prior and have no spectra mode."""
    parser.add_argument("--spectra", choices = SPECTRA_MODES, default = "ceiling") #lensed
    parser.add_argument("--iterative_delens", action = "store_true",
                        help = "--spectra delensed: iterate the delensing efficiency "
                               "against the reconstruction noise to a fixed point instead "
                               "of taking one QE pass. Better delensing quiets the "
                               "estimator, which improves delensing; this is what "
                               "iterative lensing reconstruction does, and it is closer to "
                               "the MAP reconstruction map_joint actually performs")
    parser.add_argument("--nphi_source", choices = NPHI_SOURCES, default = "covariance",
                        help = "where the quadratic-estimator noise N_phi comes from: the "
                               "phi block's C_phi + N_phi in every spectra mode, and for "
                               "--spectra delensed also the N_L behind the per-L delensing "
                               "fraction Alens_L = N_L / (C_L + N_L). 'covariance' "
                               "(default) is the box's QE matrix, the very N_phi the "
                               "sampler is built from (anisotropic on the square box); "
                               "'hu_okamoto' is the analytic flat-sky N^(0) integral "
                               "(isotropic, no box), interpolated onto the rfft grid like "
                               "C_phi")
    add_qe_response_argument(parser)
    parser.add_argument("--stability", action = "store_true",
                        help = "also recompute at 2x the step size and report the drift")
    return parser


def add_transfer_function_argument(parser):
    """--transfer_function: the empirical delensing correction, for --spectra delensed.

    Its own helper rather than a line in add_spectra_arguments because only the modules that
    actually thread it through to covariance_stencil should advertise it - a flag that parses
    and is then ignored is worse than no flag. fisher_forecast's main() calls this; a sibling
    adopting it needs the same call plus one pass-through into covariance_stencil.
    """
    parser.add_argument("--transfer_function", type = str, default = None,
                        help = "path to a merge_delensed_spectra.py transfer_function.npz. "
                               "With --spectra delensed, rescales CAMB's delensed spectrum "
                               "at every stencil point by the empirically measured R(l), so "
                               "the forecast uses the delensed spectrum this codebase's "
                               "lense_flow and map_joint actually produce. Rejected with any "
                               "other --spectra, and rejected if it was measured on a "
                               "different box than this run")
    return parser


def sampled_from_args(parser, args):
    """{param: bool} from --params, rejecting unknown names through the parser."""
    is_sampled = {name: (args.params is None or name in args.params)
                  for name in PARAM_ORDER}
    if args.params is not None:
        unknown = [name for name in args.params if name not in PARAM_ORDER]
        if unknown:
            parser.error(f"unknown parameters {unknown}; choose from {PARAM_ORDER}")
    return is_sampled


def box_subtitle(spectra, args):
    """The first line of every figure subtitle: the spectra mode and the box."""
    return (f"{spectra}  |  nside {args.nside}, {args.theta_pix:g}', "
            f"{args.noise:g} uK-arcmin")


def run_config(spectra, args, names, **extra):
    """The run configuration stored in every fisher<suffix>.npz."""
    return dict(spectra = spectra, nside = args.nside, theta_pix = args.theta_pix,
                noise_level = args.noise, l_knee = args.l_knee, beam_fwhm = args.beam_fwhm,
                qe_response = getattr(args, "qe_response", DEFAULT_QE_RESPONSE),
                #np.savez cannot store None, so "no empirical correction" is the empty string
                transfer_function = getattr(args, "transfer_function", None) or "",
                step_fracs = np.array([FD_STEP_FRAC[name] for name in names]), **extra)


def main():
    parser = argparse.ArgumentParser(
        description = "Gaussian Fisher forecast for the LCDM parameters on the "
                      "sample_lcdm.py flat-sky box - the covariance-block trace formula")
    add_box_arguments(parser)
    add_spectra_arguments(parser)
    add_transfer_function_argument(parser)
    args = parser.parse_args()
    is_sampled = sampled_from_args(parser, args)

    def run(step_fracs = None, spectra = args.spectra, verbose = True):
        #the ceiling comparison below re-runs at spectra = "ceiling", where a transfer
        #function is rejected - it only ever applies to the delensed spectrum
        return forecast(args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
                        spectra = spectra, step_fracs = step_fracs, l_knee = args.l_knee,
                        beam_fwhm = args.beam_fwhm, verbose = verbose,
                        iterative_delens = args.iterative_delens,
                        nphi_source = args.nphi_source,
                        qe_response = args.qe_response,
                        transfer_function = (args.transfer_function
                                             if spectra == "delensed" else None))

    fisher, names = run()
    covariance = covariance_from_fisher(fisher, names)
    report_sigmas(covariance, names, args.spectra, METHOD)

    if args.spectra == "lensed":
        ceiling, _ = run(spectra = "ceiling", verbose = False)
        report_ceiling_ratio(fisher, ceiling, METHOD)

    if args.stability:
        step_stability(lambda fracs: run(step_fracs = fracs, verbose = False), METHOD)

    save_outputs(fisher, covariance, names, METHOD, SUFFIX, LABEL,
                 box_subtitle(args.spectra, args),
                 run_config(args.spectra, args, names,
                            iterative_delens = args.iterative_delens,
                            nphi_source = args.nphi_source))


if __name__ == "__main__":
    main()
