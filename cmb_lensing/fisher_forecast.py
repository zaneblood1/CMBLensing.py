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

The RECONSTRUCTION is frozen across that stencil by default (constant_nphi = True): N_phi -
and, for "delensed", the delensing fraction Alens_L - are evaluated once at the fiducial
cosmology, so only the signal carries theta dependence. "--vary_nphi" rebuilds both at every
stencil point instead, which adds dN_phi/dtheta to the phi block's log-derivative. See
DEFAULT_CONSTANT_NPHI for why frozen is the default, and for what the varying number does
and does not mean - it is a bound on the size of that modelling choice, not a better
forecast.

Parameters with is_sampled[name] = False are held FIXED, not marginalized over; F is
n_sampled x n_sampled.

Usage:
    python -m cmb_lensing.fisher_forecast                       #nside 128, T-only, 2.5 uK-arcmin
    python -m cmb_lensing.fisher_forecast --spectra lensed
    python -m cmb_lensing.fisher_forecast --spectra delensed --iterative_delens
    python -m cmb_lensing.fisher_forecast --spectra lensed --vary_nphi
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
_GRID_CACHE = {}

#where the model spectra at each stencil point come from:
#  "camb"  (DEFAULT) a direct CAMB run per cosmology (camb_cls_at_params) - the physics
#  "grid"  the 5D CAMB grid spline sample_lcdm evaluates at every theta proposal
#          (grid_cls_at_params) - the model the chains actually run on. The chains' data
#          come from load_sim's direct CAMB call while their likelihood goes through the
#          grid, so the grid's DERIVATIVES set the posterior width; this source makes the
#          forecast use the same ones. The grid-vs-CAMB value offset at the fiducial point
#          biases the chains instead, which no Fisher matrix sees
CL_SOURCES = ("camb", "grid")
DEFAULT_CL_SOURCE = "camb"
#sample_lcdm.CAMB_GRID_PATH, spelled out so the forecasts need not import the sampler
CAMB_GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camb_splines",
                              "camb_grid_spline.npz")
#grid spectrum -> the cls-dict key it stands in for (same units, same 2..3999 ells: the
#grid jobs run simulate.dl2cl's conventions at lmax = CAMB_LMAX)
GRID_CLS_KEYS = {"scalar_TT": "tt", "total_TT": "tt_lensed", "phi": "pp"}

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
#  "measured"    the EMPIRICAL N_L^eff written by merge_phi_noise.py: the noise map_joint's
#                MAP reconstruction actually achieves, from the cross correlation of the
#                estimate with the true phi over many realizations at the fiducial cosmology
#                (cmb_lensing/phi_noise.py). Neither of the other two describes a MAP
#                estimator at all - they both describe a quadratic one - so this is the only
#                source that matches the reconstruction this codebase performs. It needs a
#                merged npz, passed as --phi_noise / measured_phi_noise; isotropic, and put
#                on the rfft grid the same way "hu_okamoto" is
NPHI_SOURCES = ("covariance", "hu_okamoto", "measured")

#whether the RECONSTRUCTION - N_phi, and for "delensed" the delensing fraction Alens_L - is
#held at the fiducial cosmology across the finite-difference stencil or rebuilt at every
#point from that point's own Cls:
#  True   (DEFAULT) frozen. N_phi is a property of the ESTIMATOR and the experiment rather
#         than of the model being constrained, so only the signal carries theta dependence -
#         the same convention that makes the instrumental C_n theta-independent, and the
#         same thing the sampler does (its QE norm stays at the param_init cosmology for the
#         whole chain, where it is only preconditioning G and the phi mass matrix and
#         cancels). It is also what a real quadratic-estimator analysis does:
#         realization-dependent N0 (RDN0) computes the bias from the DATA rather than from
#         the sampled cosmology precisely so that it stops depending on theta, and the
#         normalization dependence that survives is handled as a linear nuisance correction
#         rather than as constraining power
#  False  N_phi(theta) and Alens_L(theta) are rebuilt at every stencil point, so the phi
#         block's log-derivative picks up dN_phi/dtheta and, for "delensed", the f block is
#         lensed with a theta-dependent residual fraction. That derivative is real - N_phi
#         genuinely moves with the cosmology - but it is not INDEPENDENT information:
#         N_L^(0) is a deterministic functional of the estimator's filter (C^TT + N) and its
#         response, so dN_phi/dtheta lies entirely inside the span of dC^TT/dtheta, which
#         the f block already counts at first order. The block-diagonal contraction in
#         _fisher_from_blocks adds the two as though they were independent measurements -
#         the same double counting "lensed" and "delensed" already have. Quote it as a BOUND
#         on how much the frozen-N_phi modelling choice is worth, not as the production
#         number
#Expect the difference to be small wherever C_phi dominates N_phi over the modes carrying
#the Fisher weight: the qe_response switch moves N_phi by 0.4-2.6% and the marginalized
#sigmas by < 0.35% on nside 64 / 5' / 5 uK, and one FD step perturbs C^TT far less than
#swapping the whole response spectrum does. It should grow on a noisier or finer box.
#Cost: 2 * n_sampled + 1 quadratic-estimator evaluations instead of one - cheap for
#"covariance" (FFT convolutions), the dominant cost for "hu_okamoto" (a polar quadrature at
#every point), and with iterative_delens the whole fixed-point iteration runs per point.
#qe_response = "gradient" additionally costs one ~5 s CAMB post-processing call per stencil
#point instead of one for the entire run. "measured" cannot vary at all - an empirical
#N_L^eff was measured at ONE cosmology, so there is nothing in it to differentiate, and
#covariance_stencil rejects the combination
DEFAULT_CONSTANT_NPHI = True

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

#how _radial_cl_profile averages the 2D N_phi inside each annulus before handing it to the
#log-log interpolation. The two have to agree about what the profile IS between its samples:
#  "geometric"  (DEFAULT) the mean of log N at the geometric-mean l. interpolate_spectrum
#               models the profile as a local power law, and for a power law this pair is
#               EXACT rather than approximate, so the annulus width stops mattering
#  "arithmetic" the DOF-weighted mean of N at the DOF-weighted mean l. What this module did
#               before 2026-09-17; pass it to reproduce those numbers
#N_phi falls roughly two decades across four annuli, so an arithmetic mean is dominated by a
#bin's LOWEST-l modes while its centre sits at the high-l end - it overestimates N_phi, and
#the error grows without bound as the bins widen. Scored against the 2D matrix by pushing the
#profile back onto the grid and comparing per-mode Wiener retention (nside 128 / 2.5' / 5
#uK-arcmin), at the production one-fundamental annuli: arithmetic -3.59%, geometric -0.01%.
#At four fundamentals: arithmetic -64.7%, geometric -0.54%. It moves the C_phi-weighted mean
#delensing efficiency from 0.798 to 0.875, and so Alens_L at every stencil point
RADIAL_MEAN_TYPES = ("geometric", "arithmetic")
DEFAULT_RADIAL_MEAN = "geometric"

#how measured_phi_noise_cl continues the empirical N_L^eff OUTSIDE the |L| bands it was
#measured in. Inside them both modes are the same log-log interpolation:
#  "hold"         (the default HERE) clamp to the nearest measured value. N_eff falls steeply
#                 with L, so continuing it as a power law below the lowest band inflates it
#                 enormously by the time it reaches L = 2 and drives Alens_L -> 1 (no
#                 delensing) across exactly the multipoles carrying most of C_phi. There is
#                 nothing to extrapolate FROM either: the box holds no modes below its
#                 fundamental. This is also what _radial_cl_profile effectively gives the
#                 analytic sources, so "measured" and "covariance" then differ only where a
#                 measurement exists
#  "extrapolate"  continue the measured power law in log-log, the same rule
#                 interpolate_spectrum applies to every other spectrum. Honest about the
#                 profile's local slope and consistent with how C_phi reaches the same
#                 multipoles, but that slope is fitted to the two end bands and the lever arm
#                 down to L = 2 is long
#The block forecasts keep "hold" so their numbers are unchanged; fisher_forecast_from_1st_principles
#defaults its own --measured_extend to "extrapolate". Neither is obviously right - the values
#below the fundamental are a CONVENTION, and the flag exists so the choice is visible
MEASURED_EXTEND_MODES = ("hold", "extrapolate")
DEFAULT_MEASURED_EXTEND = "hold"

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


def camb_results_at_params(params, camb_lmax = None):
    """(CAMBparams, CAMBdata) at an arbitrary 5-parameter point, memoized.

    Set up through simulate.camb_parameters, so the run is identical to load_sim's: H0 solved
    from cosmomc_theta, r = 0, mnu = 0.06, tau = 0.05, nt = 0, k_pivot = 0.05, Alens = 1,
    non-linear lensing. The results object is kept (not just its spectra) because
    delensed_cls_at_params needs get_partially_lensed_cls on it.

    `camb_lmax` defaults to the module's CAMB_LMAX; covariance_stencil passes the box's own
    camb_lmax_for_grid instead, so nothing is extrapolated inside the grid. The memo is keyed
    on it as well as on the cosmology - two runs at the same point but different lmax are
    different spectra, and returning the shorter one for the longer request would silently
    truncate the axis every caller derives from the array's length.
    """
    camb_lmax = CAMB_LMAX if camb_lmax is None else int(camb_lmax)
    key = (_camb_key(params), camb_lmax)
    if key in _CAMB_RESULTS_CACHE:
        return _CAMB_RESULTS_CACHE[key]

    pars = camb_parameters(None, params["ombh2"], params["omch2"],
                           params["theta_MC_100"] / 100, 0.0, DEFAULT_MNU, DEFAULT_TAUREIO,
                           np.exp(params["logA"]) * 1e-10, 0, params["ns"], camb_lmax,
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


def camb_cls_at_params(params, camb_lmax = None):
    """Full cls dict from one CAMB run at an arbitrary 5-parameter point.

    precompute_camb_1d.camb_cls_at only moves a single parameter off GROUND_TRUTH and
    only returns (TT, PP); the Fisher stencil needs an arbitrary point and the lensed
    spectra too. The spectra are exactly what simulate._run_camb extracts (same calls,
    same units, same _extract_all_cls), landing on ells 2..camb_lmax-1 with no
    extrapolation.

    `camb_lmax` defaults to CAMB_LMAX, which is load_sim's data-map support; the grid
    forecasts pass camb_lmax_for_grid so the spectra cover every mode the box carries. Every
    downstream axis is derived from the returned arrays' LENGTH rather than from CAMB_LMAX,
    so raising it propagates on its own.
    """
    camb_lmax = CAMB_LMAX if camb_lmax is None else int(camb_lmax)
    key = (_camb_key(params), camb_lmax)
    if key in _CAMB_CACHE:
        return _CAMB_CACHE[key]

    pars, results = camb_results_at_params(params, camb_lmax = camb_lmax)
    power_spectra = results.get_cmb_power_spectra(pars, lmax = camb_lmax - 1,
                                                  CMB_unit = "muK")
    lens_potential = results.get_lens_potential_cls(lmax = camb_lmax - 1,
                                                    CMB_unit = "muK")[:, :2]
    cls = _extract_all_cls(jnp.asarray(power_spectra["unlensed_scalar"]),
                           jnp.asarray(power_spectra["tensor"]),
                           jnp.asarray(power_spectra["total"]),
                           jnp.asarray(lens_potential), camb_lmax, camb_lmax)

    if not bool(jnp.all(jnp.isfinite(cls["total_TT"]))):
        raise RuntimeError(
            f"CAMB returned non-finite Cls at {dict((k, float(params[k])) for k in PARAM_ORDER)}. "
            f"check the stencil point is reachable by CAMB (H0 is solved from "
            f"cosmomc_theta inside DEFAULT_THETA_H0_RANGE), or shrink FD_STEP_FRAC "
            f"for that parameter."
        )

    _CAMB_CACHE[key] = cls
    return cls


def grid_cls_at_params(params, path = CAMB_GRID_PATH):
    """A cls dict holding ONLY scalar_TT, total_TT and phi, read off the 5D CAMB grid spline
    at an arbitrary 5-parameter point - the Cls sample_lcdm's theta step sees there.

    Evaluated with CambGrid.cl_local, which memory-maps each ~1.1 GB coefficient table and
    reads only the 4^5 spline support around the point (~16 MB), rather than building the
    ~2.1 GB-per-spectrum NdBSpline the sampler holds. Values are identical either way. The
    grid carries ells 2..CAMB_LMAX-1 only; callers needing more must extrapolate, as
    covar_matrix_from_cls does for the sampler.

    Raises, like camb_cls_at_params, if the point is outside the grid's box (NaN spline).
    """
    from cmb_lensing.camb_grid_interp import load_camb_grid
    key = (_camb_key(params), os.path.abspath(path))
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    grid = load_camb_grid(path)
    point = np.array([[float(params[name]) for name in PARAM_ORDER]])
    cls = {name: jnp.asarray(grid.cl_local(spectrum, point)[0])
           for name, spectrum in GRID_CLS_KEYS.items()}
    if not all(bool(jnp.all(jnp.isfinite(cl))) for cl in cls.values()):
        raise RuntimeError(
            f"the CAMB grid at {path} returned non-finite Cls at "
            f"{dict((k, float(params[k])) for k in PARAM_ORDER)} - the point is outside the "
            f"grid's box, or its theta_MC_100 is unreachable in the grid's H0 range")

    _GRID_CACHE[key] = cls
    return cls


def gradient_cls_at_params(params, camb_lmax = None):
    """C_l^(T grad T): the lensed TEMPERATURE-GRADIENT cross spectrum, on CAMB's 2..camb_lmax-1.

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
    limited by the run's own max_l (camb_lmax plus CAMB's lens margin, 4200 at the default
    4000). Measured against a max_l = 5700 run at GROUND_TRUTH: agreement is 8e-5 at l = 1000,
    1.4e-3 at l = 3000, 3e-3 at l = 3500, degrading to 3% in the last ~100 multipoles. Those
    multipoles carry little of the estimator's weight and the error is far below the
    correction being made; raising camb_lmax to the grid's corner (camb_lmax_for_grid, what
    covariance_stencil now does) pushes that degradation past the range the box uses.

    Memoized separately from camb_cls_at_params rather than folded into it: the call costs
    ~5 s (its own flat-sky lensing pass), and the reconstruction noise is only ever needed at
    the FIDUCIAL cosmology, so the 2 * n_sampled stencil points must not pay for it.
    """
    camb_lmax = CAMB_LMAX if camb_lmax is None else int(camb_lmax)
    key = (_camb_key(params), camb_lmax)
    if key in _GRADIENT_CACHE:
        return _GRADIENT_CACHE[key]

    _, results = camb_results_at_params(params, camb_lmax = camb_lmax)
    #_scale_cls leaves this in the l(l+1)/2pi convention with raw_cl = False, i.e. exactly
    #what dl2cl's default (non-phi, non-tphi) branch inverts, and "muK" matches the TT units
    gradient = results.get_lensed_gradient_cls(lmax = camb_lmax - 1, CMB_unit = "muK")
    spectrum = dl2cl(jnp.asarray(gradient[:, 0]), camb_lmax, camb_lmax)

    if not bool(jnp.all(jnp.isfinite(spectrum))) or bool(jnp.any(spectrum <= 0)):
        raise RuntimeError(
            f"CAMB returned a non-positive or non-finite T-grad-T spectrum at "
            f"{dict((k, float(params[k])) for k in PARAM_ORDER)}. The quadratic estimator's "
            f"response needs it positive on the whole axis; check the point is reachable by "
            f"CAMB and that DoLensing is on.")

    _GRADIENT_CACHE[key] = spectrum
    return spectrum


def cls_with_qe_response(params, qe_response = DEFAULT_QE_RESPONSE, camb_lmax = None):
    """camb_cls_at_params, carrying whatever extra spectrum `qe_response` needs.

    "unlensed" (the default) needs nothing beyond scalar_TT, so the dict comes back
    unchanged and no extra CAMB work is done at all; "gradient" attaches "gradient_TT".
    Only the fiducial point is built this way (see gradient_cls_at_params for why the
    gradient spectrum is not folded into camb_cls_at_params), which is enough because the
    reconstruction noise is frozen there across the whole stencil. `camb_lmax` is passed
    through to both, so the response shares the temperature spectra's multipole axis.
    """
    _check_qe_response(qe_response)
    cls = dict(camb_cls_at_params(params, camb_lmax = camb_lmax))
    if qe_response == "gradient":
        cls["gradient_TT"] = gradient_cls_at_params(params, camb_lmax = camb_lmax)
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

    Independent of load_phi_noise: R corrects the delensed TEMPERATURE spectrum for the
    difference between CAMB's lensing calculation and this box's, while N_L^eff replaces the
    quadratic estimator's reconstruction noise. They answer different questions and may be
    used together or separately.
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


def has_transfer_derivatives(merged):
    """True if the merged npz carries dR/dtheta (merge_delensed_spectra.py --shifted_dirs)."""
    return "transfer_derivative" in merged and len(merged["derivative_names"]) > 0


def check_transfer_derivatives(merged, param_ground, names, path = ""):
    """Validate a transfer function's dR/dtheta against the stencil it is about to feed.

    The linear model R_0 + (theta - theta_0) dR/dtheta is expanded about the cosmology R_0 was
    measured at, so that must BE the forecast's fiducial point - otherwise even the centre of
    the stencil would sit on an extrapolation. A sampled parameter with no measured
    derivative is allowed (R is then held flat in that direction, exactly the old
    construction) but is announced, since it is easy to forget one.
    """
    if not has_transfer_derivatives(merged):
        return []
    reference = {str(name): float(value)
                 for name, value in zip(merged["param_names"], merged["params"])}
    off = [name for name in PARAM_ORDER
           if not np.isclose(reference[name], param_ground[name], rtol = 1e-12, atol = 0)]
    if off:
        raise ValueError(
            f"the transfer function {path} was measured about {reference} but this forecast's "
            f"fiducial point differs in {off}. dR/dtheta is a linear expansion about the "
            f"measurement's cosmology, so the two must coincide.")
    measured = [str(name) for name in merged["derivative_names"]]
    flat = [name for name in names if name not in measured]
    if flat:
        print(f"  WARNING: no measured dR/dtheta for sampled {flat}; R is held flat in "
              f"{'that direction' if len(flat) == 1 else 'those directions'}")
    return flat


def transfer_at_params(merged, params = None):
    """R(l) on its measured bands at `params`: R_0 + sum_i (theta_i - theta_0,i) dR/dtheta_i.

    Without derivatives in the npz, or with params = None, this is R_0 - the flat
    construction. Parameters with no measured derivative contribute nothing.
    """
    transfer = np.asarray(merged["transfer"], dtype = np.float64)
    if params is None or not has_transfer_derivatives(merged):
        return transfer
    reference = {str(name): float(value)
                 for name, value in zip(merged["param_names"], merged["params"])}
    for name, derivative in zip(merged["derivative_names"], merged["transfer_derivative"]):
        name = str(name)
        transfer = transfer + (float(params[name]) - reference[name]) * np.asarray(derivative)
    return transfer


def apply_transfer_function(cl, cl_ells, merged, params = None, verbose = False):
    """Multiply a delensed spectrum by R(l), interpolated onto its multipole axis.

    `params` is the cosmology the spectrum belongs to. When the npz carries dR/dtheta
    (merge_delensed_spectra.py --shifted_dirs) R is evaluated there through the linear model
    in transfer_at_params, so the stencil's derivative picks up C dR/dtheta alongside
    R dC/dtheta; without derivatives, or with params = None, R_0 is applied everywhere.

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
    transfer = transfer_at_params(merged, params)
    factor = np.interp(np.asarray(cl_ells), band_ells, transfer)
    if verbose:
        print(f"  transfer function R(l): {len(band_ells)} bands over "
              f"l = {band_ells[0]:.0f}..{band_ells[-1]:.0f}, R in "
              f"{np.min(transfer):.4f}..{np.max(transfer):.4f} "
              f"({int(merged['n_realizations'])} realizations); held constant outside")
        if has_transfer_derivatives(merged):
            for name, derivative, scheme, n in zip(merged["derivative_names"],
                                                   merged["transfer_derivative"],
                                                   merged["derivative_schemes"],
                                                   merged["derivative_n_realizations"]):
                drift = np.max(np.abs(derivative)) * PARAM_SIGMA[str(name)]
                print(f"    dR/d{name}: {scheme} difference, {int(n)} realizations, "
                      f"R moves up to {drift:.2e} per sigma")
        else:
            print(f"    no dR/dtheta measured: R held flat across the stencil")
    return jnp.asarray(cl) * jnp.asarray(factor)


def load_delensed_covariance(path):
    """The empirical delensed covariance stencil written by merge_delensed_covariance.py.

    Unlike the transfer function, this is not a correction to CAMB: it holds the realization
    mean of |rfft2(L^-1(phi_hat) L(phi) f)|^2 / nside^2 - the box's own lense_flow and
    map_joint - at theta_0 and at theta_0 +/- h_i for every parameter it was measured for,
    on common random numbers. See cmb_lensing/delensed_covariance.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no delensed covariance at {path}. Produce one by running "
            f"sampling_chains/get_delensed_covariance.sh and then "
            f"merge_delensed_covariance.py --covariance_dir <its out_dir>.")
    merged = dict(np.load(path, allow_pickle = True))
    missing = [key for key in ("delensed_fid", "delensed_plus", "delensed_minus", "names",
                               "steps", "params") if key not in merged]
    if missing:
        raise ValueError(f"{path} is missing {missing}; it does not look like a "
                         f"merge_delensed_covariance.py product")
    return merged


def check_delensed_covariance(merged, nside, theta_pix, noise_level, l_knee, param_ground,
                              names, path = ""):
    """Validate an empirical delensed covariance against the stencil it is about to replace.

    The box must match (the matrices live on its rfft grid, and the reconstruction noise
    sets them), the centre must BE the forecast's fiducial point, and every sampled
    parameter must have been measured - there is no model to fall back on for a missing
    direction. Returns the finite-difference steps h_i for `names`, which the forecast must
    then use for EVERY block so the phi block is differenced over the same interval.
    """
    measured_box = dict(nside = int(merged["nside"]), theta_pix = float(merged["theta_pix"]),
                        noise_level = float(merged["noise_level"]),
                        l_knee = float(merged["l_knee"]))
    wanted_box = dict(nside = int(nside), theta_pix = float(theta_pix),
                      noise_level = float(noise_level), l_knee = float(l_knee))
    if measured_box != wanted_box:
        raise ValueError(f"the delensed covariance {path} was measured at {measured_box} but "
                         f"this forecast runs at {wanted_box} - re-run "
                         f"get_delensed_covariance.sh at this box.")
    reference = {str(name): float(value)
                 for name, value in zip(merged["param_names"], merged["params"])}
    off = [name for name in PARAM_ORDER
           if not np.isclose(reference[name], param_ground[name], rtol = 1e-12, atol = 0)]
    if off:
        raise ValueError(f"the delensed covariance {path} was measured about {reference} but "
                         f"this forecast's fiducial point differs in {off}")
    measured = [str(name) for name in merged["names"]]
    missing = [name for name in names if name not in measured]
    if missing:
        raise ValueError(f"the delensed covariance {path} has no stencil points for sampled "
                         f"{missing} (it has {measured}). Re-run get_delensed_covariance.sh "
                         f"with them in derivative_params, or drop them from --params.")
    shape = (int(nside), int(nside) // 2 + 1)
    if tuple(np.shape(merged["delensed_fid"])) != shape:
        raise ValueError(f"{path} holds matrices of shape {np.shape(merged['delensed_fid'])}, "
                         f"expected {shape}")
    return [float(merged["steps"][measured.index(name)]) for name in names]


def load_phi_noise(path):
    """The empirical effective reconstruction noise N_L^eff written by merge_phi_noise.py.

    N_L^eff is what map_joint's MAP reconstruction actually achieves on THIS box, measured
    from the cross correlation of the estimate with the true phi over many realizations at
    the fiducial cosmology - see cmb_lensing/phi_noise.py for the algebra. Returns the whole
    npz as a dict; `band_ells` and `n_eff` are what measured_phi_noise_cl uses and the rest
    is the configuration it was measured at.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no effective phi noise at {path}. Produce one by running "
            f"sampling_chains/get_effective_phi_noise.sh and then "
            f"merge_phi_noise.py --noise_dir <its out_dir>.")
    merged = dict(np.load(path, allow_pickle = True))
    missing = [key for key in ("band_ells", "n_eff") if key not in merged]
    if missing:
        raise ValueError(f"{path} is missing {missing}; it does not look like a "
                         f"merge_phi_noise.py product")
    return merged


def check_phi_noise(merged, nside, theta_pix, noise_level, l_knee, path = ""):
    """Refuse an N_L^eff measured on a different box than the forecast is running on.

    N_eff absorbs the box's own resolution and periodicity along with the reconstruction's
    quality, so it is only meaningful at the configuration it was measured at - applying an
    nside 64 / 5' measurement to an nside 128 / 2.5' forecast would import the wrong noise
    entirely. Mirrors load_hessian_directory's refusal to average mixed configurations.
    """
    measured = dict(nside = int(merged["nside"]), theta_pix = float(merged["theta_pix"]),
                    noise_level = float(merged["noise_level"]),
                    l_knee = float(merged["l_knee"]))
    wanted = dict(nside = int(nside), theta_pix = float(theta_pix),
                  noise_level = float(noise_level), l_knee = float(l_knee))
    if measured != wanted:
        raise ValueError(
            f"the effective phi noise {path} was measured at {measured} but this forecast "
            f"runs at {wanted}. N_eff absorbs the box's resolution and its reconstruction "
            f"noise, so it cannot be carried across - re-run get_effective_phi_noise.sh at "
            f"this box.")


def measured_phi_noise_cl(merged, ells, verbose = False,
                          extend = DEFAULT_MEASURED_EXTEND):
    """N_L^eff interpolated onto `ells`: log-log INSIDE the measured bands, `extend` outside.

    The measurement lives in |L| bands spanning exactly the modes the rfft grid carries (its
    fundamental to its corner mode), while `ells` is CAMB's full 2..CAMB_LMAX-1 axis, which
    reaches far below the fundamental and past the corner. Inside the bands N_eff is a smooth
    positive spectrum and log-log interpolation is the same rule covar_matrix_from_cls applies
    to C_phi, so signal and noise reach the grid on an identical footing.

    Outside them `extend` decides - see MEASURED_EXTEND_MODES. The default "hold" clamps to
    the nearest measured value rather than continuing a power law. N_eff falls steeply with
    L - a factor of ~30 per decade on the boxes this is run on - so extrapolating it down
    from the lowest band to L = 2 inflates it by nine orders of magnitude and drives Alens_L
    to 1 (no delensing) across exactly the multipoles where C_phi is largest, purely as an
    artifact of the lever arm. There is nothing to extrapolate FROM in any case: the box
    holds no modes below its fundamental, so neither the measurement nor the forecast has
    anything to say about them. "extrapolate" takes the opposite convention, continuing the
    measured slope the way every other spectrum here is continued; it is what
    fisher_forecast_from_1st_principles defaults to.

    That makes the values below the fundamental a convention, and the one chosen here matches
    what the analytic sources already do through _radial_cl_profile, so "measured" and
    "covariance" differ only where a measurement actually exists. The phi block never
    contracts those multipoles - the rfft grid carries no such modes - but Alens_L does use
    them, since CAMB delenses on its own full multipole axis.
    """
    band_ells = np.asarray(merged["band_ells"])
    n_eff = np.asarray(merged["n_eff"])
    if extend not in MEASURED_EXTEND_MODES:
        raise ValueError(f"extend must be one of {MEASURED_EXTEND_MODES}, got {extend!r}")

    if extend == "hold":
        #np.interp clamps to the end values outside the sample points, which is the held
        #continuation; in log-log the interior is identical to interpolate_spectrum
        values = np.exp(np.interp(np.log(np.asarray(ells)), np.log(band_ells),
                                  np.log(n_eff)))
    else:
        #the same log-log power-law continuation every other spectrum in this module gets
        values = interpolate_spectrum(np.asarray(ells), band_ells, n_eff)

    if verbose:
        outside = ((np.asarray(ells) < band_ells[0]) |
                   (np.asarray(ells) > band_ells[-1])).sum()
        print(f"  measured N_phi: {len(band_ells)} bands over "
              f"L = {band_ells[0]:.0f}..{band_ells[-1]:.0f} "
              f"({int(merged['n_realizations'])} realizations, map_joint "
              f"{int(merged['map_joint_steps'])} steps); "
              + ("held constant" if extend == "hold" else "log-log extrapolated")
              + f" outside ({outside} multipoles of the requested axis)")
    return values


def delensed_cls_at_params(params, alens, transfer = None, camb_lmax = None):
    """camb_cls_at_params plus "delensed_TT": CAMB lensing the unlensed spectra at `params`
    with C_L^phiphi scaled by the per-L `alens` (zero-based in L, Alens_L = 1 meaning no
    delensing at that L). CAMB's get_partially_lensed_cls reruns its full non-perturbative
    correlation-function lensing with the scaled potential - NOT a linear interpolation
    between the unlensed and lensed spectra, which it differs from at the percent level -
    so the result is the delensed TT a Wiener-filtered reconstruction with residual
    fraction Alens_L would leave. Cheap (~0.02 s) once the results object is cached.

    Multipoles past the end of `alens` (CAMB lenses with C_L^phiphi up to Params.max_l,
    which exceeds camb_lmax by CAMB's lens margin) carry its last value.

    `alens` is by default frozen at the fiducial cosmology and re-applied at every stencil
    point, which is the whole construction: the reconstruction is a property of the
    experiment and CAMB supplies the theta dependence, so the derivative this feeds the
    stencil is d/dtheta of CAMB's partially lensed spectrum at a fixed delensing fraction.
    Under covariance_stencil's constant_nphi = False the caller instead hands in the
    Alens_L of the point being evaluated, and the delensing fraction carries theta too - see
    DEFAULT_CONSTANT_NPHI.

    `transfer` optionally applies the empirical transfer function R(l) (load_transfer_function)
    so that the spectrum becomes the one the BOX's own lense_flow and map_joint produce rather
    than CAMB's. It multiplies at every stencil point. By default R is measured once at the
    fiducial cosmology and CAMB supplies the theta dependence, so the derivative this feeds
    the stencil is R * dC_CAMB/dtheta - holding R fixed in theta is an assumption, which
    sampling_chains/compare_transfer_functions.py tests. If the npz also carries dR/dtheta
    (merge_delensed_spectra.py --shifted_dirs), R is evaluated at `params` through
    R_0 + (theta - theta_0) dR/dtheta and the assumption is dropped.
    """
    cls = dict(camb_cls_at_params(params, camb_lmax = camb_lmax))
    pars, results = camb_results_at_params(params, camb_lmax = camb_lmax)
    #the delensed spectrum must land on the SAME axis as the rest of the dict, so its length
    #is taken from a spectrum that is already there rather than from the module constant
    lmax = 2 + cls["scalar_TT"].shape[0]
    scaling = np.full(pars.max_l + 1, float(alens[-1]))
    scaling[:len(alens)] = alens
    partial = results.get_partially_lensed_cls(scaling, lmax = lmax - 1,
                                               CMB_unit = "muK")
    delensed = dl2cl(jnp.asarray(partial[:, 0]), lmax, lmax)
    if transfer is not None:
        ells = jnp.arange(2, 2 + delensed.shape[0]).astype(jnp.float64)
        delensed = apply_transfer_function(delensed, ells, transfer, params = params)
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
                         l_cutoff, lmax_prime = None):
    """(noise, mask, beam) on the rfft grid - the theta-INDEPENDENT part of every block.

    `lmax_prime` is the multipole axis the analytic noise is built on. It defaults to
    load_sim's, but covariance_blocks passes the length of the cls it was handed so that
    C_n and C_f share one axis when CAMB has been run past CAMB_LMAX. The noise is analytic
    and smooth, so extending it changes nothing - this only keeps the two aligned.
    """
    lmax_prime = min(DEFAULT_MAX_ELL, CAMB_LMAX) if lmax_prime is None else int(lmax_prime)
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
                                     mask, beam, pix_width) #/ 1.5 #/ NPHI_FAC


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


def camb_lmax_for_grid(ell_grid):
    """The CAMB lmax that covers every mode the rfft grid carries - the stencil's default.

    covar_matrix_from_cls log-log EXTRAPOLATES past the end of a spectrum's support, so any
    grid mode above CAMB's last multipole is filled by a power law fitted to CAMB's final two
    points rather than by CAMB. At nside 128 / 2.5' the corner is 6109 against CAMB_LMAX's
    3999, which puts 32.6% of the grid's DOF on that continuation. It is a bad continuation:
    the damping tail falls faster than any power law, so measured against CAMB run to 6400 the
    extrapolated LENSED TT is 2.5x too small at the corner, and - because the blocks contract
    dln(C_f + C_n)/dtheta, where getting C wrong moves the C/(C+N) weight - the contracted
    derivative is off by 55% there. Propagated through the delensed forecast that is 19% on
    sigma(theta_MC_100), 8% on sigma(logA) and roughly a halving of r(omch2, theta_MC_100).

    Floored at DEFAULT_MAX_ELL so a coarse box never runs CAMB to LESS than it used to: at 5'
    the corner is 3055, and cutting the axis there would shorten the range qe_noise_cl
    integrates over and delensing_alens scales, changing results for a reason that has
    nothing to do with extrapolation. Rounded up so the corner mode itself is interior.

    This does NOT touch DEFAULT_MAX_ELL, which anchors load_sim, the 1D spline caches and the
    5D grid - all built on 2..3999. Those must keep sharing one axis with each other (a
    one-multipole mismatch measurably biased the ombh2 conditional); only the forecast's own
    direct-CAMB path moves.
    """
    return max(DEFAULT_MAX_ELL, int(np.ceil(grid_max_ell(ell_grid))) + 1)


def qe_noise_spectrum(ells, cl_tt_response, cl_tt_lensed, noise_tt,
                      num_l = QE_NUM_L, num_angles = QE_NUM_ANGLES):
    """N_L^phiphi: the TT quadratic estimator's N^(0) noise, Hu & Okamoto (2002), flat sky.

        N_L = [ int d^2l / (2 pi)^2  f(l, L - l)^2 / (2 Ct_l Ct_|L - l|) ]^-1

        f(l, l') = C_l^grad (L . l) + C_l'^grad (L . l')      the lensing response
        Ct_l     = C_l^lensed + N_l                           the observed spectrum

    `cl_tt_response` is the spectrum the response f is built from, and every caller in this
    package selects it through `qe_response` (QE_RESPONSE_SOURCES), which DEFAULTS to the
    unlensed C_l^TT of Hu & Okamoto's original expression. Passing "gradient" instead gives
    the lensed temperature-gradient spectrum C_l^(T grad T) (gradient_cls_at_params): the two
    differ by the lensing correction to the response, and the gradient spectrum is the
    correct weight, resumming the N^(2) bias into the normalization (Lewis, Challinor &
    Hanson 2011). The unlensed default is what reproduces the pre-2026-09-15 numbers.

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


def _radial_cl_profile(matrix, ell_grid, weights, pix_width, ell_axis, label = "N_phi",
                       radial_mean = DEFAULT_RADIAL_MEAN):
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

    `radial_mean` picks how the entries inside an annulus are combined; see RADIAL_MEAN_TYPES
    for why the default is geometric and what the arithmetic one costs. The choice only
    matters because the profile is steep: on a flat one the two coincide.

    The profile is then log-log interpolated onto `ell_axis` with the SAME "extrapolate at
    both ends" convention covar_matrix_from_cls uses, so the two paths treat the edges of
    the support identically. A warning fires when `ell_axis` reaches past the grid's largest
    mode, where that extrapolation is doing real work: N_phi rises steeply there and a
    power-law continuation of it is a guess, not a measurement.

    NOTE the low-l edge is extrapolated too, and silently: below the grid's fundamental the
    box holds no modes at all, so the profile there is pure power-law continuation. Measured
    against the analytic Hu & Okamoto N_L, which needs no box, that continuation is good to
    14% at L = 2 and 21% at L = 60 on nside 128 / 2.5'. measured_phi_noise_cl takes the
    opposite convention and HOLDS its end value instead - see its docstring for why the two
    differ.
    """
    if radial_mean not in RADIAL_MEAN_TYPES:
        raise ValueError(f"radial_mean must be one of {RADIAL_MEAN_TYPES}, got "
                         f"{radial_mean!r}")
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

    #both the values and the bin centres are averaged in the SAME space, so that the pair
    #handed to interpolate_spectrum is self-consistent. the logs are taken only on the
    #usable entries - the rest are zeros the mask above already removed
    geometric = radial_mean == "geometric"
    value_terms = np.log(values[usable]) if geometric else values[usable]
    ell_terms = np.log(ells[usable]) if geometric else ells[usable]

    total = np.bincount(index[usable], weights = dof[usable] * value_terms, minlength = n_bin)
    norm = np.bincount(index[usable], weights = dof[usable], minlength = n_bin)
    centre = np.bincount(index[usable], weights = dof[usable] * ell_terms, minlength = n_bin)

    filled = norm > 0
    profile_ell = centre[filled] / norm[filled]
    profile = total[filled] / norm[filled]
    if geometric:
        profile_ell = np.exp(profile_ell)
        profile = np.exp(profile)

    #past the grid's largest mode there is no measurement left to interpolate between, so
    #that - not the last annulus's centre - is where the extrapolation starts doing work
    grid_max = float(np.max(ells[usable]))
    if np.max(ell_axis) > grid_max:
        print(f"  WARNING: {label} is log-log extrapolated past l = {grid_max:.0f} (the "
              f"grid's largest mode) out to l = {np.max(ell_axis):.0f}")

    return interpolate_spectrum(ell_axis, profile_ell, profile)


def qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, l_cutoff,
                nphi_source, filter_tt = None, qe_response = DEFAULT_QE_RESPONSE,
                measured_phi_noise = None, radial_mean = DEFAULT_RADIAL_MEAN):
    """N_L^phiphi on CAMB's integer multipoles 2..CAMB_LMAX-1, from any of NPHI_SOURCES.

    `filter_tt` is the TT spectrum the estimator's filter sees (default: the lensed one);
    iterative delensing passes the delensed spectrum. `qe_response` picks the spectrum the
    estimator's RESPONSE is built from, independently of `nphi_source` and applying to both
    of the quadratic-estimator sources (QE_RESPONSE_SOURCES). `measured_phi_noise` is the
    load_phi_noise dict, required by - and used only by - nphi_source = "measured", which
    ignores `filter_tt` and `qe_response` entirely because it is a measurement of the MAP
    reconstruction rather than a calculation of a quadratic estimator's noise. See
    NPHI_SOURCES for what the three sources are and how they differ.

    The two analytic sources are capped at the box: "covariance" because
    scalar_quadratic_estimate only ever sees the rfft grid, "hu_okamoto" because its
    integration axis is cut at grid_max_ell. The returned axis is CAMB's in every case, with
    the same log-log continuation past the grid's largest mode.
    """
    if nphi_source not in NPHI_SOURCES:
        raise ValueError(f"nphi_source must be one of {NPHI_SOURCES}, got {nphi_source!r}")
    ells = np.arange(2, 2 + cls["total_TT"].shape[0], dtype = np.float64)
    filter_tt = cls["total_TT"] if filter_tt is None else filter_tt

    if nphi_source == "measured":
        if measured_phi_noise is None:
            raise ValueError(
                "nphi_source = 'measured' needs the merge_phi_noise.py npz. Pass it as "
                "--phi_noise <path> (fisher_forecast) or measured_phi_noise = "
                "load_phi_noise(path) when calling this module directly. Only "
                "fisher_forecast.py threads it through so far; the sibling forecast modules "
                "still take their N_phi from the two analytic sources.")
        return measured_phi_noise_cl(measured_phi_noise, ells)

    if nphi_source == "covariance":
        matrix = qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                                 beam_fwhm, l_cutoff, filter_tt = filter_tt,
                                 qe_response = qe_response)
        weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                                   (nside, nside // 2 + 1))
        return _radial_cl_profile(matrix, ell_grid, weights, pix_width, ells,
                                  label = "N_phi (covariance source)",
                                  radial_mean = radial_mean)

    #the noise shares the cls' axis, whatever camb_lmax built them on
    lmax_prime = 2 + len(ells)
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
                  nphi_source, filter_tt = None, qe_response = DEFAULT_QE_RESPONSE,
                  measured_phi_noise = None, radial_mean = DEFAULT_RADIAL_MEAN):
    """The 2D N_phi the phi block adds to C_phi on the rfft grid, from any of NPHI_SOURCES.

    "covariance" is qe_noise_matrix itself. "hu_okamoto" and "measured" take their 1D N_L
    from qe_noise_cl and put it on the grid through covar_matrix_from_cls - the same log-log
    interpolation onto ell_grid, the same 1/pix_width**2 rescale and the same zeroed
    origin C_phi goes through - so signal and noise add on an identical footing and the block
    is a function of |l| alone, which the box matrix is not. `filter_tt`, `qe_response` and
    `measured_phi_noise` as in qe_noise_cl.
    """
    if nphi_source == "covariance":
        return qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                               beam_fwhm, l_cutoff, filter_tt = filter_tt,
                               qe_response = qe_response)
    nphi_cl = qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                          l_cutoff, nphi_source, filter_tt = filter_tt,
                          qe_response = qe_response,
                          measured_phi_noise = measured_phi_noise,
                          radial_mean = radial_mean)
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
    #C_phi lives on 2..lmax-1, so Alens_L zero-based in L is two entries longer
    alens = np.ones(2 + len(cphi))
    alens[2:] = nphi / (cphi + nphi)
    return alens


def iterative_delensing(param_ground, cls, nside, pix_width, ell_grid, noise_level,
                        l_knee, beam_fwhm, l_cutoff, nphi_source, max_iterations = 25,
                        tolerance = 1e-6, verbose = True,
                        qe_response = DEFAULT_QE_RESPONSE, camb_lmax = None,
                        radial_mean = DEFAULT_RADIAL_MEAN):
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
    alens = np.ones(2 + cls["phi"].shape[0])
    filter_tt = cls["total_TT"]
    converged = False
    for iteration in range(1, max_iterations + 1):
        nphi_cl = qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                              beam_fwhm, l_cutoff, nphi_source, filter_tt = filter_tt,
                              qe_response = qe_response, radial_mean = radial_mean)
        updated = delensing_alens(cls, nphi_cl)
        shift = float(np.max(np.abs(updated - alens)))
        alens = updated
        filter_tt = delensed_cls_at_params(param_ground, alens,
                                           camb_lmax = camb_lmax)["delensed_TT"]
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
    Hu & Okamoto spectrum on the grid, per nphi_source). This function does not decide
    where in the stencil it came from - covariance_stencil's `constant_nphi` does, and by
    default it is evaluated ONCE at the fiducial cosmology and passed in frozen at every
    point. Freezing is the default deliberately: N_phi is a property of the estimator and
    the experiment, not of the model being constrained, so only the signal should carry
    theta dependence - the same convention that makes the instrumental C_n
    theta-independent, and the same thing the sampler does (its QE norm stays at the
    param_init cosmology for the whole chain). Letting it move credits the forecast with
    information from dN_phi/dtheta, which a real C_l^phiphi likelihood removes with a
    realization-dependent N0; see DEFAULT_CONSTANT_NPHI.
    """
    ells = jnp.arange(2, 2 + cls["total_TT"].shape[0]).astype(jnp.float64)
    phi_ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)

    def covar(cl, ell_axis):
        return covar_matrix_from_cls(nside, pix_width, ell_grid, ell_axis, cl,
                                     origin_value = 0)

    noise, mask, beam = _instrument_matrices(nside, pix_width, ell_grid, noise_level,
                                             l_knee, beam_fwhm, l_cutoff,
                                             lmax_prime = 2 + cls["total_TT"].shape[0])

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

    if spectra in ("unlensed",):
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
        return {"TT": covar(cls["total_TT"], ells) + noise,
                "PP": phi_block}
    
    if spectra == "ceiling":
        #the CAMB-predicted T-phi cross spectrum (the ISW-lensing correlation), the
        #off-diagonal of the per-mode (T, phi) covariance. It changes sign at high ell
        #(61 negative entries near ell ~1100 at GROUND_TRUTH, at the 1e-16 level), so it
        #cannot go through covar's log-log interpolation; it is interpolated LINEARLY
        #onto the grid with the same origin and 1/pix_width**2 conventions. NOTE every
        #block contraction in this module treats a key as an independent auto-spectrum,
        #which is not what a cross spectrum is - it only means something inside a
        #per-mode (TT, TP; TP, PP) matrix
        return {"TlTl": covar(cls["total_TT"], ells) + noise,
                "TT": covar(cls["scalar_TT"], ells) + noise,
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
                          qe_response = DEFAULT_QE_RESPONSE,
                          measured_phi_noise = None, camb_lmax = None,
                          radial_mean = DEFAULT_RADIAL_MEAN):
    """(N_phi, Alens_L) at one cosmology - the reconstruction a stencil point sees.

    Both are properties of the estimator and the experiment rather than of the model being
    constrained, so by default they are evaluated once at the fiducial point and held fixed
    across the whole finite-difference stencil; covariance_blocks explains why letting them
    move credits the forecast with dN/dtheta information a real analysis removes. This
    function itself is agnostic - it builds the reconstruction implied by whatever `cls_fid`
    / `param_ground` it is handed, and covariance_stencil's `constant_nphi` decides whether
    that is the fiducial point alone (True, the default) or every stencil point (False).

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
    if iterative_delens and nphi_source == "measured":
        raise ValueError(
            "iterative delensing cannot be combined with nphi_source = 'measured'. The "
            "iteration exists to guess what a MAP reconstruction would achieve by quieting a "
            "quadratic estimator's filter; a measured N_eff already IS what the MAP achieved, "
            "so iterating it would re-apply the correction on top of the measurement.")
    if nphi_source not in NPHI_SOURCES:
        raise ValueError(f"nphi_source must be one of {NPHI_SOURCES}, got {nphi_source!r}")
    _check_qe_response(qe_response)

    nphi = qe_noise_grid(cls_fid, nside, pix_width, ell_grid, noise_level, l_knee,
                         beam_fwhm, l_cutoff, nphi_source, qe_response = qe_response,
                         measured_phi_noise = measured_phi_noise,
                         radial_mean = radial_mean)
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
            qe_response = qe_response, camb_lmax = camb_lmax,
            radial_mean = radial_mean)
        #the iteration quiets BOTH the delensing and the phi block's estimator
        nphi = qe_noise_grid(cls_fid, nside, pix_width, ell_grid, noise_level, l_knee,
                             beam_fwhm, l_cutoff, nphi_source, filter_tt = delensed_tt,
                             qe_response = qe_response, radial_mean = radial_mean)
        if verbose:
            print(f"  converged {converged} after {iterations} iterations")
    else:
        nphi_cl = qe_noise_cl(cls_fid, nside, pix_width, ell_grid, noise_level, l_knee,
                              beam_fwhm, l_cutoff, nphi_source, qe_response = qe_response,
                              measured_phi_noise = measured_phi_noise,
                              radial_mean = radial_mean)
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
                       qe_response = DEFAULT_QE_RESPONSE, phi_noise = None,
                       transfer_function = None, camb_lmax = None,
                       radial_mean = DEFAULT_RADIAL_MEAN,
                       constant_nphi = DEFAULT_CONSTANT_NPHI,
                       delensed_covariance = None, empirical_phi_noise = False,
                       empirical_phi_block = False, freeze_phi_noise = False,
                       return_applier = False):
    """The CAMB / covariance-block stencil the flat-sky-grid forecasts are built from.

    Returns (names, steps, weights, blocks_fid, blocks_plus, blocks_minus): the sampled
    parameter names, their FD steps, the real-DOF weights w_k, and the block dicts at the
    fiducial point and at theta +/- h for every sampled parameter. "blocks" contracts these
    directly; fisher_forecast_from_cls reads the same stencil so its bandpower Fisher can
    never drift from this one through a differently-built stencil, and
    fisher_forecast_full_sky's default ell_source azimuthally reduces these very blocks.

    `camb_lmax` defaults to camb_lmax_for_grid, i.e. CAMB is run far enough that every mode
    the box carries is interior to its spectra and nothing inside the grid comes from
    covar_matrix_from_cls's power-law continuation. Pass an explicit value (CAMB_LMAX
    reproduces the pre-2026-09-17 numbers) to pin it.

    `constant_nphi` (True by default) holds the reconstruction - N_phi, and for "delensed"
    the delensing fraction Alens_L - at the fiducial cosmology across the whole stencil.
    False rebuilds both at every point from that point's own Cls, so the phi block's
    log-derivative carries dN_phi/dtheta and the delensed f block is lensed with a
    theta-dependent residual fraction. See DEFAULT_CONSTANT_NPHI for what that number means;
    it is not compatible with nphi_source = "measured".

    `delensed_covariance` (spectra = "delensed" only) is a merge_delensed_covariance.py npz.
    Its empirical matrices REPLACE the f_delensed block's signal at the centre and at every
    +/- h point (C_n is still added analytically), and its steps h_i replace FD_STEP_FRAC's for
    every block, since the phi block must be differenced over the same interval. CAMB's
    delensed spectrum is then computed but never contracted. Incompatible with
    `transfer_function`, which corrects the very spectrum this replaces.

    `empirical_phi_noise` (needs `delensed_covariance` with phi moments) replaces the phi
    block's N_phi by the per-mode map_joint noise measured in the same jobs - frozen at
    theta_0 under constant_nphi, measured at every stencil point otherwise. See
    _apply_empirical_phi_noise.

    `empirical_phi_block` (needs `delensed_covariance` with the merged `phi_auto_*` grids)
    replaces the WHOLE phi block by the realization-mean auto-power of the reconstruction,
    <|phi_hat|^2> per mode, at the centre and every +/- point - no truth, no C_phi + N split,
    the phi analog of the empirical f_delensed block. `constant_nphi` does not apply to it:
    each point's measured auto-power is used as it is. Exclusive with `empirical_phi_noise`.

    `freeze_phi_noise` (with `empirical_phi_block` only) splits that auto-power into its
    truth-correlated part <B>^2/<D> = rho^2 C_phi and its noise part <A> - <B>^2/<D> =
    Var(n), and uses the signal part at every stencil point plus the noise part at theta_0:
    the reconstruction's own noise, in its own units, held fixed in theta.

    `return_applier` (needs `delensed_covariance`) also returns the function that swaps a set
    of per-mode grids into copies of this stencil's base blocks, which forecast_jackknife
    calls once per leave-one-out set.
    """
    if spectra not in SPECTRA_MODES:
        raise ValueError(f"spectra must be one of {SPECTRA_MODES}, got {spectra!r}")

    #an empirical N_L^eff is a MEASUREMENT taken at one cosmology, so there is no theta
    #dependence in it to differentiate - varying it would silently re-apply the same
    #spectrum at every stencil point and report a dN_phi/dtheta of exactly zero, which
    #looks like a physical result rather than the missing input it is
    if not constant_nphi and nphi_source == "measured":
        raise ValueError(
            "constant_nphi = False cannot be combined with nphi_source = 'measured'. The "
            "empirical N_L^eff from merge_phi_noise.py was measured at a single cosmology, "
            "so it carries no theta dependence to finite-difference and every stencil point "
            "would see the identical spectrum. Either freeze the reconstruction, or pick a "
            "source that is computed from the Cls ('covariance' or 'hu_okamoto').")

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

    #the measured N_eff replaces the quadratic estimator's N_phi in the phi block of EVERY
    #spectra mode, and additionally sets Alens_L for "delensed" - the same two roles
    #nphi_source has played since 2026-09-14. It is independent of the transfer function
    #above: R corrects the delensed temperature spectrum, N_eff the reconstruction noise, and
    #a "delensed" run may carry either, both, or neither
    measured_phi_noise = None
    if phi_noise is not None:
        if nphi_source != "measured":
            raise ValueError(f"--phi_noise was given but nphi_source is {nphi_source!r}, so "
                             f"the measurement would be parsed and then ignored. Pass "
                             f"--nphi_source measured to use it.")
        measured_phi_noise = load_phi_noise(phi_noise)
        check_phi_noise(measured_phi_noise, nside, theta_pix, noise_level, l_knee,
                        path = phi_noise)

    names, steps, fracs = sampled_names_and_steps(is_sampled, param_ground, step_fracs)
    if transfer is not None:
        check_transfer_derivatives(transfer, param_ground, names, path = transfer_function)

    empirical = None
    if empirical_phi_block and empirical_phi_noise:
        raise ValueError("pass empirical_phi_block OR empirical_phi_noise, not both: the "
                         "first replaces the whole phi block by <|phi_hat|^2>, the second "
                         "only its noise term")
    if freeze_phi_noise and not empirical_phi_block:
        raise ValueError("freeze_phi_noise only applies to empirical_phi_block: it freezes "
                         "the noise part of the measured <|phi_hat|^2>")
    if empirical_phi_block and delensed_covariance is None:
        raise ValueError("empirical_phi_block needs delensed_covariance: <|phi_hat|^2> is "
                         "measured by the get_delensed_covariance.sh jobs and lives in the "
                         "same merged npz")
    if empirical_phi_noise and delensed_covariance is None:
        raise ValueError("empirical_phi_noise needs delensed_covariance: the per-mode phi "
                         "noise is measured by the same get_delensed_covariance.sh jobs and "
                         "lives in the same merged npz")
    if empirical_phi_noise and nphi_source == "measured":
        raise ValueError("empirical_phi_noise replaces N_phi with the per-mode measurement, "
                         "so nphi_source = 'measured' (the band N_eff of merge_phi_noise.py) "
                         "would be a second, conflicting empirical noise")
    if delensed_covariance is not None:
        if spectra != "delensed":
            raise ValueError(f"an empirical delensed covariance only applies to spectra = "
                             f"'delensed'; got {spectra!r}")
        if transfer is not None:
            raise ValueError("pass --delensed_covariance OR --transfer_function, not both: "
                             "the transfer function corrects CAMB's delensed spectrum, and "
                             "the empirical covariance replaces that spectrum outright")
        if step_fracs is not None:
            raise ValueError("the finite-difference steps are fixed by the empirical "
                             "delensed covariance's stencil, so step_fracs (and "
                             "--stability) cannot be applied")
        empirical = load_delensed_covariance(delensed_covariance)
        steps = check_delensed_covariance(empirical, nside, theta_pix, noise_level, l_knee,
                                          param_ground, names, path = delensed_covariance)
        if empirical_phi_noise and not bool(empirical.get("has_phi_moments", False)):
            raise ValueError(f"{delensed_covariance} carries no per-mode phi moments: its jobs "
                             f"predate them. Re-run get_delensed_covariance.sh and "
                             f"merge_delensed_covariance.py, or drop --empirical_phi_noise.")
        if freeze_phi_noise and "phi_signal_fid" not in empirical:
            raise ValueError(f"{delensed_covariance} carries no phi_signal / phi_hat_noise "
                             f"grids. If its job files have the phi moments, re-run "
                             f"merge_delensed_covariance.py to add them.")
        if empirical_phi_block and "phi_auto_fid" not in empirical:
            raise ValueError(f"{delensed_covariance} carries no phi_auto grids. If its job "
                             f"files have the phi moments, re-run merge_delensed_covariance.py "
                             f"to add them; otherwise re-run get_delensed_covariance.sh.")
        #the empirical f block carries whatever N_phi convention its map_joint runs used;
        #merged files from before the flag existed froze it only under "fiducial"
        measured_constant = (bool(empirical["constant_nphi"]) if "constant_nphi" in empirical
                             else str(empirical["reconstruction"]) == "fiducial")
        #(irrelevant under empirical_phi_noise: the phi block's N then IS the measurement,
        #frozen or not by constant_nphi, rather than a QE N^(0) to be matched against it)
        if (measured_constant != bool(constant_nphi) and not empirical_phi_noise
                and not empirical_phi_block):
            print(f"  WARNING: the empirical delensed covariance was measured with N_phi "
                  f"{'frozen at' if measured_constant else 'rebuilt away from'} theta_0, but "
                  f"this forecast {'freezes' if constant_nphi else 'varies'} the phi block's "
                  f"N_phi (constant_nphi = {constant_nphi}); the two blocks sit on different "
                  f"reconstruction conventions")

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
    #run CAMB far enough that the grid's corner mode is interior to its spectra, so no mode
    #the box actually carries is filled by covar_matrix_from_cls's power-law continuation
    camb_lmax = camb_lmax_for_grid(ell_grid) if camb_lmax is None else int(camb_lmax)

    def reconstruction_at(params, loud):
        """(N_phi, Alens_L) implied by the cosmology `params` - see DEFAULT_CONSTANT_NPHI.

        cls_with_qe_response rather than camb_cls_at_params so that qe_response = "gradient"
        finds its "gradient_TT" key at this point too; a no-op on the default "unlensed",
        and with constant_nphi only the fiducial point ever calls this at all.
        """
        cls = cls_with_qe_response(params, qe_response, camb_lmax = camb_lmax)
        return frozen_reconstruction(cls, spectra, nside, pix_width, ell_grid,
                                     noise_level, l_knee, beam_fwhm, l_cutoff,
                                     iterative_delens, loud,
                                     nphi_source = nphi_source,
                                     param_ground = params,
                                     qe_response = qe_response,
                                     measured_phi_noise = measured_phi_noise,
                                     camb_lmax = camb_lmax,
                                     radial_mean = radial_mean)

    nphi_fid, alens_fid = reconstruction_at(param_ground, verbose)

    def blocks_at(params):
        #with constant_nphi the frozen per-L Alens gives every stencil point its own
        #CAMB-delensed TT and the frozen N_phi sets the phi block's noise floor; without it
        #both are rebuilt from this point's own Cls, so dln(C_phi + N_phi)/dtheta picks up
        #dN_phi/dtheta and the delensed spectrum is lensed with a theta-dependent residual
        #fraction. The frozen R rescales the delensed spectrum onto the box's own lensing
        #calculation either way. N_phi comes back so the caller can report how far it moved
        nphi, alens = ((nphi_fid, alens_fid) if constant_nphi
                       else reconstruction_at(params, False))
        cls = (camb_cls_at_params(params, camb_lmax = camb_lmax) if alens is None
               else delensed_cls_at_params(params, alens, transfer = transfer,
                                           camb_lmax = camb_lmax))
        return covariance_blocks(cls, spectra, nside, pix_width, ell_grid, noise_level,
                                 l_knee, beam_fwhm, l_cutoff, nphi), nphi

    if verbose:
        ells_on_grid = ell_grid[ell_grid > 0]
        print(f"Fisher forecast [{spectra}]: nside {nside}, theta_pix {theta_pix}', "
              f"{noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  box {nside * theta_pix / 60:.2f} deg, ell in "
              f"[{float(jnp.min(ells_on_grid)):.0f}, {float(jnp.max(ells_on_grid)):.0f}], "
              f"{int(jnp.sum(weights))} real DOF")
        print(f"  CAMB to l = {camb_lmax - 1} (grid corner "
              f"{grid_max_ell(ell_grid):.0f}"
              f"{', CAMB_LMAX floor' if camb_lmax == DEFAULT_MAX_ELL else ''})")
        print(f"  sampled: {names}")
        print(f"  reconstruction: "
              + ("N_phi"
                 + (" and Alens_L" if spectra == "delensed" else "")
                 + (" frozen at the fiducial cosmology" if constant_nphi
                    else " rebuilt at every stencil point (constant_nphi = False)")))
        if transfer is not None:
            apply_transfer_function(jnp.ones(1), jnp.ones(1), transfer, verbose = True)
        if measured_phi_noise is not None:
            measured_phi_noise_cl(measured_phi_noise, np.array([100.0]), verbose = True)
        print(f"  running {2 * len(names) + 1} CAMB calls"
              + ("..." if constant_nphi
                 else f" and {2 * len(names) + 1} quadratic-estimator evaluations..."))

    blocks_fid, _ = blocks_at(param_ground)

    blocks_plus, blocks_minus = [], []
    for name, step in zip(names, steps):
        up = dict(param_ground)
        up[name] = param_ground[name] + step
        down = dict(param_ground)
        down[name] = param_ground[name] - step
        plus, nphi_plus = blocks_at(up)
        minus, nphi_minus = blocks_at(down)
        blocks_plus.append(plus)
        blocks_minus.append(minus)
        if verbose:
            print(f"    {name}: h = {step:.6g} ({step / PARAM_SIGMA[name]:g} sigma)")
            if not constant_nphi:
                #|dN_phi/dtheta * h| / N_phi, i.e. exactly the piece this flag adds to the
                #phi block's log-derivative. Compare it against dln(C_phi + N_phi) from the
                #signal alone: where C_phi dominates N_phi it is diluted by N/(C + N)
                safe = jnp.where(nphi_fid > 0, nphi_fid, 1.0)
                drift = jnp.where(nphi_fid > 0,
                                  jnp.abs(nphi_plus - nphi_minus) / (2 * safe), 0.0)
                mean = float(jnp.sum(weights * drift) / jnp.sum(weights))
                print(f"      N_phi moves over +/-h: max |dN/N| {float(jnp.max(drift)):.3%}, "
                      f"DOF-weighted mean {mean:.3%}")

    def apply_empirical(grids, loud):
        """The stencil's blocks with the empirical ones swapped in from `grids` - the merged
        npz, or one leave-one-out set of the same per-mode grids
        (delensed_covariance.leave_one_out_grids). Works on COPIES of the base blocks, so it
        can be called once per jackknife sample; the configuration (names, realization count,
        smoothing) is always read from the merged npz."""
        fid = dict(blocks_fid)
        plus = [dict(block) for block in blocks_plus]
        minus = [dict(block) for block in blocks_minus]
        measured_names = [str(name) for name in empirical["names"]]

        fid["f_delensed"] = jnp.asarray(grids["delensed_fid"]) + noise
        for i, name in enumerate(names):
            j = measured_names.index(name)
            plus[i]["f_delensed"] = jnp.asarray(grids["delensed_plus"][j]) + noise
            minus[i]["f_delensed"] = jnp.asarray(grids["delensed_minus"][j]) + noise
        if loud:
            delensed_by = str(empirical.get("delensing_phi", "map_joint"))
            print(f"  f_delensed block: EMPIRICAL ({int(empirical['n_realizations'])} "
                  f"realizations, reconstruction at the {empirical['reconstruction']} "
                  f"cosmology, delensed by "
                  + ("the Wiener-filtered true phi" if delensed_by == "wiener_truth"
                     else "map_joint's phi_hat")
                  + f") from {delensed_covariance}")

        if empirical_phi_block:
            if freeze_phi_noise:
                #signal part rho^2 C_phi at each point + the noise part Var(n) at theta_0; at
                #the centre the sum is exactly <|phi_hat|^2>(theta_0)
                frozen_noise = jnp.asarray(grids["phi_hat_noise_fid"])
                fid["phi"] = jnp.asarray(grids["phi_signal_fid"]) + frozen_noise
                for i, name in enumerate(names):
                    j = measured_names.index(name)
                    plus[i]["phi"] = jnp.asarray(grids["phi_signal_plus"][j]) + frozen_noise
                    minus[i]["phi"] = jnp.asarray(grids["phi_signal_minus"][j]) + frozen_noise
            else:
                fid["phi"] = jnp.asarray(grids["phi_auto_fid"])
                for i, name in enumerate(names):
                    j = measured_names.index(name)
                    plus[i]["phi"] = jnp.asarray(grids["phi_auto_plus"][j])
                    minus[i]["phi"] = jnp.asarray(grids["phi_auto_minus"][j])
            if loud:
                smoothing = float(empirical.get("phi_smooth_delta_ell", 0.0))
                print(f"  phi block: EMPIRICAL <|phi_hat|^2> per mode"
                      + (" (signal part <B>^2/<D> at every stencil point, noise part frozen "
                         "at theta_0)" if freeze_phi_noise else " at every stencil point")
                      + " "
                      f"({int(empirical['n_realizations'])} realizations, reconstruction at "
                      f"the {empirical['reconstruction']} cosmology"
                      + (f", smoothed over |L| bands of {smoothing:g}" if smoothing > 0
                         else "")
                      + "); constant_nphi does not apply")

        if empirical_phi_noise:
            _apply_empirical_phi_noise(grids, names, steps, param_ground, fid, plus, minus,
                                       nside, pix_width, ell_grid, weights, camb_lmax,
                                       constant_nphi, loud, measured_names = measured_names,
                                       smoothing = float(empirical.get("phi_smooth_delta_ell",
                                                                       0.0)))
        return fid, plus, minus

    if empirical is not None:
        #the same analytic C_n covariance_blocks adds, on the same multipole axis
        cls_fid = camb_cls_at_params(param_ground, camb_lmax = camb_lmax)
        noise, _, _ = _instrument_matrices(nside, pix_width, ell_grid, noise_level, l_knee,
                                           beam_fwhm, l_cutoff,
                                           lmax_prime = 2 + cls_fid["total_TT"].shape[0])
        result = apply_empirical(empirical, verbose)
        if return_applier:
            return (names, steps, weights) + result + (apply_empirical,)
        return (names, steps, weights) + result
    if return_applier:
        raise ValueError("return_applier needs delensed_covariance: there is no empirical "
                         "block to resample otherwise")

    return names, steps, weights, blocks_fid, blocks_plus, blocks_minus


def _apply_empirical_phi_noise(empirical, names, steps, param_ground, blocks_fid, blocks_plus,
                               blocks_minus, nside, pix_width, ell_grid, weights, camb_lmax,
                               constant_nphi, verbose, measured_names = None,
                               smoothing = None):
    """Rebuild the "phi" block at every stencil point from the per-mode empirical noise.

    The merged npz carries r_k^2 = <B>^2 / (<A><D>) per mode at theta_0 and theta_0 +/- h_i
    (cmb_lensing/delensed_covariance.py), and C_phi + N_eff = C_phi / r^2 per mode. C_phi is
    this forecast's own CAMB C_phi at each point, so the signal is exactly what every other
    block uses:
      constant_nphi = True   N_eff frozen at theta_0:  C_phi(theta) + C_phi(0) (1/r_0^2 - 1)
      constant_nphi = False  N_eff measured at theta:  C_phi(theta) / r^2(theta), per mode
    A mode with no measurement (mean cross power <= 0 at the centre - or, when varying, at
    either end of that parameter's step) is given NO phi information: its +/- blocks are set
    equal to the centre, so its log-derivative is zero. Edits the block dicts in place.
    """
    #the configuration comes separately when `empirical` is a leave-one-out grid set, which
    #carries only the grids
    if measured_names is None:
        measured_names = [str(name) for name in empirical["names"]]
    if smoothing is None:
        smoothing = float(empirical.get("phi_smooth_delta_ell", 0.0))

    def cphi_at(params):
        cls = camb_cls_at_params(params, camb_lmax = camb_lmax)
        ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)
        return covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cls["phi"],
                                     origin_value = 0)

    def safe(r_squared, mask):
        return jnp.where(mask, jnp.asarray(np.nan_to_num(r_squared, nan = 1.0)), 1.0)

    mask_fid = jnp.asarray(empirical["phi_measured_fid"], dtype = bool)
    r2_fid = safe(empirical["phi_r2_fid"], mask_fid)
    cphi_fid = cphi_at(param_ground)
    noise_fid = cphi_fid * (1.0 / r2_fid - 1.0)
    #unmeasured modes keep the block they had (CAMB C_phi + the QE N_phi): with plus = minus
    #= centre below, its value only enters through modes that carry no information anyway
    block_fid = jnp.where(mask_fid, cphi_fid + noise_fid, blocks_fid["phi"])
    blocks_fid["phi"] = block_fid

    dropped = []
    for i, (name, step) in enumerate(zip(names, steps)):
        j = measured_names.index(name)
        up, down = dict(param_ground), dict(param_ground)
        up[name] = param_ground[name] + step
        down[name] = param_ground[name] - step
        if constant_nphi:
            mask = mask_fid
            plus = cphi_at(up) + noise_fid
            minus = cphi_at(down) + noise_fid
        else:
            #each stencil point's own per-mode N_eff: C_phi(theta) / r^2(theta)
            mask_plus = jnp.asarray(empirical["phi_measured_plus"][j], dtype = bool)
            mask_minus = jnp.asarray(empirical["phi_measured_minus"][j], dtype = bool)
            mask = mask_fid & mask_plus & mask_minus
            plus = cphi_at(up) / safe(empirical["phi_r2_plus"][j], mask_plus)
            minus = cphi_at(down) / safe(empirical["phi_r2_minus"][j], mask_minus)
        blocks_plus[i]["phi"] = jnp.where(mask, plus, block_fid)
        blocks_minus[i]["phi"] = jnp.where(mask, minus, block_fid)
        dropped.append(1 - float(jnp.sum(weights * mask) / jnp.sum(weights)))

    if verbose:
        print(f"  phi block: EMPIRICAL per-mode N_eff"
              + (f" (moments smoothed over |L| bands of {smoothing:g})" if smoothing > 0
                 else "")
              + (", frozen at theta_0" if constant_nphi
                 else ", measured at every stencil point (theta-dependent)")
              + "; DOF without a measurement (no phi information): "
              + ", ".join(f"{name} {fraction:.1%}" for name, fraction in zip(names, dropped)))


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
             phi_noise = None, transfer_function = None, camb_lmax = None,
             radial_mean = DEFAULT_RADIAL_MEAN,
             constant_nphi = DEFAULT_CONSTANT_NPHI, delensed_covariance = None,
             empirical_phi_noise = False, empirical_phi_block = False,
             freeze_phi_noise = False):
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
        nphi_source:  where N_phi comes from - the phi block in every spectra mode, and for
                      "delensed" also the N_L behind Alens_L. "covariance" (the box's QE
                      matrix, default), "hu_okamoto" (the analytic N^(0) quadrature) or
                      "measured" (the empirical N_L^eff) - see NPHI_SOURCES
        qe_response:  which TT spectrum the quadratic estimator's response is built from,
                      "unlensed" (default, as before) or "gradient" - see
                      QE_RESPONSE_SOURCES
        phi_noise:    path to a merge_phi_noise.py effective_phi_noise.npz, required by and
                      only valid with nphi_source = "measured". Its N_L^eff is the noise
                      map_joint's MAP reconstruction actually achieves on this box, so the
                      forecast weights the phi block - and delenses the temperature block -
                      by what THIS codebase reconstructs rather than by a quadratic
                      estimator's N^(0). See load_phi_noise and cmb_lensing/phi_noise.py
        transfer_function: "delensed" only: path to a merge_delensed_spectra.py npz. Rescales
                      CAMB's delensed spectrum at every stencil point by the empirically
                      measured R(l), so the forecast contracts the delensed spectrum THIS
                      codebase's lense_flow and map_joint actually produce rather than
                      CAMB's - see load_transfer_function and cmb_lensing/delensed_spectrum.py.
                      Independent of `phi_noise`: that one replaces the reconstruction NOISE,
                      this one corrects the delensed TEMPERATURE spectrum
        camb_lmax:    how far CAMB is run. Defaults to camb_lmax_for_grid(ell_grid), which
                      covers the box's corner mode so no grid mode is filled by a power-law
                      continuation of CAMB's last two multipoles; pass CAMB_LMAX to
                      reproduce the pre-2026-09-17 numbers
        radial_mean:  how _radial_cl_profile combines the 2D N_phi inside each annulus,
                      "geometric" (default) or "arithmetic" - see RADIAL_MEAN_TYPES. Applies
                      to nphi_source = "covariance", the only one that starts from a matrix
        constant_nphi: True (default) freezes the reconstruction at the fiducial cosmology
                      across the whole finite-difference stencil - N_phi, and for
                      "delensed" the delensing fraction Alens_L - so that only the signal
                      carries theta dependence. False rebuilds both at every stencil point,
                      so dln(C_phi + N_phi)/dtheta gains dN_phi/dtheta and the delensed f
                      block is lensed with a theta-dependent residual fraction. The varying
                      number bounds how much the freezing choice is worth; it is not a
                      better forecast, because dN_phi/dtheta lies in the span of
                      dC^TT/dtheta that the f block already counts, and a real QE analysis
                      removes it with a realization-dependent N0. See DEFAULT_CONSTANT_NPHI.
                      Incompatible with nphi_source = "measured"
        delensed_covariance: "delensed" only: path to a merge_delensed_covariance.py npz.
                      Its empirical per-mode delensed covariances (the box's own lense_flow
                      and map_joint, on common random numbers at theta_0 and theta_0 +/- h)
                      replace CAMB's delensed signal at every stencil point, and its h_i
                      replace FD_STEP_FRAC. See cmb_lensing/delensed_covariance.py.
                      Incompatible with transfer_function and step_fracs
        empirical_phi_noise: with delensed_covariance (whose jobs stored the phi moments):
                      the phi block becomes C_phi + the per-mode map_joint noise N_eff
                      measured in the same jobs, instead of CAMB C_phi + a QE N_phi.
                      constant_nphi = True freezes N_eff at theta_0; False uses the N_eff
                      measured at each stencil point, so dN_eff/dtheta enters the Fisher
        empirical_phi_block: with delensed_covariance: the whole phi block becomes the
                      measured <|phi_hat|^2> per mode at every stencil point (no truth, no
                      C_phi + N split). Exclusive with empirical_phi_noise; constant_nphi
                      does not apply
        freeze_phi_noise: with empirical_phi_block: the block becomes the truth-correlated
                      part <B>^2/<D> of <|phi_hat|^2> at each stencil point plus its noise
                      part <A> - <B>^2/<D> at theta_0

    Returns:
        (fisher, names) - the n_sampled x n_sampled matrix and the parameter names in
        OUTPUT_PARAM_ORDER order (chain_analysis.py's order, not PARAM_ORDER).
    """
    names, steps, weights, fid, plus, minus = covariance_stencil(
        nside, theta_pix, noise_level, is_sampled, param_ground, spectra, step_fracs,
        l_knee, beam_fwhm, l_cutoff, verbose, iterative_delens = iterative_delens,
        nphi_source = nphi_source, qe_response = qe_response, phi_noise = phi_noise,
        transfer_function = transfer_function, camb_lmax = camb_lmax,
        radial_mean = radial_mean, constant_nphi = constant_nphi,
        delensed_covariance = delensed_covariance, empirical_phi_noise = empirical_phi_noise,
        empirical_phi_block = empirical_phi_block, freeze_phi_noise = freeze_phi_noise)
    return _fisher_from_blocks(plus, minus, fid, steps, weights), names


def forecast_jackknife(nside, theta_pix, noise_level, is_sampled, param_ground,
                       delensed_covariance, covariance_dir = None, spectra = "delensed",
                       l_knee = 0, beam_fwhm = 0, l_cutoff = 10_000, verbose = True,
                       **stencil_kwargs):
    """forecast(..., delensed_covariance = ...) plus its delete-one jackknife samples.

    The empirical blocks are realization means, so the forecast built from them carries Monte
    Carlo error. This recomputes the WHOLE forecast with each realization left out: the
    per-mode grids are re-formed from the delete-one means by the same
    delensed_covariance.forecast_grids the merge used, swapped into the same stencil (CAMB,
    C_n, N_phi and every non-empirical block are built once), and contracted. The caller
    inverts each sample and jackknifes whatever it needs - sigmas, correlations, angles -
    through the full nonlinearity of inversion and marginalization.

    `covariance_dir` is the directory of per-realization job files the merged npz was made
    from; by default the npz's own directory (where merge_delensed_covariance.py writes it).
    The two must hold the same realizations (checked by seed). Any smoothing the merge applied
    (`phi_smooth_delta_ell`) is re-applied to every delete-one set. `stencil_kwargs` are
    covariance_stencil's keyword arguments (nphi_source, camb_lmax, constant_nphi,
    empirical_phi_block, freeze_phi_noise, empirical_phi_noise, ...).

    Returns (fisher, names, leave_one_out) with `leave_one_out` an
    (n_realizations, n_sampled, n_sampled) stack of delete-one Fisher matrices.
    """
    from cmb_lensing.delensed_covariance import leave_one_out_grids

    if covariance_dir is None:
        covariance_dir = os.path.dirname(os.path.abspath(delensed_covariance))
    names, steps, weights, fid, plus, minus, apply_empirical = covariance_stencil(
        nside, theta_pix, noise_level, is_sampled, param_ground, spectra, None, l_knee,
        beam_fwhm, l_cutoff, verbose, delensed_covariance = delensed_covariance,
        return_applier = True, **stencil_kwargs)
    fisher = _fisher_from_blocks(plus, minus, fid, steps, weights)

    merged = load_delensed_covariance(delensed_covariance)
    seeds, samples = leave_one_out_grids(
        covariance_dir, smooth_delta_ell = float(merged.get("phi_smooth_delta_ell", 0.0)),
        verbose = False)
    merged_seeds = [int(seed) for seed in merged["seeds"]]
    if seeds != merged_seeds:
        raise ValueError(f"{covariance_dir} holds realizations (seeds) different from those "
                         f"{delensed_covariance} was merged from ({len(seeds)} vs "
                         f"{len(merged_seeds)}); re-run merge_delensed_covariance.py on that "
                         f"directory so the jackknife resamples the same set")

    leave_one_out = []
    for grids in samples:
        loo_fid, loo_plus, loo_minus = apply_empirical(grids, False)
        leave_one_out.append(_fisher_from_blocks(loo_plus, loo_minus, loo_fid, steps, weights))
    if verbose:
        print(f"  jackknife: {len(leave_one_out)} delete-one forecasts from {covariance_dir}")
    return fisher, names, np.array(leave_one_out)


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
                               "C_phi; 'measured' is the empirical N_L^eff of map_joint's "
                               "own MAP reconstruction, which needs --phi_noise")
    add_qe_response_argument(parser)
    parser.add_argument("--stability", action = "store_true",
                        help = "also recompute at 2x the step size and report the drift")
    return parser


def add_constant_nphi_argument(parser):
    """--vary_nphi: let the reconstruction move with the cosmology across the stencil.

    Its own helper rather than a line in add_spectra_arguments because only the modules that
    actually thread it through to covariance_stencil should advertise it - a flag that parses
    and is then ignored is worse than no flag. fisher_forecast's main() calls this; a sibling
    adopting it needs the same call plus one pass-through into covariance_stencil.
    """
    parser.add_argument("--vary_nphi", action = "store_true",
                        help = "rebuild the quadratic-estimator noise N_phi - and, for "
                               "--spectra delensed, the delensing fraction Alens_L - at "
                               "every finite-difference point instead of freezing both at "
                               "the fiducial cosmology (constant_nphi = False). The phi "
                               "block's log-derivative then carries dN_phi/dtheta. That "
                               "derivative is real, but it is not independent information: "
                               "N^(0) is a deterministic functional of the estimator's TT "
                               "filter and response, so it lies in the span of "
                               "dC^TT/dtheta, which the f block already counts - and a real "
                               "analysis removes it with a realization-dependent N0. Use it "
                               "to bound how much the frozen-N_phi choice is worth, not as "
                               "the production number. Rejected with --nphi_source measured")
    return parser


def add_phi_noise_argument(parser):
    """--phi_noise: the empirical effective reconstruction noise, for --nphi_source measured.

    Its own helper rather than a line in add_spectra_arguments because only the modules that
    actually thread it through to covariance_stencil should advertise it - a flag that parses
    and is then ignored is worse than no flag. fisher_forecast's main() calls this; a sibling
    adopting it needs the same call plus one pass-through into covariance_stencil.
    """
    parser.add_argument("--phi_noise", type = str, default = None,
                        help = "path to a merge_phi_noise.py effective_phi_noise.npz. With "
                               "--nphi_source measured, the phi block's N_phi - and, for "
                               "--spectra delensed, the delensing fraction Alens_L - come "
                               "from the noise map_joint's MAP reconstruction actually "
                               "achieves rather than from a quadratic estimator's N^(0). "
                               "Rejected with any other --nphi_source, and rejected if it "
                               "was measured on a different box than this run")
    return parser


def add_radial_mean_argument(parser):
    """--radial_mean: how the 2D N_phi is collapsed onto a 1D ell axis.

    Only applies to nphi_source = "covariance", the one source that starts from a matrix;
    "hu_okamoto" is a 1D quadrature already and "measured" comes binned off disk. Its own
    helper for the same reason --camb_lmax is: only the modules threading it should offer it.
    """
    parser.add_argument("--radial_mean", type = str, default = DEFAULT_RADIAL_MEAN,
                        choices = RADIAL_MEAN_TYPES,
                        help = "annulus average inside _radial_cl_profile. 'geometric' "
                               "(default) is the mean of log N at the geometric-mean l, "
                               "which is exact for the local power law the log-log "
                               "interpolation assumes; 'arithmetic' is the pre-2026-09-17 "
                               "behaviour and biases N_phi high by ~3.6%%")
    return parser


def add_camb_lmax_argument(parser):
    """--camb_lmax: how far CAMB is run, defaulting to the box's own corner mode.

    Its own helper for the same reason --transfer_function is: only the modules that thread
    it into covariance_stencil should advertise it. The siblings still take CAMB's shipped
    2..CAMB_LMAX-1 axis, so a flag on their CLIs would parse and be ignored.
    """
    parser.add_argument("--camb_lmax", type = int, default = None,
                        help = "how far to run CAMB. Default: camb_lmax_for_grid, i.e. far "
                               "enough that the rfft grid's corner mode is interior to the "
                               "spectra and no grid mode is filled by covar_matrix_from_cls's "
                               f"power-law continuation. Pass {DEFAULT_MAX_ELL} to reproduce "
                               "the pre-2026-09-17 numbers")
    return parser


def add_transfer_function_argument(parser):
    """--transfer_function: the empirical delensing correction, for --spectra delensed.

    Its own helper rather than a line in add_spectra_arguments because only the modules that
    actually thread it through to covariance_stencil should advertise it - a flag that parses
    and is then ignored is worse than no flag. fisher_forecast's main() calls this; a sibling
    adopting it needs the same call plus one pass-through into covariance_stencil.

    Independent of --phi_noise, which the same main() also offers: this one corrects the
    delensed TEMPERATURE spectrum for the difference between CAMB's lensing and the box's,
    that one replaces the reconstruction NOISE. A delensed run may pass either or both.
    """
    parser.add_argument("--transfer_function", type = str, default = None,
                        help = "path to a merge_delensed_spectra.py transfer_function.npz. "
                               "With --spectra delensed, rescales CAMB's delensed spectrum "
                               "at every stencil point by the empirically measured R(l), so "
                               "the forecast uses the delensed spectrum this codebase's "
                               "lense_flow and map_joint actually produce. If the npz also "
                               "carries dR/dtheta (merged with --shifted_dirs), R is "
                               "evaluated at each stencil point instead of held flat. "
                               "Rejected with any other --spectra, and rejected if it was "
                               "measured on a different box than this run")
    return parser


def add_delensed_covariance_argument(parser):
    """--delensed_covariance: the empirical delensed covariance stencil, for --spectra delensed.

    Its own helper for the same reason --transfer_function is: only the modules that thread
    it into covariance_stencil should advertise it.
    """
    parser.add_argument("--delensed_covariance", type = str, default = None,
                        help = "path to a merge_delensed_covariance.py "
                               "delensed_covariance.npz. With --spectra delensed, the f "
                               "block's signal at theta_0 and theta_0 +/- h is the "
                               "EMPIRICAL per-mode covariance of the map_joint-delensed "
                               "noiseless field (common random numbers across the stencil) "
                               "rather than CAMB's partially lensed spectrum, and the "
                               "finite-difference steps come from that file. Rejected with "
                               "any other --spectra, with --transfer_function, with "
                               "--stability, and if it was measured on a different box")
    parser.add_argument("--empirical_phi_noise", action = "store_true",
                        help = "with --delensed_covariance: take the phi block's noise from "
                               "the per-mode map_joint reconstruction noise measured by the "
                               "same jobs (N_eff = C_phi (1/r^2 - 1) per rfft mode) instead of "
                               "a quadratic-estimator N_phi. Frozen at the fiducial cosmology "
                               "by default; add --vary_nphi to use the N_eff measured at each "
                               "stencil point, i.e. a theta-dependent empirical noise")
    parser.add_argument("--empirical_phi_block", action = "store_true",
                        help = "with --delensed_covariance: replace the WHOLE phi block by the "
                               "measured auto-power of map_joint's reconstruction, "
                               "<|phi_hat|^2> per rfft mode, at the centre and every +/- "
                               "stencil point (no ground truth, no C_phi + N split). "
                               "Exclusive with --empirical_phi_noise; --vary_nphi does not "
                               "apply to it")
    parser.add_argument("--freeze_phi_noise", action = "store_true",
                        help = "with --empirical_phi_block: split <|phi_hat|^2> into its "
                               "truth-correlated part <B>^2/<D> (B = Re phi_hat phi*, "
                               "D = |phi|^2) and its noise part <A> - <B>^2/<D>, and use the "
                               "signal part at every stencil point plus the noise part at "
                               "the fiducial cosmology")
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
    #only marked when it is NOT the default, so every figure produced so far is unchanged
    varying = "  |  N_phi(theta)" if getattr(args, "vary_nphi", False) else ""
    return (f"{spectra}  |  nside {args.nside}, {args.theta_pix:g}', "
            f"{args.noise:g} uK-arcmin{varying}")


def run_config(spectra, args, names, **extra):
    """The run configuration stored in every fisher<suffix>.npz."""
    step_fracs = np.array([FD_STEP_FRAC[name] for name in names])
    #an empirical delensed covariance fixes the steps to the ones it was measured at
    if getattr(args, "delensed_covariance", None) and spectra == "delensed":
        merged = load_delensed_covariance(args.delensed_covariance)
        measured = [str(name) for name in merged["names"]]
        step_fracs = np.array([float(merged["steps"][measured.index(name)]) /
                               PARAM_SIGMA[name] for name in names])
    return dict(spectra = spectra, nside = args.nside, theta_pix = args.theta_pix,
                noise_level = args.noise, l_knee = args.l_knee, beam_fwhm = args.beam_fwhm,
                qe_response = getattr(args, "qe_response", DEFAULT_QE_RESPONSE),
                #True for a module that does not offer --vary_nphi, which is what the
                #siblings' frozen reconstruction actually is
                constant_nphi = not getattr(args, "vary_nphi", False),
                #np.savez cannot store None, so "no empirical measurement" is the empty string
                phi_noise = getattr(args, "phi_noise", None) or "",
                transfer_function = getattr(args, "transfer_function", None) or "",
                delensed_covariance = getattr(args, "delensed_covariance", None) or "",
                empirical_phi_noise = bool(getattr(args, "empirical_phi_noise", False)),
                empirical_phi_block = bool(getattr(args, "empirical_phi_block", False)),
                freeze_phi_noise = bool(getattr(args, "freeze_phi_noise", False)),
                #0 means "not pinned", i.e. camb_lmax_for_grid chose it from the box
                camb_lmax = getattr(args, "camb_lmax", None) or 0,
                radial_mean = getattr(args, "radial_mean", DEFAULT_RADIAL_MEAN),
                step_fracs = step_fracs, **extra)


def main():
    parser = argparse.ArgumentParser(
        description = "Gaussian Fisher forecast for the LCDM parameters on the "
                      "sample_lcdm.py flat-sky box - the covariance-block trace formula")
    add_box_arguments(parser)
    add_spectra_arguments(parser)
    add_constant_nphi_argument(parser)
    add_phi_noise_argument(parser)
    add_transfer_function_argument(parser)
    add_delensed_covariance_argument(parser)
    add_camb_lmax_argument(parser)
    add_radial_mean_argument(parser)
    args = parser.parse_args()
    is_sampled = sampled_from_args(parser, args)
    if args.delensed_covariance and args.stability:
        parser.error("--stability cannot be combined with --delensed_covariance: the step is "
                     "fixed by the measured stencil. Re-run get_delensed_covariance.sh at a "
                     "second step_sigma and compare the two forecasts instead")

    def run(step_fracs = None, spectra = args.spectra, verbose = True):
        #the measured N_phi applies in every spectra mode, so it passes through to the
        #ceiling comparison below unchanged; the transfer function does not, since it is
        #rejected outside "delensed" - it only ever corrects the delensed spectrum
        return forecast(args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
                        spectra = spectra, step_fracs = step_fracs, l_knee = args.l_knee,
                        beam_fwhm = args.beam_fwhm, verbose = verbose,
                        iterative_delens = args.iterative_delens,
                        nphi_source = args.nphi_source,
                        qe_response = args.qe_response,
                        phi_noise = args.phi_noise,
                        transfer_function = (args.transfer_function
                                             if spectra == "delensed" else None),
                        camb_lmax = args.camb_lmax,
                        radial_mean = args.radial_mean,
                        constant_nphi = not args.vary_nphi,
                        delensed_covariance = (args.delensed_covariance
                                               if spectra == "delensed" else None),
                        empirical_phi_block = (args.empirical_phi_block
                                               and spectra == "delensed"),
                        freeze_phi_noise = (args.freeze_phi_noise
                                            and spectra == "delensed"),
                        empirical_phi_noise = (args.empirical_phi_noise
                                               and spectra == "delensed"))

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
