"""Gaussian Fisher forecast for the LCDM parameters on sample_lcdm.py's flat-sky box.

This answers "what is the best a power-spectrum analysis of this data set could do?" -
the baseline that sample_joint should beat, since the Gibbs sampler works with the full
hierarchical likelihood (phi ~ N(0, Cphi), f ~ N(0, Cf), d = M B L(phi) f + n) and
therefore also sees the lensing-induced non-Gaussianity that a 2-point analysis discards.

The forecast is computed on the SAME flat-sky rfft grid the sampler runs on (gen_ell_grid
at the given nside / theta_pix), not with a full-sky (2l+1) sum, so the box geometry, the
finite ell_min = 2*pi/L and the mode count all match the real run.

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

Three spectra modes, bracketing the answer. Every mode carries a lensing block
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
             the UNLENSED field would carry, not what the instrument returns. Its value is
             as a validation target: it is the ensemble average of Louis's first term, so a
             Monte-Carlo estimate of -<d2/dtheta2 log p(f, phi | theta)> over prior draws
             must reproduce it. With the noise terms in it is no longer a strict upper
             bound on the other two modes - at nside 64 / 5 uK-arcmin it now sits only
             ~1.0-1.2x above "lensed" per parameter, where the noiseless version was far
             looser.

WHAT THIS IS NOT. None of these is the marginal Fisher of p(d | theta) = the integral of
p(d, f, phi | theta) over the fields, which is what sample_joint actually targets. That
one cannot be evaluated pointwise; it needs Louis's identity

    I_marg = E[-d2 log p(d, x | theta)] - Cov[d log p(d, x | theta)]

with BOTH expectations over the conditional p(f, phi | d, theta) - i.e. over draws from
the Gibbs sampler at fixed theta, NOT over prior draws. Prior draws give "ceiling" above.
The ratio F_ceiling / F_lensed printed by __main__ is a useful advance warning of how
severe the cancellation in that subtraction will be.

TWO FISHER CODE PATHS. Both consume the SAME finite-difference stencil (one set of
2 * n_sampled + 1 CAMB runs, one set of covariance blocks), and both sum over blocks -
with spectra = "ceiling" that sum is exactly F_total = F_field + F_phi. They differ only
in what is contracted:

  "blocks" (the original) F_ij = 1/2 sum_k w_k dln(C_k)/dtheta_i dln(C_k)/dtheta_j.
           The standard zero-mean Gaussian Fisher derived above - the information the
           MAP-LEVEL likelihood log p(x | theta) carries about theta, with the two powers
           of C^-1 that the trace formula demands.
  "cls"    F_ij = (dC/dtheta_i) . C^-1 . (dC/dtheta_j), i.e. a matrix-vector product
           against the fiducial inverse covariance, contracted with the other derivative
           vector: sum_k w_k (dC_k/dtheta_i) C_k^-1 (dC_k/dtheta_j). C is diagonal in the
           Fourier basis, so C^-1 v is an entrywise divide, and w_k again counts the real
           DOF each rfft entry stands for. C^-1 is frozen at the fiducial point; only the
           derivative vectors move.

           CAVEAT, read before quoting a number from this path: it carries ONE power of
           C^-1 where the trace formula carries two, so it is not the Gaussian Fisher of
           the same likelihood and it is not dimensionless in the same way. Under a
           theta-independent rescaling C -> a C (and covar_matrix_from_cls does apply one,
           the 1/pix_width**2 factor) the "blocks" Fisher is invariant while this one
           scales as a. Its ABSOLUTE sigmas therefore inherit that arbitrary
           normalization; what is meaningful and rescaling-invariant is its SHAPE - the
           correlation matrix, the degeneracy directions, the relative ordering of the
           parameters. Compare correlation_matrix_from_cls.png against
           correlation_matrix_from_blocks.png, not the two sigma columns.

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
    python -m cmb_lensing.fisher_forecast                       #nside 64, T-only, 5 uK-arcmin
    python -m cmb_lensing.fisher_forecast --spectra ceiling
    python -m cmb_lensing.fisher_forecast --nside 128 --noise 1.0 --stability

Writes one set of outputs per code path into cmb_lensing/fisher_output/, tagged
"_from_blocks" or "_from_cls":
    fisher_matrix_from_{blocks,cls}.png       F_ij
    covariance_matrix_from_{blocks,cls}.png   F^-1 and the marginalized sigmas
    correlation_matrix_from_{blocks,cls}.png  F^-1 normalized to unit diagonal
    fisher_from_{blocks,cls}.npz              all three plus the run configuration
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.simulate import (_camb_via_callback, _extract_all_cls,
                                  covar_matrix_from_cls, noise_cls, get_beam, get_mask,
                                  scalar_quadratic_estimate, get_g_matrix_lcdm,
                                  get_d_tt_matrix, load_sim)
from cmb_lensing.statistics import logpdf, mixed_logpdf
from cmb_lensing.mixing import mix
from cmb_lensing.fields import dot
from cmb_lensing.matrix_operators import pinv, log_det
from cmb_lensing.constants import (DEFAULT_MAX_ELL, DEFAULT_A_LENSE, DEFAULT_K_PIVOT,
                                   DEFAULT_MNU, DEFAULT_TAUREIO, NPHI_FAC)
from cmb_lensing.precompute_camb_1d import (PARAM_ORDER, GROUND_TRUTH, PARAM_SIGMA,
                                            CAMB_LMAX)


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

SPECTRA_MODES = ("lensed", "unlensed", "ceiling")

#the two Fisher contractions - see "TWO FISHER CODE PATHS" in the module docstring. both
#read the same stencil; only _fisher_from_blocks / _fisher_from_cls differ
FISHER_METHODS = ("blocks", "cls")

#methods that need (data, f, phi) realizations rather than a covariance stencil, so they
#cannot share forecast_all's single-stencil path
REALIZATION_METHODS = ("logpdf", "mixed")

#suffix appended to every output file so the paths never overwrite each other.
#"marginal" is written by marginal_fisher.py, not by this module's CLI - it is
#deliberately absent from FISHER_METHODS, which drives --method
METHOD_SUFFIX = {"blocks": "_from_blocks", "cls": "_from_cls",
                 "logpdf": "_from_logpdf", "mixed": "_from_mixed_logpdf",
                 "marginal": "_marginal"}

#CAMB is deterministic in the parameters, and the stencil re-queries the fiducial point
#for every parameter, so memoize whole cls dicts across calls
_CAMB_CACHE = {}


def output_dir():
    #mirrors sample_lcdm.py's sample_lcdm_output convention: a directory next to the
    #package modules, so the same relative layout works on the laptop and the cluster
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisher_output")


# ── CAMB ──────────────────────────────────────────────────────────────────

def camb_cls_at_params(params):
    """Full cls dict from one CAMB run at an arbitrary 5-parameter point.

    precompute_camb_1d.camb_cls_at only moves a single parameter off GROUND_TRUTH and
    only returns (TT, PP); the Fisher stencil needs an arbitrary point and the lensed
    spectra too. Fixed (non-sampled) CAMB parameters match load_sim's defaults:
    H0 = None (solved from cosmomc_theta), r = 0, mnu = 0.06, tau = 0.05, nt = 0,
    k_pivot = 0.05, Alens = 1. Cls land on ells 2..CAMB_LMAX-1, pure CAMB values with no
    extrapolation - the same support as load_sim's data-map path.
    """
    key = tuple(round(float(params[name]), 12) for name in PARAM_ORDER)
    if key in _CAMB_CACHE:
        return _CAMB_CACHE[key]

    cosmomc_theta = params["theta_MC_100"] / 100
    As = np.exp(params["logA"]) * 1e-10
    unlensed_scalar, tensor, total, lens_potential = _camb_via_callback(
        None, params["ombh2"], params["omch2"], cosmomc_theta, 0.0, DEFAULT_MNU, DEFAULT_TAUREIO,
        As, 0, params["ns"], CAMB_LMAX, DEFAULT_K_PIVOT, DEFAULT_A_LENSE
    )
    cls = _extract_all_cls(unlensed_scalar, tensor, total, lens_potential,
                           CAMB_LMAX, CAMB_LMAX)

    #_camb_callback_fn swallows CAMB failures and returns NaN arrays (so a sampler
    #proposal rejects instead of crashing). here that silently poisons the whole matrix,
    #so fail loudly with the offending point
    if not bool(jnp.all(jnp.isfinite(cls["total_TT"]))):
        raise RuntimeError(
            f"CAMB returned non-finite Cls at {dict((k, float(params[k])) for k in PARAM_ORDER)}. "
            f"check the stencil point is reachable by CAMB (H0 is solved from "
            f"cosmomc_theta inside DEFAULT_THETA_H0_RANGE), or shrink FD_STEP_FRAC "
            f"for that parameter."
        )

    _CAMB_CACHE[key] = cls
    return cls


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
                    l_cutoff):
    """N_phi: the scalar quadratic-estimate reconstruction noise at this cosmology.

    Built exactly the way sample_lcdm.py builds it (unlensed Cf, lensed Cfl, the same mask
    and beam, and the same division by NPHI_FAC), so the phi / PP block below carries the
    very reconstruction noise the sampler's G matrix and phi mass matrix are built from.
    Temperature-only, matching everything else in this module.
    """
    ells = jnp.arange(2, 2 + cls["total_TT"].shape[0]).astype(jnp.float64)
    noise, mask, beam = _instrument_matrices(nside, pix_width, ell_grid, noise_level,
                                             l_knee, beam_fwhm, l_cutoff)

    def covar(cl):
        return covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl,
                                     origin_value = 0)

    return scalar_quadratic_estimate(noise, covar(cls["scalar_TT"]),
                                     covar(cls["total_TT"]), mask, beam,
                                     pix_width) / NPHI_FAC


def covariance_blocks(cls, spectra, nside, pix_width, ell_grid,
                      noise_level, l_knee, beam_fwhm, l_cutoff, nphi):
    """The theta-dependent covariance block(s) whose Fisher information we are counting.

    Every block is built through the same covar_matrix_from_cls the sampler uses, so the
    signal and noise share a normalization (the 1/pix_width**2 rescale). The trace formula
    is invariant under any theta-independent rescaling of C, so only that relative
    normalization matters.

    `nphi` is the QE reconstruction noise from qe_noise_matrix, evaluated ONCE at the
    fiducial cosmology and passed in frozen. Freezing it is deliberate: N_phi is a property
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

    if spectra == "ceiling":
        #conditional on (f, phi) the data term carries no theta dependence at all, so the
        #complete-data information is the two priors - but the fields are known only to
        #within the noise, so C_f picks up C_n and C_phi picks up N_phi. No beam or mask
        #on the f block: this is what a measurement of the UNLENSED field would carry,
        #not what the instrument returns
        return {"f_unlensed": covar(cls["scalar_TT"], ells) + noise,
                "phi": phi_block}

    #load_sim forms data = mask * beam * lensed + noise, so the observed covariance
    #carries (mask * beam)**2 on the signal only
    source = "total_TT" if spectra == "lensed" else "scalar_TT"
    return {"TT": (mask * beam)**2 * covar(cls[source], ells) + noise,
            "PP": phi_block}


def _stencil_offsets(n_param):
    """Central-difference stencil points as offset tuples in units of h.

    (0,)*n for the centre, +/-1 in one slot for the first derivatives, and the four
    (+/-1, +/-1) corners of each (i, j) pair for the mixed second derivatives. That is
    1 + 2n + 4*n*(n-1)/2 points - 19 for the usual three parameters.
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


def _fisher_from_cls(blocks_plus, blocks_minus, blocks_fid, steps, weights):
    """F_ij = (dC/dtheta_i) . C^-1 . (dC/dtheta_j), summed over blocks.

    The alternative contraction to _fisher_from_blocks. Written out on this codebase's
    array layout, where C is diagonal in the Fourier basis:

        F_ij = sum_blocks sum_k w_k (dC_k/dtheta_i) C_k^-1 (dC_k/dtheta_j)

    read as the matrix-vector product it is: C^-1 acts on the dtheta_j derivative vector
    first (an entrywise divide, since C is diagonal), and the result is contracted with
    the dtheta_i derivative vector under the w_k DOF weights - the same weights and the
    same inner product statistics.logpdf uses, so each rfft entry is counted once per
    independent real mode it stands for.

    C^-1 is the FIDUCIAL covariance, held fixed; only the derivative vectors move across
    the stencil. The derivatives are central finite differences of the same blocks the
    other path uses, and since the noise / N_phi terms in a block are theta-independent,
    dC/dtheta is exactly the finite-difference gradient of the Cls mapped onto the grid.

    With spectra = "ceiling" the block sum is F_total = F_field + F_phi, the two blocks
    being C_f + C_n and C_phi + N_phi.

    NOTE this is NOT algebraically the Gaussian Fisher _fisher_from_blocks computes: it
    carries one power of C^-1 rather than two, so it is not invariant under a
    theta-independent rescaling of C and its absolute normalization is not meaningful.
    See the CAVEAT in the module docstring before quoting sigmas from it.
    """
    n_param = len(steps)
    fisher = np.zeros((n_param, n_param))

    for name in blocks_fid:
        c_fid = blocks_fid[name]
        #the [0, 0] origin is set to zero by construction (origin_value = 0) and carries
        #no information; guard the division rather than letting it produce NaN
        good = c_fid > 0
        safe = jnp.where(good, c_fid, 1.0)

        derivatives = []
        for i in range(n_param):
            derivative = (blocks_plus[i][name] - blocks_minus[i][name]) / (2 * steps[i])
            derivatives.append(jnp.where(good, derivative, 0.0))

        for i in range(n_param):
            #C^-1 dC/dtheta_i: the matrix acting on the vector, before the dot product
            solved = jnp.where(good, derivatives[i] / safe, 0.0)
            for j in range(i, n_param):
                value = float(jnp.sum(weights * derivatives[j] * solved))
                fisher[i, j] += value
                if i != j:
                    fisher[j, i] += value

    return fisher


#which contraction each method name selects; both take the identical stencil arguments
_FISHER_BUILDERS = {"blocks": _fisher_from_blocks, "cls": _fisher_from_cls}


# ── Main entry point ──────────────────────────────────────────────────────

def _stencil(nside, theta_pix, noise_level, is_sampled, param_ground, spectra,
             step_fracs, l_knee, beam_fwhm, l_cutoff, verbose):
    """The CAMB / covariance finite-difference stencil both Fisher paths are built from.

    Returns (names, steps, weights, blocks_fid, blocks_plus, blocks_minus). Factored out
    of forecast() so that one set of 2 * n_sampled + 1 CAMB runs feeds both contractions
    and neither can drift from the other through a differently-built stencil.
    """
    if spectra not in SPECTRA_MODES:
        raise ValueError(f"spectra must be one of {SPECTRA_MODES}, got {spectra!r}")

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
    steps = [fracs[name] * PARAM_SIGMA[name] for name in names]

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    #w_k = independent real DOF per rfft entry; get_fourier_weights indexes the half-axis
    #(columns), so it broadcasts across rows
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))

    #the QE reconstruction noise is evaluated once, at the fiducial cosmology, and held
    #fixed across the whole finite-difference stencil - see covariance_blocks for why
    nphi_fid = qe_noise_matrix(camb_cls_at_params(param_ground), nside, pix_width,
                               ell_grid, noise_level, l_knee, beam_fwhm, l_cutoff)

    def blocks_at(params):
        return covariance_blocks(camb_cls_at_params(params), spectra, nside,
                                 pix_width, ell_grid, noise_level, l_knee,
                                 beam_fwhm, l_cutoff, nphi_fid)

    if verbose:
        ells_on_grid = ell_grid[ell_grid > 0]
        print(f"Fisher forecast [{spectra}]: nside {nside}, theta_pix {theta_pix}', "
              f"{noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  box {nside * theta_pix / 60:.2f} deg, ell in "
              f"[{float(jnp.min(ells_on_grid)):.0f}, {float(jnp.max(ells_on_grid)):.0f}], "
              f"{int(jnp.sum(weights))} real DOF")
        print(f"  sampled: {names}")
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


def forecast_all(nside, theta_pix, noise_level, is_sampled, param_ground,
                 spectra = "lensed", step_fracs = None, l_knee = 0, beam_fwhm = 0,
                 l_cutoff = 10_000, verbose = True, methods = FISHER_METHODS):
    """Every requested Fisher contraction, off ONE shared stencil.

    Args are forecast()'s, plus `methods` - any subset of FISHER_METHODS. Returns
    ({method: fisher}, names). Prefer this over calling forecast() once per method: the
    CAMB runs are memoized either way, but this also shares the covariance-block builds,
    and it guarantees the two matrices describe the identical stencil.
    """
    unknown = [m for m in methods if m not in FISHER_METHODS]
    if unknown:
        raise ValueError(f"unknown Fisher method(s) {unknown}; choose from {FISHER_METHODS}")

    names, steps, weights, fid, plus, minus = _stencil(
        nside, theta_pix, noise_level, is_sampled, param_ground, spectra, step_fracs,
        l_knee, beam_fwhm, l_cutoff, verbose)

    return ({method: _FISHER_BUILDERS[method](plus, minus, fid, steps, weights)
             for method in methods}, names)


def forecast(nside, theta_pix, noise_level, is_sampled, param_ground,
             spectra = "lensed", step_fracs = None, l_knee = 0, beam_fwhm = 0,
             l_cutoff = 10_000, verbose = True, method = "blocks"):
    """Gaussian Fisher matrix for the sampled LCDM parameters on an nside x nside box.

    Args:
        nside:        pixels per side (the sampler's nside)
        theta_pix:    pixel width in arcmin
        noise_level:  white-noise level in uK-arcmin
        is_sampled:   {param_name: bool}. False parameters are held FIXED (not
                      marginalized), matching the sampler's should_sample semantics
        param_ground: {param_name: float} fiducial point. All five must be present,
                      since the four unsampled ones still set the cosmology
        spectra:      "lensed" (default), "unlensed" or "ceiling" - see the module
                      docstring for what each one does and does not bound
        step_fracs:   override for FD_STEP_FRAC (fractions of PARAM_SIGMA)
        l_knee:       1/f noise knee. Defaults to 0 (pure white noise), matching every
                      sampler entry point; load_sim's own default of 100 does not apply
        beam_fwhm:    beam FWHM in arcmin, default 0 as in load_sim
        method:       which contraction to use, "blocks" (default, the trace formula -
                      this is what chain_analysis.py's triangle plot expects) or "cls"
                      (dC C^-1 dC). See "TWO FISHER CODE PATHS" in the module docstring

    Returns:
        (fisher, names) - the n_sampled x n_sampled matrix and the parameter names in
        OUTPUT_PARAM_ORDER order (chain_analysis.py's order, not PARAM_ORDER).
    """
    fishers, names = forecast_all(nside, theta_pix, noise_level, is_sampled, param_ground,
                                  spectra = spectra, step_fracs = step_fracs,
                                  l_knee = l_knee, beam_fwhm = beam_fwhm,
                                  l_cutoff = l_cutoff, verbose = verbose,
                                  methods = (method,))
    return fishers[method], names


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


def step_stability(nside, theta_pix, noise_level, is_sampled, param_ground,
                   spectra = "lensed", factor = 2.0, methods = FISHER_METHODS, **kwargs):
    """Recompute the forecast at `factor` x the step size and report the drift.

    A well-converged Fisher is step-independent. Drift means either the step is too large
    (real curvature in Cl(theta)) or too small (CAMB's own accuracy floor amplified by
    1/h). Second derivatives would amplify this by 1/h**2; here the log-derivative is
    first order, so a few percent drift is already worth chasing.

    Both contractions are checked (they share the stencil, so the doubled-step run is one
    extra set of CAMB calls for the pair, not one per method) and reported separately -
    they weight the same derivative vectors differently, so a step that has converged for
    one has not necessarily converged for the other. Returns {method: drift array}.
    """
    base = {name: FD_STEP_FRAC[name] for name in PARAM_ORDER}
    doubled = {name: factor * value for name, value in base.items()}

    fishers_a, names = forecast_all(nside, theta_pix, noise_level, is_sampled,
                                    param_ground, spectra = spectra, step_fracs = base,
                                    methods = methods, **kwargs)
    fishers_b, _ = forecast_all(nside, theta_pix, noise_level, is_sampled, param_ground,
                                spectra = spectra, step_fracs = doubled, verbose = False,
                                methods = methods,
                                **{k: v for k, v in kwargs.items() if k != "verbose"})

    drifts = {}
    for method in methods:
        sigma_a = np.sqrt(np.diag(covariance_from_fisher(fishers_a[method], names)))
        sigma_b = np.sqrt(np.diag(covariance_from_fisher(fishers_b[method], names)))
        drift = np.abs(sigma_b / sigma_a - 1)
        drifts[method] = drift

        print(f"\nStep stability [{method}] (h vs {factor:g}h), fractional change in "
              f"forecast sigma:")
        for name, value in zip(names, drift):
            flag = "  <-- unstable" if value > 0.01 else ""
            print(f"  {name:<14s} {value:.2%}{flag}")
        if np.max(drift) > 0.01:
            print("  tune FD_STEP_FRAC until every entry is well under 1%")

    return drifts



# ── Realization-averaged logpdf Hessian ───────────────────────────────────

def _theta_covariances(params, nside, pix_width, ell_grid):
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


def _prior_log_density(field, phi, cf_op, cphi_op):
    """The theta-DEPENDENT half of statistics.logpdf: the f and phi Gaussian priors.

    statistics.logpdf sums six terms under one -1/2 prefactor - a data quadratic form, two
    prior quadratic forms, and the noise / field / phi log-determinants. Only the four
    listed here move with theta. This reproduces them with the very same primitives logpdf
    uses (dot, pinv, log_det on the same operator objects), so the theta-dependent part is
    still evaluated by the codebase's own machinery; what is dropped is the data term and
    the noise log-determinant, which are constant in theta and therefore cancel identically
    in any central difference.

    Note the omitted data term is the one containing the lensing, which is why dropping it
    removes essentially all of the cost.
    """
    field_product = dot(field, pinv(cf_op) * field)
    phi_product = dot(phi, pinv(cphi_op) * phi)
    return -float(jnp.real(field_product + phi_product
                           + log_det(cf_op) + log_det(cphi_op)) / 2)


def logpdf_hessian_one_realization(data_set, names, param_ground, steps, stencil,
                                   nside, pix_width, ell_grid, fast = False):
    """-d2/dtheta_i dtheta_j log p(d, f, phi | theta) for ONE (data, f, phi) realization.

    Finite-differences the codebase's OWN statistics.logpdf - the same function the sampler
    evaluates - with (d, f, phi) held fixed and theta moved only through C_f and C_phi.
    Nothing about the log-density is assumed or re-derived: the data term, both quadratic
    forms and all three log-determinants are evaluated by the real code at every stencil
    point. That is the whole point of this path, and it is why it re-runs the lensing 19
    times per realization even though the data term cannot actually move.

    `fast = True` evaluates only the theta-dependent prior terms (_prior_log_density),
    skipping the lensing solve inside the data term. The omitted terms are constant across
    the stencil, so this is not an approximation but the removal of an exact cancellation;
    measured agreement is 5e-11 relative, which is float64 round-off amplified by the
    second difference's division by h^2, not a real discrepancy.
    forecast_from_logpdf verifies the equality on the first realization before trusting it.

    The stencil is the shared 1 + 2n + 4*n*(n-1)/2 point central-difference set:
        H_ii = [l(+i) - 2 l(0) + l(-i)] / h_i^2
        H_ij = [l(+i+j) - l(+i-j) - l(-i+j) + l(-i-j)] / (4 h_i h_j)
    """
    n_param = len(names)
    values = {}
    for offset in stencil:
        params = dict(param_ground)
        for i, name in enumerate(names):
            params[name] = param_ground[name] + offset[i] * steps[i]
        cf, cphi = _theta_covariances(params, nside, pix_width, ell_grid)
        cf_op = data_set.field_covariance.replace(scalar_matrix = cf)
        cphi_op = data_set.phi_covariance.replace(scalar_matrix = cphi)
        if fast:
            values[offset] = _prior_log_density(data_set.unlensed_field, data_set.phi,
                                                cf_op, cphi_op)
        else:
            values[offset] = float(logpdf(
                data_set.unlensed_field, data_set.phi, data_set.data,
                data_set.noise_covariance, cphi_op, cf_op,
                data_set.mask, data_set.beam))

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
    return -hessian


def forecast_from_logpdf(nside, theta_pix, noise_level, is_sampled, param_ground,
                         n_realizations = 100, step_fracs = None, l_knee = 0,
                         beam_fwhm = 0, map_seed = 20260908, fast = False,
                         verify_fast = True, verbose = True):
    """F_ij = < -d2 log p(d, f, phi | theta) / dtheta_i dtheta_j > over prior realizations.

    Each realization is a fresh load_sim at param_ground, which draws f and phi
    INDEPENDENTLY from their priors at that cosmology and lenses them into d. The Hessian
    of the codebase's own logpdf is then finite-differenced in theta at that fixed
    (d, f, phi) and averaged.

    WHAT THIS CONVERGES TO. Because the draws come from the PRIOR, this is the
    complete-data (Fisher) information of p(d, f, phi | theta), which is Louis's FIRST TERM
    ONLY. It is NOT the marginal Fisher of p(d | theta) that sample_joint targets:

        d2 log p(d|theta) = E[d2 log p(d,x|theta) | d] + Cov[d log p(d,x|theta) | d]

    and this path computes only the first expectation. Averaging over more realizations
    drives down the Monte Carlo error on that term; it does not recover the second, which
    is an expectation-level difference rather than a sampling fluctuation. Concretely: on
    prior draws the omitted Cov term equals this one exactly, so the marginal information
    is ZERO while this returns the full prior Fisher - see marginal_fisher.py's zero test.
    Use cmb_lensing/marginal_fisher.py for the marginal Fisher; use this to check that
    first term against the real code.

    WHAT IT IS GOOD FOR. It assumes nothing. It differentiates statistics.logpdf itself, so
    it independently validates
      * marginal_fisher.prior_information's analytic algebra (they must agree), and
      * the claim that log p(d | f, phi) is theta-independent - if any theta dependence
        lurked in the data term, it would appear here and nowhere else.
    Note it will NOT reproduce spectra = "ceiling": ceiling adds C_n to the field block and
    N_phi to the lensing block, whereas the honest complete-data Hessian uses the bare
    priors. The gap between them is exactly that heuristic.

    Cost is n_realizations * (1 + 2n + 2n(n-1)) logpdf calls, each running the lensing RK4,
    so this is by far the slowest path here - about 1900 lensing solves for the default
    100 realizations and three parameters.

    `fast = True` removes the stencil's lensing solves by evaluating only the
    theta-dependent prior terms (see _prior_log_density). The dropped terms - the data
    quadratic form and the noise log-determinant - are constant across the stencil, so they
    cancel exactly in every central difference; this is an exact simplification, not an
    approximation. Because that exactness is a property of the MODEL rather than of this
    function, `verify_fast` (default True) computes the first realization BOTH ways and
    raises if they disagree, so the assumption is re-checked on every run instead of being
    trusted forever. If someone later makes log p(d | f, phi) theta-dependent, that check
    fires rather than silently returning a wrong Fisher.

    Fast mode also passes ONE set of spectra to every load_sim call via its
    `precomputed_cls` argument. Each realization sits at the same cosmology and differs
    only by seed, so load_sim's own CAMB evaluation was repeated identically per
    realization; those spectra are already in _CAMB_CACHE from the stencil's centre point,
    so reusing them costs nothing.

    MEASURED: 117.7 s -> 1.8 s for 6 realizations at nside 64, three parameters - 64x.
    Skipping the stencil's lensing alone was only 4x (115.8 s -> 29.0 s); the repeated CAMB
    call inside load_sim turned out to be ~94% of what remained, since CAMB at lmax 4000
    costs a few seconds and everything else in a realization costs ~0.3 s. What is left per
    realization is the field draws, the covariance builds and the one lensing that makes
    the data map - none of which can be cached, because the draw genuinely changes with the
    seed.

    Returns (fisher, names, per_realization) with names in OUTPUT_PARAM_ORDER order.
    """
    missing = [name for name in PARAM_ORDER if name not in param_ground]
    if missing:
        raise ValueError(f"param_ground is missing {missing}; all five of {PARAM_ORDER} "
                         f"are needed because the unsampled ones still fix the cosmology")

    names = [name for name in OUTPUT_PARAM_ORDER if is_sampled.get(name, False)]
    if not names:
        raise ValueError("is_sampled selects no parameters - nothing to forecast")

    #load_sim has no beam_fwhm argument - it always builds a zero-FWHM beam - so unlike
    #the stencil paths this one cannot honour a beam. It would not change the answer (the
    #beam sits only in the theta-independent data term, which cancels in the theta
    #Hessian), but silently ignoring the argument would be worse than refusing it
    if beam_fwhm:
        raise ValueError(f"forecast_from_logpdf cannot apply beam_fwhm = {beam_fwhm}: "
                         f"load_sim builds its own zero-FWHM beam and exposes no override. "
                         f"The beam lives entirely in the theta-independent data term, so "
                         f"it would not change this Fisher anyway.")

    fracs = dict(FD_STEP_FRAC)
    if step_fracs is not None:
        fracs.update(step_fracs)
    steps = [fracs[name] * PARAM_SIGMA[name] for name in names]
    stencil = _stencil_offsets(len(names))

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    camb_kwargs = dict(param_ground)
    camb_kwargs["cosmomc_theta"] = camb_kwargs.pop("theta_MC_100") / 100
    camb_kwargs["As"] = float(np.exp(camb_kwargs.pop("logA")) * 1e-10)

    if verbose:
        print(f"Fisher forecast [logpdf{', fast' if fast else ''}]: nside {nside}, "
              f"theta_pix {theta_pix}', {noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  sampled: {names}")
        print(f"  {n_realizations} realizations x {len(stencil)} evaluations = "
              f"{n_realizations * len(stencil)} "
              f"{'prior-term evaluations (no lensing)' if fast else 'lensing solves'}")

    #in fast mode reuse ONE set of spectra across every realization. They are already in
    #_CAMB_CACHE from the stencil's centre point, and every realization sits at the same
    #cosmology (only the seed changes), so load_sim's own CAMB call is pure repetition -
    #it was the floor that kept the measured speedup at ~4x rather than the stencil ratio.
    #Left off the slow path so that stays a genuinely independent evaluation.
    shared_cls = camb_cls_at_params(param_ground) if fast else None

    per_realization = []
    for index in range(n_realizations):
        data_set = load_sim(nside, theta_pix, "I", map_seed + index, **camb_kwargs,
                            uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                            precomputed_cls = shared_cls)
        hessian = logpdf_hessian_one_realization(
            data_set, names, param_ground, steps, stencil, nside, pix_width, ell_grid,
            fast = fast)

        #re-check the exact-cancellation argument on the first realization rather than
        #trusting it: the full path evaluates the data term (and its lensing) at every
        #stencil point, so any theta dependence hiding there shows up as a mismatch
        if fast and verify_fast and index == 0:
            reference = logpdf_hessian_one_realization(
                data_set, names, param_ground, steps, stencil, nside, pix_width, ell_grid,
                fast = False)
            scale = np.sqrt(np.outer(np.abs(np.diag(reference)), np.abs(np.diag(reference))))
            error = np.max(np.abs(hessian - reference) / scale)
            if verbose:
                print(f"    fast-mode check on realization 0: max relative difference "
                      f"{error:.2e}")
            if error > 1e-6:
                raise RuntimeError(
                    f"fast mode disagrees with the full logpdf by {error:.2e} (relative) "
                    f"on the first realization. The two can only differ if "
                    f"log p(d | f, phi) has acquired theta dependence, which would break "
                    f"the cancellation fast mode relies on - and would also invalidate "
                    f"marginal_fisher.py's analytic derivation. Re-run with fast = False.")

        per_realization.append(hessian)
        if verbose and (index + 1) % 10 == 0:
            running = np.mean(per_realization, axis = 0)
            print(f"    {index + 1}/{n_realizations}  running diagonal "
                  f"{np.array2string(np.diag(running), precision = 4)}")

    per_realization = np.array(per_realization)
    return np.mean(per_realization, axis = 0), names, per_realization


# ── Mixed-parametrization Hessian ─────────────────────────────────────────

def _mixed_theta_matrices(params, data_set, nside, pix_width, ell_grid, qe_frozen):
    """(C_f, C_phi, D, G) operators at one cosmology, built the way sample_lcdm builds them.

    D = get_d_tt_matrix(C_f, C_n) and G = get_g_matrix_lcdm(C_phi, N_phi) - the same two
    constructors _recompute_cosmo_matrices uses - so the mixing here matches the sampler's.

    `qe_frozen` is the quadratic-estimate norm, held FIXED across the stencil. That mirrors
    the sampler, whose refresh_qe is plumbed but not exposed, so its QE norm stays at the
    param_init cosmology for a whole chain. Letting N_phi move with theta would credit the
    Fisher with information from dN_phi/dtheta that no chain actually uses.
    """
    cf, cphi = _theta_covariances(params, nside, pix_width, ell_grid)
    cf_op = data_set.field_covariance.replace(scalar_matrix = cf)
    cphi_op = data_set.phi_covariance.replace(scalar_matrix = cphi)
    d_op = data_set.mixing_d.replace(
        scalar_matrix = get_d_tt_matrix(cf, data_set.noise_covariance.scalar_matrix))
    g_op = data_set.mixing_g.replace(scalar_matrix = get_g_matrix_lcdm(cphi, qe_frozen))
    return cf_op, cphi_op, d_op, g_op


def mixed_logpdf_hessian_one_realization(data_set, names, param_ground, steps, stencil,
                                         nside, pix_width, ell_grid):
    """-d2/dtheta_i dtheta_j mixed_logpdf for ONE realization, at fixed (f_mixed, phi_mixed).

    The mixed-coordinate counterpart of logpdf_hessian_one_realization. The fields are drawn
    at param_ground and mixed ONCE with D and G at param_ground; that mixed pair is then held
    fixed while theta moves through C_f, C_phi, D, G and the -logdet(G) - logdet(D) Jacobian.

    READ THIS BEFORE COMPARING IT TO THE OTHER PATHS. mixed_logpdf is a properly normalized
    density in (f_mixed, phi_mixed) - it carries the Jacobian for exactly that reason - so
    this is a legitimate Fisher for the model written in mixed coordinates, and Louis's
    identity in these coordinates would recover the same MARGINAL Fisher. But the split
    between Louis's two terms is NOT parametrization invariant: only their difference is.
    So this Hessian is NOT the same quantity as the unmixed logpdf path's, and it does not
    converge to the bare-prior complete-data Fisher that path can be checked against. Treat
    the two as different decompositions, not as estimates of one number.

    It is also far more expensive per evaluation: unmix runs an INVERSE lensing solve and
    logpdf then runs a forward one, so every stencil point costs two lensing solves, and
    nothing cancels across the stencil because D, G and the unmixed fields all move with
    theta. There is no fast mode here - that is what mixing costs.

    NOTE the argument-order trap: mixed_logpdf takes (..., mixing_g, mixing_d) while
    mix / unmix take (..., mixing_d, mixing_g). They are passed correctly below.
    """
    n_param = len(names)

    #the QE norm is evaluated once at the fiducial cosmology and frozen, matching the sampler
    qe_frozen = data_set.quadratic_estimate.scalar_matrix

    #mix ONCE, at the fiducial cosmology; this pair is the "sample" held fixed
    _, _, d_fid, g_fid = _mixed_theta_matrices(param_ground, data_set, nside, pix_width,
                                               ell_grid, qe_frozen)
    mixed_field, mixed_phi = mix(data_set.unlensed_field, data_set.phi, d_fid, g_fid)

    values = {}
    for offset in stencil:
        params = dict(param_ground)
        for i, name in enumerate(names):
            params[name] = param_ground[name] + offset[i] * steps[i]
        cf_op, cphi_op, d_op, g_op = _mixed_theta_matrices(
            params, data_set, nside, pix_width, ell_grid, qe_frozen)
        values[offset] = float(mixed_logpdf(
            mixed_field, mixed_phi, data_set.data, data_set.noise_covariance,
            cphi_op, cf_op, data_set.mask, data_set.beam, g_op, d_op))

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
    return -hessian


def mixed_hessian_realization(nside, theta_pix, noise_level, is_sampled, param_ground,
                              map_seed, step_fracs = None, l_knee = 0, shared_cls = None):
    """One realization end to end: simulate, mix, finite-difference. Returns (hessian, names).

    This is what run_single_mixed_hessian.py calls - one slurm job, one realization, one
    npz - so that the 100-realization average is built in parallel instead of sequentially.
    """
    names = [name for name in OUTPUT_PARAM_ORDER if is_sampled.get(name, False)]
    if not names:
        raise ValueError("is_sampled selects no parameters - nothing to forecast")

    fracs = dict(FD_STEP_FRAC)
    if step_fracs is not None:
        fracs.update(step_fracs)
    steps = [fracs[name] * PARAM_SIGMA[name] for name in names]
    stencil = _stencil_offsets(len(names))

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    camb_kwargs = dict(param_ground)
    camb_kwargs["cosmomc_theta"] = camb_kwargs.pop("theta_MC_100") / 100
    camb_kwargs["As"] = float(np.exp(camb_kwargs.pop("logA")) * 1e-10)

    #reuse the fiducial spectra rather than re-running CAMB inside load_sim (see
    #load_sim's precomputed_cls); every realization sits at the same cosmology
    if shared_cls is None:
        shared_cls = camb_cls_at_params(param_ground)
    data_set = load_sim(nside, theta_pix, "I", map_seed, **camb_kwargs,
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = shared_cls)

    hessian = mixed_logpdf_hessian_one_realization(
        data_set, names, param_ground, steps, stencil, nside, pix_width, ell_grid)
    return hessian, names, steps


def forecast_from_mixed_logpdf(nside, theta_pix, noise_level, is_sampled, param_ground,
                               n_realizations = 100, step_fracs = None, l_knee = 0,
                               map_seed = 20260908, verbose = True):
    """Sequential driver for the mixed-coordinate Hessian - the local / small-N path.

    For the full 100 realizations this is slow (two lensing solves per stencil point per
    realization, ~3800 solves), which is what sampling_chains/mixed_hessian.sh exists for:
    it fans the same per-realization work across 100 slurm jobs and
    load_hessian_directory averages the saved results. Use this for smoke tests and for
    reproducing a single job locally.

    Returns (fisher, names, per_realization).
    """
    names = [name for name in OUTPUT_PARAM_ORDER if is_sampled.get(name, False)]
    if verbose:
        print(f"Fisher forecast [mixed]: nside {nside}, theta_pix {theta_pix}', "
              f"{noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  sampled: {names}")
        print(f"  {n_realizations} realizations x {len(_stencil_offsets(len(names)))} "
              f"mixed_logpdf evaluations, 2 lensing solves each")

    shared_cls = camb_cls_at_params(param_ground)
    per_realization = []
    for index in range(n_realizations):
        hessian, names, _ = mixed_hessian_realization(
            nside, theta_pix, noise_level, is_sampled, param_ground, map_seed + index,
            step_fracs = step_fracs, l_knee = l_knee, shared_cls = shared_cls)
        per_realization.append(hessian)
        if verbose and (index + 1) % 10 == 0:
            running = np.mean(per_realization, axis = 0)
            print(f"    {index + 1}/{n_realizations}  running diagonal "
                  f"{np.array2string(np.diag(running), precision = 4)}")

    per_realization = np.array(per_realization)
    return np.mean(per_realization, axis = 0), names, per_realization


def load_hessian_directory(directory, verbose = True):
    """Average per-realization Hessian npz files written by run_single_mixed_hessian.py.

    This is the parallel counterpart to the sequential forecast_from_* drivers: 100 slurm
    jobs each write one hessian_<index>.npz, and this collects them. Files are checked for
    a consistent configuration (parameters, box, noise, method) so results from two
    different runs cannot be silently averaged together.

    Returns (fisher, names, per_realization, metadata).
    """
    paths = sorted(glob.glob(os.path.join(directory, "hessian_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"no hessian_*.npz in {directory}. Run sampling_chains/mixed_hessian.sh first, "
            f"or point --hessian_dir at the out_dir that script writes to.")

    hessians, seeds, config = [], [], None
    for path in paths:
        data = np.load(path, allow_pickle = True)
        current = (tuple(data["names"]), int(data["nside"]), float(data["theta_pix"]),
                   float(data["noise_level"]), str(data["method"]))
        if config is None:
            config = current
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; a forecast cannot mix configurations")
        hessians.append(data["hessian"])
        seeds.append(int(data["map_seed"]))

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                         f"realizations would be double counted. Each slurm job must use "
                         f"a distinct seed - check the map_prefix loop in mixed_hessian.sh.")

    per_realization = np.array(hessians)
    #npz round-trips strings as np.str_; they compare and hash like str but print as
    #np.str_('omch2'), so convert back for readable reports and clean npz round-trips
    names, method = [str(name) for name in config[0]], str(config[4])
    if verbose:
        print(f"Averaging {len(paths)} realizations from {directory}")
        print(f"  method {method}, parameters {names}, nside {config[1]}, "
              f"{config[2]:g}', {config[3]:g} uK-arcmin")
        spread = np.std(per_realization, axis = 0) / np.sqrt(len(paths))
        mean = np.mean(per_realization, axis = 0)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            relative = np.abs(spread / mean)
        print(f"  standard error on the mean, relative: max {np.nanmax(relative):.3f}")

    metadata = {"method": method, "nside": config[1], "theta_pix": config[2],
                "noise_level": config[3], "n_realizations": len(paths)}
    return np.mean(per_realization, axis = 0), names, per_realization, metadata

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


# ── CLI ───────────────────────────────────────────────────────────────────

#one-line reminder of which contraction produced a figure, stamped into every title
METHOD_LABEL = {
    "blocks": r"blocks:  $F_{ij} = \frac{1}{2}\sum_k w_k\,\partial_i\ln C_k\,\partial_j\ln C_k$",
    "cls": r"cls:  $F_{ij} = \partial_i C \cdot C^{-1} \cdot \partial_j C$",
    "marginal": r"marginal (Louis):  $I = \mathbb{E}[-\partial^2 \ell_c] - \mathrm{Cov}[\partial \ell_c]$",
    "logpdf": r"logpdf:  $F_{ij} = \langle -\partial_i \partial_j \log p(d, f, \phi\,|\,\theta) \rangle$",
    "mixed": r"mixed:  $F_{ij} = \langle -\partial_i \partial_j \log p(d, f^\circ, \phi^\circ\,|\,\theta) \rangle$",
}


def correlation_from_covariance(covariance):
    """F^-1 normalized to unit diagonal - the degeneracy structure with the units divided out."""
    sigmas = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigmas, sigmas)


def write_outputs(fisher, covariance, names, directory, method, subtitle, config):
    """Write the three figures and the npz for one code path, tagged with its suffix.

    `config` is the run configuration (nside, noise, ...) stored alongside the matrices in
    the npz so a saved forecast can be traced back to the box it was computed on.
    """
    suffix = METHOD_SUFFIX[method]
    full_subtitle = f"{subtitle}\n{METHOD_LABEL[method]}"

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


def averaged_hessian_covariance(fisher, names, metadata):
    """Invert an averaged Hessian, blaming too few realizations rather than the stencil."""
    try:
        return covariance_from_fisher(fisher, names)
    except RuntimeError as error:
        raise RuntimeError(
            f"the averaged Hessian is not positive definite over "
            f"{metadata['n_realizations']} realizations. For a realization-averaged path "
            f"that usually means too few realizations rather than a bad finite-difference "
            f"step - add more jobs and re-run the averaging. Underlying: {error}") from error


def _report(covariance, names, spectra, method):
    sigmas = np.sqrt(np.diag(covariance))
    print(f"\nForecast 1-sigma errors [{spectra}, {method}]:")
    if method == "cls":
        #see the CAVEAT in the module docstring: one power of C^-1, so these scale with
        #covar_matrix_from_cls's arbitrary theta-independent normalization
        print("  (this path's ABSOLUTE sigmas carry an arbitrary normalization - read the "
              "correlations\n   and the relative ordering, not the numbers against "
              "PARAM_SIGMA)")
    print(f"  {'parameter':<14s} {'sigma':>12s} {'sigma/PARAM_SIGMA':>20s}")
    for i, name in enumerate(names):
        print(f"  {name:<14s} {sigmas[i]:>12.4g} {sigmas[i] / PARAM_SIGMA[name]:>20.3g}")

    if len(names) > 1:
        correlation = covariance / np.outer(sigmas, sigmas)
        pairs = [(abs(correlation[i, j]), correlation[i, j], names[i], names[j])
                 for i in range(len(names)) for j in range(i + 1, len(names))]
        _, value, first, second = max(pairs)
        print(f"  strongest degeneracy: {first} - {second}  (r = {value:+.2f})")


def main():
    parser = argparse.ArgumentParser(
        description = "Gaussian Fisher forecast for the LCDM parameters on the "
                      "sample_lcdm.py flat-sky box")
    parser.add_argument("--nside", type = int, default = 64)
    parser.add_argument("--theta_pix", type = float, default = 5.0,
                        help = "pixel width in arcmin")
    parser.add_argument("--noise", type = float, default = 5.0,
                        help = "white noise level in uK-arcmin")
    parser.add_argument("--l_knee", type = float, default = 0.0,
                        help = "1/f knee; 0 (default) matches every sampler entry point")
    parser.add_argument("--beam_fwhm", type = float, default = 0.0)
    parser.add_argument("--spectra", choices = SPECTRA_MODES, default = "ceiling") #lensed
    parser.add_argument("--params", nargs = "*", default = None,
                        help = f"subset to sample; default all of {PARAM_ORDER}")
    parser.add_argument("--method",
                        choices = FISHER_METHODS + REALIZATION_METHODS + ("both", "all"),
                        default = "both",
                        help = "which Fisher path to write outputs for. 'both' (default) "
                               "runs the two stencil contractions off one shared stencil; "
                               "'logpdf' averages -d2 logpdf over (data, f, phi) "
                               "realizations (slow, and it is Louis's first term only, "
                               "NOT the marginal Fisher - see marginal_fisher.py); 'all' "
                               "runs everything")
    parser.add_argument("--realizations", type = int, default = 100,
                        help = "(data, f, phi) draws averaged by --method logpdf")
    parser.add_argument("--hessian_dir", type = str, default = None,
                        help = "average per-realization hessian_*.npz files written by "
                               "sampling_chains/mixed_hessian.sh instead of computing them "
                               "here. This is the time-feasible route for the mixed path: "
                               "100 slurm jobs run the realizations in parallel and this "
                               "just collects them. Overrides --method")
    parser.add_argument("--fast_logpdf", action = "store_true",
                        help = "--method logpdf: evaluate only the theta-dependent prior "
                               "terms, skipping the lensing solve in the theta-INDEPENDENT "
                               "data term. Exact (those terms cancel in every central "
                               "difference), and it reuses one set of CAMB spectra across "
                               "realizations: 64x faster measured, self-verified against "
                               "the full logpdf on the first realization")
    parser.add_argument("--stability", action = "store_true",
                        help = "also recompute at 2x the step size and report the drift")
    args = parser.parse_args()

    is_sampled = {name: (args.params is None or name in args.params)
                  for name in PARAM_ORDER}
    if args.params is not None:
        unknown = [name for name in args.params if name not in PARAM_ORDER]
        if unknown:
            parser.error(f"unknown parameters {unknown}; choose from {PARAM_ORDER}")

    if args.hessian_dir is not None:
        fisher, names, per_realization, metadata = load_hessian_directory(args.hessian_dir)
        method = metadata["method"]
        covariance = averaged_hessian_covariance(fisher, names, metadata)
        _report(covariance, names, "prior draws", method)

        directory = output_dir()
        os.makedirs(directory, exist_ok = True)
        subtitle = (f"prior draws  |  nside {metadata['nside']}, "
                    f"{metadata['theta_pix']:g}', {metadata['noise_level']:g} uK-arcmin"
                    f"  |  {metadata['n_realizations']} realizations (parallel)")
        written = write_outputs(
            fisher, covariance, names, directory, method, subtitle,
            dict(spectra = "prior draws", nside = metadata["nside"],
                 theta_pix = metadata["theta_pix"], noise_level = metadata["noise_level"],
                 l_knee = 0.0, beam_fwhm = 0.0,
                 step_fracs = np.array([FD_STEP_FRAC[name] for name in names]),
                 n_realizations = metadata["n_realizations"],
                 standard_error = np.std(per_realization, axis = 0)
                                  / np.sqrt(len(per_realization))))
        print(f"\nWrote {', '.join(written)} to {directory}")
        return

    if args.method == "both":
        requested = FISHER_METHODS
    elif args.method == "all":
        requested = FISHER_METHODS + REALIZATION_METHODS
    else:
        requested = (args.method,)
    methods = tuple(m for m in requested if m in FISHER_METHODS)

    fishers, names = ({}, None)
    if methods:
        fishers, names = forecast_all(args.nside, args.theta_pix, args.noise, is_sampled,
                                      GROUND_TRUTH, spectra = args.spectra,
                                      l_knee = args.l_knee, beam_fwhm = args.beam_fwhm,
                                      methods = methods)

    if "mixed" in requested:
        mixed_fisher, mixed_names, _ = forecast_from_mixed_logpdf(
            args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
            n_realizations = args.realizations, l_knee = args.l_knee)
        if names is not None and mixed_names != names:
            raise RuntimeError(f"parameter order disagrees between paths: {names} vs "
                               f"{mixed_names}")
        names = mixed_names
        fishers = dict(fishers)
        fishers["mixed"] = mixed_fisher
        methods = methods + ("mixed",)

    if "logpdf" in requested:
        #a separate path: it needs field realizations, not a covariance stencil, and it is
        #a prior expectation, so args.spectra does not apply to it
        logpdf_fisher, logpdf_names, _ = forecast_from_logpdf(
            args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
            n_realizations = args.realizations, l_knee = args.l_knee,
            beam_fwhm = args.beam_fwhm, fast = args.fast_logpdf)
        if names is not None and logpdf_names != names:
            raise RuntimeError(f"parameter order disagrees between paths: {names} vs "
                               f"{logpdf_names}")
        names = logpdf_names
        fishers = dict(fishers)
        fishers["logpdf"] = logpdf_fisher
        methods = methods + ("logpdf",)
    covariances = {method: covariance_from_fisher(fishers[method], names)
                   for method in methods}
    for method in methods:
        #the logpdf path draws from the prior and never consults --spectra, so labelling
        #its report with args.spectra would claim a mode it did not use
        _report(covariances[method], names,
                "prior draws" if method in REALIZATION_METHODS else args.spectra, method)

    if args.spectra == "lensed":
        #the ceiling / baseline ratio sets how badly Louis's identity will cancel when
        #the marginal Fisher is estimated from Gibbs draws - worth knowing in advance
        stencil_methods = tuple(m for m in methods if m in FISHER_METHODS)
        ceilings, _ = forecast_all(args.nside, args.theta_pix, args.noise, is_sampled,
                                   GROUND_TRUTH, spectra = "ceiling",
                                   l_knee = args.l_knee, beam_fwhm = args.beam_fwhm,
                                   verbose = False, methods = stencil_methods)
        for method in ceilings:
            ratio = np.sqrt(np.diag(ceilings[method])) / np.sqrt(np.diag(fishers[method]))
            print(f"\n  [{method}] F_ceiling / F_lensed per parameter (diagonal, sqrt): "
                  f"{np.array2string(ratio, precision = 1)}")
        print("  the marginal Fisher sits between these two; a large ratio means Louis's "
              "identity\n  will subtract two nearly-equal numbers and needs many Gibbs draws")

    if args.stability:
        step_stability(args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
                       spectra = args.spectra, l_knee = args.l_knee,
                       beam_fwhm = args.beam_fwhm,
                       methods = tuple(m for m in methods if m in FISHER_METHODS))

    directory = output_dir()
    os.makedirs(directory, exist_ok = True)
    subtitle = (f"{args.spectra}  |  nside {args.nside}, {args.theta_pix:g}', "
                f"{args.noise:g} uK-arcmin")
    #the logpdf path ignores --spectra (it is a prior expectation, not a spectra choice)
    #and its accuracy is set by the realization count, so it gets its own subtitle
    logpdf_subtitle = (f"prior draws  |  nside {args.nside}, {args.theta_pix:g}', "
                       f"{args.noise:g} uK-arcmin  |  {args.realizations} realizations")
    config = dict(spectra = args.spectra, nside = args.nside, theta_pix = args.theta_pix,
                  noise_level = args.noise, l_knee = args.l_knee,
                  beam_fwhm = args.beam_fwhm,
                  step_fracs = np.array([FD_STEP_FRAC[name] for name in names]))

    if "logpdf" in methods:
        config["n_realizations"] = args.realizations
        config["fast_logpdf"] = args.fast_logpdf

    written = []
    for method in methods:
        written += write_outputs(fishers[method], covariances[method], names, directory,
                                 method,
                                 logpdf_subtitle if method in REALIZATION_METHODS
                                 else subtitle,
                                 config)
    print(f"\nWrote {', '.join(written)} to {directory}")


if __name__ == "__main__":
    main()
