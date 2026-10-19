"""Gaussian Fisher forecast for the LCDM parameters on sample_lcdm.py's flat-sky box.

The forecast is a finite difference of two covariance blocks on the SAME rfft grid the
sampler runs on (gen_ell_grid at the given nside / theta_pix), contracted with the
zero-mean Gaussian Fisher formula specialized to a covariance diagonal in Fourier space,

    F_ij = 1/2 * sum_blocks sum_k w_k * dln(C_k)/dtheta_i * dln(C_k)/dtheta_j

with w_k = util.get_fourier_weights, the number of independent real degrees of freedom of
each rfft entry (2 in the bulk, 1 on the self-conjugate columns kx = 0 and kx = Nyquist;
they sum to nside**2 and are the weights statistics.logpdf is built from). See "Derivation"
below. The two blocks are

    f block    C_TT^delensed + C_n      the temperature field after delensing, plus the
                                        instrument noise
    phi block  C_phi + N_phi            the lensing potential measured to within the
                                        reconstruction noise N_phi

THE TWO REPORTED BOUNDS. Both blocks are driven by one N_phi: it is the phi block's noise,
and it sets how well the temperature is delensed - Alens_L = N_L / (C_L + N_L), with N_L
the azimuthal average of N_phi (radial_cl_profile). N_phi is the box's quadratic-estimator
noise matrix (qe_noise_matrix = simulate.scalar_quadratic_estimate without NPHI_FAC), and
the TT spectrum in the estimator's FILTER (`cfl_tt`) brackets the answer:

  "qe_unlensed"  UPPER bound on the information. The filter carries the fully UNLENSED
                 C_TT (cfl_tt -> cf_tt): the estimator sees a map with no lensing variance
                 left in it, the lowest N_phi any quadratic reconstruction could reach
  "qe_delensed"  LOWER bound on the information. The filter carries the PARTIALLY delensed
                 C_TT, iterated from the lensed spectrum to its fixed point
                 (iterative_delensing): the noise a real iterative QE reconstruction gets

and two empirical N_phi measured on this box at the fiducial cosmology, for comparison:

  "map_joint"    map_joint's per-mode reconstruction noise N_eff = C_phi (1/r^2 - 1), the
                 `nphi` of a merge_map_joint_phi_noise.py npz (or the legacy `phi_noise_fid`
                 of an older merge_delensed_covariance.py npz), passed as --phi_noise
  "sample_joint" the sampler's effective noise from the posterior-mean phi of fixed-theta
                 chains, the `nphi` of a merge_sampler_effective_noise.py npz (--phi_noise)

THE TWO TEMPERATURE BLOCKS. By default C_TT^delensed is CAMB's own (get_partially_lensed_cls,
the non-perturbative correlation-function lensing with C_L^phiphi scaled by Alens_L), re-run
at every stencil point at the frozen Alens_L. --delensed_covariance replaces it with the
EMPIRICAL per-mode covariance of the box's own lense_flow delensing (W phi_true with
W = C_phi / (C_phi + N_phi) per mode, common random numbers across the stencil;
cmb_lensing/delensed_covariance.py), value and finite difference alike, and its steps h_i
replace FD_STEP_FRAC for both blocks. The file records which N_phi its W was built from, and
the forecast warns when that is not the N_phi it is running with.

THE RECONSTRUCTION IS FROZEN. N_phi and Alens_L are evaluated once at the fiducial cosmology
and held across the stencil, so only the signal carries theta dependence - N_phi is a
property of the estimator and the experiment, the same convention that makes C_n
theta-independent, and the empirical N_phi have no theta dependence to difference anyway.

Derivation (scalar case, in this codebase's normalization). statistics.logpdf is built
from primal_dot and primal_log_det, i.e. schematically

    log L = -1/2 sum_k w_k [ |x_k|^2 / (nside^2 C_k) + ln C_k ] + const

with E|x_k|^2 = nside^2 C_k (see the field_from_covar_single_key comment). Then
E[d log L / dtheta_i] = 0, and with Var(|x_k|^2) = (nside^2 C_k)^2 * (2 / w_k) - variance
v^2 for a free complex mode, 2 v^2 for a real one - the outer product E[d_i log L d_j log L]
collapses to

    F_ij = 1/4 sum_k w_k^2 (2 / w_k) dln(C_k)/dtheta_i dln(C_k)/dtheta_j
         = 1/2 sum_k w_k dln(C_k)/dtheta_i dln(C_k)/dtheta_j

The two blocks are added as independent. Because the residual lensing left in the delensed
map tracks C_phi(theta), the f block picks up some lensing information on omch2 that the
phi block counts again - scaled down by Alens_L, so it shrinks as the delensing improves.

WHAT THIS IS NOT. Neither bound is the marginal Fisher of p(d | theta), which is what
sample_joint actually targets; the chains themselves are the measurement of that.

Finite differences. Central differences with step h_p = FD_STEP_FRAC[p] * PARAM_SIGMA[p]
from DIRECT CAMB calls (never the 5D grid spline, whose deterministic ~1e-3 lnCl error
would be differenced over the grid's coarse node spacing); 2 * n_sampled + 1 runs, memoized.
"--stability" recomputes at 2h and reports the fractional change in the forecast sigmas.
Parameters with is_sampled[name] = False are held FIXED, not marginalized over.

Usage:
    python -m cmb_lensing.fisher_forecast --nphi_source qe_unlensed --noise 5
    python -m cmb_lensing.fisher_forecast --nphi_source qe_delensed --qe_response gradient
    python -m cmb_lensing.fisher_forecast --nphi_source map_joint --phi_noise <delensed_covariance.npz>
    python -m cmb_lensing.fisher_forecast --nphi_source sample_joint --phi_noise <sampler_effective_noise.npz>
    python -m cmb_lensing.fisher_forecast --nphi_source qe_unlensed --delensed_covariance <delensed_covariance.npz>

Writes into cmb_lensing/fisher_output/, tagged with the N_phi source, the temperature block
and the noise level (e.g. "_from_blocks_qe_unlensed_camb_5.0_uk"):
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

import camb
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.simulate import (camb_parameters, _extract_all_cls, dl2cl,
                                  covar_matrix_from_cls, noise_cls, get_beam, get_mask,
                                  scalar_quadratic_estimate)
from cmb_lensing.constants import (DEFAULT_MAX_ELL, DEFAULT_A_LENSE, DEFAULT_K_PIVOT,
                                   DEFAULT_MNU, DEFAULT_TAUREIO, PARAM_ORDER, GROUND_TRUTH,
                                   PARAM_SIGMA, CAMB_LMAX, DEFAULT_ELL_CUTOFF)


METHOD = "blocks"
SUFFIX = "_from_blocks"
LABEL = r"blocks:  $F_{ij} = \frac{1}{2}\sum_k w_k\,\partial_i\ln C_k\,\partial_j\ln C_k$"

#finite-difference step for each parameter, as a FRACTION of PARAM_SIGMA. raise them if the
#forecast is noisy (CAMB's own ~1e-3 lnCl accuracy floor leaking into the difference), lower
#them if it drifts with step size (genuine curvature of Cl(theta)). verify with --stability
FD_STEP_FRAC = {
    "theta_MC_100": 0.05,
    "logA": 0.05,
    "ns": 0.05,
    "ombh2": 0.05,
    "omch2": 0.05,
}

#the order every output is laid out in - chain_analysis.py's ground_truth_values order, NOT
#PARAM_ORDER, so the matrices here read side by side with the ones made from the chains
OUTPUT_PARAM_ORDER = ["omch2", "ombh2", "ns", "theta_MC_100", "logA"]
assert sorted(OUTPUT_PARAM_ORDER) == sorted(PARAM_ORDER)

#where N_phi comes from - see the module docstring. The two "qe_" sources are the bounds and
#are computed here; the other two are read from --phi_noise
QE_NPHI_SOURCES = ("qe_unlensed", "qe_delensed")
FILE_NPHI_SOURCES = ("map_joint", "sample_joint")
NPHI_SOURCES = QE_NPHI_SOURCES + FILE_NPHI_SOURCES
DEFAULT_NPHI_SOURCE = "qe_delensed"

#the TT spectrum in the quadratic estimator's inverse-variance FILTER (scalar_quadratic_estimate's
#`cfl_tt`), qe_reconstruction's `qe_filter`: "lensed" the fully lensed C_TT (one QE pass, no
#delensing - only delensed_covariance.py's default W still uses it), "delensed" iterated to the
#partially delensed fixed point, "unlensed" the unlensed C_TT. "qe_<filter>" is the NPHI_SOURCE
QE_FILTERS = ("lensed", "delensed", "unlensed")

#which TT spectrum the quadratic estimator's RESPONSE f(l, l') is built from - the `cf_tt`
#argument of simulate.scalar_quadratic_estimate. Independent of the filter:
#  "unlensed"  (DEFAULT) C_l^TT unlensed, Hu & Okamoto's original expression and what the
#              sampler's own QE norm uses
#  "gradient"  CAMB's lensed temperature-gradient spectrum C_l^(T grad T)
#              (gradient_cls_at_params), the correct weight (Lewis, Challinor & Hanson 2011),
#              which resums the N^(2) bias into the normalization. Under the "delensed" filter
#              every iteration after the first rebuilds it from the DELENSED T-grad-T at the
#              current Alens_L (delensed_gradient_cls_at_params)
#Measured at nside 64 / 5' / 5 uK, T-only: "gradient" raises N_phi by 0.4-2.6% and moves the
#marginalized sigmas by < 0.35%
QE_RESPONSE_SOURCES = ("unlensed", "gradient")
DEFAULT_QE_RESPONSE = "unlensed"

#the per-mode noise grid load_phi_noise_matrix takes from a noise npz when no key is named,
#in order of preference: `nphi` (merge_sampler_effective_noise.py, filled on every mode, or
#merge_map_joint_phi_noise.py, NaN where unmeasured), else `phi_noise_fid` (an older
#merge_delensed_covariance.py: map_joint's N_eff at theta_0, NaN where unmeasured). Any other
#2D grid of the file must be asked for by name
PHI_NOISE_KEYS = ("nphi", "phi_noise_fid")
#the key only merge_sampler_effective_noise.py writes (how a sampler file is told apart from a
#map_joint one)
SAMPLE_JOINT_MARKER = "nphi_cross_raw"

#relative tolerance for "these two parameter vectors are the same cosmology": the values
#round-trip through np.savez as float64
PARAM_RTOL = 1e-12

#CAMB is deterministic in the parameters, and the stencil re-queries the fiducial point
#for every parameter, so memoize across calls (and across the modules that import this one)
_CAMB_CACHE = {}
_CAMB_RESULTS_CACHE = {}
_GRADIENT_CACHE = {}


def output_dir():
    #a directory next to the package modules, as sample_lcdm.py's sample_lcdm_output, so the
    #same relative layout works on the laptop and the cluster
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisher_output")


# ── CAMB ──────────────────────────────────────────────────────────────────

def _camb_key(params):
    return tuple(round(float(params[name]), 12) for name in PARAM_ORDER)


def _point_text(params):
    return dict((name, float(params[name])) for name in PARAM_ORDER)


def camb_results_at_params(params, camb_lmax = None):
    """(CAMBparams, CAMBdata) at an arbitrary 5-parameter point, memoized per (point, lmax).

    Set up through simulate.camb_parameters, so the run is identical to load_sim's: H0 solved
    from cosmomc_theta, r = 0, mnu = 0.06, tau = 0.05, nt = 0, k_pivot = 0.05, Alens = 1,
    non-linear lensing. The results object is kept because the delensed and gradient spectra
    are post-processing calls on it. `camb_lmax` defaults to CAMB_LMAX.
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
            f"CAMB failed at {_point_text(params)}: {error}. check the stencil point is "
            f"reachable by CAMB (H0 is solved from cosmomc_theta inside "
            f"DEFAULT_THETA_H0_RANGE), or shrink FD_STEP_FRAC for that parameter.") from error

    _CAMB_RESULTS_CACHE[key] = (pars, results)
    return pars, results


def camb_cls_at_params(params, camb_lmax = None):
    """Full cls dict from one CAMB run at an arbitrary 5-parameter point.

    The spectra are exactly what simulate._run_camb extracts (same calls, same units, same
    _extract_all_cls), on ells 2..camb_lmax-1 with no extrapolation. Every downstream axis is
    derived from the returned arrays' LENGTH, so raising camb_lmax propagates on its own.
    """
    camb_lmax = CAMB_LMAX if camb_lmax is None else int(camb_lmax)
    key = (_camb_key(params), camb_lmax)
    if key in _CAMB_CACHE:
        return _CAMB_CACHE[key]

    pars, results = camb_results_at_params(params, camb_lmax = camb_lmax)
    power_spectra = results.get_cmb_power_spectra(pars, lmax = camb_lmax - 1, CMB_unit = "muK")
    lens_potential = results.get_lens_potential_cls(lmax = camb_lmax - 1, CMB_unit = "muK")[:, :2]
    cls = _extract_all_cls(jnp.asarray(power_spectra["unlensed_scalar"]),
                           jnp.asarray(power_spectra["tensor"]),
                           jnp.asarray(power_spectra["total"]),
                           jnp.asarray(lens_potential), camb_lmax, camb_lmax)

    if not bool(jnp.all(jnp.isfinite(cls["total_TT"]))):
        raise RuntimeError(
            f"CAMB returned non-finite Cls at {_point_text(params)}. check the stencil point "
            f"is reachable by CAMB, or shrink FD_STEP_FRAC for that parameter.")

    _CAMB_CACHE[key] = cls
    return cls


def gradient_cls_at_params(params, camb_lmax = None):
    """C_l^(T grad T): the lensed TEMPERATURE-GRADIENT cross spectrum, on 2..camb_lmax-1.

    The spectrum the TT quadratic estimator's response should be built from (Lewis, Challinor
    & Hanson 2011, appendix C): CAMB's get_lensed_gradient_cls, column 0. Accurate to 8e-5 at
    l = 1000 and 1.4e-3 at l = 3000 against a longer run, degrading to 3% in the last ~100
    multipoles below the run's own max_l. Memoized separately from camb_cls_at_params because
    it costs ~5 s and only the fiducial point needs it.
    """
    camb_lmax = CAMB_LMAX if camb_lmax is None else int(camb_lmax)
    key = (_camb_key(params), camb_lmax)
    if key in _GRADIENT_CACHE:
        return _GRADIENT_CACHE[key]

    _, results = camb_results_at_params(params, camb_lmax = camb_lmax)
    #l(l+1)/2pi convention with raw_cl = False, exactly what dl2cl's default branch inverts
    gradient = results.get_lensed_gradient_cls(lmax = camb_lmax - 1, CMB_unit = "muK")
    spectrum = dl2cl(jnp.asarray(gradient[:, 0]), camb_lmax, camb_lmax)
    _check_positive(spectrum, f"T-grad-T spectrum at {_point_text(params)}")
    _GRADIENT_CACHE[key] = spectrum
    return spectrum


def delensed_gradient_cls_at_params(params, alens, camb_lmax = None):
    """C_l^(T grad T) of the partially DELENSED map: gradient_cls_at_params with C_L^phiphi
    scaled per multipole by `alens` (zero-based in L; multipoles past its end carry its last
    value). The response an iterated quadratic estimator needs. Alens = 1 everywhere
    reproduces gradient_cls_at_params to 3e-10. ~5 s per call, not memoized."""
    camb_lmax = CAMB_LMAX if camb_lmax is None else int(camb_lmax)
    pars, results = camb_results_at_params(params, camb_lmax = camb_lmax)
    #[L(L+1)]^2 C_L^phiphi / 2pi, zero-based to Params.max_l - the convention CAMB's clpp
    #argument takes, and what get_partially_lensed_cls scales the same way
    clpp = results.get_lens_potential_cls(lmax = pars.max_l)[:, 0] * _alens_scaling(pars, alens)
    gradient = results.get_lensed_gradient_cls(lmax = camb_lmax - 1, CMB_unit = "muK",
                                               clpp = clpp)
    spectrum = dl2cl(jnp.asarray(gradient[:, 0]), camb_lmax, camb_lmax)
    _check_positive(spectrum, "delensed T-grad-T spectrum")
    return spectrum


def delensed_cls_at_params(params, alens, camb_lmax = None):
    """camb_cls_at_params plus "delensed_TT": CAMB lensing the unlensed spectra at `params`
    with C_L^phiphi scaled by the per-L `alens` (zero-based in L, Alens_L = 1 meaning no
    delensing at that L).

    get_partially_lensed_cls reruns CAMB's full non-perturbative correlation-function lensing
    with the scaled potential - NOT a linear interpolation between the unlensed and lensed
    spectra, which it differs from at the percent level. Multipoles past the end of `alens`
    (CAMB lenses with C_L^phiphi up to Params.max_l) carry its last value. ~0.02 s once the
    results object is cached.
    """
    cls = dict(camb_cls_at_params(params, camb_lmax = camb_lmax))
    pars, results = camb_results_at_params(params, camb_lmax = camb_lmax)
    #the delensed spectrum must land on the SAME axis as the rest of the dict
    lmax = 2 + cls["scalar_TT"].shape[0]
    partial = results.get_partially_lensed_cls(_alens_scaling(pars, alens), lmax = lmax - 1,
                                               CMB_unit = "muK", lensing_method = 3)
    cls["delensed_TT"] = dl2cl(jnp.asarray(partial[:, 0]), lmax, lmax)
    return cls


def _alens_scaling(pars, alens):
    scaling = np.full(pars.max_l + 1, float(alens[-1]))
    scaling[:len(alens)] = alens
    return scaling


def _check_positive(spectrum, label):
    if not bool(jnp.all(jnp.isfinite(spectrum))) or bool(jnp.any(spectrum <= 0)):
        raise RuntimeError(f"CAMB returned a non-positive or non-finite {label}; the quadratic "
                           f"estimator's response needs it positive on the whole axis")


def cls_with_qe_response(params, qe_response = DEFAULT_QE_RESPONSE, camb_lmax = None):
    """camb_cls_at_params, carrying the extra spectrum `qe_response` needs ("gradient_TT" for
    "gradient"; nothing for "unlensed", which reads scalar_TT)."""
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
    """The TT spectrum the quadratic estimator's response is built from. RAISES rather than
    falling back to scalar_TT when "gradient" is asked for without its key - silently
    reverting is precisely the misnormalization that mode exists to remove."""
    _check_qe_response(qe_response)
    if qe_response == "unlensed":
        return cls["scalar_TT"]
    if "gradient_TT" not in cls:
        raise ValueError("qe_response = 'gradient' needs cls['gradient_TT']; build the dict "
                         "with cls_with_qe_response(params, 'gradient')")
    return cls["gradient_TT"]


def load_sim_cosmology(param_ground):
    """param_ground translated into load_sim's keyword names (cosmomc_theta, As)."""
    camb_kwargs = dict(param_ground)
    camb_kwargs["cosmomc_theta"] = camb_kwargs.pop("theta_MC_100") / 100
    camb_kwargs["As"] = float(np.exp(camb_kwargs.pop("logA")) * 1e-10)
    return camb_kwargs


def same_cosmology(first, second):
    return all(np.isclose(first[name], second[name], rtol = PARAM_RTOL, atol = 0)
               for name in PARAM_ORDER)


# ── The rfft grid ─────────────────────────────────────────────────────────

def fourier_weights(nside):
    """w_k, the independent real DOF per rfft entry, broadcast over the (nside, nside//2+1)
    grid (get_fourier_weights indexes the half axis, i.e. the columns)."""
    return jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                            (nside, nside // 2 + 1))


def _ell_axis(cl):
    return jnp.arange(2, 2 + cl.shape[0]).astype(jnp.float64)


def _covar(nside, pix_width, ell_grid, cl):
    """A Cl on 2..len+1 put on the rfft grid by covar_matrix_from_cls, origin zeroed."""
    return covar_matrix_from_cls(nside, pix_width, ell_grid, _ell_axis(cl), cl,
                                 origin_value = 0)


def _instrument_matrices(nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                         l_cutoff, lmax_prime = None):
    """(noise, mask, beam) on the rfft grid - the theta-INDEPENDENT part of every block.

    `lmax_prime` is the multipole axis the analytic noise is built on (default load_sim's);
    pass the cls' own length so C_n and C_f share one axis when CAMB ran past CAMB_LMAX.
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
    """N_phi: the scalar quadratic-estimate reconstruction noise matrix on the rfft grid.

    simulate.scalar_quadratic_estimate(noise, response, filter, mask, beam), the same QE norm
    the sampler's G matrix uses, except (1) the response spectrum is `qe_response`'s and
    (2) there is no division by NPHI_FAC, a preconditioning choice inherited from
    CMBLensing.jl that cancels inside G - so this is the physical N^(0). `filter_tt` is the
    TT spectrum in the estimator's inverse-variance filter, the lensed one by default.
    """
    noise, mask, beam = _instrument_matrices(nside, pix_width, ell_grid, noise_level,
                                             l_knee, beam_fwhm, l_cutoff)
    filter_tt = cls["total_TT"] if filter_tt is None else filter_tt
    return scalar_quadratic_estimate(noise,
                                     _covar(nside, pix_width, ell_grid,
                                            qe_response_cl(cls, qe_response)),
                                     _covar(nside, pix_width, ell_grid, filter_tt),
                                     mask, beam, pix_width)


def prior_covariances(params, nside, pix_width, ell_grid):
    """(C_f, C_phi) on the rfft grid at one cosmology - the bare priors (no C_n, no N_phi)."""
    cls = camb_cls_at_params(params)
    return (_covar(nside, pix_width, ell_grid, cls["scalar_TT"]),
            _covar(nside, pix_width, ell_grid, cls["phi"]))


def delensing_efficiency(cphi, nphi):
    """The C_phi-weighted mean Wiener factor C_phi / (C_phi + N_phi): 1 is a perfect
    reconstruction, 0 none. A one-number summary for the printed report."""
    good = (cphi > 0) & jnp.isfinite(nphi)
    rho_squared = jnp.where(good, cphi / (cphi + jnp.where(good, nphi, 0.0)), 0.0)
    weight = jnp.where(good, cphi, 0.0)
    return float(jnp.sum(weight * rho_squared) / jnp.sum(weight))


def interpolate_spectrum(ells, source_ells, spectrum):
    """A positive spectrum moved onto `ells` by log-log interpolation, extrapolated as a
    power law past either end - the jnp.interp call covar_matrix_from_cls uses."""
    return np.asarray(jnp.exp(jnp.interp(jnp.log(ells), jnp.log(source_ells),
                                        jnp.log(spectrum), left = "extrapolate",
                                        right = "extrapolate")))


def grid_max_ell(ell_grid):
    """The largest multipole the rfft grid carries: the corner mode sqrt(2) pi / pix_width
    (3055 at 5', 6109 at 2.5', for every nside)."""
    return float(np.max(np.asarray(ell_grid)))


def camb_lmax_for_grid(ell_grid):
    """The CAMB lmax that covers every mode the rfft grid carries - the stencil's default.

    covar_matrix_from_cls log-log EXTRAPOLATES past a spectrum's support, and the damping
    tail falls faster than any power law: at nside 128 / 2.5' the corner is 6109, 32.6% of
    the grid's DOF sit past CAMB_LMAX's 3999, and the extrapolated lensed TT is 2.5x too small
    at the corner (19% on sigma(theta_MC_100) in the delensed forecast). Floored at
    DEFAULT_MAX_ELL; rounded up so the corner mode itself is interior. load_sim, the 1D
    spline caches and the 5D grid all stay on 2..3999.
    """
    return max(DEFAULT_MAX_ELL, int(np.ceil(grid_max_ell(ell_grid))) + 1)


def radial_cl_profile(matrix, ell_grid, pix_width, ell_axis, label = "N_phi"):
    """Azimuthally average an rfft-grid noise matrix back onto the 1D `ell_axis`, in Cl units.

    The QE norm is anisotropic on the square box (1.6x around an annulus at L = 540, 5.7x at
    L = 2090 on nside 64 / 5'), so the 1D N_L behind Alens_L is an annulus average. The
    1/pix_width**2 of covar_matrix_from_cls is undone; annuli are one fundamental wide and
    weighted by w_k; each annulus is the mean of log N at the geometric-mean l, which is
    EXACT for the local power law the log-log interpolation assumes (an arithmetic mean
    overestimated N_phi by 3.6% at one-fundamental annuli). Non-positive and non-finite
    entries (the origin, the QE's diverged modes, an empirical file's unmeasured modes) are
    left out. The profile is log-log interpolated onto `ell_axis` and extrapolated at both
    ends, with a warning past the grid's largest mode; below the fundamental the box holds no
    modes, so that end is pure continuation (good to 14% at L = 2 against Hu & Okamoto).
    """
    nside = np.asarray(ell_grid).shape[0]
    ells = np.asarray(ell_grid).ravel()
    values = np.asarray(matrix).ravel() * pix_width**2
    dof = np.asarray(fourier_weights(nside)).ravel()

    usable = (ells > 0) & np.isfinite(values) & (values > 0)
    if not np.any(usable):
        raise RuntimeError(f"{label} has no positive entries on the rfft grid, so it "
                           f"cannot be profiled onto a 1D ell axis")

    fundamental = 2 * np.pi / (nside * pix_width)
    index = np.floor(ells / fundamental).astype(int)
    n_bin = int(index[usable].max()) + 1
    total = np.bincount(index[usable], weights = dof[usable] * np.log(values[usable]),
                        minlength = n_bin)
    norm = np.bincount(index[usable], weights = dof[usable], minlength = n_bin)
    centre = np.bincount(index[usable], weights = dof[usable] * np.log(ells[usable]),
                         minlength = n_bin)
    filled = norm > 0
    profile_ell = np.exp(centre[filled] / norm[filled])
    profile = np.exp(total[filled] / norm[filled])

    grid_max = float(np.max(ells[usable]))
    if np.max(ell_axis) > grid_max:
        print(f"  WARNING: {label} is log-log extrapolated past l = {grid_max:.0f} (the "
              f"grid's largest mode) out to l = {np.max(ell_axis):.0f}")
    return interpolate_spectrum(ell_axis, profile_ell, profile)


def delensing_alens(cls, nphi_cl):
    """Alens_L = N_L / (C_L^phiphi + N_L): the fraction of the lensing power left in the map
    after delensing with the Wiener-filtered reconstruction C / (C + N) phi_hat. Zero-based in
    L (CAMB's convention for get_partially_lensed_cls) with Alens = 1 at L = 0, 1."""
    cphi = np.asarray(cls["phi"])
    nphi = np.asarray(nphi_cl)
    alens = np.ones(2 + len(cphi))
    alens[2:] = nphi / (cphi + nphi)
    return alens


# ── The reconstruction noise ──────────────────────────────────────────────

def _qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, l_cutoff,
                 filter_tt, qe_response):
    """(N_phi matrix, its radial profile on the cls' own 2..lmax-1 axis)."""
    matrix = qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                             l_cutoff, filter_tt = filter_tt, qe_response = qe_response)
    ells = np.arange(2, 2 + cls["total_TT"].shape[0], dtype = np.float64)
    return matrix, radial_cl_profile(matrix, ell_grid, pix_width, ells,
                                     label = "N_phi (QE matrix)")


def iterative_delensing(param_ground, cls, nside, pix_width, ell_grid, noise_level,
                        l_knee, beam_fwhm, l_cutoff, max_iterations = 25,
                        tolerance = 1e-6, verbose = True,
                        qe_response = DEFAULT_QE_RESPONSE, camb_lmax = None):
    """Fixed-point iteration between the per-L delensing fraction and the QE noise.

    Delensing quiets the map, which quiets the estimator's filter, which lowers N_phi, which
    delenses better - iterated to a fixed point as iterative lensing reconstruction does
    (Hirata & Seljak 2003; Smith et al. 2012):

        Alens_0        = 1                       (the filter sees the lensed map)
        N_n            = radial profile of the QE matrix with the delensed TT_n as the filter
                         and, for qe_response = "gradient", the delensed T-grad-T_n as response
        Alens_{n+1}    = N_n / (C_phi + N_n)
        TT_{n+1}       = CAMB partially lensed at Alens_{n+1}   (delensed_cls_at_params)
        TgradT_{n+1}   = CAMB T-grad-T at Alens_{n+1}   (delensed_gradient_cls_at_params)

    `cls` must carry what `qe_response` needs (cls_with_qe_response). The first iteration is
    the one-shot lensed-filter answer. Returns (alens, nphi_cl, delensed_tt, response_cls,
    converged, iterations), with response_cls `cls` carrying the final step's response.
    """
    alens = np.ones(2 + cls["phi"].shape[0])
    filter_tt = cls["total_TT"]
    response_cls = cls
    converged = False
    for iteration in range(1, max_iterations + 1):
        _, nphi_cl = _qe_noise_cl(response_cls, nside, pix_width, ell_grid, noise_level,
                                  l_knee, beam_fwhm, l_cutoff, filter_tt, qe_response)
        updated = delensing_alens(cls, nphi_cl)
        shift = float(np.max(np.abs(updated - alens)))
        alens = updated
        filter_tt = delensed_cls_at_params(param_ground, alens, camb_lmax = camb_lmax)["delensed_TT"]
        if qe_response == "gradient":
            response_cls = dict(cls, gradient_TT = delensed_gradient_cls_at_params(
                param_ground, alens, camb_lmax = camb_lmax))
        if verbose:
            print(f"    iteration {iteration}: max Alens shift {shift:.2e}")
        if shift < tolerance:
            converged = True
            break

    if not converged and verbose:
        print(f"    WARNING: delensing iteration did not converge in {max_iterations} "
              f"steps (last shift {shift:.2e}); the reported Alens_L is the last iterate")
    return alens, nphi_cl, filter_tt, response_cls, converged, iteration


def qe_reconstruction(param_ground, nside, pix_width, ell_grid, noise_level, l_knee,
                      beam_fwhm, l_cutoff, qe_filter, qe_response = DEFAULT_QE_RESPONSE,
                      camb_lmax = None, verbose = True):
    """(N_phi matrix, Alens_L) of the box's quadratic estimator at `param_ground`.

    `qe_filter` (QE_FILTERS) is the TT spectrum in the estimator's filter: "lensed" the lensed
    C_TT, "delensed" iterated to the partially delensed fixed point (iterative_delensing; the
    matrix is then rebuilt with the final step's filter AND response, and Alens_L is the
    iteration's), "unlensed" the unlensed C_TT. Alens_L is delensing_alens of the matrix's
    radial profile otherwise. `qe_response` (QE_RESPONSE_SOURCES) applies to all three.
    """
    cls = cls_with_qe_response(param_ground, qe_response, camb_lmax = camb_lmax)
    if qe_filter == "delensed":
        if verbose:
            print(f"  iterating the QE filter from the lensed to the delensed C_TT "
                  f"({qe_response} response):")
        alens, _, delensed_tt, response_cls, converged, iterations = iterative_delensing(
            param_ground, cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
            l_cutoff, verbose = verbose, qe_response = qe_response, camb_lmax = camb_lmax)
        nphi = qe_noise_matrix(response_cls, nside, pix_width, ell_grid, noise_level, l_knee,
                               beam_fwhm, l_cutoff, filter_tt = delensed_tt,
                               qe_response = qe_response)
        if verbose:
            print(f"  converged {converged} after {iterations} iterations")
        return nphi, alens
    filter_tt = cls["scalar_TT"] if qe_filter == "unlensed" else cls["total_TT"]
    nphi, nphi_cl = _qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                                 beam_fwhm, l_cutoff, filter_tt, qe_response)
    return nphi, delensing_alens(cls, nphi_cl)


def check_box(merged, nside, theta_pix, noise_level, l_knee, path = "", what = "file"):
    """Refuse a measurement taken on a different box (nside / theta_pix / noise / l_knee):
    it absorbs the box's resolution and reconstruction noise, so it cannot be carried across."""
    measured = dict(nside = int(merged["nside"]), theta_pix = float(merged["theta_pix"]),
                    noise_level = float(merged["noise_level"]),
                    l_knee = float(merged["l_knee"]))
    wanted = dict(nside = int(nside), theta_pix = float(theta_pix),
                  noise_level = float(noise_level), l_knee = float(l_knee))
    if measured != wanted:
        raise ValueError(f"the {what} {path} was measured at {measured} but this run is at "
                         f"{wanted}; re-measure it at this box")


def _check_centre(merged, param_ground, path, what):
    measured = {str(name): float(value)
                for name, value in zip(merged["param_names"], merged["params"])}
    if not same_cosmology(measured, param_ground):
        raise ValueError(f"the {what} {path} was measured about {measured}, not at this "
                         f"run's fiducial point {_point_text(param_ground)}")


def load_phi_noise_matrix(path, nside, theta_pix, noise_level, l_knee, param_ground,
                          key = None):
    """(nphi, key, source, merged): a per-mode phi noise measured by an earlier run.

    `path` is a merged npz holding a noise grid on the rfft grid of THIS box at the fiducial
    cosmology: a merge_sampler_effective_noise.py npz (`nphi`; `nphi_smooth` or an
    `nphi_*_raw` by `key`) -> source "sample_joint"; anything else -> "map_joint": a
    merge_map_joint_phi_noise.py npz (`nphi`) or an older merge_delensed_covariance.py npz
    with phi moments (`phi_noise_fid`), both NaN where unmeasured. `key` None takes the first
    of PHI_NOISE_KEYS the file has. Refuses another box or cosmology. `nphi` is returned as
    stored.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no phi noise file at {path}")
    merged = dict(np.load(path, allow_pickle = True))
    if key is None:
        key = next((name for name in PHI_NOISE_KEYS if name in merged), None)
        if key is None:
            raise ValueError(f"{path} has none of {PHI_NOISE_KEYS}: it must be a "
                             f"merge_sampler_effective_noise.py npz or a "
                             f"merge_delensed_covariance.py npz with phi moments, or the "
                             f"grid to use must be named")
    elif key not in merged:
        raise ValueError(f"{path} has no `{key}`")
    check_box(merged, nside, theta_pix, noise_level, l_knee, path = path,
              what = "phi noise")
    _check_centre(merged, param_ground, path, "phi noise")
    nphi = np.asarray(merged[key], dtype = float)
    if nphi.shape != (nside, nside // 2 + 1):
        raise ValueError(f"{path}: `{key}` has shape {nphi.shape}, expected the rfft grid "
                         f"{(nside, nside // 2 + 1)}")
    if SAMPLE_JOINT_MARKER in merged:
        source = "sample_joint"
    else:
        source = "map_joint"
    return nphi, key, source, merged


def frozen_reconstruction(param_ground, nside, theta_pix, noise_level, l_knee, beam_fwhm,
                          l_cutoff, nphi_source, qe_response = DEFAULT_QE_RESPONSE,
                          phi_noise = None, phi_noise_key = None, camb_lmax = None,
                          verbose = True):
    """(N_phi matrix, Alens_L, description) at the fiducial point, held across the stencil.

    The "qe_" sources run qe_reconstruction with the filter their name gives. The file
    sources load `phi_noise` (load_phi_noise_matrix) and check it IS that kind of noise;
    Alens_L is then delensing_alens of its radial profile. `description` records what was
    used (for the report, the output npz and the empirical-block consistency check).
    """
    if nphi_source not in NPHI_SOURCES:
        raise ValueError(f"nphi_source must be one of {NPHI_SOURCES}, got {nphi_source!r}")
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    camb_lmax = camb_lmax_for_grid(ell_grid) if camb_lmax is None else int(camb_lmax)

    if nphi_source in QE_NPHI_SOURCES:
        if phi_noise:
            raise ValueError(f"--phi_noise was given but nphi_source is {nphi_source!r}, "
                             f"which computes N_phi itself; pass --nphi_source "
                             f"{' or '.join(FILE_NPHI_SOURCES)} to use the file")
        qe_filter = nphi_source[len("qe_"):]
        nphi, alens = qe_reconstruction(param_ground, nside, pix_width, ell_grid,
                                        noise_level, l_knee, beam_fwhm, l_cutoff, qe_filter,
                                        qe_response = qe_response, camb_lmax = camb_lmax,
                                        verbose = verbose)
        description = dict(nphi_source = nphi_source, qe_filter = qe_filter,
                           qe_response = qe_response, phi_noise = "", phi_noise_key = "")
        text = (f"box QE matrix, {qe_filter} C_TT in the filter, {qe_response} response")
    else:
        if not phi_noise:
            raise ValueError(f"nphi_source = {nphi_source!r} needs --phi_noise <npz>: a "
                             + ("merge_map_joint_phi_noise.py npz"
                                if nphi_source == "map_joint"
                                else "merge_sampler_effective_noise.py npz"))
        stored, key, source, merged = load_phi_noise_matrix(
            phi_noise, nside, theta_pix, noise_level, l_knee, param_ground,
            key = phi_noise_key or None)
        if source != nphi_source:
            raise ValueError(f"{phi_noise} holds a {source!r} noise (`{key}`), not the "
                             f"{nphi_source!r} one nphi_source asks for")
        cls = camb_cls_at_params(param_ground, camb_lmax = camb_lmax)
        ells = np.arange(2, 2 + cls["total_TT"].shape[0], dtype = np.float64)
        alens = delensing_alens(cls, radial_cl_profile(stored, ell_grid, pix_width, ells,
                                                       label = f"N_phi ({source})"))
        nphi = jnp.asarray(stored)
        unmeasured = int(np.sum(~np.isfinite(stored)))
        description = dict(nphi_source = nphi_source, qe_filter = "", qe_response = "",
                           phi_noise = str(phi_noise), phi_noise_key = key)
        text = (f"{source} noise `{key}` from {phi_noise}"
                + (f" ({merged['n_realizations']} realizations)"
                   if "n_realizations" in merged else "")
                + (f"; {unmeasured} unmeasured modes carry no phi information"
                   if unmeasured else ""))

    if verbose:
        cls = camb_cls_at_params(param_ground, camb_lmax = camb_lmax)
        alpha = delensing_efficiency(_covar(nside, pix_width, ell_grid, cls["phi"]), nphi)
        samples = ", ".join(f"L={L} {alens[L]:.2f}" for L in (50, 100, 300, 1000, 2000))
        print(f"  N_phi [{nphi_source}]: {text}")
        print(f"  delensing: C_phi-weighted mean efficiency {alpha:.3f}; residual fraction "
              f"Alens_L at {samples}")
    return nphi, alens, description


# ── The empirical delensed temperature block ──────────────────────────────

def load_delensed_covariance(path):
    """The empirical delensed covariance stencil written by merge_delensed_covariance.py:
    the realization mean of |rfft2(L^-1(W phi) L(phi) f)|^2 / nside^2 at theta_0 and
    theta_0 +/- h_i on common random numbers (cmb_lensing/delensed_covariance.py)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no delensed covariance at {path}. Produce one with "
            f"get_delensed_field_covariance/get_delensed_covariance.sh and "
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
    """Validate an empirical delensed covariance against the stencil it replaces and return
    its steps h_i for `names` (used for BOTH blocks, so the phi block is differenced over the
    same interval). The box and the centre must match and every sampled parameter must have
    been measured - there is no model to fall back on for a missing direction."""
    check_box(merged, nside, theta_pix, noise_level, l_knee, path = path,
              what = "delensed covariance")
    _check_centre(merged, param_ground, path, "delensed covariance")
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


# ── Covariance blocks and the stencil ─────────────────────────────────────

def covariance_blocks(cls, nside, pix_width, ell_grid, noise, nphi):
    """{"f_delensed": C_TT^delensed + C_n, "phi": C_phi + N_phi} on the rfft grid, built
    through the same covar_matrix_from_cls the sampler uses, so signal and noise share a
    normalization (the trace formula is invariant under any theta-independent rescaling)."""
    return {"f_delensed": _covar(nside, pix_width, ell_grid, cls["delensed_TT"]) + noise,
            "phi": _covar(nside, pix_width, ell_grid, cls["phi"]) + nphi}


def sampled_names_and_steps(is_sampled, param_ground, step_fracs):
    """(names, steps, fracs) - the sampled subset in OUTPUT_PARAM_ORDER and its FD steps."""
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


def covariance_stencil(nside, theta_pix, noise_level, is_sampled, param_ground,
                       nphi_source = DEFAULT_NPHI_SOURCE, phi_noise = None,
                       phi_noise_key = None, qe_response = DEFAULT_QE_RESPONSE,
                       delensed_covariance = None, step_fracs = None, l_knee = 0,
                       beam_fwhm = 0, l_cutoff = DEFAULT_ELL_CUTOFF, camb_lmax = None, verbose = True,
                       return_applier = False):
    """The covariance-block stencil the forecast contracts.

    Returns (names, steps, weights, blocks_fid, blocks_plus, blocks_minus, description): the
    sampled names, their FD steps, the real-DOF weights w_k, the block dicts at theta_0 and
    at theta_0 +/- h_i per sampled parameter, and frozen_reconstruction's description of the
    N_phi. N_phi and Alens_L are built once at theta_0 (frozen_reconstruction); CAMB
    re-delenses at every point. With `delensed_covariance` the f block's signal at every
    point is the empirical one (C_n is still added analytically) and its steps replace
    FD_STEP_FRAC; `return_applier` then also returns the function that swaps a set of
    per-mode grids into copies of the base blocks, which forecast_jackknife calls once per
    leave-one-out set. `camb_lmax` defaults to camb_lmax_for_grid.
    """
    names, steps, _ = sampled_names_and_steps(is_sampled, param_ground, step_fracs)
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    weights = fourier_weights(nside)
    camb_lmax = camb_lmax_for_grid(ell_grid) if camb_lmax is None else int(camb_lmax)

    empirical = None
    if delensed_covariance is not None:
        if step_fracs is not None:
            raise ValueError("the finite-difference steps are fixed by the empirical "
                             "delensed covariance's stencil, so step_fracs (and "
                             "--stability) cannot be applied")
        empirical = load_delensed_covariance(delensed_covariance)
        steps = check_delensed_covariance(empirical, nside, theta_pix, noise_level, l_knee,
                                          param_ground, names, path = delensed_covariance)
    elif return_applier:
        raise ValueError("return_applier needs delensed_covariance: there is no empirical "
                         "block to resample otherwise")

    if verbose:
        ells_on_grid = ell_grid[ell_grid > 0]
        print(f"Fisher forecast [{nphi_source}]: nside {nside}, theta_pix {theta_pix}', "
              f"{noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  box {nside * theta_pix / 60:.2f} deg, ell in "
              f"[{float(jnp.min(ells_on_grid)):.0f}, {float(jnp.max(ells_on_grid)):.0f}], "
              f"{int(jnp.sum(weights))} real DOF; CAMB to l = {camb_lmax - 1}")
        print(f"  sampled: {names}")

    nphi, alens, description = frozen_reconstruction(
        param_ground, nside, theta_pix, noise_level, l_knee, beam_fwhm, l_cutoff, nphi_source,
        qe_response = qe_response, phi_noise = phi_noise, phi_noise_key = phi_noise_key,
        camb_lmax = camb_lmax, verbose = verbose)

    cls_fid = camb_cls_at_params(param_ground, camb_lmax = camb_lmax)
    noise, _, _ = _instrument_matrices(nside, pix_width, ell_grid, noise_level, l_knee,
                                       beam_fwhm, l_cutoff,
                                       lmax_prime = 2 + cls_fid["total_TT"].shape[0])

    def blocks_at(params):
        cls = delensed_cls_at_params(params, alens, camb_lmax = camb_lmax)
        return covariance_blocks(cls, nside, pix_width, ell_grid, noise, nphi)

    blocks_fid = blocks_at(param_ground)
    blocks_plus, blocks_minus = [], []
    for name, step in zip(names, steps):
        up, down = dict(param_ground), dict(param_ground)
        up[name] = param_ground[name] + step
        down[name] = param_ground[name] - step
        blocks_plus.append(blocks_at(up))
        blocks_minus.append(blocks_at(down))
        if verbose:
            print(f"    {name}: h = {step:.6g} ({step / PARAM_SIGMA[name]:g} sigma)")

    if empirical is None:
        return names, steps, weights, blocks_fid, blocks_plus, blocks_minus, description

    measured_names = [str(name) for name in empirical["names"]]

    def apply_empirical(grids):
        """Copies of the base blocks with the empirical f_delensed swapped in from `grids`
        (the merged npz, or one delensed_covariance.leave_one_out_grids set)."""
        fid = dict(blocks_fid, f_delensed = jnp.asarray(grids["delensed_fid"]) + noise)
        plus, minus = [], []
        for i, name in enumerate(names):
            j = measured_names.index(name)
            plus.append(dict(blocks_plus[i],
                             f_delensed = jnp.asarray(grids["delensed_plus"][j]) + noise))
            minus.append(dict(blocks_minus[i],
                              f_delensed = jnp.asarray(grids["delensed_minus"][j]) + noise))
        return fid, plus, minus

    if verbose:
        print(f"  f_delensed block: EMPIRICAL ({int(empirical['n_realizations'])} "
              f"realizations) from {delensed_covariance}")
    result = (names, steps, weights) + apply_empirical(empirical) + (description,)
    return result + (apply_empirical,) if return_applier else result


def _fisher_from_blocks(blocks_plus, blocks_minus, blocks_fid, steps, weights):
    """F_ij = 1/2 sum_blocks sum_k w_k dln(C)/dtheta_i dln(C)/dtheta_j.

    Entries whose centre is not positive and finite carry no information: the zeroed origin,
    and an empirical N_phi's unmeasured (NaN) modes.
    """
    n_param = len(steps)
    fisher = np.zeros((n_param, n_param))
    for name in blocks_fid:
        c_fid = blocks_fid[name]
        good = jnp.isfinite(c_fid) & (c_fid > 0)
        safe = jnp.where(good, c_fid, 1.0)
        dlog = [jnp.where(good, (blocks_plus[i][name] - blocks_minus[i][name])
                          / (2 * steps[i] * safe), 0.0) for i in range(n_param)]
        for i in range(n_param):
            for j in range(i, n_param):
                value = 0.5 * float(jnp.sum(weights * dlog[i] * dlog[j]))
                fisher[i, j] += value
                if i != j:
                    fisher[j, i] += value
    return fisher


def forecast(nside, theta_pix, noise_level, is_sampled, param_ground,
             nphi_source = DEFAULT_NPHI_SOURCE, phi_noise = None, phi_noise_key = None,
             qe_response = DEFAULT_QE_RESPONSE, delensed_covariance = None,
             step_fracs = None, l_knee = 0, beam_fwhm = 0, l_cutoff = DEFAULT_ELL_CUTOFF,
             camb_lmax = None, verbose = True):
    """Gaussian Fisher matrix for the sampled LCDM parameters on an nside x nside box.

    Args:
        nside, theta_pix, noise_level: the box (pixels per side, arcmin, uK-arcmin)
        is_sampled:   {param_name: bool}; False parameters are held FIXED
        param_ground: {param_name: float} fiducial point, all five
        nphi_source:  NPHI_SOURCES - "qe_unlensed" (the upper bound on the information),
                      "qe_delensed" (the lower bound, default), "map_joint" or
                      "sample_joint" (read from `phi_noise`)
        phi_noise:    the npz for the file sources; `phi_noise_key` names a non-default grid
        qe_response:  QE_RESPONSE_SOURCES, the "qe_" sources' response spectrum
        delensed_covariance: a merge_delensed_covariance.py npz; its empirical delensed T
                      block replaces CAMB's at every stencil point and its steps replace
                      FD_STEP_FRAC. Incompatible with step_fracs
        step_fracs:   override for FD_STEP_FRAC (fractions of PARAM_SIGMA)
        l_knee, beam_fwhm: 1/f knee (0 = white, as every sampler entry point) and beam FWHM
        camb_lmax:    how far CAMB is run; default camb_lmax_for_grid (the box's corner)

    Returns:
        (fisher, names) - the n_sampled x n_sampled matrix and the names in
        OUTPUT_PARAM_ORDER (chain_analysis.py's order, not PARAM_ORDER).
    """
    names, steps, weights, fid, plus, minus, _ = covariance_stencil(
        nside, theta_pix, noise_level, is_sampled, param_ground, nphi_source = nphi_source,
        phi_noise = phi_noise, phi_noise_key = phi_noise_key, qe_response = qe_response,
        delensed_covariance = delensed_covariance, step_fracs = step_fracs, l_knee = l_knee,
        beam_fwhm = beam_fwhm, l_cutoff = l_cutoff, camb_lmax = camb_lmax, verbose = verbose)
    return _fisher_from_blocks(plus, minus, fid, steps, weights), names


def forecast_jackknife(nside, theta_pix, noise_level, is_sampled, param_ground,
                       delensed_covariance, covariance_dir = None, verbose = True,
                       **stencil_kwargs):
    """forecast(..., delensed_covariance = ...) plus its delete-one jackknife samples.

    The empirical f block is a realization mean, so the forecast carries Monte Carlo error.
    Each realization is left out in turn: the per-mode grids are re-formed from the delete-one
    means by delensed_covariance.leave_one_out_grids (the merge's own forecast_grids), swapped
    into the same stencil (CAMB, C_n and N_phi built once) and contracted. The caller inverts
    each sample and jackknifes whatever it needs. `covariance_dir` holds the per-realization
    job files (default: the merged npz's own directory; same seeds required).
    `stencil_kwargs` are covariance_stencil's (nphi_source, phi_noise, qe_response, ...).

    Returns (fisher, names, leave_one_out), the last an (n_realizations, n, n) stack.
    """
    from cmb_lensing.delensed_covariance import leave_one_out_grids

    if covariance_dir is None:
        covariance_dir = os.path.dirname(os.path.abspath(delensed_covariance))
    names, steps, weights, fid, plus, minus, _, apply_empirical = covariance_stencil(
        nside, theta_pix, noise_level, is_sampled, param_ground,
        delensed_covariance = delensed_covariance, verbose = verbose, return_applier = True,
        **stencil_kwargs)
    fisher = _fisher_from_blocks(plus, minus, fid, steps, weights)

    merged = load_delensed_covariance(delensed_covariance)
    seeds, samples = leave_one_out_grids(covariance_dir, verbose = False)
    if seeds != [int(seed) for seed in merged["seeds"]]:
        raise ValueError(f"{covariance_dir} holds different realizations (seeds) from those "
                         f"{delensed_covariance} was merged from; re-run "
                         f"merge_delensed_covariance.py on that directory")

    leave_one_out = []
    for grids in samples:
        loo_fid, loo_plus, loo_minus = apply_empirical(grids)
        leave_one_out.append(_fisher_from_blocks(loo_plus, loo_minus, loo_fid, steps, weights))
    if verbose:
        print(f"  jackknife: {len(leave_one_out)} delete-one forecasts from {covariance_dir}")
    return fisher, names, np.array(leave_one_out)


# ── Post-processing ───────────────────────────────────────────────────────

def covariance_from_fisher(fisher, names):
    """Invert the Fisher matrix, failing loudly on a singular / non-PSD result."""
    eigenvalues = np.linalg.eigvalsh(fisher)
    if np.any(eigenvalues <= 0):
        raise RuntimeError(
            f"Fisher matrix is not positive definite (eigenvalues {eigenvalues}). "
            f"Usually a degenerate parameter pair at this box size, or a finite-difference "
            f"step that is too small - try raising FD_STEP_FRAC and re-running --stability.")
    condition = eigenvalues[-1] / eigenvalues[0]
    if condition > 1e12:
        print(f"WARNING: Fisher condition number {condition:.3g} - the inverse is "
              f"numerically unreliable, so at least one direction is near-degenerate "
              f"over {names}")
    return np.linalg.inv(fisher)


def correlation_from_covariance(covariance):
    """F^-1 normalized to unit diagonal."""
    sigmas = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigmas, sigmas)


def step_stability(forecast_fn, method = METHOD, factor = 2.0):
    """Recompute a forecast at `factor` x the step size and report the per-parameter
    fractional drift in sigma. `forecast_fn(step_fracs) -> (fisher, names)`; the CAMB runs
    are memoized, so the doubled-step run is one extra set of calls."""
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


def report_sigmas(covariance, names, label, method = METHOD):
    """Print the marginalized 1-sigma errors and the strongest degeneracy."""
    sigmas = np.sqrt(np.diag(covariance))
    print(f"\nForecast 1-sigma errors [{label}, {method}]:")
    print(f"  {'parameter':<14s} {'sigma':>12s} {'sigma/PARAM_SIGMA':>20s}")
    for i, name in enumerate(names):
        print(f"  {name:<14s} {sigmas[i]:>12.4g} {sigmas[i] / PARAM_SIGMA[name]:>20.3g}")
    if len(names) > 1:
        correlation = covariance / np.outer(sigmas, sigmas)
        pairs = [(abs(correlation[i, j]), correlation[i, j], names[i], names[j])
                 for i in range(len(names)) for j in range(i + 1, len(names))]
        _, value, first, second = max(pairs)
        print(f"  strongest degeneracy: {first} - {second}  (r = {value:+.2f})")


# ── Plotting ──────────────────────────────────────────────────────────────

def annotated_heatmap(axis, matrix, names, title, colorbar_label):
    """Heatmap coloured by the correlation-normalized matrix (bounded, readable) with the
    raw entries annotated - the raw values span many orders of magnitude across parameters
    with wildly different units, so colouring by them directly is useless."""
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


def plot_annotated_matrix(matrix, param_names, title, path, cmap = "coolwarm", norm = None,
                          fmt = "{:+.3f}"):
    """A square matrix as a coloured grid with every entry printed in its box (black or white
    text by the luminance behind it). `norm` defaults to a linear scale symmetric about 0."""
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
    plt.savefig(path, dpi = 150, bbox_inches = "tight")
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
    """Write the three figures and the npz for one run, tagged with `suffix`. `config` is
    stored alongside the matrices so a saved forecast can be traced back to its run.
    Returns the file names written."""
    full_subtitle = f"{subtitle}\n{label}"
    plot_fisher_matrix(fisher, names, os.path.join(directory, f"fisher_matrix{suffix}.png"),
                       full_subtitle)
    plot_covariance_matrix(covariance, names,
                           os.path.join(directory, f"covariance_matrix{suffix}.png"),
                           full_subtitle)
    correlation = correlation_from_covariance(covariance)
    plot_annotated_matrix(correlation, names, f"Forecast Correlation Matrix\n{full_subtitle}",
                          os.path.join(directory, f"correlation_matrix{suffix}.png"),
                          norm = matplotlib.colors.Normalize(vmin = -1, vmax = 1))
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


# ── CLI ───────────────────────────────────────────────────────────────────

def add_box_arguments(parser):
    """--nside / --theta_pix / --noise / --l_knee / --beam_fwhm / --params."""
    parser.add_argument("--nside", type = int, default = 128)
    parser.add_argument("--theta_pix", type = float, default = 2.5,
                        help = "pixel width in arcmin")
    parser.add_argument("--noise", type = float, default = 2.5,
                        help = "white noise level in uK-arcmin")
    parser.add_argument("--l_knee", type = float, default = 0.0,
                        help = "1/f knee; 0 (default) matches every sampler entry point")
    parser.add_argument("--beam_fwhm", type = float, default = 0.0)
    parser.add_argument("--params", nargs = "*", default = ["logA", "omch2", "theta_MC_100"],
                        help = f"subset to sample, from {PARAM_ORDER}")
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


def box_subtitle(label, args):
    """The first line of every figure subtitle: what was run, and the box."""
    return (f"{label}  |  nside {args.nside}, {args.theta_pix:g}', "
            f"{args.noise:g} uK-arcmin")


def run_config(args, names, steps = None, **extra):
    """The run configuration stored in every fisher<suffix>.npz. `steps` are the FD steps
    actually used (default FD_STEP_FRAC's); np.savez cannot store None, so absent paths are
    the empty string."""
    if steps is None:
        steps = [FD_STEP_FRAC[name] * PARAM_SIGMA[name] for name in names]
    return dict(nside = args.nside, theta_pix = args.theta_pix, noise_level = args.noise,
                l_knee = args.l_knee, beam_fwhm = args.beam_fwhm,
                steps = np.asarray(steps, dtype = float),
                step_fracs = np.array([step / PARAM_SIGMA[name]
                                       for name, step in zip(names, steps)]), **extra)


def main():
    parser = argparse.ArgumentParser(
        description = "Gaussian Fisher forecast for the LCDM parameters on the "
                      "sample_lcdm.py flat-sky box: delensed T block + phi block")
    add_box_arguments(parser)
    parser.add_argument("--nphi_source", choices = NPHI_SOURCES, default = DEFAULT_NPHI_SOURCE,
                        help = "the reconstruction noise N_phi in the phi block and behind "
                               "Alens_L. 'qe_unlensed': the box QE with the unlensed C_TT in "
                               "its filter, the UPPER bound on the information; "
                               "'qe_delensed' (default): the filter iterated to the "
                               "partially delensed C_TT, the LOWER bound; 'map_joint' / "
                               "'sample_joint': the noise measured for map_joint / the "
                               "sampler, from --phi_noise")
    parser.add_argument("--phi_noise", type = str, default = None,
                        help = "--nphi_source map_joint: a merge_map_joint_phi_noise.py npz "
                               "(its nphi) or an older merge_delensed_covariance.py npz "
                               "with phi moments (its phi_noise_fid); sample_joint: a "
                               "merge_sampler_effective_noise.py npz (its nphi). Refused if "
                               "measured at another box or cosmology")
    parser.add_argument("--phi_noise_key", type = str, default = None,
                        help = "which 2D grid of --phi_noise to use (default nphi, else "
                               "phi_noise_fid), e.g. nphi_smooth")
    parser.add_argument("--qe_response", choices = QE_RESPONSE_SOURCES,
                        default = DEFAULT_QE_RESPONSE,
                        help = "the TT spectrum weighting the QE response for the qe_ "
                               "sources: 'unlensed' (default, Hu & Okamoto; what the sampler "
                               "uses) or 'gradient' (CAMB's lensed T-grad-T, iterated with "
                               "the filter under qe_delensed)")
    parser.add_argument("--delensed_covariance", type = str, default = None,
                        help = "a merge_delensed_covariance.py npz: the f block's delensed "
                               "signal at theta_0 and theta_0 +/- h is the EMPIRICAL "
                               "per-mode covariance of the box's own delensing instead of "
                               "CAMB's partially lensed spectrum, and the steps come from "
                               "that file. Warns when its W was built from a different N_phi")
    parser.add_argument("--camb_lmax", type = int, default = None,
                        help = "how far to run CAMB; default: the rfft grid's corner mode "
                               f"(camb_lmax_for_grid). {DEFAULT_MAX_ELL} reproduces the "
                               f"load_sim support")
    parser.add_argument("--stability", action = "store_true",
                        help = "also recompute at 2x the step size and report the drift "
                               "(not with --delensed_covariance, whose steps are fixed)")
    args = parser.parse_args()
    is_sampled = sampled_from_args(parser, args)
    if args.delensed_covariance and args.stability:
        parser.error("--stability cannot be combined with --delensed_covariance: the step is "
                     "fixed by the measured stencil")

    def run(step_fracs = None, verbose = True):
        return covariance_stencil(
            args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
            nphi_source = args.nphi_source, phi_noise = args.phi_noise,
            phi_noise_key = args.phi_noise_key, qe_response = args.qe_response,
            delensed_covariance = args.delensed_covariance, step_fracs = step_fracs,
            l_knee = args.l_knee, beam_fwhm = args.beam_fwhm, camb_lmax = args.camb_lmax,
            verbose = verbose)

    names, steps, weights, fid, plus, minus, description = run()
    fisher = _fisher_from_blocks(plus, minus, fid, steps, weights)
    covariance = covariance_from_fisher(fisher, names)
    t_block = "empirical" if args.delensed_covariance else "camb"
    label = f"N_phi {args.nphi_source}, {t_block} delensed T"
    report_sigmas(covariance, names, label)

    if args.stability:
        def forecast_fn(fracs):
            stencil = run(step_fracs = fracs, verbose = False)
            return _fisher_from_blocks(stencil[4], stencil[5], stencil[3], stencil[1],
                                       stencil[2]), stencil[0]
        step_stability(forecast_fn)

    save_outputs(fisher, covariance, names, METHOD,
                 f"{SUFFIX}_{args.nphi_source}_{t_block}_{args.noise}_uk", LABEL,
                 box_subtitle(label, args),
                 run_config(args, names, steps = steps, t_block = t_block,
                            delensed_covariance = args.delensed_covariance or "",
                            camb_lmax = args.camb_lmax or 0, **description))


if __name__ == "__main__":
    main()
