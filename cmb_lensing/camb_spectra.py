"""Direct CAMB spectra at an LCDM point, and the rfft-grid matrices built from them.

Shared by the Fisher forecast and the measurement modules (delensed covariance, phi noise):

    camb_cls_at_params    the cls dict of one CAMB run, exactly what load_sim extracts
    load_sim_cosmology    a parameter dict in load_sim's keyword names
    covar_from_cl         a C_l put on the rfft grid (covar_matrix_from_cls units)
    instrument_matrices   the analytic noise, mask and beam
    qe_noise_matrix       the quadratic-estimate N0 of the box, load_sim's estimator

Every spectrum comes from a DIRECT CAMB call, never the 5D grid spline.
"""

import numpy as np
import camb
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from cmb_lensing.simulate import (camb_parameters, _extract_all_cls, _covar_or_zeros,
                                  noise_cls, get_beam, get_mask, scalar_quadratic_estimate,
                                  polar_quadratic_estimate)
from cmb_lensing.constants import (DEFAULT_MAX_ELL, DEFAULT_A_LENSE, DEFAULT_K_PIVOT,
                                   DEFAULT_MNU, DEFAULT_TAUREIO, PARAM_ORDER, CAMB_LMAX)

#the spectra in the quadratic estimator's inverse-variance filter: the unlensed or the fully
#lensed ones. Its response is always built from the unlensed spectra
QE_FILTERS = ("unlensed", "lensed")
DEFAULT_QE_FILTER = "unlensed"
#the cls-dict prefix of each filter's spectra
FILTER_CL_PREFIX = {"unlensed": "scalar", "lensed": "total"}
POLS = ("I", "P", "IP")
DEFAULT_POL = "I"

#decimals a parameter value is rounded to in the CAMB cache key
CACHE_KEY_DECIMALS = 12
#CAMB is deterministic in the parameters and a stencil re-queries its centre, so memoize
_CAMB_CACHE = {}


def camb_cls_at_params(params, camb_lmax = None):
    """Full cls dict from one CAMB run at a 5-parameter point, memoized per (point, lmax).

    Set up through simulate.camb_parameters, so the run is identical to load_sim's (H0 solved
    from cosmomc_theta, r = 0, mnu = 0.06, tau = 0.05, non-linear lensing), and the spectra
    are what simulate._run_camb extracts (same calls, units and _extract_all_cls), on ells
    2..camb_lmax-1. `camb_lmax` defaults to CAMB_LMAX. Raises if CAMB fails.
    """
    camb_lmax = CAMB_LMAX if camb_lmax is None else int(camb_lmax)
    key = tuple(round(float(params[name]), CACHE_KEY_DECIMALS) for name in PARAM_ORDER) + (camb_lmax,)
    if key not in _CAMB_CACHE:
        pars = camb_parameters(None, params["ombh2"], params["omch2"],
                               params["theta_MC_100"] / 100, 0.0, DEFAULT_MNU, DEFAULT_TAUREIO,
                               np.exp(params["logA"]) * 1e-10, 0, params["ns"], camb_lmax,
                               DEFAULT_K_PIVOT, DEFAULT_A_LENSE)
        results = camb.get_results(pars)
        power_spectra = results.get_cmb_power_spectra(pars, lmax = camb_lmax - 1,
                                                      CMB_unit = "muK")
        lens_potential = results.get_lens_potential_cls(lmax = camb_lmax - 1,
                                                        CMB_unit = "muK")[:, :2]
        _CAMB_CACHE[key] = _extract_all_cls(jnp.asarray(power_spectra["unlensed_scalar"]),
                                            jnp.asarray(power_spectra["tensor"]),
                                            jnp.asarray(power_spectra["total"]),
                                            jnp.asarray(lens_potential), camb_lmax, camb_lmax)
    return _CAMB_CACHE[key]


def load_sim_cosmology(param_ground):
    """param_ground translated into load_sim's keyword names (cosmomc_theta, As)."""
    camb_kwargs = dict(param_ground)
    camb_kwargs["cosmomc_theta"] = camb_kwargs.pop("theta_MC_100") / 100
    camb_kwargs["As"] = float(np.exp(camb_kwargs.pop("logA")) * 1e-10)
    return camb_kwargs


def covar_from_cl(nside, pix_width, ell_grid, cl):
    """A C_l on ells 2..len+1 put on the rfft grid by covar_matrix_from_cls, origin zeroed.
    An identically zero spectrum (the unlensed BB at r = 0) gives a zero matrix."""
    ells = jnp.arange(2, 2 + cl.shape[0]).astype(jnp.float64)
    return _covar_or_zeros(nside, pix_width, ell_grid, ells, cl, origin_value = 0)


def instrument_matrices(nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, l_cutoff,
                        lmax_prime = None):
    """(noise, mask, beam) on the rfft grid, none of them depending on the cosmology.

    `noise` is the temperature noise C_n^TT; the E and B noise are each twice it (noise_cls).
    `lmax_prime` is the multipole axis the noise and beam are built on (default load_sim's).
    """
    lmax_prime = min(DEFAULT_MAX_ELL, CAMB_LMAX) if lmax_prime is None else int(lmax_prime)
    n_tt, _, _, _ = noise_cls(lmax_prime, noise_level, beam_fwhm = beam_fwhm, l_knee = l_knee)
    noise = covar_from_cl(nside, pix_width, ell_grid, n_tt)
    beam = get_beam(nside, pix_width, ell_grid, lmax_prime, beam_fwhm = beam_fwhm)
    mask = get_mask(l_cutoff, nside, pix_width, ell_grid)
    return noise, mask, beam


def qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, l_cutoff,
                    pol = DEFAULT_POL, qe_filter = DEFAULT_QE_FILTER):
    """N_phi: the quadratic-estimate reconstruction noise N^(0) of the box, on the rfft grid.

    The estimator load_sim builds its QE norm from - the TT one for pol "I", the EB one for
    "P" and "IP" - without the division by NPHI_FAC (a preconditioning choice that cancels
    inside G), so this is the physical N^(0). The response is always built from the unlensed
    spectra; `qe_filter` (QE_FILTERS) puts the unlensed or the lensed spectra in the
    inverse-variance filter.
    """
    noise, mask, beam = instrument_matrices(nside, pix_width, ell_grid, noise_level, l_knee,
                                            beam_fwhm, l_cutoff)
    filter_prefix = FILTER_CL_PREFIX[qe_filter]

    def response(component):
        return covar_from_cl(nside, pix_width, ell_grid, cls[f"scalar_{component}"])

    def filter_matrix(component):
        return covar_from_cl(nside, pix_width, ell_grid, cls[f"{filter_prefix}_{component}"])

    if pol == "I":
        return scalar_quadratic_estimate(noise, response("TT"), filter_matrix("TT"), mask, beam,
                                         pix_width)
    polarization_noise = 2 * noise
    #polar_quadratic_estimate accumulates in a complex array whose imaginary part is zero
    return jnp.real(polar_quadratic_estimate(response("EE"), response("BB"),
                                             filter_matrix("EE"), filter_matrix("BB"),
                                             polarization_noise, polarization_noise,
                                             mask, mask, beam, beam, pix_width))
