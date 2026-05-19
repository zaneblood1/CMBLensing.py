import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
jax.config.update("jax_enable_x64", True)
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from functools import partial
import time
from cmb_lensing.util import *
from cmb_lensing.lense_flow import *
from cmb_lensing.dataset import *
from cmb_lensing.statistics import *
from cmb_lensing.constants import *
from cmb_lensing.simulate import *

#NOTE the following file / code is simliar to similate.py except it uses the NN emulator
#cosmopower-jax instead of CAMB which allows a JIT-ted entry point into the load_sim() method.
#However, this also comes with certain drawbacks: 1) Only temperature fields are supported,
#2) cosmopower-jax only produces spectra up to the ~2500 ell range so extrapolation must be 
#carried out further than in the CAMB case...

# ── Power Spectrum Processing ─────────────────────────────────────────────

def process_cls(cl_xx, lmax, lmax_prime, is_phi = False):

    ell = jnp.arange(2, lmax).astype(jnp.float64)
    ell_prime = jnp.arange(2, lmax_prime).astype(jnp.float64)
    #Take away the monopole and dipole and rescale temperature field
    cl_xx = cl_xx[2:]
    if not is_phi:
         cl_xx = 2 * jnp.pi * cl_xx

    #Now interpolate from current low ell predictions out to higher ell
    def exponential_interpolate(ell, ell_prime, cl_xx):
        return jnp.exp(jnp.interp(
            jnp.log(ell), jnp.log(ell_prime), jnp.log(cl_xx),
            left="extrapolate", right="extrapolate"
        ))
    
    def linear_interpolate(ell, ell_prime, cl_xx):
        return jnp.interp(ell, ell_prime, cl_xx, left=0.0, right=0.0)

    operands = ell, ell_prime, cl_xx
    cl_xx = jax.lax.cond(
        jnp.all(cl_xx > 0),
        exponential_interpolate, 
        linear_interpolate, 
        *operands
    )

    return cl_xx

# ── Covariance and Field Generation ──────────────────────────────────────

@partial(jax.jit, static_argnames = ["nside", "rescale"])
def covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cls, origin_value=None,
                          rescale=True, use_linear_interpolation=False, fill_value=None):

    def exponential_interpolate(ell_grid, ells, cls):
        ls_safe = jnp.where(ell_grid.flatten() == 0, 1e-10, ell_grid.flatten())
        result = jnp.exp(jnp.interp(
            jnp.log(ls_safe), jnp.log(ells), jnp.log(cls),
            left="extrapolate", right="extrapolate"
        )).reshape((nside, nside // 2 + 1))
        return result

    def linear_interpolate(ell_grid, ells, cls):
        fv = fill_value if fill_value is not None else 0.0
        result = jnp.interp(ell_grid.flatten(), ells, cls, left=fv, right=fv).reshape((nside, nside // 2 + 1))
        return result

    result = jax.lax.cond(
        jnp.logical_and(jnp.all(cls > 0), jnp.logical_not(use_linear_interpolation)),
        exponential_interpolate,
        linear_interpolate,
        ell_grid, ells, cls
    )

    if origin_value is not None:
        result = result.at[0, 0].set(origin_value)
    if rescale:
        result = result / pix_width**2
    return result

#NOTE these mask and beam methods are identical to the definitions in simulate.py  
#but we have to copy them over so that the correct JIT-ted version of
#"covar_matrix_from_cls" gets called by the fast_simulate code
def get_mask(l_cutoff, nside, pix_width, ell_grid):
    screen_cls = low_pass(l_cutoff)
    ell = jnp.arange(2, len(screen_cls)).astype(jnp.float64)
    return covar_matrix_from_cls(nside, pix_width, ell_grid, ell,
                                 screen_cls[2:], origin_value=1, rescale=False)

def get_beam(nside, pix_width, ell_grid, lmax_prime, beam_fwhm=0):
    ell_prime = jnp.arange(2, lmax_prime).astype(jnp.float64)
    b = jnp.sqrt(beam_cls(beam_fwhm, ell_prime))
    return covar_matrix_from_cls(nside, pix_width, ell_grid, ell_prime, b,
                                 origin_value=0, rescale=False,
                                 use_linear_interpolation=True, fill_value=1.0)

# ── D Temperature-Only Matrix ──────────────────────────────────────────────

def get_d_matrix(cf_tt, cn_tt):
    pre_factor = jnp.deg2rad(5 / ARCMIN_PER_DEGREE)**2
    identity = jnp.ones(cn_tt.shape)
    cf_inv_tt = jnp.nan_to_num(reciprocal_matrix(cf_tt), nan=0, posinf=0, neginf=0)
    sum_tt = cf_tt + pre_factor * identity + 2 * cn_tt
    d_tt = jnp.sqrt(sum_tt* cf_inv_tt)
    return d_tt

# ── CosmoPowerJAX Interface ────────────────────────────────────────────────────────

def _run_cpj(om_b, om_cdm, h, tau_reio, ns, As):
    cosmo_params = jnp.array([om_b, om_cdm, h, tau_reio, ns, As])
    emulator = CPJ(probe = "cmb_tt")
    cl_tt = emulator.predict(cosmo_params)
    emulator = CPJ(probe = "cmb_pp")
    cl_pp = emulator.predict(cosmo_params)
    return cl_tt, cl_pp

def _extract_all_cls(power_spectra, lens_potential, lmax, lmax_prime):
    cls = {}
    cls["lensed_TT"] = process_cls(power_spectra, lmax, lmax_prime)
    cls["phi"] = process_cls(lens_potential, lmax, lmax_prime, is_phi = True)
    return cls

# ── Dataset Construction ─────────────────────────────────────────────────

def _build_dataset_t(nside, theta_pix, pix_width, cphi, qe, phi_real,
                     cf, cn, d_tt, mask, beam,
                     field_t, lensed_t, data_t):
    _, phi_cov, qe_op, phi = build_phi_template(nside, theta_pix, pix_width, cphi, qe, phi_real)
    return DataSetT(
        noise_covariance=phi_cov.replace(scalar_matrix = cn),
        field_covariance=phi_cov.replace(scalar_matrix = cf),
        mixing_d=phi_cov.replace(scalar_matrix=d_tt),
        phi_covariance=phi_cov,
        mask=phi_cov.replace(scalar_matrix=mask),
        beam=phi_cov.replace(scalar_matrix=beam),
        quadratic_estimate=qe_op,
        data=phi.replace(scalar_matrix=data_t),
        unlensed_field=phi.replace(scalar_matrix=field_t),
        lensed_field=phi.replace(scalar_matrix=lensed_t),
        phi=phi,
    )


# ── Main Simulation Entry Point ──────────────────────────────────────────

@partial(jax.jit, static_argnames = ["nside", "theta_pix", "lmax"])
def load_sim(nside, theta_pix, master_seed = None, uk_arcmin_t = 3, 
             om_b = 0.0224567, om_cdm = 0.118489, h = 0.68, tau_reio = 0.055, 
             ns = 0.968602, As = 3.043, nphi_fac = 2, lmax = 17_000):

    lmax_prime = min(lmax, CPJ_MAX_ELL)

    power_spectra, lens_potential = _run_cpj(om_b, om_cdm, h, tau_reio, ns, As)
    cls = _extract_all_cls(power_spectra, lens_potential, lmax, lmax_prime)

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    if master_seed is None:
        master_seed = np.random.randint(0, 2**31)
    keys = jax.random.split(jax.random.PRNGKey(master_seed), 100)
    ells = jnp.arange(2, lmax).astype(jnp.float64)

    #Lensing potential
    cphi = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cls["phi"], origin_value=0)
    phi, kc = field_from_covar(nside, cphi, keys, 0)

    #Lensed field covariances
    cfl_tt = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                   cls[f"lensed_TT"], origin_value = 0)

    #Lensed temperature field
    lensed_t, kc = field_from_covar(nside, cfl_tt, keys, kc)
    #Convert to micro-Kelvin unit system
    lensed_t =  MUK_FACTOR * lensed_t

    #Instrument response
    mask = get_mask(3000, nside, pix_width, ell_grid)
    beam = get_beam(nside, pix_width, ell_grid, lmax_prime)

    #Unlensed temperature field
    field_t = primal_lense_flow(lensed_t, phi, pix_width, n = 10, 
                                direction = INVERSE_LENSE, adjoint = False)
    
    #Unlensed field covariances
    ell_tt, cl_tt = calculate_unbinned_cl(field_t, nside*theta_pix/ARCMIN_PER_DEGREE)
    cf_tt = covar_matrix_from_cls(nside, pix_width, ell_grid, ell_tt,
                                  cl_tt, origin_value = 0)

    # Noise covariance and white noise
    n_tt, _, _, _ = noise_cls(lmax_prime, uk_arcmin_t)
    ell_prime = jnp.arange(2, lmax_prime)
    cn_tt = covar_matrix_from_cls(nside, pix_width, ell_grid, ell_prime, n_tt, origin_value = 0)
    wn_t, kc = field_from_covar(nside, cn_tt, keys, kc)

    #Data = Mask * Beam * Lensed + Noise
    data_t = mask * beam * jfft.rfft2(lensed_t) + jfft.rfft2(wn_t)

    # Convert unlensed fields to Fourier space
    field_t = jfft.rfft2(field_t)
    lensed_t = jfft.rfft2(lensed_t)

    # D matrix
    d_tt = get_d_matrix(cf_tt, cn_tt)

    # Quadratic estimate
    qe = scalar_quadratic_estimate(cn_tt, cf_tt, cfl_tt,
                                   mask, beam, pix_width) / nphi_fac

    # Build dataset for the requested polarization
    return _build_dataset_t(
        nside, theta_pix, pix_width, cphi, qe, phi,
        cf_tt, cn_tt, d_tt, mask, beam,
        field_t, lensed_t, data_t
    )

#NOTE this is practially identical to the regular cmb_lensing.simulate.batch_simulated_trials 
#except for the lack of the polarity key argument
def batch_simulated_trials(num_trials=10, nside=256, theta_pix=2,
                           uk_arcmin_t=10, lmax=17_000):
    def parallel_sim(seed):
        return load_sim(nside=nside, theta_pix=theta_pix,
                        master_seed=seed, uk_arcmin_t=uk_arcmin_t, lmax=lmax)

    seeds = jnp.asarray([np.random.randint(0, 2**31) for _ in range(num_trials)])
    return jax.vmap(parallel_sim)(seeds)

#NOTE this is practially identical to the regular cmb_lensing.simulate.get_avg_cls 
#except for the lack of the polarity key argument
def get_avg_cls(theta_pix, num_trials=100, nside=256,
                uk_arcmin_t=10, lmax=17000, delta_l=50):
    trial_results = batch_simulated_trials(
        num_trials=num_trials, nside=nside, theta_pix=theta_pix,
        uk_arcmin_t=uk_arcmin_t, lmax=lmax
    )

    specs = get_field_specs("I")
    results = {}

    for key, ds_attr, component in specs:
        batched = getattr(getattr(trial_results, ds_attr), component)
        cls_sum = None
        for trial in range(num_trials):
            ell, cls = primal_power_spectra(batched[trial], theta_pix, delta_l=delta_l)
            cls_sum = cls if cls_sum is None else cls_sum + cls
        results[key] = (ell, cls_sum / num_trials)

    return results

if __name__ == "__main__":

    start_time = time.time()
    results = get_avg_cls(theta_pix = 2, num_trials = 100, nside = 256,
                          uk_arcmin_t = 10, lmax = 17000, delta_l = 50)
    end_time = time.time()
    print(f"Total run time = {end_time - start_time}")