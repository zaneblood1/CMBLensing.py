import matplotlib
matplotlib.use("Agg")
from cmb_lensing.simulate import *
from cmb_lensing.simulate import _covar_or_zeros
from cmb_lensing.wiener_filter import *
from cmb_lensing.util import *
from cmb_lensing.map_joint import *
from cmb_lensing.mixing import *
from cmb_lensing.constants import *
from cmb_lensing.precompute_camb_1d import load_camb_spline_predictors, GROUND_TRUTH, PARAM_BOUNDS, PARAM_SIGMA
from cmb_lensing.camb_grid_interp import (load_camb_grid_predictors,
                                          load_camb_grid, GridPredictors)
import os
import random

PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_ORDER)}

USE_CAMB_SPLINE = False
_camb_spline_predictors = {}

def get_camb_spline_predictors(param_name):
    #lazy per-parameter singletons so the same function objects are reused every call -
    #they are passed as static jit args to _recompute_cosmo_matrices, so fresh objects
    #would retrace
    if param_name not in _camb_spline_predictors:
        _camb_spline_predictors[param_name] = load_camb_spline_predictors(param_name)
    return _camb_spline_predictors[param_name]

USE_CAMB_GRID = True
CAMB_GRID_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "cmb_lensing", "camb_splines", "camb_grid_spline.npz")
_camb_grid_predictors = None

if USE_CAMB_GRID and USE_CAMB_SPLINE:
    raise ValueError("USE_CAMB_GRID and USE_CAMB_SPLINE are mutually exclusive Cl sources "
                     "- the 1D caches pin four parameters at ground truth, the 5D grid "
                     "does not")

def get_camb_grid_predictors():
    #single lazy singleton for the whole process: the ~2.3 GB of spline coefficients is
    #read once, and the same two function objects are reused on every call because they
    #are passed as static jit args to _recompute_cosmo_matrices (fresh closures would
    #force a retrace on every Gibbs sweep)
    global _camb_grid_predictors
    if _camb_grid_predictors is None:
        _camb_grid_predictors = load_camb_grid_predictors(CAMB_GRID_PATH)
    return _camb_grid_predictors

def camb_grid_bb_is_zero():
    #whether the merged grid's unlensed BB is the identically-zero r = 0 spectrum. Static
    #per grid file, so the covariance builders can route BB around the log-interpolating
    #path (which would turn exact zeros into NaN) at trace time
    return load_camb_grid(CAMB_GRID_PATH).bb_is_zero

def wrap_temp_predictors(predict_tt, predict_pp):
    """Lift a TT/PP-only predictor pair into the
    GridPredictors namedtuple shape the rest of the sampler passes around. The missing
    spectra are None, which is only valid for pol = 'I' - the 5D
    grid is the only Cl source that serves polarization and lensed spectra."""
    return GridPredictors(tt = predict_tt, ee = None, bb = None, pp = predict_pp,
                          tt_lensed = None, ee_lensed = None, bb_lensed = None,
                          te_rho = None)

#The jitted theta kernels and eval_logpdf_batch take raw arrays, not field/operator
#structs. For pol = 'I' those are the (nside, nside//2 + 1) matrices as always; for
#pol = 'IP' the T/E/B field matrices are stacked into one (3, nside, nside//2 + 1)
#array and the operator blocks into a (4, ...) TT/TE/EE/BB stack; for pol = 'P'
#(FlatS2 fields, DiagonalEB operators) the E/B field matrices form a (2, ...) stack and
#the operator blocks a (2, ...) EE/BB stack. Every kernel signature thus stays a flat
#list of arrays
def field_matrix_stack(field, pol):
    if pol == "IP":
        return jnp.stack([field.scalar_matrix, field.polar_matrix_1,
                          field.polar_matrix_2])
    if pol == "P":
        return jnp.stack([field.polar_matrix_1, field.polar_matrix_2])
    return field.scalar_matrix

def op_matrix_stack(op, pol):
    if pol == "IP":
        return jnp.stack([op.matrix_TT, op.matrix_TE, op.matrix_EE, op.matrix_BB])
    if pol == "P":
        return jnp.stack([op.matrix_EE, op.matrix_BB])
    return op.scalar_matrix

def op_shared_matrix(op, pol):
    #the mask and beam apply the same matrix to every block, so they stay un-stacked
    if pol == "IP":
        return op.matrix_TT
    if pol == "P":
        return op.matrix_EE
    return op.scalar_matrix

#draw (t, e, b) map-space realizations against a BlockTEB covariance: T and E drawn
#CORRELATED to the TE block exactly as load_sim's data draw (E ~ N(0, C_EE), then
#T = (C_TE/C_EE) E + an independent piece with variance C_TT - C_TE^2/C_EE, >= 0 by the
#|rho| <= 1 construction), B against its own covariance (an exactly-zero block - unlensed
#BB at r = 0 - yields an exactly zero field). With a zero TE block this reduces to three
#independent draws, so the same helper serves the field AND the noise simulations
def _field_matrices_from_teb_covar(covar, nside, key):
    key_t, key_e, key_b = jax.random.split(key, 3)
    conditional_tt = jnp.maximum(
        covar.matrix_TT - covar.matrix_TE**2 * reciprocal_matrix(covar.matrix_EE), 0.0)
    t_indep = field_from_covar_single_key(nside, conditional_tt, key_t)
    field_e = field_from_covar_single_key(nside, covar.matrix_EE, key_e)
    field_b = field_from_covar_single_key(nside, covar.matrix_BB, key_b)
    te_ratio = covar.matrix_TE * reciprocal_matrix(covar.matrix_EE)
    field_t = jfft.irfft2(te_ratio * jfft.rfft2(field_e)) + t_indep
    return field_t, field_e, field_b

#pol = "P" counterpart for a DiagonalEB covariance: E and B are independent draws
#(the E/B sector of the TEB draw above with no T to correlate against)
def _field_matrices_from_eb_covar(covar, nside, key):
    key_e, key_b = jax.random.split(key)
    field_e = field_from_covar_single_key(nside, covar.matrix_EE, key_e)
    field_b = field_from_covar_single_key(nside, covar.matrix_BB, key_b)
    return field_e, field_b

#sample the field
@jax.jit
def gibbs_sample_f(field_start, data_field, phi, args, rng_key):

    key_f, key_n = jax.random.split(rng_key)
    teb = isinstance(args["field_covariance"], BlockTEB)
    eb = isinstance(args["field_covariance"], DiagonalEB)

    #Run a new simulation for f ~ N(0, Cf(thetas))
    if teb:
        t, e, b = _field_matrices_from_teb_covar(args["field_covariance"],
                                                 data_field.nside, key_f)
        new_field = field_start.replace(scalar_matrix = jfft.rfft2(t),
                                        polar_matrix_1 = jfft.rfft2(e),
                                        polar_matrix_2 = jfft.rfft2(b))
    elif eb:
        e, b = _field_matrices_from_eb_covar(args["field_covariance"],
                                             data_field.nside, key_f)
        new_field = field_start.replace(polar_matrix_1 = jfft.rfft2(e),
                                        polar_matrix_2 = jfft.rfft2(b))
    else:
        new_field_matrix = field_from_covar_single_key(data_field.nside,
                            args["field_covariance"].scalar_matrix, key_f)
        new_field = field_start.replace(scalar_matrix = jfft.rfft2(new_field_matrix))

    #Run a new simulation for n ~ N(0, Cn)
    if teb:
        t, e, b = _field_matrices_from_teb_covar(args["noise_covariance"],
                                                 data_field.nside, key_n)
        new_noise = field_start.replace(scalar_matrix = jfft.rfft2(t),
                                        polar_matrix_1 = jfft.rfft2(e),
                                        polar_matrix_2 = jfft.rfft2(b))
    elif eb:
        e, b = _field_matrices_from_eb_covar(args["noise_covariance"],
                                             data_field.nside, key_n)
        new_noise = field_start.replace(polar_matrix_1 = jfft.rfft2(e),
                                        polar_matrix_2 = jfft.rfft2(b))
    else:
        new_noise_matrix = field_from_covar_single_key(data_field.nside,
                            args["noise_covariance"].scalar_matrix, key_n)
        new_noise = field_start.replace(scalar_matrix = jfft.rfft2(new_noise_matrix))

    #d = M * B * L(phi) * f + n
    lensed_field = qu2eb(fourier(lense_flow(map(eb2qu(new_field)), map(phi), 
                         n = 10, direction = FORWARD_LENSE, adjoint = False)))
    new_data = args["mask"] * args["beam"] * lensed_field + new_noise
    
    #Call the wiener filter with the field_start initial guess and data difference
    #between new and old simulations as the data term. We also use the current phi 
    #and covariance matrices from our sampling algorithms
    data_diff = data_field - new_data
    delta_field = wiener_filter(field_start, phi, data_diff,
                                args["field_covariance"], args["noise_covariance"],
                                args["mask"], args["beam"])

    #Return the new simulated unlensed field plus the wiener filter contribution
    return new_field + delta_field

#sample the lensing potential phi
@jax.jit
def gibbs_sample_phi(mixed_phi, mixed_temp, data_field, rng_key,
                     args, iter, num_burn_in_always_accept):

    always_accept = (iter < num_burn_in_always_accept)
    mass_matrix = get_mass_matrix(args["phi_covariance"], args["quadratic_estimate"],
                                      args["mixing_g"])

    mixed_phi, delta_h, accept = hmc_step(mixed_phi, always_accept, mixed_phi.nside, mass_matrix,
                                          mixed_temp, data_field, args["noise_covariance"], 
                                          args["phi_covariance"], args["field_covariance"], 
                                          args["mask"], args["beam"], args["mixing_d"], 
                                          args["mixing_g"], rng_key)
    return mixed_phi, delta_h, accept

#The mass matrix used in the HMC steps for sampling the lensing potential
#TODO implement power operator for matrix objects to avoid repeated multiplication
def get_mass_matrix(cphi, nphi, mixing_g):
    return pinv(mixing_g) * pinv(mixing_g) * (pinv(cphi)+ pinv(nphi))

#Hamiltonian Monte Carlo Step for the lensing potential
def hmc_step(x, always_accept, nside, mass_matrix,
             mixed_field, data, noise_covariance, 
             phi_covariance, field_covariance, mask, beam, 
             mixing_d, mixing_g, rng_key):
    
    #generate a random kick in momentum space. the momentum is conjugate to (mixed) PHI,
    #so its template is x - using mixed_field would break for pol = "IP", where the
    #field is a FlatS02 but phi stays a scalar FlatS0
    rng_key_1, rng_key_2 = jax.random.split(rng_key)
    p_matrix = field_from_covar_single_key(nside, mass_matrix.scalar_matrix, rng_key_1)
    p = x.replace(scalar_matrix = jfft.rfft2(p_matrix))

    #perform the integration of hamilton's equations using realization of mass_matrix
    #as initial random momentum and current mixed phi value as the starting "position"
    delta_h, x_test, _ = symplectic_integrate(x, p, mixed_field, data, noise_covariance, 
                                              phi_covariance, field_covariance, mask, beam, 
                                              mixing_d, mixing_g, mass_matrix)
    
    #Always accept if change in Hamiltonian is positive, otherwise accept probabilistically
    accept = jnp.logical_or(always_accept, jnp.log(jax.random.uniform(rng_key_2)) < delta_h)
    def on_accept(x, x_test):
        _ = x
        return x_test
    def on_decline(x, x_test):
        _ = x_test
        return x
    operands = x, x_test
    x = jax.lax.cond(
        accept,
        on_accept,
        on_decline,
        *operands
    )
    return x, delta_h, accept

#------------------ symplectic integration ---------------------------------------
#NOTE num_steps * step_size = path_length must be tuned... Too large and 
#you can overshoot and end up in physically impossible / divergent solutions...
#Too small and you may not have enough momentum to escape local minima
#and converge on the true global minimum
def symplectic_integrate(x0, p0, mixed_field, data, noise_covariance, 
                        phi_covariance, field_covariance, mask, beam, 
                        mixing_d, mixing_g, mass_matrix,
                        num_steps = 10, step_size = 0.05):
    
    #Get the mixed phi gradient at a certain mixed_phi value with all other
    #inputs held constant
    def mixed_grad_phi_partial(mixed_phi):
        return mixed_grad_phi_logpdf(mixed_field, mixed_phi, data, noise_covariance, 
                                     phi_covariance, field_covariance, 
                                     mask, beam, mixing_d, mixing_g)
    
    #Get the logpdf at a certain mixed_phi value with all other inputs held constant
    def logpdf_partial(mixed_phi):
        return mixed_logpdf(mixed_field, mixed_phi, data, noise_covariance, 
                 phi_covariance, field_covariance, mask, beam, 
                 mixing_g, mixing_d)
    
    #The hamiltonian used in Hamiltonian Monte Carlo methods of distribution sampling
    def hamiltonian(x, p):
        return logpdf_partial(x) - dot(p, (pinv(mass_matrix) * p)/2)

    def loop_body(_, state):
          x, p, gradient = state
          prev_gradient = gradient
          x = x - step_size * pinv(mass_matrix) * (p - 0.5 * step_size * gradient)
          gradient = mixed_grad_phi_partial(x)
          p = p - 0.5 * step_size * (gradient + prev_gradient)
          return (x, p, gradient)

    gradient = mixed_grad_phi_partial(x0)
    x, p, gradient = jax.lax.fori_loop(0, num_steps, loop_body, (x0, p0, gradient))
    delta_h = hamiltonian(x, p) - hamiltonian(x0, p0)
    return delta_h, x, p

#symmetric Gaussian random-walk Metropolis-within-Gibbs step for a single cosmological parameter
def metropolis_sample_theta(eval_logpdf_grid, theta_name, theta_old, lo, hi, lcdm_acceptance, log_acceptance,
                            proposal_sigma, rng_key, num_steps = 1):
    if proposal_sigma is None:
        raise ValueError("metropolis sampler requires a proposal_sigma for this parameter")

    theta_current = float(theta_old)
    for _ in range(num_steps):
        rng_key, k_prop, k_acc = jax.random.split(rng_key, 3)
        theta_prop = theta_current + proposal_sigma * float(jax.random.normal(k_prop))

        #reject out-of-bounds proposals BEFORE touching the interpolator
        if not (lo <= theta_prop <= hi):
            continue

        logpdfs = eval_logpdf_grid(jnp.array([theta_current, theta_prop]))
        delta_h = float(logpdfs[1] - logpdfs[0])

        #accept if log(u) < delta_h (same acceptance test as hmc_step). NaN-safe: a non-finite
        #logpdf makes delta_h nan and (x < nan) is False, so the step is rejected
        if float(jnp.log(jax.random.uniform(k_acc))) < delta_h:
            theta_current = theta_prop
            if log_acceptance:
                lcdm_acceptance[theta_name].append(int(True))
                print(f"{theta_name} accept rate = {np.sum(np.array(lcdm_acceptance[theta_name])) / len(lcdm_acceptance[theta_name])}")
        elif log_acceptance:
            lcdm_acceptance[theta_name].append(int(False))
            print(f"{theta_name} accept rate = {np.sum(np.array(lcdm_acceptance[theta_name])) / len(lcdm_acceptance[theta_name])}")

    return np.asarray(theta_current, dtype = np.float64)

def make_eval_logpdf_batch(predictors,
                           mixed_temp_matrix, mixed_phi_matrix, data_matrix,
                           qe_scalar, cn_scalar,
                           mask_matrix, beam_matrix, fourier_weights,
                           nside, pix_width, theta_pix, ell_grid,
                           pol = "I", bb_is_zero = True):

    #reconstruct Flax structs from raw arrays inside JIT for stable pytree tracing
    def _scalar_field(m):
        return FlatS0(scalar_matrix = m, fourier_weights = fourier_weights,
                      nside = nside, theta_pix = theta_pix, pix_width = pix_width,
                      basis = Basis.FOURIER, parametrization = Parametrization.T)
    def _op(m):
        return DiagonalScalar(scalar_matrix = m, fourier_weights = fourier_weights,
                              nside = nside, theta_pix = theta_pix, pix_width = pix_width)
    def _field(stack):
        if pol == "IP":
            return FlatS02(scalar_matrix = stack[0], polar_matrix_1 = stack[1],
                           polar_matrix_2 = stack[2], fourier_weights = fourier_weights,
                           nside = nside, theta_pix = theta_pix, pix_width = pix_width,
                           basis = Basis.FOURIER, parametrization = Parametrization.EB)
        if pol == "P":
            return FlatS2(polar_matrix_1 = stack[0], polar_matrix_2 = stack[1],
                          fourier_weights = fourier_weights,
                          nside = nside, theta_pix = theta_pix, pix_width = pix_width,
                          basis = Basis.FOURIER, parametrization = Parametrization.EB)
        return _scalar_field(stack)
    
    def _op_teb(tt, te, ee, bb):
        return BlockTEB(matrix_TT = tt, matrix_TE = te, matrix_ET = te,
                        matrix_EE = ee, matrix_BB = bb,
                        fourier_weights = fourier_weights, nside = nside,
                        theta_pix = theta_pix, pix_width = pix_width)
    def _op_eb(ee, bb):
        return DiagonalEB(matrix_EE = ee, matrix_BB = bb,
                          fourier_weights = fourier_weights, nside = nside,
                          theta_pix = theta_pix, pix_width = pix_width)
    def _block_op(stack):
        if pol == "IP":
            return _op_teb(stack[0], stack[1], stack[2], stack[3])
        if pol == "P":
            return _op_eb(stack[0], stack[1])
        return _op(stack)

    mixed_temp = _field(mixed_temp_matrix)
    mixed_phi = _scalar_field(mixed_phi_matrix)
    data_field = _field(data_matrix)
    noise_covariance = _block_op(cn_scalar)

    if pol == "IP":
        zero_block = jnp.zeros_like(mask_matrix)
        mask = _op_teb(mask_matrix, zero_block, mask_matrix, mask_matrix)
        beam = _op_teb(beam_matrix, zero_block, beam_matrix, beam_matrix)
    elif pol == "P":
        mask = _op_eb(mask_matrix, mask_matrix)
        beam = _op_eb(beam_matrix, beam_matrix)
    else:
        mask = _op(mask_matrix)
        beam = _op(beam_matrix)

    def eval_logpdf_batch(params_batch):
        M = params_batch.shape[0]

        cl_pp_batch = predictors.pp(params_batch)
        if pol != "P":
            cl_tt_batch = predictors.tt(params_batch)
        if pol in ("IP", "P"):
            cl_ee_batch = predictors.ee(params_batch)
            if not bb_is_zero:
                cl_bb_batch = predictors.bb(params_batch)
        if pol == "IP":
            te_rho_batch = predictors.te_rho(params_batch)

        def single_logpdf(i):

            cl_pp = cl_pp_batch[i]

            #POLARIZATION ONLY BRANCH
            if pol == "P":
                cl_ee = cl_ee_batch[i]
                #ells sized to the predictor output (see the T branches below)
                ells = jnp.arange(2, 2 + cl_ee.shape[-1])
                cphi = covar_matrix_from_cls(nside, pix_width,
                                             ell_grid, ells,
                                             cl_pp, origin_value = 0)
                cn_ee, cn_bb = cn_scalar[0], cn_scalar[1]
                cf_ee = covar_matrix_from_cls(nside, pix_width,
                                              ell_grid, ells,
                                              cl_ee, origin_value = 0)
                if bb_is_zero:
                    #an out-of-box row is already NaN through cf_ee, so the exact-zero
                    #BB block never weakens the rejected-proposal semantics
                    cf_bb = jnp.zeros_like(cf_ee)
                else:
                    cf_bb = covar_matrix_from_cls(nside, pix_width,
                                                  ell_grid, ells,
                                                  cl_bb_batch[i], origin_value = 0)

                g = get_g_matrix_lcdm(cphi, qe_scalar)
                d_ee, d_bb = get_d_eb_matrix(cf_ee, cf_bb, cn_ee, cn_bb)

                return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                                    noise_covariance, _op(cphi),
                                    _op_eb(cf_ee, cf_bb),
                                    mask, beam, _op(g), _op_eb(d_ee, d_bb))

            cl_tt = cl_tt_batch[i]
            #ells sized to the predictor output so the extrapolation anchor matches the data map's
            ells = jnp.arange(2, 2 + cl_tt.shape[-1])

            cf_tt = covar_matrix_from_cls(nside, pix_width,
                                          ell_grid, ells,
                                          cl_tt, origin_value = 0)
            cphi = covar_matrix_from_cls(nside, pix_width,
                                         ell_grid, ells,
                                         cl_pp, origin_value = 0)

            #POLARIZATION AND TEMPERATURE BRANCH
            if pol == "IP":
                cn_tt, cn_te = cn_scalar[0], cn_scalar[1]
                cn_ee, cn_bb = cn_scalar[2], cn_scalar[3]
                cf_ee = covar_matrix_from_cls(nside, pix_width,
                                              ell_grid, ells,
                                              cl_ee_batch[i], origin_value = 0)
                cf_te = te_covar_from_rho(te_rho_batch[i], ells, ell_grid,
                                          cf_tt, cf_ee)
                if bb_is_zero:
                    cf_bb = jnp.zeros_like(cf_tt)
                else:
                    cf_bb = covar_matrix_from_cls(nside, pix_width,
                                                  ell_grid, ells,
                                                  cl_bb_batch[i], origin_value = 0)

                g = get_g_matrix_lcdm(cphi, qe_scalar)
                d_tt, d_te, d_ee, d_bb = get_d_teb_matrix(cf_tt, cf_te, cf_ee,
                                                              cf_bb, cn_tt, cn_te,
                                                              cn_ee, cn_bb)
               
                return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                                    noise_covariance, _op(cphi),
                                    _op_teb(cf_tt, cf_te, cf_ee, cf_bb),
                                    mask, beam, _op(g),
                                    _op_teb(d_tt, d_te, d_ee, d_bb))
            
            #TEMPERATURE ONLY DEFAULT
            g = get_g_matrix_lcdm(cphi, qe_scalar)
            d = get_d_tt_matrix(cf_tt, cn_scalar)
            return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                                noise_covariance, _op(cphi), _op(cf_tt),
                                mask, beam, _op(g), _op(d))

        #Evaluate the logpdf at M different points in parameter space
        logpdfs = jax.vmap(single_logpdf)(jnp.arange(M))
        return logpdfs

    return eval_logpdf_batch

def gibbs_sample_theta(theta_key_idx, theta_range, theta_old, lcdm_acceptance, log_acceptance,
                       mixed_temp_matrix, mixed_phi_matrix, data_matrix,
                       current_params, rng_key, nside, pix_width, theta_pix,
                       ell_grid, qe_scalar, cn_scalar, mask_matrix, beam_matrix,
                       fourier_weights, proposal_sigma = None, metropolis_num_steps = 1,
                       pol = "I", bb_is_zero = True):

    #Check whether we are doing 1-D parameter inference or N-D parameter inference
    theta_name = PARAM_ORDER[int(theta_key_idx)]
    if USE_CAMB_GRID:
        predictors = get_camb_grid_predictors()
    else:
        predictors = wrap_temp_predictors(*get_camb_spline_predictors(theta_name))

    #evaluate the mixed logpdf over an arbitrary (M, 5) batch of parameter vectors
    eval_logpdf_batch = make_eval_logpdf_batch(predictors,
                                               mixed_temp_matrix, mixed_phi_matrix,
                                               data_matrix, 
                                               qe_scalar, cn_scalar, mask_matrix,
                                               beam_matrix, fourier_weights,
                                               nside, pix_width, theta_pix, ell_grid,
                                               pol = pol, bb_is_zero = bb_is_zero)

    #evaluate along one coordinate axis: tile current_params and overwrite the sampled column
    def eval_logpdf_grid(theta_grid):
        M = theta_grid.shape[0]
        params_batch = jnp.tile(current_params, (M, 1))
        params_batch = params_batch.at[:, theta_key_idx].set(theta_grid)
        return eval_logpdf_batch(params_batch)

    lo, hi = float(theta_range[0]), float(theta_range[-1])
    return metropolis_sample_theta(eval_logpdf_grid, theta_name, theta_old, lo, hi, lcdm_acceptance, log_acceptance,
                                   proposal_sigma, rng_key, num_steps = metropolis_num_steps)

#jitted core of the per-iteration covariance/mixing recompute
@partial(jax.jit, static_argnames = ["predictors", "nside", "pix_width", "refresh_qe"])
def _recompute_cosmo_matrices(current_params, ell_grid,
                              qe_scalar, cn_scalar,
                              mask_matrix, beam_matrix,
                              predictors, nside, pix_width,
                              refresh_qe = False):
    x = current_params[None, :]
    cl_tt = predictors.tt(x)[0]
    cl_pp = predictors.pp(x)[0]
    ells = jnp.arange(2, 2 + cl_tt.shape[-1])

    cf = covar_matrix_from_cls(nside, pix_width,
                               ell_grid, ells,
                               cl_tt, origin_value = 0)
    cphi = covar_matrix_from_cls(nside, pix_width,
                                 ell_grid, ells,
                                 cl_pp, origin_value = 0)

    if refresh_qe:
        cl_tt_lensed = predictors.tt_lensed(x)[0]
        cfl = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                    cl_tt_lensed, origin_value = 0)
        qe = scalar_quadratic_estimate(cn_scalar, cf, cfl,
                                       mask_matrix, beam_matrix, pix_width) / NPHI_FAC
    else:
        qe = qe_scalar

    g = get_g_matrix_lcdm(cphi, qe)
    d = get_d_tt_matrix(cf, cn_scalar)
    return g, d, cf, cphi, qe


@partial(jax.jit, static_argnames = ["predictors", "nside", "pix_width", "refresh_qe", "bb_is_zero"])
def _recompute_cosmo_matrices_teb(current_params, ell_grid,
                                  qe_scalar, cn_stack,
                                  mask_matrix, beam_matrix,
                                  predictors, nside, pix_width,
                                  refresh_qe = False, bb_is_zero = True):
    x = current_params[None, :]
    cl_tt = predictors.tt(x)[0]
    cl_ee = predictors.ee(x)[0]
    te_rho = predictors.te_rho(x)[0]
    cl_pp = predictors.pp(x)[0]
    ells = jnp.arange(2, 2 + cl_tt.shape[-1])
    cn_tt, cn_te, cn_ee, cn_bb = cn_stack[0], cn_stack[1], cn_stack[2], cn_stack[3]

    cf_tt = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                  cl_tt, origin_value = 0)
    cf_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                  cl_ee, origin_value = 0)
    cf_te = te_covar_from_rho(te_rho, ells, ell_grid, cf_tt, cf_ee)
    if bb_is_zero:
        cf_bb = jnp.zeros_like(cf_tt)
    else:
        cl_bb = predictors.bb(x)[0]
        cf_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                      cl_bb, origin_value = 0)
    cphi = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                 cl_pp, origin_value = 0)

    if refresh_qe:
        cl_ee_lensed = predictors.ee_lensed(x)[0]
        cl_bb_lensed = predictors.bb_lensed(x)[0]
        cfl_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                       cl_ee_lensed, origin_value = 0)
        cfl_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                       cl_bb_lensed, origin_value = 0)
        qe = polar_quadratic_estimate(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb,
                                      mask_matrix, mask_matrix, beam_matrix, beam_matrix,
                                      pix_width) / NPHI_FAC
    else:
        qe = qe_scalar

    g = get_g_matrix_lcdm(cphi, qe)
    d_tt, d_te, d_ee, d_bb = get_d_teb_matrix(cf_tt, cf_te, cf_ee, cf_bb,
                                              cn_tt, cn_te, cn_ee, cn_bb)
    return (g, jnp.stack([d_tt, d_te, d_ee, d_bb]),
            jnp.stack([cf_tt, cf_te, cf_ee, cf_bb]), cphi, qe)

@partial(jax.jit, static_argnames = ["predictors", "nside", "pix_width", "refresh_qe", "bb_is_zero"])
def _recompute_cosmo_matrices_eb(current_params, ell_grid, 
                                 qe_scalar, cn_stack,
                                 mask_matrix, beam_matrix,
                                 predictors, nside, pix_width,
                                 refresh_qe = False, bb_is_zero = True):
    x = current_params[None, :]
    cl_ee = predictors.ee(x)[0]
    cl_pp = predictors.pp(x)[0]
    ells = jnp.arange(2, 2 + cl_ee.shape[-1])
    cn_ee, cn_bb = cn_stack[0], cn_stack[1]

    cf_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                  cl_ee, origin_value = 0)
    if bb_is_zero:
        cf_bb = jnp.zeros_like(cf_ee)
    else:
        cl_bb = predictors.bb(x)[0]
        cf_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                      cl_bb, origin_value = 0)
    cphi = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                 cl_pp, origin_value = 0)

    if refresh_qe:
        cl_ee_lensed = predictors.ee_lensed(x)[0]
        cl_bb_lensed = predictors.bb_lensed(x)[0]
        cfl_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                       cl_ee_lensed, origin_value = 0)
        cfl_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                       cl_bb_lensed, origin_value = 0)
        qe = polar_quadratic_estimate(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb,
                                      mask_matrix, mask_matrix, beam_matrix, beam_matrix,
                                      pix_width) / NPHI_FAC
    else:
        qe = qe_scalar

    g = get_g_matrix_lcdm(cphi, qe)
    d_ee, d_bb = get_d_eb_matrix(cf_ee, cf_bb, cn_ee, cn_bb)
    return g, jnp.stack([d_ee, d_bb]), jnp.stack([cf_ee, cf_bb]), cphi, qe

def get_new_cosmo_matrices(current_params, predictors, args,
                           pol = "I", refresh_qe = False):
 
    if pol == "IP":
        return _recompute_cosmo_matrices_teb(
            current_params, args["ell_grid"],
            args["quadratic_estimate"].scalar_matrix,
            op_matrix_stack(args["noise_covariance"], pol),
            op_shared_matrix(args["mask"], pol), op_shared_matrix(args["beam"], pol),
            predictors, args["nside"], args["pix_width"],
            refresh_qe = refresh_qe, bb_is_zero = camb_grid_bb_is_zero())
    
    if pol == "P":
        return _recompute_cosmo_matrices_eb(
            current_params, args["ell_grid"],
            args["quadratic_estimate"].scalar_matrix,
            op_matrix_stack(args["noise_covariance"], pol),
            op_shared_matrix(args["mask"], pol), op_shared_matrix(args["beam"], pol),
            predictors, args["nside"], args["pix_width"],
            refresh_qe = refresh_qe, bb_is_zero = camb_grid_bb_is_zero())
    
    return _recompute_cosmo_matrices(current_params,
                                     args["ell_grid"], 
                                     args["quadratic_estimate"].scalar_matrix,
                                     args["noise_covariance"].scalar_matrix,
                                     op_shared_matrix(args["mask"], pol),
                                     op_shared_matrix(args["beam"], pol),
                                     predictors, args["nside"], args["pix_width"],
                                     refresh_qe = refresh_qe)

#Lighter-weight version of the above method to just compute the field covariance(s) and
#not D, G, Cphi... Eager-only (uses the _covar_or_zeros value check), which is fine for
#its single call site in add_starting_matrices_to_args
def get_new_cf_matrix(current_params, predictors, args, pol = "I"):
    """Compute the field covariance from a parameter vector. pol = "I" returns the TT
    matrix; pol = "IP" returns the (cf_tt, cf_te, cf_ee, cf_bb) tuple; pol = "P" the
    (cf_ee, cf_bb) pair."""

    x = current_params[None, :]

    if pol == "P":
        cl_ee = predictors.ee(x)[0]
        cl_bb = predictors.bb(x)[0]
        ells = jnp.arange(2, 2 + cl_ee.shape[-1])
        cf_ee = covar_matrix_from_cls(args["nside"], args["pix_width"],
                                      args["ell_grid"], ells,
                                      cl_ee, origin_value = 0)
        cf_bb = _covar_or_zeros(args["nside"], args["pix_width"],
                                args["ell_grid"], ells,
                                cl_bb, origin_value = 0)
        return cf_ee, cf_bb
    
    cl_tt = predictors.tt(x)[0]
    #ells sized to the predictor output
    ells = jnp.arange(2, 2 + cl_tt.shape[-1])
    cf_tt = covar_matrix_from_cls(args["nside"], args["pix_width"],
                                  args["ell_grid"], ells,
                                  cl_tt, origin_value = 0)
    if pol == "IP":
        cl_ee = predictors.ee(x)[0]
        cl_bb = predictors.bb(x)[0]
        te_rho = predictors.te_rho(x)[0]
        cf_ee = covar_matrix_from_cls(args["nside"], args["pix_width"],
                                      args["ell_grid"], ells,
                                      cl_ee, origin_value = 0)
        cf_te = te_covar_from_rho(te_rho, ells, args["ell_grid"], cf_tt, cf_ee)
        cf_bb = _covar_or_zeros(args["nside"], args["pix_width"],
                                args["ell_grid"], ells,
                                cl_bb, origin_value = 0)
        return cf_tt, cf_te, cf_ee, cf_bb
    
    return cf_tt

def add_metadata_to_args(args, data_set, lmax):
    #This set of data is generally needed for computing covariance
    #matrices from a set of Cls. The lmax variable determines how far out
    #in the multipole range we will interpolate the Cls which only go out
    #to lmax_prime in multipole
    args["lmax"] = lmax
    args["lmax_prime"] = min(lmax, DEFAULT_MAX_ELL)
    args["nside"] = data_set.nside
    args["pix_width"] = data_set.pix_width
    ell_grid, _ = gen_ell_grid(data_set.nside, data_set.theta_pix)
    args["ell_grid"] = ell_grid
    args["ells"] = jnp.arange(2, lmax + 1).astype(jnp.float64)
    return args

#Convert dictionaries between emulator and load_sim() naming convetion
def to_camb_naming_conv(param_init):
    param_init["cosmomc_theta"] = param_init["theta_MC_100"] / 100
    del param_init["theta_MC_100"]
    param_init["As"] = jnp.exp(param_init["logA"]) * 1e-10
    del param_init["logA"]
    return param_init

#replace the four independent blocks of a symmetric BlockTEB operator (ET mirrors TE)
def _replace_teb_blocks(op, tt, te, ee, bb):
    return op.replace(matrix_TT = tt, matrix_TE = te, matrix_ET = te,
                      matrix_EE = ee, matrix_BB = bb)

#replace the two diagonal blocks of a DiagonalEB operator (pol = "P")
def _replace_eb_blocks(op, ee, bb):
    return op.replace(matrix_EE = ee, matrix_BB = bb)

def add_starting_matrices_to_args(args, data_set, noise_level, param_init, current_params,
                                  predictors, pol = "I"):

    #We can comfortably reuse the noise covariance, mask, and beam
    #from the ground truth data set for our sampling algorithm
    args["noise_covariance"] = data_set.noise_covariance
    args["mask"] = data_set.mask
    args["beam"] = data_set.beam
    
    #We also need to store the fiducial phi covariance matrix which contains the
    #ground truth information needed for the G mixing matrix
    args["cphi_fid"] = data_set.phi_covariance.scalar_matrix

    #Note the QE depends on Cfl which will be affected by the ground truth
    #cosmological parameters therefore using the ground truth QE norm is somewhat
    #cheating since we are using extra information besides just the data map...
    #We should therefore initialize the QE norm to be computed based on our initial parameter guesses
    param_init = to_camb_naming_conv(param_init)
    #the lensed covariances feeding the QE norm (and through it the G matrix) must come
    #from the same Cl model as the rest of the pipeline 
    initial_cond = load_sim(data_set.nside, data_set.theta_pix, pol,
                            np.random.randint(0, 2**31), **param_init,
                            uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = 0)
    cf = get_new_cf_matrix(current_params, predictors, args, pol = pol)

    if pol == "IP":
        cf_tt, cf_te, cf_ee, cf_bb = cf
        args["field_covariance"] = _replace_teb_blocks(data_set.field_covariance,
                                                       cf_tt, cf_te, cf_ee, cf_bb)
        qe_matrix = polar_quadratic_estimate(
            cf_ee, cf_bb,
            initial_cond.lensed_field_covariance.matrix_EE,
            initial_cond.lensed_field_covariance.matrix_BB,
            args["noise_covariance"].matrix_EE,
            args["noise_covariance"].matrix_BB,
            op_shared_matrix(args["mask"], pol), op_shared_matrix(args["mask"], pol),
            op_shared_matrix(args["beam"], pol), op_shared_matrix(args["beam"], pol),
            data_set.pix_width) / NPHI_FAC
    elif pol == "P":
        cf_ee, cf_bb = cf
        args["field_covariance"] = _replace_eb_blocks(data_set.field_covariance,
                                                      cf_ee, cf_bb)
        qe_matrix = polar_quadratic_estimate(
            cf_ee, cf_bb,
            initial_cond.lensed_field_covariance.matrix_EE,
            initial_cond.lensed_field_covariance.matrix_BB,
            args["noise_covariance"].matrix_EE,
            args["noise_covariance"].matrix_BB,
            op_shared_matrix(args["mask"], pol), op_shared_matrix(args["mask"], pol),
            op_shared_matrix(args["beam"], pol), op_shared_matrix(args["beam"], pol),
            data_set.pix_width) / NPHI_FAC
    else:
        args["field_covariance"] = data_set.field_covariance.replace(scalar_matrix = cf)
        qe_matrix = scalar_quadratic_estimate(args["noise_covariance"].scalar_matrix,
                                              args["field_covariance"].scalar_matrix,
                                              initial_cond.lensed_field_covariance.scalar_matrix,
                                              args["mask"].scalar_matrix,
                                              args["beam"].scalar_matrix,
                                              data_set.pix_width) / NPHI_FAC
    args["quadratic_estimate"] = data_set.quadratic_estimate.replace(scalar_matrix = qe_matrix)

    #We must also initialize the mixing D & G matrices and Cphi
    #to the proper values according to our starting guesses for the cosmological parameters
    g, d, _, cphi, _ = get_new_cosmo_matrices(current_params, predictors,
                                              args, pol = pol)
    args["phi_covariance"] = data_set.phi_covariance.replace(scalar_matrix = cphi)
    args["mixing_g"] = data_set.mixing_g.replace(scalar_matrix = g)
    if pol == "IP":
        args["mixing_d"] = _replace_teb_blocks(data_set.mixing_d,
                                               d[0], d[1], d[2], d[3])
    elif pol == "P":
        args["mixing_d"] = _replace_eb_blocks(data_set.mixing_d, d[0], d[1])
    else:
        args["mixing_d"] = data_set.mixing_d.replace(scalar_matrix = d)

    return args

#The data set that was passed to sample_joint uses the ground truth covariance matrices
#so we must change them to the data that corresponds to our initial position in
#cosmological parameter space. The args operators were built by replace()-ing the
#data_set's own operators, so they already carry the right operator types for either
#polarization mode
def set_initial_ds_conditions(data_set, args):
    data_set = data_set.replace(
            phi_covariance = args["phi_covariance"],
            field_covariance = args["field_covariance"],
            mixing_d = args["mixing_d"],
            quadratic_estimate = args["quadratic_estimate"]
        )
    return data_set

def update_args_after_sample(current_params, predictors, args,
                             pol = "I", refresh_qe = False):
    g, d, cf, cphi, qe = get_new_cosmo_matrices(current_params, predictors,
                                                args, pol = pol,
                                                refresh_qe = refresh_qe)
    args["phi_covariance"] = args["phi_covariance"].replace(scalar_matrix = cphi)
    if pol == "IP":
        args["field_covariance"] = _replace_teb_blocks(args["field_covariance"],
                                                       cf[0], cf[1], cf[2], cf[3])
        args["mixing_d"] = _replace_teb_blocks(args["mixing_d"],
                                                d[0], d[1], d[2], d[3])
    elif pol == "P":
        args["field_covariance"] = _replace_eb_blocks(args["field_covariance"],
                                                      cf[0], cf[1])
        args["mixing_d"] = _replace_eb_blocks(args["mixing_d"], d[0], d[1])
    else:
        args["field_covariance"] = args["field_covariance"].replace(scalar_matrix = cf)
        args["mixing_d"] = args["mixing_d"].replace(scalar_matrix = (d))
    args["mixing_g"] = args["mixing_g"].replace(scalar_matrix = (g))
    #with refresh_qe the returned qe is the freshly interpolated-lensed-Cl norm at the
    #new theta, so the phi mass matrix and G track theta; without it qe passes through
    args["quadratic_estimate"] = args["quadratic_estimate"].replace(scalar_matrix = qe)
    return args

#algorithm to jointly sample cosmological parameters
def sample_joint(data_set, param_init, proposal_sigmas, param_ranges, should_sample, noise_level, 
                 advanced_logging, fixed_fields = False, phi_init = "MAP",
                 iters_per_chain = 10_000, num_burn_in_fix_theta = 100, 
                 num_burn_in_always_accept = 0, seed = None, map_idx = 1, sub_chain_idx = 1,  
                 lmax = DEFAULT_MAX_ELL, metropolis_num_steps = 1, hpc_path = None):

    #polarization mode follows the dataset 
    if isinstance(data_set, DataSetTEB):
        pol = "IP"
    elif isinstance(data_set, DataSetEB):
        pol = "P"
    elif isinstance(data_set, DataSetT):
        pol = "I"
    else:
        raise ValueError(f"sample_joint supports DataSetT (pol = 'I'), DataSetEB "
                         f"(pol = 'P') and DataSetTEB (pol = 'IP') datasets, got "
                         f"{type(data_set).__name__}")
    if pol != "I" and not USE_CAMB_GRID:
        raise ValueError(f"pol = '{pol}' requires USE_CAMB_GRID - the "
                         "1D CAMB caches have no EE/BB spectra")
    bb_is_zero = camb_grid_bb_is_zero() if USE_CAMB_GRID else True

    #Determine whether to use a 1-D interpolator or a 5-D interpolator
    if USE_CAMB_GRID:
        predictors = get_camb_grid_predictors()
        start_tt = predictors.tt(jnp.array([[param_init[k] for k in PARAM_ORDER]]))
        if not bool(jnp.all(jnp.isfinite(start_tt))):
            raise ValueError(
                f"the starting parameters {dict((k, param_init[k]) for k in PARAM_ORDER)} "
                f"fall outside the 5D CAMB grid, so the initial covariances would be NaN. "
                f"Check that theta_MC_100 is reachable at this ombh2/omch2 - see "
                f"theta_grid in {CAMB_GRID_PATH}")
    elif USE_CAMB_SPLINE:
        sampled_names = [name for name in PARAM_ORDER if should_sample.get(name, False)]
        if len(sampled_names) != 1:
            raise ValueError(f"USE_CAMB_SPLINE requires exactly one sampled parameter "
                             f"(the caches are 1D), got {sampled_names}")
        predictors = wrap_temp_predictors(*get_camb_spline_predictors(sampled_names[0]))
    else:
        raise ValueError("One of either USE_CAMB_GRID or USE_CAMB_SPLINE must be set to True")

    #Set the initial parameter values to the user specified starting guesses
    param_vals = {}
    for theta, theta_val in param_init.items():
        param_vals[theta] = [theta_val]
    #Also store the current parameter values in a JAX array
    current_params = jnp.array([param_init[k] for k in PARAM_ORDER], dtype = jnp.float64)

    #Add data that will be used throughout the sampling algorithm to an args dictionary
    args = {}
    args = add_metadata_to_args(args, data_set, lmax)
    args = add_starting_matrices_to_args(args, data_set, noise_level,
                                         param_init, current_params,
                                         predictors, pol = pol)

    #change data_set covariance matrices from ground truth to initial
    #starting point in cosmological parameter space
    data_set = set_initial_ds_conditions(data_set, args)
    data_field = data_set.data

    #Use a seed to get reproduceable results if so desired
    #otherwise use machine entropy to generate a random number
    if seed is not None:
        sub_key = jax.random.PRNGKey(seed)
    else:
        sub_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    #Setting the starting point in phi-space to be the MAP estimate (the mode) plus a random realization
    #of the Hessian pre-conditioner empirically seems to decrease the burn-in time (DEFAULT)
    if phi_init == "MAP":
        _, phi_map = map_joint(data_set)
        inv_mass_matrix = pinv(pinv(args["phi_covariance"]) + pinv(args["quadratic_estimate"]))
        rng_key, sub_key = jax.random.split(sub_key)
        phi_rng_matrix = field_from_covar_single_key(data_set.data.nside, inv_mass_matrix.scalar_matrix, rng_key)
        phi_rng = phi_map.replace(scalar_matrix = jfft.rfft2(phi_rng_matrix))
        phi = phi_rng + phi_map     
    else:
        phi = 0*data_set.phi

    #We repeatedly draw a fresh temperature / polarization field, so we need a "zeroes" object
    #of the same class as the temperature / polarization field
    field_zeroes = 0*data_set.unlensed_field

    #the stacked data matrices handed to every jitted theta kernel
    data_stack = field_matrix_stack(data_field, pol)

    #lists used to track sampling acceptance rates
    phi_acceptance = []
    mixed_logpdf_values = []
    lcdm_acceptance = {}
    lcdm_acceptance["theta_MC_100"] = []
    lcdm_acceptance["ombh2"] = []
    lcdm_acceptance["omch2"] = []
    lcdm_acceptance["ns"] = []
    lcdm_acceptance["logA"] = []

    for iter in range(1, iters_per_chain + 1):

        #The "fixed_fields" flag can be used for quick debugging of the theta step 
        #assuming we have perfect knowledge of the ground truth (f, phi) pair
        if not fixed_fields:
            #1. sample the temperature field
            rng_key, sub_key = jax.random.split(sub_key)
            temp_field = gibbs_sample_f(field_zeroes, data_field, phi, args, rng_key)

            #2. mix the fields
            mixed_temp, mixed_phi = mix(temp_field, phi, args["mixing_d"], args["mixing_g"])

            #3. sample the lensing potential phi
            rng_key, sub_key = jax.random.split(sub_key)
            mixed_phi, delta_h, accept = gibbs_sample_phi(mixed_phi, mixed_temp, data_field, rng_key,
                                                    args, iter, num_burn_in_always_accept)
            if advanced_logging["phi_acceptance"]:
                phi_acceptance.append(int(accept))
                print(f"Phi accept rate = {np.sum(np.array(phi_acceptance)) / len(phi_acceptance)}")
                print(f"delta_H = {delta_h}")
        else:
            mixed_temp, mixed_phi = mix(data_set.unlensed_field, data_set.phi, args["mixing_d"], args["mixing_g"])

        #4. sample your cosmo parameters
        if iter >= num_burn_in_fix_theta:
            for theta, proposal_sigma in list(proposal_sigmas.items()):
                if should_sample[theta]:
                    rng_key, sub_key = jax.random.split(sub_key)
                    theta_key_idx = PARAM_INDEX[theta]
                    theta_val = gibbs_sample_theta(jnp.array(theta_key_idx), param_ranges[theta], 
                                                   jnp.array(param_vals[theta][-1]),
                                                   lcdm_acceptance, advanced_logging["lcdm_acceptance"],
                                                   field_matrix_stack(mixed_temp, pol),
                                                   mixed_phi.scalar_matrix, data_stack,
                                                   current_params, rng_key, 
                                                   args["nside"], args["pix_width"], data_field.theta_pix,
                                                   args["ell_grid"],
                                                   args["quadratic_estimate"].scalar_matrix,
                                                   op_matrix_stack(args["noise_covariance"], pol),
                                                   op_shared_matrix(args["mask"], pol), 
                                                   op_shared_matrix(args["beam"], pol),
                                                   data_field.fourier_weights,
                                                   proposal_sigma = proposal_sigma,
                                                   metropolis_num_steps = metropolis_num_steps,
                                                   pol = pol, bb_is_zero = bb_is_zero)
                    param_vals[theta].append(theta_val)
                    current_params = current_params.at[theta_key_idx].set(theta_val)

            #5. recompute mixing and covariance matrices using the newly sampled parameter values
            args = update_args_after_sample(current_params, predictors,
                                            args, pol = pol)

        #6. unmix the fields using the updated version of the G & D matrices
        if not fixed_fields:
            _, phi = unmix(mixed_temp, mixed_phi, args["mixing_d"], args["mixing_g"])

        #7. plot certain diagnostics if user specified
        if advanced_logging["plot_log_pdf"]:
            mixed_logpdf_value = mixed_logpdf(mixed_temp, mixed_phi, data_field, args["noise_covariance"], 
                                              args["phi_covariance"], args["field_covariance"], 
                                              args["mask"], args["beam"], 
                                              args["mixing_g"], args["mixing_d"])
            mixed_logpdf_values.append(mixed_logpdf_value)
            plt.figure(figsize = (16, 10))
            plt.plot(mixed_logpdf_values, marker = "o")
            plt.title("Mixed LogPDF Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("Mixed LogPDF Value")
            plt.legend()
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "cmb_lensing", "sample_lcdm_output", "logpdf_progress.png")
            plt.savefig(path)
            plt.close() 

        if advanced_logging["plot_lcdm_sigmas"]:
            plt.figure(figsize = (16, 10))
            for name in PARAM_ORDER:
                if should_sample.get(name, False):
                    normalized = (np.array(param_vals[name]) - GROUND_TRUTH[name]) / PARAM_SIGMA[name]
                    plt.plot(normalized, label = name, marker = "o")
            plt.axhline(0, color = "black", label = "Zero Sigma")
            plt.axhline(1, color = "grey", label = "+/- 1 Sigma")
            plt.axhline(-1, color = "grey")
            plt.title("(mean - sample)/sigma")
            plt.xlabel("Iteration")
            plt.ylabel("Standard Deviations")
            plt.legend()
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "cmb_lensing", "sample_lcdm_output", "lcdm_progress.png")
            plt.savefig(path)
            plt.close()
        
        #Log chains to .txt files if we are running a larger experiment on an HPC
        if hpc_path is not None:
            for theta, _ in list(param_vals.items()):
                if should_sample[theta]:
                    theta_path = hpc_path + f"{theta}_map_{map_idx}_chain_{sub_chain_idx}_history.txt"
                    with open(theta_path, "a") as file:
                        file.write(str(param_vals[theta][-1]) + "\n")

    return param_vals

if __name__ == "__main__":

    #ground-truth (fiducial) parameter values
    ground_truth_params = {}
    ground_truth_params["ombh2"] = GROUND_TRUTH["ombh2"]
    ground_truth_params["omch2"] = GROUND_TRUTH["omch2"]
    ground_truth_params["cosmomc_theta"] = GROUND_TRUTH["theta_MC_100"] / 100
    ground_truth_params["As"] = jnp.exp(GROUND_TRUTH["logA"]) * 1e-10
    ground_truth_params["ns"] = GROUND_TRUTH["ns"]

    #Generate a "ground truth" simulated data set
    nside = 256
    theta_pix = 2.5
    pol = "I"
    master_seed = 469134
    noise_level = 1
    data_set = load_sim(nside, theta_pix, pol, master_seed, **ground_truth_params,
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = 0)

    #Starting points in parameter space
    param_init = {}
    param_init["ombh2"] = PARAM_BOUNDS["ombh2"][-1]
    param_init["omch2"] =  GROUND_TRUTH["omch2"]
    param_init["theta_MC_100"] =  GROUND_TRUTH["theta_MC_100"]
    param_init["logA"] =  GROUND_TRUTH["logA"]
    param_init["ns"] = PARAM_BOUNDS["ns"][0]

    #Whether or not to sample each parameter
    should_sample = {}
    should_sample["ombh2"] = True
    should_sample["omch2"] = False
    should_sample["theta_MC_100"] = False
    should_sample["logA"] = False
    should_sample["ns"] = True

    #Width of the proposed Gaussian distribution used in the Metropolis
    #step for sampling the LCDM parameters. These should be tuned to around
    #a 44 - 50% acceptance rate
    proposal_sigmas = {}
    proposal_sigmas["ombh2"] = 1e-4
    proposal_sigmas["omch2"] = 8e-4
    proposal_sigmas["theta_MC_100"] = 3e-3
    proposal_sigmas["logA"] = 2e-2
    proposal_sigmas["ns"] = 5e-3

    #allowed search ranges for each of the LCDM parameters
    param_ranges = {}
    param_ranges["ombh2"] = PARAM_BOUNDS["ombh2"]
    param_ranges["omch2"] = PARAM_BOUNDS["omch2"]
    param_ranges["theta_MC_100"] = PARAM_BOUNDS["theta_MC_100"]
    param_ranges["logA"] = PARAM_BOUNDS["logA"]
    param_ranges["ns"] = PARAM_BOUNDS["ns"]

    #Whether or not to log certain diagnostic statistics
    advanced_logging = {}
    advanced_logging["phi_acceptance"] = True
    advanced_logging["lcdm_acceptance"] = True
    advanced_logging["plot_log_pdf"] = True
    advanced_logging["plot_lcdm_sigmas"] = True

    #run the sampling algorithm.
    param_distributions = sample_joint(data_set, param_init, proposal_sigmas, param_ranges, 
                                       should_sample, noise_level, advanced_logging, 
                                       fixed_fields = True, phi_init = "ZEROES",
                                       iters_per_chain = 10_000, num_burn_in_fix_theta = 0, 
                                       seed = 67)

