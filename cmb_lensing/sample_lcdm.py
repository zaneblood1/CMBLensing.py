from cmb_lensing.simulate import *
from cmb_lensing.wiener_filter import *
from cmb_lensing.util import *
from cmb_lensing.map_joint import *
from cmb_lensing.mixing import *
from cmb_lensing.constants import *
import cambemul
from cambemul.emulator import build_model
import os

#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_log_compiles", True)

#parameter ordering used by the emulator (must match cambemul param_names)
PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_ORDER)}

def prepare_emulator_jax(emulator):
    """Extract emulator internals into a JIT-friendly form.

    Returns (predict_tt_fn, predict_pp_fn, emu_params) where the predict
    functions close over the Flax models and normalization constants, and
    emu_params is a pytree of Flax weights passable through JIT.
    """
    params_tt, meta_tt, _ = emulator.members["uTT"]
    params_pp, meta_pp, _ = emulator.members["PP"]

    model_tt = build_model(meta_tt["config"])
    model_pp = build_model(meta_pp["config"])

    #pre-convert normalization constants to JAX arrays on device
    tt_x_mean = jnp.array(meta_tt["x_mean"], dtype = jnp.float32)
    tt_x_std = jnp.array(meta_tt["x_std"], dtype = jnp.float32)
    tt_t_mean = jnp.array(meta_tt["t_mean"], dtype = jnp.float64)
    tt_t_std = jnp.array(meta_tt["t_std"], dtype = jnp.float64)
    tt_pca_basis_T = jnp.array(meta_tt["pca_basis"], dtype = jnp.float64).T
    tt_pca_mean = jnp.array(meta_tt["pca_mean"], dtype = jnp.float64)

    pp_x_mean = jnp.array(meta_pp["x_mean"], dtype = jnp.float32)
    pp_x_std = jnp.array(meta_pp["x_std"], dtype = jnp.float32)
    pp_t_mean = jnp.array(meta_pp["t_mean"], dtype = jnp.float64)
    pp_t_std = jnp.array(meta_pp["t_std"], dtype = jnp.float64)
    pp_pca_basis_T = jnp.array(meta_pp["pca_basis"], dtype = jnp.float64).T
    pp_pca_mean = jnp.array(meta_pp["pca_mean"], dtype = jnp.float64)

    emu_params = {"tt": params_tt, "pp": params_pp}
    emu_meta = {"tt": meta_tt, "pp": meta_pp}

    def predict_tt(emu_params, x):
        """Pure-JAX uTT prediction. x is (N, 5) float."""
        xn = ((x - tt_x_mean) / tt_x_std).astype(jnp.float32)
        out = model_tt.apply(emu_params["tt"], xn)
        coeffs = out * tt_t_std + tt_t_mean
        return jnp.power(10.0, coeffs @ tt_pca_basis_T + tt_pca_mean)

    def predict_pp(emu_params, x):
        """Pure-JAX PP prediction. x is (N, 5) float."""
        xn = ((x - pp_x_mean) / pp_x_std).astype(jnp.float32)
        out = model_pp.apply(emu_params["pp"], xn)
        coeffs = out * pp_t_std + pp_t_mean
        return jnp.power(10.0, coeffs @ pp_pca_basis_T + pp_pca_mean)

    return (predict_tt, predict_pp, emu_params, emu_meta, model_tt, model_pp,
            tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean,
            pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean)

#sample the field
@jax.jit
def gibbs_sample_f(field_start, data_field, phi, args, rng_key):
    #NOTE really the only thing we use the "original" data set here for
    #besides meta data is the original "data" field that we are using as 
    #our input to our data --> LCDM parameters black box...
    key_f, key_n = jax.random.split(rng_key)

    #Run a new simulation for f ~ N(0, Cf(thetas))
    new_field_matrix = field_from_covar_single_key(data_field.nside, 
                        args["field_covariance"].scalar_matrix, key_f)
    #Convert raw matrix to instance of FlatS0
    new_field = field_start.replace(scalar_matrix = jfft.rfft2(new_field_matrix))

    #Run a new simulation for n ~ N(0, Cn)
    new_noise_matrix = field_from_covar_single_key(data_field.nside, 
                        args["noise_covariance"].scalar_matrix, key_n)
    #Convert raw matrix to instance of FlatS0
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
    return pinv(mixing_g) * pinv(mixing_g) * (pinv(cphi) + pinv(nphi))

#Hamiltonian Monte Carlo Step for the lensing potential
def hmc_step(x, always_accept, nside, mass_matrix,
             mixed_field, data, noise_covariance, 
             phi_covariance, field_covariance, mask, beam, 
             mixing_d, mixing_g, rng_key):
    
    #generate a random kick in momentum space
    rng_key_1, rng_key_2 = jax.random.split(rng_key)
    p_matrix = field_from_covar_single_key(nside, mass_matrix.scalar_matrix, rng_key_1)
    p = mixed_field.replace(scalar_matrix = jfft.rfft2(p_matrix))

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
                        num_steps = 30, step_size = 0.01):
    
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

@partial(jax.jit, static_argnames = ["model_tt", "model_pp", "lmax", "lmax_prime",
                                     "nside", "pix_width", "theta_pix",
                                     "over_relaxation_num_samps"])
def gibbs_sample_theta(theta_key_idx, theta_old, theta_range,
                       mixed_temp_matrix, mixed_phi_matrix, data_matrix,
                       current_params, emu_params, model_tt,
                       model_pp, rng_key, lmax, lmax_prime, nside, pix_width, theta_pix,
                       ell_grid, ells,
                       cphi_fid, qe_scalar, cn_scalar, mask_matrix, beam_matrix,
                       fourier_weights,
                       tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean,
                       pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean,
                       over_relaxation_num_samps = -1):
    """Sample a single cosmological parameter via grid evaluation + inverse CDF."""

    #reconstruct Flax structs from raw arrays inside JIT for stable pytree tracing
    def _field(m):
        return FlatS0(scalar_matrix = m, fourier_weights = fourier_weights,
                      nside = nside, theta_pix = theta_pix, pix_width = pix_width,
                      basis = Basis.FOURIER, parametrization = Parametrization.T)
    def _op(m):
        return DiagonalScalar(scalar_matrix = m, fourier_weights = fourier_weights,
                              nside = nside, theta_pix = theta_pix, pix_width = pix_width)

    mixed_temp = _field(mixed_temp_matrix)
    mixed_phi = _field(mixed_phi_matrix)
    data_field = _field(data_matrix)
    noise_covariance = _op(cn_scalar)
    mask = _op(mask_matrix)
    beam = _op(beam_matrix)

    #batch-predict all grid points at once: build (N, 5) parameter array
    #by tiling current_params and overwriting the sampled column
    N = theta_range.shape[0]
    params_batch = jnp.tile(current_params, (N, 1))
    params_batch = params_batch.at[:, theta_key_idx].set(theta_range)

    def predict_tt(emu_params, x):
        xn = ((x - tt_x_mean) / tt_x_std).astype(jnp.float32)
        out = model_tt.apply(emu_params["tt"], xn)
        coeffs = out * tt_t_std + tt_t_mean
        return jnp.power(10.0, coeffs @ tt_pca_basis_T + tt_pca_mean)

    def predict_pp(emu_params, x):
        xn = ((x - pp_x_mean) / pp_x_std).astype(jnp.float32)
        out = model_pp.apply(emu_params["pp"], xn)
        coeffs = out * pp_t_std + pp_t_mean
        return jnp.power(10.0, coeffs @ pp_pca_basis_T + pp_pca_mean)

    cl_tt_batch = predict_tt(emu_params, params_batch)
    cl_pp_batch = predict_pp(emu_params, params_batch)

    def single_logpdf(i):
        cl_tt = interpolate_cls(cl_tt_batch[i], lmax, lmax_prime)
        cl_pp = interpolate_cls(cl_pp_batch[i], lmax, lmax_prime)

        cf = covar_matrix_from_cls(nside, pix_width,
                                   ell_grid, ells,
                                   cl_tt, origin_value = 0)
        cphi = covar_matrix_from_cls(nside, pix_width,
                                     ell_grid, ells,
                                     cl_pp, origin_value = 0)

        g = get_g_matrix_lcdm(cphi_fid, cphi, qe_scalar)
        d = get_d_tt_matrix(cf, jnp.zeros_like(cf), cn_scalar, 1, 1)

        return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                            noise_covariance, _op(cphi), _op(cf),
                            mask, beam, _op(g), _op(d))

    logpdf_values = jax.vmap(single_logpdf)(jnp.arange(N))
    theta_new = grid_and_sample(logpdf_values, theta_range, rng_key,
                                theta_old, over_relaxation_num_samps)
    return theta_new

#sample single parameter "theta" via inverse CDF
@partial(jax.jit, static_argnames = ["over_relaxation_num_samps"])
def grid_and_sample(logpdf_values, theta_values, sub_key, theta_old,
                    over_relaxation_num_samps = -1):

    #shift for numerical stability then smooth
    logpdf_values = logpdf_values - jnp.max(logpdf_values)
    logpdf_values = loess(theta_values, logpdf_values)

    #convert to PDF and build a piecewise-linear CDF via cumulative trapezoid
    pdf = jnp.exp(logpdf_values)
    dx = jnp.diff(theta_values)
    areas = (pdf[:-1] + pdf[1:]) / 2 * dx
    cdf_values = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(areas)])
    cdf_values = cdf_values / cdf_values[-1]

    if over_relaxation_num_samps != -1:
        #generate a set of samples if over relaxation is enabled and choose
        #theta_new = theta_{K - i + 1} where K is the size of the set of samples
        #and the index "i" is chosen such that theta_i < theta_old < theta_{i+1}
        sample_set = []
        for _ in over_relaxation_num_samps:
            rng_key, sub_key = jax.random.split(sub_key)
            random_number = jax.random.uniform(rng_key)
            sampled_theta = jnp.interp(random_number, cdf_values, theta_values)
            sample_set.append(sampled_theta)

        #convert to jnp array and sort
        sorted_set = jnp.sort(jnp.array(sample_set))
        index = jnp.searchsorted(sorted_set, theta_old, side = 'right')
        chosen_index = over_relaxation_num_samps - index + 1
        #clamp to avoid out-of-bounds problems
        chosen_index = jnp.min(len(sorted_set)-1, jnp.max(0, chosen_index))
        chosen_theta = sorted_set[chosen_index]
        return chosen_theta

    #inverse CDF sampling: interpolate theta as a function of CDF
    random_number = jax.random.uniform(sub_key)
    sampled_theta = jnp.interp(random_number, cdf_values, theta_values)
    return sampled_theta

@partial(jax.jit, static_argnames = ["lmax", "lmax_prime"])
def interpolate_cls(cls, lmax, lmax_prime):

    ell = jnp.arange(2, lmax + 1).astype(jnp.float64)
    ell_prime = jnp.arange(2, lmax_prime + 1).astype(jnp.float64)

    def exponential_interpolate(cls):
        return jnp.exp(jnp.interp(
            jnp.log(ell), jnp.log(ell_prime), jnp.log(cls),
            left="extrapolate", right="extrapolate"
        ))

    def linear_interpolate(cls):
        return jnp.interp(ell, ell_prime, cls, left=0.0, right=0.0)

    cls = jax.lax.cond(
        jnp.all(cls > 0),
        exponential_interpolate,
        linear_interpolate,
        cls
    )

    return cls

#JIT-able version of get_new_cosmo_matrices using the pure-JAX emulator
def get_new_cosmo_matrices(current_params, predict_tt, predict_pp, emu_params, args):
    """Compute (g, d, cf, cphi) from a parameter vector using the pure-JAX emulator."""
    x = current_params[None, :]
    cl_tt = predict_tt(emu_params, x)[0]
    cl_pp = predict_pp(emu_params, x)[0]

    cl_tt = interpolate_cls(cl_tt, args["lmax"], args["lmax_prime"])
    cl_pp = interpolate_cls(cl_pp, args["lmax"], args["lmax_prime"])

    cf = covar_matrix_from_cls(args["nside"], args["pix_width"],
                               args["ell_grid"], args["ells"],
                               cl_tt, origin_value = 0)
    cphi = covar_matrix_from_cls(args["nside"], args["pix_width"],
                                 args["ell_grid"], args["ells"],
                                 cl_pp, origin_value = 0)

    g = get_g_matrix_lcdm(args["cphi_fid"], cphi, args["quadratic_estimate"].scalar_matrix)
    d = get_d_tt_matrix(cf, jnp.zeros_like(cf), args["noise_covariance"].scalar_matrix, 1, 1)
    return g, d, cf, cphi

#Lighter-weight version of the above method to just compute the field covariance and not D, G, Cphi...
def get_new_cf_matrix(current_params, predict_tt, emu_params, args):
    """Compute (g, d, cf, cphi) from a parameter vector using the pure-JAX emulator."""
    x = current_params[None, :]
    cl_tt = predict_tt(emu_params, x)[0]
    cl_tt = interpolate_cls(cl_tt, args["lmax"], args["lmax_prime"])
    cf = covar_matrix_from_cls(args["nside"], args["pix_width"],
                               args["ell_grid"], args["ells"],
                               cl_tt, origin_value = 0)
    return cf

def get_starting_f_and_phi(f_start, phi_start, data_set, args, sub_key):

    #run MAP_joint if necessary
    if phi_start == "MAP" or f_start == "MAP":
        temp_joint, phi_joint = map_joint(data_set)

    #initialize the starting phi to either MAP, random realization, or zeroes
    if phi_start == "MAP":
        phi = phi_joint
    elif phi_start == "RNG":
        rng_key, sub_key = jax.random.split(sub_key)
        phi_matrix = field_from_covar_single_key(data_set.data.nside,
                           args["phi_covariance"].scalar_matrix, rng_key)
        phi = data_set.phi.replace(scalar_matrix = jfft.rfft2(phi_matrix))
    elif phi_start == "ZEROES":
        phi = 0*data_set.phi
    else:
        raise KeyError

    #initialize the starting field to either MAP, random realization, or zeroes
    if f_start == "MAP":
        temp_field = temp_joint
    elif f_start == "RNG":
        rng_key, sub_key = jax.random.split(sub_key)
        temp_matrix = field_from_covar_single_key(data_set.data.nside,
                           args["field_covariance"].scalar_matrix, rng_key)
        temp_field = data_set.phi.replace(scalar_matrix = jfft.rfft2(temp_matrix))
    elif phi_start == "ZEROES":
        temp_field = 0*data_set.unlensed_field
    else:
        raise KeyError

    return temp_field, phi

def add_metadata_to_args(args, data_set, lmax):
    #This set of data is generally needed for computing covariance
    #matrices from a set of Cls. The lmax variable determines how far out
    #in the multipole range we will interpolate the Cls which only go out
    #to lmax_prime in multipole
    args["lmax"] = lmax
    args["lmax_prime"] = min(lmax, EMULATOR_MAX_ELL)
    args["nside"] = data_set.nside
    args["pix_width"] = data_set.pix_width
    ell_grid, _ = gen_ell_grid(data_set.nside, data_set.theta_pix)
    args["ell_grid"] = ell_grid
    args["ells"] = jnp.arange(2, lmax + 1).astype(jnp.float64)
    return args

#Convert dictionaries between emulator and load_sim() naming convetion
def emul_2_camb_naming(param_init):
    param_init["cosmomc_theta"] = param_init["theta_MC_100"] / 100
    del param_init["theta_MC_100"]
    param_init["As"] = jnp.exp(param_init["logA"]) * 1e-10
    del param_init["logA"]
    return param_init

def add_starting_matrices_to_args(args, data_set, noise_level, param_init, current_params,
                                  predict_tt, predict_pp, emu_params):
    
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
    param_init = emul_2_camb_naming(param_init)
    initial_cond = load_sim(data_set.nside, data_set.theta_pix, "I", 
                            np.random.randint(0, 2**31), **param_init,
                            uk_arcmin_t = noise_level, r = 0, nt = 0)
    cf = get_new_cf_matrix(current_params, predict_tt, emu_params, args)
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
    g, d, _, cphi = get_new_cosmo_matrices(current_params, predict_tt, predict_pp,
                                           emu_params, args)
    args["phi_covariance"] = data_set.phi_covariance.replace(scalar_matrix = cphi)
    args["mixing_g"] = data_set.mixing_g.replace(scalar_matrix = g)
    args["mixing_d"] = data_set.mixing_d.replace(scalar_matrix = d)

    return args

#The data set that was passed to sample_joint uses the ground truth covariance matrices
#so we must change them to the data that corresponds to our initial position in 
#cosmological parameter space
def set_initial_ds_conditions(data_set, args):
    data_set = data_set.replace(
            phi_covariance = data_set.phi_covariance.replace(
                scalar_matrix = args["phi_covariance"].scalar_matrix
            ),
            field_covariance = data_set.field_covariance.replace(
                scalar_matrix = args["field_covariance"].scalar_matrix
            ),
            mixing_d = data_set.mixing_d.replace(
                scalar_matrix = args["mixing_d"].scalar_matrix
            ),
            quadratic_estimate = data_set.quadratic_estimate.replace(
                scalar_matrix = args["quadratic_estimate"].scalar_matrix
            )
        )
    return data_set

def update_args_after_sample(current_params, predict_tt, predict_pp,
                             emu_params, args): 
    g, d, cf, cphi = get_new_cosmo_matrices(current_params, predict_tt, predict_pp,
                                            emu_params, args)
    args["phi_covariance"] = args["phi_covariance"].replace(scalar_matrix = cphi)
    args["field_covariance"] = args["field_covariance"].replace(scalar_matrix = cf)
    args["mixing_g"] = args["mixing_g"].replace(scalar_matrix = g)
    args["mixing_d"] = args["mixing_d"].replace(scalar_matrix = d)
    return args

#algorithm to jointly sample cosmological parameters
def sample_joint(data_set, param_init, param_ranges, should_sample, noise_level, iters_per_chain = 500,
                 num_burn_in_fix_theta = 100, num_burn_in_always_accept = 0, seed = None,
                 phi_start = "MAP", f_start = "MAP", over_relaxation_num_samps = -1, lmax = 17_000):
    
    #Prepare the JIT-friendly emulator (build models once, extract weights)
    emulator = cambemul.loademul(os.getcwd() + "/camb_emulator")
    (predict_tt, predict_pp, emu_params, emu_meta, model_tt, model_pp,
     tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean,
     pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean) = prepare_emulator_jax(emulator)

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
                                         predict_tt, predict_pp, emu_params)

    #change data_set covariance matrices from ground truth to initial
    #starting point in cosmological parameter space
    data_set = set_initial_ds_conditions(data_set, args)
    data_field = data_set.data

    #Use a seed to get reproduceable results if so desired
    if seed is not None:
        sub_key = jax.random.PRNGKey(seed)
    else:
        sub_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    #choose the starting point for (f, phi) in (f, phi, theta) cosmological parameter space
    temp_field, phi = get_starting_f_and_phi(f_start, phi_start, data_set, args, sub_key)

    #run the chain for the maximum specified number if iterations
    for iter in range(1, iters_per_chain):

        #1. sample the temperature field
        start_time = time.time()
        rng_key, sub_key = jax.random.split(sub_key)
        temp_field = gibbs_sample_f(temp_field, data_field, phi, args, rng_key)
        end_time = time.time()
        print(f"sample f time = {end_time- start_time}")

        #2. mix the fields
        start_time = time.time()
        mixed_temp, mixed_phi = mix(temp_field, phi, args["mixing_d"], args["mixing_g"])
        end_time = time.time()
        print(f"mixing time = {end_time - start_time}")

        #3. sample the lensing potential phi
        start_time = time.time()
        rng_key, sub_key = jax.random.split(sub_key)
        mixed_phi, _, _ = gibbs_sample_phi(mixed_phi, mixed_temp, data_field, rng_key,
                                           args, iter, num_burn_in_always_accept)
        end_time = time.time()
        print(f"sample phi time = {end_time - start_time}")

        start_time = time.time()
        #4. sample your cosmo parameters
        if iter >= num_burn_in_fix_theta:
            for theta, theta_range in param_ranges.items():
                if should_sample[theta]:
                    rng_key, sub_key = jax.random.split(sub_key)
                    theta_key_idx = PARAM_INDEX[theta]
                    theta_val = gibbs_sample_theta(jnp.array(theta_key_idx), jnp.array(param_vals[theta][-1]),
                                                   theta_range, mixed_temp.scalar_matrix,
                                                   mixed_phi.scalar_matrix, data_field.scalar_matrix,
                                                   current_params,
                                                   emu_params, model_tt,
                                                   model_pp, rng_key, args["lmax"], args["lmax_prime"],
                                                   args["nside"], args["pix_width"],
                                                   data_field.theta_pix,
                                                   args["ell_grid"], args["ells"],
                                                   args["cphi_fid"], args["quadratic_estimate"].scalar_matrix,
                                                   args["noise_covariance"].scalar_matrix,
                                                   args["mask"].scalar_matrix, args["beam"].scalar_matrix,
                                                   data_field.fourier_weights,
                                                   tt_x_mean, tt_x_std, tt_t_mean,
                                                   tt_t_std, tt_pca_basis_T, tt_pca_mean,
                                                   pp_x_mean, pp_x_std, pp_t_mean,
                                                   pp_t_std, pp_pca_basis_T, pp_pca_mean,
                                                   over_relaxation_num_samps = over_relaxation_num_samps)
                    param_vals[theta].append(theta_val)
                    current_params = current_params.at[theta_key_idx].set(theta_val)

            #5. recompute mixing and covariance matrices using the newly sampled parameter values
            #NOTE this is only necessary if we are past the burn-in phase
            args = update_args_after_sample(current_params, predict_tt, predict_pp,
                                            emu_params, args)
        
        #TODO fix the fact that this is re-compiling with each loop.... :(
        end_time = time.time()
        print(f"sample 5 thetas time = {end_time - start_time}")
            
            # -------------------------------------------------------- DEBUG --------------------------------------------------------
            #Store the sampled a_phi value to a debug text file...
            # ombh2_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/chains_lcdm/ombh2_chain_{seed}_history.txt"
            # omch2_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/chains_lcdm/omch2_chain_{seed}_history.txt"
            # theta_MC_100_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/chains_lcdm/theta_MC_100_chain_{seed}_history.txt"
            # logA_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/chains_lcdm/logA_chain_{seed}_history.txt"
            # ns_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/chains_lcdm/ns_chain_{seed}_history.txt"
            # with open(ombh2_file_path, "a") as file:
            #     file.write(str(param_vals["ombh2"][-1]) + "\n")
            # with open(omch2_file_path, "a") as file:
            #     file.write(str(param_vals["omch2"][-1]) + "\n")
            # with open(theta_MC_100_file_path, "a") as file:
            #     file.write(str(param_vals["theta_MC_100"][-1]) + "\n")
            # with open(logA_file_path, "a") as file:
            #     file.write(str(param_vals["logA"][-1]) + "\n")
            # with open(ns_file_path, "a") as file:
            #     file.write(str(param_vals["ns"][-1]) + "\n")
            # -------------------------------------------------------- DEBUG --------------------------------------------------------


        #6. unmix the fields using the updated version of the G & D matrices
        temp_field, phi = unmix(mixed_temp, mixed_phi, args["mixing_d"], args["mixing_g"])

    #Return your learned distributions at the end of the chain
    #for each parameter that was sampled
    return param_vals

if __name__ == "__main__":

    #initial starting guesses for parameters (fiducial values)
    ground_truth_params = {}
    ground_truth_params["ombh2"] = 0.022386
    ground_truth_params["omch2"] = 0.109381
    ground_truth_params["cosmomc_theta"] = 0.01031732
    ground_truth_params["As"] = jnp.exp(3.218387) * 1e-10
    ground_truth_params["ns"] = 0.959814

    #Generate a "ground truth" simulated data set
    nside = 128
    theta_pix = 2.5
    pol = "I"
    master_seed = 33
    noise_level = 5

    #For the paramaters we want to infer, the tensor-to-scalar ratio should be zero
    #TODO we need to make sure the default parameters in load_sim() match the default 
    #parameter's in Yuuki's emulator code... 
    data_set = load_sim(nside, theta_pix, pol, master_seed, **ground_truth_params,
                        uk_arcmin_t = noise_level, r = 0, nt = 0)
    
    #switch back to cambemul naming conventions...
    #NOTE if we only sample ombh2 the other parameters should be fixed at their fiducial
    #values for better convergence...
    param_init = {}
    param_init["ombh2"] = 0.024389 #+5 sigma from Yuuki's mean for training
    param_init["omch2"] = 0.079704 #-5 sigma from mean
    param_init["theta_MC_100"] = 0.900723 #-5 sigma from mean #NOTE +5 here and -5 for logA seems to break CAMB
    param_init["logA"] = 3.782861 #+5 sigma from mean
    param_init["ns"] = 1.042186 #+5 sigma from mean

    #allowed search / sample range for parameters... The min and max values are +/- 5 std
    #from the training mean for the CAMB emulator
    SEARCH_PRECISION = 200
    param_ranges = {}
    param_ranges["ombh2"] = jnp.linspace(0.020413, 0.024389, SEARCH_PRECISION)
    param_ranges["omch2"] = jnp.linspace(0.079704, 0.155541, SEARCH_PRECISION)
    param_ranges["theta_MC_100"] = jnp.linspace(0.900723, 1.156063, SEARCH_PRECISION)
    param_ranges["logA"] = jnp.linspace(2.661635, 3.782861, SEARCH_PRECISION)
    param_ranges["ns"] = jnp.linspace(0.867143, 1.042186, SEARCH_PRECISION)

    #Whether or not to sample each parameter
    #NOTE just sampling ombh2 for the time being while we get up and running
    should_sample = {}
    should_sample["ombh2"] = True
    should_sample["omch2"] = True
    should_sample["theta_MC_100"] = True
    should_sample["logA"] = True
    should_sample["ns"] = True

    #run the sampling algorithm
    param_distributions = sample_joint(data_set, param_init, param_ranges, should_sample, noise_level,
                                       iters_per_chain = 5000, num_burn_in_fix_theta = 0, 
                                       over_relaxation_num_samps = -1, seed = 67,
                                       num_burn_in_always_accept = 0, phi_start = "MAP", 
                                       f_start = "MAP")
    print(f"Done!")

