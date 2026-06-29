from cmb_lensing.simulate import *
from cmb_lensing.wiener_filter import *
from cmb_lensing.util import *
from cmb_lensing.map_joint import *
from cmb_lensing.mixing import *
from cmb_lensing.constants import *
from scipy.integrate import quad
from scipy.optimize import brentq
import os
#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_log_compiles", True)

#sample the field
@jax.jit
def gibbs_sample_f(field_start, data_field, phi, args, rng_key):

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

@partial(jax.jit, static_argnames = ["use_priors", "over_relaxation_num_samps"])
def gibbs_sample_theta(theta, theta_range, mixed_temp, mixed_phi, a_phi, r, a_phi_fid,
                       data_field, args, rng_key, theta_old, over_relaxation_num_samps = -1, 
                       use_priors = False):
    
    #NOTE comment this out for the time being since I don't want to have to deal with making the 
    #if-else branch lax-compatible right now since this script is simply just trying to 
    #reproduce Marius' results and will likely not be included in the final version of the code ...
    
    # rng_key_1, rng_key_2 = jax.random.split(rng_key)
    # if theta == AR_KEYS["r"]:
    #     sampled_theta = gibbs_sample_r(theta_range, a_phi, mixed_temp, mixed_phi, args["mixing_g"], 
    #                                    data_field, args["noise_covariance"], 
    #                                    args["phi_covariance"], args["scalar_field_covariance"], 
    #                                    args["tensor_field_covariance"], args["mask"], args["beam"],
    #                                    args["r_fid"], rng_key_1, theta_old, over_relaxation_num_samps,
    #                                    use_priors)
    # if theta == AR_KEYS["a_phi"]:
    #     sampled_theta = gibbs_sample_a_phi(theta_range, a_phi_fid, r, mixed_temp, mixed_phi, 
    #                                        args["mixing_d"], args["quadratic_estimate"],
    #                                        data_field, args["noise_covariance"], 
    #                                        args["unscaled_phi_covariance"], args["field_covariance"],
    #                                        args["mask"], args["beam"], rng_key_2, theta_old, over_relaxation_num_samps, 
    #                                        use_priors)

    return gibbs_sample_a_phi(theta_range, a_phi_fid, r, mixed_temp, mixed_phi, 
                              args["mixing_d"], args["quadratic_estimate"],
                              data_field, args["noise_covariance"],
                              args["unscaled_phi_covariance"], args["field_covariance"],
                              args["mask"], args["beam"], rng_key, theta_old, over_relaxation_num_samps, 
                              use_priors)

def gibbs_sample_a_phi(a_phi_range, a_phi_fid, r, mixed_field, mixed_phi, mixing_d, quadratic_estimate,
                       data, noise_covariance, phi_covariance, field_covariance,
                       mask, beam, rng_key, old_a_phi, over_relaxation_num_samps = -1,
                       use_priors = False):

    #Get the logpdf at a certain a_phi value with all other inputs held constant
    def logpdf_partial(a_phi):

        #first get the G matrix for a given a_phi
        mixing_g_matrix = get_g_matrix(a_phi_fid * phi_covariance.scalar_matrix, 
                                       quadratic_estimate.scalar_matrix, a_phi_fid, 
                                       a_phi = a_phi)
        mixing_g = mixing_d.replace(scalar_matrix = mixing_g_matrix)

        #then call logpdf in the unmixed parametrization
        mixed_logpdf_value = mixed_logpdf(mixed_field, mixed_phi, data, noise_covariance, 
                                          a_phi * phi_covariance, field_covariance, mask, beam, 
                                          mixing_g, mixing_d)
        if use_priors:
            return mixed_logpdf_value - 0.5*(jnp.log(a_phi) + jnp.log(r))
        return mixed_logpdf_value
    
    #Evaluate the logpdf at all possible A_phi values with other inputs held constant
    logpdf_values = jax.vmap(logpdf_partial)(a_phi_range)
    a_phi = grid_and_sample(logpdf_values, a_phi_range, rng_key, old_a_phi, over_relaxation_num_samps)
    return a_phi

def gibbs_sample_r(r_range, a_phi, mixed_field, mixed_phi, mixing_g,
                   data, noise_covariance, phi_covariance, scalar_field_covariance, 
                   tensor_field_covariance, mask, beam, r_fid, rng_key, old_r,
                   over_relaxation_num_samps = -1, use_priors = False):

    #Get the logpdf at a certain r-value with all other inputs held constant
    def logpdf_partial(r):

        #first get the D matrix for a given r-value
        mixing_d_matrix = get_d_tt_matrix(scalar_field_covariance.scalar_matrix,
                                          tensor_field_covariance.scalar_matrix,
                                          noise_covariance.scalar_matrix,
                                          r, r_fid)
        mixing_d = mixing_g.replace(scalar_matrix = mixing_d_matrix)

        #also recompute the total field covariance
        field_covariance = scalar_field_covariance + (r/r_fid)*tensor_field_covariance
        
        #then call logpdf in the unmixed parametrization
        mixed_logpdf_value =  mixed_logpdf(mixed_field, mixed_phi, data, noise_covariance, 
                                           phi_covariance, field_covariance, mask, beam, 
                                           mixing_g, mixing_d)
        if use_priors:
            return mixed_logpdf_value - 0.5*(jnp.log(a_phi)+jnp.log(r))
        return mixed_logpdf_value
    
    #Evaluate the logpdf at all possible A_phi values with other inputs held constant
    logpdf_values = jax.vmap(logpdf_partial)(r_range)
    r = grid_and_sample(logpdf_values, r_range, rng_key, old_r, over_relaxation_num_samps)
    return r

#sample single parameter "theta" via inverse CDF using adaptive quadrature
#and root-finding, matching Julia's CMBLensing.jl grid_and_sample implementation.
#scipy calls are wrapped in jax.pure_callback for JIT compatibility.
def grid_and_sample(logpdf_values, theta_values, sub_key, theta_old,
                    over_relaxation_num_samps = -1):

    random_number = jax.random.uniform(sub_key)

    def _grid_and_sample_internal(theta_values, logpdf_values, random_number):
        xs = np.asarray(theta_values, dtype = np.float64)
        logpdfs = np.asarray(logpdf_values, dtype = np.float64)

        #trim leading/trailing zero-probability regions (matches Julia's findnext/findprev isfinite)
        finite = np.isfinite(logpdfs)
        first_finite = int(np.argmax(finite))
        last_finite = len(finite) - 1 - int(np.argmax(finite[::-1])) #::-1 syntax reverses a python array [a, b, c] --> [c, b, a]
        xs = xs[first_finite:last_finite + 1]
        logpdfs = logpdfs[first_finite:last_finite + 1]

        #shift for numerical stability then smooth
        logpdfs = logpdfs - np.max(logpdfs)
        xmin, xmax = float(xs[0]), float(xs[-1])
        interp_logpdfs = np.array(loess(xs, logpdfs, span = 0.25), dtype = np.float64)

        #callable interpolant over the smoothed log PDF
        def interp_logpdf(x):
            return float(np.interp(x, xs, interp_logpdfs))

        def nan2zero(x):
            return 0.0 if np.isnan(x) else x

        #normalize the PDF via adaptive quadrature (matches Julia's quadgk)
        def cdf(x):
            result, _ = quad(lambda t: nan2zero(np.exp(interp_logpdf(t))),
                             xmin, float(x), limit = 500, epsrel = 1e-4)
            return result

        logA = nan2zero(np.log(cdf(xmax)))
        interp_logpdfs -= logA
        logpdfs = interp_logpdfs

        #re-create interpolant with the normalized values
        def interp_logpdf_norm(x):
            return float(np.interp(x, xs, interp_logpdfs))

        def cdf_norm(x):
            result, _ = quad(lambda t: nan2zero(np.exp(interp_logpdf_norm(t))),
                             xmin, float(x), limit = 500, epsrel = 1e-4)
            return result

        #bracket the sample by finding where logpdf > peak - 1000
        peak = np.max(logpdfs)
        above = logpdfs > (peak - 1000)
        xmin_prime = float(xs[np.argmax(above)])
        xmax_prime = float(xs[len(xs) - 1 - np.argmax(above[::-1])])

        #inverse transform sampling via Brent root-finding (matches Julia's find_zero)
        r = float(random_number)
        cdf_lo = cdf_norm(xmin_prime)
        cdf_hi = cdf_norm(xmax_prime)

        if (cdf_lo - r) * (cdf_hi - r) >= 0:
            if logpdfs[0] > logpdfs[-1]:
                sampled = xmin_prime
            else:
                sampled = xmax_prime
        else:
            sampled = brentq(lambda x: cdf_norm(x) - r,
                             xmin_prime, xmax_prime,
                             xtol = (xmax - xmin) * 1e-4)

        return np.array(sampled, dtype = np.float64)

    sampled_theta = jax.pure_callback(
        _grid_and_sample_internal,
        jax.ShapeDtypeStruct((), jnp.float64),
        theta_values, logpdf_values, random_number
    )
    return sampled_theta

#sample single parameter "theta" via inverse CDF using cumulative trapezoid
def grid_and_sample_cumsum(logpdf_values, theta_values, sub_key, theta_old, iter,
                           over_relaxation_num_samps = -1):

    #shift for numerical stability then smooth
    logpdf_values = logpdf_values - jnp.max(logpdf_values)
    interp_logpdf_values = jax.pure_callback(
        lambda x, y: loess(x, y, span = 0.25),
        jax.ShapeDtypeStruct(logpdf_values.shape, logpdf_values.dtype),
        theta_values, logpdf_values
    )

    #convert to PDF and build a piecewise-linear CDF via cumulative trapezoid
    pdf = jnp.exp(interp_logpdf_values)
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

#NOTE prior experience with the MAP_joint() algorithm makes me think it is not worthwhile
#to JIT the main entry point here due to the for-loop compilation but it may be worth experimenting with
def sample_joint(data_set, param_init, param_ranges, should_sample, a_phi_fid, iters_per_chain = 500, 
                 num_burn_in_fix_theta = 100, num_burn_in_always_accept = 0, seed = None, map = None,
                 phi_start = "MAP", f_start = "MAP", over_relaxation_num_samps = -1,
                 use_priors = False):

    #Set the initial parameter values to the user specified starting guesses
    param_vals = {}
    for theta, theta_val in param_init.items():
        param_vals[theta] = [theta_val]

    args = {}
    args["noise_covariance"] = data_set.noise_covariance
    args["r_fid"] = data_set.fid_r
    args["a_phi_fid"] = a_phi_fid
    args["unscaled_phi_covariance"] = data_set.unscaled_phi_covariance
    args["phi_covariance"] = param_init["a_phi"] * args["unscaled_phi_covariance"]
    args["field_covariance"] = data_set.field_covariance
    args["mask"] = data_set.mask
    args["beam"] = data_set.beam
    args["quadratic_estimate"] = data_set.quadratic_estimate
    args["mixing_d"] = data_set.mixing_d
    data_field = data_set.data
    zeroes = 0*data_field

    #We must initialize the mixing  G matrix
    #to the proper values according to our starting guess for the cosmological parameters
    if should_sample["a_phi"]:
        mixing_g_matrix = get_g_matrix(args["a_phi_fid"] * args["unscaled_phi_covariance"].scalar_matrix, 
                                       args["quadratic_estimate"].scalar_matrix, args["a_phi_fid"], 
                                       a_phi = param_init["a_phi"])
        args["mixing_g"] = data_set.mixing_g.replace(scalar_matrix = mixing_g_matrix)
    else:
        args["mixing_g"] = data_set.mixing_g

    #Use a seed to get reproduceable results if so desired
    # if seed is not None:
    #     sub_key = jax.random.PRNGKey(seed)
    # else:
    sub_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    #run MAP_joint if necessary
    if phi_start == "MAP" or f_start == "MAP":
        data_set = data_set.replace(
            phi_covariance = data_set.phi_covariance.replace(
                scalar_matrix = args["phi_covariance"].scalar_matrix
            )
        )
        temp_joint, phi_joint = map_joint(data_set)
    
    #choose the starting point for (f, phi) in (f, phi, theta) parameter space
    if phi_start == "MAP":
        phi = phi_joint
    elif phi_start == "RNG":
        rng_key, sub_key = jax.random.split(sub_key)
        phi_matrix = field_from_covar_single_key(data_set.data.nside, 
                     args["phi_covariance"].scalar_matrix, rng_key)
        phi = data_set.phi.replace(scalar_matrix = jfft.rfft2(phi_matrix))
    else: 
        phi = zeroes

    if f_start == "MAP":
        temp_field = temp_joint
    elif f_start == "RNG":
        rng_key, sub_key = jax.random.split(sub_key)
        temp_matrix = field_from_covar_single_key(data_set.data.nside, 
                      args["field_covariance"].scalar_matrix, rng_key)
        temp_field = data_set.unlensed_field.replace(scalar_matrix = jfft.rfft2(temp_matrix))
    else: 
        temp_field = zeroes

    #run the chain for the maximum specified number if iterations
    for iter in range(1, iters_per_chain + 1):

        #1. sample the temperature field (using fstart = nothing EACH time...)
        rng_key, sub_key = jax.random.split(sub_key)
        temp_field = gibbs_sample_f(zeroes, data_field, phi, args, rng_key)

        #2. mix the fields
        mixed_temp, mixed_phi = mix(temp_field, phi, args["mixing_d"], args["mixing_g"])

        #julia_mixed_f = precision_load(f"/home/zane-blood/Desktop/julia_chain_debug/mixed_f_{iter}.npz")
        #julia_mixed_phi = precision_load(f"/home/zane-blood/Desktop/julia_chain_debug/mixed_phi_{iter}.npz")
        #print(f"F Fractional Difference = {jnp.linalg.norm(julia_mixed_f - mixed_temp.scalar_matrix)/jnp.linalg.norm(julia_mixed_f)}")
        #print(f"Phi Fractional Difference = {jnp.linalg.norm(julia_mixed_phi - mixed_phi.scalar_matrix)/jnp.linalg.norm(julia_mixed_phi)}")
        
        #3. sample the lensing potential phi
        rng_key, sub_key = jax.random.split(sub_key)
        mixed_phi, _, _ = gibbs_sample_phi(mixed_phi, mixed_temp, data_field, rng_key,
                                           args, iter, num_burn_in_always_accept)
        
        #4. sample your cosmo parameters
        if iter >= num_burn_in_fix_theta:
            for theta, theta_range in param_ranges.items():
                if should_sample[theta]:
                    rng_key, sub_key = jax.random.split(sub_key)
                    theta_val = gibbs_sample_theta(AR_KEYS[theta], theta_range, mixed_temp, mixed_phi, 
                                                   param_vals["a_phi"][-1], param_vals["r"][-1], a_phi_fid,
                                                   data_field, args, rng_key, param_vals[theta][-1],
                                                   over_relaxation_num_samps = over_relaxation_num_samps, 
                                                   use_priors = use_priors)
                    param_vals[theta].append(theta_val)

        #5. recompute mixing and covariance matrices using the newly sampled parameter values
        if should_sample["a_phi"]:
            args["phi_covariance"] = param_vals["a_phi"][-1] * args["unscaled_phi_covariance"]
            mixing_g_matrix = get_g_matrix(a_phi_fid * args["unscaled_phi_covariance"].scalar_matrix, 
                                           args["quadratic_estimate"].scalar_matrix, a_phi_fid, 
                                           a_phi = param_vals["a_phi"][-1])
            args["mixing_g"] = args["mixing_g"].replace(scalar_matrix = mixing_g_matrix)

            # # -------------------------------------------------------- DEBUG --------------------------------------------------------
            # #Store the sampled a_phi value to a debug text file...
            # file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/chains_v7_aphi_0dot75/map_{map}_chain_{seed}_history.txt"
            # os.makedirs(os.path.dirname(file_path), exist_ok = True)
            # with open(file_path, "a") as file:
            #     file.write(str(param_vals["a_phi"][-1]) + "\n")
            # # -------------------------------------------------------- DEBUG --------------------------------------------------------

        #6. unmix the fields using the updated version of the G & D matrices
        temp_field, phi = unmix(mixed_temp, mixed_phi, args["mixing_d"], args["mixing_g"])
        print(np.array(param_vals["a_phi"]))

    #Return your learned distributions at the end of the chain
    #for each parameter that was sampled    
    return param_vals

if __name__ == "__main__":

    #Generate a "ground truth" simulated data set
    nside = 128
    theta_pix = 2.5
    pol = "I"
    master_seed = 33
    a_phi_ground = 0.75
    r_ground = 0.2
    noise_level = 5
    data_set = load_sim(nside, theta_pix, pol, master_seed, 
                        uk_arcmin_t = noise_level, a_phi = a_phi_ground, 
                        r = r_ground)

    #Plug the ground truth data into sample_joint() to 
    #try and learn the r & a_phi distributions
    start_time = time.time()

    #initial starting guesses for parameters
    param_init = {}
    param_init["a_phi"] = 1
    param_init["r"] = 0.2

    #allowed search / sample range for parameters
    param_ranges = {}
    param_ranges["a_phi"] = jnp.linspace(0.1, 2.5, 200)
    param_ranges["r"] = jnp.linspace(0.005, 0.04, 200)

    #Whether or not to sample each parameter
    should_sample = {}
    should_sample["r"] = False
    should_sample["a_phi"] = True

    #time and run the sampling algorithm
    param_distributions = sample_joint(data_set, param_init, param_ranges, should_sample, a_phi_ground,
                                       iters_per_chain = 5000, num_burn_in_fix_theta = 0, 
                                       over_relaxation_num_samps = -1, use_priors = False,
                                       num_burn_in_always_accept = 0, map = 0, seed = None,
                                       phi_start = "RNG", f_start = "ZEROES")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Done! Total time = {total_time}")

