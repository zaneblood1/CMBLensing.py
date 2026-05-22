from cmb_lensing.simulate import *
from cmb_lensing.wiener_filter import *
from cmb_lensing.util import *
from cmb_lensing.map_joint import *

#jax.config.update("jax_disable_jit", True)
jax.config.update("jax_log_compiles", True)

#sample the field
@jax.jit
def gibbs_sample_f(field_start, data_set, args, rng_key):
    #NOTE really the only thing we use the "original" data set here for
    #besides meta data is the original "data" field that we are using as 
    #our input to our data --> LCDM parameters black box...
    key_f, key_n = jax.random.split(rng_key)

    #Run a new simulation for f ~ N(0, Cf(thetas))
    new_field_matrix = field_from_covar_single_key(data_set.nside, 
                        args["field_covariance"].scalar_matrix, key_f)
    #Convert raw matrix to instance of FlatS0
    new_field = field_start.replace(scalar_matrix = jfft.rfft2(new_field_matrix))

    #Run a new simulation for n ~ N(0, Cn)
    new_noise_matrix = field_from_covar_single_key(data_set.nside, 
                        args["noise_covariance"].scalar_matrix, key_n)
    #Convert raw matrix to instance of FlatS0
    new_noise = field_start.replace(scalar_matrix = jfft.rfft2(new_noise_matrix))

    #d = M * B * L(phi) * f + n
    lensed_field = qu2eb(fourier(lense_flow(map(eb2qu(new_field)), map(args["phi"]), 
                              n = 10, direction = FORWARD_LENSE, adjoint = False)))
    new_data = args["mask"] * args["beam"] * lensed_field + new_noise
    
    #Call the wiener filter with the field_start initial guess and data difference
    #between new and old simulations as the data term. We also use the current phi 
    #and covariance matrices from our sampling algorithms
    data_diff = data_set.data - new_data
    delta_field = wiener_filter(field_start, args["phi"], data_diff, 
                                args["field_covariance"], args["noise_covariance"], 
                                args["mask"], args["beam"])

    #Return the new simulated unlensed field plus the wiener filter contribution
    return new_field + delta_field

#sample the lensing potential phi
@jax.jit
def gibbs_sample_phi(mixed_phi, cphi, nphi, mixing_g, 
                     mixed_field, data, noise_covariance, 
                     field_covariance, mask, beam, rng_key, 
                     mixing_d, a_phi, step, num_burn_in_always_accept):
    
    always_accept = (step < num_burn_in_always_accept)
    mass_matrix = get_mass_matrix(cphi, nphi, mixing_g, a_phi)

    mixed_phi, delta_h, accept = hmc_step(mixed_phi, always_accept, mixed_phi.nside, mass_matrix, 
                                          mixed_field, data, noise_covariance, 
                                          cphi, field_covariance, mask, beam, 
                                          mixing_d, mixing_g, a_phi, rng_key)
    return mixed_phi, delta_h, accept

#The mass matrix used in the HMC steps for sampling the lensing potential
def get_mass_matrix(cphi, nphi, mixing_g, aphi):
    return pinv(mixing_g) * pinv(mixing_g) * (pinv(aphi * cphi) + pinv(nphi))

#Hamiltonian Monte Carlo Step for the lensing potential
def hmc_step(x, always_accept, nside, mass_matrix, 
             mixed_field, data, noise_covariance, 
             phi_covariance, field_covariance, mask, beam, 
             mixing_d, mixing_g, a_phi, rng_key):
    
    rng_key_1, rng_key_2 = jax.random.split(rng_key)
    p_matrix = field_from_covar_single_key(nside, mass_matrix.scalar_matrix, rng_key_1)
    p = FlatS0(scalar_matrix = jfft.rfft2(p_matrix))
    delta_h, x_test, _ = symplectic_integrate(x, p, mixed_field, data, noise_covariance, 
                                              phi_covariance, field_covariance, mask, beam, 
                                              mixing_d, mixing_g, mass_matrix, a_phi)
    
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

#symplectic integration
def symplectic_integrate(x0, p0, mixed_field, data, noise_covariance, 
                        phi_covariance, field_covariance, mask, beam, 
                        mixing_d, mixing_g, mass_matrix,
                        a_phi, num_steps = 25, step_size = 0.01):
    
    #Get the mixed phi gradient at a certain mixed_phi value with all other
    #inputs held constant
    def mixed_grad_phi_partial(mixed_phi):
        return mixed_grad_phi_logpdf(mixed_field, mixed_phi, data, noise_covariance, 
                                     a_phi * phi_covariance, field_covariance, 
                                     mask, beam, mixing_d, mixing_g)
    
    #Get the logpdf at a certain mixed_phi value with all other inputs held constant
    def logpdf_partial(mixed_phi):
        #first unmix the fields
        field, phi = unmix(mixed_field, mixed_phi, mixing_d, mixing_g)
        #then call logpdf in the unmixed parametrization
        return logpdf(field, phi, data, noise_covariance, phi_covariance, 
                      field_covariance, mask, beam, mixing_g, a_phi = a_phi)
    
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

#sample cosmological parameters "thetas"
#NOTE we are only implementing a_phi right now
@jax.jit
def gibbs_sample_theta(a_phi_range, mixed_field, mixed_phi, mixing_d, quadratic_estimate,
                       data, noise_covariance, phi_covariance, field_covariance, 
                       mask, beam, rng_key):

    #Get the logpdf at a certain a_phi value with all other inputs held constant
    def logpdf_partial(a_phi):
        #first get the G matrix for a given a_phi
        mixing_g_matrix = get_g_matrix(phi_covariance.scalar_matrix, 
                                       quadratic_estimate.scalar_matrix, 
                                       a_phi)
        mixing_g = mixing_d.replace(scalar_matrix = mixing_g_matrix)
        #now unmix the fields at the appropriate G matrix
        field, phi = unmix(mixed_field, mixed_phi, mixing_d, mixing_g)
        #then call logpdf in the unmixed parametrization
        return logpdf(field, phi, data, noise_covariance, phi_covariance, 
                      field_covariance, mask, beam, mixing_g, a_phi = a_phi)
    
    #Evaluate the logpdf at all possible A_phi values with other inputs held constant
    logpdf_values = jax.vmap(logpdf_partial)(a_phi_range)
    a_phi = grid_and_sample(logpdf_values, a_phi_range, rng_key)
    return a_phi

#sample single parameter "theta" via inverse CDF
def grid_and_sample(logpdf_values, theta_values, rng_key):

    #shift for numerical stability then smooth
    logpdf_values = logpdf_values - jnp.max(logpdf_values)
    logpdf_values = loess(theta_values, logpdf_values)

    #convert to PDF and build a piecewise-linear CDF via cumulative trapezoid
    pdf = jnp.exp(logpdf_values)
    dx = jnp.diff(theta_values)
    areas = (pdf[:-1] + pdf[1:]) / 2 * dx
    cdf_values = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(areas)])
    cdf_values = cdf_values / cdf_values[-1]

    #inverse CDF sampling: interpolate theta as a function of CDF
    random_number = jax.random.uniform(rng_key)
    return jnp.interp(random_number, cdf_values, theta_values)

#NOTE prior experience with the MAP_joint() algoritm makes me think it is not worthwhile
#to JIT the main entry point here due to the for-loop compilation but it may be worth experimenting with
def sample_joint(data_set, num_burn_in_fix_theta = 10, 
                 iters_per_chain = 60, num_burn_in_always_accept = 10):

    #Initialize the starting guess for Aphi to be 1 (ground truth is 0.75 in our load_sim() method)
    cosmo_params = {}
    #NOTE using a python int here would re-trigger JIT-compilation and 
    #lead to performance loss in terms of run-time
    a_phi = jnp.float64(1.0)
    cosmo_params["a_phi"] = a_phi
    a_phi_range = jnp.linspace(0.5, 1.5, 30)
    a_phi_distribution = []

    #Store other necessary data in the args object
    args = {}
    args["noise_covariance"] = data_set.noise_covariance
    args["phi_covariance"] = data_set.phi_covariance
    args["field_covariance"] = data_set.field_covariance
    args["quadratic_estimate"] = data_set.quadratic_estimate
    args["mask"] = data_set.mask
    args["beam"] = data_set.beam
    mixing_d = data_set.mixing_d
    #The mixing G matrix depends on the a_phi we are trying to learn so we should
    #initialize it to our initial guess of a_phi
    mixing_g_matrix = get_g_matrix(args["phi_covariance"].scalar_matrix, 
                                   args["quadratic_estimate"].scalar_matrix, 
                                   a_phi)
    mixing_g = data_set.phi_covariance.replace(scalar_matrix = mixing_g_matrix)

    #Use a key to get reproduceable results
    sub_key = jax.random.PRNGKey(67)
    #Run a new simulation for phi ~ N(0, Cphi(a_phi))
    rng_key, sub_key = jax.random.split(sub_key)
    phi_matrix = field_from_covar_single_key(data_set.nside, 
                        a_phi * args["phi_covariance"].scalar_matrix, rng_key)
    #Convert raw matrix to instance of FlatS0
    phi = data_set.data.replace(scalar_matrix = jfft.rfft2(phi_matrix))
    args["phi"] = phi
    temp_field = 0*data_set.data
    #NOTE using f_init = zeros() and phi_init ~ N(0, Cphi(a_phi))
    #seem to give good results empirically...
    for iter in range(iters_per_chain):

        #1. sample the temperature field
        rng_key, sub_key = jax.random.split(sub_key)
        temp_field = gibbs_sample_f(temp_field, data_set, args, rng_key)

        #2. mix the fields
        phi = args["phi"]
        mixed_temp, mixed_phi = mix(temp_field, phi, mixing_d, mixing_g)

        #3. sample the lensing potential phi
        rng_key, sub_key = jax.random.split(sub_key)
        mixed_phi, _, _ = gibbs_sample_phi(mixed_phi, args["phi_covariance"], args["quadratic_estimate"],
                            mixing_g, mixed_temp, data_set.data, args["noise_covariance"], 
                            args["field_covariance"], args["mask"], args["beam"], rng_key,
                            mixing_d, cosmo_params["a_phi"], iter, num_burn_in_always_accept)
        
        #4. sample your cosmo parameters (just A_phi at the moment)
        if iter > num_burn_in_fix_theta:
            rng_key, sub_key = jax.random.split(sub_key)
            a_phi = gibbs_sample_theta(a_phi_range, mixed_temp, mixed_phi, mixing_d, args["quadratic_estimate"],
                                    data_set.data, args["noise_covariance"], args["phi_covariance"], 
                                    args["field_covariance"], args["mask"], args["beam"], rng_key)
            cosmo_params["a_phi"] = a_phi
            a_phi_distribution.append(a_phi)

        #5. recompute the mixing G matrix since it depends on the new value of a_phi
        #NOTE once we add in other cosmological parameters we will also need to update their
        #covariance matrices here as well
        mixing_g_matrix = get_g_matrix(args["phi_covariance"].scalar_matrix, 
                                       args["quadratic_estimate"].scalar_matrix, 
                                       a_phi)
        mixing_g = data_set.phi_covariance.replace(scalar_matrix = mixing_g_matrix)

        #6. unmix the fields using the updated version of G
        temp_field, phi = unmix(mixed_temp, mixed_phi, mixing_d, mixing_g)
        args["phi"] = phi

    #Return your learned distribution at the end of the chain    
    return np.array(a_phi_distribution)

if __name__ == "__main__":
    #Generate a "ground truth" simulated data set
    data_set = load_sim(nside = 256, theta_pix = 2.5, pol = "I", master_seed = 67, a_phi = 0.75)
    #Plug the ground truth into sample_joint() to try and learn the a_phi distribution
    a_phi_distribution = sample_joint(data_set, num_chains = 1, num_burn_in_fix_theta = 0, 
                                      iters_per_chain = 60, num_burn_in_always_accept = 0)
    print("Done!")

