from cmb_lensing.simulate import *
from cmb_lensing.wiener_filter import *
from cmb_lensing.util import *
from cmb_lensing.map_joint import *
import os
#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_log_compiles", True)

#sample the field
@jax.jit
def gibbs_sample_f(field_start, data_field, args, rng_key):

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
    lensed_field = qu2eb(fourier(lense_flow(map(eb2qu(new_field)), map(args["phi"]), 
                              n = 10, direction = FORWARD_LENSE, adjoint = False)))
    new_data = args["mask"] * args["beam"] * lensed_field + new_noise
    
    #Call the wiener filter with the field_start initial guess and data difference
    #between new and old simulations as the data term. We also use the current phi 
    #and covariance matrices from our sampling algorithms
    data_diff = data_field - new_data
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
                     mixing_d, step, num_burn_in_always_accept):
    
    always_accept = (step < num_burn_in_always_accept)
    mass_matrix = get_mass_matrix(cphi, nphi, mixing_g)

    mixed_phi, delta_h, accept = hmc_step(mixed_phi, always_accept, mixed_phi.nside, mass_matrix, 
                                          mixed_field, data, noise_covariance, 
                                          cphi, field_covariance, mask, beam, 
                                          mixing_d, mixing_g, rng_key)
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
    
    rng_key_1, rng_key_2 = jax.random.split(rng_key)
    p_matrix = field_from_covar_single_key(nside, mass_matrix.scalar_matrix, rng_key_1)
    p = mixed_field.replace(scalar_matrix = jfft.rfft2(p_matrix))
    delta_h, x_test, _ = symplectic_integrate(x, p, mixed_field, data, noise_covariance, 
                                              phi_covariance, field_covariance, mask, beam, 
                                              mixing_d, mixing_g, mass_matrix)
    
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

#------------------ symplectic integration ------------------------------------
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
    #NOTE what will be the difference between using mixed v.s. non-mixed here?
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

#sample cosmological parameters "thetas"
@jax.jit
def gibbs_sample_theta(theta_key, theta_range, cosmo_params, mixed_field, 
                       mixed_phi, data, noise_covariance, mask, beam, rng_key):

    #Get the logpdf at a certain theta value with all other inputs held constant
    def logpdf_partial(theta_val):

        #run a new load_sim with all other thetas held constant but varying over theta_key
        cosmo_params[theta_key] = theta_val
        trial_sim = load_sim(nside = 128, theta_pix = 2.5, pol = "I", 
                             master_seed = 67, **cosmo_params)
        #NOTE load_sim() is overkill here and performance poor since we only need the
        #new covariance matrices no GRF realizations of phi, f, data, etc... Should be
        #swapped out for something clearner...
        
        #now unmix the fields at the appropriate G & D matrices
        mixing_g = trial_sim.mixing_g
        mixing_d = trial_sim.mixing_d

        #then call logpdf in the mixed parametrization
        return mixed_logpdf(mixed_field, mixed_phi, data, noise_covariance, trial_sim.phi_covariance, 
                            trial_sim.field_covariance, mask, beam, mixing_g, mixing_d)
    
    #Evaluate the logpdf at all possible A_phi values with other inputs held constant
    logpdf_values = jax.vmap(logpdf_partial)(theta_range)
    a_phi = grid_and_sample(logpdf_values, theta_range, rng_key)
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

#NOTE prior experience with the MAP_joint() algorithm makes me think it is not worthwhile
#to JIT the main entry point here due to the for-loop compilation but it may be worth 
#experimenting with
def sample_joint(data_field, cosmo_params, cosmo_ranges, num_burn_in_fix_theta = 10, 
                 noise_level = 3, iters_per_chain = 500, num_burn_in_always_accept = 10, 
                 seed = 0):

    #we will store the learned parameter distributions in a dictionary
    cosmo_distributions = {}
    cosmo_distributions["ombh2"] = [] 
    cosmo_distributions["omch2"] = [] 
    cosmo_distributions["tau"] = [] 
    cosmo_distributions["ns"] = [] 
    cosmo_distributions["As"] = [] 
    cosmo_distributions["H0"] = [] 

    #TODO we need to be a lot less sloppy with magic numbers here and what gets passed to who...
    #run a simulation at the initial parameter values for all theta cosmological parameters
    #to get the baseline initial covariance matrices and store these matrices in the args object
    initial_sim = load_sim(nside = 128, theta_pix = 2.5, pol = "I", **cosmo_params,
                            master_seed = 37, uk_arcmin_t = noise_level)
    args = {}
    args["noise_covariance"] = initial_sim.noise_covariance
    args["phi_covariance"] = initial_sim.phi_covariance
    args["field_covariance"] = initial_sim.field_covariance
    args["quadratic_estimate"] = initial_sim.quadratic_estimate
    args["mask"] = initial_sim.mask
    args["beam"] = initial_sim.beam
    args["mixing_d"] = initial_sim.mixing_d
    args["mixing_g"] = initial_sim.mixing_g

    #Use a key to get reproduceable results
    sub_key = jax.random.PRNGKey(seed)
    rng_key, sub_key = jax.random.split(sub_key)

    #using f_init = zeros() and phi_init ~ N(0, Cphi(theta)) are
    #good pre-conditioners for the (f, phi, theta) initial position
    phi_matrix = field_from_covar_single_key(data_field.nside, 
                       args["phi_covariance"].scalar_matrix, rng_key)
    phi = initial_sim.phi.replace(scalar_matrix = jfft.rfft2(phi_matrix))
    temp_field = zero_scalar_field_like(phi)

    for iter in range(iters_per_chain):

        #1. sample the temperature field
        rng_key, sub_key = jax.random.split(sub_key)
        temp_field = gibbs_sample_f(temp_field, data_field, args, rng_key)

        #2. mix the fields
        mixed_temp, mixed_phi = mix(temp_field, phi, args["mixing_d"], args["mixing_g"])

        #3. sample the lensing potential phi
        rng_key, sub_key = jax.random.split(sub_key)
        mixed_phi, _, _ = gibbs_sample_phi(mixed_phi, args["phi_covariance"], 
                                           args["quadratic_estimate"], args["mixing_g"], 
                                           mixed_temp, data_field, args["noise_covariance"], 
                                           args["field_covariance"], args["mask"], args["beam"], 
                                           rng_key, args["mixing_d"], iter, 
                                           num_burn_in_always_accept)
        
        #4. sample your cosmo parameters
        if iter >= num_burn_in_fix_theta:
            for theta, theta_range in cosmo_ranges.items():
                rng_key, sub_key = jax.random.split(sub_key)
                theta_val = gibbs_sample_theta(theta, theta_range, cosmo_params, mixed_temp, 
                                               mixed_phi, data_field, args["noise_covariance"], 
                                               args["mask"], args["beam"], rng_key)
                cosmo_params[theta] = theta_val
                cosmo_distributions[theta].append(theta_val)

        #5. recompute Cf, Cphi, QE, G, and D matrices with the updated theta values
        rng_key, sub_key = jax.random.split(sub_key)
        new_sim = load_sim(nside = 128, theta_pix = 2.5, pol = "I", **cosmo_params,
                           master_seed = jax.random.randint(rng_key, shape = (), 
                           minval = 0, maxval = 2**31), uk_arcmin_t = noise_level)
        args["phi_covariance"] = new_sim.phi_covariance
        args["field_covariance"] = new_sim.field_covariance
        args["quadratic_estimate"] = new_sim.quadratic_estimate
        args["mixing_d"] = new_sim.mixing_d
        args["mixing_g"] = new_sim.mixing_g

        #6. unmix the fields using the updated version of G
        temp_field, phi = unmix(mixed_temp, mixed_phi, args["mixing_d"], args["mixing_g"])

    #Return your learned distributions at the end of the chain    
    return cosmo_distributions

if __name__ == "__main__":

    #Generate a "ground truth" simulated data set
    noise_level = 5
    data_set = load_sim(nside = 128, theta_pix = 2.5,
                        uk_arcmin_t = noise_level, pol = "I", 
                        master_seed = 67, a_phi = 0.75)
    data_field = data_set.data

    #initialize the cosmo parameters and their ranges
    cosmo_params = {} #TODO replace with realistic values
    cosmo_params["ombh2"] = 0
    cosmo_params["omch2"] = 0
    cosmo_params["tau"] = 0
    cosmo_params["ns"] = 0
    cosmo_params["As"] = 0
    cosmo_params["H0"] = 0
    cosmo_ranges = {} #TODO replace with realistic ranges
    cosmo_ranges["ombh2"] = jnp.linspace(0.1, 2, 50) 
    cosmo_ranges["omch2"] = jnp.linspace(0.1, 2, 50) 
    cosmo_ranges["tau"] = jnp.linspace(0.1, 2, 50) 
    cosmo_ranges["ns"] = jnp.linspace(0.1, 2, 50) 
    cosmo_ranges["As"] = jnp.linspace(0.1, 2, 50) 
    cosmo_ranges["H0"] = jnp.linspace(0.1, 2, 50) 

    #Plug the ground truth data into sample_joint() to try and learn the theta distributions
    start_time = time.time()
    a_phi_distribution = sample_joint(data_field, **cosmo_params, **cosmo_ranges, 
                                      num_burn_in_fix_theta = 100, noise_level = noise_level, 
                                      iters_per_chain = 500, num_burn_in_always_accept = 0, 
                                      seed = 0)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Done! Total time = {total_time}")

