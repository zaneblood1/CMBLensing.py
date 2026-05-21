from cmb_lensing.simulate import *
from cmb_lensing.wiener_filter import *
from cmb_lensing.util import *
from cmb_lensing.map_joint import *
import optimistix as optx
#jax.config.update("jax_disable_jit", True)
jax.config.update("jax_log_compiles", True)

#sample the field
@jax.jit
def gibbs_sample_f(field_start, data_set, cosmo_params, args, rng_key):
    #NOTE really the only thing we use the "original" data set here for
    #besides meta data is the original "data" field that we are using as 
    #our input to our data --> LCDM parameters black box...

    #Run a new simulation
    new_sim = load_sim(data_set.nside, data_set.theta_pix, pol = "I", 
                 master_seed = jax.random.randint(rng_key, shape=(), minval=0, maxval=2**31), 
                 **cosmo_params)

    #Extract the unlensed field and data objects from the new simulation run
    new_field = new_sim.unlensed_field
    new_data = new_sim.data

    #Call the wiener filter with the field_start initial guess and data difference
    #between new and old simulations as the data term. We also use the current phi 
    #and covariance matrices from our sampling algorithms
    data_diff = data_set.data - new_data
    change_in_field = wiener_filter(field_start, args["phi"], data_diff, 
                                args["field_covariance"], args["noise_covariance"], 
                                args["mask"], args["beam"])

    #Return the new simulated unlensed field plus the wiener filter contribution
    return new_field + change_in_field

#sample the lensing potential phi
@jax.jit
def gibbs_sample_phi(mixed_phi, cphi, nphi, mixing_g, 
                     mixed_field, data, noise_covariance, 
                     field_covariance, mask, beam, rng_key_1, rng_key_2, 
                     mixing_d, a_phi, step, num_burn_in_always_accept):
    
    always_accept = (step < num_burn_in_always_accept)
    mass_matrix = get_mass_matrix(cphi, nphi, mixing_g, a_phi)

    mixed_phi, delta_h, accept = hmc_step(mixed_phi, always_accept, mixed_phi.nside, mass_matrix, 
                                          mixed_field, data, noise_covariance, 
                                          cphi, field_covariance, mask, beam, 
                                          mixing_d, mixing_g, a_phi, rng_key_1, rng_key_2)
    return mixed_phi, delta_h, accept

#The mass matrix used in the HMC steps for sampling the lensing potential
def get_mass_matrix(cphi, nphi, mixing_g, aphi):
    return pinv(mixing_g) * pinv(mixing_g) * (pinv(aphi * cphi) + pinv(nphi))

#Hamiltonian Monte Carlo Step for the lensing potential
def hmc_step(x, always_accept, nside, mass_matrix, 
             mixed_field, data, noise_covariance, 
             phi_covariance, field_covariance, mask, beam, 
             mixing_d, mixing_g, a_phi, rng_key_1, rng_key_2):
    
    p_matrix = field_from_covar_single_key(nside, mass_matrix.scalar_matrix, rng_key_1)
    p = FlatS0(scalar_matrix = jfft.rfft2(p_matrix))
    delta_h, x_test, p = symplectic_integrate(x, p, mixed_field, data, noise_covariance, 
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
                        a_phi, num_steps = 50, step_size = 0.1):
    
    #Get the mixed phi gradient at a certain mixed_phi value with all other
    #inputs held constant
    def mixed_grad_phi_partial(mixed_phi):
        return mixed_grad_phi_logpdf(mixed_field, mixed_phi, data, noise_covariance, 
                                     phi_covariance, field_covariance, 
                                     mask, beam, mixing_d, mixing_g)
    
    #Get the logpdf at a certain mixed_phi value with all other inputs held constant
    def logpdf_partial(mixed_phi):
        #first unmix the fields
        field, phi = unmix(mixed_field, mixed_phi, mixing_d, mixing_g)
        #then call logpdf in the unmixed parametrization
        return logpdf(field, phi, data, noise_covariance, phi_covariance, 
                      field_covariance, mask, beam, a_phi)
    
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
def gibbs_sample_theta(a_phi_range, mixed_field, mixed_phi, mixing_d, mixing_g,
                       data, noise_covariance, phi_covariance, field_covariance, 
                       mask, beam, rng_key):

    #Get the logpdf at a certain a_phi value with all other inputs held constant
    def logpdf_partial(a_phi):
        #first unmix the fields
        field, phi = unmix(mixed_field, mixed_phi, mixing_d, mixing_g)
        #then call logpdf in the unmixed parametrization
        return logpdf(field, phi, data, noise_covariance, phi_covariance, 
                      field_covariance, mask, beam, a_phi)
    
    #Evaluate the logpdf at all possible A_phi values with other inputs held constant
    logpdf_values = jax.vmap(logpdf_partial)(a_phi_range)
    a_phi = grid_and_sample(logpdf_values, a_phi_range, rng_key)
    return a_phi

#sample single parameter "theta" via inverse transform
def grid_and_sample(logpdf_values, theta_values, rng_key):

    #do some prep work
    theta_max = theta_values[-1]
    logpdf_values = logpdf_values - jnp.max(logpdf_values)

    #interpolate the logpdf values using a custom loess implementation
    logpdf_values = loess(theta_values, logpdf_values)
    
    #Compute the cumulative distribution function
    def cdf(theta):
      idx = jnp.searchsorted(theta_values, theta)
      mask = jnp.arange(len(theta_values)) < idx
      #NOTE we must do "masking" like this since we can't slice
      #with a dynamic value in JIT-ted code...
      y = jnp.where(mask, jnp.exp(logpdf_values), 0.0)
      return jax.scipy.integrate.trapezoid(y, theta_values)
    
    #normalize (normalizing logpdfs means subtracting the full log area of the PDF)
    log_norm = jnp.log(cdf(theta_max))
    logpdf_values = logpdf_values - log_norm

    #bracket the sample by conservatively saying we won't draw a
    #point with loglikelihood < -1000 of the peak
    min_idx = jnp.argmax(logpdf_values > (jnp.max(logpdf_values)-1000))
    max_idx = len(logpdf_values) - 1 - jnp.argmax(jnp.flip(logpdf_values > (jnp.max(logpdf_values)-1000)))
    theta_min_prime = theta_values[min_idx]
    theta_max_prime = theta_values[max_idx]

    #Use inverse transform sampling to find a sampled theta
    random_number = jax.random.uniform(rng_key)
    bracket_check = (cdf(theta_min_prime) - random_number) * (cdf(theta_max_prime) - random_number) >= 0
    def early_return(_):
        return jax.lax.cond(logpdf_values[0] < logpdf_values[-1],
                            lambda: theta_min_prime,
                            lambda: theta_max_prime)
    def root_find(_):
        sol = optx.root_find(
            lambda x, args: cdf(x) - random_number,
            optx.Bisection(rtol = 1e-3, atol = 1e-3),
            jnp.array(theta_min_prime + theta_max_prime)/2,
            options = dict(lower = theta_min_prime, upper = theta_max_prime),
            max_steps = 16_384) #Default number of max steps is 256 here we use 2^14
        return sol.value
    
    best_theta = jax.lax.cond(bracket_check, early_return, root_find, None)
    return best_theta

#TODO get this JIT-ted so it is not horrendously slow...
#main entry point for sampling algorithm
def sample_joint(data_set, num_chains = 1, num_burn_in = 10, 
                 iters_per_chain = 60, num_burn_in_always_accept = 0):

    #Initialize the starting guess for Aphi to be 1
    cosmo_params = {}
    #NOTE using a python int here would re-trigger JIT-compilation and 
    #lead to performance loss in terms of run-time
    cosmo_params["a_phi"] = jnp.float64(1.0)

    #Store other necessary data in the args object
    args = {}
    args["noise_covariance"] = data_set.noise_covariance
    args["phi_covariance"] = data_set.phi_covariance
    args["field_covariance"] = data_set.field_covariance
    args["quadratic_estimate"] = data_set.quadratic_estimate
    args["mask"] = data_set.mask
    args["beam"] = data_set.beam
    mixing_d = data_set.mixing_d
    mixing_g = data_set.mixing_g

    a_phi_range = jnp.linspace(0.5, 1.5, 30)
    a_phi_distribution = []

    #We need one random number for the f step, 2 for the phi step 
    #and one for each theta step on every iteration
    num_keys = 4 * iters_per_chain
    master_key = jax.random.PRNGKey(67) #67 creates reproduceable results
    rng_keys = jax.random.split(master_key, num_keys)

    start_time = time.time()
    #0. Do we actually need to initialize (f, phi) here to (fJ, phiJ)
    #using MAP_joint() or should they actually be initialized to zero?
    temp_field, phi = map_joint(data_set, num_steps = 30, constant_step = True)
    end_time = time.time()
    args["phi"] = phi

    for iter in range(iters_per_chain):

        #1. sample the temperature field
        rng_key = rng_keys[iter]
        temp_field = gibbs_sample_f(temp_field, data_set, cosmo_params, args, rng_key)

        #2. mix the fields
        phi = args["phi"]
        mixed_temp, mixed_phi = mix(temp_field, phi, mixing_d, mixing_g)

        #3. sample the lensing potential phi
        rng_key_1 = rng_keys[iter + 1]
        rng_key_2 = rng_keys[iter + 2]
        mixed_phi, _, _ = gibbs_sample_phi(mixed_phi, args["phi_covariance"], 
                            args["quadratic_estimate"], mixed_temp, data_set.data, args["noise_covariance"], 
                            args["field_covariance"], args["mask"], args["beam"], rng_key_1, rng_key_2,
                            mixing_d, cosmo_params["a_phi"], iter, num_burn_in_always_accept)
        
        #4. sample your cosmo parameters (just A_phi at the moment)
        rng_key = rng_keys[iter + 3]
        a_phi = gibbs_sample_theta(a_phi_range, mixed_temp, mixed_phi, mixing_d, mixing_g,
                                  data_set.data, args["noise_covariance"], args["phi_covariance"], 
                                  args["field_covariance"], args["mask"], args["beam"], rng_key)
        cosmo_params["a_phi"] = a_phi
        a_phi_distribution.append(a_phi)

        #5. unmix the fields
        temp_field, phi = unmix(mixed_temp, mixed_phi, mixing_d, mixing_g)
        args["phi"] = phi

    #Return your learned distribution at the end of the chain    
    return np.array(a_phi_distribution)

if __name__ == "__main__":
    #TODO now that we are using CAMB again we need to implement the G matrix for real inside of it
    data_set = load_sim(nside = 256, theta_pix = 2.5, pol = "I", master_seed = 67)
    a_phi_distribution = sample_joint(data_set, num_chains = 1, num_burn_in = 10, 
                                      iters_per_chain = 60, num_burn_in_always_accept = 10)
    print("Done!")

