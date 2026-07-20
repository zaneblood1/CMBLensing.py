#Prototype: vmap-across-thetas variant of sample_lcdm_legacy.sample_joint.
#
#The sequential loop in sample_lcdm_legacy samples each cosmological parameter
#conditioned on the JUST-updated values of the others (a Gauss-Seidel Gibbs sweep).
#This prototype instead samples every enabled parameter SIMULTANEOUSLY, each
#conditioned on the same pre-sweep `current_params` (a Jacobi sweep). That lets the
#expensive per-parameter grid evaluation be batched into a single XLA program via
#jax.vmap instead of a Python for-loop, which is far cheaper than spawning worker
#processes and doesn't oversubscribe the CPU cores.
#
#STATISTICAL CAVEAT: the Jacobi update is NOT the same Markov chain as the
#sequential sweep. For correlated parameters it can mix worse or bias the chain,
#so this is meant for A/B convergence testing against sample_lcdm_legacy, not as a
#drop-in correctness-preserving replacement. When only one parameter is enabled in
#`should_sample`, the two are identical (a one-element sweep has no ordering).
from cmb_lensing.sample_lcdm_legacy import *
import random

#Pure-JAX core extracted from gibbs_sample_theta: everything EXCEPT the final
#host-side grid_and_sample call. Given a parameter to vary (theta_key_idx) and the
#grid of values to try (theta_range), returns the log pdf evaluated on that grid,
#shape (N,). Written to be vmap-able over (theta_key_idx, theta_range) with all
#other arguments shared (the Jacobi conditioning on a single `current_params`).
def theta_logpdf_grid(theta_key_idx, theta_range,
                      current_params, emu_params, model_tt, model_pp,
                      nside, pix_width, theta_pix, ell_grid,
                      mixed_temp_matrix, mixed_phi_matrix, data_matrix,
                      cphi_fid, qe_scalar, cn_scalar, mask_matrix, beam_matrix,
                      fourier_weights, tt_norm, pp_norm):

    (tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean) = tt_norm
    (pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean) = pp_norm

    #reconstruct Flax structs from raw arrays for stable pytree tracing
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

    #build the (N, 5) parameter batch: tile current_params and overwrite the
    #sampled column with theta_range. Done with a one-hot where() rather than a
    #dynamic-index .at[:, idx].set(...) so it batches cleanly under an outer vmap
    #(no scatter with a batched column index).
    N = theta_range.shape[0]
    onehot = (jnp.arange(current_params.shape[0]) == theta_key_idx)
    tiled = jnp.tile(current_params, (N, 1))
    params_batch = jnp.where(onehot[None, :], theta_range[:, None], tiled)

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
        cl_tt = cl_tt_batch[i]
        cl_pp = cl_pp_batch[i]
        emu_ells = jnp.arange(2, EMULATOR_MAX_ELL + 1)

        cf = covar_matrix_from_cls(nside, pix_width, ell_grid, emu_ells,
                                   cl_tt, origin_value = 0)
        cphi = covar_matrix_from_cls(nside, pix_width, ell_grid, emu_ells,
                                     cl_pp, origin_value = 0)

        g = get_g_matrix_lcdm(cphi_fid, cphi, qe_scalar)
        d = get_d_tt_matrix(cf, jnp.zeros_like(cf), cn_scalar, 1, 1)

        return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                            noise_covariance, _op(cphi), _op(cf),
                            mask, beam, _op(g), _op(d))

    return jax.vmap(single_logpdf)(jnp.arange(N))

#Sample every enabled parameter at once. Returns a list of (theta_name, sampled_value)
#in the order the names appear in `sampled_thetas`. The grid log pdfs for all K enabled
#parameters are computed in a single vmapped XLA program; the K host-side inverse-CDF
#draws (grid_and_sample -> scipy) stay in a Python loop since they are cheap and not
#JAX-traceable.
def gibbs_sample_thetas_vmap(sampled_thetas, param_ranges, param_vals,
                             mixed_temp, mixed_phi, data_field,
                             current_params, emu_params, model_tt, model_pp,
                             args, tt_norm, pp_norm, keys,
                             over_relaxation_num_samps = -1):

    #stack the per-parameter grids into (K, N) and their column indices into (K,)
    theta_idxs = jnp.array([PARAM_INDEX[t] for t in sampled_thetas])
    theta_ranges = jnp.stack([param_ranges[t] for t in sampled_thetas])
    theta_olds = [param_vals[t][-1] for t in sampled_thetas]

    #closure binds all shared args; vmap maps only over (theta_key_idx, theta_range).
    #Capturing current_params/mixed_* as constants IS the Jacobi conditioning: every
    #parameter sees the same pre-sweep state.
    def grid_closure(theta_key_idx, theta_range):
        return theta_logpdf_grid(theta_key_idx, theta_range,
                                 current_params, emu_params, model_tt, model_pp,
                                 args["nside"], args["pix_width"], data_field.theta_pix,
                                 args["ell_grid"],
                                 mixed_temp.scalar_matrix, mixed_phi.scalar_matrix,
                                 data_field.scalar_matrix,
                                 args["cphi_fid"], args["quadratic_estimate"].scalar_matrix,
                                 args["noise_covariance"].scalar_matrix,
                                 args["mask"].scalar_matrix, args["beam"].scalar_matrix,
                                 data_field.fourier_weights, tt_norm, pp_norm)

    #(K, N) log pdfs, one row per enabled parameter, all in one XLA dispatch
    all_logpdfs = jax.vmap(grid_closure)(theta_idxs, theta_ranges)

    #host-side inverse-CDF draw per parameter (scipy quad/brentq/loess)
    results = []
    for j, theta in enumerate(sampled_thetas):
        theta_val = grid_and_sample(all_logpdfs[j], theta_ranges[j], keys[j],
                                    theta_olds[j], over_relaxation_num_samps)
        results.append((theta, theta_val))
    return results

#Drop-in variant of sample_lcdm_legacy.sample_joint using the Jacobi/vmap theta sweep.
#Only the "sample cosmo parameters" block differs from the legacy version; the f/phi
#Gibbs steps and all setup reuse the legacy helpers unchanged.
def sample_joint(data_set, param_init, param_ranges, should_sample, noise_level, iters_per_chain = 500,
                 num_burn_in_fix_theta = 100, num_burn_in_always_accept = 0, seed = None, map = None,
                 phi_start = "MAP", f_start = "MAP", over_relaxation_num_samps = -1, lmax = 17_000):

    emulator = cambemul.loademul("/home/zane-blood/Desktop/cmb_lensing/camb_emulator")
    #emulator = cambemul.loademul("/resnick/groups/wugroup/zblood/cmb_lensing/camb_emulator")
    (predict_tt, predict_pp, emu_params, emu_meta, model_tt, model_pp,
     tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean,
     pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean) = prepare_emulator_jax(emulator)

    #bundle the emulator normalization constants for the vmapped grid core
    tt_norm = (tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean)
    pp_norm = (pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean)

    param_vals = {}
    for theta, theta_val in param_init.items():
        param_vals[theta] = [theta_val]
    current_params = jnp.array([param_init[k] for k in PARAM_ORDER], dtype = jnp.float64)

    args = {}
    args = add_metadata_to_args(args, data_set, lmax)
    args = add_starting_matrices_to_args(args, data_set, noise_level,
                                         param_init, current_params,
                                         predict_tt, predict_pp, emu_params)

    data_set = set_initial_ds_conditions(data_set, args)
    data_field = data_set.data

    sub_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    temp_field, phi = get_starting_f_and_phi(f_start, phi_start, data_set, args, sub_key)
    zeroes = 0*temp_field

    start_time = time.time()
    for iter in range(1, iters_per_chain + 1):

        #1. sample the temperature field
        rng_key, sub_key = jax.random.split(sub_key)
        temp_field = gibbs_sample_f(zeroes, data_field, phi, args, rng_key)

        #2. mix the fields
        mixed_temp, mixed_phi = mix(temp_field, phi, args["mixing_d"], args["mixing_g"])

        #3. sample the lensing potential phi
        rng_key, sub_key = jax.random.split(sub_key)
        mixed_phi, _, _ = gibbs_sample_phi(mixed_phi, mixed_temp, data_field, rng_key,
                                           args, iter, num_burn_in_always_accept)

        #4. sample your cosmo parameters -- Jacobi sweep: every enabled parameter is
        #sampled from its conditional at the SAME pre-sweep current_params, so there is
        #no ordering and no per-parameter update in between (that is the whole point of
        #being able to vmap). current_params is updated once, after all draws.
        if iter >= num_burn_in_fix_theta:
            sampled_thetas = [t for t in param_ranges if should_sample[t]]

            if sampled_thetas:
                #one rng key per enabled parameter (order-independent, no shuffle needed)
                keys = jax.random.split(sub_key, len(sampled_thetas) + 1)
                sub_key = keys[0]
                theta_keys = keys[1:]

                #start_time = time.time()
                results = gibbs_sample_thetas_vmap(sampled_thetas, param_ranges, param_vals,
                                                   mixed_temp, mixed_phi, data_field,
                                                   current_params, emu_params, model_tt, model_pp,
                                                   args, tt_norm, pp_norm, theta_keys,
                                                   over_relaxation_num_samps = over_relaxation_num_samps)
                #end_time = time.time()
                #print(f"sample {len(sampled_thetas)} thetas (vmap) time = {end_time - start_time}")

                #apply all draws simultaneously (Jacobi update)
                for theta, theta_val in results:
                    param_vals[theta].append(theta_val)
                    current_params = current_params.at[PARAM_INDEX[theta]].set(theta_val)

            ombh2 = (np.array(param_vals["ombh2"]) - 0.022386) / 0.0004006
            omch2 = (np.array(param_vals["omch2"]) - 0.109381) / 0.0059354
            logA = (np.array(param_vals["logA"]) - 3.218387) / 0.1128948
            ns = (np.array(param_vals["ns"]) - 0.959814) / 0.0164744
            theta_MC_100 = (np.array(param_vals["theta_MC_100"]) - 1.031732) / 0.0262018

            plt.figure(figsize = (16, 10))
            plt.plot(ombh2, label = "ombh2", marker = "o")
            plt.plot(omch2, label = "omch2", marker = "o")
            plt.plot(logA, label = "logA", marker = "o")
            plt.plot(ns, label = "ns", marker = "o")
            plt.plot(theta_MC_100, label = "theta_MC_100", marker = "o")
            plt.axhline(0, color = "black", label = "Zero Sigma")
            plt.axhline(1, color = "grey", label = "+/- 1 Sigma")
            plt.axhline(-1, color = "grey")
            plt.title("(mean - sample)/sigma")
            plt.xlabel("iteration")
            plt.ylabel("standard deviations")
            plt.legend()
            plt.savefig("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/chain_progress.png")
            plt.close()

            #5. recompute mixing and covariance matrices at the newly sampled parameters
            args = update_args_after_sample(current_params, predict_tt, predict_pp,
                                            emu_params, args)

        #6. unmix the fields using the updated G & D matrices
        temp_field, phi = unmix(mixed_temp, mixed_phi, args["mixing_d"], args["mixing_g"])

    #end_time = time.time()
    #total_time = end_time - start_time
    #file_path = "/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/"
    #np.savetxt(file_path + f"sample_lcdm_vmap_time_4_cores.txt", np.array([total_time]))

    return param_vals

if __name__ == "__main__":

    ground_truth_params = {}
    ground_truth_params["ombh2"] = 0.022386
    ground_truth_params["omch2"] = 0.109381
    ground_truth_params["cosmomc_theta"] = 0.01031732
    ground_truth_params["As"] = jnp.exp(3.218387) * 1e-10
    ground_truth_params["ns"] = 0.959814

    nside = 128
    theta_pix = 2.5
    pol = "I"
    master_seed = 16725
    noise_level = 5

    data_set = load_sim(nside, theta_pix, pol, master_seed, **ground_truth_params,
                        uk_arcmin_t = noise_level, r = 0, nt = 0)

    param_init = {}
    param_init["ombh2"] = 0.024389 #+5 sigma from Yuuki's mean for training
    param_init["omch2"] = 0.079704 #-5 sigma from mean
    param_init["theta_MC_100"] = 0.900723 #-5 sigma from mean #NOTE +5 here and -5 for logA seems to break CAMB
    param_init["logA"] = 3.782861 #+5 sigma from mean
    param_init["ns"] = 1.042186 #+5 sigma from mean

    SEARCH_PRECISION = 50
    MAX_BUFFER_FACTOR = 1
    MIN_BUFFER_FACTOR = 1
    param_ranges = {}
    param_ranges["ombh2"] = jnp.linspace(MIN_BUFFER_FACTOR * 0.020413, 0.024389 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)
    param_ranges["omch2"] = jnp.linspace(MIN_BUFFER_FACTOR * 0.079704, 0.155541 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)
    param_ranges["theta_MC_100"] = jnp.linspace(MIN_BUFFER_FACTOR * 0.900723, 1.156063 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)
    param_ranges["logA"] = jnp.linspace(MIN_BUFFER_FACTOR * 2.661635, 3.782861 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)
    param_ranges["ns"] = jnp.linspace(MIN_BUFFER_FACTOR * 0.867143, 1.042186 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)

    #enable several parameters so the vmap sweep actually batches >1 theta
    should_sample = {}
    should_sample["ombh2"] = True
    should_sample["omch2"] = True
    should_sample["theta_MC_100"] = True
    should_sample["logA"] = True
    should_sample["ns"] = True

    #start_time = time.time()
    param_distributions = sample_joint(data_set, param_init, param_ranges, should_sample, noise_level,
                                       iters_per_chain = 10_000, num_burn_in_fix_theta = 200,
                                       over_relaxation_num_samps = 20, seed = 67,
                                       num_burn_in_always_accept = 0, phi_start = "MAP",
                                       f_start = "MAP")
    #end_time = time.time()
    #print(f"total time = {end_time - start_time}")
