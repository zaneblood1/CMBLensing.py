#Prototype: process-parallel Jacobi theta sweep for sample_lcdm.
#
#Motivation (measured on the nside=128 setup, over_relaxation_num_samps=20):
#  - the vmapped log-pdf grid (K thetas x N grid points) costs ~10.8 s / sweep and
#    does NOT get faster with more cores -- XLA-on-CPU will not spread that one
#    process's work across cores, and jitting does not change it (genuine compute).
#  - grid_and_sample (scipy quad/brentq/loess) costs ~1.1 s / theta = ~5.7 s / sweep
#    and is serial, GIL-bound Python.
#Both costs are embarrassingly parallel across the K independent Jacobi thetas
#(every theta is sampled from its conditional at the SAME pre-sweep state), so the
#only way to actually use the extra --cpus-per-task cores is Python-level process
#parallelism: give each theta its own worker process, which runs BOTH its JAX grid
#and its grid_and_sample. That divides the ~16 s theta block by ~K.
#
#Same STATISTICAL CAVEAT as sample_lcdm_vmap: the Jacobi update is a different Markov
#chain than the sequential (Gauss-Seidel) legacy sweep. Use for A/B convergence
#testing, not as a correctness-preserving drop-in. With one enabled parameter the
#two are identical.
#
#THREADS: K worker processes each running XLA can oversubscribe the node. Because a
#single XLA process already refuses to scale this workload past a couple of cores,
#capping each worker to a few threads and running K of them is what gives real
#multicore use. Pass worker_xla_threads (or set XLA_FLAGS/OMP_NUM_THREADS in the
#SLURM script); the cap is applied to the *worker* environment before their JAX
#imports, leaving the parent process uncapped.
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

#--------------------------------------------------------------------------------
#Pure-numpy inverse-CDF sampler. Byte-for-byte the same algorithm as
#sample_lcdm_legacy.grid_and_sample's inner _grid_and_sample_internal, but takes
#pre-drawn uniform randoms (so a worker process needs no JAX for the RNG) instead of
#splitting jax PRNGKeys internally.
def grid_and_sample_numpy(logpdf_values, theta_values, uniforms, theta_old,
                          over_relaxation_num_samps = -1):
    from cmb_lensing.util import loess

    xs = np.asarray(theta_values, dtype = np.float64)
    logpdfs = np.asarray(logpdf_values, dtype = np.float64)

    #trim leading/trailing zero-probability regions
    finite = np.isfinite(logpdfs)
    first_finite = int(np.argmax(finite))
    last_finite = len(finite) - 1 - int(np.argmax(finite[::-1]))
    xs = xs[first_finite:last_finite + 1]
    logpdfs = logpdfs[first_finite:last_finite + 1]

    #shift for numerical stability then smooth
    logpdfs = logpdfs - np.max(logpdfs)
    xmin, xmax = float(xs[0]), float(xs[-1])
    interp_logpdfs = np.array(loess(xs, logpdfs, span = 0.25), dtype = np.float64)

    def interp_logpdf(x):
        return float(np.interp(x, xs, interp_logpdfs))

    def nan2zero(x):
        return 0.0 if np.isnan(x) else x

    #normalize the PDF via adaptive quadrature
    def cdf(x):
        result, _ = quad(lambda t: nan2zero(np.exp(interp_logpdf(t))),
                         xmin, float(x), limit = 500, epsrel = 1e-4)
        return result

    logA = nan2zero(np.log(cdf(xmax)))
    interp_logpdfs -= logA
    logpdfs = interp_logpdfs

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

    if over_relaxation_num_samps != -1:
        sample_set = []
        for i in range(over_relaxation_num_samps):
            r = float(uniforms[i])
            cdf_lo = cdf_norm(xmin_prime)
            cdf_hi = cdf_norm(xmax_prime)
            if (cdf_lo - r) * (cdf_hi - r) >= 0:
                sampled_theta = xmin_prime if logpdfs[0] > logpdfs[-1] else xmax_prime
            else:
                sampled_theta = brentq(lambda x: cdf_norm(x) - r,
                                       xmin_prime, xmax_prime,
                                       xtol = (xmax - xmin) * 1e-4)
            sample_set.append(sampled_theta)

        sorted_set = np.sort(np.array(sample_set))
        index = int(np.searchsorted(sorted_set, theta_old, side = "right"))
        chosen_index = over_relaxation_num_samps - index - 1
        chosen_index = min(len(sorted_set) - 1, max(0, chosen_index))
        return float(sorted_set[chosen_index])

    r = float(uniforms[0])
    cdf_lo = cdf_norm(xmin_prime)
    cdf_hi = cdf_norm(xmax_prime)
    if (cdf_lo - r) * (cdf_hi - r) >= 0:
        return float(xmin_prime if logpdfs[0] > logpdfs[-1] else xmax_prime)
    return float(brentq(lambda x: cdf_norm(x) - r,
                        xmin_prime, xmax_prime,
                        xtol = (xmax - xmin) * 1e-4))

#--------------------------------------------------------------------------------
#Worker-process state. Populated once per worker by _worker_init (spawn), reused
#across every sweep. Holds the emulator/models and the sweep-invariant "static"
#arrays (data, covariances, mask, beam, geometry) plus the jitted single-theta grid
#function, so each per-sweep task only ships the tiny dynamic state.
_W = {}

def _worker_init(emulator_path, static):
    #Imports are done here (not at module top) so the parent process that owns the
    #pool does not pay the JAX/emulator import cost, and so that any XLA thread cap
    #inherited via the environment is already in effect for this child's JAX import.
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import cambemul
    from cmb_lensing.sample_lcdm_legacy import prepare_emulator_jax
    from cmb_lensing.sample_lcdm_vmap import theta_logpdf_grid

    (predict_tt, predict_pp, emu_params, emu_meta, model_tt, model_pp,
     tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean,
     pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean) = \
        prepare_emulator_jax(cambemul.loademul(emulator_path))

    tt_norm = (tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean)
    pp_norm = (pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean)

    #move the sweep-invariant arrays onto the device once
    S = {}
    for k, v in static.items():
        S[k] = jnp.asarray(v) if isinstance(v, np.ndarray) else v

    #single-theta grid closure: static arrays/emulator captured as constants, only
    #the per-sweep dynamic state is a traced argument, so it compiles exactly once.
    def _grid(theta_idx, theta_range, current_params, mixed_temp_m, mixed_phi_m):
        return theta_logpdf_grid(theta_idx, theta_range, current_params,
                                 emu_params, model_tt, model_pp,
                                 S["nside"], S["pix_width"], S["theta_pix"], S["ell_grid"],
                                 mixed_temp_m, mixed_phi_m, S["data"],
                                 S["cphi_fid"], S["qe"], S["cn"], S["mask"], S["beam"],
                                 S["fourier_weights"], tt_norm, pp_norm)

    _W["jnp"] = jnp
    _W["grid_fn"] = jax.jit(_grid)

#One theta, end to end, inside a worker: JAX log-pdf grid then the numpy inverse-CDF
#draw. Returns a plain python float.
def _worker_sample_theta(payload):
    (theta_idx, theta_range, theta_old, current_params,
     mixed_temp_m, mixed_phi_m, uniforms, orns) = payload
    jnp = _W["jnp"]
    logpdf = _W["grid_fn"](jnp.array(theta_idx),
                           jnp.asarray(theta_range),
                           jnp.asarray(current_params),
                           jnp.asarray(mixed_temp_m),
                           jnp.asarray(mixed_phi_m))
    logpdf = np.asarray(logpdf)
    return grid_and_sample_numpy(logpdf, theta_range, uniforms, theta_old, orns)

#--------------------------------------------------------------------------------
#Manages the persistent worker pool and issues one Jacobi theta sweep per call.
class ParallelThetaSampler:
    def __init__(self, sampled_thetas, param_ranges, emulator_path,
                 static, over_relaxation_num_samps = -1, worker_xla_threads = None,
                 rng = None):
        self.sampled_thetas = list(sampled_thetas)
        self.theta_idx = {t: i for i, t in enumerate(sampled_thetas)}
        #resolve each theta's column index in the 5-vector via the emulator ordering
        from cmb_lensing.sample_lcdm_legacy import PARAM_INDEX
        self.param_index = PARAM_INDEX
        self.param_ranges = {t: np.asarray(param_ranges[t], dtype = np.float64)
                             for t in sampled_thetas}
        self.orns = over_relaxation_num_samps
        self.n_uniforms = max(1, over_relaxation_num_samps)
        self.rng = rng if rng is not None else np.random.default_rng()

        #Cap worker XLA/BLAS threads via the environment BEFORE the pool spawns, so
        #children inherit it at import time. The parent's JAX is already initialized
        #and unaffected.
        if worker_xla_threads is not None:
            os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                f" --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={worker_xla_threads}").strip()
            os.environ["OMP_NUM_THREADS"] = str(worker_xla_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(worker_xla_threads)

        ctx = mp.get_context("spawn")
        self.pool = ProcessPoolExecutor(max_workers = len(sampled_thetas),
                                        mp_context = ctx,
                                        initializer = _worker_init,
                                        initargs = (emulator_path, static))

    #Sample every enabled parameter in parallel, each conditioned on the same
    #pre-sweep state. Returns [(theta_name, sampled_value), ...].
    def sample(self, current_params, mixed_temp_matrix, mixed_phi_matrix, theta_olds):
        cp = np.asarray(current_params, dtype = np.float64)
        mt = np.asarray(mixed_temp_matrix)
        mp_ = np.asarray(mixed_phi_matrix)

        payloads = []
        for t in self.sampled_thetas:
            uniforms = self.rng.random(self.n_uniforms)
            payloads.append((self.param_index[t], self.param_ranges[t],
                             float(theta_olds[t]), cp, mt, mp_, uniforms, self.orns))

        #map preserves order, so results line up with self.sampled_thetas
        results = list(self.pool.map(_worker_sample_theta, payloads))
        return list(zip(self.sampled_thetas, results))

    def close(self):
        self.pool.shutdown(wait = True)


#--------------------------------------------------------------------------------
#Drop-in sample_joint using the process-parallel Jacobi theta sweep. The f/phi Gibbs
#steps and all setup reuse the sample_lcdm_legacy / sample_lcdm_vmap helpers. Only
#the "sample cosmo parameters" block differs.
def sample_joint(data_set, param_init, param_ranges, should_sample, noise_level,
                 iters_per_chain = 500, num_burn_in_fix_theta = 100,
                 num_burn_in_always_accept = 0, seed = None, map = None,
                 phi_start = "MAP", f_start = "MAP", over_relaxation_num_samps = -1,
                 lmax = 17_000,
                 emulator_path = "/resnick/groups/wugroup/zblood/cmb_lensing/camb_emulator",
                 worker_xla_threads = None):

    #import the JAX-heavy helpers lazily so importing this module (e.g. by the spawned
    #workers to find _worker_init) stays cheap
    import time
    import jax
    import jax.numpy as jnp
    from cmb_lensing.sample_lcdm_legacy import (
        prepare_emulator_jax, PARAM_ORDER, PARAM_INDEX,
        add_metadata_to_args, add_starting_matrices_to_args, set_initial_ds_conditions,
        get_starting_f_and_phi, gibbs_sample_f, gibbs_sample_phi, update_args_after_sample)
    from cmb_lensing.mixing import mix, unmix

    (predict_tt, predict_pp, emu_params, emu_meta, model_tt, model_pp,
     *_norm) = prepare_emulator_jax(cambemul_load(emulator_path))

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
    zeroes = 0 * temp_field

    #build the persistent worker pool once. The static (sweep-invariant) arrays are
    #captured now: data, fiducial phi covariance, quadratic-estimate norm, noise
    #covariance, mask, beam, and geometry never change across sweeps.
    sampled_thetas = [t for t in param_ranges if should_sample[t]]
    sampler = None
    if sampled_thetas:
        static = {
            "nside": int(args["nside"]),
            "pix_width": float(args["pix_width"]),
            "theta_pix": float(data_field.theta_pix),
            "ell_grid": np.asarray(args["ell_grid"]),
            "data": np.asarray(data_field.scalar_matrix),
            "cphi_fid": np.asarray(args["cphi_fid"]),
            "qe": np.asarray(args["quadratic_estimate"].scalar_matrix),
            "cn": np.asarray(args["noise_covariance"].scalar_matrix),
            "mask": np.asarray(args["mask"].scalar_matrix),
            "beam": np.asarray(args["beam"].scalar_matrix),
            "fourier_weights": np.asarray(data_field.fourier_weights),
        }
        sampler = ParallelThetaSampler(sampled_thetas, param_ranges, emulator_path,
                                       static, over_relaxation_num_samps,
                                       worker_xla_threads)

    start_time = time.time()
    try:
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

            #4. sample cosmo parameters -- process-parallel Jacobi sweep
            if iter >= num_burn_in_fix_theta and sampler is not None:
                theta_olds = {t: param_vals[t][-1] for t in sampled_thetas}

                t0 = time.time()
                results = sampler.sample(current_params,
                                         mixed_temp.scalar_matrix,
                                         mixed_phi.scalar_matrix,
                                         theta_olds)
                print(f"sample {len(sampled_thetas)} thetas (parallel) time = {time.time() - t0}")

                #apply all draws simultaneously (Jacobi update)
                for theta, theta_val in results:
                    param_vals[theta].append(theta_val)
                    current_params = current_params.at[PARAM_INDEX[theta]].set(theta_val)

                #5. recompute mixing/covariance matrices at the newly sampled parameters
                args = update_args_after_sample(current_params, predict_tt, predict_pp,
                                                emu_params, args)

            #6. unmix the fields using the updated G & D matrices
            temp_field, phi = unmix(mixed_temp, mixed_phi, args["mixing_d"], args["mixing_g"])
    finally:
        if sampler is not None:
            sampler.close()

    end_time = time.time()
    total_time = end_time - start_time
    file_path = "/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/"
    np.savetxt(file_path + f"sample_lcdm_parallel_time_16_cores_3_workers.txt", np.array([total_time]))
    return param_vals


#tiny indirection so the emulator import lives next to its use and is easy to stub
def cambemul_load(path):
    import cambemul
    return cambemul.loademul(path)


if __name__ == "__main__":
    import jax.numpy as jnp
    from cmb_lensing.simulate import load_sim

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
    param_init["ombh2"] = 0.024389
    param_init["omch2"] = 0.079704
    param_init["theta_MC_100"] = 0.900723
    param_init["logA"] = 3.782861
    param_init["ns"] = 1.042186

    SEARCH_PRECISION = 50
    param_ranges = {}
    param_ranges["ombh2"] = jnp.linspace(0.020413, 0.024389, SEARCH_PRECISION)
    param_ranges["omch2"] = jnp.linspace(0.079704, 0.155541, SEARCH_PRECISION)
    param_ranges["theta_MC_100"] = jnp.linspace(0.900723, 1.156063, SEARCH_PRECISION)
    param_ranges["logA"] = jnp.linspace(2.661635, 3.782861, SEARCH_PRECISION)
    param_ranges["ns"] = jnp.linspace(0.867143, 1.042186, SEARCH_PRECISION)

    should_sample = {}
    should_sample["ombh2"] = True
    should_sample["omch2"] = True
    should_sample["theta_MC_100"] = True
    should_sample["logA"] = True
    should_sample["ns"] = True

    param_distributions = sample_joint(data_set, param_init, param_ranges, should_sample,
                                       noise_level, iters_per_chain = 20,
                                       num_burn_in_fix_theta = 0,
                                       over_relaxation_num_samps = 20,
                                       phi_start = "MAP", f_start = "MAP",
                                       worker_xla_threads = 3)
