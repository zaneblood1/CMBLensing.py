from cmb_lensing.simulate import *
from cmb_lensing.wiener_filter import *
from cmb_lensing.util import *
from cmb_lensing.map_joint import *
from cmb_lensing.mixing import *
from cmb_lensing.constants import *
from cmb_lensing.mode_diagnostics import PhiModeRecorder, analyze_phi_modes
from cmb_lensing.hvp_mass_matrix import (build_hvp_mass_matrix, compare_to_analytic,
                                         build_hvp_correction_factor, apply_correction)
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import cambemul
from cambemul.emulator import build_model
import os
import random
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
                                args["mask"], args["beam"], maxiter = 1_000_000, tol = 1e-5)

    #Return the new simulated unlensed field plus the wiener filter contribution
    return new_field + delta_field

#sample the lensing potential phi
@jax.jit
def gibbs_sample_phi(mixed_phi, mixed_temp, data_field, rng_key,
                     args, iter, num_burn_in_always_accept, mass_matrix = None):

    always_accept = (iter < num_burn_in_always_accept)
    #use a precomputed metric (e.g. the HVP-diagonal mass matrix) when supplied, else the
    #analytic approximation M = pinv(G)^2 (pinv(Cphi) + pinv(Nphi))
    if mass_matrix is None:
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
                        num_steps = 30, step_size = 0.033333333):
    
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

#continuous peak of a gridded log pdf via the cubic-spline vertex (falls back to the best
#node when too few finite samples remain for a spline). used to center the adaptive
#recenter pass so the window follows the true between-node peak, not a grid node
def _continuous_peak(theta_grid, logpdf_values):
    xs = np.asarray(theta_grid, dtype = np.float64)
    ys = np.asarray(logpdf_values, dtype = np.float64)
    finite = np.isfinite(ys)
    if not finite.any():
        return float(xs[len(xs) // 2])
    first = int(np.argmax(finite))
    last = len(finite) - 1 - int(np.argmax(finite[::-1]))
    xs = xs[first:last + 1]
    ys = ys[first:last + 1] - np.max(ys[first:last + 1])
    if len(xs) >= 4:
        spline = CubicSpline(xs, ys)
        crit = spline.derivative().roots(extrapolate = False)
        candidates = np.concatenate([crit, [xs[0], xs[-1]]])
        return float(candidates[np.argmax(spline(candidates))])
    return float(xs[np.argmax(ys)])

#symmetric Gaussian random-walk Metropolis-within-Gibbs step for a single cosmological
#parameter. this is the alternative to the grid + inverse-CDF path in gibbs_sample_theta:
#instead of reconstructing the whole 1D conditional and drawing an independent sample from
#it (which lets the adaptive recenter chase a spurious far mode when f/phi are themselves
#sampled, producing huge jumps), we propose theta' ~ N(theta_old, proposal_sigma) and
#accept/reject. proposal_sigma is the "never too far from the previous value" knob, and
#unlike truncating the grid this is an EXACT MCMC update for the true conditional.
#eval_logpdf_grid is the same closure the grid path builds inside gibbs_sample_theta, so
#passing a length-2 array evaluates logpdf at both theta_old and the proposal in one batched
#emulator/covariance call. both logpdfs are recomputed every call because f and phi change
#between Gibbs sweeps - no cross-sweep caching is valid.
def metropolis_sample_theta(eval_logpdf_grid, theta_old, lo0, hi0,
                            proposal_sigma, rng_key, accept_history, num_steps = 1):
    if proposal_sigma is None:
        raise ValueError("metropolis sampler requires a proposal_sigma for this parameter")

    theta_current = float(theta_old)
    for _ in range(num_steps):
        rng_key, k_prop, k_acc = jax.random.split(rng_key, 3)
        theta_prop = theta_current + proposal_sigma * float(jax.random.normal(k_prop))

        #reject out-of-bounds proposals BEFORE touching the emulator: keeping the old value
        #is the exact MH treatment of a target with zero density outside the emulator-validated
        #range [lo0, hi0], and it guarantees we never query the emulator out of range. the
        #symmetric Gaussian proposal makes q(a->b) = q(b->a), so no proposal correction term
        if not (lo0 <= theta_prop <= hi0):
            continue

        logpdfs = eval_logpdf_grid(jnp.array([theta_current, theta_prop]))
        delta_h = float(logpdfs[1] - logpdfs[0])

        #accept if log(u) < delta_h (same acceptance test as hmc_step). NaN-safe: a non-finite
        #logpdf makes delta_h nan and (x < nan) is False, so the step is rejected
        if float(jnp.log(jax.random.uniform(k_acc))) < delta_h:
            theta_current = theta_prop
            accept_history.append(1)
        else:
            accept_history.append(0)
        print(f"Accept Rate = {np.sum(np.array(accept_history))/len(accept_history)}")

    return np.asarray(theta_current, dtype = np.float64)

#@partial(jax.jit, static_argnames = ["model_tt", "model_pp", "lmax", "lmax_prime",
#                                     "nside", "pix_width", "theta_pix",
#                                     "over_relaxation_num_samps"])
def gibbs_sample_theta(theta_key_idx, theta_old, theta_range,
                       mixed_temp_matrix, accept_history, mixed_phi_matrix, data_matrix,
                       current_params, emu_params, model_tt,
                       model_pp, rng_key, lmax, lmax_prime, nside, pix_width, theta_pix,
                       ell_grid, ells,
                       cphi_fid, cf_fid, qe_scalar, cn_scalar, mask_matrix, beam_matrix,
                       fourier_weights,
                       tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean,
                       pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean,
                       over_relaxation_num_samps = -1, sampler = "grid",
                       proposal_sigma = None, metropolis_num_steps = 1):
    """Sample a single cosmological parameter. sampler = "grid" (default) reconstructs the
    1D conditional and draws an independent inverse-CDF sample; sampler = "metropolis" takes
    a Gaussian random-walk Metropolis step of scale proposal_sigma around theta_old instead."""

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

    #evaluate the mixed logpdf over an arbitrary theta grid. all the emulator/covariance
    #work is a pure function of the grid, so wrap it once and reuse it for the coarse pass
    #and each adaptive recenter pass below
    def eval_logpdf_grid(theta_grid):
        #batch-predict all grid points at once: build (N, 5) parameter array by tiling
        #current_params and overwriting the sampled column
        M = theta_grid.shape[0]
        params_batch = jnp.tile(current_params, (M, 1))
        params_batch = params_batch.at[:, theta_key_idx].set(theta_grid)

        cl_tt_batch = predict_tt(emu_params, params_batch)
        cl_pp_batch = predict_pp(emu_params, params_batch)

        def single_logpdf(i):
            # cl_tt = interpolate_cls(cl_tt_batch[i], lmax, lmax_prime)
            # cl_pp = interpolate_cls(cl_pp_batch[i], lmax, lmax_prime)
            cl_tt = cl_tt_batch[i]
            cl_pp = cl_pp_batch[i]
            emu_ells = jnp.arange(2, EMULATOR_MAX_ELL + 1)

            cf = covar_matrix_from_cls(nside, pix_width,
                                       ell_grid, emu_ells,
                                       cl_tt, origin_value = 0)
            cphi = covar_matrix_from_cls(nside, pix_width,
                                         ell_grid, emu_ells,
                                         cl_pp, origin_value = 0)

            g = (get_g_matrix_lcdm(cphi_fid, cphi, qe_scalar, cn_scalar))
            #d = get_d_tt_matrix(cf, jnp.zeros_like(cf), cn_scalar, 1, 1)
            d = (get_d_tt_matrix(cf, cf_fid, cn_scalar))

            return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                                noise_covariance, _op(cphi), _op(cf),
                                mask, beam, _op(g), _op(d))

        return jax.vmap(single_logpdf)(jnp.arange(M))

    #adaptive recenter: the conditional peak (std ~1e-4) is far narrower than one node of
    #the initial grid (spacing ~5e-3), so a single fixed grid reconstructs the peak the
    #same wrong way every iteration -> a consistent directional bias that never averages
    #out. re-center a narrower grid on the peak and re-evaluate, so resolution near the
    #peak improves without a huge node count and node placement stays centered (symmetric)
    #each step. clamp windows to the original range so we never query the emulator outside
    #its validated bounds. NUM_REFINE / SHRINK are the tuning knobs: with SHRINK = 0.15 the
    #~0.25-wide theta_MC_100 grid narrows to spacing ~8e-4 then ~1e-4 over two passes
    NUM_REFINE = 2
    SHRINK = 0.1
    N = theta_range.shape[0]
    lo0, hi0 = float(theta_range[0]), float(theta_range[-1])

    #separate Metropolis random-walk path: propose locally around theta_old and accept/reject
    #instead of reconstructing and drawing from the full conditional. proposal_sigma bounds the
    #step size, which suppresses the spurious far jumps the grid path produces once f is sampled
    if sampler == "metropolis":
        return metropolis_sample_theta(eval_logpdf_grid, theta_old, lo0, hi0,
                                       proposal_sigma, rng_key, accept_history,
                                       num_steps = metropolis_num_steps)

    theta_grid = theta_range
    logpdf_values = eval_logpdf_grid(theta_grid)
    for _ in range(NUM_REFINE):
        peak = _continuous_peak(theta_grid, logpdf_values)
        span = (float(theta_grid[-1]) - float(theta_grid[0])) * SHRINK
        lo = max(peak - 0.5 * span, lo0)
        hi = min(peak + 0.5 * span, hi0)
        theta_grid = jnp.linspace(lo, hi, N)
        logpdf_values = eval_logpdf_grid(theta_grid)

    theta_new = grid_and_sample(logpdf_values, theta_grid, rng_key,
                                theta_old, over_relaxation_num_samps)
    return theta_new

#sample single parameter "theta" via inverse CDF
#@partial(jax.jit, static_argnames = ["over_relaxation_num_samps"])
def grid_and_sample(logpdf_values, theta_values, sub_key, theta_old,
                    over_relaxation_num_samps = -1):

    rng_key, sub_key = jax.random.split(sub_key)
    random_number = jax.random.uniform(rng_key)

    def _grid_and_sample_internal(theta_values, logpdf_values, random_number, over_relaxation_num_samps, sub_key):
        xs = np.asarray(theta_values, dtype = np.float64)
        logpdfs = np.asarray(logpdf_values, dtype = np.float64)

        #trim leading/trailing zero-probability regions (matches Julia's findnext/findprev isfinite)
        finite = np.isfinite(logpdfs)
        first_finite = int(np.argmax(finite))
        #::-1 syntax reverses a python array [a, b, c] --> [c, b, a]
        last_finite = len(finite) - 1 - int(np.argmax(finite[::-1]))
        xs = xs[first_finite:last_finite + 1]
        logpdfs = logpdfs[first_finite:last_finite + 1]

        #shift for numerical stability then smooth
        logpdfs = logpdfs - np.max(logpdfs)
        xmin, xmax = float(xs[0]), float(xs[-1])
        interp_logpdfs = np.array(loess(xs, logpdfs, span = 0.25), dtype = np.float64)
        #interp_logpdfs = logpdfs

        #cubic-spline interpolant over the log PDF. linear (np.interp) chords the concave
        #peak between grid nodes and clips its tip, snapping the reconstructed mode to a
        #node and biasing the inverse-CDF draw toward whichever bracketing node is higher.
        #a cubic spline represents the local curvature so the peak sits at its true
        #between-node location. fall back to linear if too few finite nodes remain for a
        #meaningful cubic fit
        def make_log_interpolant(values):
            if len(xs) >= 4:
                spline = CubicSpline(xs, values)
                return lambda x: float(spline(x))
            return lambda x: float(np.interp(x, xs, values))

        interp_logpdf = make_log_interpolant(interp_logpdfs)

        def nan2zero(x):
            return 0.0 if np.isnan(x) else x

        #normalize the PDF via adaptive quadrature (matches Julia's quadgk)
        def cdf(x):
            result, _ = quad(lambda t: nan2zero(np.exp(interp_logpdf(t))),
                             xmin, float(x), limit = 10_000, epsrel = (xmax - xmin) * 1e-5)
            return result

        #normalize the log PDF by its total integral. guard against underflow or a
        #non-finite total (e.g. a razor-thin finite region) which would otherwise leave
        #the CDF un-normalized (cdf_hi != 1) and push nearly every draw into the
        #edge-snap fallback below, pinning the chain against the search-range wall
        total_mass = cdf(xmax)
        if total_mass > 0 and np.isfinite(total_mass):
            logA = np.log(total_mass)
        else:
            logA = 0.0
        interp_logpdfs -= logA
        logpdfs = interp_logpdfs

        #re-create interpolant with the normalized values
        interp_logpdf_norm = make_log_interpolant(interp_logpdfs)

        def cdf_norm(x):
            result, _ = quad(lambda t: nan2zero(np.exp(interp_logpdf_norm(t))),
                             xmin, float(x), limit = 10_000, epsrel = (xmax - xmin) * 1e-5)
            return result

        #bracket the sample by finding where logpdf > peak - 1000
        peak = np.max(logpdfs)
        above = logpdfs > (peak - 1000)
        xmin_prime = float(xs[np.argmax(above)])
        xmax_prime = float(xs[len(xs) - 1 - np.argmax(above[::-1])])

        #invert the normalized CDF for a target uniform value r. clamp r into the
        #achievable range [cdf_lo, cdf_hi] and only return a bracket endpoint for the
        #(post-normalization, negligible) trimmed tails; otherwise root-solve for an
        #interior theta. the old code deterministically snapped to a grid edge whenever
        #r was unbracketed, which pinned the chain against the search-range wall
        cdf_lo = cdf_norm(xmin_prime)
        cdf_hi = cdf_norm(xmax_prime)

        #continuous mode: locate the peak of the interpolant itself, not the best grid
        #node (xs[argmax] can only ever return a linspace value, which is the grid-snap
        #artifact we are removing). use the spline's stationary points when available
        if len(xs) >= 4:
            mode_spline = CubicSpline(xs, interp_logpdfs)
            crit = mode_spline.derivative().roots(extrapolate = False)
            candidates = np.concatenate([crit, [xs[0], xs[-1]]])
            mode = float(candidates[np.argmax(mode_spline(candidates))])
        else:
            mode = float(xs[np.argmax(interp_logpdfs)])
        plt.figure()
        xs_dense = np.linspace(xmin, xmax, 20 * len(xs))
        plt.plot(xs_dense, [interp_logpdf_norm(x) for x in xs_dense])
        plt.plot(xs, interp_logpdfs, ".", markersize = 4)
        plt.axvline(mode, label = f"mode = {mode}")
        plt.legend()
        plt.savefig("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/ombh2_distribution.png")
        plt.close()

        def sample_from_cdf(r):
            if cdf_hi <= cdf_lo:
                #degenerate: no resolvable probability mass between the brackets
                return xmin_prime if logpdfs[0] > logpdfs[-1] else xmax_prime
            if r <= cdf_lo:
                return xmin_prime
            if r >= cdf_hi:
                return xmax_prime
            return brentq(lambda x: cdf_norm(x) - r,
                          xmin_prime, xmax_prime,
                          xtol = (xmax - xmin) * 1e-6)

        if over_relaxation_num_samps != -1:
            #generate a set of samples if over relaxation is enabled and choose
            #theta_new = theta_{K - i + 1} where K is the size of the set of samples
            #and the index "i" is chosen such that theta_i < theta_old < theta_{i+1}
            sample_set = []
            for _ in range(over_relaxation_num_samps):
                rng_key, sub_key = jax.random.split(sub_key)
                random_number = jax.random.uniform(rng_key)
                sampled_theta = sample_from_cdf(float(random_number))
                sample_set.append(sampled_theta)

            #convert to jnp array and sort
            sorted_set = jnp.sort(jnp.array(sample_set))
            index = jnp.searchsorted(sorted_set, theta_old, side = 'right')
            chosen_index = over_relaxation_num_samps - index
            #clamp to avoid out-of-bounds problems
            chosen_index = min(len(sorted_set)-1, max(0, chosen_index))
            chosen_theta = sorted_set[chosen_index]
            return np.array(chosen_theta, dtype = np.float64)
        
        #inverse transform sampling via Brent root-finding (matches Julia's find_zero)
        sampled = sample_from_cdf(float(random_number))
        return np.array(sampled, dtype = np.float64)

    # sampled_theta = jax.pure_callback(
    #     _grid_and_sample_internal,
    #     jax.ShapeDtypeStruct((), jnp.float64),
    #     theta_values, logpdf_values, random_number
    # )
    return _grid_and_sample_internal(theta_values, logpdf_values, random_number, over_relaxation_num_samps, sub_key)

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

#jitted core of the per-iteration covariance/mixing recompute. Splits static shape/emulator
#args (predict_*/nside/pix_width/lmax) from dynamic arrays so the internal control-flow
#(covar_matrix_from_cls / interpolate_cls conds) compiles once and is cached across Gibbs
#iterations, instead of re-tracing eagerly every call (the "<lambda> for pjit" log flood).
#predict_tt/predict_pp are static (hashable closures, created once in prepare_emulator_jax).
@partial(jax.jit, static_argnames = ["predict_tt", "predict_pp", "nside", "pix_width", "lmax", "lmax_prime"])
def _recompute_cosmo_matrices(current_params, emu_params, ell_grid, ells,
                              cphi_fid, cf_fid, qe_scalar, cn_scalar,
                              predict_tt, predict_pp, nside, pix_width, lmax, lmax_prime):
    x = current_params[None, :]
    cl_tt = predict_tt(emu_params, x)[0]
    cl_pp = predict_pp(emu_params, x)[0]

    emul_ells = jnp.arange(2, EMULATOR_MAX_ELL + 1)

    cf = covar_matrix_from_cls(nside, pix_width,
                               ell_grid, emul_ells,
                               cl_tt, origin_value = 0)
    cphi = covar_matrix_from_cls(nside, pix_width,
                                 ell_grid, emul_ells,
                                 cl_pp, origin_value = 0)

    g = (get_g_matrix_lcdm(cphi_fid, cphi, qe_scalar, cn_scalar))
    #d = get_d_tt_matrix(cf, jnp.zeros_like(cf), cn_scalar, 1, 1)
    d = (get_d_tt_matrix(cf, cf_fid, cn_scalar))
    return g, d, cf, cphi

#JIT-able version of get_new_cosmo_matrices using the pure-JAX emulator.
#Thin eager wrapper: unpacks args (mixed static/dynamic) and delegates to the jitted core.
def get_new_cosmo_matrices(current_params, predict_tt, predict_pp, emu_params, args):
    """Compute (g, d, cf, cphi) from a parameter vector using the pure-JAX emulator."""
    return _recompute_cosmo_matrices(current_params, emu_params,
                                     args["ell_grid"], args["ells"],
                                     args["cphi_fid"], args["cf_fid"],
                                     args["quadratic_estimate"].scalar_matrix,
                                     args["noise_covariance"].scalar_matrix,
                                     predict_tt, predict_pp,
                                     args["nside"], args["pix_width"],
                                     args["lmax"], args["lmax_prime"])

#Lighter-weight version of the above method to just compute the field covariance and not D, G, Cphi...
def get_new_cf_matrix(current_params, predict_tt, emu_params, args):
    """Compute (g, d, cf, cphi) from a parameter vector using the pure-JAX emulator."""
    x = current_params[None, :]
    cl_tt = predict_tt(emu_params, x)[0]
    #cl_tt = interpolate_cls(cl_tt, args["lmax"], args["lmax_prime"])
    emu_ells = jnp.arange(2, EMULATOR_MAX_ELL + 1)
    cf = covar_matrix_from_cls(args["nside"], args["pix_width"],
                               args["ell_grid"], emu_ells,
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
    elif f_start == "ZEROES":
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
    args["cf_fid"] = data_set.field_covariance.scalar_matrix

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
    args["mixing_g"] = data_set.mixing_g.replace(scalar_matrix = (g))
    args["mixing_d"] = data_set.mixing_d.replace(scalar_matrix = (d))

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
    args["mixing_g"] = args["mixing_g"].replace(scalar_matrix = (g))
    args["mixing_d"] = args["mixing_d"].replace(scalar_matrix = (d))
    return args

#algorithm to jointly sample cosmological parameters
def sample_joint(data_set, param_init, param_ranges, should_sample, noise_level, 
                 iters_per_chain = 500,
                 num_burn_in_fix_theta = 100, num_burn_in_always_accept = 0, seed = None, map = None,
                 phi_start = "MAP", f_start = "MAP", over_relaxation_num_samps = -1, lmax = 17_000,
                 record_phi_modes = False, mode_diag_dir = "/home/zane-blood/Desktop/cmb_lensing/cmb_lensing",
                 mode_diag_burn_in = 0, use_hvp_mass_matrix = False, hvp_num_probes = 16,
                 hvp_eps = 1e-2, hvp_floor_frac = 0.1, hvp_refresh_every = 1,
                 hvp_warmup_iters = None, theta_samplers = None, proposal_sigmas = None,
                 metropolis_num_steps = 1):

    #theta_samplers / proposal_sigmas are optional per-parameter dicts (keyed by parameter
    #name, like should_sample) selecting the per-parameter theta sampler. theta_samplers maps
    #a name to "grid" (default) or "metropolis"; proposal_sigmas gives the random-walk step
    #scale for any parameter set to "metropolis". Both default to None -> every parameter uses
    #the grid sampler, exactly reproducing the previous behavior.

    #Step 0 mixing diagnostic: records the (unmixed) phi each sweep to localize stuck modes
    #phi_recorder = PhiModeRecorder() if record_phi_modes else None
    #Step 1 mass matrix: freeze the expensive HVP "shape" correction R = HVP/analytic, refreshed
    #every hvp_refresh_every sweeps DURING warmup (while theta migrates from a far init), then
    #frozen. Each sweep M = R * analytic_M(theta) so the cheap analytic part tracks theta.
    #Only post-warmup samples are valid for inference (adaptation must stop before you collect).
    #hvp_R = None
    #hvp_compare_done = False
    #theta only starts migrating at iter >= num_burn_in_fix_theta, so the metric warmup must
    #span the migration that FOLLOWS it, not precede it. Default: refresh for the whole run
    #(fine for the debug/ergodicity test); for final inference set this to freeze R after theta
    #has visibly converged and discard pre-freeze samples.
    #if hvp_warmup_iters is None:
    #    hvp_warmup_iters = iters_per_chain

    #Prepare the JIT-friendly emulator (build models once, extract weights)
    emulator = cambemul.loademul("/resnick/groups/wugroup/zblood/cmb_lensing/camb_emulator")
    #emulator = cambemul.loademul("/home/zane-blood/Desktop/cmb_lensing/camb_emulator")
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
    # if seed is not None:
    #     sub_key = jax.random.PRNGKey(seed)
    # else:
    #     sub_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
    #sub_key = jax.random.PRNGKey(seed)
    sub_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    #choose the starting point for (f, phi) in (f, phi, theta) cosmological parameter space
    #_, phi = get_starting_f_and_phi(f_start, phi_start, data_set, args, sub_key)
    zeroes = 0*data_field #phi_ground
    phi = zeroes #phi_ground
    #phi = phi_ground
    #temp_field = f_ground

    #run the chain for the maximum specified number if iterations
    #start_time = time.time()
    mc_accept_history = []
    phi_accept_history = []
    delta_h_history = []
    for iter in range(1, iters_per_chain + 1):

        #1. sample the temperature field
        #start_time = time.time()
        rng_key, sub_key = jax.random.split(sub_key)
        temp_field = gibbs_sample_f(zeroes, data_field, phi, args, rng_key)
        #data_set = set_initial_ds_conditions(data_set, args)
        #temp_field, _ = map_joint(data_set)
        #end_time = time.time()
        #print(f"sample f time = {end_time- start_time}")
        #temp_field = f_ground

        #2. mix the fields
        #start_time = time.time()
        mixed_temp, mixed_phi = mix(temp_field, phi, args["mixing_d"], args["mixing_g"])
        #end_time = time.time()
        #print(f"mixing time = {end_time - start_time}")

        # #Step 1: HVP metric. Refresh the frozen shape-correction R every hvp_refresh_every
        # #sweeps while theta is still migrating (iter <= hvp_warmup_iters), then freeze it. Each
        # #sweep M = R * analytic_M(current theta), so the cheap analytic factor tracks theta.
        # hvp_mass_matrix = None
        # if use_hvp_mass_matrix:
        #     analytic_mass = get_mass_matrix(args["phi_covariance"], args["quadratic_estimate"],
        #                                     args["mixing_g"])
        #     in_warmup = iter <= hvp_warmup_iters
        #     need_refresh = (hvp_R is None) or (in_warmup and (iter % hvp_refresh_every == 0))
        #     if need_refresh:
        #         rng_key, sub_key = jax.random.split(sub_key)
        #         #one-time before/after comparison plot at the current theta
        #         if not hvp_compare_done:
        #             compare_to_analytic(analytic_mass, mixed_phi, mixed_temp, data_field, args,
        #                                 rng_key, mode_diag_dir, num_probes = hvp_num_probes,
        #                                 eps = hvp_eps, tag = f"hvp_seed_{seed}")
        #             hvp_compare_done = True
        #         hvp_R = build_hvp_correction_factor(analytic_mass, mixed_phi, mixed_temp,
        #                                             data_field, args, rng_key,
        #                                             num_probes = hvp_num_probes, eps = hvp_eps,
        #                                             floor_frac = hvp_floor_frac)
        #         print(f"[hvp] refreshed correction factor R at iter {iter} "
        #               f"(warmup={in_warmup}, R median={float(jnp.median(hvp_R)):.3f})")
        #     hvp_mass_matrix = apply_correction(analytic_mass, hvp_R)

        #3. sample the lensing potential phi
        #start_time = time.time()
        rng_key, sub_key = jax.random.split(sub_key)
        mixed_phi, delta_h, accept = gibbs_sample_phi(mixed_phi, mixed_temp, data_field, rng_key,
                                                      args, iter, num_burn_in_always_accept,
                                                      mass_matrix = None)
        # phi_accept_history.append(accept)
        # delta_h_history.append(delta_h)
        # # # #end_time = time.time()
        # # # #print(f"sample phi time = {end_time - start_time}")
        # print(f"accept history = {np.array(phi_accept_history)}")
        # print(f"delta_h history = {np.array(delta_h_history)}")
        # plt.figure()
        # plt.imshow(jfft.irfft2(mixed_phi.scalar_matrix), cmap = "coolwarm")
        # plt.colorbar()
        # plt.title(f"mixed_phi_iter_{iter}")
        # plt.savefig(f"/home/zane-blood/Desktop/cmb_lensing/mixed_phi.png")
        # plt.close()
        
        #data_set = set_initial_ds_conditions(data_set, args)
        #temp_field, phi = map_joint(data_set)
        #mixed_temp, mixed_phi = mix(f_ground, phi_ground, args["mixing_d"], args["mixing_g"])
        #start_time = time.time()
        #4. sample your cosmo parameters
        if iter >= num_burn_in_fix_theta:
            #start_time = time.time()
            #randomly shuffle the order in which we sample the cosmo parameters each loop
            #(host-side shuffle, avoids the ~7s device dispatch/sync of jax.random.permutation)
            shuffled_items = list(param_ranges.items())
            random.shuffle(shuffled_items)
            #end_time = time.time()
            #print(f"shuffle time = {end_time - start_time}")

            #start_time = time.time()
            for theta, theta_range in shuffled_items:
                if should_sample[theta]:
                    rng_key, sub_key = jax.random.split(sub_key)
                    theta_key_idx = PARAM_INDEX[theta]
                    theta_sampler = "grid" if theta_samplers is None else theta_samplers.get(theta, "grid")
                    theta_proposal_sigma = None if proposal_sigmas is None else proposal_sigmas.get(theta)
                    #larger steps in beginning then clamp to min value for later iterations
                    theta_proposal_sigma = max((0.95**iter)*theta_proposal_sigma, 2e-4)
                    theta_val = gibbs_sample_theta(jnp.array(theta_key_idx), jnp.array(param_vals[theta][-1]),
                                                   theta_range, mixed_temp.scalar_matrix, mc_accept_history,
                                                   mixed_phi.scalar_matrix, data_field.scalar_matrix,
                                                   current_params,
                                                   emu_params, model_tt,
                                                   model_pp, rng_key, args["lmax"], args["lmax_prime"],
                                                   args["nside"], args["pix_width"],
                                                   data_field.theta_pix,
                                                   args["ell_grid"], args["ells"],
                                                   args["cphi_fid"], args["cf_fid"], args["quadratic_estimate"].scalar_matrix,
                                                   args["noise_covariance"].scalar_matrix,
                                                   args["mask"].scalar_matrix, args["beam"].scalar_matrix,
                                                   data_field.fourier_weights,
                                                   tt_x_mean, tt_x_std, tt_t_mean,
                                                   tt_t_std, tt_pca_basis_T, tt_pca_mean,
                                                   pp_x_mean, pp_x_std, pp_t_mean,
                                                   pp_t_std, pp_pca_basis_T, pp_pca_mean,
                                                   over_relaxation_num_samps = over_relaxation_num_samps,
                                                   sampler = theta_sampler,
                                                   proposal_sigma = theta_proposal_sigma,
                                                   metropolis_num_steps = metropolis_num_steps)
                    param_vals[theta].append(theta_val)
                    current_params = current_params.at[theta_key_idx].set(theta_val)

            #5. recompute mixing and covariance matrices using the newly sampled parameter values
            #NOTE this is only necessary if we are past the burn-in phase
            args = update_args_after_sample(current_params, predict_tt, predict_pp,
                                            emu_params, args)
            
        # -------------------------------------------------------- DEBUG --------------------------------------------------------
        #Store the sampled a_phi value to a debug text file...
        ombh2_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/lcdm_chains_v8/ombh2/ombh2_map_{map}_chain_{seed}_history.txt"
        # omch2_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/lcdm_chains_v3/omch2/omch2_map_{map}_history.txt"
        #theta_MC_100_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/lcdm_chains_v5/theta_MC_100/theta_MC_100_map_{map}_chain_{seed}_history.txt"
        #logA_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/lcdm_chains_v5/logA/logA_map_{map}_chain_{seed}_history.txt"
        # ns_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/lcdm_chains_v3/ns/ns_map_{map}_history.txt"
        with open(ombh2_file_path, "a") as file:
            file.write(str(param_vals["ombh2"][-1]) + "\n")
        # with open(omch2_file_path, "a") as file:
        #     file.write(str(param_vals["omch2"][-1]) + "\n")
        #with open(theta_MC_100_file_path, "a") as file:
        #    file.write(str(param_vals["theta_MC_100"][-1]) + "\n")
        #with open(logA_file_path, "a") as file:
        #    file.write(str(param_vals["logA"][-1]) + "\n")
        # with open(ns_file_path, "a") as file:
        #     file.write(str(param_vals["ns"][-1]) + "\n")
        # -------------------------------------------------------- DEBUG --------------------------------------------------------

        #end_time = time.time()
        #print(f"sample 1 thetas time = {end_time - start_time}")

        # ombh2 = (np.array(param_vals["ombh2"]) - 0.022386) / 0.0004006
        # #omch2 = (np.array(param_vals["omch2"]) - 0.109381) / 0.0059354
        # #logA = (np.array(param_vals["logA"]) - 3.218387) / 0.1128948
        # #ns = (np.array(param_vals["ns"]) - 0.959814) / 0.0164744
        # #theta_MC_100 = (np.array(param_vals["theta_MC_100"]) - 1.031732) / 0.0262018

        # plt.figure(figsize = (16, 10))
        # plt.plot(ombh2, label = "ombh2", marker = "o")
        # #plt.plot(omch2, label = "omch2", marker = "o")
        # #plt.plot(logA, label = "logA", marker = "o")
        # #plt.plot(ns, label = "ns", marker = "o")
        # #plt.plot(theta_MC_100, label = "theta_MC_100", marker = "o")
        # plt.axhline(0, color = "black", label = "Zero Sigma")
        # plt.axhline(1, color = "grey", label = "+/- 1 Sigma")
        # plt.axhline(-1, color = "grey")
        # plt.title("(mean - sample)/sigma")
        # plt.xlabel("iteration")
        # #plt.ylim([-0.5, 0.5])
        # plt.ylabel("standard deviations")
        # plt.legend()
        # plt.savefig("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/ombh2_progress.png")
        # plt.close()

        #6. unmix the fields using the updated version of the G & D matrices
        _, phi = unmix(mixed_temp, mixed_phi, args["mixing_d"], args["mixing_g"])
        #print(f"ombh2 history = {np.array(ombh2)}")

    #     #record the physical (unmixed) phi in its fixed Fourier basis for the mixing diagnostic
    #     if phi_recorder is not None:
    #         phi_recorder.record(phi)

    # #Step 0 diagnostic: dump the phi history and localize which modes fail to decorrelate
    # if phi_recorder is not None:
    #     stack = phi_recorder.stack()
    #     np.save(f"{mode_diag_dir}/phi_mode_history_seed_{seed}.npy", stack)
    #     analyze_phi_modes(stack, args["nside"], args["pix_width"], mode_diag_dir,
    #                       burn_in = mode_diag_burn_in, tag = f"phi_seed_{seed}")

    #Return your learned distributions at the end of the chain
    #for each parameter that was sampled
    # end_time = time.time()
    # total_time = end_time - start_time
    # file_path = "/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/"
    # np.savetxt(file_path + f"sample_lcdm_legacy_time_4_cores.txt", np.array([total_time]))
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
    master_seed = 314150 * 10 #NOTE originally 11 
    #seed 15 = slightly low
    #seed 14 = slightly low
    #seed 12 = slightly low
    #seed 10 = slightly low
    #seed 18 = medium low

    #seed 13 = slightly high
    #seed 16 = medium high
    #seed 17 = medium high
    #seed 11 = super high
    noise_level = 5

    #For the paramaters we want to infer, the tensor-to-scalar ratio should be zero
    #TODO we need to make sure the default parameters in load_sim() match the default 
    #parameter's in Yuuki's emulator code... 
    data_set = load_sim(nside, theta_pix, pol, master_seed, **ground_truth_params,
                        uk_arcmin_t = noise_level, r = 0, nt = 0)
    f_ground = data_set.unlensed_field
    phi_ground = data_set.phi

    #switch back to cambemul naming conventions...
    #NOTE if we only sample ombh2 the other parameters should be fixed at their fiducial
    #values for better convergence...
    param_init = {}
    param_init["ombh2"] = 0.024389 #0.022386 #0.024389 #0.022386 #0.020413 #-5 sigma from Yuuki's mean for training
    param_init["omch2"] = 0.109381 #0.079704 #-5 sigma from mean
    param_init["theta_MC_100"] = 1.031732 #1.156063 #-5 sigma from mean #NOTE +5 here and -5 for logA seems to break CAMB
    param_init["logA"] = 3.218387 #3.782861 #+5 sigma from mean
    param_init["ns"] = 0.959814 #1.042186 #0.867143 #+5 sigma from mean

    #allowed search / sample range for parameters... The min and max values are +/- 5 std
    #from the training mean for the CAMB emulator
    SEARCH_PRECISION = 50
    MAX_BUFFER_FACTOR = 1 #1.1 #Use a buffer to avoid getting trapped at the search boundaries
    MIN_BUFFER_FACTOR = 1 #MAX_BUFFER_FACTOR - 1
    param_ranges = {}
    param_ranges["ombh2"] = jnp.linspace(MIN_BUFFER_FACTOR * 0.020413, 0.024389 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)
    param_ranges["omch2"] = jnp.linspace(MIN_BUFFER_FACTOR * 0.079704, 0.155541 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)
    param_ranges["theta_MC_100"] = jnp.linspace(MIN_BUFFER_FACTOR * 0.900723, 1.156063 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)
    param_ranges["logA"] = jnp.linspace(MIN_BUFFER_FACTOR * 2.661635, 3.782861 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)
    param_ranges["ns"] = jnp.linspace(MIN_BUFFER_FACTOR * 0.867143, 1.042186 * MAX_BUFFER_FACTOR, SEARCH_PRECISION)

    #Whether or not to sample each parameter
    #NOTE just sampling ombh2 for the time being while we get up and running
    should_sample = {}
    should_sample["ombh2"] = True
    should_sample["omch2"] = False
    should_sample["theta_MC_100"] = False
    should_sample["logA"] = False
    should_sample["ns"] = False

    #run the sampling algorithm
    #start_time = time.time()
    param_distributions = sample_joint(data_set, param_init, param_ranges, should_sample, noise_level,
                                       f_ground, phi_ground, record_phi_modes = False,
                                       use_hvp_mass_matrix = False, theta_samplers = {"ombh2": "metropolis"}, 
                                       proposal_sigmas = {"ombh2": 1e-3}, 
                                       iters_per_chain = 10_000, num_burn_in_fix_theta = 0, 
                                       over_relaxation_num_samps = -1, seed = 67,
                                       num_burn_in_always_accept = 0, phi_start = "MAP", 
                                       f_start = "MAP")
    #end_time = time.time()
    #file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/lcdm_data_v2/"
    #os.makedirs(file_path, exist_ok = True)
    #np.savez(file_path + f"learned_legacy_ombh2_distribution.npz", np.array(param_distributions["ombh2"]))

    #record the time as well
    #total_time = end_time - start_time
    #np.savetxt(file_path + f"sample_lcdm_legacy_time.txt", np.array([total_time]))

