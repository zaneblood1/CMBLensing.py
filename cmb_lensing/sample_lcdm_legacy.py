#the chain loop savefigs progress figures every sweep and never shows a window; the
#default TkAgg GUI backend aborts long runs when tkinter objects get garbage-collected
#off the main thread ("RuntimeError: main thread is not in main loop" spam, then a fatal
#"Tcl_AsyncDelete: async handler deleted by the wrong thread" - observed after ~1200
#sweeps). Agg is headless and safe for pure-savefig use; it must be selected BEFORE the
#cmb_lensing imports below, which pull in pyplot via statistics.py
import matplotlib
matplotlib.use("Agg")

from cmb_lensing.simulate import *
#underscore-prefixed CAMB helpers are skipped by "import *", so pull them in explicitly
from cmb_lensing.simulate import _camb_via_callback, _extract_all_cls, _covar_or_zeros
from cmb_lensing.wiener_filter import *
from cmb_lensing.util import *
from cmb_lensing.map_joint import *
from cmb_lensing.mixing import *
from cmb_lensing.constants import *
from cmb_lensing.precompute_camb_1d import (load_camb_spline_predictors, GROUND_TRUTH,
                                            TRAINING_SIGMA, PARAM_BOUNDS)
from cmb_lensing.camb_grid_interp import (load_camb_grid_predictors,
                                          load_camb_grid_predictors_grad,
                                          load_camb_grid, GridPredictors)
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

#when True, every theta-step Cl prediction (eval_logpdf_grid and the post-sample
#covariance recompute) uses the cached-CAMB 1D spline for the sampled parameter from
#precompute_camb_1d.py instead of the emulator. Only valid while a single parameter is
#sampled - the 1D cache holds the other four at their ground-truth values (the loader
#raises if any of them moves off its cached fixed value). The data map must also be
#CAMB-generated: load_sim(..., use_emulator_cls = False)
USE_CAMB_SPLINE = False
_camb_spline_predictors = {}

def get_camb_spline_predictors(param_name):
    #lazy per-parameter singletons so the same function objects are reused every call -
    #they are passed as static jit args to _recompute_cosmo_matrices, so fresh objects
    #would retrace
    if param_name not in _camb_spline_predictors:
        _camb_spline_predictors[param_name] = load_camb_spline_predictors(param_name)
    return _camb_spline_predictors[param_name]

#when True, every Cl prediction comes from the 5D tensor-product cubic spline built by
#performance_testing/sampling_chains/merge_camb_grid.py over
#(H0, logA, ns, ombh2, omch2) with tau pinned at 0.05. Unlike the 1D caches above this
#interpolates all five parameters simultaneously, so every parameter may be sampled at
#once - which is the whole point of the grid. Mutually exclusive with USE_CAMB_SPLINE;
#like it, the data map must be CAMB-generated (load_sim(..., use_emulator_cls = False)).
#
#The grid is laid out in H0, not theta_MC_100, because a rectangular theta box has corners
#CAMB cannot solve. camb_grid_interp converts theta -> H0 internally using the theta table
#recorded at every node, so the sampler keeps working in theta_MC_100 throughout and
#PARAM_ORDER is unchanged.
USE_CAMB_GRID = True
CAMB_GRID_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "performance_testing", "sampling_chains",
                              "multi_param_CAMB_grid", "camb_grid_spline.npz")
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

_camb_grid_predictors_grad = None

def get_camb_grid_predictors_grad():
    #lazy singleton for the DIFFERENTIABLE grid predictors the GHMC theta update
    #differentiates through: numpy callback forward pass + analytic spline-derivative
    #backward pass (jax.custom_vjp). These share the numpy coefficient tables with
    #get_camb_grid_predictors, costing no extra memory - unlike the pure-JAX predictors
    #(load_camb_grid_predictors_jax), whose tables jit re-embeds as ~2.3 GB of constants
    #in every traced executable and which OOM this 16 GB machine
    global _camb_grid_predictors_grad
    if _camb_grid_predictors_grad is None:
        _camb_grid_predictors_grad = load_camb_grid_predictors_grad(CAMB_GRID_PATH)
    return _camb_grid_predictors_grad

def camb_grid_bb_is_zero():
    #whether the merged grid's unlensed BB is the identically-zero r = 0 spectrum. Static
    #per grid file, so the covariance builders can route BB around the log-interpolating
    #path (which would turn exact zeros into NaN) at trace time
    return load_camb_grid(CAMB_GRID_PATH).bb_is_zero

def wrap_predictors(predict_tt, predict_pp):
    """Lift a TT/PP-only predictor pair (the emulator or the 1D CAMB caches) into the
    GridPredictors namedtuple shape the rest of the sampler passes around. The missing
    spectra are None, which is only valid for pol = 'I' with refresh_qe off - the 5D
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

#PCA reparametrization of the theta step. A pilot chain (save_pca = True) writes the
#empirical 5x5 parameter covariance here, eigendecomposed; a production chain
#(use_pca = True) then proposes along the eigen-directions instead of the coordinate axes,
#which decorrelates the degenerate LCDM parameters and lets one global proposal_sigma (in
#units of conditional standard deviations) replace the five hand-tuned per-parameter sigmas
THETA_PCA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "performance_testing", "sampling_chains", "theta_pca.npz")

def compute_and_save_pca(param_vals, pca_path, burn_in = 0):
    """Empirical PCA of a chain: drop the first burn_in entries of every parameter history,
    compute the 5x5 covariance in PARAM_ORDER, eigendecompose it, and save to pca_path.
    Requires every parameter to have been sampled - a frozen parameter has zero variance,
    which load_pca_directions rejects."""
    n = min(len(param_vals[name]) for name in PARAM_ORDER)
    if n - burn_in < 10:
        raise ValueError(f"not enough post-burn-in samples for a covariance estimate "
                         f"(shortest history has {n} entries, burn_in = {burn_in})")
    samples = np.array([np.asarray(param_vals[name][burn_in:n], dtype = np.float64)
                        for name in PARAM_ORDER])
    theta0 = samples.mean(axis = 1)
    covariance = np.cov(samples)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    np.savez(pca_path, theta0 = theta0, covariance = covariance,
             eigvals = eigvals, eigvecs = eigvecs)
    print(f"saved theta PCA ({n - burn_in} samples) to {pca_path}")
    for i in range(len(PARAM_ORDER)):
        combo = " ".join(f"{eigvecs[j, i]:+.3f}*{PARAM_ORDER[j]}"
                         for j in range(len(PARAM_ORDER)))
        print(f"  sigma = {np.sqrt(max(eigvals[i], 0.0)):.3e} along {combo}")

def load_pca_directions(pca_path):
    """Load a saved PCA and return a (5, 5) array whose row i is eigenvector i scaled by
    sqrt(eigenvalue i) - so a unit step along a row is a one-conditional-sigma move."""
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"no theta PCA at {pca_path} - run a pilot chain with "
                                f"save_pca = True (and use_pca = False) first")
    data = np.load(pca_path)
    eigvals, eigvecs = data["eigvals"], data["eigvecs"]
    #the cutoff is relative: a frozen parameter's zero variance shows up as round-off
    #(~1e-16 of the largest eigenvalue), while genuinely degenerate-but-sampled
    #combinations sit many orders of magnitude above it
    if not np.all(np.isfinite(eigvals)) or np.any(eigvals <= 1e-12 * np.max(eigvals)):
        raise ValueError(f"theta PCA at {pca_path} has zero/non-positive eigenvalues "
                         f"{eigvals} - the pilot chain must sample every parameter")
    return (eigvecs * np.sqrt(eigvals)).T

def load_theta_whitening(pca_path, sampled_idx, scales):
    """Educated-guess whitening transform for the gradient-based theta update
    (use_ghmc_theta): returns (T, T_inv) with theta_sampled = T @ u, chosen so the
    posterior of u is approximately N(0, I) and the IDENTITY mass matrix is already
    ideal - no adaptation warmup needed (a window-adaptation warmup's cost scales with
    map size: hours at nside 256).

    With a pilot chain (theta_pca.npz): T = Cholesky factor of the sampled-subset
    covariance, which captures the strong LCDM correlations (logA-ns ~ -0.9,
    ombh2-omch2 ~ +0.86). The correlations must live in the position transform because
    blackjax's ghmc cannot take a dense mass matrix (a (d, d) momentum_inverse_scale
    diverges 100% of the time - measured - despite what its docstring claims).
    Fallback without a pilot: T = diag(TRAINING_SIGMA), plain per-parameter whitening."""
    scales = np.asarray(scales, dtype = np.float64)
    if os.path.exists(pca_path):
        covariance = np.load(pca_path)["covariance"]
        sub = covariance[np.ix_(list(sampled_idx), list(sampled_idx))]
        transform = np.linalg.cholesky(sub)
        print(f"theta whitening: pilot-covariance Cholesky from {pca_path}")
    else:
        transform = np.diag(scales)
        print("theta whitening: TRAINING_SIGMA diagonal (no pilot covariance found)")
    return jnp.asarray(transform), jnp.asarray(np.linalg.inv(transform))

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
    #pol = "IP" carries BlockTEB covariances and FlatS02 fields, pol = "P" DiagonalEB
    #covariances and FlatS2 fields; field_start must be shaped like the UNLENSED FIELD
    #(0*data_set.unlensed_field), not like phi. the isinstance is resolved at trace time
    #(operator structure is static under jit)
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
        #E and B are independent (the EB block is diagonal); a zero BB covariance
        #(unlensed B at r = 0) yields an exactly zero B field
        e, b = _field_matrices_from_eb_covar(args["field_covariance"],
                                             data_field.nside, key_f)
        new_field = field_start.replace(polar_matrix_1 = jfft.rfft2(e),
                                        polar_matrix_2 = jfft.rfft2(b))
    else:
        new_field_matrix = field_from_covar_single_key(data_field.nside,
                            args["field_covariance"].scalar_matrix, key_f)
        #Convert raw matrix to instance of FlatS0
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
                                args["mask"], args["beam"], maxiter = 25)

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

#------------------ preconditioned Crank-Nicolson (pCN) --------------------------
#Gradient-free alternative to the HMC step for the mixed phi conditional. pCN exploits
#the exactly-Gaussian prior on mixed phi: phi = pinv(G) * mixed_phi with phi ~ N(0, Cphi)
#implies mixed_phi ~ N(0, G^2 * Cphi). The AR(1) proposal
#    mixed_phi' = sqrt(1 - beta^2) * mixed_phi + beta * xi,   xi ~ N(0, G^2 * Cphi)
#preserves that prior exactly for any beta in (0, 1], so the Metropolis test involves
#only the likelihood part Phi(x) = mixed_logpdf(x) + 0.5 * <x, (G^2 Cphi)^-1 x>
#(the added quadratic form cancels the Gaussian prior term inside mixed_logpdf, and the
#logdet constants cancel in the acceptance difference since theta is fixed here).
#Each proposal costs one mixed_logpdf (~2 lense flow integrations) and no gradients,
#vs ~7 flow integrations per leapfrog step in the HMC path.
@jax.jit
def gibbs_sample_phi_pcn(mixed_phi, mixed_temp, data_field, rng_key,
                         args, iter, num_burn_in_always_accept,
                         beta, num_steps):

    always_accept = (iter < num_burn_in_always_accept)
    prior_covariance = args["mixing_g"] * args["mixing_g"] * args["phi_covariance"]

    #the likelihood part of the mixed phi conditional: mixed_logpdf with the Gaussian
    #prior quadratic form added back so the prior cancels from the acceptance ratio
    def likelihood_partial(x):
        full_logpdf = mixed_logpdf(mixed_temp, x, data_field, args["noise_covariance"],
                                   args["phi_covariance"], args["field_covariance"],
                                   args["mask"], args["beam"],
                                   args["mixing_g"], args["mixing_d"])
        prior_product = jnp.real(dot(x, pinv(prior_covariance) * x))
        return full_logpdf + 0.5 * prior_product

    def on_accept(x, x_test):
        _ = x
        return x_test
    def on_decline(x, x_test):
        _ = x_test
        return x

    def loop_body(_, state):
        x, log_lik, key, num_accepted, delta_lik_sum = state
        key, key_xi, key_u = jax.random.split(key, 3)

        #draw xi ~ N(0, G^2 * Cphi) with the same machinery as every other field draw
        xi_matrix = field_from_covar_single_key(mixed_phi.nside,
                                                prior_covariance.scalar_matrix, key_xi)
        xi = x.replace(scalar_matrix = jfft.rfft2(xi_matrix))

        #prior-preserving AR(1) proposal
        x_test = jnp.sqrt(1 - beta**2) * x + beta * xi
        log_lik_test = likelihood_partial(x_test)

        #likelihood-only acceptance test. NaN-safe: a non-finite log_lik_test makes the
        #comparison False so the proposal is rejected (unless in the always-accept phase)
        delta_lik = log_lik_test - log_lik
        accept = jnp.logical_or(always_accept, jnp.log(jax.random.uniform(key_u)) < delta_lik)
        x = jax.lax.cond(accept, on_accept, on_decline, x, x_test)
        log_lik = jnp.where(accept, log_lik_test, log_lik)
        return (x, log_lik, key, num_accepted + accept, delta_lik_sum + delta_lik)

    log_lik_start = likelihood_partial(mixed_phi)
    state = (mixed_phi, log_lik_start, rng_key, 0.0, 0.0)
    mixed_phi, _, _, num_accepted, delta_lik_sum = jax.lax.fori_loop(0, num_steps,
                                                                     loop_body, state)

    #return the mean likelihood change and acceptance fraction across the pCN steps,
    #mirroring the (delta_h, accept) diagnostics of the HMC path
    return mixed_phi, delta_lik_sum / num_steps, num_accepted / num_steps

#------------------ generalized HMC (partial momentum refreshment) ---------------
#GHMC (Horowitz 1991) alternative to the HMC step for the mixed phi conditional. plain
#MALA is HMC with ONE leapfrog step and a FULL momentum refresh, which makes it diffusive
#and pins its stable step size to the stiffest mode. GHMC keeps the one-leapfrog-step
#cost (~1 gradient + 1 logpdf per update) but only PARTIALLY refreshes the momentum
#    u' = sqrt(1 - alpha^2) * u + alpha * xi,   xi ~ N(0, I)
#so momentum persists across updates (and across Gibbs sweeps) and trajectories stay
#ballistic like HMC instead of diffusive like MALA. the standard price: on rejection the
#momentum must be FLIPPED (u -> -u), which reverses the trajectory, so GHMC only mixes
#well at HIGH acceptance - run with a step size giving >~90% acceptance and let the
#persistence (small alpha) supply the long trajectories that num_steps used to.
#
#the momentum is stored WHITENED (u ~ N(0, I), p = sqrt(M) * u) and carried across Gibbs
#sweeps in the chain state. this matters because the mass matrix M depends on theta:
#after a theta update a raw p ~ N(0, M_old) would be wrongly distributed, but u ~ N(0, I)
#stays exactly correct under any M, so the persistent momentum survives theta moves for
#free. field_from_covar_single_key draws irfft2(rfft2(white) * sqrt(covar)), so
#p = sqrt(M) * rfft2(white) reproduces the HMC momentum draw distribution exactly
#(including full power in the self-conjugate rfft columns)
@jax.jit
def gibbs_sample_phi_ghmc(mixed_phi, momentum_u, mixed_temp, data_field, rng_key,
                          args, iter, num_burn_in_always_accept,
                          alpha, step_size, num_steps):

    always_accept = (iter < num_burn_in_always_accept)
    mass_matrix = get_mass_matrix(args["phi_covariance"], args["quadratic_estimate"],
                                  args["mixing_g"])
    sqrt_mass = mass_matrix.replace(scalar_matrix = jnp.sqrt(mass_matrix.scalar_matrix))
    inv_mass = pinv(mass_matrix)
    inv_sqrt_mass = pinv(sqrt_mass)

    def logpdf_partial(x):
        return mixed_logpdf(mixed_temp, x, data_field, args["noise_covariance"],
                            args["phi_covariance"], args["field_covariance"],
                            args["mask"], args["beam"],
                            args["mixing_g"], args["mixing_d"])

    def mixed_grad_phi_partial(x):
        return mixed_grad_phi_logpdf(mixed_temp, x, data_field, args["noise_covariance"],
                                     args["phi_covariance"], args["field_covariance"],
                                     args["mask"], args["beam"],
                                     args["mixing_d"], args["mixing_g"])

    #kinetic term, identical to the one inside symplectic_integrate's hamiltonian
    def kinetic(p):
        return dot(p, (inv_mass * p) / 2)

    def on_accept(x, x_test):
        _ = x
        return x_test
    def on_decline(x, x_test):
        _ = x_test
        return x

    def loop_body(_, state):
        x, u, log_post, gradient, key, num_accepted, delta_h_sum = state
        key, key_xi, key_acc = jax.random.split(key, 3)

        #partial momentum refresh in whitened coordinates - leaves N(0, I) invariant
        white = jax.random.normal(key_xi, shape = (x.nside, x.nside))
        xi = x.replace(scalar_matrix = jfft.rfft2(white))
        u = jnp.sqrt(1 - alpha**2) * u + alpha * xi
        p = sqrt_mass * u

        #one leapfrog step, same update and sign conventions as symplectic_integrate
        x_test = x - step_size * inv_mass * (p - 0.5 * step_size * gradient)
        gradient_test = mixed_grad_phi_partial(x_test)
        p_test = p - 0.5 * step_size * (gradient_test + gradient)

        #Metropolis test on the (implicitly momentum-flipped) proposal - the kinetic term
        #is even in p so the flip does not change delta_h, and on accept the flip is undone
        #so the trajectory keeps moving forward. NaN-safe: a non-finite logpdf makes the
        #comparison False so the proposal is rejected (unless in the always-accept phase)
        log_post_test = logpdf_partial(x_test)
        delta_h = (log_post_test - kinetic(p_test)) - (log_post - kinetic(p))
        accept = jnp.logical_or(always_accept, jnp.log(jax.random.uniform(key_acc)) < delta_h)

        #on accept keep the evolved momentum (re-whitened); on reject FLIP the persistent
        #momentum - the flip is what keeps the partial-refresh kernel reversible
        x = jax.lax.cond(accept, on_accept, on_decline, x, x_test)
        u = jax.lax.cond(accept, on_accept, on_decline, -1 * u, inv_sqrt_mass * p_test)
        gradient = jax.lax.cond(accept, on_accept, on_decline, gradient, gradient_test)
        log_post = jnp.where(accept, log_post_test, log_post)
        return (x, u, log_post, gradient, key, num_accepted + accept, delta_h_sum + delta_h)

    log_post_start = logpdf_partial(mixed_phi)
    gradient_start = mixed_grad_phi_partial(mixed_phi)
    state = (mixed_phi, momentum_u, log_post_start, gradient_start, rng_key, 0.0, 0.0)
    mixed_phi, momentum_u, _, _, _, num_accepted, delta_h_sum = jax.lax.fori_loop(
        0, num_steps, loop_body, state)

    #return the mean delta_H and acceptance fraction across the GHMC updates, mirroring
    #the (delta_h, accept) diagnostics of the HMC and pCN paths
    return mixed_phi, momentum_u, delta_h_sum / num_steps, num_accepted / num_steps

#------------------ symplectic integration ---------------------------------------
#NOTE num_steps * step_size = path_length must be tuned... Too large and 
#you can overshoot and end up in physically impossible / divergent solutions...
#Too small and you may not have enough momentum to escape local minima
#and converge on the true global minimum
def symplectic_integrate(x0, p0, mixed_field, data, noise_covariance, 
                        phi_covariance, field_covariance, mask, beam, 
                        mixing_d, mixing_g, mass_matrix,
                        num_steps = 5, step_size = 0.1):
    
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
def metropolis_sample_theta(eval_logpdf_grid, theta_old, lo0, hi0, theta_key,
                            ombh2_accept_history, omch2_accept_history,
                            ns_accept_history, theta_MC_100_accept_history, logA_accept_history,
                            proposal_sigma, rng_key, num_steps = 1):
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
            if theta_key == "ns":
                ns_accept_history.append(1)
            if theta_key == "logA":
                logA_accept_history.append(1)
            if theta_key == "ombh2":
                ombh2_accept_history.append(1)
            if theta_key == "omch2":
                omch2_accept_history.append(1)
            if theta_key == "theta_MC_100":
                theta_MC_100_accept_history.append(1)
        else:
            if theta_key == "ns":
                ns_accept_history.append(0)
            if theta_key == "logA":
                logA_accept_history.append(0)
            if theta_key == "ombh2":
                ombh2_accept_history.append(0)
            if theta_key == "omch2":
                omch2_accept_history.append(0)
            if theta_key == "theta_MC_100":
                theta_MC_100_accept_history.append(0)
        if theta_key == "ns":
            print(f"ns accept rate = {np.sum(np.array(ns_accept_history))/len(ns_accept_history)}")
        if theta_key == "logA":
            print(f"logA accept rate = {np.sum(np.array(logA_accept_history))/len(logA_accept_history)}")
        if theta_key == "ombh2":
            print(f"ombh2 accept rate = {np.sum(np.array(ombh2_accept_history))/len(ombh2_accept_history)}")
        if theta_key == "omch2":
            print(f"omch2 accept rate = {np.sum(np.array(omch2_accept_history))/len(omch2_accept_history)}")
        if theta_key == "theta_MC_100":
            print(f"theta_MC_100 accept rate = {np.sum(np.array(theta_MC_100_accept_history))/len(theta_MC_100_accept_history)}")

    return np.asarray(theta_current, dtype = np.float64)

def metropolis_sample_direction(eval_logpdf_batch, params_old, direction, direction_idx,
                                bounds_lo, bounds_hi, pca_accept_history,
                                proposal_sigma, rng_key, num_steps = 1):
    """Gaussian random-walk Metropolis on the FULL parameter vector along one PCA
    direction. direction is an eigenvector scaled to one conditional standard deviation,
    so proposal_sigma is in sigma units and a single value serves every direction."""
    if proposal_sigma is None:
        raise ValueError("metropolis sampler requires a proposal_sigma")

    params_current = np.asarray(params_old, dtype = np.float64).copy()
    for _ in range(num_steps):
        rng_key, k_prop, k_acc = jax.random.split(rng_key, 3)
        params_prop = params_current + proposal_sigma * float(jax.random.normal(k_prop)) * direction

        #reject out-of-box proposals BEFORE evaluating, exactly as metropolis_sample_theta
        #does: the target has zero density outside the grid support and the proposal is
        #symmetric, so keeping the old value is exact MH with no correction term. A
        #theta_MC_100 unreachable at this ombh2/omch2 (the box cannot see that - the grid
        #is laid out in H0) is still caught by the grid's NaN -> rejected-step backstop
        if not (np.all(params_prop >= bounds_lo) and np.all(params_prop <= bounds_hi)):
            continue

        logpdfs = eval_logpdf_batch(jnp.stack([jnp.array(params_current),
                                               jnp.array(params_prop)]))
        delta_h = float(logpdfs[1] - logpdfs[0])

        #accept if log(u) < delta_h. NaN-safe: a non-finite logpdf makes delta_h nan and
        #(x < nan) is False, so the step is rejected
        if float(jnp.log(jax.random.uniform(k_acc))) < delta_h:
            params_current = np.asarray(params_prop, dtype = np.float64)
            pca_accept_history.append(1)
        else:
            pca_accept_history.append(0)
        print(f"pca direction {direction_idx} accept rate = "
              f"{np.sum(np.array(pca_accept_history)) / len(pca_accept_history)}")
    return params_current

def metropolis_sample_joint(eval_logpdf_batch, params_old, proposal_sigmas,
                            bounds_lo, bounds_hi, joint_accept_history,
                            rng_key, num_steps = 1):
    """Plain joint Gaussian random-walk Metropolis: kick all five parameters at once with
    independent Gaussians (one tunable sigma per parameter) and take a single
    accept/reject on the joint logpdf change. proposal_sigmas is a (5,) array in
    PARAM_ORDER. Under strong correlations the sigmas must shrink toward the CONDITIONAL
    widths for proposals to be accepted, so mixing along a degeneracy ridge stays slow -
    this is the simple baseline the PCA eigen-direction sampler improves on."""
    params_current = np.asarray(params_old, dtype = np.float64).copy()
    for _ in range(num_steps):
        rng_key, k_prop, k_acc = jax.random.split(rng_key, 3)
        kick = proposal_sigmas * np.asarray(jax.random.normal(k_prop,
                                                              shape = params_current.shape))
        params_prop = params_current + kick

        #reject out-of-box proposals BEFORE evaluating (exact MH for a zero-density-
        #outside-the-box target with a symmetric proposal); the grid's NaN -> rejected-step
        #backstop still covers theta_MC_100 unreachable at this ombh2/omch2
        if not (np.all(params_prop >= bounds_lo) and np.all(params_prop <= bounds_hi)):
            continue

        logpdfs = eval_logpdf_batch(jnp.stack([jnp.array(params_current),
                                               jnp.array(params_prop)]))
        delta_h = float(logpdfs[1] - logpdfs[0])

        #accept if log(u) < delta_h. NaN-safe: a non-finite logpdf makes delta_h nan and
        #(x < nan) is False, so the step is rejected
        if float(jnp.log(jax.random.uniform(k_acc))) < delta_h:
            params_current = np.asarray(params_prop, dtype = np.float64)
            joint_accept_history.append(1)
        else:
            joint_accept_history.append(0)
        print(f"joint MH accept rate = "
              f"{np.sum(np.array(joint_accept_history)) / len(joint_accept_history)}")
    return params_current

#@partial(jax.jit, static_argnames = ["model_tt", "model_pp", "lmax", "lmax_prime",
#Gaussian priors on individual cosmological parameters, added to EVERY theta logpdf
#evaluation (all sampler paths - grid, Metropolis, PCA, joint MH, GHMC - share
#make_eval_logpdf_batch, so the posterior they target changes consistently, and the FD
#gradients pick the prior up automatically). Entries are name: (mean, sigma). The ombh2
#BBN-style prior below existed because the T-only map constrains the ns-ombh2 plane
#poorly (marginal corr ~ -0.76). With pol = "IP" the EE spectrum breaks that degeneracy
#from the data itself, so the prior is OFF by default - re-enable it to compare against
#the T-only-with-prior chains
THETA_PRIORS = {}
#THETA_PRIORS = {"ombh2": (GROUND_TRUTH["ombh2"], 3.6e-4)}

def make_eval_logpdf_batch(predictors, emu_params, 
                           mixed_temp_matrix, mixed_phi_matrix, data_matrix,
                           cphi_fid, cf_fid, qe_scalar, cn_scalar,
                           mask_matrix, beam_matrix, fourier_weights,
                           nside, pix_width, theta_pix, ell_grid,
                           pol = "I", bb_is_zero = True, fixed_field = False,
                           freeze_g = False, cn_mix_scalar = None):
    """Build eval_logpdf_batch((M, 5) PARAM_ORDER batch) -> (M,) mixed logpdfs at the
    given (fixed) mixed fields, rebuilding the covariances and G/D mixing matrices from
    predicted Cls at every row. Shared by every theta update path: the grid / Metropolis /
    PCA / joint-MH paths in gibbs_sample_theta, and the GHMC logdensity in
    sample_joint - which differentiates through it, so predictors must then be the
    gradient-capable grid predictors from load_camb_grid_predictors_grad, not the plain
    pure_callback ones.

    fixed_field = True replaces the per-row G/D with the IDENTITY (see sample_joint's
    fixed_field_theta flag): unmix then returns the pinned fields unchanged for every
    theta row and logdet(G) = logdet(D) = 0, so the batch evaluates the UNMIXED
    conditional p(theta | f, phi, d) instead of the mixed one. Requires the caller to
    have mixed with identity matrices too - a theta-dependent outer mixing would make
    the pinned-field chain a fixed-point iteration whose attractor sits away from the
    conditional peak (the +1-training-sigma theta_MC_100 parking seen on pol = "IP").

    freeze_g = True replaces ONLY the per-row G with the identity while D stays live
    (see sample_joint's freeze_g_mixing flag): unmix then returns the pinned phi for
    every theta row (logdet(G) = 0) but keeps the theta-dependent D transform of the
    genuinely resampled f, whose Jacobian logdet(D) is still paid. Requires the caller
    to have mixed with identity G too. Mutually exclusive with fixed_field, which pins
    both matrices.

    cn_mix_scalar decouples the MIXING noise from the likelihood noise (sample_joint's
    mixing_noise_uk_arcmin): the per-row D and G are built from cn_mix_scalar while the
    likelihood's noise covariance stays cn_scalar. G/D are pure reparametrizations
    (their Jacobians are paid in mixed_logpdf), so any choice here changes mixing speed
    only, never the posterior. None means cn_scalar (no decoupling). The qe_scalar the
    caller passes must be built at the same mixing noise level for consistency.

    pol = "I": mixed_temp_matrix / data_matrix / cn_scalar / cf_fid are single
    (nside, nside//2 + 1) matrices and the logpdf is the T-only one.
    pol = "IP": the field arguments are (3, ...) T/E/B stacks (field_matrix_stack) and
    the operator arguments (4, ...) TT/TE/EE/BB stacks (op_matrix_stack), the fields
    are FlatS02, the covariances BlockTEB with a REAL TE block: the te_rho predictor
    supplies the correlation ratio and cf_te = rho * sqrt(cf_tt * cf_ee)
    (te_covar_from_rho), matching load_sim's correlated T/E draws; the D matrix comes
    from get_d_teb_matrix's coupled T/E block. With bb_is_zero (the r = 0 grids) the BB
    covariance is exactly zero and is routed around the log-interpolating covariance
    builder, which would NaN it. The QE norm qe_scalar is FIXED across the batch - it
    is refreshed between sweeps (refresh_qe), never within a theta update, so the
    accept/reject compares both points under one mixing.
    pol = "P": polarization only - the field arguments are (2, ...) E/B stacks, the
    operator arguments (2, ...) EE/BB stacks, the fields FlatS2 and the covariances
    DiagonalEB. No TT or TE enters at all: only the ee (and, off the r = 0 grids, bb)
    predictors are evaluated and the D matrix is get_d_eb_matrix's diagonal E/B pair.
    bb_is_zero is handled as in "IP"."""

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
    #the likelihood noise above always stays the true cn_scalar; only D/G see this
    if cn_mix_scalar is None:
        cn_mix_scalar = cn_scalar
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

        cl_pp_batch = predictors.pp(emu_params, params_batch)
        if pol != "P":
            cl_tt_batch = predictors.tt(emu_params, params_batch)
        if pol in ("IP", "P"):
            cl_ee_batch = predictors.ee(emu_params, params_batch)
            if not bb_is_zero:
                cl_bb_batch = predictors.bb(emu_params, params_batch)
        if pol == "IP":
            te_rho_batch = predictors.te_rho(emu_params, params_batch)

        def single_logpdf(i):

            cl_pp = cl_pp_batch[i]

            if pol == "P":
                cl_ee = cl_ee_batch[i]
                #ells sized to the predictor output (see the T branches below)
                emu_ells = jnp.arange(2, 2 + cl_ee.shape[-1])
                cphi = covar_matrix_from_cls(nside, pix_width,
                                             ell_grid, emu_ells,
                                             cl_pp, origin_value = 0)
                #the MIXING noise blocks (see the "IP" branch below); only D/G use them
                cn_ee, cn_bb = cn_mix_scalar[0], cn_mix_scalar[1]
                cf_ee = covar_matrix_from_cls(nside, pix_width,
                                              ell_grid, emu_ells,
                                              cl_ee, origin_value = 0)
                if bb_is_zero:
                    #an out-of-box row is already NaN through cf_ee, so the exact-zero
                    #BB block never weakens the rejected-proposal semantics
                    cf_bb = jnp.zeros_like(cf_ee)
                else:
                    cf_bb = covar_matrix_from_cls(nside, pix_width,
                                                  ell_grid, emu_ells,
                                                  cl_bb_batch[i], origin_value = 0)

                if fixed_field:
                    g = jnp.ones_like(cphi)
                    d_ee = d_bb = jnp.ones_like(cphi)
                else:
                    if freeze_g:
                        g = jnp.ones_like(cphi)
                    else:
                        g = get_g_matrix_lcdm(cphi_fid, cphi, qe_scalar, cn_ee)
                    d_ee, d_bb = get_d_eb_matrix(cf_ee, cf_bb, cn_ee, cn_bb)

                return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                                    noise_covariance, _op(cphi),
                                    _op_eb(cf_ee, cf_bb),
                                    mask, beam, _op(g), _op_eb(d_ee, d_bb))

            cl_tt = cl_tt_batch[i]
            #ells sized to the predictor output so the extrapolation anchor matches the
            #data map's: emulator returns 2..4000, the CAMB spline cache returns 2..3999
            emu_ells = jnp.arange(2, 2 + cl_tt.shape[-1])

            cf_tt = covar_matrix_from_cls(nside, pix_width,
                                          ell_grid, emu_ells,
                                          cl_tt, origin_value = 0)
            cphi = covar_matrix_from_cls(nside, pix_width,
                                         ell_grid, emu_ells,
                                         cl_pp, origin_value = 0)

            if pol == "IP":
                #the MIXING noise blocks - equal to the true noise unless
                #mixing_noise_uk_arcmin decouples them; only D/G consume these
                cn_tt, cn_te = cn_mix_scalar[0], cn_mix_scalar[1]
                cn_ee, cn_bb = cn_mix_scalar[2], cn_mix_scalar[3]
                cf_ee = covar_matrix_from_cls(nside, pix_width,
                                              ell_grid, emu_ells,
                                              cl_ee_batch[i], origin_value = 0)
                cf_te = te_covar_from_rho(te_rho_batch[i], emu_ells, ell_grid,
                                          cf_tt, cf_ee)
                if bb_is_zero:
                    #an out-of-box row is already NaN through cf_tt, so the exact-zero
                    #BB block never weakens the rejected-proposal semantics
                    cf_bb = jnp.zeros_like(cf_tt)
                else:
                    cf_bb = covar_matrix_from_cls(nside, pix_width,
                                                  ell_grid, emu_ells,
                                                  cl_bb_batch[i], origin_value = 0)

                if fixed_field:
                    g = jnp.ones_like(cphi)
                    d_tt = d_ee = d_bb = jnp.ones_like(cphi)
                    d_te = jnp.zeros_like(cphi)
                else:
                    if freeze_g:
                        g = jnp.ones_like(cphi)
                    else:
                        g = get_g_matrix_lcdm(cphi_fid, cphi, qe_scalar, cn_tt)
                    d_tt, d_te, d_ee, d_bb = get_d_teb_matrix(cf_tt, cf_te, cf_ee,
                                                              cf_bb, cn_tt, cn_te,
                                                              cn_ee, cn_bb)
               
                return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                                    noise_covariance, _op(cphi),
                                    _op_teb(cf_tt, cf_te, cf_ee, cf_bb),
                                    mask, beam, _op(g),
                                    _op_teb(d_tt, d_te, d_ee, d_bb))
               

            if fixed_field:
                g = jnp.ones_like(cphi)
                d = jnp.ones_like(cf_tt)
            else:
                if freeze_g:
                    g = jnp.ones_like(cphi)
                else:
                    g = (get_g_matrix_lcdm(cphi_fid, cphi, qe_scalar, cn_mix_scalar))
                #d = get_d_tt_matrix(cf_tt, jnp.zeros_like(cf_tt), cn_mix_scalar, 1, 1)
                d = (get_d_tt_matrix(cf_tt, cf_fid, cn_mix_scalar))
                # ------------------ ZXB DEBUG -----------------
                #d = jnp.ones_like(d)
                #g = jnp.ones_like(g)
                # ------------------ ZXB DEBUG -----------------
            return mixed_logpdf(mixed_temp, mixed_phi, data_field,
                                noise_covariance, _op(cphi), _op(cf_tt),
                                mask, beam, _op(g), _op(d))

        logpdfs = jax.vmap(single_logpdf)(jnp.arange(M))
        #Gaussian parameter priors (THETA_PRIORS). Normalization constants are dropped -
        #only logpdf differences enter every acceptance test and gradient
        if THETA_PRIORS:
            prior_idx = jnp.array([PARAM_INDEX[name] for name in THETA_PRIORS],
                                  dtype = jnp.int32)
            prior_mu = jnp.array([THETA_PRIORS[name][0] for name in THETA_PRIORS])
            prior_sigma = jnp.array([THETA_PRIORS[name][1] for name in THETA_PRIORS])
            deviations = (params_batch[:, prior_idx] - prior_mu) / prior_sigma
            logpdfs = logpdfs - 0.5 * jnp.sum(deviations**2, axis = 1)
        return logpdfs

    return eval_logpdf_batch

def gibbs_sample_theta(theta_key_idx, theta_old, theta_range, 
                       mixed_temp_matrix, mixed_phi_matrix, data_matrix,
                       current_params, emu_params, model_tt,
                       model_pp, rng_key, lmax, lmax_prime, nside, pix_width, theta_pix,
                       ell_grid, ells, theta_key, ombh2_accept_history, omch2_accept_history,
                       ns_accept_history, theta_MC_100_accept_history, logA_accept_history,
                       cphi_fid, cf_fid, qe_scalar, cn_scalar, mask_matrix, beam_matrix,
                       fourier_weights,
                       tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean,
                       pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean,
                       over_relaxation_num_samps = -1, sampler = "grid",
                       proposal_sigma = None, metropolis_num_steps = 1,
                       direction = None, direction_idx = None, pca_accept_history = None,
                       bounds_lo = None, bounds_hi = None,
                       joint_sigmas = None, joint_accept_history = None,
                       pol = "I", bb_is_zero = True, fixed_field = False,
                       freeze_g = False, cn_mix_scalar = None):
    """Sample a single cosmological parameter. sampler = "grid" (default) reconstructs the
    1D conditional and draws an independent inverse-CDF sample; sampler = "metropolis" takes
    a Gaussian random-walk Metropolis step of scale proposal_sigma around theta_old instead.
    Two full-vector paths override the above and return the updated FULL parameter vector,
    not a scalar (theta_key_idx/theta_old/theta_range are ignored): if direction is given
    (the PCA path), one Metropolis step along that scaled eigen-direction; if joint_sigmas
    is given, one joint Metropolis step kicking all five parameters independently."""

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

    #swap in the cached-CAMB predictors (same signature, emu_params ignored). The 5D grid
    #needs no per-parameter dispatch - it interpolates every column of params_batch, so the
    #same set of predictors serves whichever parameter is currently being sampled
    theta_name = PARAM_ORDER[int(theta_key_idx)]
    if USE_CAMB_GRID:
        predictors = get_camb_grid_predictors()
    elif USE_CAMB_SPLINE:
        predictors = wrap_predictors(*get_camb_spline_predictors(theta_name))
    else:
        predictors = wrap_predictors(predict_tt, predict_pp)

    #evaluate the mixed logpdf over an arbitrary (M, 5) batch of parameter vectors. all
    #the emulator/covariance work is a pure function of the batch, so build it once and
    #reuse it for the axis-aligned grid/Metropolis paths and the PCA direction path
    eval_logpdf_batch = make_eval_logpdf_batch(predictors, emu_params,
                                               mixed_temp_matrix, mixed_phi_matrix,
                                               data_matrix, cphi_fid, cf_fid,
                                               qe_scalar, cn_scalar, mask_matrix,
                                               beam_matrix, fourier_weights,
                                               nside, pix_width, theta_pix, ell_grid,
                                               pol = pol, bb_is_zero = bb_is_zero,
                                               fixed_field = fixed_field,
                                               freeze_g = freeze_g,
                                               cn_mix_scalar = cn_mix_scalar)

    #evaluate along one coordinate axis: tile current_params and overwrite the sampled column
    def eval_logpdf_grid(theta_grid):
        M = theta_grid.shape[0]
        params_batch = jnp.tile(current_params, (M, 1))
        params_batch = params_batch.at[:, theta_key_idx].set(theta_grid)
        return eval_logpdf_batch(params_batch)

    #PCA path: one Metropolis update of the FULL parameter vector along a scaled
    #eigen-direction. Returns the new 5-vector, unlike the scalar axis-aligned paths below
    if direction is not None:
        return metropolis_sample_direction(eval_logpdf_batch, current_params, direction,
                                           direction_idx, bounds_lo, bounds_hi,
                                           pca_accept_history, proposal_sigma, rng_key,
                                           num_steps = metropolis_num_steps)

    #joint-MH path: one Metropolis update of the FULL parameter vector with an independent
    #Gaussian kick per parameter. Also returns the new 5-vector
    if joint_sigmas is not None:
        return metropolis_sample_joint(eval_logpdf_batch, current_params, joint_sigmas,
                                       bounds_lo, bounds_hi, joint_accept_history,
                                       rng_key, num_steps = metropolis_num_steps)

    #adaptive recenter: the conditional peak (std ~1e-4) is far narrower than one node of
    #the initial grid (spacing ~5e-3), so a single fixed grid reconstructs the peak the
    #same wrong way every iteration -> a consistent directional bias that never averages
    #out. re-center a narrower grid on the peak and re-evaluate, so resolution near the
    #peak improves without a huge node count and node placement stays centered (symmetric)
    #each step. clamp windows to the original range so we never query the emulator outside
    #its validated bounds. NUM_REFINE / SHRINK are the tuning knobs: with SHRINK = 0.15 the
    #~0.25-wide theta_MC_100 grid narrows to spacing ~8e-4 then ~1e-4 over two passes
    NUM_REFINE = 0
    SHRINK = 0.1
    N = theta_range.shape[0]
    lo0, hi0 = float(theta_range[0]), float(theta_range[-1])

    #separate Metropolis random-walk path: propose locally around theta_old and accept/reject
    #instead of reconstructing and drawing from the full conditional. proposal_sigma bounds the
    #step size, which suppresses the spurious far jumps the grid path produces once f is sampled
    if sampler == "metropolis":
        return metropolis_sample_theta(eval_logpdf_grid, theta_old, lo0, hi0, theta_key,
                                       ombh2_accept_history, omch2_accept_history,
                                       ns_accept_history, theta_MC_100_accept_history, logA_accept_history,
                                       proposal_sigma, rng_key,
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
                                theta_old, over_relaxation_num_samps,
                                param_name = theta_name)
    return theta_new

#sample single parameter "theta" via inverse CDF
#@partial(jax.jit, static_argnames = ["over_relaxation_num_samps"])
def grid_and_sample(logpdf_values, theta_values, sub_key, theta_old,
                    over_relaxation_num_samps = -1, param_name = "theta"):

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

        #-----------------------------------------------------
        # polynomial_coeffs = np.polyfit(xs, logpdfs, 2)
        # polynomial_func = np.poly1d(polynomial_coeffs)
        # interp_logpdfs = polynomial_func(xs)
        #-----------------------------------------------------

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
        plt.savefig(f"/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/{param_name}_distribution.png")
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

#CAMB-based replacement for the emulator predict_tt/predict_pp calls. Given a parameter
#vector [theta_MC_100, logA, ns, ombh2, omch2], run CAMB (through the pure_callback in
#simulate.py, so it stays JIT-compatible) and return the unlensed-scalar TT and
#lensing-potential PP Cls interpolated onto ells 2..lmax, mirroring load_sim's
#cls["scalar_TT"] / cls["phi"]. The non-sampled CAMB parameters use the same fixed values
#as simulate.load_sim's defaults.
def _camb_cls_lcdm(current_params, lmax, lmax_prime):
    cosmomc_theta = current_params[PARAM_INDEX["theta_MC_100"]] / 100
    As = jnp.exp(current_params[PARAM_INDEX["logA"]]) * 1e-10
    ns = current_params[PARAM_INDEX["ns"]]
    ombh2 = current_params[PARAM_INDEX["ombh2"]]
    omch2 = current_params[PARAM_INDEX["omch2"]]

    #fixed (non-sampled) parameters: H0=None, r=0, mnu=0.06, tau=0.05, nt=0, k_pivot=0.05, Alens=1
    unlensed_scalar, tensor, total, lens_potential = _camb_via_callback(
        None, ombh2, omch2, cosmomc_theta, 0.0, 0.06, 0.05, As, 0, ns,
        lmax_prime, 0.05, 1
    )
    cls = _extract_all_cls(unlensed_scalar, tensor, total, lens_potential, lmax, lmax_prime)
    return cls["scalar_TT"], cls["phi"]

#jitted core of the per-iteration covariance/mixing recompute. Splits static shape/emulator
#args (predictors/nside/pix_width/lmax) from dynamic arrays so the internal control-flow
#(covar_matrix_from_cls / interpolate_cls conds) compiles once and is cached across Gibbs
#iterations, instead of re-tracing eagerly every call (the "<lambda> for pjit" log flood).
#predictors is static (a namedtuple of hashable closures created once per process).
#
#With refresh_qe the quadratic-estimate norm is ALSO rebuilt here, from the interpolated
#lensed Cls at the current theta, so the G matrix and the phi HMC mass matrix
#(pinv(G)^2 (pinv(Cphi) + pinv(Nphi))) track theta continuously instead of being frozen
#at the initial cosmology; without it the passed-in qe_scalar is returned unchanged
@partial(jax.jit, static_argnames = ["predictors", "nside", "pix_width", "lmax",
                                     "lmax_prime", "refresh_qe"])
def _recompute_cosmo_matrices(current_params, emu_params, ell_grid, ells,
                              cphi_fid, cf_fid, qe_scalar, cn_scalar,
                              mask_matrix, beam_matrix,
                              predictors, nside, pix_width, lmax, lmax_prime,
                              refresh_qe = False):
    x = current_params[None, :]
    cl_tt = predictors.tt(emu_params, x)[0]
    cl_pp = predictors.pp(emu_params, x)[0]
    #ells sized to the predictor output (emulator: 2..4000, CAMB spline cache: 2..3999)
    emul_ells = jnp.arange(2, 2 + cl_tt.shape[-1])

    cf = covar_matrix_from_cls(nside, pix_width,
                               ell_grid, emul_ells,
                               cl_tt, origin_value = 0)
    cphi = covar_matrix_from_cls(nside, pix_width,
                                 ell_grid, emul_ells,
                                 cl_pp, origin_value = 0)

    if refresh_qe:
        cl_tt_lensed = predictors.tt_lensed(emu_params, x)[0]
        cfl = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                    cl_tt_lensed, origin_value = 0)
        qe = scalar_quadratic_estimate(cn_scalar, cf, cfl,
                                       mask_matrix, beam_matrix, pix_width) / NPHI_FAC
    else:
        qe = qe_scalar

    g = (get_g_matrix_lcdm(cphi_fid, cphi, qe, cn_scalar))
    #d = get_d_tt_matrix(cf, jnp.zeros_like(cf), cn_scalar, 1, 1)
    d = (get_d_tt_matrix(cf, cf_fid, cn_scalar))
    return g, d, cf, cphi, qe

#TEB counterpart: TT/TE/EE/BB covariance stacks (TE reconstructed from the te_rho
#correlation-ratio spline), the coupled-block D matrix (identity B block on the r = 0
#grids), and the POLARIZATION quadratic estimate for the refresh_qe path. Returned d and
#cf are (4, nside, nside//2 + 1) stacks in (TT, TE, EE, BB) order, matching
#op_matrix_stack
@partial(jax.jit, static_argnames = ["predictors", "nside", "pix_width", "lmax",
                                     "lmax_prime", "refresh_qe", "bb_is_zero"])
def _recompute_cosmo_matrices_teb(current_params, emu_params, ell_grid, ells,
                                  cphi_fid, qe_scalar, cn_stack,
                                  mask_matrix, beam_matrix,
                                  predictors, nside, pix_width, lmax, lmax_prime,
                                  refresh_qe = False, bb_is_zero = True):
    x = current_params[None, :]
    cl_tt = predictors.tt(emu_params, x)[0]
    cl_ee = predictors.ee(emu_params, x)[0]
    te_rho = predictors.te_rho(emu_params, x)[0]
    cl_pp = predictors.pp(emu_params, x)[0]
    emul_ells = jnp.arange(2, 2 + cl_tt.shape[-1])
    cn_tt, cn_te, cn_ee, cn_bb = cn_stack[0], cn_stack[1], cn_stack[2], cn_stack[3]

    cf_tt = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                  cl_tt, origin_value = 0)
    cf_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                  cl_ee, origin_value = 0)
    cf_te = te_covar_from_rho(te_rho, emul_ells, ell_grid, cf_tt, cf_ee)
    if bb_is_zero:
        cf_bb = jnp.zeros_like(cf_tt)
    else:
        cl_bb = predictors.bb(emu_params, x)[0]
        cf_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                      cl_bb, origin_value = 0)
    cphi = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                 cl_pp, origin_value = 0)

    if refresh_qe:
        cl_ee_lensed = predictors.ee_lensed(emu_params, x)[0]
        cl_bb_lensed = predictors.bb_lensed(emu_params, x)[0]
        cfl_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                       cl_ee_lensed, origin_value = 0)
        cfl_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                       cl_bb_lensed, origin_value = 0)
        qe = polar_quadratic_estimate(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb,
                                      mask_matrix, mask_matrix, beam_matrix, beam_matrix,
                                      pix_width) / NPHI_FAC
    else:
        qe = qe_scalar

    g = get_g_matrix_lcdm(cphi_fid, cphi, qe, cn_tt)
    d_tt, d_te, d_ee, d_bb = get_d_teb_matrix(cf_tt, cf_te, cf_ee, cf_bb,
                                              cn_tt, cn_te, cn_ee, cn_bb)
    return (g, jnp.stack([d_tt, d_te, d_ee, d_bb]),
            jnp.stack([cf_tt, cf_te, cf_ee, cf_bb]), cphi, qe)

#Polarization-only counterpart: EE/BB covariance stack, the diagonal E/B D matrix
#(identity B block on the r = 0 grids) and the polarization quadratic estimate for the
#refresh_qe path. Returned d and cf are (2, nside, nside//2 + 1) stacks in (EE, BB)
#order, matching op_matrix_stack. No TT/TE predictor is ever evaluated
@partial(jax.jit, static_argnames = ["predictors", "nside", "pix_width", "lmax",
                                     "lmax_prime", "refresh_qe", "bb_is_zero"])
def _recompute_cosmo_matrices_eb(current_params, emu_params, ell_grid, ells,
                                 cphi_fid, qe_scalar, cn_stack,
                                 mask_matrix, beam_matrix,
                                 predictors, nside, pix_width, lmax, lmax_prime,
                                 refresh_qe = False, bb_is_zero = True):
    x = current_params[None, :]
    cl_ee = predictors.ee(emu_params, x)[0]
    cl_pp = predictors.pp(emu_params, x)[0]
    emul_ells = jnp.arange(2, 2 + cl_ee.shape[-1])
    cn_ee, cn_bb = cn_stack[0], cn_stack[1]

    cf_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                  cl_ee, origin_value = 0)
    if bb_is_zero:
        cf_bb = jnp.zeros_like(cf_ee)
    else:
        cl_bb = predictors.bb(emu_params, x)[0]
        cf_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                      cl_bb, origin_value = 0)
    cphi = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                 cl_pp, origin_value = 0)

    if refresh_qe:
        cl_ee_lensed = predictors.ee_lensed(emu_params, x)[0]
        cl_bb_lensed = predictors.bb_lensed(emu_params, x)[0]
        cfl_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                       cl_ee_lensed, origin_value = 0)
        cfl_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, emul_ells,
                                       cl_bb_lensed, origin_value = 0)
        qe = polar_quadratic_estimate(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb,
                                      mask_matrix, mask_matrix, beam_matrix, beam_matrix,
                                      pix_width) / NPHI_FAC
    else:
        qe = qe_scalar

    g = get_g_matrix_lcdm(cphi_fid, cphi, qe, cn_ee)
    d_ee, d_bb = get_d_eb_matrix(cf_ee, cf_bb, cn_ee, cn_bb)
    return g, jnp.stack([d_ee, d_bb]), jnp.stack([cf_ee, cf_bb]), cphi, qe

#Thin eager wrapper: unpacks args (mixed static/dynamic) and delegates to the jitted core
#for the dataset's polarization mode.
def get_new_cosmo_matrices(current_params, predictors, emu_params, args,
                           pol = "I", refresh_qe = False):
    """Compute (g, d, cf, cphi, qe) from a parameter vector. For pol = "IP" the d and cf
    entries are (4, ...) TT/TE/EE/BB stacks, for pol = "P" (2, ...) EE/BB stacks. The
    noise covariance handed to the jitted kernels is the MIXING one - inside them it
    only ever feeds D, G and the refreshed QE norm, never a likelihood term - so
    mixing_noise_uk_arcmin decoupling applies to every recompute automatically (falls
    back to the true noise when absent)."""
    cn_mix = args.get("mixing_noise_covariance", args["noise_covariance"])
    if pol == "IP":
        return _recompute_cosmo_matrices_teb(
            current_params, emu_params, args["ell_grid"], args["ells"],
            args["cphi_fid"],
            args["quadratic_estimate"].scalar_matrix,
            op_matrix_stack(cn_mix, pol),
            op_shared_matrix(args["mask"], pol), op_shared_matrix(args["beam"], pol),
            predictors, args["nside"], args["pix_width"],
            args["lmax"], args["lmax_prime"],
            refresh_qe = refresh_qe, bb_is_zero = camb_grid_bb_is_zero())
    if pol == "P":
        return _recompute_cosmo_matrices_eb(
            current_params, emu_params, args["ell_grid"], args["ells"],
            args["cphi_fid"],
            args["quadratic_estimate"].scalar_matrix,
            op_matrix_stack(cn_mix, pol),
            op_shared_matrix(args["mask"], pol), op_shared_matrix(args["beam"], pol),
            predictors, args["nside"], args["pix_width"],
            args["lmax"], args["lmax_prime"],
            refresh_qe = refresh_qe, bb_is_zero = camb_grid_bb_is_zero())
    return _recompute_cosmo_matrices(current_params, emu_params,
                                     args["ell_grid"], args["ells"],
                                     args["cphi_fid"], args["cf_fid"],
                                     args["quadratic_estimate"].scalar_matrix,
                                     cn_mix.scalar_matrix,
                                     op_shared_matrix(args["mask"], pol),
                                     op_shared_matrix(args["beam"], pol),
                                     predictors, args["nside"], args["pix_width"],
                                     args["lmax"], args["lmax_prime"],
                                     refresh_qe = refresh_qe)

#Lighter-weight version of the above method to just compute the field covariance(s) and
#not D, G, Cphi... Eager-only (uses the _covar_or_zeros value check), which is fine for
#its single call site in add_starting_matrices_to_args
def get_new_cf_matrix(current_params, predictors, emu_params, args, pol = "I"):
    """Compute the field covariance from a parameter vector. pol = "I" returns the TT
    matrix; pol = "IP" returns the (cf_tt, cf_te, cf_ee, cf_bb) tuple; pol = "P" the
    (cf_ee, cf_bb) pair."""
    x = current_params[None, :]
    if pol == "P":
        cl_ee = predictors.ee(emu_params, x)[0]
        cl_bb = predictors.bb(emu_params, x)[0]
        emu_ells = jnp.arange(2, 2 + cl_ee.shape[-1])
        cf_ee = covar_matrix_from_cls(args["nside"], args["pix_width"],
                                      args["ell_grid"], emu_ells,
                                      cl_ee, origin_value = 0)
        cf_bb = _covar_or_zeros(args["nside"], args["pix_width"],
                                args["ell_grid"], emu_ells,
                                cl_bb, origin_value = 0)
        return cf_ee, cf_bb
    cl_tt = predictors.tt(emu_params, x)[0]
    #ells sized to the predictor output (emulator: 2..4000, CAMB spline cache: 2..3999)
    emu_ells = jnp.arange(2, 2 + cl_tt.shape[-1])

    cf_tt = covar_matrix_from_cls(args["nside"], args["pix_width"],
                                  args["ell_grid"], emu_ells,
                                  cl_tt, origin_value = 0)
    if pol == "IP":
        cl_ee = predictors.ee(emu_params, x)[0]
        cl_bb = predictors.bb(emu_params, x)[0]
        te_rho = predictors.te_rho(emu_params, x)[0]
        cf_ee = covar_matrix_from_cls(args["nside"], args["pix_width"],
                                      args["ell_grid"], emu_ells,
                                      cl_ee, origin_value = 0)
        cf_te = te_covar_from_rho(te_rho, emu_ells, args["ell_grid"], cf_tt, cf_ee)
        cf_bb = _covar_or_zeros(args["nside"], args["pix_width"],
                                args["ell_grid"], emu_ells,
                                cl_bb, origin_value = 0)
        return cf_tt, cf_te, cf_ee, cf_bb
    return cf_tt

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
    #args["lmax_prime"] = min(lmax, EMULATOR_MAX_ELL)
    args["lmax_prime"] = min(lmax, 4000)
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

#replace the four independent blocks of a symmetric BlockTEB operator (ET mirrors TE)
def _replace_teb_blocks(op, tt, te, ee, bb):
    return op.replace(matrix_TT = tt, matrix_TE = te, matrix_ET = te,
                      matrix_EE = ee, matrix_BB = bb)

#replace the two diagonal blocks of a DiagonalEB operator (pol = "P")
def _replace_eb_blocks(op, ee, bb):
    return op.replace(matrix_EE = ee, matrix_BB = bb)

def add_starting_matrices_to_args(args, data_set, noise_level, param_init, current_params,
                                  predictors, emu_params, pol = "I",
                                  mixing_noise_uk_arcmin = None):

    #We can comfortably reuse the noise covariance, mask, and beam
    #from the ground truth data set for our sampling algorithm
    args["noise_covariance"] = data_set.noise_covariance
    args["mask"] = data_set.mask
    args["beam"] = data_set.beam

    #mixing-noise decoupling: everything on the MIXING / preconditioning side (the D and
    #G matrices, the QE norm feeding G and the phi HMC mass matrix) is built from this
    #covariance, while the likelihood keeps the true noise_covariance. G/D are pure
    #reparametrizations, so this changes mixing speed only, never the posterior - it
    #lets a low-noise likelihood run with the mixing tuned as if the noise were e.g.
    #5 uk-arcmin, where the parametrization is known to mix well. Implemented as a
    #FLOOR by scaling the true covariance (exact for the white noise currently used:
    #Cn scales as uk_arcmin^2), so mixing_noise_uk_arcmin <= noise_level is a no-op
    if mixing_noise_uk_arcmin is None:
        args["mixing_noise_covariance"] = args["noise_covariance"]
    else:
        mix_scale = max(1.0, (mixing_noise_uk_arcmin / noise_level)**2)
        nc = args["noise_covariance"]
        if pol == "IP":
            args["mixing_noise_covariance"] = _replace_teb_blocks(
                nc, mix_scale * nc.matrix_TT, mix_scale * nc.matrix_TE,
                mix_scale * nc.matrix_EE, mix_scale * nc.matrix_BB)
        elif pol == "P":
            args["mixing_noise_covariance"] = _replace_eb_blocks(
                nc, mix_scale * nc.matrix_EE, mix_scale * nc.matrix_BB)
        else:
            #cn_tt, _, _, _ = noise_cls(args["lmax_prime"], mixing_noise_uk_arcmin, 
            #                           beam_fwhm=0, l_knee=100, alpha_knee=3)
            args["mixing_noise_covariance"] = nc.replace(
                scalar_matrix = mix_scale * nc.scalar_matrix)
    #We also need to store the fiducial phi covariance matrix which contains the
    #ground truth information needed for the G mixing matrix
    args["cphi_fid"] = data_set.phi_covariance.scalar_matrix
    #the fiducial field covariance, stacked (TT, EE, BB) for pol = "IP" so it can be
    #handed straight to the jitted theta kernels
    args["cf_fid"] = op_matrix_stack(data_set.field_covariance, pol)

    #Note the QE depends on Cfl which will be affected by the ground truth
    #cosmological parameters therefore using the ground truth QE norm is somewhat
    #cheating since we are using extra information besides just the data map...
    #We should therefore initialize the QE norm to be computed based on our initial parameter guesses
    param_init = emul_2_camb_naming(param_init)
    #the lensed covariances feeding the QE norm (and through it the G matrix) must come
    #from the same Cl model as the rest of the pipeline: CAMB when the spline/grid is on
    initial_cond = load_sim(data_set.nside, data_set.theta_pix, pol,
                            np.random.randint(0, 2**31), **param_init,
                            uk_arcmin_t = noise_level, r = 0, nt = 0,
                            use_emulator_cls = not (USE_CAMB_SPLINE or USE_CAMB_GRID))
    cf = get_new_cf_matrix(current_params, predictors, emu_params, args, pol = pol)
    if pol == "IP":
        cf_tt, cf_te, cf_ee, cf_bb = cf
        args["field_covariance"] = _replace_teb_blocks(data_set.field_covariance,
                                                       cf_tt, cf_te, cf_ee, cf_bb)
        #the QE norm only ever serves the mixing side (G and the phi mass matrix), so it
        #is built at the MIXING noise level
        qe_matrix = polar_quadratic_estimate(
            cf_ee, cf_bb,
            initial_cond.lensed_field_covariance.matrix_EE,
            initial_cond.lensed_field_covariance.matrix_BB,
            args["mixing_noise_covariance"].matrix_EE,
            args["mixing_noise_covariance"].matrix_BB,
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
            args["mixing_noise_covariance"].matrix_EE,
            args["mixing_noise_covariance"].matrix_BB,
            op_shared_matrix(args["mask"], pol), op_shared_matrix(args["mask"], pol),
            op_shared_matrix(args["beam"], pol), op_shared_matrix(args["beam"], pol),
            data_set.pix_width) / NPHI_FAC
    else:
        args["field_covariance"] = data_set.field_covariance.replace(scalar_matrix = cf)
        qe_matrix = scalar_quadratic_estimate(args["mixing_noise_covariance"].scalar_matrix,
                                              args["field_covariance"].scalar_matrix,
                                              initial_cond.lensed_field_covariance.scalar_matrix,
                                              args["mask"].scalar_matrix,
                                              args["beam"].scalar_matrix,
                                              data_set.pix_width) / NPHI_FAC
    args["quadratic_estimate"] = data_set.quadratic_estimate.replace(scalar_matrix = qe_matrix)

    #We must also initialize the mixing D & G matrices and Cphi
    #to the proper values according to our starting guesses for the cosmological parameters
    g, d, _, cphi, _ = get_new_cosmo_matrices(current_params, predictors,
                                              emu_params, args, pol = pol)
    args["phi_covariance"] = data_set.phi_covariance.replace(scalar_matrix = cphi)
    args["mixing_g"] = data_set.mixing_g.replace(scalar_matrix = (g))
    if pol == "IP":
        args["mixing_d"] = _replace_teb_blocks(data_set.mixing_d,
                                               d[0], d[1], d[2], d[3])
    elif pol == "P":
        args["mixing_d"] = _replace_eb_blocks(data_set.mixing_d, d[0], d[1])
    else:
        args["mixing_d"] = data_set.mixing_d.replace(scalar_matrix = (d))

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

def update_args_after_sample(current_params, predictors, emu_params, args,
                             pol = "I", refresh_qe = False, fixed_field = False,
                             freeze_g = False):
    g, d, cf, cphi, qe = get_new_cosmo_matrices(current_params, predictors,
                                                emu_params, args, pol = pol,
                                                refresh_qe = refresh_qe)
    args["phi_covariance"] = args["phi_covariance"].replace(scalar_matrix = cphi)
    if pol == "IP":
        args["field_covariance"] = _replace_teb_blocks(args["field_covariance"],
                                                       cf[0], cf[1], cf[2], cf[3])
    elif pol == "P":
        args["field_covariance"] = _replace_eb_blocks(args["field_covariance"],
                                                      cf[0], cf[1])
    else:
        args["field_covariance"] = args["field_covariance"].replace(scalar_matrix = cf)
    #fixed_field: the mixing matrices were pinned to the identity at setup and MUST stay
    #theta-independent - re-mixing pinned fields at the current theta turns the chain
    #into a fixed-point iteration biased away from the conditional peak. freeze_g pins
    #ONLY G the same way (phi is the pinned variable there); D keeps tracking theta
    #because f is genuinely resampled each sweep
    if not fixed_field:
        if pol == "IP":
            args["mixing_d"] = _replace_teb_blocks(args["mixing_d"],
                                                   d[0], d[1], d[2], d[3])
        elif pol == "P":
            args["mixing_d"] = _replace_eb_blocks(args["mixing_d"], d[0], d[1])
        else:
            args["mixing_d"] = args["mixing_d"].replace(scalar_matrix = (d))
        if not freeze_g:
            args["mixing_g"] = args["mixing_g"].replace(scalar_matrix = (g))
    #with refresh_qe the returned qe is the freshly interpolated-lensed-Cl norm at the
    #new theta, so the phi mass matrix and G track theta; without it qe passes through
    args["quadratic_estimate"] = args["quadratic_estimate"].replace(scalar_matrix = qe)
    return args

#algorithm to jointly sample cosmological parameters
def sample_joint(data_set, param_init, param_ranges, should_sample, noise_level, 
                 f_ground, phi_ground, iters_per_chain = 500,
                 num_burn_in_fix_theta = 100, 
                 num_burn_in_fix_ombh2 = 0,
                 num_burn_in_always_accept = 0, seed = None, map = None,
                 over_relaxation_num_samps = -1, lmax = 4000,
                 metropolis_num_steps = 1, phi_sampler = "hmc", pcn_beta = 0.1,
                 ghmc_alpha = 0.3,
                 ghmc_step_size = 0.06, ghmc_num_steps = 1, ghmc_warmup_hmc_iters = 0,
                 use_pca = False, save_pca = False, pca_path = THETA_PCA_PATH,
                 pca_proposal_sigma = 2.4, pca_burn_in = 0,
                 use_joint_mh = False, joint_proposal_sigmas = None,
                 use_ghmc_theta = False, ghmc_theta_step_size = 0.1,
                 ghmc_theta_alpha = 0.1, ghmc_theta_delta = 0.1,
                 ghmc_theta_target_accept = 0.95, ghmc_theta_adapt_sweeps = 100,
                 theta_fd_step = 1e-3,
                 refresh_qe = None, fixed_field_theta = False,
                 freeze_g_mixing = False, mixing_noise_uk_arcmin = None):

    #polarization mode follows the dataset: DataSetTEB samples T + E + B (T and E
    #correlated through the physical TE block), DataSetEB is the polarization-only
    #E + B path (FlatS2 fields, DiagonalEB operators - no T, no TE), DataSetT the
    #T-only path
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
        raise ValueError(f"pol = '{pol}' requires USE_CAMB_GRID - the emulator and the "
                         "1D CAMB caches have no EE/BB spectra")
    #refresh_qe: recompute the quadratic-estimate norm from the interpolated LENSED Cls
    #after every theta update, so the G matrix and the phi HMC mass matrix track theta.
    #Defaults to on whenever the 5D grid (the only Cl source carrying lensed spectra) is
    #active; a pre-lensed-Cl grid file raises with a regeneration pointer on first use
    if refresh_qe is None:
        refresh_qe = USE_CAMB_GRID
    if refresh_qe and not USE_CAMB_GRID:
        raise ValueError("refresh_qe requires USE_CAMB_GRID - only the 5D grid carries "
                         "the lensed TT/EE/BB spectra the QE norm is built from")
    if freeze_g_mixing and fixed_field_theta:
        raise ValueError("freeze_g_mixing and fixed_field_theta are mutually exclusive - "
                         "fixed_field_theta already pins BOTH mixing matrices to the "
                         "identity, freeze_g_mixing pins only G and resamples f")
    bb_is_zero = camb_grid_bb_is_zero() if USE_CAMB_GRID else True

    #full-vector theta-update setup (PCA eigen-directions and/or plain joint MH), checked
    #before any heavy loading so a bad configuration fails instantly. All of these flags
    #require every parameter to be sampled: full-vector moves touch all five at once
    #(use_pca / use_joint_mh), and a frozen parameter has zero variance so its covariance
    #row would be singular (save_pca)
    if use_pca and use_joint_mh:
        raise ValueError("use_pca and use_joint_mh are mutually exclusive theta updates")
    if use_pca or save_pca:
        not_sampled = [name for name in PARAM_ORDER if not should_sample.get(name, False)]
        if not_sampled:
            raise ValueError(f"use_pca/save_pca require sampling all parameters, but "
                             f"{not_sampled} have should_sample = False")
    if use_pca or use_joint_mh:
        # if not USE_CAMB_GRID:
        #     raise ValueError("use_pca/use_joint_mh require USE_CAMB_GRID - full-vector "
        #                      "proposals move all five parameters at once, which the 1D "
        #                      "caches cannot evaluate")
        theta_bounds_lo = np.array([float(param_ranges[name][0]) for name in PARAM_ORDER])
        theta_bounds_hi = np.array([float(param_ranges[name][-1]) for name in PARAM_ORDER])
    if use_pca:
        #row i = eigenvector i scaled to one conditional standard deviation
        pca_directions = load_pca_directions(pca_path)
        pca_accept_histories = [[] for _ in range(len(PARAM_ORDER))]
    if use_joint_mh:
        #unlike use_pca, joint MH works on any subset of parameters: should_sample = False
        #entries get proposal sigma 0, so they receive no kick and never move off their
        #param_init values
        sampled_names = [name for name in PARAM_ORDER if should_sample.get(name, False)]
        if not sampled_names:
            raise ValueError("use_joint_mh requires at least one sampled parameter")
        if joint_proposal_sigmas is None:
            raise ValueError("use_joint_mh requires joint_proposal_sigmas - a dict with "
                             "one proposal sigma per sampled parameter")
        missing = [name for name in sampled_names if name not in joint_proposal_sigmas]
        if missing:
            raise ValueError(f"joint_proposal_sigmas is missing {missing}")
        joint_sigmas_vec = np.array([float(joint_proposal_sigmas[name])
                                     if should_sample.get(name, False) else 0.0
                                     for name in PARAM_ORDER])
        joint_accept_history = []
    if use_ghmc_theta:
        #gradient-based theta update on the differentiable 5D grid spline, replacing
        #the Metropolis kick: use_ghmc_theta runs ONE leapfrog step per sweep with
        #persistent momentum (generalized HMC, mirroring gibbs_sample_phi_ghmc). Like
        #use_joint_mh it works on any subset of parameters - frozen entries stay at
        #their param_init values. (A blackjax NUTS variant used to live here too; it
        #was removed because naive autodiff of mixed_logpdf is wrong w.r.t. the G
        #mixing matrix and blackjax NUTS never worked for this problem - GHMC with the
        #finite-difference logdensity gradient is the surviving gradient-based path)
        if use_pca or use_joint_mh:
            raise ValueError("use_ghmc_theta is mutually exclusive with "
                             "use_pca/use_joint_mh - pick one theta update")
        if not USE_CAMB_GRID:
            raise ValueError("use_ghmc_theta requires USE_CAMB_GRID - the "
                             "logdensity differentiates through the 5D grid spline")
        ghmc_sampled_names = [name for name in PARAM_ORDER
                              if should_sample.get(name, False)]
        if not ghmc_sampled_names:
            raise ValueError("use_ghmc_theta requires at least one sampled parameter")
        ghmc_sampled_idx = tuple(PARAM_INDEX[name] for name in ghmc_sampled_names)
        ghmc_theta_accept_history = []

    #Prepare the JIT-friendly emulator (build models once, extract weights)
    #emulator = cambemul.loademul("/resnick/groups/wugroup/zblood/cmb_lensing/camb_emulator")
    emulator = cambemul.loademul("/home/zane-blood/Desktop/cmb_lensing/camb_emulator")
    (predict_tt, predict_pp, emu_params, emu_meta, model_tt, model_pp,
     tt_x_mean, tt_x_std, tt_t_mean, tt_t_std, tt_pca_basis_T, tt_pca_mean,
     pp_x_mean, pp_x_std, pp_t_mean, pp_t_std, pp_pca_basis_T, pp_pca_mean) = prepare_emulator_jax(emulator)

    #swap in the cached-CAMB predictors for every covariance recompute
    #(add_starting_matrices_to_args, set_initial_ds_conditions, update_args_after_sample).
    #the 5D grid interpolates all five parameters at once, so there is no restriction on
    #how many may be sampled - unlike the 1D caches below
    if USE_CAMB_GRID:
        predictors = get_camb_grid_predictors()
        #the covariance recomputes are NOT proposals - a NaN here poisons the run rather
        #than rejecting a step - so fail loudly if the starting point is outside the grid
        #(most likely a theta_MC_100 unreachable at this ombh2/omch2, since the grid is
        #laid out in H0 over a fixed range)
        start_tt = predictors.tt(None, jnp.array([[param_init[k] for k in PARAM_ORDER]]))
        if not bool(jnp.all(jnp.isfinite(start_tt))):
            raise ValueError(
                f"the starting parameters {dict((k, param_init[k]) for k in PARAM_ORDER)} "
                f"fall outside the 5D CAMB grid, so the initial covariances would be NaN. "
                f"Check that theta_MC_100 is reachable at this ombh2/omch2 - see "
                f"theta_grid in {CAMB_GRID_PATH}")
    elif USE_CAMB_SPLINE:
        #the 1D cache fixes every other parameter at ground truth, so exactly one
        #parameter may be sampled - its spline serves all recomputes since only its
        #column ever moves
        sampled_names = [name for name in PARAM_ORDER if should_sample.get(name, False)]
        if len(sampled_names) != 1:
            raise ValueError(f"USE_CAMB_SPLINE requires exactly one sampled parameter "
                             f"(the caches are 1D), got {sampled_names}")
        predictors = wrap_predictors(*get_camb_spline_predictors(sampled_names[0]))
    else:
        predictors = wrap_predictors(predict_tt, predict_pp)

    #Set the initial parameter values to the user specified starting guesses
    param_vals = {}
    for theta, theta_val in param_init.items():
        param_vals[theta] = [theta_val]
    #Also store the current parameter values in a JAX array
    current_params = jnp.array([param_init[k] for k in PARAM_ORDER], dtype = jnp.float64)

    #Add data that will be used throughout the sampling algorithm to an args dictionary
    args = {}
    args = add_metadata_to_args(args, data_set, lmax)
    #mixing_noise_uk_arcmin: build D, G, the QE norm and the phi mass matrix at this
    #EFFECTIVE noise level while the likelihood keeps the true noise_level. Exact for
    #any value (G/D are reparametrizations whose Jacobians mixed_logpdf pays); useful
    #because the mixed parametrization is known to mix well around ~5 uk-arcmin and to
    #stall at much lower noise. Implemented as a floor, so values <= noise_level are a
    #no-op. None disables the decoupling entirely
    if mixing_noise_uk_arcmin is not None:
        print(f"mixing-noise decoupling ON: D/G/QE built at "
              f"{mixing_noise_uk_arcmin} uk-arcmin, likelihood at {noise_level}")
    args = add_starting_matrices_to_args(args, data_set, noise_level,
                                         param_init, current_params,
                                         predictors, emu_params, pol = pol,
                                         mixing_noise_uk_arcmin = mixing_noise_uk_arcmin)
    #the stacked mixing-noise matrices handed to every theta path (equal to the true
    #noise stack when decoupling is off); constant for the whole run
    cn_mix_stack = op_matrix_stack(args["mixing_noise_covariance"], pol)

    #fixed_field_theta: validation mode for chains that pin f and phi at ground truth
    #(the f/phi sampling steps commented out below). The theta step then targets the
    #UNMIXED conditional p(theta | f, phi, d): the mixing matrices are pinned to the
    #IDENTITY here (and left untouched by update_args_after_sample), and every theta
    #path builds its logpdf batch with identity per-row G/D (make_eval_logpdf_batch's
    #fixed_field flag), so logdet(G) = logdet(D) = 0 and unmix returns the pinned fields
    #for every candidate theta. Without this, re-mixing the pinned fields with
    #D/G(theta_current) after each accepted theta makes the chain a stochastic
    #fixed-point iteration - NOT MCMC on p(theta | f, phi, d) - with an attractor away
    #from the conditional peak (measured at +1.0 TRAINING sigma in theta_MC_100 for
    #pol = "IP", where the coupled T/E D block destabilizes the truth fixed point; the
    #T-only drift map is contractive, which is why "I" chains never showed it). Do NOT
    #enable while actually sampling f or phi - the mixed parametrization exists because
    #unmixed phi HMC is poorly conditioned, and full Gibbs with theta-dependent mixing
    #is exact when the fields are genuinely resampled each sweep
    if fixed_field_theta:
        ones = jnp.ones_like(args["mixing_g"].scalar_matrix)
        args["mixing_g"] = args["mixing_g"].replace(scalar_matrix = ones)
        if pol == "IP":
            args["mixing_d"] = _replace_teb_blocks(args["mixing_d"], ones,
                                                   jnp.zeros_like(ones), ones, ones)
        elif pol == "P":
            args["mixing_d"] = _replace_eb_blocks(args["mixing_d"], ones, ones)
        else:
            args["mixing_d"] = args["mixing_d"].replace(scalar_matrix = ones)

    #freeze_g_mixing: validation mode for chains that pin ONLY phi at ground truth while
    #genuinely resampling f each sweep (the loop below skips the phi sampling and unmix
    #steps under this flag; the f draw stays). A theta-dependent mixing of a variable is
    #exact iff that variable is resampled from its conditional each sweep, so D may keep
    #tracking theta (its
    #Jacobian is paid in mixed_logpdf and f is freshly drawn) but G must be pinned to
    #the IDENTITY here (and left untouched by update_args_after_sample): re-mixing the
    #pinned phi with G(theta_current) would hand the data term a likelihood bonus at
    #theta = theta_current through the phi channel - the same self-reinforcing
    #fixed-point pathology fixed_field_theta guards against, just via G instead of D.
    #With identity G the lensing in mix/unmix acts at the pinned phi for every candidate
    #theta, so its Jacobian is constant and the chain is exact MCMC on
    #p(f, theta | phi = phi_ground, d) - a data-coupled validation target whose theta
    #marginal conditions on perfect phi knowledge (tighter than the full-Gibbs marginal)
    if freeze_g_mixing:
        ones = jnp.ones_like(args["mixing_g"].scalar_matrix)
        args["mixing_g"] = args["mixing_g"].replace(scalar_matrix = ones)

    #change data_set covariance matrices from ground truth to initial
    #starting point in cosmological parameter space
    data_set = set_initial_ds_conditions(data_set, args)
    data_field = data_set.data

    #Use a seed to get reproduceable results if so desired
    # if seed is not None:
    #     sub_key = jax.random.PRNGKey(seed)
    # else:
    #sub_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
    sub_key = jax.random.PRNGKey(seed)
    rng_key, sub_key = jax.random.split(sub_key)
    #sub_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
    #NOTE that map_joint for 256-square, theta_pix = 1 arcmin map took almost 12 minutes to finish...
    #choose the starting point for (f, phi) in (f, phi, theta) cosmological parameter space
    #_, phi_map = get_starting_f_and_phi("ZEROES", "MAP", data_set, args, rng_key)
    #np.savez("/home/zane-blood/Desktop/MAP_estimate.npz", phi_map.scalar_matrix)
    #phi_map = data_set.phi.replace(scalar_matrix = np.load("/home/zane-blood/Desktop/MAP_estimate.npz")["arr_0"])
    #_, phi_map = map_joint(data_set)
    #np.savez("/home/zane-blood/Desktop/MAP_estimate.npz", phi_map.scalar_matrix)
    #phi_map = data_set.phi.replace(scalar_matrix = np.load("/home/zane-blood/Desktop/MAP_estimate.npz")["arr_0"])
    # inv_mass_matrix = pinv(pinv(args["phi_covariance"]) + pinv(args["quadratic_estimate"]))
    # phi_rng_matrix = field_from_covar_single_key(data_set.data.nside, 
    #                                              inv_mass_matrix.scalar_matrix, 
    #                                              rng_key)
    # phi_rng = phi_map.replace(scalar_matrix = jfft.rfft2(phi_rng_matrix))
    # phi = (phi_rng + phi_map)
    mixed_logpdf_values = []

    #phi is a scalar field regardless of polarization mode, so its template comes from
    #data_set.phi, never from the (possibly FlatS02) data field
    zeroes = 0*data_set.phi
    phi = zeroes
    #the f-sampling template must be shaped like the unlensed FIELD instead - a FlatS02
    #for pol = "IP" - both as the drawn-realization struct and as the Wiener CG start
    field_zeroes = 0*data_set.unlensed_field
    #freeze_g_mixing conditions the whole chain on phi = phi_ground, so the pinned value
    #must be the ground truth regardless of the starting-point choice above
    if freeze_g_mixing:
        phi = phi_ground

    #the stacked data matrices handed to every jitted theta kernel ((3, ...) T/E/B for
    #pol = "IP", the plain matrix for "I")
    data_stack = field_matrix_stack(data_field, pol)

    ghmc_u = None
    if phi_sampler == "ghmc":
        rng_key, sub_key = jax.random.split(sub_key)
        ghmc_u = data_set.phi.replace(scalar_matrix = jfft.rfft2(
            jax.random.normal(rng_key, shape = (data_field.nside, data_field.nside))))

    #GHMC theta update: build the differentiable logdensity machinery once.
    #blackjax is imported lazily so the other theta paths keep working without it
    if use_ghmc_theta:
        import blackjax
        ghmc_predictors = get_camb_grid_predictors_grad()
        #run-constant matrices for the theta logdensity (the per-sweep mixed matrices and
        #the QE norm are kernel ARGUMENTS instead - the QE moves with theta when
        #refresh_qe is on, so baking it into the trace would go stale)
        ghmc_cn_stack = op_matrix_stack(args["noise_covariance"], pol)
        ghmc_mask_matrix = op_shared_matrix(args["mask"], pol)
        ghmc_beam_matrix = op_shared_matrix(args["beam"], pol)

        #GHMC runs in whitened coordinates theta_sampled = T @ u. The raw parameter
        #scales span ~1e-4..1e-1 and are strongly correlated; sampling them directly
        #with an identity mass matrix crashes the step size (observed: 3e-8). T comes
        #from the pilot covariance's Cholesky (or TRAINING_SIGMA without a pilot),
        #making the identity mass matrix the right geometry from the first step - no
        #warmup phase needed. The flat prior is uniform so the constant linear
        #transform needs no Jacobian term
        ghmc_scales = jnp.array([TRAINING_SIGMA[PARAM_ORDER[i]]
                                 for i in ghmc_sampled_idx])
        ghmc_T, ghmc_T_inv = load_theta_whitening(pca_path, ghmc_sampled_idx,
                                                  np.asarray(ghmc_scales))

        def ghmc_logdensity_from(mixed_temp_matrix, mixed_phi_matrix, frozen_params,
                                 qe_matrix):
            """Logdensity over the SAMPLED parameter sub-vector at the given mixed
            fields. Everything else it closes over (data, fiducial covariances, mask,
            beam, noise) is constant for the whole run, so it is safe to bake into a jit
            trace; the per-sweep mixed matrices, the frozen-parameter template and the
            QE norm (which tracks theta under refresh_qe) are arguments precisely
            because they are not."""
            eval_logpdf_batch = make_eval_logpdf_batch(
                ghmc_predictors, None,
                mixed_temp_matrix, mixed_phi_matrix, data_stack,
                args["cphi_fid"], args["cf_fid"],
                qe_matrix, ghmc_cn_stack,
                ghmc_mask_matrix, ghmc_beam_matrix,
                data_field.fourier_weights, args["nside"], args["pix_width"],
                data_field.theta_pix, args["ell_grid"],
                pol = pol, bb_is_zero = bb_is_zero,
                fixed_field = fixed_field_theta,
                freeze_g = freeze_g_mixing,
                cn_mix_scalar = cn_mix_stack)

            n_dim = len(ghmc_sampled_idx)

            def embed(us):
                #(M, n_dim) whitened positions -> (M, 5) full parameter vectors
                xs = us @ ghmc_T.T
                full = jnp.tile(frozen_params, (us.shape[0], 1))
                for k, i in enumerate(ghmc_sampled_idx):
                    full = full.at[:, i].set(xs[:, k])
                return full

            #central-difference gradient in the whitened coordinates, attached via
            #jax.custom_vjp - deliberately NOT reverse-mode autodiff: plain
            #jax.grad(mixed_logpdf) is WRONG with respect to the G mixing matrix
            #(measured ~3000x too large with a flipped sign; the cotangent path through
            #unmix/the hand-written lensing adjoint was never built for naive autodiff,
            #root cause still open), and an autodiff logdensity option that existed here
            #was removed after it made every gradient-based trajectory divergent. The
            #value used for acceptance stays exact; only the gradient is FD, which
            #leaves the MH-corrected chain exact regardless of truncation error. One
            #BATCHED eval_logpdf_batch call of 2*n_dim + 1 rows serves value + gradient
            #together - theta is only 4-5 dimensional, so this costs a handful of
            #logpdf evals and sidesteps reverse-mode entirely
            def value_and_fd_grad(u):
                offsets = theta_fd_step * jnp.eye(n_dim)
                us = jnp.concatenate([u[None, :], u[None, :] + offsets,
                                      u[None, :] - offsets], axis = 0)
                lps = eval_logpdf_batch(embed(us))
                lp0 = jnp.where(jnp.isfinite(lps[0]), lps[0], -jnp.inf)
                grad = (lps[1:n_dim + 1] - lps[n_dim + 1:]) / (2 * theta_fd_step)
                #out-of-box stencil points give non-finite differences; zero those
                #gradient components so the trajectory state itself stays finite (the
                #-inf value already flags the divergence)
                grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
                return lp0, grad

            @jax.custom_vjp
            def logdensity(u):
                return value_and_fd_grad(u)[0]
            def logdensity_fwd(u):
                lp, grad = value_and_fd_grad(u)
                return lp, grad
            def logdensity_bwd(grad, cotangent):
                return (cotangent * grad,)
            logdensity.defvjp(logdensity_fwd, logdensity_bwd)
            return logdensity

        #the whitening transform above IS the educated-guess tuning, so the momentum
        #metric is the identity and the step size starts at its flag value (then tunes
        #online via Robbins-Monro in the loop)
        ghmc_inverse_mass_matrix = jnp.ones(len(ghmc_sampled_idx))

        #generalized HMC: persistent momentum and slice variables carried across
        #sweeps (blackjax stores them in GHMCState). The momentum lives in the
        #FIXED whitened metric defined by ghmc_inverse_mass_matrix, so it stays
        #correctly distributed across sweeps even as the conditional's mixing
        #matrices move; the state's logdensity/gradient are recomputed inside the
        #kernel each sweep for the same reason
        from blackjax.mcmc.ghmc import GHMCState

        #the kernel step is jitted ONCE with the per-sweep arrays as arguments - fresh
        #closures over each sweep's mixing matrices would retrace (and recompile the
        #lensing gradient graph) every iteration
        @jax.jit
        def ghmc_kernel_step(step_key, position, momentum, slice_var, step_size,
                             mixed_temp_matrix, mixed_phi_matrix, frozen_params,
                             qe_matrix):
            logdensity = ghmc_logdensity_from(mixed_temp_matrix, mixed_phi_matrix,
                                              frozen_params, qe_matrix)
            alg = blackjax.ghmc(logdensity, step_size = step_size,
                                momentum_inverse_scale = ghmc_inverse_mass_matrix,
                                alpha = ghmc_theta_alpha, delta = ghmc_theta_delta)
            ld, ld_grad = jax.value_and_grad(logdensity)(position)
            state = GHMCState(position = position, momentum = momentum,
                              logdensity = ld, logdensity_grad = ld_grad,
                              slice = slice_var)
            state, info = alg.step(step_key, state)
            return (state.position, state.momentum, state.slice,
                    info.acceptance_rate, info.is_divergent)

        #same initialization blackjax's ghmc.init uses
        rng_key, sub_key = jax.random.split(sub_key)
        k_momentum, k_slice = jax.random.split(rng_key)
        ghmc_theta_momentum = jax.random.normal(k_momentum,
                                                (len(ghmc_sampled_idx),))
        ghmc_theta_slice = jax.random.uniform(k_slice, minval = -1.0, maxval = 1.0)
        ghmc_theta_step = float(ghmc_theta_step_size)

    #run the chain for the maximum specified number if iterations
    #start_time = time.time()
    ombh2_accept_history = []
    omch2_accept_history = []
    ns_accept_history = []
    theta_MC_100_accept_history = []
    logA_accept_history = []

    phi_accept_history = []
    delta_h_history = []
    # #2. mix the fields
    #which_theta_idx = 0
    #mixed_temp, mixed_phi = mix(f_ground, phi_ground, args["mixing_d"], args["mixing_g"])
    for iter in range(1, iters_per_chain + 1):

        #1. sample the temperature field. freeze_g_mixing REQUIRES this draw: the
        #theta-dependent D mixing of f is exact only because f is resampled from its
        #conditional each sweep - mixing a pinned f with D(theta_current) would
        #reintroduce the fixed-point pathology through the D channel
        # rng_key, sub_key = jax.random.split(sub_key)
        # temp_field = gibbs_sample_f(field_zeroes, data_field, phi, args, rng_key)

        #2. mix the fields (under freeze_g_mixing phi is pinned at phi_ground and the
        #identity G makes mixed_phi = phi_ground itself)

        # start = time.perf_counter()
        mixed_temp, mixed_phi = mix(f_ground, phi_ground, args["mixing_d"], args["mixing_g"])
        # end = time.perf_counter()
        # print(f"Single Mixing Step = {end - start} Seconds")

        #3. sample the lensing potential phi. Skipped under freeze_g_mixing: phi stays
        #pinned at phi_ground, which is exactly why G must be frozen at the identity -
        #re-mixing a pinned phi with G(theta_current) would make the chain a
        #fixed-point iteration through the phi channel instead of MCMC on
        #p(f, theta | phi = phi_ground, d)
        # phi_sampler = "hmc"
        # if not freeze_g_mixing:
        #     rng_key, sub_key = jax.random.split(sub_key)
        #     if phi_sampler == "ghmc" and iter > ghmc_warmup_hmc_iters:
        #         #accept is the acceptance FRACTION across the ghmc_num_steps inner updates
        #         mixed_phi, ghmc_u, delta_h, accept = gibbs_sample_phi_ghmc(mixed_phi, ghmc_u,
        #                                                                    mixed_temp, data_field,
        #                                                                    rng_key, args, iter,
        #                                                                    num_burn_in_always_accept,
        #                                                                    ghmc_alpha, ghmc_step_size,
        #                                                                    ghmc_num_steps)
        #     else:
        #         mixed_phi, delta_h, accept = gibbs_sample_phi(mixed_phi, mixed_temp, data_field, rng_key,
        #                                                       args, iter, num_burn_in_always_accept,
        #                                                       mass_matrix = None)
        #     phi_accept_history.append(accept)
        #     delta_h_history.append(delta_h)
        #     print(f"delta_H = {delta_h}")
        #     print(f"phi accept rate = {np.sum(np.array(phi_accept_history)) / len(phi_accept_history)}")
        #     num_positive = 0
        #     for delta_h_value in delta_h_history:
        #         if delta_h_value > 0:
        #             num_positive += 1
        #     print(f"percent delta_h positive = {num_positive / len(delta_h_history)}")

        #4. sample your cosmo parameters
        if iter >= num_burn_in_fix_theta:
            if use_ghmc_theta:
                #GHMC path: ONE leapfrog step per sweep with persistent momentum - the
                #trajectory is effectively continued across sweeps, so no multi-step
                #integration is needed per theta sample
                rng_key, sub_key = jax.random.split(sub_key)
                u = ghmc_T_inv @ current_params[jnp.array(ghmc_sampled_idx)]
                (u, ghmc_theta_momentum, ghmc_theta_slice, accept_rate,
                 divergent) = ghmc_kernel_step(
                    rng_key, u, ghmc_theta_momentum, ghmc_theta_slice,
                    jnp.float64(ghmc_theta_step),
                    field_matrix_stack(mixed_temp, pol), mixed_phi.scalar_matrix,
                    current_params, args["quadratic_estimate"].scalar_matrix)
                new_x = ghmc_T @ u
                for k, i in enumerate(ghmc_sampled_idx):
                    current_params = current_params.at[i].set(new_x[k])
                ghmc_theta_accept_history.append(float(accept_rate))
                #online Robbins-Monro step-size tuning toward the target acceptance for
                #the first ghmc_theta_adapt_sweeps sweeps (discard those as burn-in),
                #frozen afterwards. GHMC needs HIGH acceptance: every rejection flips
                #the persistent momentum and degrades it back toward a random walk
                if iter <= ghmc_theta_adapt_sweeps:
                    ghmc_theta_step *= float(np.exp(
                        0.1 * (float(accept_rate) - ghmc_theta_target_accept)))
                if bool(divergent):
                    print("GHMC theta: divergent step")
                print(f"GHMC theta mean acceptance = "
                      f"{np.mean(np.array(ghmc_theta_accept_history)):.3f}, "
                      f"step size = {ghmc_theta_step:.3e}")
                for name in PARAM_ORDER:
                    param_vals[name].append(float(current_params[PARAM_INDEX[name]]))
            elif use_joint_mh:
                if iter <= 200:
                    joint_sigmas_vec_prime = 4*joint_sigmas_vec
                else:
                    joint_sigmas_vec_prime = 0.5*joint_sigmas_vec
                #joint-MH path: one Metropolis update kicking all five parameters at once
                #with independent per-parameter Gaussian sigmas
                rng_key, sub_key = jax.random.split(sub_key)
                current_params = jnp.array(gibbs_sample_theta(
                    jnp.array(0), jnp.array(param_vals[PARAM_ORDER[0]][-1]),
                    param_ranges[PARAM_ORDER[0]], field_matrix_stack(mixed_temp, pol),
                    mixed_phi.scalar_matrix, data_stack,
                    current_params,
                    emu_params, model_tt,
                    model_pp, rng_key, args["lmax"], args["lmax_prime"],
                    args["nside"], args["pix_width"],
                    data_field.theta_pix,
                    args["ell_grid"], args["ells"], PARAM_ORDER[0], ombh2_accept_history, omch2_accept_history,
                    ns_accept_history, theta_MC_100_accept_history, logA_accept_history,
                    args["cphi_fid"], args["cf_fid"], args["quadratic_estimate"].scalar_matrix,
                    op_matrix_stack(args["noise_covariance"], pol),
                    op_shared_matrix(args["mask"], pol), op_shared_matrix(args["beam"], pol),
                    data_field.fourier_weights,
                    tt_x_mean, tt_x_std, tt_t_mean,
                    tt_t_std, tt_pca_basis_T, tt_pca_mean,
                    pp_x_mean, pp_x_std, pp_t_mean,
                    pp_t_std, pp_pca_basis_T, pp_pca_mean,
                    sampler = "metropolis",
                    metropolis_num_steps = metropolis_num_steps,
                    joint_sigmas = joint_sigmas_vec_prime,
                    joint_accept_history = joint_accept_history,
                    bounds_lo = theta_bounds_lo, bounds_hi = theta_bounds_hi,
                    pol = pol, bb_is_zero = bb_is_zero,
                    fixed_field = fixed_field_theta,
                    freeze_g = freeze_g_mixing,
                    cn_mix_scalar = cn_mix_stack))
                for name in PARAM_ORDER:
                    param_vals[name].append(float(current_params[PARAM_INDEX[name]]))
            elif use_pca:
                #PCA path: one Metropolis update along each scaled eigen-direction, in a
                #freshly shuffled order each loop. Every update moves all five parameters
                #at once, so the whole vector is appended once per iteration
                direction_order = list(range(len(PARAM_ORDER)))
                random.shuffle(direction_order)
                for i in direction_order:
                    rng_key, sub_key = jax.random.split(sub_key)
                    current_params = jnp.array(gibbs_sample_theta(
                        jnp.array(0), jnp.array(param_vals[PARAM_ORDER[0]][-1]),
                        param_ranges[PARAM_ORDER[0]], field_matrix_stack(mixed_temp, pol),
                        mixed_phi.scalar_matrix, data_stack,
                        current_params,
                        emu_params, model_tt,
                        model_pp, rng_key, args["lmax"], args["lmax_prime"],
                        args["nside"], args["pix_width"],
                        data_field.theta_pix,
                        args["ell_grid"], args["ells"], PARAM_ORDER[0], ombh2_accept_history, omch2_accept_history,
                        ns_accept_history, theta_MC_100_accept_history, logA_accept_history,
                        args["cphi_fid"], args["cf_fid"], args["quadratic_estimate"].scalar_matrix,
                        op_matrix_stack(args["noise_covariance"], pol),
                        op_shared_matrix(args["mask"], pol), op_shared_matrix(args["beam"], pol),
                        data_field.fourier_weights,
                        tt_x_mean, tt_x_std, tt_t_mean,
                        tt_t_std, tt_pca_basis_T, tt_pca_mean,
                        pp_x_mean, pp_x_std, pp_t_mean,
                        pp_t_std, pp_pca_basis_T, pp_pca_mean,
                        sampler = "metropolis",
                        proposal_sigma = pca_proposal_sigma,
                        metropolis_num_steps = metropolis_num_steps,
                        direction = pca_directions[i], direction_idx = i,
                        pca_accept_history = pca_accept_histories[i],
                        bounds_lo = theta_bounds_lo, bounds_hi = theta_bounds_hi,
                        pol = pol, bb_is_zero = bb_is_zero,
                        fixed_field = fixed_field_theta,
                        freeze_g = freeze_g_mixing,
                        cn_mix_scalar = cn_mix_stack))
                for name in PARAM_ORDER:
                    param_vals[name].append(float(current_params[PARAM_INDEX[name]]))
            else:
                #randomly shuffle the order in which we sample the cosmo parameters each loop
                shuffled_items = list(param_ranges.items())
                #random.shuffle(shuffled_items)
                for theta, theta_range in shuffled_items:
                    if should_sample[theta]:
                        rng_key, sub_key = jax.random.split(sub_key)
                        theta_key_idx = PARAM_INDEX[theta]
                        theta_sampler = "metropolis"
                        if theta == "ombh2":
                            theta_proposal_sigma = 1.25e-4 
                        elif theta == "omch2":
                            theta_proposal_sigma = 1e-3
                        elif theta == "logA":
                            theta_proposal_sigma = 2.5e-2
                        elif theta == "theta_MC_100":
                            theta_proposal_sigma = 3.5e-3
                        elif theta == "ns":
                            theta_proposal_sigma = 0.5e-2 
                        theta_val = gibbs_sample_theta(jnp.array(theta_key_idx), jnp.array(param_vals[theta][-1]),
                                                    theta_range, field_matrix_stack(mixed_temp, pol),
                                                    mixed_phi.scalar_matrix, data_stack,
                                                    current_params,
                                                    emu_params, model_tt,
                                                    model_pp, rng_key, args["lmax"], args["lmax_prime"],
                                                    args["nside"], args["pix_width"],
                                                    data_field.theta_pix,
                                                    args["ell_grid"], args["ells"], theta, ombh2_accept_history, omch2_accept_history,
                                                    ns_accept_history, theta_MC_100_accept_history, logA_accept_history,
                                                    args["cphi_fid"], args["cf_fid"], args["quadratic_estimate"].scalar_matrix,
                                                    op_matrix_stack(args["noise_covariance"], pol),
                                                    op_shared_matrix(args["mask"], pol), op_shared_matrix(args["beam"], pol),
                                                    data_field.fourier_weights,
                                                    tt_x_mean, tt_x_std, tt_t_mean,
                                                    tt_t_std, tt_pca_basis_T, tt_pca_mean,
                                                    pp_x_mean, pp_x_std, pp_t_mean,
                                                    pp_t_std, pp_pca_basis_T, pp_pca_mean,
                                                    over_relaxation_num_samps = over_relaxation_num_samps,
                                                    sampler = theta_sampler,
                                                    proposal_sigma = theta_proposal_sigma,
                                                    metropolis_num_steps = metropolis_num_steps,
                                                    pol = pol, bb_is_zero = bb_is_zero,
                                                    fixed_field = fixed_field_theta,
                                                    freeze_g = freeze_g_mixing,
                                                    cn_mix_scalar = cn_mix_stack)
                        param_vals[theta].append(theta_val)
                        current_params = current_params.at[theta_key_idx].set(theta_val)
                #DEBUG... Try a Gauss-Seidel update instead of Jacobian...
                # current_params = current_params.at[PARAM_INDEX["ombh2"]].set(param_vals["ombh2"][-1])
                # current_params = current_params.at[PARAM_INDEX["omch2"]].set(param_vals["omch2"][-1])
                # current_params = current_params.at[PARAM_INDEX["ns"]].set(param_vals["ns"][-1])
                # current_params = current_params.at[PARAM_INDEX["logA"]].set(param_vals["logA"][-1])
                # current_params = current_params.at[PARAM_INDEX["theta_MC_100"]].set(param_vals["theta_MC_100"][-1])

            #5. recompute mixing and covariance matrices using the newly sampled parameter
            #values (with refresh_qe the quadratic-estimate norm - and through it the G
            #matrix and the phi mass matrix - is rebuilt from the interpolated lensed Cls
            #at the new theta as well)
            args = update_args_after_sample(current_params, predictors,
                                            emu_params, args, pol = pol,
                                            refresh_qe = refresh_qe,
                                            fixed_field = fixed_field_theta,
                                            freeze_g = freeze_g_mixing)
            
        # -------------------------------------------------------- DEBUG --------------------------------------------------------
        #Store the sampled a_phi value to a debug text file...
        # ombh2_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/grape_shot_joint_inference_08_20_26/ombh2_map_{map}_chain_{seed}_history.txt"
        # ns_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/grape_shot_joint_inference_08_20_26/ns_map_{map}_chain_{seed}_history.txt"
        # omch2_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/grape_shot_joint_inference_08_20_26/omch2_map_{map}_chain_{seed}_history.txt"
        # logA_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/grape_shot_joint_inference_08_20_26/logA_map_{map}_chain_{seed}_history.txt"
        # theta_MC_100_file_path = f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/sampling_chains/grape_shot_joint_inference_08_20_26/theta_MC_100_map_{map}_chain_{seed}_history.txt"
        # with open(ombh2_file_path, "a") as file:
        #    file.write(str(param_vals["ombh2"][-1]) + "\n")
        # with open(ns_file_path, "a") as file:
        #     file.write(str(param_vals["ns"][-1]) + "\n")
        # with open(omch2_file_path, "a") as file:
        #    file.write(str(param_vals["omch2"][-1]) + "\n")
        # with open(logA_file_path, "a") as file:
        #     file.write(str(param_vals["logA"][-1]) + "\n")
        # with open(theta_MC_100_file_path, "a") as file:
        #    file.write(str(param_vals["theta_MC_100"][-1]) + "\n")
        # -------------------------------------------------------- DEBUG --------------------------------------------------------

        plt.figure(figsize = (16, 10))
        for name in PARAM_ORDER:
            if should_sample.get(name, False):
                normalized = (np.array(param_vals[name]) - GROUND_TRUTH[name]) / TRAINING_SIGMA[name]
                plt.plot(normalized, label = name, marker = "o")
        plt.axhline(0, color = "black", label = "Zero Sigma")
        plt.axhline(1, color = "grey", label = "+/- 1 Sigma")
        plt.axhline(-1, color = "grey")
        #plt.ylim([-5, +5])
        plt.title("(mean - sample)/sigma")
        plt.xlabel("iteration")
        plt.ylabel("standard deviations")
        plt.legend()
        plt.savefig("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/lcdm_progress.png")
        plt.close()

        #6. unmix the fields using the updated version of the G & D matrices
        #skipped under freeze_g_mixing: phi was never sampled, so with identity G the
        #pinned phi_ground is already the unmixed phi and the inverse lensing is wasted
        # if not freeze_g_mixing:
        # _, phi = unmix(mixed_temp, mixed_phi, args["mixing_d"], args["mixing_g"])

        mixed_logpdf_value = mixed_logpdf(mixed_temp, mixed_phi, data_field, args["noise_covariance"], 
                                          args["phi_covariance"], args["field_covariance"], 
                                          args["mask"], args["beam"], 
                                          args["mixing_g"], args["mixing_d"])
        mixed_logpdf_values.append(mixed_logpdf_value)
        print(f"LogPDF value = {mixed_logpdf_value}")
        plt.figure()
        plt.plot(mixed_logpdf_values, marker = "o")
        plt.title("LogPDF Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("LogPDF Value")
        plt.legend()
        plt.savefig("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/logpdf_progress.png")
        plt.close()        

    # #save the empirical PCA of this chain so the next run can propose along the
    # #eigen-directions (use_pca = True)
    # if save_pca:
    #     compute_and_save_pca(param_vals, pca_path, burn_in = pca_burn_in)

    return param_vals

if __name__ == "__main__":

    #ground-truth (fiducial) parameter values in load_sim naming. single source of truth:
    #GROUND_TRUTH in precompute_camb_1d.py - the values baked into the CAMB spline caches
    ground_truth_params = {}
    ground_truth_params["ombh2"] = GROUND_TRUTH["ombh2"]
    ground_truth_params["omch2"] = GROUND_TRUTH["omch2"]
    ground_truth_params["cosmomc_theta"] = GROUND_TRUTH["theta_MC_100"] / 100
    ground_truth_params["As"] = jnp.exp(GROUND_TRUTH["logA"]) * 1e-10
    ground_truth_params["ns"] = GROUND_TRUTH["ns"]

    #Generate a "ground truth" simulated data set. pol = "IP" runs the joint T + E + B
    #analysis: EE's acoustic structure plus the physical TE cross-correlation break the
    #ns-ombh2 degeneracy that the T-only map leaves (marginal corr ~ -0.76, previously
    #patched over with the BBN-style THETA_PRIORS ombh2 prior, now off by default). T
    #and E are drawn CORRELATED to the CAMB TE spectrum, carried through the analysis
    #as the correlation ratio te_rho (grid-splined; see te_covar_from_rho), and the
    #unlensed B has zero power at r = 0 - see get_d_teb_matrix / the 5D grid docs.
    #Requires a grid file carrying the EE/BB, lensed and te_rho spectra (regenerate
    #with camb_grid.sh + merge_camb_grid.py - te_rho only needs a RE-MERGE of existing
    #slabs); set pol = "I" to fall back to the temperature-only analysis, or pol = "P"
    #for the polarization-only (E + B, no T / TE) analysis - all three share the same
    #sampler code paths (sample_joint infers the mode from the dataset type)
    nside = 128 #256
    theta_pix = 2.5
    pol = "I"
    master_seed = 1283746 * 4 #1871263
    noise_level = 5
    data_set = load_sim(nside, theta_pix, pol, master_seed, **ground_truth_params,
                        uk_arcmin_t = noise_level, r = 0, nt = 0,
                        use_emulator_cls = not (USE_CAMB_SPLINE or USE_CAMB_GRID))
    f_ground = data_set.unlensed_field
    phi_ground = data_set.phi

    #switch back to cambemul naming conventions...
    #NOTE with USE_CAMB_GRID the 5D spline interpolates all five parameters at once, so
    #every one of these may be given a real starting guess and sampled jointly. All five
    #start at ground truth so the chain needs minimal burn-in before its samples are
    #usable for the PCA covariance estimate
    param_init = {}
    param_init["ombh2"] = GROUND_TRUTH["ombh2"] #PARAM_BOUNDS["ombh2"][0] #
    param_init["omch2"] =  PARAM_BOUNDS["omch2"][-1] #GROUND_TRUTH["omch2"] #
    param_init["theta_MC_100"] = PARAM_BOUNDS["theta_MC_100"][-1] #GROUND_TRUTH["theta_MC_100"] #
    param_init["logA"] = PARAM_BOUNDS["logA"][0] #GROUND_TRUTH["logA"] #
    param_init["ns"] = GROUND_TRUTH["ns"] #PARAM_BOUNDS["ns"][-1] #
    # param_init["ombh2"] = GROUND_TRUTH["ombh2"] #
    # param_init["omch2"] =  GROUND_TRUTH["omch2"] #
    # param_init["theta_MC_100"] = GROUND_TRUTH["theta_MC_100"] #
    # param_init["logA"] = GROUND_TRUTH["logA"] #
    # param_init["ns"] = GROUND_TRUTH["ns"] #

    #allowed search / sample range for parameters... The min and max values are +/- 5 std
    #from the training mean for the CAMB emulator
    SEARCH_PRECISION = 50
    #endpoints come from PARAM_BOUNDS in precompute_camb_1d.py so the sampler's search
    #range always matches the support of the cached CAMB spline grids
    param_ranges = {}
    for name, (lo, hi) in PARAM_BOUNDS.items():
        param_ranges[name] = jnp.linspace(lo, hi, SEARCH_PRECISION)

    #Whether or not to sample each parameter - all five sampled jointly off the 5D grid.
    #use_pca/save_pca require all five True; use_joint_mh accepts any subset (False
    #entries are frozen at param_init)
    should_sample = {}
    should_sample["ombh2"] = False
    should_sample["omch2"] = True
    should_sample["theta_MC_100"] = True
    should_sample["logA"] = True
    should_sample["ns"] = False

    #the 1D CAMB spline caches hold every non-sampled parameter at GROUND_TRUTH; force
    #param_init to match so a stale starting guess cannot trip the cache's fixed-value
    #assertion (only the sampled parameter keeps its custom starting point). The 5D grid
    #has no such constraint - every parameter is free to start anywhere inside the box -
    #so this clamp is skipped when USE_CAMB_GRID is on
    if USE_CAMB_SPLINE and not USE_CAMB_GRID:
        for name in PARAM_ORDER:
            if not should_sample[name]:
                param_init[name] = GROUND_TRUTH[name]

    #proposal sigmas for the plain joint-MH theta update (use_joint_mh = True): the
    #CONDITIONAL widths 1/sqrt(diag(C^-1)) of the pilot covariance in theta_pca.npz,
    #scaled by the 2.38/sqrt(d) optimal-scaling rule. A diagonal joint proposal has to
    #stay inside the conditional widths to be accepted under strong correlations, so
    #these are the right starting point - tune toward ~23% acceptance from here
    joint_proposal_sigmas = {}
    joint_proposal_sigmas["theta_MC_100"] = 5e-3 #0.125*1e-2 #6.2e-4
    joint_proposal_sigmas["logA"] = 1e-2 #0.125*5e-2 #6.7e-3
    joint_proposal_sigmas["ns"] = 0.125*1e-2 #4.8e-4 #4.8e-3
    joint_proposal_sigmas["ombh2"] = 0.125*5e-4 #1.9e-5 #1.9e-4
    joint_proposal_sigmas["omch2"] = 5e-4 #0.125*5e-3 #6.3e-4

    #run the sampling algorithm.
    #Theta-update options (mutually exclusive full-vector paths; with all False the
    #axis-aligned per-parameter Metropolis runs):
    # - use_ghmc_theta = True: generalized HMC - ONE gradient-based leapfrog step per
    #   sweep with persistent momentum, via the differentiable 5D grid spline. Inverse
    #   mass matrix guessed from the pilot covariance (theta_pca.npz, identity if
    #   absent); step size tuned online over the first ghmc_theta_adapt_sweeps sweeps.
    #   No warmup phase at all - this is the production gradient-based configuration.
    #   (blackjax NUTS was removed: naive autodiff of mixed_logpdf is broken w.r.t. the
    #   G mixing matrix and NUTS never functioned for this problem)
    # - use_joint_mh = True: plain joint MH - all five parameters kicked at once with the
    #   per-parameter sigmas above, one accept/reject on the joint logpdf change
    # - use_pca = True: Metropolis along the sigma-scaled eigen-directions of the pilot
    #   covariance written by a previous save_pca = True run (pca_proposal_sigma is in
    #   conditional-sigma units). Keeping save_pca = True refreshes THETA_PCA_PATH from
    #   the finished chain
    param_distributions = sample_joint(data_set, param_init, param_ranges, should_sample, noise_level,
                                       f_ground, phi_ground, phi_sampler = "ghmc",
                                       iters_per_chain = 10_000, num_burn_in_fix_theta = 0,
                                       over_relaxation_num_samps = -1, seed = 67,
                                       num_burn_in_always_accept = 0,
                                       use_pca = False, save_pca = False,
                                       pca_burn_in = 0, pca_proposal_sigma = 1,
                                       use_joint_mh = False,
                                       joint_proposal_sigmas = joint_proposal_sigmas,
                                       use_ghmc_theta = False,
                                       #fixed_field_theta: only for chains with BOTH
                                       #f and phi pinned at ground truth (identity
                                       #mixing, theta targets the UNMIXED conditional).
                                       #Must stay False when the fields are sampled -
                                       #full Gibbs needs the real D/G
                                       fixed_field_theta = False,
                                       #freeze_g_mixing: pin ONLY phi at phi_ground
                                       #(the phi HMC and unmix steps are skipped) while
                                       #f is still resampled every sweep. G is frozen
                                       #at the identity so the pinned phi is never
                                       #re-mixed at the current theta (that would be
                                       #the fixed-point pathology through the phi
                                       #channel); D keeps tracking theta, which is
                                       #exact because f is freshly drawn. The chain is
                                       #then exact MCMC on p(f, theta | phi_ground, d)
                                       #at a fraction of full-Gibbs cost
                                       freeze_g_mixing = False,
                                       #mixing_noise_uk_arcmin: build D/G/QE/mass matrix
                                       #at this effective noise level while the
                                       #likelihood kee128 x 128, theta_pix = 2.5, noise_level = 5, no 1/f or beam or mask, mixing turned on, Temperature only, CAMB interpolator, (f, phi) @ GROUNDps noise_level. Exact for any
                                       #value (pure reparametrization). Set to 5 for
                                       #low-noise runs to test whether the mixed
                                       #parametrization's stalling is its noise
                                       #dependence; None = no decoupling
                                       mixing_noise_uk_arcmin = None)

