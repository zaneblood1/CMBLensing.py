"""Posterior draws of the mixed pair (f°, phi°) given the data, with theta HELD FIXED.

sample_lcdm.sample_joint's Gibbs sweep minus its theta step: f from its exact conditional
(gibbs_sample_f), a mix, one HMC step on phi° (gibbs_sample_phi) and an unmix, with C_f /
C_phi / D / G built once at a fixed cosmology by the same constructors the sampler uses
(mixed_theta_matrices). sampler_phi_noise.py runs one such chain per realization at the
ground truth and cross correlates the posterior-mean phi with the true phi to measure the
sampler's effective reconstruction noise.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from cmb_lensing.simulate import get_d_tt_matrix, get_g_matrix_lcdm
from cmb_lensing.mixing import mix, unmix
from cmb_lensing.fisher_forecast import prior_covariances
from cmb_lensing.constants import DEFAULT_HMC_NUM_STEPS, DEFAULT_HMC_STEP_SIZE

#fold_in data separating a chain's MCMC randomness from its data realization's. load_sim
#starts from PRNGKey(map_seed) and splits off one subkey per draw (rng_key, subkey =
#split(rng_key)), and under the partitionable threefry JAX uses split(k)[i] == fold_in(k, i),
#so every key load_sim touches descends from fold_in(PRNGKey(map_seed), 0 or 1). A chain rooted
#at PRNGKey(map_seed) itself would reuse those keys on its first sweep; rooted at
#fold_in(PRNGKey(map_seed), CHAIN_STREAM) with CHAIN_STREAM not 0 or 1 it never can
CHAIN_STREAM = 2_026_092_301

def chain_key(map_seed):
    """The MCMC root key of one chain on the realization load_sim draws at `map_seed`:
    distinct per `chain_index` and disjoint from every key load_sim uses."""
    stream = jax.random.fold_in(jax.random.PRNGKey(map_seed), CHAIN_STREAM)
    return stream

def mixed_theta_matrices(params, data_set, nside, pix_width, ell_grid, qe_frozen):
    """(C_f, C_phi, D, G) operators at one cosmology, built the way sample_lcdm builds them.

    D = get_d_tt_matrix(C_f, C_n) and G = get_g_matrix_lcdm(C_phi, N_phi) - the same two
    constructors _recompute_cosmo_matrices uses - so the mixing here matches the sampler's.
    `qe_frozen` is the quadratic-estimate norm matrix G is built from (load_sim's own
    `quadratic_estimate.scalar_matrix`, for sampler_phi_noise), as the sampler holds its
    QE norm fixed for a whole chain.
    T-only.
    """
    cf, cphi = prior_covariances(params, nside, pix_width, ell_grid)
    cf_op = data_set.field_covariance.replace(scalar_matrix = cf)
    cphi_op = data_set.phi_covariance.replace(scalar_matrix = cphi)
    d_op = data_set.mixing_d.replace(scalar_matrix = get_d_tt_matrix(cf, data_set.noise_covariance.scalar_matrix))
    g_op = data_set.mixing_g.replace(scalar_matrix = get_g_matrix_lcdm(cphi, qe_frozen))
    return cf_op, cphi_op, d_op, g_op

def posterior_mixed_draws(data_set, cf_op, cphi_op, d_fid, g_fid, n_draws, rng_key,
                          on_draw = None, verbose = True, hmc_step_size = DEFAULT_HMC_STEP_SIZE, 
                          hmc_num_steps = DEFAULT_HMC_NUM_STEPS):
    """Draws of the mixed pair (f°, phi°) from p(f°, phi° | d, theta), theta held fixed.

    The same Gibbs sweep sample_lcdm.sample_joint runs, minus its theta step: f from its
    exact conditional (gibbs_sample_f), a mix, one HMC step on phi° (gibbs_sample_phi), and
    an unmix to hand the next f step the new phi. D and G never change, so the (f°, phi°)
    pair at the end of each sweep IS a posterior draw in fixed mixed coordinates. The chain
    starts at phi = 0 and hands EVERY sweep to the caller, the first one included: there is
    deliberately no burn-in and no thinning here, since both are properties of the
    realization and are chosen in post-processing. `rng_key` is the chain's own key
    (chain_key).

    `on_draw(sweep_index, mixed_field, mixed_phi, accepted)` CONSUMES each sweep as it is
    produced (`accepted` is that sweep's phi HMC decision) and is the way callers should use
    this: a mixed pair is two (nside, nside//2+1) complex128 arrays - 260 kB per draw at
    nside 128 - so with a callback nothing is accumulated and the returned draw list is
    empty; without one every draw is kept, which only suits short diagnostic chains.

    Returns (draws, diagnostics) with diagnostics carrying the per-sweep phi acceptances,
    their mean and the sweep count.
    """
    #imported here: sample_lcdm pulls in the whole sampler stack (and the 5D grid machinery)
    from cmb_lensing.sample_lcdm import gibbs_sample_f, gibbs_sample_phi

    #the posterior is taken under EXACTLY these operators
    data_set = data_set.replace(field_covariance = cf_op, phi_covariance = cphi_op,
                                mixing_d = d_fid, mixing_g = g_fid)
    args = dict(noise_covariance = data_set.noise_covariance, mask = data_set.mask,
                beam = data_set.beam, field_covariance = cf_op, phi_covariance = cphi_op,
                mixing_d = d_fid, mixing_g = g_fid,
                quadratic_estimate = data_set.quadratic_estimate)

    phi = 0 * data_set.phi
    field_zeroes = 0 * data_set.unlensed_field
    key = rng_key
    draws, accepts = [], []

    for sweep in range(1, n_draws + 1):
        key, key_f, key_phi = jax.random.split(key, 3)
        field = gibbs_sample_f(field_zeroes, data_set.data, phi, args, key_f)
        mixed_field, mixed_phi = mix(field, phi, d_fid, g_fid)
        mixed_phi, _, accept = gibbs_sample_phi(mixed_phi, mixed_field, data_set.data,
                                                key_phi, args, sweep, 0, 
                                                hmc_num_steps = hmc_num_steps,
                                                hmc_step_size = hmc_step_size)
        accepts.append(bool(accept))
        _, phi = unmix(mixed_field, mixed_phi, d_fid, g_fid)
        if on_draw is None:
            draws.append((mixed_field, mixed_phi))
        else:
            #consumed here and dropped: only the callback's summary survives the sweep
            on_draw(sweep, mixed_field, mixed_phi, accepts[-1])
        if verbose and sweep % 50 == 0:
            print(f"    sweep {sweep}/{n_draws}: phi acceptance {np.mean(accepts):.3f}")

    diagnostics = dict(phi_accepts = np.array(accepts),
                       phi_acceptance = float(np.mean(accepts)), n_sweeps = n_draws)
    return draws, diagnostics
