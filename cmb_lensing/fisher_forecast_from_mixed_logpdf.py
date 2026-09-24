"""The realization-averaged Hessian of statistics.mixed_logpdf, "mixed":

    F_ij = < -d2 log p(d, f°, phi° | theta) / dtheta_i dtheta_j >  at a fixed mixed pair

The mixed-coordinate counterpart of fisher_forecast_from_logpdf. Each realization draws
(f, phi) at the fiducial cosmology with load_sim, mixes them ONCE with D and G at that
cosmology (f° = L(phi) D f, phi° = G phi - the sampler's coordinates), then holds that
mixed pair fixed while theta moves through C_f, C_phi, D, G and the -logdet(G) - logdet(D)
Jacobian inside mixed_logpdf.

READ THIS BEFORE COMPARING IT TO THE OTHER METHODS. mixed_logpdf is a properly normalized
density in (f°, phi°) - it carries the Jacobian for exactly that reason - so this is a
legitimate Fisher for the model written in mixed coordinates, and Louis's identity in
these coordinates would recover the same MARGINAL Fisher as in the unmixed ones. But the
split between Louis's two terms is NOT parametrization invariant: only their difference
is. So this Hessian is NOT the same quantity as the unmixed logpdf method's, and it does
not converge to the bare-prior complete-data Fisher that one can be checked against. Treat
the two as different decompositions, not as estimates of one number.

COST. Far more expensive per evaluation than the unmixed path: unmix runs an INVERSE
lensing solve and logpdf then runs a forward one, so every stencil point costs two lensing
solves, and nothing cancels across the stencil because D, G and the unmixed fields all move
with theta. There is no fast mode here - that is what mixing costs. Sequentially that is
~3800 lensing solves for 100 realizations at 19 stencil points, which is what
sampling_chains/mixed_hessian.sh exists for: it runs one slurm job per realization
(run_single_mixed_hessian.py -> mixed_hessian_realization, one small npz each; MEASURED
1m41s per job at nside 128 / 2.5' / 3 params) and `--hessian_dir` averages the collected
files here. forecast_from_mixed_logpdf is the single entry point: without `hessian_dir` it
is the sequential driver (smoke tests, reproducing a single job locally); with it, it
averages the cached files after checking they were made for the requested configuration.

The QE norm is evaluated once at the fiducial cosmology and frozen across the stencil,
mirroring the sampler (whose refresh_qe is plumbed but not exposed, so its QE norm stays at
the param_init cosmology for a whole chain).

LOUIS MODE (--louis). The average above is the COMPLETE-data information: it counts f° and
phi° as observed. The marginal Fisher of p(d | theta) - the curvature the sampler's
theta posterior actually has - follows from Louis's identity,

    I(d) = E_{f°,phi°|d}[ -d2 l_c ] - Cov_{f°,phi°|d}[ d l_c ],

so louis_realization draws (f°, phi°) from their posterior at theta_0 with the sampler's
own Gibbs sweep (theta held fixed), runs the same stencil at every draw - the score comes
free from its +/-h points - and subtracts the posterior covariance of the score from the
posterior mean of -H. Each realization also evaluates the plain Hessian at its true fields:
by the tower property the complete term and that must agree on average, which checks the
chain. Cost is (1 + draws) stencils plus the Gibbs sweeps per sub-chain - hours rather
than minutes at nside 128 - and the files are method "mixed_louis", written with
SUFFIX "_from_mixed_louis".

The jobs store the RAW chain and every choice about it is made in post-processing
(louis_information_from_chains), the same burn-in -> autocorrelation -> IAT -> prune order
chain_analysis.py applies to the theta chains:
  * NO burn-in is cut in the job. Every sweep from the first is differentiated and saved,
    and `burn_in` is applied at merge time, so a burn-in that turns out too short is fixed
    by re-merging instead of re-running.
  * Each realization runs SEVERAL independent sub-chains (same data map, different MCMC
    randomness; chain_key), one slurm job each. That divides the wall clock and makes the
    Gelman-Rubin R-hat of every score component and Hessian entry measurable.
  * Each sub-chain is thinned by ONE stride: the largest IAT over its score components AND
    its Hessian entries (louis_quantities), since both terms are built from the pruned
    samples and either can decorrelate more slowly than the other.
chain_analysis.py plots the per-quantity ACFs, IATs, R-hats and traces from the same files.

Usage:
    python -m cmb_lensing.fisher_forecast_from_mixed_logpdf --realizations 2 --nside 64
    python -m cmb_lensing.fisher_forecast_from_mixed_logpdf --hessian_dir <mixed_hessian_output>
    python -m cmb_lensing.fisher_forecast_from_mixed_logpdf --louis --realizations 1 --nside 64 \\
        --hessian_dir ""

Writes fisher_matrix_from_mixed_logpdf.png, covariance_matrix_from_mixed_logpdf.png,
correlation_matrix_from_mixed_logpdf.png and fisher_from_mixed_logpdf.npz into
cmb_lensing/fisher_output/.
"""

import argparse
import glob
import os

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid
from cmb_lensing.simulate import load_sim, get_d_tt_matrix, get_g_matrix_lcdm
from cmb_lensing.statistics import mixed_logpdf
from cmb_lensing.mixing import mix, unmix
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH
from cmb_lensing.fisher_forecast import (camb_cls_at_params,
                                         load_sim_cosmology, prior_covariances,
                                         sampled_names_and_steps, stencil_offsets,
                                         hessian_from_stencil, covariance_from_fisher,
                                         report_sigmas, save_outputs, add_box_arguments,
                                         sampled_from_args, box_subtitle, run_config)


METHOD = "mixed"
SUFFIX = "_from_mixed_logpdf"
LABEL = r"mixed:  $F_{ij} = \langle -\partial_i \partial_j \log p(d, f^\circ, \phi^\circ\,|\,\theta) \rangle$"

#the realization methods draw from the prior and never consult a spectra mode; this is
#what their reports and figures are labelled with instead
SPECTRA = "prior draws"

#Louis's observed information in the same coordinates: the mixed Hessian averaged over
#POSTERIOR draws of (f°, phi°) given d, minus the posterior covariance of the score
LOUIS_METHOD = "mixed_louis"
LOUIS_SUFFIX = "_from_mixed_louis"
LOUIS_LABEL = (r"mixed Louis:  $F = \langle -\partial^2 \ell_c \rangle_{f^\circ,\phi^\circ|d}"
               r" - {\rm Cov}_{f^\circ,\phi^\circ|d}[\partial \ell_c]$")
#Gibbs sweeps per sub-chain, ALL of them differentiated and saved - burn-in included. So
#`draws` is a raw chain length, not a count of independent samples: the merge cuts
#DEFAULT_LOUIS_BURN of them (a POST-PROCESSING choice, never applied in the job) and prunes
#the rest by the stride the chain itself earns
DEFAULT_LOUIS_DRAWS = 150
DEFAULT_LOUIS_BURN = 100
#independent sub-chains per data realization (same map, different MCMC randomness), for the
#sequential driver; mixed_hessian.sh sets its own count, one slurm job per sub-chain
DEFAULT_LOUIS_CHAINS = 4
#which stride the post-processing prunes each sub-chain by, mirroring chain_analysis.py's
#USE_ZERO_CROSSING_PRUNE: False -> the integrated autocorrelation time, True -> the lag just
#before the ACF first crosses zero. Either way it is the MAXIMUM over every score component
#and Hessian entry of that sub-chain
LOUIS_USE_ZERO_CROSSING = False
#a realization whose largest R-hat exceeds this is flagged by the merge (not dropped)
LOUIS_R_HAT_WARN = 1.1
#fold_in data separating the MCMC randomness from the data realization's. load_sim draws its
#fields from split(PRNGKey(map_seed), 100) and, under the partitionable threefry JAX uses,
#split(k, n)[i] == split(k, m)[i] and fold_in(k, i) == split(k, n)[i] - so a chain rooted
#at PRNGKey(map_seed) (as it once was) reused the data's own keys on its first sweep, and
#any fold_in below 100 would too. This constant sits far outside that range
LOUIS_CHAIN_STREAM = 2_026_092_301


def _mixed_theta_matrices(params, data_set, nside, pix_width, ell_grid, qe_frozen):
    """(C_f, C_phi, D, G) operators at one cosmology, built the way sample_lcdm builds them.

    D = get_d_tt_matrix(C_f, C_n) and G = get_g_matrix_lcdm(C_phi, N_phi) - the same two
    constructors _recompute_cosmo_matrices uses - so the mixing here matches the sampler's.

    `qe_frozen` is the quadratic-estimate norm, held FIXED across the stencil. That mirrors
    the sampler, whose refresh_qe is plumbed but not exposed, so its QE norm stays at the
    param_init cosmology for a whole chain. Letting N_phi move with theta would credit the
    Fisher with information from dN_phi/dtheta that no chain actually uses.
    """
    cf, cphi = prior_covariances(params, nside, pix_width, ell_grid)
    cf_op = data_set.field_covariance.replace(scalar_matrix = cf)
    cphi_op = data_set.phi_covariance.replace(scalar_matrix = cphi)
    d_op = data_set.mixing_d.replace(
        scalar_matrix = get_d_tt_matrix(cf, data_set.noise_covariance.scalar_matrix))
    g_op = data_set.mixing_g.replace(scalar_matrix = get_g_matrix_lcdm(cphi, qe_frozen))
    return cf_op, cphi_op, d_op, g_op


def _mixed_stencil_values(mixed_field, mixed_phi, data_set, names, param_ground, steps,
                          stencil, nside, pix_width, ell_grid, qe_frozen):
    """mixed_logpdf at every stencil point, with the mixed pair (f°, phi°) held fixed.

    NOTE the argument-order trap: mixed_logpdf takes (..., mixing_g, mixing_d) while
    mix / unmix take (..., mixing_d, mixing_g). They are passed correctly below.
    """
    values = {}
    for offset in stencil:
        params = dict(param_ground)
        for i, name in enumerate(names):
            params[name] = param_ground[name] + offset[i] * steps[i]
        cf_op, cphi_op, d_op, g_op = _mixed_theta_matrices(
            params, data_set, nside, pix_width, ell_grid, qe_frozen)
        values[offset] = float(mixed_logpdf(
            mixed_field, mixed_phi, data_set.data, data_set.noise_covariance,
            cphi_op, cf_op, data_set.mask, data_set.beam, g_op, d_op))
    return values


def score_from_stencil(values, steps):
    """d log p / dtheta_i by central differences, from the +/-1 points stencil_offsets holds."""
    n_param = len(steps)
    score = np.zeros(n_param)
    for i in range(n_param):
        plus = tuple(1 if k == i else 0 for k in range(n_param))
        minus = tuple(-1 if k == i else 0 for k in range(n_param))
        score[i] = (values[plus] - values[minus]) / (2 * steps[i])
    return score


def mixed_logpdf_hessian_one_realization(data_set, names, param_ground, steps, stencil,
                                         nside, pix_width, ell_grid):
    """-d2/dtheta_i dtheta_j mixed_logpdf for ONE realization, at fixed (f_mixed, phi_mixed).

    The fields are drawn at param_ground and mixed ONCE with D and G at param_ground; that
    mixed pair is then held fixed while theta moves through C_f, C_phi, D, G and the
    -logdet(G) - logdet(D) Jacobian. See the module docstring for why this is a different
    decomposition from the unmixed logpdf Hessian, not an estimate of the same number.
    """
    #the QE norm is evaluated once at the fiducial cosmology and frozen, matching the sampler
    qe_frozen = data_set.quadratic_estimate.scalar_matrix

    #mix ONCE, at the fiducial cosmology; this pair is the "sample" held fixed
    _, _, d_fid, g_fid = _mixed_theta_matrices(param_ground, data_set, nside, pix_width,
                                               ell_grid, qe_frozen)
    mixed_field, mixed_phi = mix(data_set.unlensed_field, data_set.phi, d_fid, g_fid)

    values = _mixed_stencil_values(mixed_field, mixed_phi, data_set, names, param_ground,
                                   steps, stencil, nside, pix_width, ell_grid, qe_frozen)
    return -hessian_from_stencil(values, steps)


def autocorrelation(series, max_lag = None):
    """Normalized autocorrelation of a 1D sequence at lags 0, 1, ... max_lag.

    Same estimator chain_analysis.py uses on the theta chains, kept local because this
    module cannot import from the gitignored sampling_chains/ directory.
    """
    series = np.asarray(series, dtype = np.float64)
    n = len(series)
    max_lag = n // 2 if max_lag is None else min(max_lag, n)
    centred = series - np.mean(series)
    variance = np.var(series)
    if not np.isfinite(variance) or variance <= 0:
        #a constant series has no autocorrelation to measure; treat it as independent
        return np.zeros(max_lag)
    acf = np.correlate(centred, centred, mode = "full")[n - 1:]
    return acf[:max_lag] / (variance * n)


def integrated_autocorrelation_time(acf, c = 5.0):
    """Sokal's automatic windowing: tau = 1 + 2 sum_k rho_k truncated at the lag M < c tau.

    Takes the ACF, not the series, exactly as chain_analysis.integrated_autocorrelation_time
    does, so the stride derived here means the same thing as the one the theta chains are
    pruned by.
    """
    tau = 1.0
    for lag in range(1, len(acf)):
        tau += 2.0 * acf[lag]
        if lag >= c * tau:
            return tau
    return tau


def first_zero_crossing_lag(acf):
    """Smallest lag just before the ACF first crosses zero; chain_analysis's other stride."""
    acf = np.asarray(acf)
    non_positive = np.nonzero(acf <= 0)[0]
    if len(non_positive) == 0:
        return max(1, len(acf) - 1)
    return max(1, int(non_positive[0]) - 1)


def gelman_rubin(chains):
    """Gelman-Rubin R-hat of a scalar across sub-chains; chain_analysis.gelman_rubin's estimator.

    Chains are trimmed to the shortest. NaN with fewer than two chains, and 1 for a quantity
    that is constant within every chain and equal across them (nothing to converge).
    """
    if len(chains) < 2:
        return np.nan
    n = min(len(chain) for chain in chains)
    chains = np.array([np.asarray(chain, dtype = np.float64)[:n] for chain in chains])
    chain_means = np.mean(chains, axis = 1)
    within = np.mean(np.var(chains, axis = 1, ddof = 1))
    between = n * np.var(chain_means, ddof = 1)
    if within <= 0:
        return 1.0 if between <= 0 else np.inf
    var_hat = (n - 1) / n * within + between / n
    return float(np.sqrt(var_hat / within))


def chain_key(map_seed, sub_chain_index):
    """The MCMC root key of one sub-chain: distinct per sub-chain, disjoint from load_sim's."""
    stream = jax.random.fold_in(jax.random.PRNGKey(map_seed), LOUIS_CHAIN_STREAM)
    return jax.random.fold_in(stream, sub_chain_index)


def louis_quantities(hessians, scores, names = None):
    """(labels, series): every per-sweep scalar Louis's identity averages, one column each.

    The score components first, then the upper triangle of the Hessian (it is symmetric, so
    the lower triangle adds nothing). These are the quantities whose ACF / IAT / R-hat are
    measured, and the pruning stride is the largest IAT among them.
    """
    hessians, scores = np.asarray(hessians), np.asarray(scores)
    n_param = scores.shape[1]
    names = [str(i) for i in range(n_param)] if names is None else [str(n) for n in names]
    rows, cols = np.triu_indices(n_param)
    labels = ([f"score[{names[i]}]" for i in range(n_param)]
              + [f"H[{names[i]},{names[j]}]" for i, j in zip(rows, cols)])
    return labels, np.concatenate([scores, hessians[:, rows, cols]], axis = 1)


def posterior_mixed_draws(data_set, cf_op, cphi_op, d_fid, g_fid, n_draws, rng_key,
                          on_draw = None, verbose = True):
    """Draws of the mixed pair (f°, phi°) from p(f°, phi° | d, theta_0), theta held fixed.

    The same Gibbs sweep sample_lcdm.sample_joint runs, minus its theta step: f from its
    exact conditional (gibbs_sample_f), a mix, one HMC step on phi° (gibbs_sample_phi), and
    an unmix to hand the next f step the new phi. D and G never change, so the (f°, phi°)
    pair at the end of each sweep IS a posterior draw in exactly the fixed mixed coordinates
    the stencil differentiates in - no re-mixing needed. The chain starts at phi = 0 and
    hands EVERY sweep to the caller, the first one included.

    There is deliberately no burn-in and no thinning here. How long a chain takes to reach
    the typical set and how fast it then decorrelates are properties of the realization, not
    of the box, so fixing either up front would be guessing a different number for every
    job. The raw chain is saved instead and cut / pruned in post-processing - the same order
    chain_analysis.py uses on the theta chains (burn-in -> autocorrelation ->
    integrated_autocorrelation_time -> prune_chains). `rng_key` is the sub-chain's own key
    (chain_key), so sub-chains of one realization share the data and nothing else.

    `on_draw(sweep_index, mixed_field, mixed_phi, accepted)` CONSUMES each sweep as it is
    produced (`accepted` is that sweep's phi HMC decision, so a caller that checkpoints
    mid-chain can save the acceptance record so far) and is the way callers should use
    this: a mixed pair is two
    (nside, nside//2+1) complex128 arrays - 260 kB per draw at nside 128, so a few thousand
    sweeps would be gigabytes of fields held for no reason. With a callback nothing is
    accumulated and the returned draw list is empty; without one every draw is kept, which
    only suits short diagnostic chains.

    Returns (draws, diagnostics) with draws the kept pairs (empty when `on_draw` is given)
    and diagnostics carrying the per-sweep phi acceptances (so the rate can be recomputed
    after any burn-in), their mean and the sweep count.
    """
    #imported here: sample_lcdm pulls in the whole sampler stack, which the plain mixed
    #Hessian path never needs
    from cmb_lensing.sample_lcdm import gibbs_sample_f, gibbs_sample_phi
    from cmb_lensing.map_joint import map_joint

    #the posterior is taken under EXACTLY the operators the stencil's centre point uses, so
    #the draws and the derivatives describe the same density
    data_set = data_set.replace(field_covariance = cf_op, phi_covariance = cphi_op,
                                mixing_d = d_fid, mixing_g = g_fid)
    args = dict(noise_covariance = data_set.noise_covariance, mask = data_set.mask,
                beam = data_set.beam, field_covariance = cf_op, phi_covariance = cphi_op,
                mixing_d = d_fid, mixing_g = g_fid,
                quadratic_estimate = data_set.quadratic_estimate)

    phi = 0*data_set.phi #map_joint(data_set)
    field_zeroes = 0 * data_set.unlensed_field
    key = rng_key

    draws, accepts = [], []
    for sweep in range(1, n_draws + 1):
        key, key_f, key_phi = jax.random.split(key, 3)
        field = gibbs_sample_f(field_zeroes, data_set.data, phi, args, key_f)
        mixed_field, mixed_phi = mix(field, phi, d_fid, g_fid)
        mixed_phi, _, accept = gibbs_sample_phi(mixed_phi, mixed_field, data_set.data,
                                                key_phi, args, sweep, 0)
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


def louis_realization(nside, theta_pix, noise_level, is_sampled, param_ground, map_seed,
                      n_draws = DEFAULT_LOUIS_DRAWS, sub_chain_index = 0,
                      step_fracs = None, l_knee = 0, shared_cls = None, verbose = True,
                      on_sweep = None):
    """ONE sub-chain of one realization of Louis's observed information in mixed coordinates.

        I(d) = E_{f°,phi°|d}[ -d2 l_c ] - Cov_{f°,phi°|d}[ d l_c ]

    with l_c = mixed_logpdf, the expectation and covariance taken over posterior draws of
    the mixed pair at theta_0 (posterior_mixed_draws), and every draw differentiated on the
    same 19-point stencil as the plain mixed Hessian - which also yields its score from the
    +/-h points at no extra cost. Averaged over data realizations this is the MARGINAL Fisher
    of p(d | theta): the first term alone (what mixed_hessian_realization averages to) counts
    f and phi as observed; the second removes what marginalizing over them loses.

    Also returns the plain mixed Hessian at the TRUE mixed pair of the same realization: its
    average and the average of the first term estimate the same number, which checks the
    posterior draws. It depends only on `map_seed`, so every sub-chain returns the same one.

    This returns the RAW chain - `n_draws` sweeps from the very first, no burn-in, no
    thinning - as per-sweep `hessians` / `scores` / `phi_accepts`. The information itself is
    built afterwards by louis_information_from_chains from all the sub-chains of the
    realization (`sub_chain_index` selects this one's MCMC randomness via chain_key; the
    data map is the same for every index).

    `on_sweep(snapshot)` is the CHECKPOINT hook: it is called once as soon as the truth
    Hessian exists (zero sweeps) and again after every differentiated sweep, with the same
    dict this function returns but holding only the sweeps done so far and
    `finished = False`. run_single_mixed_hessian.py rewrites its npz there, so a job killed
    at the wall-clock limit keeps everything up to its last sweep, and a running chain can be
    copied off and diagnosed before it finishes. The return value has `finished = True`.
    """
    names, steps, _ = sampled_names_and_steps(is_sampled, param_ground, step_fracs)
    stencil = stencil_offsets(len(names))
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)

    if shared_cls is None:
        shared_cls = camb_cls_at_params(param_ground)
    data_set = load_sim(nside, theta_pix, "I", map_seed, **load_sim_cosmology(param_ground),
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = shared_cls)

    qe_frozen = data_set.quadratic_estimate.scalar_matrix
    cf_op, cphi_op, d_fid, g_fid = _mixed_theta_matrices(param_ground, data_set, nside,
                                                         pix_width, ell_grid, qe_frozen)
    stencil_args = (data_set, names, param_ground, steps, stencil, nside, pix_width,
                    ell_grid, qe_frozen)

    hessian_truth = mixed_logpdf_hessian_one_realization(
        data_set, names, param_ground, steps, stencil, nside, pix_width, ell_grid)

    if verbose:
        print(f"realization seed {map_seed}, sub-chain {sub_chain_index}: sampling "
              f"(f°, phi°) | d at theta_0 - {n_draws} raw sweeps (no burn-in, un-thinned)")
    #differentiate each draw as the chain produces it, so the only thing that outlives a
    #sweep is its (n, n) Hessian and (n,) score - the mixed pair itself is two complex
    #(nside, nside//2+1) arrays and holding thousands of them would be gigabytes
    hessians, scores, accepts = [], [], []
    n_param = len(names)

    def snapshot(finished):
        #reshape keeps the (0, n, n) / (0, n) shapes of an empty chain, so a checkpoint
        #taken before the first sweep has the same layout as every later one
        return dict(hessian_truth = hessian_truth,
                    hessians = np.array(hessians).reshape(-1, n_param, n_param),
                    scores = np.array(scores).reshape(-1, n_param),
                    phi_accepts = np.array(accepts, dtype = bool), names = names,
                    steps = steps, n_sweeps = len(scores), n_draws = n_draws,
                    phi_acceptance = float(np.mean(accepts)) if accepts else np.nan,
                    sub_chain_index = sub_chain_index, finished = finished)

    def differentiate(index, mixed_field, mixed_phi, accepted):
        values = _mixed_stencil_values(mixed_field, mixed_phi, *stencil_args)
        hessians.append(-hessian_from_stencil(values, steps))
        scores.append(score_from_stencil(values, steps))
        accepts.append(bool(accepted))
        if on_sweep is not None:
            on_sweep(snapshot(finished = False))
        if verbose and index % 10 == 0:
            print(f"    differentiated {index}/{n_draws} draws")

    if on_sweep is not None:
        on_sweep(snapshot(finished = False))
    posterior_mixed_draws(data_set, cf_op, cphi_op, d_fid, g_fid, n_draws,
                          chain_key(map_seed, sub_chain_index), on_draw = differentiate,
                          verbose = verbose)

    result = snapshot(finished = True)
    if verbose:
        print(f"  phi acceptance {result['phi_acceptance']:.3f}; truth Hessian diagonal "
              f"{np.array2string(np.diag(hessian_truth), precision = 4)}")
    return result


def louis_sub_chain_stride(series, use_zero_crossing = LOUIS_USE_ZERO_CROSSING):
    """(stride, iats, acfs) of one post-burn-in sub-chain, over every column of `series`.

    `series` is louis_quantities' (n_sweeps, n_quantities) array. The stride is the MAXIMUM
    over all quantities - score components and Hessian entries alike - of the per-quantity
    stride (the IAT, clamped to >= 1 the way chain_analysis.prune_chains clamps it, or the
    first zero crossing of the ACF). One stride per sub-chain because the complete term and
    the score covariance are built from the same pruned sweeps, so every quantity has to be
    decorrelated at that spacing, not just the fastest-mixing one.
    """
    acfs = [autocorrelation(series[:, q]) for q in range(series.shape[1])]
    iats = np.array([integrated_autocorrelation_time(acf) for acf in acfs])
    if use_zero_crossing:
        strides = [first_zero_crossing_lag(acf) for acf in acfs]
    else:
        strides = [max(1, int(tau)) for tau in iats]
    return int(max(strides)), iats, acfs


def louis_information_from_chains(sub_chains, burn_in = DEFAULT_LOUIS_BURN,
                                  use_zero_crossing = LOUIS_USE_ZERO_CROSSING,
                                  keep_acfs = False, names = None):
    """(information, terms) for ONE realization from its raw sub-chains.

    `sub_chains` is a list of (hessians, scores) pairs, one per sub-chain of the same data
    map, each the UN-CUT, UN-THINNED per-sweep arrays the job saved. This is the whole
    post-processing step, in chain_analysis.py's order:
      1. cut the first `burn_in` sweeps from every sub-chain;
      2. Gelman-Rubin R-hat of every quantity (louis_quantities) across the sub-chains, on
         the cut but un-thinned chains (NaN with a single sub-chain);
      3. thin each sub-chain by its own stride, the max IAT over its score components AND
         Hessian entries (louis_sub_chain_stride);
      4. pool the pruned sweeps of all sub-chains - they sample the same posterior - and
         take complete = mean(-H), missing = Cov(score) over the pool.

    Both terms come from the pruned pool, so the sample count behind them is honest; the
    stride accounts for the Hessian entries precisely because they are thinned too.
    `terms["missing_unpruned"]` is the covariance over the un-thinned pool, for comparison.
    With `keep_acfs` the per-sub-chain ACFs are returned too (for plotting); `names` only
    labels the quantities.
    """
    cut = []
    for index, (hessians, scores) in enumerate(sub_chains):
        hessians, scores = np.asarray(hessians), np.asarray(scores)
        if len(scores) - burn_in < 2:
            raise ValueError(f"sub-chain {index} has {len(scores)} sweeps, too few for a "
                             f"burn-in of {burn_in}; lower the burn-in or run longer chains")
        cut.append((hessians[burn_in:], scores[burn_in:]))

    labels, _ = louis_quantities(*cut[0], names = names)
    series = [louis_quantities(h, s)[1] for h, s in cut]
    r_hat = np.array([gelman_rubin([chain[:, q] for chain in series])
                      for q in range(len(labels))])

    pruned_h, pruned_s, strides, iats, acfs = [], [], [], [], []
    for (hessians, scores), chain in zip(cut, series):
        stride, chain_iats, chain_acfs = louis_sub_chain_stride(chain, use_zero_crossing)
        pruned_h.append(hessians[::stride])
        pruned_s.append(scores[::stride])
        strides.append(stride)
        iats.append(chain_iats)
        acfs.append(chain_acfs)
    pruned_h, pruned_s = np.concatenate(pruned_h), np.concatenate(pruned_s)

    complete = np.mean(pruned_h, axis = 0)
    missing = np.atleast_2d(np.cov(pruned_s, rowvar = False))
    missing_unpruned = np.atleast_2d(np.cov(np.concatenate([s for _, s in cut]),
                                            rowvar = False))
    iats = np.array(iats)
    terms = dict(complete = complete, missing = missing, missing_unpruned = missing_unpruned,
                 labels = labels, r_hat = r_hat, iats = iats, strides = np.array(strides),
                 #which quantity set each sub-chain's stride (the argmax of its IATs)
                 stride_setter = np.argmax(iats, axis = 1),
                 n_chains = len(cut), n_draws = int(sum(len(s) for _, s in cut)),
                 effective_draws = len(pruned_s), burn_in = burn_in)
    if keep_acfs:
        terms["acfs"] = acfs
    return complete - missing, terms


def mixed_hessian_realization(nside, theta_pix, noise_level, is_sampled, param_ground,
                              map_seed, step_fracs = None, l_knee = 0, shared_cls = None):
    """One realization end to end: simulate, mix, finite-difference.

    This is what run_single_mixed_hessian.py calls - one slurm job, one realization, one
    npz - so that the 100-realization average is built in parallel instead of sequentially.
    Returns (hessian, names, steps).
    """
    names, steps, _ = sampled_names_and_steps(is_sampled, param_ground, step_fracs)
    stencil = stencil_offsets(len(names))

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    camb_kwargs = load_sim_cosmology(param_ground)

    #reuse the fiducial spectra rather than re-running CAMB inside load_sim (see
    #load_sim's precomputed_cls); every realization sits at the same cosmology
    if shared_cls is None:
        shared_cls = camb_cls_at_params(param_ground)
    data_set = load_sim(nside, theta_pix, "I", map_seed, **camb_kwargs,
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = shared_cls)

    hessian = mixed_logpdf_hessian_one_realization(
        data_set, names, param_ground, steps, stencil, nside, pix_width, ell_grid)
    return hessian, names, steps


def forecast_from_mixed_logpdf(nside, theta_pix, noise_level, is_sampled, param_ground,
                               n_realizations = 100, step_fracs = None, l_knee = 0,
                               map_seed = 20260908, hessian_dir = None, verbose = True,
                               louis = False, louis_draws = DEFAULT_LOUIS_DRAWS,
                               louis_burn = DEFAULT_LOUIS_BURN,
                               louis_chains = DEFAULT_LOUIS_CHAINS):
    """The realization-averaged mixed-coordinate Hessian, computed here or read from disk.

    With `louis = True` each realization contributes Louis's observed information
    (louis_realization) instead of the Hessian at its true fields, so the average is the
    marginal Fisher of p(d | theta) rather than the complete-data one; cached files must
    then be louis ones (method "mixed_louis"). `louis_burn` is the POST-PROCESSING burn-in,
    cut from every raw sub-chain before anything is measured, in both modes;
    `louis_draws` (raw sweeps per sub-chain) and `louis_chains` (sub-chains per
    realization) only matter when computing here.

    With `hessian_dir = None` this is the sequential driver - the local / small-N path. For
    the full 100 realizations that is slow (two lensing solves per stencil point per
    realization), which is what sampling_chains/mixed_hessian.sh is for: it fans the same
    per-realization work across slurm jobs, one hessian_<index>.npz each. Point
    `hessian_dir` at that output directory and every file in it is averaged INSTEAD of
    computing anything; `n_realizations` and `map_seed` are then ignored (the files decide
    both), and the files must have been produced for exactly the box, noise, l_knee and
    parameter set requested here, or a ValueError says which differs - a forecast must
    describe the configuration it is labelled with.

    Returns (fisher, names, per_realization).
    """
    names, _, _ = sampled_names_and_steps(is_sampled, param_ground, step_fracs)

    if hessian_dir is not None:
        requested = dict(names = names, nside = nside, theta_pix = theta_pix,
                         noise_level = noise_level, l_knee = l_knee)
        fisher, names, per_realization, metadata = load_hessian_directory(
            hessian_dir, verbose = verbose, method = LOUIS_METHOD if louis else METHOD,
            burn_in = louis_burn)
        _check_cached_config(hessian_dir, metadata, requested)
        return fisher, names, per_realization

    if verbose:
        print(f"Fisher forecast [{LOUIS_METHOD if louis else METHOD}]: nside {nside}, "
              f"theta_pix {theta_pix}', {noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  sampled: {names}")
        draws = f"(1 + {louis_chains} x {louis_draws})" if louis else "1"
        print(f"  {n_realizations} realizations x {draws} mixed pair(s) x "
              f"{len(stencil_offsets(len(names)))} mixed_logpdf evaluations, 2 lensing "
              f"solves each")

    shared_cls = camb_cls_at_params(param_ground)
    per_realization, louis_terms = [], []
    for index in range(n_realizations):
        if louis:
            results = [louis_realization(nside, theta_pix, noise_level, is_sampled,
                                         param_ground, map_seed + index,
                                         n_draws = louis_draws, sub_chain_index = chain,
                                         step_fracs = step_fracs, l_knee = l_knee,
                                         shared_cls = shared_cls, verbose = verbose)
                       for chain in range(louis_chains)]
            hessian, terms = louis_information_from_chains(
                [(result["hessians"], result["scores"]) for result in results],
                burn_in = louis_burn, names = results[0]["names"])
            names = results[0]["names"]
            louis_terms.append(_louis_summary(terms, results[0]["hessian_truth"],
                                              [result["phi_accepts"] for result in results]))
        else:
            hessian, names, _ = mixed_hessian_realization(
                nside, theta_pix, noise_level, is_sampled, param_ground, map_seed + index,
                step_fracs = step_fracs, l_knee = l_knee, shared_cls = shared_cls)
        per_realization.append(hessian)
        if verbose and (index + 1) % 10 == 0:
            running = np.mean(per_realization, axis = 0)
            print(f"    {index + 1}/{n_realizations}  running diagonal "
                  f"{np.array2string(np.diag(running), precision = 4)}")

    per_realization = np.array(per_realization)
    if verbose and louis_terms:
        _report_louis_terms(louis_terms)
    return np.mean(per_realization, axis = 0), names, per_realization


def _check_cached_config(directory, metadata, requested):
    """Raise unless the cached Hessians were produced for the configuration requested."""
    mismatches = [f"{key}: files {metadata[key]!r}, requested {requested[key]!r}"
                  for key in requested if metadata[key] != requested[key]]
    if mismatches:
        raise ValueError(
            f"the Hessians in {directory} were not produced for the requested "
            f"configuration - " + "; ".join(mismatches) + ". Pass the box the files were "
            f"made for (hessian_directory_config reads it), or leave hessian_dir unset "
            f"to compute the realizations here")


def hessian_directory_config(directory):
    """The configuration every hessian_*.npz in `directory` was produced with.

    Reads only the metadata written by run_single_mixed_hessian.py and checks the files
    agree on it, so results from two different runs cannot be silently averaged together.
    Returns {"names", "nside", "theta_pix", "noise_level", "l_knee", "method",
    "n_realizations", "paths"}; n_realizations counts distinct map seeds, since a louis
    realization is spread over one file per sub-chain.
    """
    paths = sorted(glob.glob(os.path.join(directory, "hessian_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"no hessian_*.npz in {directory}. Run sampling_chains/mixed_hessian.sh first, "
            f"or point --hessian_dir at the out_dir that script writes to.")

    config = None
    seeds = set()
    for path in paths:
        data = np.load(path, allow_pickle = True)
        seeds.add(int(data["map_seed"]))
        #npz round-trips strings as np.str_; they compare and hash like str but print as
        #np.str_('omch2'), so convert back for readable reports and clean npz round-trips
        current = dict(names = [str(name) for name in data["names"]],
                       nside = int(data["nside"]), theta_pix = float(data["theta_pix"]),
                       noise_level = float(data["noise_level"]),
                       l_knee = float(data["l_knee"]), method = str(data["method"]))
        if config is None:
            config = current
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; a forecast cannot mix configurations")

    return dict(config, n_realizations = len(seeds), paths = paths)


def load_louis_chains(directory):
    """Every louis file in `directory`, grouped into realizations of raw sub-chains.

    Returns (metadata, realizations): metadata as hessian_directory_config gives it (with
    n_realizations corrected to the number of distinct seeds, not files), and one dict per
    map_seed, sorted by seed:
        {"map_seed", "hessian_truth", "job_burn",
         "sub_chains": [{"sub_chain_index", "hessians", "scores", "phi_accepts"}, ...]}
    `phi_accepts` is None for files written before per-sweep acceptances were saved.
    `job_burn` is the burn-in the JOB already cut (0 for current files, whose chains start
    at sweep 1; older files cut `louis_burn` in the job and saved it as `n_burn`) - any
    merge-time burn-in comes on top of it.

    Refuses duplicate (map_seed, sub_chain_index) pairs, and sub-chains of one seed that
    disagree on the truth Hessian (which depends only on the data map, so a disagreement
    means two different maps were given one seed).
    """
    metadata = hessian_directory_config(directory)
    if metadata["method"] != LOUIS_METHOD:
        raise ValueError(f"{directory} holds '{metadata['method']}' Hessians, not "
                         f"'{LOUIS_METHOD}' ones")

    by_seed = {}
    for path in metadata["paths"]:
        data = np.load(path, allow_pickle = True)
        if "scores" not in data.files:
            raise ValueError(f"{path} predates the saved per-sweep chains; it cannot be "
                             f"re-burned or re-thinned")
        seed = int(data["map_seed"])
        #files written before sub-chains existed are sub-chain 0 of their realization
        index = int(data["sub_chain_index"]) if "sub_chain_index" in data.files else 0
        realization = by_seed.setdefault(seed, dict(map_seed = seed, sub_chains = [],
                                                    hessian_truth = data["hessian_truth"],
                                                    job_burn = int(data["n_burn"])
                                                    if "n_burn" in data.files else 0))
        if any(chain["sub_chain_index"] == index for chain in realization["sub_chains"]):
            raise ValueError(f"{directory} holds sub-chain {index} of map_seed {seed} twice; "
                             f"it would be double counted - check the loops in "
                             f"mixed_hessian.sh")
        if not np.allclose(data["hessian_truth"], realization["hessian_truth"],
                           rtol = 1e-8, atol = 0):
            raise ValueError(f"{path}: the truth Hessian differs from another sub-chain of "
                             f"map_seed {seed}, so they were not run on the same data map")
        realization["sub_chains"].append(dict(
            sub_chain_index = index, hessians = data["hessians"], scores = data["scores"],
            phi_accepts = data["phi_accepts"] if "phi_accepts" in data.files else None,
            #False for a checkpoint of a job still running or killed at its wall clock;
            #files from before checkpointing were only ever written on completion
            finished = bool(data["finished"]) if "finished" in data.files else True))

    realizations = [by_seed[seed] for seed in sorted(by_seed)]
    for realization in realizations:
        realization["sub_chains"].sort(key = lambda chain: chain["sub_chain_index"])
    return metadata, realizations


def usable_louis_chains(realizations, burn_in, verbose = True):
    """Drop sub-chains too short to outlive `burn_in`, and realizations left with none.

    Jobs checkpoint every sweep, so a directory can hold sub-chains that are still running
    or were killed at the wall clock. Those are valid PREFIXES of their chains and are used
    as they are; only one with fewer than burn_in + 2 sweeps (nothing left to take a
    covariance of) is set aside, and reported, rather than failing the whole merge.
    Returns the kept realizations (new dicts; the input is not modified).
    """
    kept, dropped = [], []
    for realization in realizations:
        chains = [chain for chain in realization["sub_chains"]
                  if len(chain["scores"]) >= burn_in + 2]
        dropped += [(realization["map_seed"], chain["sub_chain_index"], len(chain["scores"]))
                    for chain in realization["sub_chains"]
                    if len(chain["scores"]) < burn_in + 2]
        if chains:
            kept.append(dict(realization, sub_chains = chains))
    if verbose:
        unfinished = [(realization["map_seed"], chain["sub_chain_index"], len(chain["scores"]))
                      for realization in kept for chain in realization["sub_chains"]
                      if not chain["finished"]]
        if unfinished:
            lengths = [length for _, _, length in unfinished]
            print(f"  {len(unfinished)} sub-chain(s) are UNFINISHED checkpoints (running or "
                  f"killed), used as the {min(lengths)}-{max(lengths)} sweeps they reached")
        if dropped:
            print(f"  set aside {len(dropped)} sub-chain(s) with fewer than {burn_in + 2} "
                  f"sweeps (map_seed, sub-chain, sweeps): {dropped[:10]}"
                  + (" ..." if len(dropped) > 10 else ""))
        lost = len(realizations) - len(kept)
        if lost:
            print(f"  {lost} realization(s) have no usable sub-chain yet and are left out")
    if not kept:
        raise ValueError(f"no sub-chain has more than burn_in + 1 = {burn_in + 1} sweeps yet; "
                         f"wait for the jobs or lower the burn-in")
    return kept


def louis_realization_information(realization, burn_in = DEFAULT_LOUIS_BURN,
                                  keep_acfs = False, names = None):
    """louis_information_from_chains on one realization as load_louis_chains returns it."""
    return louis_information_from_chains(
        [(chain["hessians"], chain["scores"]) for chain in realization["sub_chains"]],
        burn_in = burn_in, keep_acfs = keep_acfs, names = names)


def _louis_summary(terms, hessian_truth, phi_accepts, burn_in = None):
    """The per-realization numbers _report_louis_terms aggregates.

    Acceptance is recomputed after the burn-in from the per-sweep record when there is one
    (a pre-sub-chain file has only its whole-chain rate, which is used as is).
    """
    burn_in = terms["burn_in"] if burn_in is None else burn_in
    acceptance = [np.mean(accepts[burn_in:]) for accepts in phi_accepts
                  if accepts is not None]
    return dict(complete = terms["complete"], missing = terms["missing"],
                truth = np.asarray(hessian_truth),
                acceptance = np.mean(acceptance) if acceptance else np.nan,
                labels = terms["labels"], r_hat = terms["r_hat"], iats = terms["iats"],
                strides = terms["strides"], stride_setter = terms["stride_setter"],
                effective = terms["effective_draws"], swept = terms["n_draws"],
                n_chains = terms["n_chains"])


def load_hessian_directory(directory, verbose = True, method = METHOD,
                           burn_in = DEFAULT_LOUIS_BURN):
    """Average the per-realization Hessian npz files written by run_single_mixed_hessian.py.

    The parallel counterpart to computing the realizations in forecast_from_mixed_logpdf
    (which calls this when given hessian_dir): slurm jobs each write one npz, and this
    collects them, refusing mixed configurations and duplicate seeds. `method` is what the
    files must hold: METHOD for plain mixed Hessians (one hessian_<index>.npz per
    realization, averaged as they are), LOUIS_METHOD for Louis's observed information.

    Louis files hold one RAW sub-chain each (hessian_<index>_chain_<sub>.npz); they are
    grouped by map_seed (load_louis_chains) and each realization's information is rebuilt
    HERE from its sub-chains' per-sweep Hessians and scores - `burn_in` cut, R-hat measured,
    each sub-chain thinned by its max IAT over scores and Hessian entries, pooled
    (louis_information_from_chains). Nothing the job computed about the chain is trusted,
    so burn-in and thinning can be changed by re-merging alone. `burn_in` is ignored for
    plain Hessians.

    Returns (fisher, names, per_realization, metadata) with metadata as
    hessian_directory_config gives it (n_realizations counting seeds, not files).
    """
    if method == LOUIS_METHOD:
        metadata, realizations = load_louis_chains(directory)
        if verbose:
            print(f"Merging mixed_louis sub-chains from {directory}")
        realizations = usable_louis_chains(realizations, burn_in, verbose = verbose)
        hessians, louis_terms = [], []
        for realization in realizations:
            information, terms = louis_realization_information(realization, burn_in,
                                                               names = metadata["names"])
            hessians.append(information)
            louis_terms.append(_louis_summary(
                terms, realization["hessian_truth"],
                [chain["phi_accepts"] for chain in realization["sub_chains"]]))
        job_burns = sorted({realization["job_burn"] for realization in realizations})
    else:
        metadata = hessian_directory_config(directory)
        if metadata["method"] != method:
            raise ValueError(f"{directory} holds '{metadata['method']}' Hessians, not "
                             f"'{method}' ones")
        hessians, seeds, louis_terms = [], [], []
        for path in metadata["paths"]:
            data = np.load(path, allow_pickle = True)
            seeds.append(int(data["map_seed"]))
            hessians.append(data["hessian"])
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                             f"realizations would be double counted. Each slurm job must "
                             f"use a distinct seed - check the map_prefix loop in "
                             f"mixed_hessian.sh.")

    per_realization = np.array(hessians)
    if verbose:
        print(f"Averaging {len(hessians)} cached realizations from {directory}")
        print(f"  method {metadata['method']}, parameters {metadata['names']}, nside "
              f"{metadata['nside']}, {metadata['theta_pix']:g}', "
              f"{metadata['noise_level']:g} uK-arcmin, l_knee {metadata['l_knee']:g}")
        if louis_terms:
            print(f"  burn-in cut at merge: {burn_in} sweeps per sub-chain (on top of "
                  f"{job_burns} already cut in the job{'s' if len(job_burns) > 1 else ''})")
        spread = np.std(per_realization, axis = 0) / np.sqrt(len(hessians))
        mean = np.mean(per_realization, axis = 0)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            relative = np.abs(spread / mean)
        print(f"  standard error on the mean, relative: max {np.nanmax(relative):.3f}")
        if louis_terms:
            _report_louis_terms(louis_terms)

    return np.mean(per_realization, axis = 0), metadata["names"], per_realization, metadata


def _report_louis_terms(louis_terms):
    """The two Louis terms, and the checks that the posterior draws are doing their job.

    E_d[E_{f°,phi°|d}(-H)] and E over prior draws of -H at the true fields are the same
    number by the tower property, so the complete term must agree with the truth Hessian
    within their errors; a real gap means the posterior chain is not sampling p(f°, phi° | d).

    The chain diagnostics are per realization (and per sub-chain) by construction: each
    data realization mixes at its own rate. Reported per quantity (every score component and
    Hessian entry): the mean and std over realizations of the Gelman-Rubin R-hat across that
    realization's sub-chains, the mean IAT over all sub-chains, and how often that quantity
    was the slowest one, i.e. set its sub-chain's stride.
    """
    summary = {key: [terms[key] for terms in louis_terms]
             for key in ("complete", "missing", "truth", "acceptance", "strides",
                         "effective", "swept", "n_chains")}
    complete, missing, truth = (np.array(summary[key]) for key in ("complete", "missing",
                                                                   "truth"))
    acceptance = np.array(summary["acceptance"])
    stride = np.concatenate(summary["strides"])
    effective, swept = np.array(summary["effective"]), np.array(summary["swept"])
    n = len(louis_terms)

    def mean_and_error(stack):
        return np.mean(stack, axis = 0), np.std(stack, axis = 0) / np.sqrt(n)

    for label, stack in (("complete  <-H>_post", complete), ("missing   Cov_post[t]", missing),
                         ("truth     -H(true f, phi)", truth)):
        mean, error = mean_and_error(stack)
        print(f"  {label}: diagonal {np.array2string(np.diag(mean), precision = 4)} "
              f"+/- {np.array2string(np.diag(error), precision = 4)}")
    fraction = np.diag(np.mean(missing, axis = 0)) / np.diag(np.mean(complete, axis = 0))
    print(f"  missing-information fraction (diagonal): "
          f"{np.array2string(fraction, precision = 3)}")
    difference = np.diag(np.mean(complete - truth, axis = 0))
    difference_error = np.diag(np.std(complete - truth, axis = 0)) / np.sqrt(n)
    print(f"  complete - truth (should be 0): {np.array2string(difference, precision = 4)} "
          f"+/- {np.array2string(difference_error, precision = 4)}")
    acceptance_text = (f"{np.nanmean(acceptance):.3f}" if np.any(np.isfinite(acceptance))
                       else "n/a (files predate per-sweep acceptances)")
    print(f"  phi acceptance (post burn-in) {acceptance_text}; sub-chains per "
          f"realization {np.min(summary['n_chains'])}-{np.max(summary['n_chains'])}")

    #per-quantity convergence and mixing: R-hat across each realization's sub-chains, IAT
    #over every sub-chain, and how often each quantity was the slowest (set the stride)
    labels = louis_terms[0]["labels"]
    r_hat = np.array([terms["r_hat"] for terms in louis_terms])
    iats = np.concatenate([terms["iats"] for terms in louis_terms])
    setters = np.concatenate([terms["stride_setter"] for terms in louis_terms])
    has_r_hat = np.any(np.isfinite(r_hat))
    width = max(len(label) for label in labels)
    print(f"  {'quantity':<{width}}  {'R-hat mean +/- std':>20}  {'IAT mean (max)':>16}  "
          f"sets stride")
    for q, label in enumerate(labels):
        r_text = (f"{np.nanmean(r_hat[:, q]):.4f} +/- {np.nanstd(r_hat[:, q]):.4f}"
                  if has_r_hat else "n/a (1 sub-chain)")
        print(f"  {label:<{width}}  {r_text:>20}  "
              f"{np.mean(iats[:, q]):>7.2f} ({np.max(iats[:, q]):>6.2f})  "
              f"{np.mean(setters == q):>10.0%}")
    if has_r_hat:
        worst = np.nanmax(r_hat, axis = 1)
        flagged = np.sum(worst > LOUIS_R_HAT_WARN)
        print(f"  realizations with max R-hat > {LOUIS_R_HAT_WARN}: {flagged}/{n}"
              + ("  <- raise the burn-in or the chain length" if flagged else ""))

    print(f"  pruning stride per sub-chain (max IAT over scores AND Hessian entries): "
          f"{np.min(stride):.0f}-{np.max(stride):.0f} (median {np.median(stride):.0f}), "
          f"leaving {np.min(effective):.0f}-{np.max(effective):.0f} pooled samples of "
          f"{int(np.median(swept))} post-burn-in sweeps per realization")
    #a covariance from a handful of samples is mostly noise, whatever the stride is
    if np.min(effective) < 10:
        print(f"  WARNING: the worst realization is pruned to {np.min(effective):.0f} "
              f"samples, so its missing term is badly determined; raise louis_draws or "
              f"louis_chains for this box")


def averaged_hessian_covariance(fisher, names, n_realizations):
    """Invert an averaged Hessian, blaming too few realizations rather than the stencil."""
    try:
        return covariance_from_fisher(fisher, names)
    except RuntimeError as error:
        raise RuntimeError(
            f"the averaged Hessian is not positive definite over {n_realizations} "
            f"realizations. For a realization-averaged method that usually means too few "
            f"realizations rather than a bad finite-difference step - add more jobs and "
            f"re-run the averaging. Underlying: {error}") from error


def main():
    parser = argparse.ArgumentParser(
        description = "Fisher forecast from the realization-averaged Hessian of "
                      "statistics.mixed_logpdf at a fixed mixed (f, phi) pair")
    add_box_arguments(parser)
    parser.add_argument("--realizations", type = int, default = 100,
                        help = "(data, f, phi) draws to average over, sequentially. "
                               "Ignored with --hessian_dir, where the files decide")
    parser.add_argument("--hessian_dir", type = str, default = "/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/mixed_hessian_output/",
                        help = "average per-realization hessian_*.npz files written by "
                               "sampling_chains/mixed_hessian.sh instead of computing them "
                               "here. This is the time-feasible route: 100 slurm jobs run "
                               "the realizations in parallel and this just collects them. "
                               "The box arguments (--nside, --theta_pix, --noise, "
                               "--l_knee, --params) are then taken from the files, not "
                               "from the command line. Pass an empty string to compute "
                               "the realizations here instead")
    parser.add_argument("--louis", action = "store_true",
                        help = "Louis's observed information instead of the complete-data "
                               "Hessian: each realization's mixed Hessian is averaged over "
                               "posterior draws of (f, phi) given d and the posterior "
                               "covariance of the score is subtracted, giving the MARGINAL "
                               "Fisher of p(d | theta). With --hessian_dir it is inferred "
                               "from the files")
    parser.add_argument("--louis_draws", type = int, default = DEFAULT_LOUIS_DRAWS,
                        help = "raw Gibbs sweeps per sub-chain, burn-in included "
                               "(sequential mode only)")
    parser.add_argument("--louis_chains", type = int, default = DEFAULT_LOUIS_CHAINS,
                        help = "independent sub-chains per data realization, for R-hat "
                               "(sequential mode only; the files decide with --hessian_dir)")
    parser.add_argument("--louis_burn", type = int, default = DEFAULT_LOUIS_BURN,
                        help = "burn-in cut from every raw sub-chain at MERGE time, in both "
                               "modes - re-merge with a different value, never re-run")
    args = parser.parse_args()
    hessian_dir = args.hessian_dir or None

    if hessian_dir is not None:
        #the files fix the configuration; the command line's box defaults would otherwise
        #describe a box the cached Hessians were never computed for
        cached = hessian_directory_config(hessian_dir)
        args.nside, args.theta_pix = cached["nside"], cached["theta_pix"]
        args.noise, args.l_knee = cached["noise_level"], cached["l_knee"]
        args.params = cached["names"]
        args.realizations = cached["n_realizations"]
        args.louis = cached["method"] == LOUIS_METHOD
    is_sampled = sampled_from_args(parser, args)
    method, suffix, label = ((LOUIS_METHOD, LOUIS_SUFFIX, LOUIS_LABEL) if args.louis
                             else (METHOD, SUFFIX, LABEL))

    fisher, names, per_realization = forecast_from_mixed_logpdf(
        args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
        n_realizations = args.realizations, l_knee = args.l_knee, hessian_dir = hessian_dir,
        louis = args.louis, louis_draws = args.louis_draws, louis_burn = args.louis_burn,
        louis_chains = args.louis_chains)
    covariance = averaged_hessian_covariance(fisher, names, len(per_realization))
    report_sigmas(covariance, names, SPECTRA, method)

    source = "cached" if hessian_dir is not None else "sequential"
    subtitle = (f"{box_subtitle(SPECTRA, args)}  |  {len(per_realization)} realizations "
                f"({source})")
    config = run_config(SPECTRA, args, names, n_realizations = len(per_realization),
                        standard_error = np.std(per_realization, axis = 0)
                                         / np.sqrt(len(per_realization)))
    save_outputs(fisher, covariance, names, method, suffix, label, subtitle, config)


if __name__ == "__main__":
    main()
