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
posterior mean of -H. The chain is NOT thinned while it runs: every post-burn-in sweep is
kept, and the un-thinned score chain is pruned in post-processing by its own measured
stride (louis_information_from_terms), the same autocorrelation -> IAT -> prune order
chain_analysis.py applies to the theta chains. How fast a chain decorrelates is a property
of the data realization, so that stride cannot be known before the chain exists. Each realization also evaluates the plain Hessian at its true fields:
by the tower property the complete term and that must agree on average, which checks the
chain. Cost is (1 + draws) stencils plus the Gibbs sweeps per realization - hours rather
than minutes at nside 128 - and the files are method "mixed_louis", written with
SUFFIX "_from_mixed_louis".

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
#post-burn-in sweeps kept per realization, and sweeps discarded first. The chain starts at
#the MAP phi, so the burn-in only has to cover the move from the mode into the typical set.
#There is NO thinning: every post-burn-in sweep is kept and differentiated, and the
#correlation between them is measured afterwards from the un-thinned score chain
#(louis_information_from_terms). So `draws` is a raw chain length, not a count of
#independent samples - the effective number is draws / tau, which the merge reports
DEFAULT_LOUIS_DRAWS = 50
DEFAULT_LOUIS_BURN = 100
#which stride the post-processing prunes the score chain by, mirroring chain_analysis.py's
#USE_ZERO_CROSSING_PRUNE: False -> the integrated autocorrelation time, True -> the lag just
#before the ACF first crosses zero
LOUIS_USE_ZERO_CROSSING = False


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


def posterior_mixed_draws(data_set, cf_op, cphi_op, d_fid, g_fid, n_draws, n_burn,
                          chain_seed, on_draw = None, verbose = True):
    """Draws of the mixed pair (f°, phi°) from p(f°, phi° | d, theta_0), theta held fixed.

    The same Gibbs sweep sample_lcdm.sample_joint runs, minus its theta step: f from its
    exact conditional (gibbs_sample_f), a mix, one HMC step on phi° (gibbs_sample_phi), and
    an unmix to hand the next f step the new phi. D and G never change, so the (f°, phi°)
    pair at the end of each sweep IS a posterior draw in exactly the fixed mixed coordinates
    the stencil differentiates in - no re-mixing needed. The chain starts at map_joint's MAP
    phi (the sampler's default), discards `n_burn` sweeps, and keeps EVERY sweep after that.

    There is deliberately no thinning here. How fast a chain decorrelates is a property of
    the realization, not of the box, so guessing a stride up front would be guessing a
    different number for every job. The un-thinned score chain is saved instead and pruned
    in post-processing by its own measured stride - the same order chain_analysis.py uses on
    the theta chains (autocorrelation -> integrated_autocorrelation_time -> prune_chains).

    `on_draw(sweep_index, mixed_field, mixed_phi)` CONSUMES each kept sweep as it is
    produced and is the way callers should use this: a mixed pair is two
    (nside, nside//2+1) complex128 arrays - 260 kB per draw at nside 128, so a few thousand
    sweeps would be gigabytes of fields held for no reason. With a callback nothing is
    accumulated and the returned draw list is empty; without one every draw is kept, which
    only suits short diagnostic chains.

    Returns (draws, diagnostics) with draws the kept pairs (empty when `on_draw` is given)
    and diagnostics carrying the acceptance rate and the sweep count.
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
    key = jax.random.PRNGKey(chain_seed)

    draws, accepts = [], []
    kept = 0
    for sweep in range(1, n_burn + n_draws + 1):
        key, key_f, key_phi = jax.random.split(key, 3)
        field = gibbs_sample_f(field_zeroes, data_set.data, phi, args, key_f)
        mixed_field, mixed_phi = mix(field, phi, d_fid, g_fid)
        mixed_phi, _, accept = gibbs_sample_phi(mixed_phi, mixed_field, data_set.data,
                                                key_phi, args, sweep, 0)
        accepts.append(bool(accept))
        _, phi = unmix(mixed_field, mixed_phi, d_fid, g_fid)
        if sweep > n_burn:
            kept += 1
            if on_draw is None:
                draws.append((mixed_field, mixed_phi))
            else:
                #consumed here and dropped: only the callback's summary survives the sweep
                on_draw(kept, mixed_field, mixed_phi)
        if verbose and sweep % 50 == 0:
            print(f"    sweep {sweep}/{n_burn + n_draws}: phi acceptance "
                  f"{np.mean(accepts):.3f}, {kept} draws kept")

    diagnostics = dict(phi_acceptance = float(np.mean(accepts)),
                       n_sweeps = n_burn + n_draws, n_burn = n_burn)
    return draws, diagnostics


def louis_realization(nside, theta_pix, noise_level, is_sampled, param_ground, map_seed,
                      n_draws = DEFAULT_LOUIS_DRAWS, n_burn = DEFAULT_LOUIS_BURN,
                      step_fracs = None, l_knee = 0, shared_cls = None, verbose = True):
    """One realization of Louis's observed information in mixed coordinates.

        I(d) = E_{f°,phi°|d}[ -d2 l_c ] - Cov_{f°,phi°|d}[ d l_c ]

    with l_c = mixed_logpdf, the expectation and covariance taken over posterior draws of
    the mixed pair at theta_0 (posterior_mixed_draws), and every draw differentiated on the
    same 19-point stencil as the plain mixed Hessian - which also yields its score from the
    +/-h points at no extra cost. Averaged over data realizations this is the MARGINAL Fisher
    of p(d | theta): the first term alone (what mixed_hessian_realization averages to) counts
    f and phi as observed; the second removes what marginalizing over them loses.

    Also returns the plain mixed Hessian at the TRUE mixed pair of the same realization: its
    average and the average of the first term estimate the same number, which checks the
    posterior draws.

    `n_draws` is a raw chain length: every post-burn-in sweep is kept and differentiated,
    and the un-thinned scores are saved so the pruning stride can be measured from them in
    post-processing (louis_information_from_terms), per realization.
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
        print(f"realization seed {map_seed}: sampling (f°, phi°) | d at theta_0 - "
              f"{n_burn} burn-in + {n_draws} kept sweeps (un-thinned)")
    #differentiate each draw as the chain produces it, so the only thing that outlives a
    #sweep is its (n, n) Hessian and (n,) score - the mixed pair itself is two complex
    #(nside, nside//2+1) arrays and holding thousands of them would be gigabytes
    hessians, scores = [], []

    def differentiate(index, mixed_field, mixed_phi):
        values = _mixed_stencil_values(mixed_field, mixed_phi, *stencil_args)
        hessians.append(-hessian_from_stencil(values, steps))
        scores.append(score_from_stencil(values, steps))
        if verbose and index % 10 == 0:
            print(f"    differentiated {index}/{n_draws} draws")

    _, chain = posterior_mixed_draws(data_set, cf_op, cphi_op, d_fid, g_fid,
                                     n_draws, n_burn, chain_seed = map_seed,
                                     on_draw = differentiate, verbose = verbose)
    hessians, scores = np.array(hessians), np.array(scores)

    information, terms = louis_information_from_terms(hessians, scores)

    if verbose:
        print(f"  phi acceptance {chain['phi_acceptance']:.3f}; score IAT "
              f"{np.array2string(terms['score_iat'], precision = 2)} -> stride "
              f"{terms['stride']}, {terms['effective_draws']} pruned samples of "
              f"{terms['n_draws']} sweeps")
        print(f"  diagonal: complete "
              f"{np.array2string(np.diag(terms['complete']), precision = 4)}, "
              f"missing {np.array2string(np.diag(terms['missing']), precision = 4)}, "
              f"truth Hessian {np.array2string(np.diag(hessian_truth), precision = 4)}")

    return dict(louis_information = information, hessian_truth = hessian_truth,
                hessians = hessians, scores = scores, names = names, steps = steps,
                n_sweeps = chain["n_sweeps"], n_burn = chain["n_burn"],
                phi_acceptance = chain["phi_acceptance"], **terms)


def louis_information_from_terms(hessians, scores,
                                 use_zero_crossing = LOUIS_USE_ZERO_CROSSING):
    """(information, terms) from one realization's UN-THINNED per-sweep Hessians and scores.

    This is the post-processing step, and it prunes the chain exactly the way
    chain_analysis.py prunes the theta chains: autocorrelation -> a stride
    (integrated_autocorrelation_time by default, first_zero_crossing_lag with
    `use_zero_crossing`, mirroring its USE_ZERO_CROSSING_PRUNE) -> keep every stride-th
    sample. Doing it here rather than in the job is the point: each data realization mixes
    at its own rate, so the stride is measured from that realization's own chain instead of
    being guessed once for every box.

    ONE stride serves all parameters - the largest of the per-parameter ones - because the
    missing term is a single covariance matrix over the joint score, so its rows have to
    stay aligned. The Hessian term is a plain mean, which correlation leaves unbiased, so it
    keeps every sweep; only the covariance is built from the pruned chain.
    """
    hessians, scores = np.asarray(hessians), np.asarray(scores)
    n_draws = len(scores)
    complete = np.mean(hessians, axis = 0)

    acfs = [autocorrelation(scores[:, i]) for i in range(scores.shape[1])]
    score_iat = np.array([integrated_autocorrelation_time(acf) for acf in acfs])
    if use_zero_crossing:
        strides = np.array([first_zero_crossing_lag(acf) for acf in acfs])
    else:
        #clamped the way prune_chains clamps it, so an IAT below 1 cannot produce a zero step
        strides = np.array([max(1, int(tau)) for tau in score_iat])
    stride = int(np.max(strides))

    pruned = scores[0:-1:stride] if stride > 1 else scores
    missing = np.atleast_2d(np.cov(pruned, rowvar = False))
    missing_unpruned = np.atleast_2d(np.cov(scores, rowvar = False))

    terms = dict(complete = complete, missing = missing,
                 missing_unpruned = missing_unpruned, score_iat = score_iat,
                 stride = stride, n_draws = n_draws, effective_draws = len(pruned))
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
                               louis_burn = DEFAULT_LOUIS_BURN):
    """The realization-averaged mixed-coordinate Hessian, computed here or read from disk.

    With `louis = True` each realization contributes Louis's observed information
    (louis_realization) instead of the Hessian at its true fields, so the average is the
    marginal Fisher of p(d | theta) rather than the complete-data one; cached files must
    then be louis ones (method "mixed_louis").

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
            hessian_dir, verbose = verbose, method = LOUIS_METHOD if louis else METHOD)
        _check_cached_config(hessian_dir, metadata, requested)
        return fisher, names, per_realization

    if verbose:
        print(f"Fisher forecast [{LOUIS_METHOD if louis else METHOD}]: nside {nside}, "
              f"theta_pix {theta_pix}', {noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  sampled: {names}")
        draws = (1 + louis_draws) if louis else 1
        print(f"  {n_realizations} realizations x {draws} mixed pair(s) x "
              f"{len(stencil_offsets(len(names)))} mixed_logpdf evaluations, 2 lensing "
              f"solves each")

    shared_cls = camb_cls_at_params(param_ground)
    per_realization = []
    for index in range(n_realizations):
        if louis:
            result = louis_realization(nside, theta_pix, noise_level, is_sampled,
                                       param_ground, map_seed + index,
                                       n_draws = louis_draws,
                                       n_burn = louis_burn, step_fracs = step_fracs,
                                       l_knee = l_knee, shared_cls = shared_cls,
                                       verbose = verbose)
            hessian, names = result["louis_information"], result["names"]
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
    "n_realizations", "paths"}.
    """
    paths = sorted(glob.glob(os.path.join(directory, "hessian_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"no hessian_*.npz in {directory}. Run sampling_chains/mixed_hessian.sh first, "
            f"or point --hessian_dir at the out_dir that script writes to.")

    config = None
    for path in paths:
        data = np.load(path, allow_pickle = True)
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

    return dict(config, n_realizations = len(paths), paths = paths)


def load_hessian_directory(directory, verbose = True, method = METHOD):
    """Average the per-realization Hessian npz files written by run_single_mixed_hessian.py.

    The parallel counterpart to computing the realizations in forecast_from_mixed_logpdf
    (which calls this when given hessian_dir): slurm jobs each write one
    hessian_<index>.npz, and this collects them, refusing mixed configurations and
    duplicate seeds. `method` is what the files must hold: METHOD for plain mixed Hessians,
    LOUIS_METHOD for Louis's observed information (whose `hessian` field is that
    information, so the averaging is identical; its two terms are also reported).

    Returns (fisher, names, per_realization, metadata) with metadata as
    hessian_directory_config gives it.
    """
    metadata = hessian_directory_config(directory)
    if metadata["method"] != method:
        raise ValueError(f"{directory} holds '{metadata['method']}' Hessians, not "
                         f"'{method}' ones")

    hessians, seeds, louis_terms = [], [], []
    for path in metadata["paths"]:
        data = np.load(path, allow_pickle = True)
        seeds.append(int(data["map_seed"]))
        if method == LOUIS_METHOD and "scores" in data.files:
            #rebuild the information HERE from the per-draw Hessians and scores rather than
            #trusting the number the job stored: the autocorrelation correction is measured
            #per realization from its own scores, so it is applied at merge time and a file
            #written before that correction existed is still handled correctly
            information, terms = louis_information_from_terms(data["hessians"],
                                                              data["scores"])
            hessians.append(information)
            louis_terms.append((terms["complete"], terms["missing"], data["hessian_truth"],
                                float(data["phi_acceptance"]), terms["score_iat"],
                                terms["stride"], terms["effective_draws"],
                                terms["n_draws"]))
        else:
            hessians.append(data["hessian"])

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                         f"realizations would be double counted. Each slurm job must use "
                         f"a distinct seed - check the map_prefix loop in mixed_hessian.sh.")

    per_realization = np.array(hessians)
    if verbose:
        print(f"Averaging {len(hessians)} cached realizations from {directory}")
        print(f"  method {metadata['method']}, parameters {metadata['names']}, nside "
              f"{metadata['nside']}, {metadata['theta_pix']:g}', "
              f"{metadata['noise_level']:g} uK-arcmin, l_knee {metadata['l_knee']:g}")
        spread = np.std(per_realization, axis = 0) / np.sqrt(len(hessians))
        mean = np.mean(per_realization, axis = 0)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            relative = np.abs(spread / mean)
        print(f"  standard error on the mean, relative: max {np.nanmax(relative):.3f}")
        if louis_terms:
            _report_louis_terms(louis_terms)

    return np.mean(per_realization, axis = 0), metadata["names"], per_realization, metadata


def _report_louis_terms(louis_terms):
    """The two Louis terms, and the check that the posterior draws are doing their job.

    E_d[E_{f°,phi°|d}(-H)] and E over prior draws of -H at the true fields are the same
    number by the tower property, so the complete term must agree with the truth Hessian
    within their errors; a real gap means the posterior chain is not sampling p(f°, phi° | d).

    The chain diagnostics are per realization by construction: each data realization mixes
    at its own rate, so the stride its own score chain earned, the IAT behind it and the
    number of pruned samples that leaves are all reported as a spread, not a single number.
    """
    (complete, missing, truth, acceptance, score_iat, stride, effective,
     swept) = (np.array(term) for term in zip(*louis_terms))
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
    print(f"  phi acceptance {np.mean(acceptance):.3f}; score IAT, mean over realizations "
          f"{np.array2string(np.mean(score_iat, axis = 0), precision = 2)}")
    print(f"  pruning stride measured per realization: {np.min(stride):.0f}-"
          f"{np.max(stride):.0f} (median {np.median(stride):.0f}), leaving "
          f"{np.min(effective):.0f}-{np.max(effective):.0f} samples of "
          f"{int(np.median(swept))} sweeps")
    #a covariance from a handful of samples is mostly noise, whatever the stride is
    if np.min(effective) < 10:
        print(f"  WARNING: the worst realization is pruned to {np.min(effective):.0f} "
              f"samples, so its missing term is badly determined; raise louis_draws for "
              f"this box")


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
    parser.add_argument("--louis_draws", type = int, default = DEFAULT_LOUIS_DRAWS)
    parser.add_argument("--louis_burn", type = int, default = DEFAULT_LOUIS_BURN)
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
        louis = args.louis, louis_draws = args.louis_draws, louis_burn = args.louis_burn)
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
