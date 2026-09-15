"""The realization-averaged Hessian of statistics.logpdf, "logpdf":

    F_ij = < -d2 log p(d, f, phi | theta) / dtheta_i dtheta_j >  over prior draws

Each realization is a fresh load_sim at the fiducial point, which draws f and phi
INDEPENDENTLY from their priors at that cosmology and lenses them into d. The Hessian of
the codebase's OWN statistics.logpdf - the same function the sampler evaluates - is then
finite-differenced in theta at that fixed (d, f, phi) and averaged over realizations.
Nothing about the log-density is assumed or re-derived: the data term, both quadratic
forms and all three log-determinants are evaluated by the real code at every stencil point.

WHAT THIS CONVERGES TO. Because the draws come from the PRIOR, this is the complete-data
information of p(d, f, phi | theta) - Louis's FIRST TERM ONLY:

    d2 log p(d|theta) = E[d2 log p(d,x|theta) | d] + Cov[d log p(d,x|theta) | d]

and this path computes only the first expectation. Averaging over more realizations drives
down the Monte Carlo error on that term; it does not recover the second, which is an
expectation-level difference rather than a sampling fluctuation. It is NOT the marginal
Fisher of p(d | theta) that sample_joint targets. Concretely: on prior draws the omitted
Cov term equals this one exactly, so the marginal information is ZERO while this returns
the full prior Fisher.

WHAT IT IS GOOD FOR. It assumes nothing. It differentiates statistics.logpdf itself, so it
independently checks the claim that log p(d | f, phi) is theta-independent - if any theta
dependence lurked in the data term, it would appear here and nowhere else. Note it will
NOT reproduce fisher_forecast's spectra = "ceiling": ceiling adds C_n to the field block
and N_phi to the lensing block, whereas the honest complete-data Hessian uses the bare
priors. The gap between them is exactly that heuristic.

COST. n_realizations * (1 + 2n + 2n(n-1)) logpdf calls, each running the lensing RK4 -
about 1900 lensing solves for the default 100 realizations and three parameters. `fast`
removes the stencil's lensing solves by evaluating only the theta-dependent prior terms
(_prior_log_density). The dropped terms - the data quadratic form and the noise
log-determinant - are constant across the stencil, so they cancel exactly in every central
difference; this is an exact simplification, not an approximation. Because that exactness
is a property of the MODEL rather than of this function, `verify_fast` (default True)
computes the first realization BOTH ways and raises if they disagree, so the assumption is
re-checked on every run instead of being trusted forever. Fast mode also passes ONE set of
spectra to every load_sim call via its `precomputed_cls` argument, since every realization
sits at the same cosmology and differs only by seed. MEASURED: 117.7 s -> 1.8 s for 6
realizations at nside 64, three parameters - 64x. Skipping the stencil's lensing alone was
only 4x; the repeated CAMB call inside load_sim was ~94% of what remained.

Usage:
    python -m cmb_lensing.fisher_forecast_from_logpdf --fast
    python -m cmb_lensing.fisher_forecast_from_logpdf --realizations 20 --nside 64

Writes fisher_matrix_from_logpdf.png, covariance_matrix_from_logpdf.png,
correlation_matrix_from_logpdf.png and fisher_from_logpdf.npz into
cmb_lensing/fisher_output/.
"""

import argparse

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid
from cmb_lensing.simulate import load_sim
from cmb_lensing.statistics import logpdf
from cmb_lensing.fields import dot
from cmb_lensing.matrix_operators import pinv, log_det
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH
from cmb_lensing.fisher_forecast import (camb_cls_at_params, load_sim_cosmology,
                                         prior_covariances, sampled_names_and_steps,
                                         stencil_offsets, hessian_from_stencil,
                                         covariance_from_fisher, report_sigmas,
                                         save_outputs, add_box_arguments,
                                         sampled_from_args, box_subtitle, run_config)


METHOD = "logpdf"
SUFFIX = "_from_logpdf"
LABEL = r"logpdf:  $F_{ij} = \langle -\partial_i \partial_j \log p(d, f, \phi\,|\,\theta) \rangle$"

#the realization methods draw from the prior and never consult a spectra mode; this is
#what their reports and figures are labelled with instead
SPECTRA = "prior draws"


def _prior_log_density(field, phi, cf_op, cphi_op):
    """The theta-DEPENDENT half of statistics.logpdf: the f and phi Gaussian priors.

    statistics.logpdf sums six terms under one -1/2 prefactor - a data quadratic form, two
    prior quadratic forms, and the noise / field / phi log-determinants. Only the four
    listed here move with theta. This reproduces them with the very same primitives logpdf
    uses (dot, pinv, log_det on the same operator objects), so the theta-dependent part is
    still evaluated by the codebase's own machinery; what is dropped is the data term and
    the noise log-determinant, which are constant in theta and therefore cancel identically
    in any central difference.

    Note the omitted data term is the one containing the lensing, which is why dropping it
    removes essentially all of the cost.
    """
    field_product = dot(field, pinv(cf_op) * field)
    phi_product = dot(phi, pinv(cphi_op) * phi)
    return -float(jnp.real(field_product + phi_product
                           + log_det(cf_op) + log_det(cphi_op)) / 2)


def logpdf_hessian_one_realization(data_set, names, param_ground, steps, stencil,
                                   nside, pix_width, ell_grid, fast = False):
    """-d2/dtheta_i dtheta_j log p(d, f, phi | theta) for ONE (data, f, phi) realization.

    Finite-differences statistics.logpdf with (d, f, phi) held fixed and theta moved only
    through C_f and C_phi. `fast = True` evaluates only the theta-dependent prior terms
    (_prior_log_density), skipping the lensing solve inside the data term; measured
    agreement with the full evaluation is 5e-11 relative, which is float64 round-off
    amplified by the second difference's division by h^2, not a real discrepancy.

    The stencil is fisher_forecast.stencil_offsets' 1 + 2n + 4*n*(n-1)/2 point set,
    assembled by hessian_from_stencil.
    """
    values = {}
    for offset in stencil:
        params = dict(param_ground)
        for i, name in enumerate(names):
            params[name] = param_ground[name] + offset[i] * steps[i]
        cf, cphi = prior_covariances(params, nside, pix_width, ell_grid)
        cf_op = data_set.field_covariance.replace(scalar_matrix = cf)
        cphi_op = data_set.phi_covariance.replace(scalar_matrix = cphi)
        if fast:
            values[offset] = _prior_log_density(data_set.unlensed_field, data_set.phi,
                                                cf_op, cphi_op)
        else:
            values[offset] = float(logpdf(
                data_set.unlensed_field, data_set.phi, data_set.data,
                data_set.noise_covariance, cphi_op, cf_op,
                data_set.mask, data_set.beam))

    return -hessian_from_stencil(values, steps)


def forecast_from_logpdf(nside, theta_pix, noise_level, is_sampled, param_ground,
                         n_realizations = 100, step_fracs = None, l_knee = 0,
                         beam_fwhm = 0, map_seed = 20260908, fast = False,
                         verify_fast = True, verbose = True):
    """< -d2 log p(d, f, phi | theta) > over prior realizations - see the module docstring.

    Args are fisher_forecast.forecast's minus `spectra` (a prior expectation has no
    spectra mode), plus `n_realizations`, `map_seed` (the first load_sim seed; realization
    k uses map_seed + k), `fast` and `verify_fast`. Returns (fisher, names,
    per_realization) with names in OUTPUT_PARAM_ORDER order.
    """
    names, steps, _ = sampled_names_and_steps(is_sampled, param_ground, step_fracs)

    #load_sim has no beam_fwhm argument - it always builds a zero-FWHM beam - so unlike
    #the stencil methods this one cannot honour a beam. It would not change the answer (the
    #beam sits only in the theta-independent data term, which cancels in the theta
    #Hessian), but silently ignoring the argument would be worse than refusing it
    if beam_fwhm:
        raise ValueError(f"forecast_from_logpdf cannot apply beam_fwhm = {beam_fwhm}: "
                         f"load_sim builds its own zero-FWHM beam and exposes no override. "
                         f"The beam lives entirely in the theta-independent data term, so "
                         f"it would not change this Fisher anyway.")

    stencil = stencil_offsets(len(names))
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    camb_kwargs = load_sim_cosmology(param_ground)

    if verbose:
        print(f"Fisher forecast [logpdf{', fast' if fast else ''}]: nside {nside}, "
              f"theta_pix {theta_pix}', {noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  sampled: {names}")
        print(f"  {n_realizations} realizations x {len(stencil)} evaluations = "
              f"{n_realizations * len(stencil)} "
              f"{'prior-term evaluations (no lensing)' if fast else 'lensing solves'}")

    #in fast mode reuse ONE set of spectra across every realization. They are already in
    #the CAMB memo from the stencil's centre point, and every realization sits at the same
    #cosmology (only the seed changes), so load_sim's own CAMB call is pure repetition -
    #it was the floor that kept the measured speedup at ~4x rather than the stencil ratio.
    #Left off the slow path so that stays a genuinely independent evaluation.
    shared_cls = camb_cls_at_params(param_ground) if fast else None

    per_realization = []
    for index in range(n_realizations):
        data_set = load_sim(nside, theta_pix, "I", map_seed + index, **camb_kwargs,
                            uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                            precomputed_cls = shared_cls)
        hessian = logpdf_hessian_one_realization(
            data_set, names, param_ground, steps, stencil, nside, pix_width, ell_grid,
            fast = fast)

        #re-check the exact-cancellation argument on the first realization rather than
        #trusting it: the full path evaluates the data term (and its lensing) at every
        #stencil point, so any theta dependence hiding there shows up as a mismatch
        if fast and verify_fast and index == 0:
            reference = logpdf_hessian_one_realization(
                data_set, names, param_ground, steps, stencil, nside, pix_width, ell_grid,
                fast = False)
            scale = np.sqrt(np.outer(np.abs(np.diag(reference)), np.abs(np.diag(reference))))
            error = np.max(np.abs(hessian - reference) / scale)
            if verbose:
                print(f"    fast-mode check on realization 0: max relative difference "
                      f"{error:.2e}")
            if error > 1e-6:
                raise RuntimeError(
                    f"fast mode disagrees with the full logpdf by {error:.2e} (relative) "
                    f"on the first realization. The two can only differ if "
                    f"log p(d | f, phi) has acquired theta dependence, which would break "
                    f"the cancellation fast mode relies on. Re-run with fast = False.")

        per_realization.append(hessian)
        if verbose and (index + 1) % 10 == 0:
            running = np.mean(per_realization, axis = 0)
            print(f"    {index + 1}/{n_realizations}  running diagonal "
                  f"{np.array2string(np.diag(running), precision = 4)}")

    per_realization = np.array(per_realization)
    return np.mean(per_realization, axis = 0), names, per_realization


def main():
    parser = argparse.ArgumentParser(
        description = "Fisher forecast from the realization-averaged Hessian of "
                      "statistics.logpdf over prior draws of (d, f, phi)")
    add_box_arguments(parser)
    parser.add_argument("--realizations", type = int, default = 100,
                        help = "(data, f, phi) draws to average over")
    parser.add_argument("--fast", action = "store_true",
                        help = "evaluate only the theta-dependent prior terms, skipping "
                               "the lensing solve in the theta-INDEPENDENT data term. "
                               "Exact (those terms cancel in every central difference), "
                               "and it reuses one set of CAMB spectra across realizations: "
                               "64x faster measured, self-verified against the full logpdf "
                               "on the first realization")
    args = parser.parse_args()
    is_sampled = sampled_from_args(parser, args)

    fisher, names, _ = forecast_from_logpdf(
        args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
        n_realizations = args.realizations, l_knee = args.l_knee,
        beam_fwhm = args.beam_fwhm, fast = args.fast)
    covariance = covariance_from_fisher(fisher, names)
    report_sigmas(covariance, names, SPECTRA, METHOD)

    #the accuracy here is set by the realization count, so it goes on the figures
    subtitle = f"{box_subtitle(SPECTRA, args)}  |  {args.realizations} realizations"
    save_outputs(fisher, covariance, names, METHOD, SUFFIX, LABEL, subtitle,
                 run_config(SPECTRA, args, names, n_realizations = args.realizations,
                            fast_logpdf = args.fast))


if __name__ == "__main__":
    main()
