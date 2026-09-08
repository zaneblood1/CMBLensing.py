"""The MARGINAL Fisher information of p(d | theta), via Louis's identity.

fisher_forecast.py computes two Gaussian forecasts and its own docstring says neither is
the object the sampler targets: "lensed" is a two-point analysis of a surrogate Gaussian
model, "ceiling" is the complete-data information given (f, phi). Measured against the
Gibbs chains they bracket the truth rather than reproduce it - at nside 128 / 2.5' /
5 uK-arcmin the measured posterior sits between them in all three sigmas AND all three
correlations, and for omch2 - theta_MC_100 the bracket straddles zero, so the two bounds
disagree on the SIGN of a real degeneracy. This module computes the marginal information

    I(theta) = -d2/dtheta2 log p(d | theta),     p(d | theta) = INT p(d, f, phi | theta)

which is what a converged theta chain actually measures.

THE IDENTITY. The integral has no closed form, but Louis (1982) gives its second
derivative in terms of expectations over the conditional p(f, phi | d, theta) - exactly
the Gibbs conditional sample_lcdm.py already draws from:

    I_obs(theta; d) = E[ -d2 log p(d, x | theta) | d ]  -  Cov[ d log p(d, x | theta) | d ]

with x = (f, phi). Two properties of this codebase make it cheap:

 1. log p(d | f, phi) is EXACTLY theta-independent. statistics.py:logpdf forms the data
    term from mask, beam and lense_flow only, and noise_cls / mask / beam carry no
    cosmology. So the complete-data log-density splits as

        log p(d, f, phi | theta) = [theta-independent] + log p(f | theta) + log p(phi | theta)

    and BOTH the score and the Hessian involve only the two Gaussian priors - no lensing,
    no adjoint ODE, no mixing matrices.
 2. Those priors are diagonal in the Fourier basis with a known normalization. Reading it
    off statistics.py:logpdf + util.primal_dot + util.primal_log_det:

        log p(x | theta) = -1/2 sum_k w_k [ |x_k|^2 / (nside^2 C_k)  +  ln C_k ]  + const

    with w_k = get_fourier_weights, the same real-DOF weights fisher_forecast uses.

Writing u_k = |x_k|^2 / (nside^2 C_k) - which has mean 1 under the PRIOR and is Wiener
shrunk below 1 under the posterior - the theta-derivatives are analytic in the fields, and
only the Cl derivatives need finite differences:

    s_i        = 1/2 sum_k w_k  dlnC_k/di  (u_k - 1)
    d_i d_j l  = 1/2 sum_k w_k [ d2lnC_k/didj (u_k - 1)  -  dlnC_k/di dlnC_k/dj  u_k ]

so, summing both blocks (f carries C_f = unlensed scalar TT, phi carries C_phi):

    I_obs_ij = 1/2 sum_k w_k [ dlnC/di dlnC/dj <u_k> - d2lnC/didj (<u_k> - 1) ] - Cov(s_i, s_j)

Per Gibbs draw we therefore store only the n-vector s and accumulate the running mean of
u_k. No field history is written to disk.

NOTE this corrects the "ceiling" construction. The true first term uses the BARE priors
C_f and C_phi; the C_n and N_phi that fisher_forecast.covariance_blocks adds to them are a
heuristic ("the fields are known only to within the noise"). The noise properly enters
only through WHICH DRAWS are fed in - the conditional p(f, phi | d, theta) is narrower
than the prior precisely because the data constrains the fields.

A SECOND ESTIMATOR, free from the same per-draw scores. Fisher's identity says the
marginal score is the conditional mean of the complete-data score, g(d) = E[s | d], so the
EXPECTED information is I = Cov_d[g(d)] over data realizations. That form needs no
subtraction, is positive semi-definite by construction, and needs only FIRST derivatives.
Its one flaw - Monte Carlo noise in g(d) inflates the covariance - is removed exactly by
splitting each map's draws into two halves and taking the CROSS-covariance Cov_d[g_A, g_B],
since the two halves' MC errors are independent. It needs many maps where Louis needs many
draws per map, so the two estimators degrade in different directions and cross-validate.
Both are reported.

TWO THINGS THIS MODULE DELIBERATELY DOES NOT DO.

  * It works in UNMIXED (f, phi) only. D and G both depend on theta (simulate.py:
    get_d_tt_matrix, get_g_matrix_lcdm), so the mixed change of variables carries a
    theta-dependent Jacobian - mixed_logpdf subtracts logdet(G) + logdet(D) for exactly
    this reason. Differentiating in mixed coordinates would have to carry that too, and it
    is the same trap CLAUDE.md's "never re-mix pinned f/phi with theta-dependent D/G" note
    describes. The unmixed parametrization is theta-independent and the Jacobian is absent.
  * It never calls jax.grad with respect to theta. CLAUDE.md records that path as ~3000x
    wrong with a flipped sign through G. There is a second, independent reason:
    statistics.py:phi_dot_wrapper's custom VJP differentiates dot(phi, Cphi) rather than
    dot(phi, Cphi^-1 phi), so ANY autodiff through phi_covariance is silently wrong. The
    derivatives here are analytic in the fields and finite-difference in the Cls, so they
    touch neither.

Cls come from DIRECT CAMB (fisher_forecast.camb_cls_at_params), never the 5D grid, for the
reason fisher_forecast's docstring gives: a deterministic ~1e-3 lnCl interpolation error
does not average away under differencing. A useful side effect is that nothing here loads
the ~8 GB grid file.

VALIDATION. Feed the estimator draws from the PRIOR instead of the posterior and Louis
must return exactly ZERO up to Monte Carlo noise: if the conditional equals the prior then
d carries no information about theta. Algebraically both terms collapse to the same
1/2 sum_k w_k dlnC/di dlnC/dj - the first because E[u] = 1, the second because
Var(u_k) = 2/w_k - so the difference vanishes identically. That single test pins the
weights, the nside^2 normalization, the factor of 1/2, the finite-difference stencil and
every sign at once. It is the analogue of fisher_forecast's F_logA,logA = (nside^2 - 1)/2
check, and it runs in seconds with no sampler and no grid. `--validate` runs it.

Temperature only (pol = "I"), matching every block in fisher_forecast.

Usage:
    python -m cmb_lensing.marginal_fisher --validate --nside 64
    python -m cmb_lensing.marginal_fisher --nside 64 --theta_pix 5 --noise 5 \
        --n_maps 2 --n_draws 200 --burn_in 100 --params omch2 theta_MC_100 logA

Writes fisher_matrix_marginal.png, covariance_matrix_marginal.png,
correlation_matrix_marginal.png and fisher_marginal.npz to cmb_lensing/fisher_output/.
"""

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.simulate import (load_sim, covar_matrix_from_cls,
                                  field_from_covar_single_key,
                                  get_d_tt_matrix, get_g_matrix_lcdm)
from cmb_lensing.mixing import mix, unmix
from cmb_lensing.sample_lcdm import gibbs_sample_f, gibbs_sample_phi
from cmb_lensing.constants import DEFAULT_MAX_ELL
from cmb_lensing.precompute_camb_1d import (PARAM_ORDER, GROUND_TRUTH, PARAM_SIGMA,
                                            CAMB_LMAX)
from cmb_lensing.fisher_forecast import (camb_cls_at_params, qe_noise_matrix,
                                         _instrument_matrices, _stencil_offsets,
                                         FD_STEP_FRAC, OUTPUT_PARAM_ORDER,
                                         covariance_from_fisher, write_outputs,
                                         output_dir)

#the two theta-dependent Gaussian priors whose information Louis's identity counts. the
#keys name the CAMB spectrum each block's covariance is built from - "scalar_TT" is the
#UNLENSED field prior (the joint is over the unlensed f, with d = M B L(phi) f + n), and
#"phi" is the lensing potential prior. neither carries a noise term: C_n and N_phi enter
#only through which draws are fed in
PRIOR_BLOCKS = {"f": "scalar_TT", "phi": "phi"}


# ── Cl derivatives on the flat-sky grid ───────────────────────────────────

def _log_covariance_at(params, nside, pix_width, ell_grid):
    """ln C on the rfft grid for each prior block, plus the block's positivity mask.

    Returns ({block: lnC array}, {block: good mask}). The [0, 0] origin is zeroed by
    origin_value = 0 and carries no information, so it is masked out rather than allowed
    to produce log(0) = -inf.
    """
    cls = camb_cls_at_params(params)
    log_covariance, good = {}, {}
    for block, spectrum in PRIOR_BLOCKS.items():
        ells = jnp.arange(2, 2 + cls[spectrum].shape[0]).astype(jnp.float64)
        covariance = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                           cls[spectrum], origin_value = 0)
        positive = covariance > 0
        good[block] = positive
        log_covariance[block] = jnp.where(positive, jnp.log(jnp.where(positive, covariance, 1.0)), 0.0)
    return log_covariance, good


def log_cl_derivatives(names, param_ground, nside, theta_pix, step_fracs = None,
                       verbose = True):
    """First and second theta-derivatives of ln C for both prior blocks.

    Central finite differences on DIRECT CAMB runs, on the same rfft grid the sampler
    works on. Returns (derivatives, steps, fiducial) where

        derivatives[block]["dlog"][i]      = d ln C / d theta_i
        derivatives[block]["d2log"][i][j]  = d2 ln C / d theta_i d theta_j
        fiducial[block]                    = C at param_ground (for the u_k denominators)
        fiducial_good[block]               = the positivity mask

    Only ln C is differenced, never C, so the result is invariant to the theta-independent
    1/pix_width**2 rescale covar_matrix_from_cls applies.
    """
    n_param = len(names)
    fracs = dict(FD_STEP_FRAC)
    if step_fracs is not None:
        fracs.update(step_fracs)
    steps = [fracs[name] * PARAM_SIGMA[name] for name in names]

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)

    offsets = _stencil_offsets(n_param)
    if verbose:
        print(f"  {len(offsets)} CAMB stencil points for {n_param} parameters")

    log_covariance = {}
    good_all = None
    for offset in offsets:
        params = dict(param_ground)
        for i, name in enumerate(names):
            params[name] = param_ground[name] + offset[i] * steps[i]
        log_covariance[offset], good = _log_covariance_at(params, nside, pix_width, ell_grid)
        #a mode is usable only where every stencil point has positive power
        good_all = good if good_all is None else {b: good_all[b] & good[b] for b in good}

    centre = tuple([0] * n_param)

    def unit(i, sign):
        offset = [0] * n_param
        offset[i] = sign
        return tuple(offset)

    def corner(i, j, sign_i, sign_j):
        offset = [0] * n_param
        offset[i] = sign_i
        offset[j] = sign_j
        return tuple(offset)

    derivatives = {}
    for block in PRIOR_BLOCKS:
        mask = good_all[block]
        dlog = []
        for i in range(n_param):
            value = (log_covariance[unit(i, +1)][block] - log_covariance[unit(i, -1)][block]) / (2 * steps[i])
            dlog.append(jnp.where(mask, value, 0.0))

        d2log = [[None] * n_param for _ in range(n_param)]
        for i in range(n_param):
            value = (log_covariance[unit(i, +1)][block] - 2 * log_covariance[centre][block]
                     + log_covariance[unit(i, -1)][block]) / steps[i]**2
            d2log[i][i] = jnp.where(mask, value, 0.0)
        for i in range(n_param):
            for j in range(i + 1, n_param):
                value = (log_covariance[corner(i, j, +1, +1)][block]
                         - log_covariance[corner(i, j, +1, -1)][block]
                         - log_covariance[corner(i, j, -1, +1)][block]
                         + log_covariance[corner(i, j, -1, -1)][block]) / (4 * steps[i] * steps[j])
                d2log[i][j] = jnp.where(mask, value, 0.0)
                d2log[j][i] = d2log[i][j]

        derivatives[block] = {"dlog": dlog, "d2log": d2log}

    #the fiducial covariance itself, for the u_k = |x_k|^2 / (nside^2 C_k) denominators
    cls = camb_cls_at_params(param_ground)
    fiducial, fiducial_good = {}, {}
    for block, spectrum in PRIOR_BLOCKS.items():
        ells = jnp.arange(2, 2 + cls[spectrum].shape[0]).astype(jnp.float64)
        covariance = covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                           cls[spectrum], origin_value = 0)
        fiducial[block] = covariance
        fiducial_good[block] = good_all[block]

    return derivatives, steps, fiducial, fiducial_good


# ── Per-draw score accumulation ───────────────────────────────────────────

class ScoreAccumulator:
    """Accumulates the complete-data score and mean normalized power over field draws.

    add() takes one draw as {block: rfft2 array}. It stores the n-vector

        s_i = 1/2 sum_blocks sum_k w_k dlnC_k/di (u_k - 1),   u_k = |x_k|^2 / (nside^2 C_k)

    and folds u_k into a running mean. Memory is O(n_draws * n_param) plus one array per
    block - the fields themselves are never retained.
    """

    def __init__(self, derivatives, fiducial, good, weights, nside, n_param):
        self.derivatives = derivatives
        self.fiducial = fiducial
        self.good = good
        self.weights = weights
        self.nside = nside
        self.n_param = n_param
        self.scores = []
        self._u_sum = {block: jnp.zeros_like(fiducial[block]) for block in fiducial}
        self._n = 0

    def add(self, draw):
        score = np.zeros(self.n_param)
        for block, matrix in draw.items():
            good = self.good[block]
            safe = jnp.where(good, self.fiducial[block], 1.0)
            power = jnp.real(matrix * jnp.conj(matrix))
            u = jnp.where(good, power / (self.nside**2 * safe), 0.0)
            self._u_sum[block] = self._u_sum[block] + u

            residual = jnp.where(good, u - 1.0, 0.0)
            for i in range(self.n_param):
                score[i] += 0.5 * float(jnp.sum(self.weights * self.derivatives[block]["dlog"][i]
                                                * residual))
        self.scores.append(score)
        self._n += 1

    @property
    def mean_u(self):
        return {block: total / self._n for block, total in self._u_sum.items()}

    @property
    def score_array(self):
        return np.array(self.scores)


def louis_information(accumulator):
    """I_obs = -E[Hessian] - Cov[score], both expectations over the accumulated draws.

    Returns (information, expected_hessian_term, score_covariance) so the caller can
    report how severe the subtraction was.
    """
    n_param = accumulator.n_param
    weights = accumulator.weights
    mean_u = accumulator.mean_u

    hessian_term = np.zeros((n_param, n_param))
    for block in accumulator.derivatives:
        good = accumulator.good[block]
        u_mean = mean_u[block]
        excess = jnp.where(good, u_mean - 1.0, 0.0)
        dlog = accumulator.derivatives[block]["dlog"]
        d2log = accumulator.derivatives[block]["d2log"]
        for i in range(n_param):
            for j in range(i, n_param):
                value = 0.5 * float(jnp.sum(weights * (dlog[i] * dlog[j] * u_mean
                                                       - d2log[i][j] * excess)))
                hessian_term[i, j] += value
                if i != j:
                    hessian_term[j, i] += value

    scores = accumulator.score_array
    score_covariance = np.cov(scores, rowvar = False, ddof = 1)
    score_covariance = np.atleast_2d(score_covariance)

    return hessian_term - score_covariance, hessian_term, score_covariance


def prior_information(derivatives, weights, n_param):
    """1/2 sum_k w_k dlnC/di dlnC/dj summed over blocks - the bare-prior Gaussian Fisher.

    This is what BOTH Louis terms collapse to under prior draws, so it is the reference
    the zero test measures against, and it is exactly fisher_forecast._fisher_from_blocks
    evaluated on C_f and C_phi with no noise added.
    """
    fisher = np.zeros((n_param, n_param))
    for block in derivatives:
        dlog = derivatives[block]["dlog"]
        for i in range(n_param):
            for j in range(i, n_param):
                value = 0.5 * float(jnp.sum(weights * dlog[i] * dlog[j]))
                fisher[i, j] += value
                if i != j:
                    fisher[j, i] += value
    return fisher


# ── Fixed-theta field sampling ────────────────────────────────────────────

def build_fixed_theta_args(data_set, params, noise_level, l_knee = 0, beam_fwhm = 0,
                           l_cutoff = 10_000, lmax = DEFAULT_MAX_ELL):
    """The sampler's `args` dict at a fixed cosmology, built from direct CAMB.

    Mirrors sample_lcdm.add_starting_matrices_to_args for pol = "I", but takes the Cls
    from camb_cls_at_params instead of the 5D grid predictors, so nothing here loads the
    ~8 GB grid file. Because theta never moves, D, G, C_f, C_phi and the QE norm are all
    constants for the whole run and the jitted mix / unmix / gibbs kernels compile once.
    """
    args = {}
    args["lmax"] = lmax
    args["lmax_prime"] = min(lmax, DEFAULT_MAX_ELL)
    args["nside"] = data_set.nside
    args["pix_width"] = data_set.pix_width
    ell_grid, pix_width = gen_ell_grid(data_set.nside, data_set.theta_pix)
    args["ell_grid"] = ell_grid
    args["ells"] = jnp.arange(2, lmax + 1).astype(jnp.float64)

    #the instrument is theta-independent, so it comes straight from the data set
    args["noise_covariance"] = data_set.noise_covariance
    args["mask"] = data_set.mask
    args["beam"] = data_set.beam
    args["cphi_fid"] = data_set.phi_covariance.scalar_matrix

    cls = camb_cls_at_params(params)
    ells = jnp.arange(2, 2 + cls["total_TT"].shape[0]).astype(jnp.float64)
    phi_ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)

    def covar(cl, ell_axis):
        return covar_matrix_from_cls(data_set.nside, pix_width, ell_grid, ell_axis, cl,
                                     origin_value = 0)

    cf = covar(cls["scalar_TT"], ells)
    cphi = covar(cls["phi"], phi_ells)

    args["field_covariance"] = data_set.field_covariance.replace(scalar_matrix = cf)
    args["phi_covariance"] = data_set.phi_covariance.replace(scalar_matrix = cphi)

    #the QE norm the G matrix and the phi mass matrix are built from, at this cosmology
    qe_matrix = qe_noise_matrix(cls, data_set.nside, pix_width, ell_grid, noise_level,
                                l_knee, beam_fwhm, l_cutoff)
    args["quadratic_estimate"] = data_set.quadratic_estimate.replace(scalar_matrix = qe_matrix)

    args["mixing_g"] = data_set.mixing_g.replace(
        scalar_matrix = get_g_matrix_lcdm(cphi, qe_matrix))
    args["mixing_d"] = data_set.mixing_d.replace(
        scalar_matrix = get_d_tt_matrix(cf, args["noise_covariance"].scalar_matrix))

    return args


def draw_fields(data_set, args, accumulator, n_draws, burn_in, seed, verbose = True):
    """Gibbs-sample (f, phi) at FIXED theta, feeding each draw to the accumulator.

    One sweep is the sampler's steps 1-3 and 6 with the theta step removed entirely: draw
    f from its conditional, mix, take one HMC step in phi, unmix. The pair captured is the
    END-OF-SWEEP joint state - unmix's own first return value, which sample_joint discards
    into `_`. Both come back in the UNMIXED parametrization, which is the theta-independent
    one Louis's identity needs.

    Returns the phi HMC acceptance rate.
    """
    sub_key = jax.random.PRNGKey(seed)
    data_field = data_set.data
    field_zeroes = 0 * data_set.unlensed_field
    phi = 0 * data_set.phi
    accepts = []

    for iteration in range(1, burn_in + n_draws + 1):
        rng_key, sub_key = jax.random.split(sub_key)
        field = gibbs_sample_f(field_zeroes, data_field, phi, args, rng_key)

        mixed_field, mixed_phi = mix(field, phi, args["mixing_d"], args["mixing_g"])

        rng_key, sub_key = jax.random.split(sub_key)
        mixed_phi, _, accept = gibbs_sample_phi(mixed_phi, mixed_field, data_field,
                                                rng_key, args, iteration, 0)
        accepts.append(int(accept))

        field, phi = unmix(mixed_field, mixed_phi, args["mixing_d"], args["mixing_g"])

        if iteration > burn_in:
            accumulator.add({"f": field.scalar_matrix, "phi": phi.scalar_matrix})

        if verbose and iteration % 100 == 0:
            print(f"    sweep {iteration}/{burn_in + n_draws}  "
                  f"phi acceptance {np.mean(accepts):.2f}")

    return float(np.mean(accepts))


# ── Estimators ────────────────────────────────────────────────────────────

def integrated_autocorrelation(series):
    """Integrated autocorrelation time by Sokal's automatic windowing.

    Reported so the effective number of draws behind Cov(score) is visible - the Louis
    subtraction is only as good as N_eff, not N.
    """
    series = np.asarray(series, dtype = float)
    series = series - series.mean()
    n = len(series)
    if n < 4 or np.allclose(series, 0):
        return 1.0
    padded = np.zeros(2 * n)
    padded[:n] = series
    spectrum = np.fft.rfft(padded)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:n].real
    if correlation[0] <= 0:
        return 1.0
    correlation /= correlation[0]
    tau = 1.0
    for window in range(1, n):
        tau += 2 * correlation[window]
        #Sokal's window: stop once the window is 6 correlation times wide
        if window >= 6 * tau:
            break
    return max(float(tau), 1.0)


def marginal_fisher_one_map(nside, theta_pix, noise_level, names, param_ground,
                            derivatives, fiducial, good, weights, map_seed,
                            n_draws, burn_in, l_knee = 0, verbose = True):
    """Louis's observed information for ONE data realization.

    Simulates a map at param_ground, Gibbs-samples the fields at that same theta, and
    contracts the accumulated scores. Returns a dict with the information, the two terms
    it was subtracted from, the raw per-draw scores (needed by the score-covariance
    estimator) and the run diagnostics.
    """
    data_set = load_sim(nside, theta_pix, "I", map_seed,
                        **_camb_kwargs(param_ground),
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee)

    args = build_fixed_theta_args(data_set, param_ground, noise_level, l_knee = l_knee)

    accumulator = ScoreAccumulator(derivatives, fiducial, good, weights, nside, len(names))
    acceptance = draw_fields(data_set, args, accumulator, n_draws, burn_in, map_seed,
                             verbose = verbose)

    information, hessian_term, score_covariance = louis_information(accumulator)
    scores = accumulator.score_array
    tau = [integrated_autocorrelation(scores[:, i]) for i in range(len(names))]

    return {"information": information, "hessian_term": hessian_term,
            "score_covariance": score_covariance, "scores": scores,
            "phi_acceptance": acceptance, "autocorrelation": np.array(tau),
            "n_draws": n_draws}


def score_covariance_information(per_map_scores):
    """I = Cov_d[ E[s | d] ], debiased by the half-split cross-covariance.

    Fisher's identity makes the conditional mean score E[s | d] the score of the MARGINAL
    likelihood, so its covariance across data realizations IS the expected information -
    with no subtraction of two large terms. Averaging a finite number of draws leaves
    Monte Carlo noise in each g(d) which would inflate a plain Cov_d; splitting every
    map's draws into two halves and taking the cross-covariance removes that bias exactly,
    because the two halves' errors are independent.

    Needs at least n_param + 1 maps to be non-singular. Returns (information, n_maps).
    """
    first_half, second_half = [], []
    for scores in per_map_scores:
        midpoint = len(scores) // 2
        first_half.append(scores[:midpoint].mean(axis = 0))
        second_half.append(scores[midpoint:].mean(axis = 0))
    first_half = np.array(first_half)
    second_half = np.array(second_half)

    n_maps = len(first_half)
    centred_a = first_half - first_half.mean(axis = 0)
    centred_b = second_half - second_half.mean(axis = 0)
    cross = centred_a.T @ centred_b / (n_maps - 1)
    #symmetrize: Cov[g_A, g_B] and its transpose are two estimates of the same matrix
    return 0.5 * (cross + cross.T), n_maps


def _camb_kwargs(params):
    """param dict -> load_sim's CAMB keyword names, without mutating the caller's dict.

    sample_lcdm.to_camb_naming_conv does this in place (it deletes the old keys); this
    module never mutates a caller's parameter dict.
    """
    converted = dict(params)
    converted["cosmomc_theta"] = converted.pop("theta_MC_100") / 100
    converted["As"] = float(np.exp(converted.pop("logA")) * 1e-10)
    return converted


# ── Validation ────────────────────────────────────────────────────────────

def validate_zero_on_prior_draws(nside, theta_pix, names, param_ground, n_draws = 400,
                                 seed = 0, step_fracs = None):
    """THE test: on PRIOR draws Louis's identity must return exactly zero.

    If the conditional p(f, phi | d, theta) were the prior, the data would carry no
    information about theta and I_marg would vanish. Algebraically both terms collapse to
    prior_information(): the Hessian term because E[u] = 1 kills the d2lnC piece and
    leaves 1/2 sum w dlnC dlnC, the score term because u_k has variance 2/w_k so
    Cov(s)_ij = 1/4 sum w^2 dlnC dlnC (2/w) = the same thing. Their difference is
    identically zero, so any error in the weights, the nside^2 normalization, the factor
    of 1/2, the stencil or a sign shows up here as a non-zero residual.

    Needs no sampler, no data map and no 5D grid. Returns a dict of the pieces.
    """
    n_param = len(names)
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))
    derivatives, steps, fiducial, good = log_cl_derivatives(
        names, param_ground, nside, theta_pix, step_fracs = step_fracs)

    accumulator = ScoreAccumulator(derivatives, fiducial, good, weights, nside, n_param)
    key = jax.random.PRNGKey(seed)
    for _ in range(n_draws):
        draw = {}
        for block in PRIOR_BLOCKS:
            key, sub_key = jax.random.split(key)
            field_map = field_from_covar_single_key(nside, fiducial[block], sub_key)
            draw[block] = jfft.rfft2(field_map)
        accumulator.add(draw)

    information, hessian_term, score_covariance = louis_information(accumulator)
    reference = prior_information(derivatives, weights, n_param)

    return {"information": information, "hessian_term": hessian_term,
            "score_covariance": score_covariance, "prior_fisher": reference,
            "n_draws": n_draws}


def _report_validation(result, names):
    reference = result["prior_fisher"]
    scale = np.sqrt(np.outer(np.abs(np.diag(reference)), np.abs(np.diag(reference))))
    residual = np.abs(result["information"]) / scale
    hessian_error = np.abs(result["hessian_term"] - reference) / scale
    score_error = np.abs(result["score_covariance"] - reference) / scale
    expected = np.sqrt(2.0 / result["n_draws"])

    print(f"\nZERO TEST on {result['n_draws']} prior draws "
          f"(both Louis terms must equal the bare-prior Fisher, so I must vanish)")
    print(f"  bare-prior Fisher diagonal      {np.array2string(np.diag(reference), precision = 4)}")
    print(f"  |E[-H] - prior| / scale    max  {hessian_error.max():.4f}")
    print(f"  |Cov(s) - prior| / scale   max  {score_error.max():.4f}")
    print(f"  |I| / scale                max  {residual.max():.4f}   "
          f"(MC floor ~ sqrt(2/N) = {expected:.4f})")
    verdict = "PASS" if residual.max() < 8 * expected else "FAIL"
    print(f"  -> {verdict}")
    return verdict == "PASS"


# ── CLI ───────────────────────────────────────────────────────────────────

def marginal_covariance(information, names, n_draws = None, autocorrelation = None):
    """covariance_from_fisher, but with a diagnostic aimed at THIS estimator's failure mode.

    fisher_forecast's message blames the finite-difference step, which is right for a
    trace-formula Fisher built from a smooth stencil. Here a non-PSD result almost always
    means Cov(score) was estimated from too few effectively-independent draws and has
    overshot E[-Hessian]: the subtraction is exact in expectation but noisy at finite N,
    and the noise does not respect positive definiteness.
    """
    try:
        return covariance_from_fisher(information, names)
    except RuntimeError as error:
        hint = ""
        if n_draws is not None:
            effective = n_draws
            if autocorrelation is not None:
                effective = n_draws / float(np.max(autocorrelation))
            hint = (f" Each map kept {n_draws} draws (N_eff ~ {effective:.0f}), so the "
                    f"relative error on Cov(score) is ~{np.sqrt(2 / max(effective, 1)):.0%}.")
        raise RuntimeError(
            f"the marginal Fisher is not positive definite. This is a Monte Carlo "
            f"failure, not a stencil failure: Cov(score) is estimated from the Gibbs "
            f"draws and at small N it can overshoot E[-Hessian]. Raise --n_draws (and "
            f"--n_maps), not FD_STEP_FRAC.{hint} Underlying: {error}") from error


def _report(information, names, label, extra = "", n_draws = None, autocorrelation = None):
    covariance = marginal_covariance(information, names, n_draws, autocorrelation)
    sigmas = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sigmas, sigmas)
    print(f"\nMarginal forecast [{label}]{extra}")
    print(f"  {'parameter':<14s} {'sigma':>12s} {'sigma/PARAM_SIGMA':>20s}")
    for i, name in enumerate(names):
        print(f"  {name:<14s} {sigmas[i]:>12.4g} {sigmas[i] / PARAM_SIGMA[name]:>20.3g}")
    if len(names) > 1:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"  r({names[i]}, {names[j]}) = {correlation[i, j]:+.3f}")
    return covariance


def main():
    parser = argparse.ArgumentParser(
        description = "Marginal Fisher information of p(d | theta) via Louis's identity")
    parser.add_argument("--nside", type = int, default = 64)
    parser.add_argument("--theta_pix", type = float, default = 5.0,
                        help = "pixel width in arcmin")
    parser.add_argument("--noise", type = float, default = 5.0,
                        help = "white noise level in uK-arcmin")
    parser.add_argument("--l_knee", type = float, default = 0.0)
    parser.add_argument("--params", nargs = "*", default = None,
                        help = f"subset to forecast; default all of {PARAM_ORDER}")
    parser.add_argument("--n_maps", type = int, default = 4,
                        help = "data realizations; Louis averages over them and the "
                               "score-covariance estimator needs n_param + 1 at minimum")
    parser.add_argument("--n_draws", type = int, default = 400,
                        help = "retained Gibbs draws of (f, phi) per map")
    parser.add_argument("--burn_in", type = int, default = 100,
                        help = "discarded sweeps before accumulation starts")
    parser.add_argument("--map_seed", type = int, default = 1234)
    parser.add_argument("--validate", action = "store_true",
                        help = "run the prior-draw zero test and exit")
    parser.add_argument("--no_write", action = "store_true",
                        help = "skip the figures and npz")
    args = parser.parse_args()

    names = [name for name in OUTPUT_PARAM_ORDER
             if args.params is None or name in args.params]
    if args.params is not None:
        unknown = [name for name in args.params if name not in PARAM_ORDER]
        if unknown:
            parser.error(f"unknown parameters {unknown}; choose from {PARAM_ORDER}")
    if not names:
        parser.error("no parameters selected")

    if args.validate:
        result = validate_zero_on_prior_draws(args.nside, args.theta_pix, names,
                                              GROUND_TRUTH, n_draws = 400)
        ok = _report_validation(result, names)
        raise SystemExit(0 if ok else 1)

    print(f"Marginal Fisher [Louis]: nside {args.nside}, theta_pix {args.theta_pix}', "
          f"{args.noise} uK-arcmin, l_knee {args.l_knee}")
    print(f"  parameters: {names}")
    print(f"  {args.n_maps} maps x {args.n_draws} draws (+{args.burn_in} burn-in)")

    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((args.nside, args.nside // 2 + 1))),
                               (args.nside, args.nside // 2 + 1))
    derivatives, steps, fiducial, good = log_cl_derivatives(
        names, GROUND_TRUTH, args.nside, args.theta_pix)

    per_map, per_map_scores = [], []
    for index in range(args.n_maps):
        print(f"\n  map {index + 1}/{args.n_maps}")
        result = marginal_fisher_one_map(
            args.nside, args.theta_pix, args.noise, names, GROUND_TRUTH,
            derivatives, fiducial, good, weights, args.map_seed + index,
            args.n_draws, args.burn_in, l_knee = args.l_knee)
        per_map.append(result)
        per_map_scores.append(result["scores"])
        print(f"    phi acceptance {result['phi_acceptance']:.2f}, "
              f"autocorrelation {np.array2string(result['autocorrelation'], precision = 1)}, "
              f"N_eff {np.array2string(result['n_draws'] / result['autocorrelation'], precision = 0)}")

    louis = np.mean([result["information"] for result in per_map], axis = 0)
    hessian_term = np.mean([result["hessian_term"] for result in per_map], axis = 0)
    score_term = np.mean([result["score_covariance"] for result in per_map], axis = 0)

    survived = np.diag(louis) / np.diag(hessian_term)
    print(f"\n  Louis subtraction: {np.array2string(survived * 100, precision = 0)}% of "
          f"E[-H] survives Cov(s) (a small number means many draws are needed)")

    mean_tau = np.mean([result["autocorrelation"] for result in per_map], axis = 0)
    covariance = _report(louis, names, "Louis, averaged over maps",
                         f" - {args.n_maps} maps x {args.n_draws} draws",
                         n_draws = args.n_maps * args.n_draws,
                         autocorrelation = mean_tau)

    if args.n_maps > len(names):
        cross, n_maps = score_covariance_information(per_map_scores)
        try:
            _report(cross, names, "score covariance, half-split debiased",
                    f" - {n_maps} maps", n_draws = args.n_maps * args.n_draws,
                    autocorrelation = mean_tau)
        except RuntimeError as error:
            print(f"\n  score-covariance estimator not usable yet: {error}")
    else:
        print(f"\n  score-covariance estimator skipped: needs more than {len(names)} maps, "
              f"got {args.n_maps}")

    if not args.no_write:
        directory = output_dir()
        os.makedirs(directory, exist_ok = True)
        subtitle = (f"marginal  |  nside {args.nside}, {args.theta_pix:g}', "
                    f"{args.noise:g} uK-arcmin")
        config = dict(spectra = "marginal", nside = args.nside,
                      theta_pix = args.theta_pix, noise_level = args.noise,
                      l_knee = args.l_knee, beam_fwhm = 0.0,
                      step_fracs = np.array([FD_STEP_FRAC[name] for name in names]),
                      n_maps = args.n_maps, n_draws = args.n_draws,
                      hessian_term = hessian_term, score_covariance = score_term,
                      phi_acceptance = np.array([r["phi_acceptance"] for r in per_map]),
                      autocorrelation = np.array([r["autocorrelation"] for r in per_map]))
        written = write_outputs(louis, covariance, names, directory, "marginal",
                                subtitle, config)
        print(f"\nWrote {', '.join(written)} to {directory}")


if __name__ == "__main__":
    main()
