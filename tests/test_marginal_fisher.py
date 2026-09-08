"""Louis's-identity estimator checks for cmb_lensing/marginal_fisher.py.

Not a Julia benchmark - like tests/test_native_5d_interp.py this checks the module against
an exact analytic result instead of a CMBLensing.jl dump, so it needs no juliacall and no
ground_truth_data npz.

THE ZERO TEST. Feed the estimator draws from the PRIOR rather than the posterior and
Louis's identity must return exactly zero: if p(f, phi | d, theta) were the prior, the data
would carry no information about theta. Both terms collapse to the same bare-prior Fisher
1/2 sum_k w_k dlnC/di dlnC/dj - the Hessian term because E[u] = 1 kills the second-
derivative piece, the score term because u_k has variance 2/w_k - so their difference
vanishes identically. Any error in the fourier weights, the nside^2 normalization, the
factor of 1/2, the finite-difference stencil or a sign shows up as a non-zero residual.

These run real CAMB (9 stencil points for two parameters at nside 32), so budget ~1 minute.
"""

import numpy as np
import pytest

from cmb_lensing.marginal_fisher import (validate_zero_on_prior_draws, prior_information,
                                         log_cl_derivatives, _stencil_offsets)
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH

NSIDE = 32
THETA_PIX = 10.0
NAMES = ["omch2", "logA"]


def _relative(matrix, reference):
    """|matrix| normalized by sqrt(F_ii F_jj) of the reference, so every entry is O(1)."""
    scale = np.sqrt(np.outer(np.abs(np.diag(reference)), np.abs(np.diag(reference))))
    return np.abs(matrix) / scale


@pytest.fixture(scope = "module")
def zero_test():
    #the CAMB stencil dominates the runtime, so both assertions share one evaluation
    return validate_zero_on_prior_draws(NSIDE, THETA_PIX, NAMES, GROUND_TRUTH,
                                        n_draws = 600, seed = 11)


def test_stencil_covers_every_mixed_partial():
    """1 centre + 2n first-derivative points + 4 corners per pair, all distinct."""
    for n_param in (1, 2, 3, 5):
        offsets = _stencil_offsets(n_param)
        assert len(offsets) == len(set(offsets)), "duplicate stencil point"
        assert len(offsets) == 1 + 2 * n_param + 4 * n_param * (n_param - 1) // 2
        assert tuple([0] * n_param) in offsets
        #every pair (i, j) needs all four sign combinations for the mixed partial
        for i in range(n_param):
            for j in range(i + 1, n_param):
                for sign_i in (+1, -1):
                    for sign_j in (+1, -1):
                        wanted = [0] * n_param
                        wanted[i] = sign_i
                        wanted[j] = sign_j
                        assert tuple(wanted) in offsets


def test_hessian_term_reproduces_the_bare_prior_fisher(zero_test):
    """On prior draws E[u] = 1, so E[-Hessian] must collapse to 1/2 sum w dlnC dlnC.

    This isolates the first Louis term and the finite-difference derivatives from the
    score term, which is the noisy half.
    """
    error = _relative(zero_test["hessian_term"] - zero_test["prior_fisher"],
                      zero_test["prior_fisher"])
    assert error.max() < 0.02, (
        f"E[-Hessian] departs from the bare-prior Fisher by {error.max():.3f} "
        f"(relative); the Cl derivatives or the u normalization are wrong")


def test_score_covariance_reproduces_the_bare_prior_fisher(zero_test):
    """Cov(score) on prior draws must ALSO equal the bare-prior Fisher.

    It does so through a different route - Var(u_k) = 2/w_k turns
    1/4 sum w^2 dlnC dlnC (2/w) into 1/2 sum w dlnC dlnC - so this pins the fourier
    weights independently of the test above. Tolerance is the sqrt(2/N) Monte Carlo floor
    with headroom.
    """
    floor = np.sqrt(2.0 / zero_test["n_draws"])
    error = _relative(zero_test["score_covariance"] - zero_test["prior_fisher"],
                      zero_test["prior_fisher"])
    assert error.max() < 6 * floor, (
        f"Cov(score) departs from the bare-prior Fisher by {error.max():.3f} against a "
        f"Monte Carlo floor of {floor:.3f}")


def test_louis_identity_vanishes_on_prior_draws(zero_test):
    """The headline: prior draws carry no information, so I_marginal must be zero."""
    floor = np.sqrt(2.0 / zero_test["n_draws"])
    residual = _relative(zero_test["information"], zero_test["prior_fisher"])
    assert residual.max() < 6 * floor, (
        f"Louis's identity returned {residual.max():.3f} (relative) on prior draws "
        f"instead of zero, against a Monte Carlo floor of {floor:.3f}")


def test_prior_information_matches_fisher_forecast_blocks():
    """prior_information must agree with fisher_forecast's trace formula to O(h^2).

    Both compute 1/2 sum w dlnC/di dlnC/dj on the same spectra, sharing no code, so this
    checks the two stencils against each other. They are NOT algebraically identical
    though, and the tolerance has to respect that: fisher_forecast differences the
    COVARIANCE and divides, (C_+ - C_-)/(2h) / C_0, while this module differences the LOG
    covariance, (lnC_+ - lnC_-)/(2h). Both converge to dlnC/dtheta but carry different
    O(h^2) truncation terms, so at the shipped FD_STEP_FRAC of 0.05 sigma they differ by
    ~1.6e-5 relative. Measured gap against step size: 1.59e-5 at h, 3.04e-6 at h/2 (a
    factor 5.2, consistent with h^2), 1.20e-6 at h/4 (only 2.5, because CAMB's own ~1e-3
    lnCl noise amplified by 1/h starts to dominate the truncation term there). So the
    disagreement is discretization, not a bug - but demanding machine precision here would
    be asserting something the mathematics does not say.
    """
    import jax.numpy as jnp
    from cmb_lensing.util import get_fourier_weights
    from cmb_lensing.fisher_forecast import _fisher_from_blocks, camb_cls_at_params
    from cmb_lensing.simulate import covar_matrix_from_cls
    from cmb_lensing.util import gen_ell_grid
    from cmb_lensing.marginal_fisher import PRIOR_BLOCKS
    from cmb_lensing.fisher_forecast import FD_STEP_FRAC
    from cmb_lensing.precompute_camb_1d import PARAM_SIGMA

    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((NSIDE, NSIDE // 2 + 1))),
                               (NSIDE, NSIDE // 2 + 1))
    derivatives, steps, _, _ = log_cl_derivatives(NAMES, GROUND_TRUTH, NSIDE, THETA_PIX,
                                                  verbose = False)
    mine = prior_information(derivatives, weights, len(NAMES))

    #the same bare-prior blocks, built through fisher_forecast's covariance stencil
    ell_grid, pix_width = gen_ell_grid(NSIDE, THETA_PIX)

    def blocks_at(params):
        cls = camb_cls_at_params(params)
        out = {}
        for block, spectrum in PRIOR_BLOCKS.items():
            ells = jnp.arange(2, 2 + cls[spectrum].shape[0]).astype(jnp.float64)
            out[block] = covar_matrix_from_cls(NSIDE, pix_width, ell_grid, ells,
                                               cls[spectrum], origin_value = 0)
        return out

    plus, minus = [], []
    for name in NAMES:
        step = FD_STEP_FRAC[name] * PARAM_SIGMA[name]
        up = dict(GROUND_TRUTH)
        up[name] = GROUND_TRUTH[name] + step
        down = dict(GROUND_TRUTH)
        down[name] = GROUND_TRUTH[name] - step
        plus.append(blocks_at(up))
        minus.append(blocks_at(down))

    theirs = _fisher_from_blocks(plus, minus, blocks_at(GROUND_TRUTH), steps, weights)
    error = _relative(mine - theirs, theirs)
    assert error.max() < 1e-3, (
        f"prior_information and _fisher_from_blocks disagree by {error.max():.2e}, far "
        f"beyond the ~2e-5 O(h^2) gap between differencing lnC and differencing C; the "
        f"two finite-difference stencils have genuinely drifted apart")
