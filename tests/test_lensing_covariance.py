"""Checks for cmb_lensing.lensing_covariance against brute-force references on tiny grids.

No CAMB and no Julia: the spectra are synthetic power laws, which is all the kernel's
algebra needs. What is pinned here is the FFT bookkeeping - circular convolution index
conventions, the half / full plane extension, the real-DOF weights - not the physics; the
physics normalization is checked against CAMB by first_order_lensing_check at run time.
"""

import numpy as np
import pytest

from cmb_lensing.lensing_covariance import (LensingKernel, LensingBandpowerCovariance,
                                            half_to_full, full_to_half, full_wavevectors,
                                            fisher_from_bandpower_covariance,
                                            independent_modes, independent_to_full)


def _power_law(nside, pix_width, amplitude, index, ell_min = 30.0):
    """A smooth even spectrum on the rfft half plane with a zero origin."""
    kx, ky = full_wavevectors(nside, pix_width)
    ell = np.sqrt(kx**2 + ky**2)
    full = amplitude * (np.maximum(ell, ell_min) / 1000.0)**index
    full[0, 0] = 0.0
    return full_to_half(full)


@pytest.fixture(params = [8, 10])
def tiny_grid(request):
    nside = request.param
    pix_width = np.deg2rad(5.0 / 60)
    cf = _power_law(nside, pix_width, 1.0, -2.3)
    cphi = _power_law(nside, pix_width, 1e-7, -3.8)
    _, weights = independent_modes(nside)
    return nside, pix_width, cf, cphi, weights


def test_half_full_round_trip(tiny_grid):
    nside, pix_width, cf, _, _ = tiny_grid
    full = half_to_full(cf)
    #even under l -> -l: full[i, j] == full[-i, -j]
    rows = (-np.arange(nside)) % nside
    flipped = full[np.ix_(rows, rows)]
    assert np.allclose(full, flipped)
    assert np.array_equal(full_to_half(full), cf)


def test_independent_modes_bookkeeping(tiny_grid):
    nside, _, cf, _, weights = tiny_grid
    mask, _ = independent_modes(nside)
    #the weights still count nside^2 real degrees of freedom
    assert weights.sum() == nside**2
    #an even function restricted to the independent modes and folded back is itself
    restricted = np.where(mask, cf, 0.0)
    assert np.allclose(independent_to_full(restricted), half_to_full(cf))
    #each conjugate pair appears once: no two independent modes are negations of each other
    rows = (-np.arange(nside)) % nside
    full_mask = np.zeros((nside, nside), dtype = bool)
    full_mask[:, :nside // 2 + 1] = mask
    negated = full_mask[np.ix_(rows, rows)]
    self_conjugate = np.zeros((nside, nside), dtype = bool)
    for i, j in ((0, 0), (0, nside // 2), (nside // 2, 0), (nside // 2, nside // 2)):
        self_conjugate[i, j] = True
    assert not np.any(full_mask & negated & ~self_conjugate)


def test_kernel_matches_brute_force(tiny_grid):
    nside, pix_width, cf, _, _ = tiny_grid
    kernel = LensingKernel(cf, nside, pix_width)
    dense = kernel.dense()
    rng = np.random.default_rng(0)
    v = rng.standard_normal((nside, nside))
    g = rng.standard_normal((nside, nside))
    assert np.allclose(kernel.apply(v).ravel(), dense @ v.ravel(), rtol = 1e-10, atol = 1e-12)
    assert np.allclose(kernel.apply_transpose(g).ravel(), dense.T @ g.ravel(),
                       rtol = 1e-10, atol = 1e-12)


def test_kernel_preserves_evenness(tiny_grid):
    nside, pix_width, cf, cphi, _ = tiny_grid
    kernel = LensingKernel(cf, nside, pix_width)
    out = kernel.apply(half_to_full(cphi))
    rows = (-np.arange(nside)) % nside
    assert np.allclose(out, out[np.ix_(rows, rows)])
    back = kernel.apply_transpose(half_to_full(cf))
    assert np.allclose(back, back[np.ix_(rows, rows)])


def _brute_force_covariance(nside, pix_width, cf, cphi, nphi, noise, weights, cross,
                            cphi_eff):
    """Cov(mu_hat) built explicitly from the dense kernel and the half-plane formulas."""
    kernel = LensingKernel(cf, nside, pix_width)
    dense = kernel.dense()
    half_index = np.arange(nside * nside).reshape(nside, nside)[:, :nside // 2 + 1]
    rows = (-np.arange(nside)) % nside
    #the index of -L on the full plane for every half-plane L
    minus_index = np.zeros_like(half_index)
    for i in range(nside):
        for j in range(nside // 2 + 1):
            minus_index[i, j] = rows[i] * nside + ((-j) % nside)
    independent, _ = independent_modes(nside)
    tt_fid = cf + noise
    pp_fid = cphi + nphi
    good_tt = (tt_fid > 0) & independent
    good_pp = (pp_fid > 0) & independent
    l_idx = half_index[good_tt]
    l_minus = minus_index[good_tt]
    L_idx = half_index[good_pp]
    L_minus = minus_index[good_pp]
    #K_sym(l, L) = [K(l, L) + K(l, -L)] / 2 over half-plane l and L
    k_sym = 0.5 * (dense[np.ix_(l_idx, L_idx)] + dense[np.ix_(l_idx, L_minus)])
    w_L = weights[good_pp]
    c2 = cphi_eff[good_pp]**2
    n_tt, n_pp = len(l_idx), len(L_idx)
    cov = np.zeros((n_tt + n_pp, n_tt + n_pp))
    cov[:n_tt, :n_tt] = np.diag(2 * tt_fid[good_tt]**2 / weights[good_tt])
    cov[n_tt:, n_tt:] = np.diag(2 * pp_fid[good_pp]**2 / w_L)
    cov[:n_tt, :n_tt] += 2 * (k_sym * (w_L * c2)) @ k_sym.T
    if cross:
        block = 2 * k_sym * c2
        cov[:n_tt, n_tt:] = block
        cov[n_tt:, :n_tt] = block.T
    return cov


@pytest.mark.parametrize("cross", [True, False])
def test_operator_matches_brute_force_covariance(tiny_grid, cross):
    nside, pix_width, cf, cphi, weights = tiny_grid
    noise = 0.05 * np.max(cf) * np.ones_like(cf)
    noise[0, 0] = 0.0
    nphi = 3.0 * cphi
    cphi_eff = cphi if cross else cphi * nphi / np.where(cphi + nphi > 0, cphi + nphi, 1.0)
    reference = _brute_force_covariance(nside, pix_width, cf, cphi, nphi, noise, weights,
                                        cross, cphi_eff)
    operator = LensingBandpowerCovariance(cf + noise, cphi + nphi, cf, cphi_eff,
                                          cross, nside, pix_width)
    dense = operator.dense()
    assert np.allclose(dense, reference, rtol = 1e-9, atol = 1e-9 * np.abs(reference).max())
    #symmetric and positive definite, as a covariance must be
    assert np.allclose(dense, dense.T)
    assert np.linalg.eigvalsh(dense).min() > 0


def test_solve_and_fisher_against_dense_inverse(tiny_grid):
    nside, pix_width, cf, cphi, weights = tiny_grid
    noise = 0.05 * np.max(cf) * np.ones_like(cf)
    noise[0, 0] = 0.0
    nphi = 3.0 * cphi
    operator = LensingBandpowerCovariance(cf + noise, cphi + nphi, cf, cphi,
                                          True, nside, pix_width)
    rng = np.random.default_rng(1)
    derivatives = [rng.standard_normal(operator.size) * operator.diagonal**0.5
                   for _ in range(3)]
    dense = operator.dense()
    expected = np.array([[d_i @ np.linalg.solve(dense, d_j) for d_j in derivatives]
                         for d_i in derivatives])
    fisher, asymmetry = fisher_from_bandpower_covariance(operator, derivatives)
    assert asymmetry < 1e-8
    assert np.allclose(fisher, expected, rtol = 1e-7)


def test_gaussian_only_is_the_diagonal(tiny_grid):
    nside, pix_width, cf, cphi, weights = tiny_grid
    noise = 0.05 * np.max(cf) * np.ones_like(cf)
    noise[0, 0] = 0.0
    operator = LensingBandpowerCovariance(cf + noise, cphi, cf, cphi, True,
                                          nside, pix_width, non_gaussian = False)
    x = np.arange(operator.size, dtype = float) + 1.0
    assert np.allclose(operator.matvec(x), operator.diagonal * x)
    assert np.allclose(operator.solve(x), x / operator.diagonal)
