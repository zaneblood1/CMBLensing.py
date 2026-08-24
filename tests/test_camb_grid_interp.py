"""Tests for the 5D CAMB grid spline (merge_camb_grid.py + camb_grid_interp.py).

These are self-contained: the synthetic-grid tests fabricate slab files whose "Cls" are a
known cubic in the parameters, so a correct tensor-product cubic spline must reproduce
them to round-off. That pins down the parts most likely to break silently - axis ordering,
index-to-slot placement, and the theta_MC_100 -> H0 inversion - without needing CAMB.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline

from cmb_lensing.camb_grid_interp import _eval_bspline, load_camb_grid, CambGrid

MERGE_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "performance_testing", "sampling_chains", "merge_camb_grid.py")

#analytic stand-ins for the grid. lnCl is a cubic in each parameter, so the cubic spline
#is exact on it; theta is monotone in H0 so it can be inverted
def true_ln_cl(h0, log_a, ns, ombh2, omch2, ell, offset = 0.0):
    x = (h0 - 68.0) / 2.0
    return (offset + log_a + 0.3 * ns - 0.5 * x + 0.2 * x**2 - 0.05 * x**3
            + 40.0 * (ombh2 - 0.0224) - 3.0 * (omch2 - 0.1094)
            + 0.01 * ell * (1.0 + 0.1 * x))

def true_theta(h0, ombh2, omch2):
    return 1.0 + 0.003 * (h0 - 66.0) + 0.1 * (ombh2 - 0.0224) - 0.5 * (omch2 - 0.1094)

#each splined spectrum gets its own lnCl offset so a slab landing in the wrong spectrum
#slot cannot cancel out. Unlensed BB is exactly zero (the r = 0 physics the merge step
#verifies); TE crosses zero and is carried as the correlation ratio te_rho, which is
#made cubic in every parameter so the LINEAR te_rho spline must reproduce it exactly
SPECTRUM_OFFSETS = {"tt": 0.0, "ee": -1.0, "pp": -3.0,
                    "tt_lensed": 0.5, "ee_lensed": -1.5, "bb_lensed": -5.0}

def true_te_rho(h0, log_a, ns, ombh2, omch2, ell):
    #sign-crossing in ell, cubic in the parameters, |rho| well below 1
    return 0.05 * np.sin(ell) * true_ln_cl(h0, log_a, ns, ombh2, omch2, 0.0)


@pytest.fixture(scope = "module")
def synthetic_grid(tmp_path_factory):
    """Fabricate the 256 slab files a 4x4x4x4 job array would write, then merge them."""
    grid_dir = tmp_path_factory.mktemp("camb_grid")
    h0_grid = np.linspace(66.0, 70.0, 4)
    ells = np.arange(2, 12).astype(np.float64)
    axes = {"log_a": np.linspace(3.15, 3.29, 4),
            "ns": np.linspace(0.94, 0.98, 4),
            "ombh2": np.linspace(0.0220, 0.0228, 4),
            "omch2": np.linspace(0.106, 0.113, 4)}

    for a, log_a in enumerate(axes["log_a"]):
        for b, ns in enumerate(axes["ns"]):
            for c, ombh2 in enumerate(axes["ombh2"]):
                for d, omch2 in enumerate(axes["omch2"]):
                    cls = {name: np.exp(np.array(
                               [true_ln_cl(h0, log_a, ns, ombh2, omch2, ells,
                                           offset = off) for h0 in h0_grid]))
                           for name, off in SPECTRUM_OFFSETS.items()}
                    cls["bb"] = np.zeros((h0_grid.size, ells.size))
                    rho = np.array([true_te_rho(h0, log_a, ns, ombh2, omch2, ells)
                                    for h0 in h0_grid])
                    cls["te"] = rho * np.sqrt(cls["tt"] * cls["ee"])
                    cls["te_lensed"] = cls["te"]
                    np.savez(grid_dir / f"grid_logA{a}_ns{b}_ombh2{c}_omch2{d}.npz",
                             h0_grid = h0_grid, ells = ells,
                             **{f"cl_{name}": cls[name] for name in cls},
                             ok = np.ones(4, dtype = bool),
                             theta_MC_100 = true_theta(h0_grid, ombh2, omch2),
                             log_a = log_a, ns = ns, ombh2 = ombh2, omch2 = omch2,
                             log_a_index = a, ns_index = b, ombh2_index = c, omch2_index = d,
                             tau = 0.05, mnu = 0.06, lmax = 12, k_pivot = 0.05, alens = 1,
                             accuracy_boost = 1, l_sample_boost = 1, l_accuracy_boost = 1)

    out = grid_dir / "camb_grid_spline.npz"
    subprocess.run([sys.executable, MERGE_SCRIPT, "--grid_dir", str(grid_dir),
                    "--out", str(out)], check = True, capture_output = True)
    return CambGrid(str(out)), h0_grid, axes, ells


def build_coefficients(axis_values, data):
    """Same construction merge_camb_grid.py uses: not-a-knot cubic along each axis."""
    coeff = data.astype(np.float64)
    knots = []
    for axis, values in enumerate(axis_values):
        spline = make_interp_spline(values, np.moveaxis(coeff, axis, 0), k = 3)
        coeff = np.moveaxis(spline.c, 0, axis)
        knots.append(spline.t)
    return coeff, knots


def reference_eval(axis_values, data, point):
    """An independent tensor-product cubic spline: collapse one axis at a time with
    scipy's 1D interpolating spline. Slow, but it shares no code with _eval_bspline."""
    value = data.astype(np.float64)
    for axis, values in enumerate(axis_values):
        value = make_interp_spline(values, value, k = 3, axis = 0)(point[axis])
    return value


def test_bspline_matches_reference():
    """The separable stencil evaluation must equal the independent reference."""
    rng = np.random.default_rng(0)
    axis_values = [np.linspace(0, 1, n) for n in (9, 5, 5, 5, 7)]
    data = rng.normal(size = tuple(v.size for v in axis_values))
    coeff, knots = build_coefficients(axis_values, data)

    pts = rng.uniform(0.0, 1.0, size = (25, 5))
    mine = _eval_bspline(coeff, knots, pts)
    theirs = np.array([reference_eval(axis_values, data, p) for p in pts])
    assert np.abs(mine - theirs).max() < 1e-10


def test_bspline_reproduces_nodes():
    """A cubic spline interpolates: at the nodes it must return the original data."""
    rng = np.random.default_rng(1)
    axis_values = [np.linspace(0, 1, n) for n in (9, 5, 5, 5, 7)]
    data = rng.normal(size = tuple(v.size for v in axis_values))
    coeff, knots = build_coefficients(axis_values, data)
    nodes = np.stack(np.meshgrid(*axis_values, indexing = "ij"), axis = -1).reshape(-1, 5)
    got = _eval_bspline(coeff, knots, nodes)
    assert np.abs(got - data.ravel()).max() < 1e-10


def test_bspline_exact_on_low_order_polynomial():
    """A cubic spline must reproduce a cubic exactly. This is what rules out
    scipy.ndimage's folding boundary modes, which fail it even for a straight line."""
    axis_values = [np.linspace(0.1, 0.9, n) for n in (4, 4, 5)]
    mesh = np.meshgrid(*axis_values, indexing = "ij")
    poly = lambda x, y, z: 1.0 + 2.0 * x - 3.0 * y**2 + 0.5 * z**3 + 4.0 * x * y * z
    data = poly(*mesh)
    coeff, knots = build_coefficients(axis_values, data)
    rng = np.random.default_rng(5)
    pts = np.stack([rng.uniform(v[0], v[-1], 50) for v in axis_values], axis = -1)
    got = _eval_bspline(coeff, knots, pts)
    want = poly(pts[:, 0], pts[:, 1], pts[:, 2])
    assert np.abs(got - want).max() < 1e-12


def test_out_of_box_returns_nan():
    """Queries outside the box must be NaN, not extrapolated - the sampler turns a
    non-finite logpdf into a rejected proposal."""
    axis_values = [np.linspace(0, 4, 5), np.linspace(0, 4, 5)]
    data = np.ones((5, 5))
    coeff, knots = build_coefficients(axis_values, data)
    pts = np.array([[-0.01, 1.0], [1.0, 4.01], [np.nan, 1.0], [1.0, 1.0]])
    got = _eval_bspline(coeff, knots, pts)
    assert np.all(np.isnan(got[:3]))
    assert np.isfinite(got[3])


def test_merge_places_slabs_correctly(synthetic_grid):
    """Every slab must land in its own slot: the merged coefficients, evaluated at the
    grid nodes, have to reproduce the analytic values that were written."""
    grid, h0_grid, axes, ells = synthetic_grid
    rng = np.random.default_rng(2)
    for _ in range(20):
        i = [rng.integers(0, 4) for _ in range(5)]
        h0 = h0_grid[i[0]]
        log_a, ns = axes["log_a"][i[1]], axes["ns"][i[2]]
        ombh2, omch2 = axes["ombh2"][i[3]], axes["omch2"][i[4]]
        theta = true_theta(h0, ombh2, omch2)
        got_tt = grid.cl_tt(np.array([[theta, log_a, ns, ombh2, omch2]]))[0]
        want_tt = np.exp(true_ln_cl(h0, log_a, ns, ombh2, omch2, ells))
        assert np.abs(np.log(got_tt) - np.log(want_tt)).max() < 1e-6


def test_interpolates_between_nodes(synthetic_grid):
    """lnCl is a cubic in every parameter here, so the tensor-product cubic spline is
    exact between nodes too - any error is an axis-ordering or theta-inversion bug.
    Checked for every splined spectrum (unlensed TT/EE, lensed TT/EE/BB, PP)."""
    grid, h0_grid, axes, ells = synthetic_grid
    rng = np.random.default_rng(3)
    worst = {name: 0.0 for name in SPECTRUM_OFFSETS}
    for _ in range(50):
        h0 = rng.uniform(h0_grid[0], h0_grid[-1])
        log_a = rng.uniform(axes["log_a"][0], axes["log_a"][-1])
        ns = rng.uniform(axes["ns"][0], axes["ns"][-1])
        ombh2 = rng.uniform(axes["ombh2"][0], axes["ombh2"][-1])
        omch2 = rng.uniform(axes["omch2"][0], axes["omch2"][-1])
        theta = true_theta(h0, ombh2, omch2)
        pb = np.array([[theta, log_a, ns, ombh2, omch2]])
        for name, off in SPECTRUM_OFFSETS.items():
            worst[name] = max(worst[name], np.abs(
                np.log(grid.cl(name, pb)[0])
                - true_ln_cl(h0, log_a, ns, ombh2, omch2, ells, offset = off)).max())
    for name, err in worst.items():
        assert err < 1e-6, f"{name} worst |delta lnCl| = {err:.2e}"


def test_te_rho_spectrum(synthetic_grid):
    """The linearly-splined TE correlation ratio must reproduce the (cubic-in-parameters)
    truth between nodes, keep its sign structure, and agree across all three predictor
    paths - including gradients through the analytic-VJP predictors."""
    import jax
    import jax.numpy as jnp
    from cmb_lensing.camb_grid_interp import (load_camb_grid_predictors,
                                              load_camb_grid_predictors_grad,
                                              load_camb_grid_predictors_jax)

    grid, h0_grid, axes, ells = synthetic_grid
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(30):
        h0 = rng.uniform(h0_grid[0], h0_grid[-1])
        log_a = rng.uniform(axes["log_a"][0], axes["log_a"][-1])
        ns = rng.uniform(axes["ns"][0], axes["ns"][-1])
        ombh2 = rng.uniform(axes["ombh2"][0], axes["ombh2"][-1])
        omch2 = rng.uniform(axes["omch2"][0], axes["omch2"][-1])
        pb = np.array([[true_theta(h0, ombh2, omch2), log_a, ns, ombh2, omch2]])
        got = grid.cl("te_rho", pb)[0]
        want = true_te_rho(h0, log_a, ns, ombh2, omch2, ells)
        worst = max(worst, np.abs(got - want).max())
    assert worst < 1e-6, f"te_rho worst |delta rho| = {worst:.2e}"

    #out-of-box query rejects like every other spectrum
    bad = np.array([[2.0, 3.2, 0.96, 0.0224, 0.1094]])
    assert np.all(np.isnan(grid.cl("te_rho", bad)))

    #all three predictor paths agree, and the grad predictors' analytic VJP matches
    #jax.grad of the pure-JAX path (sum of values, not log - rho changes sign)
    cb = load_camb_grid_predictors(grid.path)
    gp = load_camb_grid_predictors_grad(grid.path)
    jp = load_camb_grid_predictors_jax(grid.path)
    q = jnp.array([[true_theta(68.0, 0.0224, 0.1094), 3.2, 0.96, 0.0224, 0.1094]])
    r_cb = np.asarray(cb.te_rho(None, q))
    r_jp = np.asarray(jp.te_rho(None, q))
    assert np.abs(r_cb - r_jp).max() < 1e-9
    fg = lambda x: gp.te_rho(None, x[None, :]).sum()
    fj = lambda x: jp.te_rho(None, x[None, :]).sum()
    assert abs(float(fg(q[0])) - float(fj(q[0]))) < 1e-10
    gg = np.asarray(jax.grad(fg)(q[0]))
    gj = np.asarray(jax.grad(fj)(q[0]))
    assert np.abs(gg - gj).max() < 1e-8 * max(1.0, np.abs(gj).max())


def test_zero_bb_spectrum(synthetic_grid):
    """Unlensed BB on an r = 0 grid is exact zeros inside the box, NaN outside (same
    rejected-proposal semantics as the splined spectra), and its parameter gradients
    are identically zero."""
    grid, h0_grid, axes, ells = synthetic_grid
    assert grid.bb_is_zero
    theta = true_theta(68.0, 0.0224, 0.1094)
    good = np.array([[theta, 3.2, 0.96, 0.0224, 0.1094]])
    bad_theta = np.array([[2.0, 3.2, 0.96, 0.0224, 0.1094]])
    bad_axis = np.array([[theta, 99.0, 0.96, 0.0224, 0.1094]])
    assert np.all(grid.cl("bb", good) == 0.0)
    assert np.all(np.isnan(grid.cl("bb", bad_theta)))
    assert np.all(np.isnan(grid.cl("bb", bad_axis)))
    assert np.all(grid.spline_param_grads(good, "bb") == 0.0)


def test_theta_to_h0_inversion(synthetic_grid):
    """theta_MC_100 -> H0 must round-trip, including its dependence on ombh2/omch2."""
    grid, h0_grid, axes, _ = synthetic_grid
    rng = np.random.default_rng(4)
    for _ in range(50):
        h0 = rng.uniform(h0_grid[0], h0_grid[-1])
        ombh2 = rng.uniform(axes["ombh2"][0], axes["ombh2"][-1])
        omch2 = rng.uniform(axes["omch2"][0], axes["omch2"][-1])
        got = grid.h0_from_theta(true_theta(h0, ombh2, omch2), ombh2, omch2)[0]
        assert abs(got - h0) < 1e-6, f"H0 {h0} -> {got}"


def test_theta_outside_range_is_nan(synthetic_grid):
    """A theta unreachable inside the grid's H0 range must give NaN Cls, not a clamp."""
    grid, _, axes, _ = synthetic_grid
    ombh2, omch2 = axes["ombh2"][1], axes["omch2"][1]
    for theta in (0.5, 2.0):
        assert np.isnan(grid.h0_from_theta(theta, ombh2, omch2)[0])
        assert np.all(np.isnan(grid.cl_tt(np.array([[theta, 3.2, 0.96, ombh2, omch2]]))))


def test_jax_bspline_matches_numpy():
    """The jnp evaluator must agree with the numpy stencil path to round-off, including
    NaN for out-of-box and non-finite queries."""
    import jax.numpy as jnp
    from cmb_lensing.camb_grid_interp import _eval_bspline_jax

    rng = np.random.default_rng(6)
    axis_values = [np.linspace(0, 1, n) for n in (9, 5, 7)]
    data = rng.normal(size = tuple(v.size for v in axis_values))
    coeff, knots = build_coefficients(axis_values, data)

    pts = rng.uniform(0.0, 1.0, size = (25, 3))
    pts = np.vstack([pts, [[-0.01, 0.5, 0.5], [0.5, 1.01, 0.5], [0.5, np.nan, 0.5]]])
    want = _eval_bspline(coeff, knots, pts)
    jknots = [jnp.asarray(k) for k in knots]
    got = np.array([_eval_bspline_jax(jnp.asarray(coeff), jknots, jnp.asarray(p))
                    for p in pts])
    assert np.all(np.isnan(got[-3:])) and np.all(np.isnan(want[-3:]))
    assert np.abs(got[:-3] - want[:-3]).max() < 1e-12


def test_jax_bspline_gradients():
    """On a cubic the spline is exact, so its gradient must match the analytic gradient
    of the cubic - and out-of-box queries must give finite (not NaN) gradients so a NUTS
    trajectory that steps over the boundary cannot poison the whole leapfrog."""
    import jax
    import jax.numpy as jnp
    from cmb_lensing.camb_grid_interp import _eval_bspline_jax

    axis_values = [np.linspace(0.1, 0.9, n) for n in (4, 4, 5)]
    mesh = np.meshgrid(*axis_values, indexing = "ij")
    poly = lambda x, y, z: 1.0 + 2.0 * x - 3.0 * y**2 + 0.5 * z**3 + 4.0 * x * y * z
    grad_poly = lambda x, y, z: np.array([2.0 + 4.0 * y * z,
                                          -6.0 * y + 4.0 * x * z,
                                          1.5 * z**2 + 4.0 * x * y])
    data = poly(*mesh)
    coeff, knots = build_coefficients(axis_values, data)
    jcoeff = jnp.asarray(coeff)
    jknots = [jnp.asarray(k) for k in knots]

    grad_fn = jax.grad(lambda p: _eval_bspline_jax(jcoeff, jknots, p))
    rng = np.random.default_rng(7)
    for _ in range(20):
        p = np.array([rng.uniform(v[0], v[-1]) for v in axis_values])
        got = np.asarray(grad_fn(jnp.asarray(p)))
        assert np.abs(got - grad_poly(*p)).max() < 1e-9
    #out-of-box: NaN value, finite gradient
    p_out = jnp.array([0.05, 0.5, 0.5])
    assert np.isnan(_eval_bspline_jax(jcoeff, jknots, p_out))
    assert np.all(np.isfinite(np.asarray(grad_fn(p_out))))


def test_jax_predictors_match_numpy(synthetic_grid):
    """load_camb_grid_predictors_jax must reproduce the callback path - same Cls at
    reachable points (through the theta -> H0 inversion), same NaN for unreachable theta -
    and be differentiable end to end."""
    import jax
    import jax.numpy as jnp
    from cmb_lensing.camb_grid_interp import load_camb_grid_predictors_jax

    grid, h0_grid, axes, ells = synthetic_grid
    predictors = load_camb_grid_predictors_jax(grid.path)
    predict_tt, predict_pp = predictors.tt, predictors.pp

    rng = np.random.default_rng(8)
    pts = []
    for _ in range(20):
        h0 = rng.uniform(h0_grid[0], h0_grid[-1])
        ombh2 = rng.uniform(axes["ombh2"][0], axes["ombh2"][-1])
        omch2 = rng.uniform(axes["omch2"][0], axes["omch2"][-1])
        pts.append([true_theta(h0, ombh2, omch2),
                    rng.uniform(axes["log_a"][0], axes["log_a"][-1]),
                    rng.uniform(axes["ns"][0], axes["ns"][-1]), ombh2, omch2])
    pb = jnp.array(pts)
    assert np.abs(np.log(np.asarray(predict_tt(None, pb)))
                  - np.log(grid.cl_tt(np.asarray(pb)))).max() < 1e-9
    assert np.abs(np.log(np.asarray(predict_pp(None, pb)))
                  - np.log(grid.cl_pp(np.asarray(pb)))).max() < 1e-9
    assert np.abs(np.log(np.asarray(predictors.ee(None, pb)))
                  - np.log(grid.cl("ee", np.asarray(pb)))).max() < 1e-9
    #zero-BB: exact zeros through the jnp path too
    assert np.all(np.asarray(predictors.bb(None, pb)) == 0.0)

    #unreachable theta -> NaN row, matching the callback path
    bad = jnp.array([[2.0, 3.2, 0.96, float(axes["ombh2"][1]), float(axes["omch2"][1])]])
    assert np.all(np.isnan(np.asarray(predict_tt(None, bad))))
    assert np.all(np.isnan(np.asarray(predictors.bb(None, bad))))

    #gradient of a scalar of the Cls w.r.t. the parameter vector: finite, and matching
    #central finite differences through the numpy path
    def scalar(p):
        return jnp.log(predict_tt(None, p[None, :])).sum()
    g = np.asarray(jax.grad(scalar)(pb[0]))
    assert np.all(np.isfinite(g))
    eps = 1e-6
    for a in range(5):
        dp = np.zeros(5)
        dp[a] = eps
        hi = np.log(grid.cl_tt(np.asarray(pb[0] + dp)[None, :])).sum()
        lo = np.log(grid.cl_tt(np.asarray(pb[0] - dp)[None, :])).sum()
        fd = (hi - lo) / (2 * eps)
        assert abs(g[a] - fd) < 1e-4 * max(1.0, abs(fd)), \
            f"axis {a}: grad {g[a]:.6e} vs fd {fd:.6e}"


def test_grad_predictors_match_jax_predictors(synthetic_grid):
    """The callback-forward / analytic-VJP-backward predictors must produce the same
    values AND the same gradients as jax.grad through the pure-JAX predictors - both are
    exact derivatives of the same spline, so they should agree to round-off."""
    import jax
    import jax.numpy as jnp
    from cmb_lensing.camb_grid_interp import (load_camb_grid_predictors_grad,
                                              load_camb_grid_predictors_jax)

    grid, h0_grid, axes, ells = synthetic_grid
    grad_predictors = load_camb_grid_predictors_grad(grid.path)
    jax_predictors = load_camb_grid_predictors_jax(grid.path)
    predict_tt_g = grad_predictors.tt

    rng = np.random.default_rng(9)
    for _ in range(10):
        h0 = rng.uniform(h0_grid[0], h0_grid[-1])
        ombh2 = rng.uniform(axes["ombh2"][0], axes["ombh2"][-1])
        omch2 = rng.uniform(axes["omch2"][0], axes["omch2"][-1])
        p = jnp.array([true_theta(h0, ombh2, omch2),
                       rng.uniform(axes["log_a"][0], axes["log_a"][-1]),
                       rng.uniform(axes["ns"][0], axes["ns"][-1]), ombh2, omch2])
        #every splined spectrum, through both differentiable paths
        for pg, pj in ((grad_predictors.tt, jax_predictors.tt),
                       (grad_predictors.pp, jax_predictors.pp),
                       (grad_predictors.ee, jax_predictors.ee),
                       (grad_predictors.tt_lensed, jax_predictors.tt_lensed),
                       (grad_predictors.ee_lensed, jax_predictors.ee_lensed),
                       (grad_predictors.bb_lensed, jax_predictors.bb_lensed)):
            #same nonlinear scalar of the Cls through both paths
            fg = lambda x, f = pg: jnp.log(f(None, x[None, :])).sum()
            fj = lambda x, f = pj: jnp.log(f(None, x[None, :])).sum()
            assert abs(float(fg(p)) - float(fj(p))) < 1e-10
            gg = np.asarray(jax.grad(fg)(p))
            gj = np.asarray(jax.grad(fj)(p))
            assert np.abs(gg - gj).max() < 1e-8 * max(1.0, np.abs(gj).max()), \
                f"analytic VJP {gg} vs pure-JAX grad {gj}"

    #unreachable theta: NaN value, zero (finite) gradient - and usable under jit
    bad = jnp.array([2.0, 3.2, 0.96, float(axes["ombh2"][1]), float(axes["omch2"][1])])
    f_bad = jax.jit(lambda x: jnp.log(predict_tt_g(None, x[None, :])).sum())
    assert np.isnan(float(f_bad(bad)))
    assert np.all(np.asarray(jax.grad(lambda x: jnp.where(
        jnp.isfinite(f_bad(x)), f_bad(x), 0.0))(bad)) == 0.0)


def test_predictor_signature_under_jit(synthetic_grid):
    """The predictors must be drop-in for the emulator's, i.e. callable inside jit with
    signature f(emu_params, params_batch) -> (M, n_ell)."""
    import jax
    import jax.numpy as jnp
    from cmb_lensing.camb_grid_interp import load_camb_grid_predictors

    grid, h0_grid, axes, ells = synthetic_grid
    predictors = load_camb_grid_predictors(grid.path)
    pb = jnp.array([[true_theta(68.0, 0.0224, 0.1094), 3.2, 0.96, 0.0224, 0.1094],
                    [true_theta(67.0, 0.0224, 0.1094), 3.25, 0.95, 0.0224, 0.1094]])

    @jax.jit
    def run(x):
        return (predictors.tt(None, x), predictors.pp(None, x),
                predictors.ee(None, x), predictors.bb(None, x),
                predictors.ee_lensed(None, x))

    tt, pp, ee, bb, ee_lensed = run(pb)
    for out in (tt, pp, ee, bb, ee_lensed):
        assert out.shape == (2, ells.size)
        assert np.all(np.isfinite(out))
    assert np.all(np.asarray(bb) == 0.0)
    #and it must agree with the direct numpy path
    assert np.abs(np.asarray(tt) - grid.cl_tt(np.asarray(pb))).max() < 1e-9
    assert np.abs(np.asarray(ee_lensed) - grid.cl("ee_lensed", np.asarray(pb))).max() < 1e-9


def test_old_grid_file_stays_loadable(synthetic_grid, tmp_path):
    """A pre-polarization merged file (coeff_tt / coeff_pp only) must still load and
    serve TT/PP, and raise a clear regeneration message only when a missing spectrum is
    actually requested."""
    grid, h0_grid, axes, ells = synthetic_grid
    old = dict(np.load(grid.path))
    for key in list(old):
        if key.startswith("coeff_") and key not in ("coeff_tt", "coeff_pp"):
            del old[key]
    del old["bb_is_zero"]
    old_path = tmp_path / "old_grid.npz"
    np.savez(old_path, **old)

    from cmb_lensing.camb_grid_interp import load_camb_grid_predictors
    old_grid = CambGrid(str(old_path))
    theta = true_theta(68.0, 0.0224, 0.1094)
    pb = np.array([[theta, 3.2, 0.96, 0.0224, 0.1094]])
    assert np.abs(old_grid.cl_tt(pb) - grid.cl_tt(pb)).max() == 0.0
    with pytest.raises(KeyError, match = "no ee spectrum"):
        old_grid.cl("ee", pb)
    predictors = load_camb_grid_predictors(str(old_path))
    assert np.all(np.isfinite(np.asarray(predictors.tt(None, pb))))
    with pytest.raises(KeyError, match = "no ee_lensed spectrum"):
        predictors.ee_lensed(None, pb)
