"""Regression check on camb_grid_interp.py's 5D CAMB grid evaluator.

camb_grid_interp.py used to evaluate the merged grid with ~150 lines of hand-written cubic
B-spline machinery (_axis_stencil, _axis_stencil_derivs, _eval_bspline, _eval_bspline_grads,
including a manual degree-lowering identity for the derivative weights). That was replaced
by scipy.interpolate.NdBSpline, wrapped as camb_grid_interp._TensorSpline - which works
because merge_camb_grid.py already writes exactly what NdBSpline consumes: tensor-product
cubic B-spline coefficients plus the knot vectors that go with them. The merged ~7.9 GB
grid file therefore drives either implementation unchanged.

These tests hold that swap in place. The old implementation is frozen verbatim in
handrolled_bspline_reference.py and the production path is checked against it on a grid
built through the SAME merge-time prefilter merge_camb_grid.build_coefficients_inplace
uses. That matters because the real grid is a gitignored artifact costing 875 slurm jobs
to rebuild, so a silent change in what it predicts would be expensive to notice late.

Also pinned here, because both are easy to "simplify" back into a bug:

  * the BOUNDARY_TOL clamp is still required - bare NdBSpline(extrapolate = False) rejects
    the box edges the sampler's conditional scans land on
  * RegularGridInterpolator(method = "cubic") is NOT an equivalent shortcut, even though
    it now builds an NdBSpline internally: it re-solves for coefficients with an iterative
    Krylov solver and lands ~4e-4 away from the direct solve merge_camb_grid.py does
  * scipy.ndimage.map_coordinates(order = 3) does not reproduce the spline at all

The last test exercises the real ~7.9 GB merged grid and is skipped unless it is present
AND CMB_LENSING_REAL_GRID_TESTS=1 is set, since it loads ~3.4 GB.
"""

import os
import sys
import numpy as np
import pytest
from scipy.interpolate import make_interp_spline, NdBSpline

from cmb_lensing.camb_grid_interp import _TensorSpline, GRID_AXES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handrolled_bspline_reference import _eval_bspline, _eval_bspline_grads

#the production grid's axis lengths (81 H0 x 5 logA x 5 ns x 5 ombh2 x 7 omch2), so the
#short 4-node-class axes that make the boundary treatment matter are represented
GRID_SHAPE = (81, 5, 5, 5, 7)
AXIS_RANGES = [(50.0, 130.0), (2.9, 3.2), (0.93, 0.99), (0.0210, 0.0236), (0.104, 0.136)]
#a stand-in for the ell axis; the real grid has 3998 and the evaluators only ever carry it
N_TRAILING = 64

REAL_GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "cmb_lensing", "camb_splines", "camb_grid_spline.npz")


def _build_grid(seed = 0, dtype = np.float32):
    """A synthetic 5D lnCl-like grid plus its cubic B-spline coefficients, prefiltered the
    way merge_camb_grid.build_coefficients_inplace does it (make_interp_spline per axis,
    not-a-knot ends, float32 storage)."""
    rng = np.random.default_rng(seed)
    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(AXIS_RANGES, GRID_SHAPE)]
    trailing = np.arange(2, 2 + N_TRAILING, dtype = np.float64)
    mesh = np.meshgrid(*axes, indexing = "ij")
    #smooth in every parameter, with an ell-like oscillation, like a real lnCl surface
    surface = sum(np.sin(3 * (x - x.mean()) / (x.max() - x.min())) for x in mesh)
    values = surface[..., None] * np.cos(trailing / 20.0) + np.log(trailing)
    values = values + 1e-3 * rng.standard_normal(values.shape)

    block = values.astype(np.float64)
    knots = []
    for axis in range(5):
        spline = make_interp_spline(axes[axis], np.moveaxis(block, axis, 0), k = 3)
        block = np.moveaxis(spline.c, 0, axis)
        knots.append(spline.t)
    return axes, knots, values, block.astype(dtype)


def _interior_points(axes, count, seed = 1):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(a[1], a[-2], count) for a in axes], axis = -1)


@pytest.fixture(scope = "module")
def grid():
    return _build_grid()


def test_stored_coefficients_are_already_an_ndbspline(grid):
    """The premise of the whole swap: merge_camb_grid.py's output needs no regeneration,
    because len(knots) == n + k + 1 on every axis is precisely NdBSpline's contract."""
    _, knots, _, coeff = grid
    for axis, knot in enumerate(knots):
        assert knot.size == coeff.shape[axis] + 3 + 1, GRID_AXES[axis]
    #constructing it at all is the assertion - NdBSpline validates knots vs coefficients
    NdBSpline(tuple(knots), coeff, 3, extrapolate = False)


def test_ndbspline_matches_hand_rolled_values(grid):
    axes, knots, _, coeff = grid
    points = _interior_points(axes, 32)
    hand = _eval_bspline(coeff, knots, points)
    native = _TensorSpline(coeff, knots)(points)
    assert native.shape == hand.shape
    assert np.max(np.abs(native - hand) / np.abs(hand)) < 1e-12


def test_ndbspline_matches_hand_rolled_derivatives(grid):
    """_axis_stencil_derivs' hand-written degree-lowering identity is reproduced by
    NdBSpline.__call__(nu = ...), which spline_param_grads now goes through."""
    axes, knots, _, coeff = grid
    points = _interior_points(axes, 16, seed = 2)
    hand_value, hand_grads = _eval_bspline_grads(coeff, knots, points)
    native_value, native_grads = _TensorSpline(coeff, knots).value_and_grads(points)
    assert np.max(np.abs(native_value - hand_value) / np.abs(hand_value)) < 1e-12
    scale = np.abs(hand_grads) + np.abs(hand_grads).max() * 1e-6
    assert np.max(np.abs(native_grads - hand_grads) / scale) < 1e-9


def test_interpolation_is_exact_at_the_nodes(grid):
    """Both evaluators must reproduce the sampled values at grid nodes - the property that
    distinguishes an interpolating spline from a smoothing/approximating one."""
    axes, knots, values, coeff = grid
    rng = np.random.default_rng(3)
    idx = [rng.integers(0, n, 12) for n in GRID_SHAPE]
    points = np.stack([axes[a][idx[a]] for a in range(5)], axis = -1)
    expected = values[tuple(idx)]
    native = _TensorSpline(coeff, knots)(points)
    hand = _eval_bspline(coeff, knots, points)
    #float32 coefficient storage carries ~3e-6 absolute error, per merge_camb_grid.py
    assert np.max(np.abs(hand - expected)) < 1e-4
    assert np.max(np.abs(native - expected)) < 1e-4


def test_out_of_box_queries_return_nan_not_an_exception(grid):
    """The sampler turns a NaN spectrum into a rejected proposal, so an out-of-box query
    must never raise. Both evaluators must agree on which rows are NaN."""
    axes, knots, _, coeff = grid
    points = _interior_points(axes, 6, seed = 4)
    points[0, 0] = axes[0][-1] + 10.0
    points[1, 3] = axes[3][0] - 1.0
    points[2, 2] = np.nan
    hand = _eval_bspline(coeff, knots, points)
    native = _TensorSpline(coeff, knots)(points)
    bad = np.isnan(hand).all(axis = -1)
    assert bad.tolist() == [True, True, True, False, False, False]
    assert np.array_equal(bad, np.isnan(native).all(axis = -1))
    assert np.max(np.abs(native[~bad] - hand[~bad]) / np.abs(hand[~bad])) < 1e-12


def test_boundary_tolerance_clamp_is_still_required(grid):
    """A query one ulp past a box edge - what the sampler's linspace scans and the
    theta -> H0 round trip both produce - must still evaluate. Bare NdBSpline returns NaN
    there, so the BOUNDARY_TOL clamp cannot be dropped along with the hand-rolled code."""
    axes, knots, _, coeff = grid
    points = _interior_points(axes, 4, seed = 5)
    for a in range(5):
        points[:, a] = np.nextafter(knots[a][-4], np.inf)
    bare = NdBSpline(tuple(knots), coeff, 3, extrapolate = False)(points)
    assert np.isnan(bare).all(), "expected bare NdBSpline to reject the box edge"
    hand = _eval_bspline(coeff, knots, points)
    native = _TensorSpline(coeff, knots)(points)
    assert np.isfinite(hand).all()
    assert np.isfinite(native).all()
    assert np.max(np.abs(native - hand) / np.abs(hand)) < 1e-12
    #and the clamp itself agrees with the hand-rolled screen
    valid, _ = _TensorSpline(coeff, knots)._screen(points)
    assert valid.all()


def test_ndimage_map_coordinates_does_not_reproduce_the_spline(grid):
    """Confirms merge_camb_grid.py's reason for rejecting scipy.ndimage: every
    prefilter boundary mode it offers folds the data back on itself, and with axes this
    short that error contaminates the whole axis, not just its edges."""
    from scipy import ndimage
    axes, knots, values, coeff = grid
    points = _interior_points(axes, 8, seed = 7)
    hand = _eval_bspline(coeff, knots, points)
    index = np.stack([(points[:, a] - axes[a][0]) / (axes[a][1] - axes[a][0])
                      for a in range(5)])
    worst = np.inf
    for mode in ("mirror", "reflect", "nearest", "grid-wrap"):
        got = np.stack([ndimage.map_coordinates(values[..., j].astype(np.float64), index,
                                                order = 3, mode = mode)
                        for j in range(0, N_TRAILING, 8)], axis = -1)
        want = hand[:, ::8]
        worst = min(worst, float(np.max(np.abs(got - want) / np.abs(want))))
    #NdBSpline agrees to ~1e-13; the best ndimage mode is orders of magnitude worse
    assert worst > 1e-6, f"ndimage matched to {worst:.1e} - re-examine the trade-off"


def test_regular_grid_interpolator_cubic_delegates_to_ndbspline(grid):
    """camb_grid_interp.py used to reject RegularGridInterpolator(method = "cubic") as
    "minutes per call - it rebuilds splines on every evaluation". That is no longer what
    it does: since scipy 1.12 it solves for NdBSpline coefficients ONCE in its constructor
    and every call is a plain NdBSpline evaluation. The cost moved from the call to the
    construction - so the reason to keep it out is now the construction, below."""
    from scipy.interpolate import RegularGridInterpolator
    axes, knots, values, coeff = grid
    #trailing axis trimmed hard: RGI's construction cost is linear in it (~9 s per 64
    #trailing values at this grid shape, i.e. ~9 minutes for the real 3998 ells)
    small = values[..., :4]
    rgi = RegularGridInterpolator(tuple(axes), small, method = "cubic",
                                  bounds_error = False, fill_value = np.nan)
    assert isinstance(rgi._spline, NdBSpline), (
        "RGI cubic no longer builds an NdBSpline - re-check this comparison")

    #it lands on the SAME knot vectors merge_camb_grid.py's make_interp_spline produces
    for axis, knot in enumerate(knots):
        assert np.allclose(np.asarray(rgi._spline.t[axis])[:knot.size], knot), \
            GRID_AXES[axis]

    points = _interior_points(axes, 8, seed = 8)
    hand = _eval_bspline(coeff[..., :4], knots, points)
    got = rgi(points)
    #...but NOT on the same coefficients: RGI solves the N-d interpolation system with an
    #ITERATIVE Krylov solver (scipy.sparse.linalg.gcrotmk, its hardcoded default), which
    #converges only to a tolerance, while merge_camb_grid.py's per-axis make_interp_spline
    #is a direct banded solve. The residual lands right in the error budget
    #merge_camb_grid.py already rejected scipy.ndimage over (1.9e-4 in lnCl)
    residual = float(np.max(np.abs(got - hand) / np.abs(hand)))
    assert 1e-6 < residual < 1e-2, (
        f"RGI cubic differs from the direct separable solve by {residual:.2e}; if this "
        f"has dropped to machine precision its solver default changed")


def test_regular_grid_interpolator_construction_cost_scales_with_the_ell_axis(grid):
    """Why RGI is still the wrong entry point even though its calls are now fast:
    merge_camb_grid.py has ALREADY solved for the coefficients, and handing RGI the raw
    values makes it redo that work - at a cost linear in the ell axis, on a grid whose
    ell axis is 3998 long."""
    import time
    from scipy.interpolate import RegularGridInterpolator
    axes, _, values, _ = grid
    timings = []
    for n_trailing in (4, 32):
        start = time.perf_counter()
        RegularGridInterpolator(tuple(axes), values[..., :n_trailing], method = "cubic",
                                bounds_error = False, fill_value = np.nan)
        timings.append(time.perf_counter() - start)
    #linear in the trailing axis, so 8x the ells costs roughly 8x the build
    assert timings[1] > 3 * timings[0], (
        f"RGI construction no longer scales with the ell axis ({timings}) - re-measure")


@pytest.mark.skipif(not os.path.exists(REAL_GRID_PATH),
                    reason = "the merged 5D CAMB grid is not present")
@pytest.mark.skipif(os.environ.get("CMB_LENSING_REAL_GRID_TESTS") != "1",
                    reason = "set CMB_LENSING_REAL_GRID_TESTS=1 (loads ~3.4 GB)")
def test_real_grid_ndbspline_matches_hand_rolled():
    """The same equivalence on the production grid, including the theta(H0) table that
    CambGrid builds at load time for the theta_MC_100 -> H0 inversion."""
    from cmb_lensing.camb_grid_interp import load_camb_grid
    real = load_camb_grid(REAL_GRID_PATH)
    rng = np.random.default_rng(9)

    #the theta table first: 2 interpolated axes, 2000 trailing H0 nodes, float64, small
    pts2 = np.stack([rng.uniform(real.axes[3][1], real.axes[3][-2], 16),
                     rng.uniform(real.axes[4][1], real.axes[4][-2], 16)], axis = -1)
    hand_v, hand_g = _eval_bspline_grads(real.coeff_theta_dense, real.theta_knots, pts2)
    native_v, native_g = real.theta_spline.value_and_grads(pts2)
    assert np.max(np.abs(native_v - hand_v) / np.abs(hand_v)) < 1e-12
    scale = np.abs(hand_g) + np.abs(hand_g).max() * 1e-6
    assert np.max(np.abs(native_g - hand_g) / scale) < 1e-9

    #then one full 1.06 GB spectrum, straight off the production cache
    coeff = np.load(REAL_GRID_PATH)["coeff_tt"]
    points = np.stack([rng.uniform(a[1], a[-2], 8) for a in real.axes], axis = -1)
    hand = _eval_bspline(coeff, real.knots, points)
    del coeff
    native = real.spline("tt")(points)
    assert np.max(np.abs(native - hand) / np.abs(hand)) < 1e-10
