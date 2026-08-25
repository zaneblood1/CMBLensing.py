"""Evaluate the 5D CAMB grid spline at an arbitrary LCDM parameter vector.

This is the joint-inference counterpart to precompute_camb_1d.load_camb_spline_predictors.
That one splines in a single parameter with the other four pinned at ground truth, so
sample_joint can only sample one parameter at a time. This one interpolates all five at
once, off the tensor-product cubic spline built by merge_camb_grid.py.

    from cmb_lensing.camb_grid_interp import load_camb_grid_predictors
    predictors = load_camb_grid_predictors("<path>/camb_grid_spline.npz")
    predictors.tt, predictors.ee, predictors.bb, predictors.pp  #unlensed + phi
    predictors.tt_lensed, predictors.ee_lensed, predictors.bb_lensed

Every loader returns a GridPredictors namedtuple of functions with the emulator's
predictor signature f(params_batch) -> (M, n_ell), with params_batch ordered as
sample_lcdm.PARAM_ORDER, so each drops straight into
every existing call site (make_eval_logpdf_batch, _recompute_cosmo_matrices,
get_new_cf_matrix). The unlensed spectra feed the field covariances, PP the phi
covariance, and the lensed spectra the theta-dependent quadratic-estimate (and through
it mass-matrix / G) recompute. Unlensed scalar BB is identically zero on the r = 0
grids (bb_is_zero flag from merge_camb_grid.py), so its predictor returns exact zeros -
with NaN rows for out-of-box queries, like every other spectrum. Coefficient tables are
loaded LAZILY, one spectrum at a time on first use (~1.1 GB each), so a T-only run
never pays for the polarization or lensed tables; a pre-polarization grid file stays
loadable, and only raises (with a clear message) if a spectrum it lacks is requested.

theta_MC_100 vs H0
------------------
The sampler works in theta_MC_100 but the grid is laid out in H0, because a rectangular
theta box has corners CAMB cannot solve. theta is a background quantity - it depends only
on (H0, ombh2, omch2) once mnu and tau are pinned - so run_single_camb_grid.py records it
at every node for free and merge_camb_grid.py collapses it to a 3D table. Converting a
query is then: spline that table over (ombh2, omch2) to get the monotone theta(H0) curve
at this cosmology, then invert it by interpolation. The curve is densified once at load
time (N_THETA_DENSE points) so the inversion itself contributes negligible error.

Out-of-box queries return NaN rather than extrapolating, which flows to a non-finite
logpdf and a rejected proposal - byte for byte what the direct-CAMB path does when CAMB
fails, and what the 1D caches do beyond their grid.

Why the spline is hand-rolled
-----------------------------
scipy's RegularGridInterpolator(method="cubic") on this grid shape was measured at minutes
per call - it rebuilds splines on every evaluation. Because merge_camb_grid.py already
solved for the cubic B-spline coefficients, evaluation here is just a local
4-point-per-axis separable contraction, which is both exact (it reproduces the same
interpolating cubic spline) and fast.
"""

import os
from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

#the sampler's parameter ordering (sample_lcdm.PARAM_ORDER)
PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]
#the grid's axis ordering - same, but with the acoustic axis in H0
GRID_AXES = ["H0", "logA", "ns", "ombh2", "omch2"]

#every spectrum a merged grid can serve. All predictor loaders return one of these, with
#one predictor function per field; requesting a spectrum the grid file does not hold
#raises at call time with a pointer to the regeneration scripts. te_rho is the TE
#correlation ratio Cl_TE / sqrt(Cl_TT * Cl_EE) - the sampler reconstructs the TE
#covariance from it and its own splined TT/EE, which keeps the T/E block positive
#semi-definite by construction
SPECTRA = ["tt", "ee", "bb", "pp", "tt_lensed", "ee_lensed", "bb_lensed", "te_rho"]
#spectra splined in their raw values rather than lnCl (sign-changing quantities);
#their predictors return the splined value directly, with no exp
LINEAR_SPECTRA = {"te_rho"}
GridPredictors = namedtuple("GridPredictors", SPECTRA)

#nodes used to densify the theta(H0) curve before inverting it. 2000 points over an H0
#range of ~105 leaves the inversion error orders of magnitude below the spline error
N_THETA_DENSE = 2000

#how far outside an axis a query may land, as a fraction of that axis's span, before it is
#treated as out of box rather than snapped to the edge. Covers round-off only
BOUNDARY_TOL = 1e-9

# ── Cubic B-spline evaluation ─────────────────────────────────────────────

def _axis_stencil(values, knots):
    """For each query value, the 4 coefficient indices and 4 B-spline weights that make up
    the cubic spline there. Thin wrapper over scipy's design_matrix, which handles the
    non-uniform knot spacing that the not-a-knot end condition introduces."""
    from scipy.interpolate import BSpline
    dm = BSpline.design_matrix(values, knots, 3, extrapolate = False).tocsr()
    counts = np.diff(dm.indptr)
    if not np.all(counts == 4):
        raise RuntimeError(f"expected 4 nonzero basis functions per point, got {counts}")
    indices = dm.indices.reshape(-1, 4)
    #_eval_bspline slices the stencil rather than gathering it, which is only valid if the
    #four nonzero basis functions really are consecutive (they are, for any B-spline)
    if not np.all(np.diff(indices, axis = 1) == 1):
        raise RuntimeError("B-spline stencil indices are not consecutive")
    return indices, dm.data.reshape(-1, 4)


def _axis_stencil_derivs(values, knots):
    """Derivative counterpart of _axis_stencil: for each query value, the 4 derivative
    weights dB/dx of the SAME 4 cubic basis functions _axis_stencil selects, via the
    standard degree-lowering identity B'_{m,3}(x) = 3 * (B_{m,2}(x) / (t[m+3] - t[m])
    - B_{m+1,2}(x) / (t[m+4] - t[m+1]))."""
    from scipy.interpolate import BSpline
    indices, _ = _axis_stencil(values, knots)
    dm2 = BSpline.design_matrix(values, knots, 2, extrapolate = False).tocsr()
    counts = np.diff(dm2.indptr)
    if not np.all(counts == 3):
        raise RuntimeError(f"expected 3 nonzero degree-2 basis functions per point, "
                           f"got {counts}")
    idx2 = dm2.indices.reshape(-1, 3)
    if not np.all(idx2[:, 0] == indices[:, 0] + 1):
        raise RuntimeError("degree-2 stencil misaligned with the cubic stencil")
    #b2[:, j] = B_{m0+j, 2} for j = 0..4 with m0 the first cubic index: only j = 1..3 are
    #nonzero at any interior point, and the j = 0 / j = 4 zeros make the identity's two
    #terms uniform below
    b2 = np.zeros((values.shape[0], 5))
    b2[:, 1:4] = dm2.data.reshape(-1, 3)
    t = np.asarray(knots, dtype = np.float64)
    m0 = indices[:, 0]
    dweights = np.zeros((values.shape[0], 4))
    for j in range(4):
        m = m0 + j
        d1 = t[m + 3] - t[m]
        d2 = t[m + 4] - t[m + 1]
        #zero-width spans (repeated end knots) contribute nothing, by convention
        term1 = np.where(d1 > 0, b2[:, j] / np.where(d1 > 0, d1, 1.0), 0.0)
        term2 = np.where(d2 > 0, b2[:, j + 1] / np.where(d2 > 0, d2, 1.0), 0.0)
        dweights[:, j] = 3.0 * (term1 - term2)
    return dweights


def _eval_bspline_grads(coeff, knots, points):
    """Value AND per-axis first derivative of the tensor-product cubic spline.

    Same contract as _eval_bspline, returning a second array of shape
    (M, k) + trailing with d value / d points[:, a] in slot a. NaN rows for out-of-box
    queries in both outputs. The derivative along axis a is the same separable
    contraction with the axis-a value weights swapped for _axis_stencil_derivs."""
    n_axes = len(knots)
    trailing = coeff.shape[n_axes:]
    out = np.full((points.shape[0],) + trailing, np.nan)
    gout = np.full((points.shape[0], n_axes) + trailing, np.nan)

    clamped = np.array(points, dtype = np.float64, copy = True)
    valid = np.all(np.isfinite(points), axis = 1)
    for a, knot in enumerate(knots):
        lo, hi = knot[3], knot[-4]
        tol = BOUNDARY_TOL * (hi - lo)
        valid &= (points[:, a] >= lo - tol) & (points[:, a] <= hi + tol)
        clamped[:, a] = np.clip(clamped[:, a], lo, hi)
    where = np.flatnonzero(valid)
    if where.size == 0:
        return out, gout

    stencils = [_axis_stencil(clamped[where, a], knots[a]) for a in range(n_axes)]
    dstencils = [_axis_stencil_derivs(clamped[where, a], knots[a]) for a in range(n_axes)]
    for m, target in enumerate(where):
        block = coeff[tuple(slice(stencils[a][0][m, 0], stencils[a][0][m, 0] + 4)
                            for a in range(n_axes))]
        value = block
        for a in range(n_axes):
            value = np.tensordot(stencils[a][1][m], value, axes = ([0], [0]))
        out[target] = value
        for da in range(n_axes):
            value = block
            for a in range(n_axes):
                wvec = dstencils[a][m] if a == da else stencils[a][1][m]
                value = np.tensordot(wvec, value, axes = ([0], [0]))
            gout[target, da] = value
    return out, gout


def _eval_bspline(coeff, knots, points):
    """Evaluate a tensor-product cubic B-spline.

    coeff  - spline coefficients, shape (n_0, ..., n_{k-1}) + (any trailing dims)
    knots  - list of k knot vectors, one per interpolated axis (from merge_camb_grid.py)
    points - (M, k) query points in parameter units

    Returns (M,) + trailing, NaN wherever a query falls outside the box. Trailing
    dimensions (the ell axis) are carried through untouched.

    The not-a-knot end condition matters: scipy.ndimage's prefilter modes all impose a
    folding boundary that is not exact even for a straight line, and with axes as short as
    4 nodes that error (measured at 1.9e-4 in lnCl) contaminates the entire axis rather
    than just its edges."""
    n_axes = len(knots)
    trailing = coeff.shape[n_axes:]
    out = np.full((points.shape[0],) + trailing, np.nan)

    #design_matrix raises on out-of-range input, so screen the box first. Outside queries
    #stay NaN, which the sampler turns into a rejected proposal.
    #
    #The tolerance is not cosmetic: the sampler scans each conditional on a
    #jnp.linspace(lo, hi, SEARCH_PRECISION) whose endpoints ARE the box edges, and a query
    #built by round-tripping through theta -> H0 lands a few ulp outside. Without this,
    #every scan would silently lose its two endpoints to NaN
    clamped = np.array(points, dtype = np.float64, copy = True)
    valid = np.all(np.isfinite(points), axis = 1)
    for a, knot in enumerate(knots):
        lo, hi = knot[3], knot[-4]
        tol = BOUNDARY_TOL * (hi - lo)
        valid &= (points[:, a] >= lo - tol) & (points[:, a] <= hi + tol)
        clamped[:, a] = np.clip(clamped[:, a], lo, hi)
    where = np.flatnonzero(valid)
    if where.size == 0:
        return out

    stencils = [_axis_stencil(clamped[where, a], knots[a]) for a in range(n_axes)]
    for m, target in enumerate(where):
        #The four nonzero cubic B-spline basis functions at any point are always
        #consecutive, so the stencil is a contiguous slice rather than a fancy-index
        #gather. That matters: np.ix_ would copy the whole 4^n_axes x n_ell block (16 MB
        #per query at the production grid shape) before any arithmetic happened.
        block = coeff[tuple(slice(stencils[a][0][m, 0], stencils[a][0][m, 0] + 4)
                            for a in range(n_axes))]
        #contract one axis at a time; separable contraction is ~4^n multiply-adds per ell
        #instead of materializing an outer product of the weights
        for a in range(n_axes):
            block = np.tensordot(stencils[a][1][m], block, axes = ([0], [0]))
        out[target] = block
    return out


# ── Grid file ─────────────────────────────────────────────────────────────

class CambGrid:
    """The merged 5D spline, plus the theta_MC_100 -> H0 conversion it needs."""

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no merged CAMB grid at {path} - generate the grid with "
                f"camb_grid.sh and merge it with "
                f"merge_camb_grid.py")
        data = np.load(path)
        self.path = path
        #coefficient tables are ~1.1 GB each and there are up to six of them, so they are
        #NOT read here: _coeff_cache fills lazily, per spectrum, on first use. np.load on
        #an npz keeps a file handle and reads members on indexing, which is exactly the
        #laziness needed
        self._data = data
        self._coeff_cache = {}
        #unlensed scalar BB is identically zero on r = 0 / tensors-off grids; the merge
        #step verifies that and records this flag instead of a coefficient table
        self.bb_is_zero = bool(data["bb_is_zero"]) if "bb_is_zero" in data.files else False
        self.ells = data["ells"]
        self.n_ell = self.ells.size
        self.axes = [data[f"axis_{name}"] for name in GRID_AXES]
        self.knots = [data[f"knots_{name}"] for name in GRID_AXES]

        #densify theta(H0) once at every (ombh2, omch2) node, so a query only has to spline
        #over those two axes and then invert a 1D monotone curve. The nodes are exact grid
        #points, so this is a plain 1D cubic along H0 - no 3D stencil needed
        from scipy.interpolate import CubicSpline
        self.h0_dense = np.linspace(self.axes[0][0], self.axes[0][-1], N_THETA_DENSE)
        dense = CubicSpline(self.axes[0], data["theta_grid"], axis = 0)(self.h0_dense)
        #_eval_bspline interpolates over the LEADING axes and carries the rest through as
        #trailing values, so the dense H0 axis has to sit last: the two interpolated axes
        #are ombh2 and omch2, and what comes back is a whole theta(H0) curve per query
        dense = np.moveaxis(dense, 0, -1)
        if not np.all(np.diff(dense, axis = -1) > 0):
            raise RuntimeError("theta_MC_100 is not monotonically increasing in H0 "
                               "somewhere on the grid - it cannot be inverted")
        #spline over the two node axes so queries between ombh2/omch2 nodes are
        #interpolated, not snapped to the nearest node
        from scipy.interpolate import make_interp_spline
        coeff_dense = dense
        theta_knots = []
        for axis, values in ((0, self.axes[3]), (1, self.axes[4])):
            spline = make_interp_spline(values, np.moveaxis(coeff_dense, axis, 0), k = 3)
            coeff_dense = np.moveaxis(spline.c, 0, axis)
            theta_knots.append(spline.t)
        self.coeff_theta_dense = coeff_dense
        self.theta_knots = theta_knots

    def h0_from_theta(self, theta, ombh2, omch2):
        """Invert theta_MC_100 -> H0 at each (ombh2, omch2). Returns NaN where the
        requested theta is not reachable inside the grid's H0 range."""
        pts = np.stack([np.atleast_1d(ombh2), np.atleast_1d(omch2)], axis = -1)
        curves = _eval_bspline(self.coeff_theta_dense, self.theta_knots, pts)
        theta = np.atleast_1d(theta)
        h0 = np.full(theta.shape, np.nan)
        for m in range(theta.size):
            curve = curves[m]
            if not np.all(np.isfinite(curve)):
                continue
            #same round-off tolerance as _eval_bspline: a theta taken from the edge of the
            #sampler's search range must not be rejected for being one ulp past the end of
            #the curve it was derived from
            tol = BOUNDARY_TOL * (curve[-1] - curve[0])
            if theta[m] < curve[0] - tol or theta[m] > curve[-1] + tol:
                continue
            h0[m] = np.interp(np.clip(theta[m], curve[0], curve[-1]), curve, self.h0_dense)
        return h0

    def grid_points(self, params_batch):
        """(M, 5) in PARAM_ORDER -> (M, 5) in the grid's own axes, i.e. with
        theta_MC_100 replaced by the H0 that produces it at this ombh2/omch2."""
        pb = np.atleast_2d(np.asarray(params_batch, dtype = np.float64))
        h0 = self.h0_from_theta(pb[:, 0], pb[:, 3], pb[:, 4])
        return np.stack([h0, pb[:, 1], pb[:, 2], pb[:, 3], pb[:, 4]], axis = -1)

    def has_spectrum(self, spectrum):
        if spectrum == "bb" and self.bb_is_zero:
            return True
        return f"coeff_{spectrum}" in self._data.files

    def coeff(self, spectrum):
        """Lazily load (and cache) one spectrum's coefficient table."""
        if spectrum not in self._coeff_cache:
            key = f"coeff_{spectrum}"
            if key not in self._data.files:
                raise KeyError(
                    f"the merged CAMB grid at {self.path} has no {spectrum} spectrum - it "
                    f"predates the polarization/lensed-Cl extension. Re-run "
                    f"sampling_chains/camb_grid.sh and "
                    f"merge_camb_grid.py to regenerate it")
            self._coeff_cache[spectrum] = self._data[key]
        return self._coeff_cache[spectrum]

    def _points_valid(self, points):
        """The same validity screen _eval_bspline applies: finite coordinates inside the
        box (with the round-off tolerance). Needed to give the zero-BB spectrum the same
        NaN-outside-the-box semantics as the splined spectra."""
        valid = np.all(np.isfinite(points), axis = 1)
        for a, knot in enumerate(self.knots):
            lo, hi = knot[3], knot[-4]
            tol = BOUNDARY_TOL * (hi - lo)
            valid &= (points[:, a] >= lo - tol) & (points[:, a] <= hi + tol)
        return valid

    def cl(self, spectrum, params_batch):
        """(M, 5) in PARAM_ORDER -> (M, n_ell) values of the requested spectrum (Cls for
        the lnCl-splined spectra, the correlation ratio itself for te_rho). NaN rows
        outside the box (or at an unreachable theta_MC_100), which the sampler turns into
        a rejected proposal."""
        points = self.grid_points(params_batch)
        if spectrum == "bb" and self.bb_is_zero:
            out = np.zeros((points.shape[0], self.n_ell))
            out[~self._points_valid(points)] = np.nan
            return out
        values = _eval_bspline(self.coeff(spectrum), self.knots, points)
        if spectrum in LINEAR_SPECTRA:
            return values
        return np.exp(values)

    def cl_tt(self, params_batch):
        return self.cl("tt", params_batch)

    def cl_pp(self, params_batch):
        return self.cl("pp", params_batch)

    def cls(self, params_batch):
        """(M, 5) in PARAM_ORDER -> (cl_tt, cl_pp), each (M, n_ell). NaN rows outside the
        box, which the sampler turns into a rejected proposal."""
        return self.cl_tt(params_batch), self.cl_pp(params_batch)

    def spline_param_grads(self, params_batch, spectrum):
        """d (splined quantity) / d params in PARAM_ORDER: (M, 5) -> (M, 5, n_ell), NaN
        rows where the query is out of box or its theta_MC_100 is unreachable. The
        splined quantity is lnCl for the log spectra and the raw value (the correlation
        ratio) for the LINEAR_SPECTRA.

        The grid is laid out in H0, so the chain rule runs through the theta -> H0
        inversion: with g_a = d lnCl / d grid-axis a and theta(H0, ombh2, omch2) the
        recorded background relation,
            d/d theta = g_H0 / (d theta / d H0)
            d/d ombh2 = g_ombh2 - g_H0 * (d theta / d ombh2) / (d theta / d H0)
        (implicit function theorem; omch2 analogous). The dense theta(H0) curve is
        inverted by linear interpolation, so d theta / d H0 is the containing segment's
        slope - exactly what autodiff of the jnp.interp in the pure-JAX path uses."""
        pb = np.atleast_2d(np.asarray(params_batch, dtype = np.float64))
        M = pb.shape[0]
        #the zero-BB spectrum is constant in theta, so its gradient contribution is
        #zero everywhere (the VJP multiplies by Cl = 0 anyway)
        if spectrum == "bb" and self.bb_is_zero:
            return np.zeros((M, 5, self.n_ell))
        coeff = self.coeff(spectrum)

        pts2 = np.stack([pb[:, 3], pb[:, 4]], axis = -1)
        curves, dcurves = _eval_bspline_grads(self.coeff_theta_dense, self.theta_knots,
                                              pts2)
        h0 = np.full(M, np.nan)
        dh0_dtheta = np.full(M, np.nan)
        dh0_dob = np.full(M, np.nan)
        dh0_doc = np.full(M, np.nan)
        for m in range(M):
            curve = curves[m]
            if not np.all(np.isfinite(curve)):
                continue
            tol = BOUNDARY_TOL * (curve[-1] - curve[0])
            theta = pb[m, 0]
            if not np.isfinite(theta) or theta < curve[0] - tol or theta > curve[-1] + tol:
                continue
            theta_c = np.clip(theta, curve[0], curve[-1])
            h0[m] = np.interp(theta_c, curve, self.h0_dense)
            seg = np.clip(np.searchsorted(curve, theta_c), 1, curve.size - 1)
            slope = ((curve[seg] - curve[seg - 1])
                     / (self.h0_dense[seg] - self.h0_dense[seg - 1]))
            dth_dob = np.interp(h0[m], self.h0_dense, dcurves[m, 0])
            dth_doc = np.interp(h0[m], self.h0_dense, dcurves[m, 1])
            dh0_dtheta[m] = 1.0 / slope
            dh0_dob[m] = -dth_dob / slope
            dh0_doc[m] = -dth_doc / slope

        grid_pts = np.stack([h0, pb[:, 1], pb[:, 2], pb[:, 3], pb[:, 4]], axis = -1)
        _, g = _eval_bspline_grads(coeff, self.knots, grid_pts)

        grads = np.full((M, 5, self.n_ell), np.nan)
        grads[:, 0] = g[:, 0] * dh0_dtheta[:, None]
        grads[:, 1] = g[:, 1]
        grads[:, 2] = g[:, 2]
        grads[:, 3] = g[:, 3] + g[:, 0] * dh0_dob[:, None]
        grads[:, 4] = g[:, 4] + g[:, 0] * dh0_doc[:, None]
        return grads

_loaded_grids = {}

def load_camb_grid(path):
    """Load (and cache) a merged grid. Cached so the file is opened once per process even
    though the predictors are built separately (the coefficient tables themselves load
    lazily, per spectrum)."""
    path = os.path.abspath(path)
    if path not in _loaded_grids:
        _loaded_grids[path] = CambGrid(path)
    return _loaded_grids[path]

def _missing_spectrum_predictor(grid, spectrum):
    """Stand-in slot for a spectrum the grid file does not hold: raises with a clear
    message the moment it is called (i.e. at trace time), so a pre-polarization grid
    file keeps working for T-only sampling."""
    def predict(params_batch):
        raise KeyError(
            f"the merged CAMB grid at {grid.path} has no {spectrum} spectrum - it "
            f"predates the polarization/lensed-Cl extension. Re-run "
            f"sampling_chains/camb_grid.sh and merge_camb_grid.py "
            f"to regenerate it")
    return predict

_loaded_callback_predictors = {}

def load_camb_grid_predictors(path):
    """Returns a GridPredictors namedtuple of functions with the emulator predictor
    signature f(params_batch) -> (M, n_ell), backed by the 5D grid spline.
    jit-safe via jax.pure_callback, matching
    load_camb_spline_predictors. Cached per path so consumers passing the predictors as
    static jit arguments always see the same function objects."""
    path = os.path.abspath(path)
    if path in _loaded_callback_predictors:
        return _loaded_callback_predictors[path]
    grid = load_camb_grid(path)

    def make_predictor(spectrum):
        if not grid.has_spectrum(spectrum):
            return _missing_spectrum_predictor(grid, spectrum)

        def eval_fn(params_batch):
            return grid.cl(spectrum, params_batch)

        def predict(params_batch):
            M = params_batch.shape[0]
            return jax.pure_callback(
                eval_fn, jax.ShapeDtypeStruct((M, grid.n_ell), jnp.float64),
                params_batch, vmap_method = "sequential"
            )
        return predict

    predictors = GridPredictors(*[make_predictor(name) for name in SPECTRA])
    _loaded_callback_predictors[path] = predictors
    return predictors
