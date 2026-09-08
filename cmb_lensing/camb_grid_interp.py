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
loaded LAZILY, one spectrum at a time on first use (~1.1 GB read, ~2.1 GB resident once
NdBSpline has cast it to float64), so a T-only run never pays for the polarization or
lensed tables; a pre-polarization grid file stays
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

Why NdBSpline, and not RegularGridInterpolator
---------------------------------------------
Evaluation is scipy.interpolate.NdBSpline, which consumes precomputed tensor-product
B-spline coefficients plus knots - precisely what merge_camb_grid.py already writes. The
grid file needs no regeneration for this and the arithmetic is identical; see
_TensorSpline. (This replaced a hand-rolled 4-point-per-axis separable contraction, kept
as a regression reference in tests/handrolled_bspline_reference.py, that the production
path is checked against in tests/test_native_5d_interp.py.)

RegularGridInterpolator(method="cubic") is the wrong entry point despite building an
NdBSpline of its own since scipy 1.12. It wants the raw sampled values, so it re-solves
the interpolation problem merge_camb_grid.py has already solved - at a cost linear in the
ell axis (~9 s per 64 ells at this grid shape, i.e. of order ten minutes per spectrum at
3998 ells) - and it solves it with an ITERATIVE Krylov solver (scipy.sparse.linalg.gcrotmk,
its hardcoded default) rather than the direct banded solve merge_camb_grid.py's per-axis
make_interp_spline uses. The resulting coefficients differ by ~4e-4, which is the size of
the whole lnCl error budget. scipy.ndimage's prefilter is out for the reason
merge_camb_grid.py records: every boundary mode it offers folds the data back on itself,
which is not exact even for a straight line and, on axes as short as 4 nodes, contaminates
the entire axis rather than just its edges.
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

class _TensorSpline:
    """One coefficient table wrapped as an evaluable tensor-product cubic B-spline.

    merge_camb_grid.py writes, per spectrum, the B-spline coefficients over the five
    parameter axes plus the knot vectors that go with them - which is exactly
    scipy.interpolate.NdBSpline's contract (len(knots[a]) == coeff.shape[a] + k + 1), with
    the trailing ell axis carried through untouched. So the stored grid drives NdBSpline
    directly, with no regeneration and no reinterpretation of the file format.

    What this class still owns on top of NdBSpline is the box screen. NdBSpline's
    extrapolate = False returns NaN strictly outside the knot range with no round-off
    slack, and the sampler needs slack: it scans each conditional on a
    jnp.linspace(lo, hi, SEARCH_PRECISION) whose endpoints ARE the box edges, and a query
    built by round-tripping through theta -> H0 lands a few ulp outside. Without the
    BOUNDARY_TOL clamp every scan would silently lose its two endpoints to NaN. Queries
    genuinely outside stay NaN, which the sampler turns into a rejected proposal - byte for
    byte what the direct-CAMB path does when CAMB fails.

    Memory: NdBSpline's constructor casts the coefficients to float64 (scipy's
    _get_dtype), so a ~1.1 GB float32 table becomes a ~2.1 GB float64 one. CambGrid.spline
    therefore drops its reference to the float32 array as soon as the spline is built,
    leaving one copy resident rather than two."""

    def __init__(self, coeff, knots):
        from scipy.interpolate import NdBSpline
        self.knots = [np.asarray(k, dtype = np.float64) for k in knots]
        self.n_axes = len(self.knots)
        self.trailing = tuple(coeff.shape[self.n_axes:])
        self.spline = NdBSpline(tuple(self.knots), coeff, 3, extrapolate = False)

    def _screen(self, points):
        """(valid mask, points clipped onto the closed box), with BOUNDARY_TOL of slack."""
        clamped = np.array(points, dtype = np.float64, copy = True)
        valid = np.all(np.isfinite(points), axis = 1)
        for a, knot in enumerate(self.knots):
            lo, hi = knot[3], knot[-4]
            tol = BOUNDARY_TOL * (hi - lo)
            valid &= (points[:, a] >= lo - tol) & (points[:, a] <= hi + tol)
            clamped[:, a] = np.clip(clamped[:, a], lo, hi)
        return valid, clamped

    def __call__(self, points):
        """(M, n_axes) query points in parameter units -> (M,) + trailing, NaN wherever a
        query falls outside the box."""
        points = np.atleast_2d(np.asarray(points, dtype = np.float64))
        out = np.full((points.shape[0],) + self.trailing, np.nan)
        valid, clamped = self._screen(points)
        if valid.any():
            out[valid] = self.spline(clamped[valid])
        return out

    def value_and_grads(self, points):
        """Value AND per-axis first derivative: ((M,) + trailing, (M, n_axes) + trailing),
        with d value / d points[:, a] in slot a. NaN rows out of box in both outputs."""
        points = np.atleast_2d(np.asarray(points, dtype = np.float64))
        out = np.full((points.shape[0],) + self.trailing, np.nan)
        gout = np.full((points.shape[0], self.n_axes) + self.trailing, np.nan)
        valid, clamped = self._screen(points)
        if not valid.any():
            return out, gout
        inside = clamped[valid]
        out[valid] = self.spline(inside)
        for a in range(self.n_axes):
            nu = [0] * self.n_axes
            nu[a] = 1
            gout[valid, a] = self.spline(inside, nu = nu)
        return out, gout


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
        #NOT read here: _spline_cache fills lazily, per spectrum, on first use. np.load on
        #an npz keeps a file handle and reads members on indexing, which is exactly the
        #laziness needed
        self._data = data
        self._spline_cache = {}
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
        #_TensorSpline interpolates over the LEADING axes and carries the rest through as
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
        #this one is small (a few MB) and every query needs it, so unlike the spectra it is
        #built eagerly
        self.theta_spline = _TensorSpline(coeff_dense, theta_knots)

    def h0_from_theta(self, theta, ombh2, omch2):
        """Invert theta_MC_100 -> H0 at each (ombh2, omch2). Returns NaN where the
        requested theta is not reachable inside the grid's H0 range."""
        pts = np.stack([np.atleast_1d(ombh2), np.atleast_1d(omch2)], axis = -1)
        curves = self.theta_spline(pts)
        theta = np.atleast_1d(theta)
        h0 = np.full(theta.shape, np.nan)
        for m in range(theta.size):
            curve = curves[m]
            if not np.all(np.isfinite(curve)):
                continue
            #same round-off tolerance as _TensorSpline's box screen: a theta taken from
            #the edge of the
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

    def spline(self, spectrum):
        """Lazily load one spectrum's coefficient table and cache it as an evaluable
        _TensorSpline.

        Only the spline is cached, not the array it was built from: NdBSpline holds its own
        float64 copy, so keeping the float32 original as well would double an already
        ~2.1 GB per-spectrum footprint for nothing."""
        if spectrum not in self._spline_cache:
            key = f"coeff_{spectrum}"
            if key not in self._data.files:
                raise KeyError(
                    f"the merged CAMB grid at {self.path} has no {spectrum} spectrum - it "
                    f"predates the polarization/lensed-Cl extension. Re-run "
                    f"sampling_chains/camb_grid.sh and "
                    f"merge_camb_grid.py to regenerate it")
            self._spline_cache[spectrum] = _TensorSpline(self._data[key], self.knots)
        return self._spline_cache[spectrum]

    def _points_valid(self, points):
        """The same validity screen _TensorSpline applies: finite coordinates inside the
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
        values = self.spline(spectrum)(points)
        if spectrum in LINEAR_SPECTRA:
            return values
        return np.exp(values)

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
