"""Evaluate the 5D CAMB grid spline at an arbitrary LCDM parameter vector.

This is the five-parameter counterpart of camb_grid_generation/precompute_camb_1d.py,
which puts a CubicSpline through lnCl along one parameter with the other four pinned.
Here lnCl is known on a full tensor-product grid over (H0, logA, ns, ombh2, omch2), and
the interpolant is the tensor-product cubic spline through it.

How the spline is built (spline_coefficients). A tensor-product spline is separable:
    s(x0, ..., x4) = sum_{i0..i4} c[i0, ..., i4] B_i0(x0) ... B_i4(x4)
so asking it to pass through every grid value is the 1D interpolation problem solved
along one axis at a time - a make_interp_spline (not-a-knot ends, the same as
CubicSpline's default) along H0, then along logA on the result, and so on. What comes
out is the coefficient array c and the knot vector of each axis, which is exactly what
scipy.interpolate.NdBSpline evaluates. merge_camb_grid.py does this once for every
spectrum and stores c and the knots, so loading the grid is instant.

    from cmb_lensing.camb_grid_interp import load_camb_grid_predictors
    predictors = load_camb_grid_predictors("<path>/camb_grid_spline.npz")
    predictors.tt(params_batch)  #(M, 5) in PARAM_ORDER -> (M, n_ell) Cls

Each coefficient table is loaded lazily, on first use of its spectrum (~1.1 GB read,
~2.1 GB resident since NdBSpline casts to float64). Unlensed BB is identically zero at
r = 0 (bb_is_zero), so its predictor returns zeros. te_rho = Cl_TE / sqrt(Cl_TT Cl_EE)
crosses zero, so it is splined raw instead of in log space.

theta_MC_100 vs H0. The sampler works in theta_MC_100, but the grid is laid out in H0
because a rectangular theta box has corners CAMB cannot solve. theta depends only on
(H0, ombh2, omch2), so the merge stores it as a 3D table; here it gets the same
tensor-product spline, and a query is converted by evaluating theta(H0) along a dense
H0 line at its (ombh2, omch2) and inverting that monotone curve with np.interp.

Out-of-box queries return NaN, which the sampler turns into a rejected proposal.
"""

import os
from collections import namedtuple
import numpy as np
from scipy.interpolate import make_interp_spline, NdBSpline
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

#the sampler's parameter ordering (params_batch columns)
from cmb_lensing.constants import PARAM_INDEX
#the grid's axis ordering - PARAM_ORDER with the acoustic axis in H0 instead of theta
GRID_AXES = ["H0", "logA", "ns", "ombh2", "omch2"]
GRID_INDEX = {name: i for i, name in enumerate(GRID_AXES)}

#every spectrum the grid serves, one predictor each
SPECTRA = ["tt", "ee", "bb", "pp", "tt_lensed", "ee_lensed", "bb_lensed", "te_rho"]
#spectra splined in their raw values rather than lnCl (sign-changing quantities)
LINEAR_SPECTRA = {"te_rho"}
GridPredictors = namedtuple("GridPredictors", SPECTRA)

#points on the dense H0 line the theta(H0) curve is inverted on
N_THETA_DENSE = 2000

#how far outside an axis (as a fraction of its span) a query may land and still be
#snapped onto the edge:
BOUNDARY_TOL = 1e-9


def spline_coefficients(axes, values):
    """Cubic B-spline coefficients and knots of the tensor-product spline through
    values[i0, i1, ...] on the grid axes. Axes of values beyond len(axes) (ell) are
    carried along untouched."""
    coeff = values
    knots = []
    for a in range(len(axes)):
        spline = make_interp_spline(axes[a], coeff, k = 3, axis = a)
        #make_interp_spline puts the interpolated axis first in .c; put it back
        coeff = np.moveaxis(spline.c, 0, a)
        knots.append(spline.t)
    return coeff, knots


def screen(knots, points):
    """(valid, clipped) for (M, n_axes) points: valid marks the rows inside the box
    (within BOUNDARY_TOL), clipped has every coordinate clipped onto the box."""
    valid = np.all(np.isfinite(points), axis = 1)
    clipped = points.copy()
    for a in range(points.shape[1]):
        #the box runs from the 4th knot to the 4th-last one: the not-a-knot cubic spline
        #repeats each end knot 4 times
        lo = knots[a][3]
        hi = knots[a][-4]
        tol = BOUNDARY_TOL * (hi - lo)
        valid &= (points[:, a] >= lo - tol) & (points[:, a] <= hi + tol)
        clipped[:, a] = np.clip(points[:, a], lo, hi)
    return valid, clipped


def evaluate_spline(spline, points):
    """NdBSpline at (M, n_axes) points -> (M,) + trailing, with NaN rows for points
    outside the box. Points within BOUNDARY_TOL of an edge are clipped onto it."""
    points = np.atleast_2d(np.asarray(points, dtype = np.float64))
    valid, clipped = screen(spline.t, points)
    trailing = spline.c.shape[points.shape[1]:]
    out = np.full((points.shape[0],) + trailing, np.nan)
    if valid.any():
        out[valid] = spline(clipped[valid])
    return out


class CambGrid:
    """The merged 5D spline, plus the theta_MC_100 -> H0 conversion it needs."""

    def __init__(self, path):
        self.path = path
        #np.load on an npz reads a member only when it is indexed, so the coefficient
        #tables stay on disk until spline() asks for them
        self.data = np.load(path)
        self.bb_is_zero = bool(self.data["bb_is_zero"])
        self.ells = self.data["ells"]
        self.n_ell = self.ells.size
        self.axes = [self.data[f"axis_{name}"] for name in GRID_AXES]
        self.knots = [self.data[f"knots_{name}"] for name in GRID_AXES]
        self.spline_cache = {}

        #theta_grid[H0, ombh2, omch2] gets the same tensor-product spline as the spectra
        h0_axis = self.axes[GRID_INDEX["H0"]]
        ombh2_axis = self.axes[GRID_INDEX["ombh2"]]
        omch2_axis = self.axes[GRID_INDEX["omch2"]]
        theta_coeff, theta_knots = spline_coefficients([h0_axis, ombh2_axis, omch2_axis],
                                                       self.data["theta_grid"])
        self.theta_spline = NdBSpline(tuple(theta_knots), theta_coeff, 3)
        self.h0_dense = np.linspace(h0_axis[0], h0_axis[-1], N_THETA_DENSE)

    def h0_from_theta(self, theta, ombh2, omch2):
        """H0 giving theta_MC_100 at each (ombh2, omch2); NaN where theta is not reached
        inside the grid's H0 range."""
        theta = np.atleast_1d(theta)
        ombh2 = np.atleast_1d(ombh2)
        omch2 = np.atleast_1d(omch2)
        h0 = np.full(theta.shape, np.nan)
        for m in range(theta.size):
            line = np.stack([self.h0_dense,
                             np.full(N_THETA_DENSE, ombh2[m]),
                             np.full(N_THETA_DENSE, omch2[m])], axis = -1)
            #theta(H0) at this (ombh2, omch2), increasing in H0
            curve = evaluate_spline(self.theta_spline, line)
            if not np.all(np.isfinite(curve)):
                continue
            tol = BOUNDARY_TOL * (curve[-1] - curve[0])
            if theta[m] < curve[0] - tol or theta[m] > curve[-1] + tol:
                continue
            h0[m] = np.interp(np.clip(theta[m], curve[0], curve[-1]), curve, self.h0_dense)
        return h0

    def grid_points(self, params_batch):
        """(M, 5) in PARAM_ORDER -> (M, 5) in GRID_AXES, i.e. theta_MC_100 replaced by
        the H0 that produces it at this ombh2 / omch2."""
        pb = np.atleast_2d(np.asarray(params_batch, dtype = np.float64))
        theta = pb[:, PARAM_INDEX["theta_MC_100"]]
        ombh2 = pb[:, PARAM_INDEX["ombh2"]]
        omch2 = pb[:, PARAM_INDEX["omch2"]]
        points = np.zeros_like(pb)
        for name in GRID_AXES:
            if name == "H0":
                points[:, GRID_INDEX["H0"]] = self.h0_from_theta(theta, ombh2, omch2)
            else:
                points[:, GRID_INDEX[name]] = pb[:, PARAM_INDEX[name]]
        return points

    def spline(self, spectrum):
        """One spectrum's NdBSpline, built from its stored coefficients on first use."""
        if spectrum not in self.spline_cache:
            coeff = self.data[f"coeff_{spectrum}"]
            self.spline_cache[spectrum] = NdBSpline(tuple(self.knots), coeff, 3)
        return self.spline_cache[spectrum]

    def cl(self, spectrum, params_batch):
        """(M, 5) in PARAM_ORDER -> (M, n_ell): Cls, or the ratio itself for te_rho. NaN
        rows outside the box or at an unreachable theta_MC_100."""
        points = self.grid_points(params_batch)
        if spectrum == "bb" and self.bb_is_zero:
            #zeros, with the same NaN rows as any other spectrum
            out = np.zeros((points.shape[0], self.n_ell))
            valid, _ = screen(self.knots, points)
            out[~valid] = np.nan
            return out
        values = evaluate_spline(self.spline(spectrum), points)
        if spectrum in LINEAR_SPECTRA:
            return values
        return np.exp(values)


_loaded_grids = {}

def load_camb_grid(path):
    """One CambGrid per file per process."""
    path = os.path.abspath(path)
    if path not in _loaded_grids:
        _loaded_grids[path] = CambGrid(path)
    return _loaded_grids[path]

_loaded_predictors = {}

def load_camb_grid_predictors(path):
    """GridPredictors of jit-safe functions f(params_batch) -> (M, n_ell), via
    jax.pure_callback. Cached per path: the sampler passes them as static jit arguments,
    so they must be the same function objects on every call."""
    path = os.path.abspath(path)
    if path in _loaded_predictors:
        return _loaded_predictors[path]
    grid = load_camb_grid(path)

    def make_predictor(spectrum):
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
    _loaded_predictors[path] = predictors
    return predictors
