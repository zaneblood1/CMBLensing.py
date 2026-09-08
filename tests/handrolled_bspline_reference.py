"""The hand-rolled cubic B-spline evaluator camb_grid_interp.py used before the swap to
scipy.interpolate.NdBSpline, frozen here verbatim as a regression reference.

Nothing in cmb_lensing imports this. test_native_5d_interp.py evaluates the production
NdBSpline path against it so the swap can never silently change what the merged 5D CAMB
grid predicts - which matters because that grid is a gitignored ~7.9 GB artifact that
takes 875 slurm jobs to rebuild, so a regression here would be expensive to notice late.

Kept unmodified apart from this docstring and the BOUNDARY_TOL constant being defined
locally rather than imported, so the reference stays independent of the production module.
"""

import numpy as np

#how far outside an axis a query may land, as a fraction of that axis's span, before it is
#treated as out of box rather than snapped to the edge. Covers round-off only
BOUNDARY_TOL = 1e-9


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

