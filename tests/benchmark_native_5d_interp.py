"""Timing + memory comparison of camb_grid_interp.py's production NdBSpline evaluator
against the hand-rolled one it replaced.

    python tests/benchmark_native_5d_interp.py                  #synthetic grid only
    python tests/benchmark_native_5d_interp.py --real           #also the real ~7.9 GB grid
    python tests/benchmark_native_5d_interp.py --real --rgi     #include the slow RGI build

Correctness is asserted in tests/test_native_5d_interp.py; this file only measures. The
--real pass holds one 1.06 GB float32 coefficient table (for the reference implementation,
which reads float32 in place) alongside the 2.11 GB float64 copy NdBSpline makes, so it
peaks around 5.8 GB - do not run it alongside a sampler.
"""

import argparse
import os
import resource
import sys
import time

import numpy as np
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmb_lensing.camb_grid_interp import _TensorSpline
from handrolled_bspline_reference import _eval_bspline, _eval_bspline_grads

GRID_SHAPE = (81, 5, 5, 5, 7)
AXIS_RANGES = [(50.0, 130.0), (2.9, 3.2), (0.93, 0.99), (0.0210, 0.0236), (0.104, 0.136)]
REAL_GRID_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "cmb_lensing", "camb_splines", "camb_grid_spline.npz")
#the batch sizes the sampler actually issues: metropolis_sample_theta evaluates one
#candidate at a time, make_eval_logpdf_batch vmaps a handful
BATCH_SIZES = [1, 8, 64, 256]


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def timeit(fn, repeats = 3):
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    return (time.perf_counter() - start) / repeats, out


def build_synthetic(n_trailing):
    """A synthetic grid + coefficients, prefiltered exactly as merge_camb_grid.py does."""
    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(AXIS_RANGES, GRID_SHAPE)]
    trailing = np.arange(2, 2 + n_trailing, dtype = np.float64)
    mesh = np.meshgrid(*axes, indexing = "ij")
    surface = sum(np.sin(3 * (x - x.mean()) / (x.max() - x.min())) for x in mesh)
    values = surface[..., None] * np.cos(trailing / 60.0) + np.log(trailing)

    block = values.astype(np.float64)
    knots = []
    for axis in range(5):
        spline = make_interp_spline(axes[axis], np.moveaxis(block, axis, 0), k = 3)
        block = np.moveaxis(spline.c, 0, axis)
        knots.append(spline.t)
    return axes, knots, values, block.astype(np.float32)


def interior_points(axes, count, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(a[1], a[-2], count) for a in axes], axis = -1)


def bench_evaluators(axes, knots, coeff, label):
    print(f"\n{label}")
    print(f"  coefficients {coeff.shape} {coeff.dtype} "
          f"({coeff.nbytes / 1024**3:.2f} GB)")

    start = time.perf_counter()
    native = _TensorSpline(coeff, knots)
    build_seconds = time.perf_counter() - start
    print(f"  NdBSpline construction (float32 -> float64 cast): {build_seconds:.2f} s, "
          f"c is {native.spline.c.nbytes / 1024**3:.2f} GB {native.spline.c.dtype}")
    print(f"  peak RSS so far {peak_rss_gb():.2f} GB")

    print(f"  {'M':>5}  {'hand-rolled':>13}  {'NdBSpline':>13}  {'speedup':>8}")
    for count in BATCH_SIZES:
        points = interior_points(axes, count, seed = 100 + count)
        t_hand, v_hand = timeit(lambda: _eval_bspline(coeff, knots, points), repeats = 2)
        t_native, v_native = timeit(lambda: native(points), repeats = 2)
        rel = np.max(np.abs(v_native - v_hand) / np.abs(v_hand))
        print(f"  {count:5d}  {t_hand * 1e3:10.2f} ms  {t_native * 1e3:10.2f} ms  "
              f"{t_hand / t_native:7.2f}x   (max rel diff {rel:.1e})")

    #the derivative path behind CambGrid.spline_param_grads
    points = interior_points(axes, 8, seed = 999)
    t_hand, _ = timeit(lambda: _eval_bspline_grads(coeff, knots, points), repeats = 2)
    t_native, _ = timeit(lambda: native.value_and_grads(points), repeats = 2)
    print(f"  value + 5 gradients (M = 8): hand-rolled {t_hand * 1e3:.2f} ms   "
          f"NdBSpline {t_native * 1e3:.2f} ms   {t_hand / t_native:.2f}x")
    return native


def bench_rgi(axes, values):
    """RegularGridInterpolator(method = "cubic"). Since scipy 1.12 it builds an NdBSpline
    once in its constructor rather than rebuilding splines per call, so the interesting
    number is the CONSTRUCTION time and how it scales with the ell axis."""
    from scipy.interpolate import RegularGridInterpolator
    print("\nRegularGridInterpolator(method = \"cubic\") construction scaling")
    n_ell = values.shape[-1]
    for n_trailing in (1, 4, 16, 64):
        if n_trailing > n_ell:
            break
        start = time.perf_counter()
        rgi = RegularGridInterpolator(tuple(axes), values[..., :n_trailing],
                                      method = "cubic", bounds_error = False,
                                      fill_value = np.nan)
        build = time.perf_counter() - start
        points = interior_points(axes, 8, seed = 7)
        t_call, _ = timeit(lambda: rgi(points), repeats = 2)
        print(f"  trailing {n_trailing:4d}   build {build:8.2f} s   call "
              f"{t_call * 1e3:8.2f} ms")
    print("  (build is linear in the trailing axis; the real grid has 3998 ells, so a "
          "full-size\n   build is on the order of ten minutes PER SPECTRUM - and it "
          "redoes work\n   merge_camb_grid.py has already done, with an iterative "
          "solver instead of a direct one)")


def bench_real():
    from cmb_lensing.camb_grid_interp import load_camb_grid
    if not os.path.exists(REAL_GRID_PATH):
        print(f"\nreal grid not found at {REAL_GRID_PATH} - skipping --real")
        return
    print("\n" + "=" * 78)
    print("REAL merged grid")
    grid = load_camb_grid(REAL_GRID_PATH)
    print(f"  axes {[len(a) for a in grid.axes]}   n_ell {grid.n_ell}")

    #the theta(H0) inversion table CambGrid builds at load time
    rng = np.random.default_rng(11)
    pts2 = np.stack([rng.uniform(grid.axes[3][1], grid.axes[3][-2], 16),
                     rng.uniform(grid.axes[4][1], grid.axes[4][-2], 16)], axis = -1)
    t_hand, _ = timeit(lambda: _eval_bspline(grid.coeff_theta_dense, grid.theta_knots,
                                             pts2), repeats = 5)
    t_native, _ = timeit(lambda: grid.theta_spline(pts2), repeats = 5)
    print(f"\n  theta(H0) table {grid.coeff_theta_dense.shape}: hand-rolled "
          f"{t_hand * 1e3:.3f} ms   NdBSpline {t_native * 1e3:.3f} ms   "
          f"{t_hand / t_native:.2f}x")

    start = time.perf_counter()
    #read the raw table directly: CambGrid.spline deliberately does not keep one around
    coeff = np.load(REAL_GRID_PATH)["coeff_tt"]
    print(f"  loaded coeff_tt in {time.perf_counter() - start:.1f} s   "
          f"peak RSS {peak_rss_gb():.2f} GB")
    bench_evaluators(grid.axes, grid.knots, coeff, "real coeff_tt")
    print(f"\n  final peak RSS {peak_rss_gb():.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action = "store_true",
                        help = "also benchmark the real ~7.9 GB merged grid (~3.5 GB RSS)")
    parser.add_argument("--rgi", action = "store_true",
                        help = "include the RegularGridInterpolator construction scaling "
                               "(slow)")
    parser.add_argument("--n_trailing", type = int, default = 256,
                        help = "stand-in for the ell axis on the synthetic grid")
    opts = parser.parse_args()

    print("=" * 78)
    print(f"synthetic grid, shape {GRID_SHAPE} x {opts.n_trailing}")
    axes, knots, values, coeff = build_synthetic(opts.n_trailing)
    bench_evaluators(axes, knots, coeff, "synthetic")
    if opts.rgi:
        bench_rgi(axes, values)
    if opts.real:
        bench_real()


if __name__ == "__main__":
    main()
