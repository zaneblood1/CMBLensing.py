"""2D heatmaps of the two analytic N_phi sources on the rfft grid, and of their inverses.

    python -m cmb_lensing.plot_nphi_sources [--nside ... --theta_pix ... --noise ...]

writes cmb_lensing/fisher_output/nphi_source_grids.png.

The 1D version of this comparison azimuthally averages the box matrix onto an ell axis
(_radial_cl_profile) and overplots the Hu & Okamoto spectrum. That hides the whole point of
the disagreement: the box matrix is NOT a function of |l| - scalar_quadratic_estimate
integrates over a SQUARE domain with the FFT's periodic wrap of L - l - while the Hu &
Okamoto spectrum put on the grid by covar_matrix_from_cls is isotropic by construction. This
shows both as they actually enter the phi block C_phi + N_phi.

Eight panels, two rows:
  row 1   N_phi        covariance source | hu_okamoto source | difference | fractional
  row 2   N_phi^-1     the same, for the inverse the Fisher contraction actually uses

Both matrices are multiplied by pix_width**2 on the way out, so they are honest Cls (the
convention covar_matrix_from_cls divides by and _radial_cl_profile undoes). Neither is
divided by NPHI_FAC - qe_noise_matrix already dropped that, so these are the physical N^(0).

Everything is shown in the rfft HALF plane the code actually carries - lx from 0 to the
Nyquist, ly from -Nyquist to +Nyquist, nothing mirrored into the dropped half. Only the ly
axis is fftshifted, so it reads as a signed multipole rather than in fftfreq's wrapped order,
and the aspect is equal so the box matrix's anisotropy is geometrically true.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import numpy as np

from cmb_lensing.util import gen_mesh_grid, gen_ell_grid
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH
from cmb_lensing.fisher_forecast import (add_box_arguments, add_qe_response_argument,
                                         cls_with_qe_response, qe_noise_grid, output_dir,
                                         DEFAULT_RADIAL_MEAN, RADIAL_MEAN_TYPES)


def shifted(matrix):
    """The rfft half plane as stored, with only the ly axis rolled into -Nyq .. +Nyq order.

    The array is shown as the code carries it: lx runs 0 .. Nyquist (the rfft half axis) and
    nothing is mirrored into the dropped half. Only axis 0 is fftshifted, so ly reads as a
    signed multipole instead of fftfreq's wrapped order.
    """
    return np.fft.fftshift(np.asarray(matrix), axes = 0)


def grid_extent(nside, theta_pix):
    """(lx_min, lx_max, ly_min, ly_max) in multipoles for imshow's extent."""
    lx, ly, _ = gen_mesh_grid(nside, theta_pix)
    lx = np.asarray(lx)[0]
    ly = np.fft.fftshift(np.asarray(ly)[:, 0])
    #half a cell of padding on each side so the pixel centres land on their own wavenumber
    half_x = (lx[1] - lx[0]) / 2
    half_y = (ly[1] - ly[0]) / 2
    return (lx[0] - half_x, lx[-1] + half_x, ly[0] - half_y, ly[-1] + half_y)


def draw(axis, values, extent, title, norm, cmap, figure):
    image = axis.imshow(shifted(values), origin = "lower", extent = extent,
                        aspect = "equal", norm = norm, cmap = cmap,
                        interpolation = "nearest")
    axis.set_title(title, fontsize = 9)
    axis.set_xlabel(r"$\ell_x$", fontsize = 8)
    axis.set_ylabel(r"$\ell_y$", fontsize = 8)
    axis.tick_params(labelsize = 7)
    bar = figure.colorbar(image, ax = axis, fraction = 0.046, pad = 0.03)
    bar.ax.tick_params(labelsize = 7)
    return image


def log_norm(*arrays):
    """A LogNorm spanning every positive entry of all the arrays, so panels are comparable."""
    finite = np.concatenate([a[np.isfinite(a) & (a > 0)].ravel() for a in arrays])
    return LogNorm(vmin = np.min(finite), vmax = np.max(finite))


def sym_norm(values, percentile = 99.5):
    """A symmetric SymLogNorm for a signed difference, clipped at a high percentile so one
    diverging pixel at the origin cannot flatten the whole panel."""
    finite = values[np.isfinite(values)]
    high = np.percentile(np.abs(finite), percentile)
    high = high if high > 0 else np.max(np.abs(finite))
    small = np.percentile(np.abs(finite[finite != 0]), 5) if np.any(finite != 0) else high
    return SymLogNorm(linthresh = max(small, high * 1e-4), vmin = -high, vmax = high)


def linear_norm(values, percentile = 99.5):
    """A symmetric LINEAR norm for a fractional difference - it is an O(1) number, and a log
    colour scale on it would hide exactly the structure it exists to show."""
    finite = values[np.isfinite(values)]
    high = float(np.percentile(np.abs(finite), percentile))
    return Normalize(vmin = -high, vmax = high)


def main():
    parser = argparse.ArgumentParser(
        description = "2D comparison of the covariance and hu_okamoto N_phi sources")
    add_box_arguments(parser)
    add_qe_response_argument(parser)
    parser.add_argument("--radial_mean", choices = RADIAL_MEAN_TYPES,
                        default = DEFAULT_RADIAL_MEAN)
    parser.add_argument("--l_cutoff", type = float, default = 10_000)
    parser.add_argument("--out", type = str,
                        default = os.path.join(output_dir(), "nphi_source_grids.png"))
    args = parser.parse_args()

    ell_grid, pix_width = gen_ell_grid(args.nside, args.theta_pix)
    cls = cls_with_qe_response(GROUND_TRUTH, args.qe_response)

    common = dict(filter_tt = None, qe_response = args.qe_response,
                  radial_mean = args.radial_mean)
    print("building N_phi (covariance source: the box's own QE matrix)...")
    covariance = np.asarray(qe_noise_grid(cls, args.nside, pix_width, ell_grid, args.noise,
                                          args.l_knee, args.beam_fwhm, args.l_cutoff,
                                          "covariance", **common)) * pix_width**2
    print("building N_phi (hu_okamoto source: the analytic N^(0) on the grid)...")
    okamoto = np.asarray(qe_noise_grid(cls, args.nside, pix_width, ell_grid, args.noise,
                                       args.l_knee, args.beam_fwhm, args.l_cutoff,
                                       "hu_okamoto", **common)) * pix_width**2

    #the [0, 0] origin carries no multipole (covar_matrix_from_cls zeroes it) and the
    #quadratic estimator's nan_to_num leaves exact zeros wherever its norm diverged; neither
    #belongs in a ratio or on a log colour scale
    usable = (np.asarray(ell_grid) > 0) & (covariance > 0) & (okamoto > 0)
    blank = np.where(usable, 0.0, np.nan)
    covariance = covariance + blank
    okamoto = okamoto + blank

    inverse_covariance = 1.0 / covariance
    inverse_okamoto = 1.0 / okamoto

    difference = covariance - okamoto
    fractional = difference / okamoto
    inverse_difference = inverse_covariance - inverse_okamoto
    inverse_fractional = inverse_difference / inverse_okamoto

    extent = grid_extent(args.nside, args.theta_pix)
    figure, axes = plt.subplots(2, 4, figsize = (15, 12))

    value_norm = log_norm(covariance, okamoto)
    inverse_norm = log_norm(inverse_covariance, inverse_okamoto)

    draw(axes[0, 0], covariance, extent,
         r"$N_\phi$  covariance source (box QE matrix)", value_norm, "viridis", figure)
    draw(axes[0, 1], okamoto, extent,
         r"$N_\phi$  hu_okamoto source, interpolated onto the grid",
         value_norm, "viridis", figure)
    draw(axes[0, 2], difference, extent,
         r"difference  $N^{\rm cov}_\phi - N^{\rm HO}_\phi$",
         sym_norm(difference), "RdBu_r", figure)
    draw(axes[0, 3], fractional, extent,
         r"fractional  $(N^{\rm cov}_\phi - N^{\rm HO}_\phi) / N^{\rm HO}_\phi$",
         linear_norm(fractional), "RdBu_r", figure)

    draw(axes[1, 0], inverse_covariance, extent,
         r"$N_\phi^{-1}$  covariance source", inverse_norm, "magma", figure)
    draw(axes[1, 1], inverse_okamoto, extent,
         r"$N_\phi^{-1}$  hu_okamoto source", inverse_norm, "magma", figure)
    draw(axes[1, 2], inverse_difference, extent,
         r"difference  $1/N^{\rm cov}_\phi - 1/N^{\rm HO}_\phi$",
         sym_norm(inverse_difference), "RdBu_r", figure)
    draw(axes[1, 3], inverse_fractional, extent,
         r"fractional  $(1/N^{\rm cov}_\phi - 1/N^{\rm HO}_\phi) \, N^{\rm HO}_\phi$",
         linear_norm(inverse_fractional), "RdBu_r", figure)

    figure.suptitle(
        f"Quadratic-estimator noise $N_\\phi$ on the rfft grid, in $C_L$ units  |  "
        f"nside {args.nside}, {args.theta_pix:g}', {args.noise:g} uK-arcmin, "
        f"l_knee {args.l_knee:g}, beam {args.beam_fwhm:g}'  |  "
        f"qe_response {args.qe_response}, radial_mean {args.radial_mean}", fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out), exist_ok = True)
    figure.savefig(args.out, dpi = 150)
    plt.close(figure)

    ratio = covariance[usable] / okamoto[usable]
    print(f"\n{np.sum(usable)} usable modes of {covariance.size}")
    print(f"  N^cov / N^HO: median {np.median(ratio):.4f}, "
          f"mean {np.mean(ratio):.4f}, "
          f"5-95% {np.percentile(ratio, 5):.4f} .. {np.percentile(ratio, 95):.4f}, "
          f"min {np.min(ratio):.4f}, max {np.max(ratio):.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
