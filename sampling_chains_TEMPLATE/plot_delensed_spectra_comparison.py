"""Unlensed / lensed / delensed temperature spectra, CAMB against the box, one png per noise level.

Series, all band-averaged over the same |l| annuli of the rfft grid:
    CAMB unlensed               scalar_TT
    CAMB lensed                 total_TT
    CAMB iteratively delensed   get_partially_lensed_cls at the Alens_L fisher_forecast's
                                iterative_delensing converges to: the box QE matrix (no
                                NPHI_FAC), unlensed response, the filter's lensed C_TT replaced
                                by the delensed one until it stops moving
    the empirical delensed spectra, the realization-mean `delensed_fid` of the merged
    delensed_covariance.npz in
        delensed_covariance_output_<tag>_nphi_unlensedx2
        delensed_covariance_output_fiducial_wiener_<tag>_<step>_sigma_iterative_delens_qe_gradient
        delensed_covariance_output_map_joint_noise_<tag>
        delensed_covariance_output_<tag>_sample_joint_neff

By default an empirical spectrum is drawn PAIRED: band(C_delensed) / band(C_unlensed) of the
same realizations, times CAMB's unlensed band value, which cancels the realizations' cosmic
variance (delensed_spectrum.py's paired estimator). --naive draws band(C_delensed) as measured.

Top panel: D_l = l (l + 1) C_l / 2 pi. Bottom panel: every series over CAMB unlensed.

    python plot_delensed_spectra_comparison.py [--out_dir <dir>] [--delta_ell <w>] [--naive]
                                              [--ell_max <l>]
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.simulate import covar_matrix_from_cls
from cmb_lensing.sample_lcdm import PARAM_ORDER
from cmb_lensing.fisher_forecast import camb_cls_at_params, iterative_delensing
from cmb_lensing.delensed_covariance import MERGED_NAME
from cmb_lensing.delensed_spectrum import DEFAULT_DELTA_ELL, band_edges, band_average

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE_LEVELS = (("2dot5uk", "2.5"), ("5uk", "5"), ("10uk", "10"))

#(directory glob with the noise tag left open, legend label, color, marker). The second
#pattern leaves the stencil step open and also matches "terative" spelled without its i
EMPIRICAL = (
    ("delensed_covariance_output_{tag}_nphi_unlensedx2",
     "Empirical Delensed N_Phi cfl_tt --> cf_tt", "C0", "o"),
    ("delensed_covariance_output_fiducial_wiener_{tag}_0dot*_sigma_*terative_delens_qe_gradient",
     "Empirical Delensed Iteratively Delensed N_phi Response = Gradient", "C1", "s"),
    ("delensed_covariance_output_map_joint_noise_{tag}",
     "Empirical Delensed N_Phi = MAP_Joint Noise", "C2", "^"),
    ("delensed_covariance_output_{tag}_sample_joint_neff",
     "Empirical Delensed N_Phi = Sample_Joint Effective Noise", "C5", "D"),
)
BOX_KEYS = ("nside", "theta_pix", "noise_level", "l_knee")


def load_empirical(tag):
    """[(label, color, marker, merged npz)] of the merged files that exist for this tag."""
    found = []
    for pattern, label, color, marker in EMPIRICAL:
        matches = sorted(glob.glob(os.path.join(HERE, pattern.format(tag = tag))))
        paths = [os.path.join(match, MERGED_NAME) for match in matches]
        paths = [path for path in paths if os.path.exists(path)]
        if not paths:
            print(f"{tag}: no {pattern.format(tag = tag)}/{MERGED_NAME}, series left out")
            continue
        if len(paths) > 1:
            print(f"{tag}: several directories match {pattern.format(tag = tag)}, using "
                  f"{paths[0]}")
        print(f"{tag}: {label}  <-  {paths[0]}")
        found.append((label, color, marker, np.load(paths[0], allow_pickle = True)))
    return found


def plot_noise_level(tag, label, out_dir, delta_ell, naive, ell_max):
    empirical = load_empirical(tag)
    if not empirical:
        print(f"{label} uK: no merged delensed covariance found, skipped")
        return
    first = empirical[0][3]
    for _, _, _, merged in empirical[1:]:
        for key in BOX_KEYS:
            if float(merged[key]) != float(first[key]):
                raise ValueError(f"{label} uK: the merged files disagree on {key}")
        if not np.allclose(merged["params"], first["params"], rtol = 1e-12, atol = 0):
            raise ValueError(f"{label} uK: the merged files sit at different cosmologies")

    nside, theta_pix = int(first["nside"]), float(first["theta_pix"])
    noise_level, l_knee = float(first["noise_level"]), float(first["l_knee"])
    params = {name: float(first["params"][i]) for i, name in enumerate(PARAM_ORDER)}
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))
    edges = band_edges(ell_grid, delta_ell)

    def band(matrix):
        centres, values, _ = band_average(matrix, ell_grid, weights, edges)
        return centres, values

    def camb_band(cl):
        #CAMB's C_l on the grid exactly as the merge puts it there (C_l / pix_width^2)
        ells = jnp.arange(2, 2 + cl.shape[0]).astype(jnp.float64)
        return band(np.asarray(covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl,
                                                     origin_value = 0)))[1]

    cls = camb_cls_at_params(params)
    #the converged delensed C_TT of the iteration: box QE matrix ("covariance", which carries
    #no NPHI_FAC), unlensed response, beam 0 as the data are simulated
    delensed_tt = iterative_delensing(params, cls, nside, pix_width, ell_grid, noise_level,
                                      l_knee, 0.0, 10_000, "covariance", verbose = False,
                                      qe_response = "unlensed")[2]

    centres = band(np.asarray(first["unlensed_fid"]))[0]
    camb_unlensed = camb_band(cls["scalar_TT"])
    #grid units C_l / pix_width^2 -> D_l
    to_dl = centres * (centres + 1) / (2 * np.pi) * float(pix_width)**2
    series = [("CAMB unlensed", "0.2", "-", None, camb_unlensed),
              ("CAMB lensed", "C3", "-", None, camb_band(cls["total_TT"])),
              ("CAMB iteratively delensed", "C4", "--", None, camb_band(delensed_tt))]
    for name, color, marker, merged in empirical:
        measured = band(np.asarray(merged["delensed_fid"]))[1]
        if not naive:
            measured = measured / band(np.asarray(merged["unlensed_fid"]))[1] * camb_unlensed
        series.append((name, color, "-", marker, measured))

    figure, (top, bottom) = plt.subplots(2, 1, figsize = (11, 10), sharex = True,
                                         gridspec_kw = dict(height_ratios = [3, 2]))
    #past CAMB's own range the model spectra are a log-log extrapolation and the box's lensed
    #power aliases at the grid's corner, which swamps the ratio panel
    shown = centres <= ell_max if ell_max > 0 else np.ones(len(centres), dtype = bool)
    centres, camb_unlensed, to_dl = centres[shown], camb_unlensed[shown], to_dl[shown]
    for name, color, linestyle, marker, values in series:
        values = values[shown]
        style = dict(color = color, linestyle = linestyle, marker = marker, markersize = 3,
                     linewidth = 1.3)
        top.plot(centres, values * to_dl, label = name, **style)
        bottom.plot(centres, values / camb_unlensed, label = name, **style)
        ratio = values / camb_unlensed
        print(f"{label} uK  {name[:66]:66s} / CAMB unlensed: min {np.min(ratio):.4f} "
              f"max {np.max(ratio):.4f}")

    top.set_yscale("log")
    top.set_ylabel(r"$D_\ell^{TT} = \ell(\ell+1)\,C_\ell / 2\pi$  [$\mu$K$^2$]")
    top.legend(fontsize = 8)
    bottom.set_xlabel(r"$\ell$")
    bottom.set_ylabel(r"$C_\ell\ /\ C_\ell^{\rm unlensed,\ CAMB}$")
    figure.suptitle(f"Temperature spectra | nside {nside}, {theta_pix:g}', {label} uK-arcmin | "
                    f"bands of {delta_ell:g}, empirical "
                    + ("as measured" if naive else "paired with its own unlensed"),
                    fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"delensed_spectra_comparison_{tag}.png")
    figure.savefig(path, dpi = 150)
    plt.close(figure)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    parser.add_argument("--out_dir", default = HERE)
    parser.add_argument("--delta_ell", type = float, default = DEFAULT_DELTA_ELL,
                        help = "width of the |l| bands every series is averaged in")
    parser.add_argument("--naive", action = "store_true",
                        help = "draw the empirical band(C_delensed) as measured instead of "
                               "pairing it with the same realizations' unlensed spectrum")
    parser.add_argument("--ell_max", type = float, default = 4000.0,
                        help = "largest band centre drawn (default CAMB's own range); 0 "
                               "draws every band out to the grid's corner")
    args = parser.parse_args()
    for tag, label in NOISE_LEVELS:
        plot_noise_level(tag, label, args.out_dir, args.delta_ell, args.naive, args.ell_max)


if __name__ == "__main__":
    main()
