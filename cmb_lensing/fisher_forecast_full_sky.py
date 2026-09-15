"""The textbook full-sky Fisher forecast, "full_sky":

    F_ij = sum_l (2l+1)/2 f_sky Tr[C_l^-1 dC_l/dtheta_i C_l^-1 dC_l/dtheta_j]

Same likelihood as fisher_forecast's "blocks" and the same two powers of C^-1; what
changes is the mode counting. "blocks" weights each rfft entry by its real degrees of
freedom w_k, which sum to exactly nside^2; this weights each multipole by the 2l+1 modes
a sky of area 4 pi f_sky would carry, with f_sky = (nside * pix_width)^2 / (4 pi)
(sky_fraction; the mask is all ones, so the patch IS the box). C_l is assembled as an
honest matrix over the blocks and inverted per multipole, so a cross-spectrum would slot
into the off-diagonal - this codebase carries none, so it is diagonal today and the trace
reduces to sum_blocks dln(C) dln(C).

`ell_source` picks which spectra go in, and the DEFAULT is "grid": the sum runs over the
DISTINCT |l| values the rfft grid carries (grid_ell_axis), the grid's own irregular
spacing supplies the per-multipole width, and the blocks are fisher_forecast's
covariance_stencil - azimuthally reduced onto that axis. So the Cls contracted here are
exactly the ones the SAMPLER sees: CAMB log-log interpolated onto ell_grid by
covar_matrix_from_cls, times the same mask and beam, plus the same C_n, N_phi and
delensing residual. "blocks" and this are then contractions of ONE identical set of
matrices and only the mode counting separates them, at no extra CAMB calls. The one
approximation is that a 1D ell sum cannot carry an anisotropic covariance: everything
covar_matrix_from_cls builds is isotropic to 3e-8, but N_phi (FFT convolutions on a
SQUARE box) scatters up to 78% within one |l|. Measured cost at nside 64 / 5', with mode
counting held fixed: f block exact to 5e-14, phi block 1.8%, Fisher diagonal
1.4%/0.02%/0.03%, marginalized sigmas 0.7%/0.007%/0.02%. Every run prints the per-block
scatter.

ell_source = "camb" instead uses the raw CAMB spectra on their native integer multipoles,
ells 2..CAMB_LMAX-1 - free of the box's resolution, so that is the version to quote
against the literature, but it is not the model the sampler runs on. There N_phi has to
cross bases (it only ever exists as a 2D array, since scalar_quadratic_estimate is built
from FFT convolutions), so _radial_cl_profile azimuthally averages it onto the ell axis
and log-log EXTRAPOLATES past the grid's largest mode, with a printed warning when it does.

`ell_min` / `ell_max` narrow the range (default: the source's own edges) and `delta_ell`
bins the spectra into bands on top of the source's spacing. Binning is not a speed-up -
the unbinned sum is already cheap - it pools C and dC before contracting them, which is
exact only where the spectra are linear across a band, so it answers "how much do the
features I resolve at this resolution carry" and LOSES information as delta_ell grows
past the acoustic width. See _fisher_full_sky and _ell_binner.

EXPECT IT TO DISAGREE WITH "blocks" even on identical blocks: nside^2 real DOF spread
over a SQUARE in (lx, ly) is not sum_l (2l+1) f_sky over a DISK. Over the grid's full
range that disk holds 1.595x the square's modes (sqrt = 1.263, against measured
sqrt(F_full_sky / F_blocks) = 1.10/1.42/1.27); truncated at the box's Nyquist it holds
pi/4 (measured 0.7861 vs 0.7854, sqrt(F) ratios 0.96/0.79/0.88). Both totals are printed.

Usage:
    python -m cmb_lensing.fisher_forecast_full_sky                      #the grid's ells
    python -m cmb_lensing.fisher_forecast_full_sky --ell_source camb
    python -m cmb_lensing.fisher_forecast_full_sky --delta_ell 50 --ell_max 2160

Writes fisher_matrix_full_sky.png, covariance_matrix_full_sky.png,
correlation_matrix_full_sky.png and fisher_full_sky.npz into cmb_lensing/fisher_output/.
"""

import argparse

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.simulate import noise_cls, beam_cls
from cmb_lensing.constants import DEFAULT_MAX_ELL, ARCMIN_PER_DEGREE
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, CAMB_LMAX
from cmb_lensing.fisher_forecast import (SPECTRA_MODES, camb_cls_at_params,
                                         cls_with_qe_response, DEFAULT_QE_RESPONSE,
                                         delensed_cls_at_params, sampled_names_and_steps,
                                         frozen_reconstruction, covariance_stencil,
                                         covariance_from_fisher, step_stability,
                                         report_sigmas, report_ceiling_ratio,
                                         save_outputs, add_box_arguments,
                                         add_spectra_arguments, sampled_from_args,
                                         run_config, _radial_cl_profile)


METHOD = "full_sky"
SUFFIX = "_full_sky"
LABEL = (r"full sky:  $F_{ij} = \sum_\ell \frac{2\ell+1}{2} f_{sky}\,"
         r"\mathrm{Tr}[C_\ell^{-1}\partial_i C_\ell C_\ell^{-1}\partial_j C_\ell]$")

#where the full-sky sum gets its multipoles and its spectra from:
#  "grid" (DEFAULT) the DISTINCT |l| values the rfft grid actually carries, and the blocks
#         covariance_blocks builds on that grid - i.e. the very Cls the sampler sees, log-log
#         interpolated onto ell_grid by covar_matrix_from_cls, with the same mask, beam,
#         noise, N_phi and delensing residual. delta_ell comes from the grid's own spacing
#  "camb" the raw CAMB spectra on their native integer multipoles, 2 .. CAMB_LMAX - 1
ELL_SOURCES = ("grid", "camb")

#the CAMB source's multipole support: camb_cls_at_params puts every spectrum on ells
#2 .. CAMB_LMAX - 1, so CAMB_ELL_MAX is the largest value ell_max may take there - anything
#past it would be a log-log extrapolation of the spectra rather than a CAMB value. The grid
#source has no such cap: past CAMB's last multipole covar_matrix_from_cls extrapolates, and
#reproducing that is the whole point of reading the spectra off the grid
CAMB_ELL_MIN = 2
CAMB_ELL_MAX = CAMB_LMAX - 1


def sky_fraction(nside, theta_pix):
    """f_sky for this flat-sky box: its solid angle as a fraction of the full sphere.

    The box is nside * theta_pix on a side, so its area is (nside * pix_width)**2 steradians
    and f_sky = area / (4 pi). There is no extra mask factor - simulate.load_sim forces the
    mask to all ones, so the observed patch IS the box.

    At nside 128 / 2.5' that is 5.33 deg on a side and f_sky = 6.86e-4.
    """
    pix_width = np.deg2rad(theta_pix / ARCMIN_PER_DEGREE)
    return float((nside * pix_width)**2 / (4 * np.pi))


def grid_ell_axis(ell_grid, weights, ell_min = None, ell_max = None):
    """The DISTINCT |l| values the rfft grid carries, with a per-value width and a reducer.

    This is what ell_source = "grid" sums over. The grid holds |l| = sqrt(lx^2 + ly^2) at
    every rfft entry, so its distinct values are the fundamental 2 pi / L times sqrt(i^2 +
    j^2) over the integer mode numbers - irregularly spaced, sparse near ell_min and dense
    at high ell. Summing over those and reading the spectra off the grid means the forecast
    sees the SAME Cls the sampler does: covar_matrix_from_cls's log-log interpolation onto
    this grid, with the same mask, beam, noise and N_phi covariance_blocks builds.

    Returns (ells, measure, reduce_fn, scatter_fn):

      ells       the distinct values, ascending
      measure    the width each value stands for - the midpoint interval
                 (l_{i+1} - l_{i-1}) / 2, with the two ends extended symmetrically. This is
                 the "spacing on the ell grid" that plays delta_ell's role: sum_l (2l+1)/2 *
                 measure_l is the trapezoidal form of the integral the unit-spaced CAMB sum
                 approximates, so both sources count modes on the same footing. It
                 telescopes to exactly ell_max - ell_min plus the two half-end extensions
      reduce_fn  averages any rfft-grid array onto `ells`, weighting each entry by its real
                 DOF w_k. It needs no extrapolation at all, unlike _radial_cl_profile,
                 because it never leaves the grid
      scatter_fn the DOF-weighted relative standard deviation WITHIN each distinct |l|, i.e.
                 how much reduce_fn threw away

    THE REDUCTION IS EXACT ONLY FOR AN ISOTROPIC OPERATOR, and one block here is not one.
    Anything covar_matrix_from_cls builds is a pure function of |l| (measured within-|l|
    scatter 3e-8, pure round-off), so the signal, noise, mask and beam survive the average
    untouched. But N_phi comes out of scalar_quadratic_estimate, whose FFT convolutions see
    the SQUARE box: at nside 64 / 5' its within-|l| scatter reaches 78% (median 0.2%), and
    the phi block inherits it. Averaging is unavoidable - a sum over a 1D multipole axis
    cannot represent an anisotropic covariance at all - but it is the one place this path
    departs from the matrices "blocks" contracts. _grid_cl_stencil prints the worst
    per-block scatter so the size of the approximation is visible in every run.

    HOW MUCH IT COSTS, measured at nside 64 / 5' / 5 uK-arcmin, spectra = "ceiling", by
    handing _fisher_full_sky the grid's OWN DOF weights so that mode counting drops out and
    only the reduction is left: the isotropic f block reproduces _fisher_from_blocks to
    5e-14, the phi block alone to 1.8%, and together the Fisher diagonal moves 1.4% / 0.02%
    / 0.03% and the marginalized sigmas 0.7% / 0.007% / 0.02% (omch2 / theta_MC_100 / logA).
    A single near-zero off-diagonal entry moves 17%, which is what a relative comparison of
    a number consistent with zero looks like - do not read it as the size of the effect. The
    5e-14 is what pins the reduction machinery itself.

    Entries are grouped by |l| / fundamental rounded to 1e-6, not by exact float equality:
    (0, 5) and (3, 4) are the same multipole but 25 f^2 and 9 f^2 + 16 f^2 need not agree in
    the last bit, and splitting them would hand one of the pair a near-zero `measure`.
    Distinct sqrt(i^2 + j^2) values are separated by far more than that, so nothing that is
    genuinely distinct gets merged.
    """
    flat_ell = np.asarray(ell_grid).ravel()
    flat_dof = np.asarray(jnp.real(weights)).ravel()

    #the [0, 0] origin carries no ell and is zeroed by origin_value = 0 in every block
    keep = flat_ell > 0
    if ell_min is not None:
        keep = keep & (flat_ell >= ell_min)
    if ell_max is not None:
        keep = keep & (flat_ell <= ell_max)
    if not np.any(keep):
        raise ValueError(f"no rfft-grid modes survive ell_min = {ell_min}, "
                         f"ell_max = {ell_max}; the grid's ell run "
                         f"[{flat_ell[flat_ell > 0].min():.0f}, {flat_ell.max():.0f}]")

    fundamental = float(np.min(flat_ell[flat_ell > 0]))
    _, inverse = np.unique(np.round(flat_ell[keep] / fundamental, 6),
                           return_inverse = True)

    dof = np.bincount(inverse, weights = flat_dof[keep])
    ells = np.bincount(inverse, weights = (flat_dof * flat_ell)[keep]) / dof

    #midpoint intervals, with the first and last value extended by half its one-sided gap
    if len(ells) == 1:
        measure = np.array([fundamental])
    else:
        edges = np.empty(len(ells) + 1)
        edges[1:-1] = 0.5 * (ells[1:] + ells[:-1])
        edges[0] = ells[0] - 0.5 * (ells[1] - ells[0])
        edges[-1] = ells[-1] + 0.5 * (ells[-1] - ells[-2])
        measure = np.diff(edges)

    def reduce_fn(matrix):
        flat = np.asarray(matrix).ravel()
        return np.bincount(inverse, weights = (flat_dof * flat)[keep]) / dof

    def scatter_fn(matrix):
        flat = np.asarray(matrix).ravel()
        mean = reduce_fn(flat)
        second = np.bincount(inverse, weights = (flat_dof * flat**2)[keep]) / dof
        spread = np.sqrt(np.maximum(second - mean**2, 0.0))
        return np.where(mean != 0, spread / np.abs(np.where(mean != 0, mean, 1.0)), 0.0)

    return ells, measure, reduce_fn, scatter_fn


def cl_blocks(cls, spectra, ell_axis, noise_level, l_knee, beam_fwhm, nphi_cl):
    """1D analogue of fisher_forecast.covariance_blocks: the same blocks as functions of ell.

    Same block names, same contents and the same freezing conventions - the one difference
    is that these are the CAMB spectra themselves rather than covar_matrix_from_cls's
    interpolation of them onto the rfft grid, so nothing is rescaled by 1/pix_width**2 and
    nothing is smeared across the grid's annuli. `nphi_cl` is the frozen quadratic-estimate
    noise already brought onto `ell_axis` by _radial_cl_profile. For spectra = "delensed"
    the cls dict must carry "delensed_TT" - delensed_cls_at_params adds it, from CAMB's
    get_partially_lensed_cls at the frozen per-L Alens the stencil computed.

    The beam / noise convention is copied verbatim from covariance_blocks (and so from
    load_sim): the signal carries B_l**2 while noise_cls returns a BEAM-DECONVOLVED N_l.
    Those two only compose consistently at beam_fwhm = 0, which is the default everywhere
    in this codebase; it is mirrored rather than quietly fixed so that the full-sky path
    describes the same model the other two do.
    """
    camb_ells = np.arange(2, 2 + cls["total_TT"].shape[0], dtype = np.float64)
    phi_ells = np.arange(2, 2 + cls["phi"].shape[0], dtype = np.float64)

    def on_axis(cl, source_ells):
        #log-log, matching covar_matrix_from_cls. on the default axis this is the identity:
        #CAMB, the noise and the requested ells all live on 2 .. CAMB_LMAX - 1
        return np.asarray(jnp.exp(jnp.interp(jnp.log(ell_axis), jnp.log(source_ells),
                                             jnp.log(cl), left = "extrapolate",
                                             right = "extrapolate")))

    lmax_prime = min(DEFAULT_MAX_ELL, CAMB_LMAX)
    n_tt, _, _, _ = noise_cls(lmax_prime, noise_level, beam_fwhm = beam_fwhm,
                              l_knee = l_knee)
    noise = on_axis(n_tt, np.arange(2, lmax_prime, dtype = np.float64))
    beam_squared = np.asarray(beam_cls(beam_fwhm, ell_axis))

    lensed = on_axis(cls["total_TT"], camb_ells)
    unlensed = on_axis(cls["scalar_TT"], camb_ells)
    phi_block = on_axis(cls["phi"], phi_ells) + nphi_cl

    if spectra == "delensed":
        if "delensed_TT" not in cls:
            raise ValueError("spectra = 'delensed' needs cls['delensed_TT']; build the dict "
                             "with delensed_cls_at_params at the frozen Alens_L")
        return {"f_delensed": on_axis(cls["delensed_TT"], camb_ells) + noise,
                "phi": phi_block}

    if spectra == "ceiling":
        return {"f_unlensed": unlensed + noise,
                "phi": phi_block}

    source = lensed if spectra == "lensed" else unlensed
    return {"TT": beam_squared * source + noise, "PP": phi_block}


def _ell_binner(ell_axis, f_sky, delta_ell, measure = None):
    """(bin_ells, mode_weights, bin_fn) - the (2l + 1) mode counting for the full-sky sum.

    `measure` is the width in multipoles each entry of `ell_axis` stands for. It is 1 for
    the CAMB source, whose axis is every integer multipole; for the grid source it is
    grid_ell_axis's midpoint spacing, since the grid's distinct |l| values are irregularly
    spaced and each has to be weighted by the stretch of ell it represents. Either way the
    per-multipole weight is

        w_l = (2 l + 1) / 2 * f_sky * measure_l

    which for measure = 1 is exactly the prefactor in
    F_ij = sum_l (2l+1)/2 f_sky Tr[...], and otherwise is its trapezoidal form - so the two
    sources count modes on the same footing and their answers are comparable.

    With `delta_ell = None` there is no further binning: every entry of the axis is its own
    band and `bin_fn` is the identity.

    With a band width the modes in a band are pooled: the weight becomes the sum of the
    w_l in it - the band's share of the sphere's modes - and a spectrum is replaced by its
    w_l-weighted mean across the band, which is the estimator a real bandpower analysis
    forms.

    That pooling is EXACT only where the spectra are linear across a band: averaging C and
    dC before contracting them is not the same as contracting first and then summing, and
    the difference is genuine information the binning throws away. So binning is a way to
    ask "how much do the features I resolve at this resolution carry", not a speed-up - the
    unbinned sum is already cheap. Expect the forecast to loosen once delta_ell grows past
    the width of the acoustic features (~50 in TT).
    """
    if measure is None:
        measure = np.ones_like(ell_axis)
    weights = 0.5 * f_sky * (2 * ell_axis + 1) * measure

    if delta_ell is None:
        return ell_axis, weights, (lambda values: values)

    if delta_ell < 1:
        raise ValueError(f"delta_ell must be at least 1 multipole, got {delta_ell}")

    index = ((ell_axis - ell_axis[0]) // delta_ell).astype(int)
    n_bin = int(index.max()) + 1
    norm = np.bincount(index, weights = weights, minlength = n_bin)
    #a band can be empty on the grid source, whose axis has gaps; carrying an empty band
    #through would divide by zero and contribute nothing anyway
    filled = norm > 0
    bin_ells = (np.bincount(index, weights = weights * ell_axis, minlength = n_bin)
                / np.where(filled, norm, 1.0))[filled]

    def bin_fn(values):
        return (np.bincount(index, weights = weights * values, minlength = n_bin)
                / np.where(filled, norm, 1.0))[filled]

    return bin_ells, norm[filled], bin_fn


def _fisher_full_sky(cl_plus, cl_minus, cl_fid, steps, mode_weights, bin_fn):
    """F_ij = sum_l (2l+1)/2 f_sky Tr[C_l^-1 dC_l/dtheta_i C_l^-1 dC_l/dtheta_j].

    The textbook full-sky Gaussian Fisher (Tegmark, Taylor & Heavens 1997 in the isotropic,
    all-sky limit) scaled to a cut sky by f_sky. C_l is the matrix over the blocks being
    combined - (C_TT, C_phiphi), or whichever pair the chosen `spectra` mode builds - and it
    is assembled as an honest matrix and inverted per multipole, so a cross-spectrum could
    be dropped into the off-diagonal without touching this contraction. This codebase
    carries none (simulate.py keeps only column 0 of the lens-potential Cls, so C_l^{T phi}
    is discarded), so C_l is diagonal today and the trace collapses to
    sum_blocks dln(C) dln(C) - the same quantity _fisher_from_blocks contracts, but with
    full-sky (2l + 1) f_sky mode counting rather than the rfft grid's per-entry DOF weights.

    `bin_fn` is applied to every spectrum and every derivative before the contraction, so
    with a band width set this is the bandpower Fisher of the binned data vector; see
    _ell_binner for what the binning costs.

    VALIDATION, the analogue of the (nside^2 - 1) / 2 check _fisher_from_blocks has. A
    single block exactly proportional to As has dln(C)/dlogA identically 1, so it must
    return exactly F_logA,logA = f_sky * sum_l (2l + 1) / 2 * measure_l over the summed
    range. At nside 64 / 5' that is 5516.10 on the CAMB source (ells 2..3999) and 3266.98
    on the grid source, and the code returns 5516.28 and 3267.09 - the 1.000033 excess is
    (sinh h / h)^2 from the finite central difference, not an error. Those two numbers pin
    f_sky, the (2l + 1) weights, the factor of 1/2, the grid axis's `measure` and the
    stencil at once. delta_ell = 1 reproduces the unbinned CAMB sum bit for bit, which pins
    _ell_binner. grid_ell_axis records what pins the azimuthal reduction.

    MEASURED against "blocks" at nside 64 / 5' / 5 uK-arcmin, spectra = "ceiling", for
    omch2 / theta_MC_100 / logA:

      grid source, its full range (ell 67.5..3055, 456 distinct multipoles)
          f_sky sum (2l+1) measure = 6534 against the grid's 4096, a ratio of 1.595 whose
          square root, 1.263, is what sqrt(F_full_sky / F_blocks) = 1.10 / 1.42 / 1.27
          scatters around. The disk of radius 3055 simply holds more modes than the square
          of half-side 2160 that carries the box's DOF
      camb source truncated to ell 68..2160 (the box's ell_min to its Nyquist)
          f_sky sum (2l+1) = 3220 against 4096, a ratio of 0.7861 versus pi/4 = 0.7854, and
          sqrt(F_full_sky / F_blocks) = 0.96 / 0.79 / 0.88 bracketing sqrt(pi/4) = 0.886

    In both cases the spread around the mode-count ratio is the ell dependence of where each
    parameter's information sits, and the direction of the offset is set by whether the disk
    being summed is larger or smaller than the box's square. That is the mode counting doing
    its job, not a discrepancy to chase.
    """
    names = list(cl_fid)
    n_param = len(steps)
    n_block = len(names)

    fiducial = np.stack([np.asarray(bin_fn(cl_fid[name])) for name in names], axis = -1)
    if not np.all(fiducial > 0):
        raise RuntimeError(
            f"the fiducial spectra are not all positive over the requested ell range "
            f"(blocks {names}), so C_l cannot be inverted. Usually a block that is "
            f"identically zero at this cosmology, or an ell range reaching past where the "
            f"spectra are trustworthy")
    n_ell = fiducial.shape[0]

    covariance = np.zeros((n_ell, n_block, n_block))
    for a in range(n_block):
        covariance[:, a, a] = fiducial[:, a]
    inverse = np.linalg.inv(covariance)

    derivatives = []
    for i in range(n_param):
        block = np.zeros((n_ell, n_block, n_block))
        for a, name in enumerate(names):
            block[:, a, a] = bin_fn((np.asarray(cl_plus[i][name])
                                     - np.asarray(cl_minus[i][name])) / (2 * steps[i]))
        derivatives.append(block)

    fisher = np.zeros((n_param, n_param))
    for i in range(n_param):
        left = inverse @ derivatives[i] @ inverse
        for j in range(i, n_param):
            trace = np.einsum("lab,lba->l", left, derivatives[j])
            value = float(np.sum(mode_weights * trace))
            fisher[i, j] = value
            fisher[j, i] = value

    return fisher


def _full_sky_summary(spectra, nside, theta_pix, noise_level, l_knee, f_sky, ell_grid,
                      weights, ell_source, ell_axis, bin_ells, mode_weights, delta_ell,
                      names):
    """The verbose banner both stencils print, so they cannot describe the run differently.
    The two mode counts are the comparison _fisher_full_sky's docstring makes: a disk of
    multipoles against the rfft grid's square. 2 * sum(w) undoes the 1/2 in w."""
    grid_ells = ell_grid[ell_grid > 0]
    print(f"Full-sky Fisher forecast [{spectra}, ell_source {ell_source}]: nside {nside}, "
          f"theta_pix {theta_pix}', {noise_level} uK-arcmin, l_knee {l_knee}")
    print(f"  box {nside * theta_pix / 60:.2f} deg, f_sky {f_sky:.4g}, summing ell "
          f"{np.min(ell_axis):.0f}..{np.max(ell_axis):.0f} over {len(ell_axis)} "
          f"multipole(s) in {len(bin_ells)} band(s) of width "
          f"{delta_ell if delta_ell is not None else 'the axis spacing'}")
    print(f"  {2 * float(np.sum(mode_weights)):.0f} full-sky DOF vs "
          f"{int(jnp.sum(weights))} on the rfft grid, whose ell run "
          f"[{float(jnp.min(grid_ells)):.0f}, {float(jnp.max(grid_ells)):.0f}]")
    print(f"  sampled: {names}")


def _grid_cl_stencil(nside, theta_pix, spectra, noise_level, l_knee, stencil_output,
                     ell_min, ell_max, delta_ell, verbose):
    """The stencil for ell_source = "grid": covariance_stencil's blocks, reduced onto ell.

    Takes fisher_forecast.covariance_stencil's own output - the SAME covariance blocks
    "blocks" and "cls" contract - and azimuthally reduces every one of them onto the
    distinct |l| the rfft grid carries (grid_ell_axis). So the spectra summed here are
    precisely the ones the sampler works with: CAMB log-log interpolated onto ell_grid by
    covar_matrix_from_cls, times the same mask and beam, plus the same C_n, N_phi and
    delensing residual. Nothing is re-interpolated and, unlike the CAMB source's
    _radial_cl_profile step, nothing is extrapolated - the reduction never leaves the grid.

    covar_matrix_from_cls' 1/pix_width**2 rescale rides along on every block, which is
    exactly why it is safe to leave in: the contraction carries two powers of C^-1 against
    two of dC, so any theta-independent rescaling of C cancels identically.

    The reduction is lossless for every block that is a function of |l| alone - which is all
    of them except the QE noise inside the phi block; see grid_ell_axis. The verbose banner
    reports how anisotropic each block actually was, since that is the only approximation
    separating this path's matrices from the ones "blocks" contracts.

    The grid's distinct ells are irregularly spaced, so grid_ell_axis' midpoint spacing is
    handed to _ell_binner as the per-multipole width - that is the "delta_ell from the grid"
    this source is built around. `delta_ell` is then an OPTIONAL further coarsening on top.

    Returns (names, steps, mode_weights, bin_fn, cl_fid, cl_plus, cl_minus).
    """
    names, steps, weights, blocks_fid, blocks_plus, blocks_minus = stencil_output

    ell_grid, _ = gen_ell_grid(nside, theta_pix)
    f_sky = sky_fraction(nside, theta_pix)
    ell_axis, measure, reduce_fn, scatter_fn = grid_ell_axis(ell_grid, weights,
                                                             ell_min, ell_max)
    bin_ells, mode_weights, bin_fn = _ell_binner(ell_axis, f_sky, delta_ell,
                                                 measure = measure)

    if verbose:
        _full_sky_summary(spectra, nside, theta_pix, noise_level, l_knee, f_sky, ell_grid,
                          weights, "grid", ell_axis, bin_ells, mode_weights, delta_ell,
                          names)
        print("  reusing the covariance blocks 'blocks' / 'cls' contract - no extra CAMB "
              "calls")
        #the only approximation between this path and "blocks": a 1D ell sum cannot carry
        #an anisotropic covariance, so anything not a function of |l| alone gets averaged
        worst = {name: float(np.max(scatter_fn(block)))
                 for name, block in blocks_fid.items()}
        print("  within-|l| scatter lost to the azimuthal average: "
              + ", ".join(f"{name} {value:.1e}" for name, value in worst.items()))

    def reduce_blocks(blocks):
        return {name: reduce_fn(block) for name, block in blocks.items()}

    return (names, steps, mode_weights, bin_fn, reduce_blocks(blocks_fid),
            [reduce_blocks(b) for b in blocks_plus],
            [reduce_blocks(b) for b in blocks_minus])


def _camb_cl_stencil(nside, theta_pix, noise_level, is_sampled, param_ground, spectra,
                     step_fracs, l_knee, beam_fwhm, l_cutoff, ell_min, ell_max, delta_ell,
                     verbose, iterative_delens = False, nphi_source = "covariance",
                     qe_response = DEFAULT_QE_RESPONSE):
    """The stencil for ell_source = "camb": raw CAMB spectra on integer multipoles.

    The same CAMB runs covariance_stencil uses (camb_cls_at_params memoizes across both,
    so asking for "blocks" and this together costs one set of runs, not two) and the same
    frozen reconstruction - but the blocks stay as spectra on CAMB's own multipole axis
    instead of being interpolated onto the rfft grid. N_phi has to cross bases for the phi
    block (the box matrix, azimuthally averaged by _radial_cl_profile); with
    spectra = "delensed" the per-L Alens itself comes from either N_phi source, see
    fisher_forecast.frozen_reconstruction.

    Returns (names, steps, mode_weights, bin_fn, cl_fid, cl_plus, cl_minus).
    """
    if spectra not in SPECTRA_MODES:
        raise ValueError(f"spectra must be one of {SPECTRA_MODES}, got {spectra!r}")

    names, steps, fracs = sampled_names_and_steps(is_sampled, param_ground, step_fracs)

    ell_min = CAMB_ELL_MIN if ell_min is None else int(ell_min)
    ell_max = CAMB_ELL_MAX if ell_max is None else int(ell_max)
    if ell_min < CAMB_ELL_MIN:
        raise ValueError(f"ell_min must be at least {CAMB_ELL_MIN} (CAMB's first "
                         f"multipole), got {ell_min}")
    if ell_max > CAMB_ELL_MAX:
        raise ValueError(f"ell_max {ell_max} is past CAMB's support: camb_cls_at_params "
                         f"returns ells {CAMB_ELL_MIN} .. {CAMB_ELL_MAX}, so anything "
                         f"beyond it would be a log-log extrapolation of the spectra "
                         f"rather than a CAMB value. ell_source = 'grid' has no such cap - "
                         f"extrapolating there is what the sampler itself does")
    if ell_max <= ell_min:
        raise ValueError(f"ell_max ({ell_max}) must exceed ell_min ({ell_min})")
    ell_axis = np.arange(ell_min, ell_max + 1, dtype = np.float64)

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    #the same real-DOF weights covariance_stencil uses, here only to weight the annular
    #average that brings the 2D quadratic-estimate noise onto the 1D axis
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))
    f_sky = sky_fraction(nside, theta_pix)

    #cls_with_qe_response, not camb_cls_at_params, so qe_response = "gradient" finds its
    #"gradient_TT" key; a no-op on the default "unlensed", and only this fiducial point pays
    cls_fid = cls_with_qe_response(param_ground, qe_response)
    nphi_fid, alens = frozen_reconstruction(cls_fid, spectra, nside, pix_width, ell_grid,
                                            noise_level, l_knee, beam_fwhm, l_cutoff,
                                            iterative_delens, verbose,
                                            nphi_source = nphi_source,
                                            param_ground = param_ground,
                                            qe_response = qe_response)
    nphi_cl = _radial_cl_profile(nphi_fid, ell_grid, weights, pix_width, ell_axis)

    bin_ells, mode_weights, bin_fn = _ell_binner(ell_axis, f_sky, delta_ell)

    if verbose:
        _full_sky_summary(spectra, nside, theta_pix, noise_level, l_knee, f_sky, ell_grid,
                          weights, "camb", ell_axis, bin_ells, mode_weights, delta_ell,
                          names)
        print(f"  running {2 * len(names) + 1} CAMB calls...")

    def blocks_at(params):
        #the frozen per-L Alens gives every stencil point its own CAMB-delensed TT
        cls = (camb_cls_at_params(params) if alens is None
               else delensed_cls_at_params(params, alens))
        return cl_blocks(cls, spectra, ell_axis, noise_level, l_knee, beam_fwhm, nphi_cl)

    cl_fid = blocks_at(param_ground)

    cl_plus, cl_minus = [], []
    for name, step in zip(names, steps):
        up = dict(param_ground)
        up[name] = param_ground[name] + step
        down = dict(param_ground)
        down[name] = param_ground[name] - step
        cl_plus.append(blocks_at(up))
        cl_minus.append(blocks_at(down))
        if verbose:
            print(f"    {name}: h = {step:.6g} ({fracs[name]:g} sigma)")

    return names, steps, mode_weights, bin_fn, cl_fid, cl_plus, cl_minus


def forecast_full_sky(nside, theta_pix, noise_level, is_sampled, param_ground,
                      spectra = "lensed", step_fracs = None, l_knee = 0, beam_fwhm = 0,
                      l_cutoff = 10_000, verbose = True, iterative_delens = False,
                      ell_source = "grid", ell_min = None, ell_max = None,
                      delta_ell = None, nphi_source = "covariance",
                      qe_response = DEFAULT_QE_RESPONSE):
    """The (2l+1)/2 f_sky Fisher for the sampled LCDM parameters on an nside x nside box.

    Same arguments and return value as fisher_forecast.forecast (nphi_source included), plus:
        ell_source:   where the multipoles and spectra come from. "grid" (default) sums
                      over the distinct |l| the rfft grid carries, reading the blocks off
                      that grid - the same Cls the sampler sees, with the grid's own
                      spacing as the band width. "camb" uses the raw CAMB spectra on
                      their native integer multipoles
        ell_min:      first multipole in the sum. None (default) starts at the bottom of
                      the chosen source's range
        ell_max:      last multipole in the sum. None (default) runs to the top of the
                      chosen source's range
        delta_ell:    band width to bin the spectra into before contracting, ON TOP of
                      the source's own spacing. None (default) keeps every multipole of
                      the axis separate
    """
    if ell_source not in ELL_SOURCES:
        raise ValueError(f"ell_source must be one of {ELL_SOURCES}, got {ell_source!r}")

    if ell_source == "grid":
        stencil_output = covariance_stencil(
            nside, theta_pix, noise_level, is_sampled, param_ground, spectra, step_fracs,
            l_knee, beam_fwhm, l_cutoff, verbose, iterative_delens = iterative_delens,
            nphi_source = nphi_source, qe_response = qe_response)
        pieces = _grid_cl_stencil(nside, theta_pix, spectra, noise_level, l_knee,
                                  stencil_output, ell_min, ell_max, delta_ell, verbose)
    else:
        pieces = _camb_cl_stencil(
            nside, theta_pix, noise_level, is_sampled, param_ground, spectra, step_fracs,
            l_knee, beam_fwhm, l_cutoff, ell_min, ell_max, delta_ell, verbose,
            iterative_delens = iterative_delens, nphi_source = nphi_source,
            qe_response = qe_response)

    names, steps, mode_weights, bin_fn, fid, plus, minus = pieces
    return _fisher_full_sky(plus, minus, fid, steps, mode_weights, bin_fn), names


def main():
    parser = argparse.ArgumentParser(
        description = "Full-sky (2l+1)/2 f_sky Fisher forecast for the LCDM parameters, "
                      "on the sample_lcdm.py flat-sky box's multipoles or CAMB's")
    add_box_arguments(parser)
    add_spectra_arguments(parser)
    parser.add_argument("--ell_source", choices = ELL_SOURCES, default = "grid",
                        help = "where the multipoles and spectra come from. 'grid' "
                               "(default) sums over the distinct |l| the rfft grid carries "
                               "and reads the blocks off that grid, so the spectra are "
                               "exactly the ones the sampler sees and the grid's own "
                               "spacing plays delta_ell; 'camb' uses the raw CAMB spectra "
                               "on their native integer multipoles, which is the version "
                               "to quote against the literature")
    parser.add_argument("--delta_ell", type = int, default = None,
                        help = "bin the spectra into bands of this width before "
                               "contracting them, ON TOP of the source's own spacing. The "
                               "default keeps every multipole of the axis separate; "
                               "binning pools C and dC first and so throws away the "
                               "structure inside a band - see _ell_binner")
    parser.add_argument("--ell_min", type = float, default = None,
                        help = f"first multipole in the sum. Default: the bottom of "
                               f"--ell_source's range ({CAMB_ELL_MIN} for camb, the box's "
                               f"fundamental 2 pi / L for grid)")
    parser.add_argument("--ell_max", type = float, default = None,
                        help = f"last multipole in the sum. Default: the top of "
                               f"--ell_source's range ({CAMB_ELL_MAX} for camb, the grid's "
                               f"corner for grid). Cap it at the box's Nyquist ell to "
                               f"compare mode counts like for like against 'blocks'")
    args = parser.parse_args()
    is_sampled = sampled_from_args(parser, args)

    def run(step_fracs = None, spectra = args.spectra, verbose = True):
        return forecast_full_sky(args.nside, args.theta_pix, args.noise, is_sampled,
                                 GROUND_TRUTH, spectra = spectra, step_fracs = step_fracs,
                                 l_knee = args.l_knee, beam_fwhm = args.beam_fwhm,
                                 verbose = verbose,
                                 iterative_delens = args.iterative_delens,
                                 ell_source = args.ell_source, ell_min = args.ell_min,
                                 ell_max = args.ell_max, delta_ell = args.delta_ell,
                                 nphi_source = args.nphi_source,
                                 qe_response = args.qe_response)

    fisher, names = run()
    covariance = covariance_from_fisher(fisher, names)
    report_sigmas(covariance, names, args.spectra, METHOD)

    if args.spectra == "lensed":
        ceiling, _ = run(spectra = "ceiling", verbose = False)
        report_ceiling_ratio(fisher, ceiling, METHOD)

    if args.stability:
        step_stability(lambda fracs: run(step_fracs = fracs, verbose = False), METHOD)

    #the ell source, range and band width are what need stamping onto the figures - with
    #ell_source = "camb" the forecast is not tied to the box's resolution at all
    f_sky = sky_fraction(args.nside, args.theta_pix)
    ell_range = ("the source's full range" if args.ell_min is None and args.ell_max is None
                 else f"ell {args.ell_min if args.ell_min is not None else 'min'}.."
                      f"{args.ell_max if args.ell_max is not None else 'max'}")
    subtitle = (f"{args.spectra}, ell_source {args.ell_source}  |  "
                f"f_sky {f_sky:.3g}, {args.noise:g} uK-arcmin  |  {ell_range}"
                + (f", bands of {args.delta_ell}" if args.delta_ell is not None
                   else ", every multipole"))
    #np.savez cannot store None, so an unset limit is recorded as NaN = "the source's own
    #edge", and an unset delta_ell as NaN = "the axis's own spacing"
    config = run_config(args.spectra, args, names,
                        iterative_delens = args.iterative_delens,
                        nphi_source = args.nphi_source,
                        ell_source = args.ell_source,
                        ell_min = args.ell_min if args.ell_min is not None else np.nan,
                        ell_max = args.ell_max if args.ell_max is not None else np.nan,
                        delta_ell = args.delta_ell if args.delta_ell is not None else np.nan,
                        f_sky = f_sky)
    save_outputs(fisher, covariance, names, METHOD, SUFFIX, LABEL, subtitle, config)


if __name__ == "__main__":
    main()
