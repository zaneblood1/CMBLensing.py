"""The BANDPOWER Fisher forecast, "cls": F_ij = d_i mu^T Cov(mu_hat)^-1 d_j mu.

The data vector mu is the per-mode power of every covariance block fisher_forecast builds
(C_TT and C_phiphi on the rfft grid) and Cov(mu_hat) is the covariance of those ESTIMATES,
not of the map. With a covariance that is diagonal in the modes - Var(C^_k) = 2 C_k^2 / w_k
for Gaussian fields with w_k real degrees of freedom per mode - the contraction collapses to

    F = sum_k w_k / (2 C^2) d_iC d_jC = 1/2 sum_k w_k d_i lnC d_j lnC

which is EXACTLY fisher_forecast's "blocks" trace formula, and that is what this module
returns for spectra = "ceiling" / "unlensed" (verified to 1e-17).

Its reason to exist is what the trace form cannot express. For spectra = "lensed" and
"delensed" it adds the NON-GAUSSIAN covariance gravitational lensing induces
(lensing_covariance.py): the realization's phi power moves the TT power at every multipole
through the first-order flat-sky kernel K(l, L) = dC_lensed(l)/dC_phi(L) (Hu 2000;
Benoit-Levy, Smith & Hu 2012), giving a TT-TT covariance 2 K diag(C_eff^2) K^T that
correlates TT power across multipoles, and - for "lensed", where the phi-phi estimate and
the TT power share one realization of phi - a TT-phiphi cross term 2 K diag(C_phi^2)
(Schmittfull+13 / Peloton+17's signal term). C_eff is the full C_phi for "lensed" and the
per-mode residual C_phi N_phi / (C_phi + N_phi) for "delensed", whose cross term vanishes
(a Wiener residual is uncorrelated with the recovered part). Those off-diagonal terms are
the only place in any of these forecasts where information can be SUBTRACTED rather than
summed mode by mode, which is what lets this path describe the TT two-point function and
the lensing four-point function sharing information instead of double counting it.

The operator is applied matrix-free (three FFT convolutions per kernel apply, circular
because the box is periodic) and inverted by preconditioned conjugate gradients on the
stacked (TT, phiphi) vector of independent modes, so it stays exact per mode - nothing is
banded or densified. It is off for "ceiling" / "unlensed" (there is no residual lensing to
induce it) and can be switched off everywhere with cls_non_gaussian = False, which
reproduces "blocks" exactly.

MEASURED at nside 128 / 2.5' / 5 uK, T-only: about a 1% effect. Per-mode TT-TT
correlations reach 0.045 at ell 2500 and TT-phiphi 0.12 against the lowest phi mode -
the literature's size once bandpower binning is undone - so on this box it does not move
the "lensed" forecast (r(omch2, theta) +0.085 -> +0.060). Every verbose run prints the
kernel's first-order correction against CAMB's C_lensed - C_unlensed per annulus
(0.93-1.00 below ell 1000 at nside 128; the ~1000-1500 annulus is where that correction
crosses zero). tests/test_lensing_covariance.py pins the operator against a brute-force
kernel matrix on nside 8/10.

The stencil is fisher_forecast.covariance_stencil - the very same blocks "blocks"
contracts, so the two can never drift apart through a differently-built stencil, and the
CAMB runs are shared through camb_cls_at_params's memo.

Usage:
    python -m cmb_lensing.fisher_forecast_from_cls --spectra lensed
    python -m cmb_lensing.fisher_forecast_from_cls --spectra lensed --cls_gaussian_only
    python -m cmb_lensing.fisher_forecast_from_cls --spectra delensed --iterative_delens

Writes fisher_matrix_from_cls.png, covariance_matrix_from_cls.png,
correlation_matrix_from_cls.png and fisher_from_cls.npz into cmb_lensing/fisher_output/.
"""

import argparse

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid
from cmb_lensing.simulate import covar_matrix_from_cls
from cmb_lensing.lensing_covariance import (LensingBandpowerCovariance,
                                            fisher_from_bandpower_covariance,
                                            first_order_lensing_check)
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH
from cmb_lensing.fisher_forecast import (camb_cls_at_params, covariance_stencil,
                                         cls_with_qe_response, DEFAULT_QE_RESPONSE,
                                         camb_lmax_for_grid,
                                         frozen_reconstruction, covariance_from_fisher,
                                         step_stability, report_sigmas,
                                         report_ceiling_ratio, save_outputs,
                                         add_box_arguments, add_spectra_arguments,
                                         sampled_from_args, box_subtitle, run_config)


METHOD = "cls"
SUFFIX = "_from_cls"
LABEL = (r"cls:  $F_{ij} = \partial_i \mu^T\,\mathrm{Cov}(\hat\mu)^{-1}\,\partial_j \mu$"
         r"  (lensing non-Gaussian covariance)")

#which stencil block holds the temperature power and which the phi power, per spectra
#mode - the non-Gaussian bandpower covariance needs to tell them apart. "ceiling" and
#"unlensed" carry no residual lensing, so they get no non-Gaussian term and are absent
_CLS_BLOCK_KEYS = {"lensed": ("TT", "PP"), "delensed": ("f_delensed", "phi")}


def _fisher_from_cls(blocks_plus, blocks_minus, blocks_fid, steps, weights,
                     lensing_cov = None, tt_key = None, pp_key = None):
    """Bandpower-data-vector Fisher: F_ij = d_i mu^T Cov(mu_hat)^-1 d_j mu.

    `lensing_cov = None` (the Gaussian case). The estimates are independent with
    Var(C_k) = 2 C_k^2 / w_k, so the contraction is EXACTLY fisher_forecast's
    _fisher_from_blocks. That identity is worth keeping: the trace formula and the
    data-vector formula are the same estimator whenever the covariance is diagonal, and
    this branch is what spectra = "ceiling" / "unlensed" use.

    `lensing_cov` set (a lensing_covariance.LensingBandpowerCovariance built on the
    fiducial blocks by _lensing_bandpower_covariance). Now the covariance couples modes:
    the realization's lensing power moves the TT power at every multipole through the
    first-order lensing kernel, so C^_TT is correlated across ell, and the same
    realization's phi power is what C^_phiphi measures, so the two blocks are correlated
    too. `tt_key` / `pp_key` name the temperature and the phi block in the stencil dicts.
    The derivative vectors are stacked over the operator's independent-mode layout and
    Cov^-1 dmu is solved by preconditioned conjugate gradients. The Gaussian diagonal of
    that operator is the same 2 C^2 / w as the branch above, so switching the
    non-Gaussian terms off inside the operator reproduces "blocks" too.
    """
    names = list(blocks_fid)
    n_param = len(steps)

    if lensing_cov is None:
        fisher = np.zeros((n_param, n_param))
        for name in names:
            c_fid = blocks_fid[name]
            good = c_fid > 0
            safe = jnp.where(good, c_fid, 1.0)
            #Var(C^_k) = 2 C_k^2 / w_k, and the [0, 0] origin carries no information
            inverse_variance = jnp.where(good, weights / (2.0 * safe**2), 0.0)
            derivatives = [(blocks_plus[i][name] - blocks_minus[i][name]) / (2 * steps[i])
                           for i in range(n_param)]
            for i in range(n_param):
                for j in range(i, n_param):
                    value = float(jnp.sum(inverse_variance * derivatives[i] * derivatives[j]))
                    fisher[i, j] += value
                    if i != j:
                        fisher[j, i] += value
        return fisher

    if tt_key is None or pp_key is None or set(names) != {tt_key, pp_key}:
        raise ValueError(f"the non-Gaussian bandpower covariance needs exactly a TT and a "
                         f"phi-phi block; got blocks {names} with tt_key = {tt_key!r}, "
                         f"pp_key = {pp_key!r}")

    derivatives = []
    for i in range(n_param):
        d_tt = np.asarray((blocks_plus[i][tt_key] - blocks_minus[i][tt_key]) / (2 * steps[i]))
        d_pp = np.asarray((blocks_plus[i][pp_key] - blocks_minus[i][pp_key]) / (2 * steps[i]))
        derivatives.append(lensing_cov.stack(d_tt, d_pp))

    fisher, asymmetry = fisher_from_bandpower_covariance(lensing_cov, derivatives)
    if asymmetry > 1e-6:
        print(f"  WARNING: the bandpower Fisher came back asymmetric at the {asymmetry:.1e} "
              f"level before symmetrization - the conjugate-gradient solves have not "
              f"converged tightly enough; tighten LensingBandpowerCovariance.solve's rtol")
    return fisher


def _lensing_bandpower_covariance(spectra, blocks_fid, nside, pix_width, ell_grid,
                                  noise_level, l_knee, beam_fwhm, l_cutoff, param_ground,
                                  iterative_delens, weights, verbose,
                                  nphi_source = "covariance",
                                  qe_response = DEFAULT_QE_RESPONSE):
    """The lensing_covariance operator at this spectra mode, or None.

    Only "lensed" and "delensed" carry residual lensing in their temperature block, so only
    they get the non-Gaussian terms. The phi power whose sample variance leaks into the TT
    power is the FULL C_phi for "lensed" (nothing is delensed) and the per-mode residual
    C_phi N_phi / (C_phi + N_phi) for "delensed", with the same frozen N_phi the stencil
    used (iterated if the stencil iterated). The TT-phiphi cross term is on for "lensed",
    where the phi-phi estimate and the TT power share the full realization, and off for
    "delensed", where the residual left in the map is uncorrelated with the recovered
    part the phi-phi estimate measures - see lensing_covariance.py.

    With verbose the first-order lensing kernel is checked against CAMB's own lensing
    correction on this grid, annulus by annulus, and the ratios are printed.
    """
    if spectra not in _CLS_BLOCK_KEYS:
        return None
    tt_key, pp_key = _CLS_BLOCK_KEYS[spectra]

    #the same camb_lmax covariance_stencil defaults to, so this kernel and the blocks it
    #corrects are built from one set of CAMB spectra rather than two of different reach
    camb_lmax = camb_lmax_for_grid(ell_grid)
    cls_fid = cls_with_qe_response(param_ground, qe_response, camb_lmax = camb_lmax)
    ells = jnp.arange(2, 2 + cls_fid["scalar_TT"].shape[0]).astype(jnp.float64)
    phi_ells = jnp.arange(2, 2 + cls_fid["phi"].shape[0]).astype(jnp.float64)

    def covar(cl, ell_axis):
        return covar_matrix_from_cls(nside, pix_width, ell_grid, ell_axis, cl,
                                     origin_value = 0)

    cf_unlensed = np.asarray(covar(cls_fid["scalar_TT"], ells))
    cphi = np.asarray(covar(cls_fid["phi"], phi_ells))
    #the same frozen reconstruction the stencil built, recomputed rather than threaded
    #through covariance_stencil's return value; camb_cls_at_params is memoized so this
    #costs one QE
    nphi, _ = frozen_reconstruction(cls_fid, spectra, nside, pix_width, ell_grid,
                                    noise_level, l_knee, beam_fwhm, l_cutoff,
                                    iterative_delens, verbose = False,
                                    nphi_source = nphi_source, param_ground = param_ground,
                                    qe_response = qe_response, camb_lmax = camb_lmax)
    nphi = np.asarray(nphi)

    if spectra == "lensed":
        cphi_eff, cross = cphi, True
    else:
        total = cphi + nphi
        cphi_eff = np.where(total > 0, cphi * nphi / np.where(total > 0, total, 1.0), 0.0)
        cross = False

    operator = LensingBandpowerCovariance(blocks_fid[tt_key], blocks_fid[pp_key],
                                          cf_unlensed, cphi_eff, cross, nside, pix_width)
    if verbose:
        cfl = np.asarray(covar(cls_fid["total_TT"], ells))
        rows = first_order_lensing_check(operator.kernel, cphi, cf_unlensed, cfl,
                                         ell_grid, weights)
        print("  first-order lensing kernel vs CAMB's C_lensed - C_unlensed, per annulus "
              "(near 1 below ell ~1500 validates the kernel; the drift above is the "
              "first-order approximation itself):")
        print("    " + ", ".join(f"[{low}, {high}) {ratio:.2f}" for low, high, ratio in rows))
    return operator


def forecast_from_cls(nside, theta_pix, noise_level, is_sampled, param_ground,
                      spectra = "lensed", step_fracs = None, l_knee = 0, beam_fwhm = 0,
                      l_cutoff = 10_000, verbose = True, cls_non_gaussian = True,
                      iterative_delens = False, nphi_source = "covariance",
                      qe_response = DEFAULT_QE_RESPONSE):
    """The bandpower Fisher for the sampled LCDM parameters on an nside x nside box.

    Same arguments and return value as fisher_forecast.forecast (nphi_source included),
    plus `cls_non_gaussian`
    (default True): include the lensing-induced mode coupling in the bandpower covariance
    for spectra = "lensed" / "delensed". False keeps the covariance diagonal, which makes
    the result identical to "blocks".
    """
    names, steps, weights, fid, plus, minus = covariance_stencil(
        nside, theta_pix, noise_level, is_sampled, param_ground, spectra, step_fracs,
        l_knee, beam_fwhm, l_cutoff, verbose, iterative_delens = iterative_delens,
        nphi_source = nphi_source, qe_response = qe_response)

    lensing_cov = None
    if cls_non_gaussian:
        ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
        lensing_cov = _lensing_bandpower_covariance(
            spectra, fid, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
            l_cutoff, param_ground, iterative_delens, weights, verbose,
            nphi_source = nphi_source, qe_response = qe_response)
    tt_key, pp_key = _CLS_BLOCK_KEYS.get(spectra, (None, None))

    fisher = _fisher_from_cls(plus, minus, fid, steps, weights, lensing_cov = lensing_cov,
                              tt_key = tt_key, pp_key = pp_key)
    return fisher, names


def main():
    parser = argparse.ArgumentParser(
        description = "Bandpower Fisher forecast with the lensing-induced non-Gaussian "
                      "covariance, on the sample_lcdm.py flat-sky box")
    add_box_arguments(parser)
    add_spectra_arguments(parser)
    parser.add_argument("--cls_gaussian_only", action = "store_true",
                        help = "drop the lensing-induced non-Gaussian bandpower covariance "
                               "and keep the per-mode Gaussian diagonal, which makes this "
                               "method reproduce blocks EXACTLY. The default keeps it for "
                               "--spectra lensed / delensed; ceiling / unlensed never "
                               "carry it - see lensing_covariance.py")
    args = parser.parse_args()
    is_sampled = sampled_from_args(parser, args)

    def run(step_fracs = None, spectra = args.spectra, verbose = True):
        return forecast_from_cls(args.nside, args.theta_pix, args.noise, is_sampled,
                                 GROUND_TRUTH, spectra = spectra, step_fracs = step_fracs,
                                 l_knee = args.l_knee, beam_fwhm = args.beam_fwhm,
                                 verbose = verbose,
                                 cls_non_gaussian = not args.cls_gaussian_only,
                                 iterative_delens = args.iterative_delens,
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

    save_outputs(fisher, covariance, names, METHOD, SUFFIX, LABEL,
                 box_subtitle(args.spectra, args),
                 run_config(args.spectra, args, names,
                            cls_non_gaussian = not args.cls_gaussian_only,
                            iterative_delens = args.iterative_delens,
                            nphi_source = args.nphi_source))


if __name__ == "__main__":
    main()
