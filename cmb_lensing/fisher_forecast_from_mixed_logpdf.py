"""The realization-averaged Hessian of statistics.mixed_logpdf, "mixed":

    F_ij = < -d2 log p(d, f°, phi° | theta) / dtheta_i dtheta_j >  at a fixed mixed pair

The mixed-coordinate counterpart of fisher_forecast_from_logpdf. Each realization draws
(f, phi) at the fiducial cosmology with load_sim, mixes them ONCE with D and G at that
cosmology (f° = L(phi) D f, phi° = G phi - the sampler's coordinates), then holds that
mixed pair fixed while theta moves through C_f, C_phi, D, G and the -logdet(G) - logdet(D)
Jacobian inside mixed_logpdf.

READ THIS BEFORE COMPARING IT TO THE OTHER METHODS. mixed_logpdf is a properly normalized
density in (f°, phi°) - it carries the Jacobian for exactly that reason - so this is a
legitimate Fisher for the model written in mixed coordinates, and Louis's identity in
these coordinates would recover the same MARGINAL Fisher as in the unmixed ones. But the
split between Louis's two terms is NOT parametrization invariant: only their difference
is. So this Hessian is NOT the same quantity as the unmixed logpdf method's, and it does
not converge to the bare-prior complete-data Fisher that one can be checked against. Treat
the two as different decompositions, not as estimates of one number.

COST. Far more expensive per evaluation than the unmixed path: unmix runs an INVERSE
lensing solve and logpdf then runs a forward one, so every stencil point costs two lensing
solves, and nothing cancels across the stencil because D, G and the unmixed fields all move
with theta. There is no fast mode here - that is what mixing costs. Sequentially that is
~3800 lensing solves for 100 realizations at 19 stencil points, which is what
sampling_chains/mixed_hessian.sh exists for: it runs one slurm job per realization
(run_single_mixed_hessian.py -> mixed_hessian_realization, one small npz each; MEASURED
1m41s per job at nside 128 / 2.5' / 3 params) and `--hessian_dir` averages the collected
files here. forecast_from_mixed_logpdf is the single entry point: without `hessian_dir` it
is the sequential driver (smoke tests, reproducing a single job locally); with it, it
averages the cached files after checking they were made for the requested configuration.

The QE norm is evaluated once at the fiducial cosmology and frozen across the stencil,
mirroring the sampler (whose refresh_qe is plumbed but not exposed, so its QE norm stays at
the param_init cosmology for a whole chain).

Usage:
    python -m cmb_lensing.fisher_forecast_from_mixed_logpdf --realizations 2 --nside 64
    python -m cmb_lensing.fisher_forecast_from_mixed_logpdf --hessian_dir <mixed_hessian_output>

Writes fisher_matrix_from_mixed_logpdf.png, covariance_matrix_from_mixed_logpdf.png,
correlation_matrix_from_mixed_logpdf.png and fisher_from_mixed_logpdf.npz into
cmb_lensing/fisher_output/.
"""

import argparse
import glob
import os

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid
from cmb_lensing.simulate import load_sim, get_d_tt_matrix, get_g_matrix_lcdm
from cmb_lensing.statistics import mixed_logpdf
from cmb_lensing.mixing import mix
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH
from cmb_lensing.fisher_forecast import (camb_cls_at_params,
                                         load_sim_cosmology, prior_covariances,
                                         sampled_names_and_steps, stencil_offsets,
                                         hessian_from_stencil, covariance_from_fisher,
                                         report_sigmas, save_outputs, add_box_arguments,
                                         sampled_from_args, box_subtitle, run_config)


METHOD = "mixed"
SUFFIX = "_from_mixed_logpdf"
LABEL = r"mixed:  $F_{ij} = \langle -\partial_i \partial_j \log p(d, f^\circ, \phi^\circ\,|\,\theta) \rangle$"

#the realization methods draw from the prior and never consult a spectra mode; this is
#what their reports and figures are labelled with instead
SPECTRA = "prior draws"


def _mixed_theta_matrices(params, data_set, nside, pix_width, ell_grid, qe_frozen):
    """(C_f, C_phi, D, G) operators at one cosmology, built the way sample_lcdm builds them.

    D = get_d_tt_matrix(C_f, C_n) and G = get_g_matrix_lcdm(C_phi, N_phi) - the same two
    constructors _recompute_cosmo_matrices uses - so the mixing here matches the sampler's.

    `qe_frozen` is the quadratic-estimate norm, held FIXED across the stencil. That mirrors
    the sampler, whose refresh_qe is plumbed but not exposed, so its QE norm stays at the
    param_init cosmology for a whole chain. Letting N_phi move with theta would credit the
    Fisher with information from dN_phi/dtheta that no chain actually uses.
    """
    cf, cphi = prior_covariances(params, nside, pix_width, ell_grid)
    cf_op = data_set.field_covariance.replace(scalar_matrix = cf)
    cphi_op = data_set.phi_covariance.replace(scalar_matrix = cphi)
    d_op = data_set.mixing_d.replace(
        scalar_matrix = get_d_tt_matrix(cf, data_set.noise_covariance.scalar_matrix))
    g_op = data_set.mixing_g.replace(scalar_matrix = get_g_matrix_lcdm(cphi, qe_frozen))
    return cf_op, cphi_op, d_op, g_op


def mixed_logpdf_hessian_one_realization(data_set, names, param_ground, steps, stencil,
                                         nside, pix_width, ell_grid):
    """-d2/dtheta_i dtheta_j mixed_logpdf for ONE realization, at fixed (f_mixed, phi_mixed).

    The fields are drawn at param_ground and mixed ONCE with D and G at param_ground; that
    mixed pair is then held fixed while theta moves through C_f, C_phi, D, G and the
    -logdet(G) - logdet(D) Jacobian. See the module docstring for why this is a different
    decomposition from the unmixed logpdf Hessian, not an estimate of the same number.

    NOTE the argument-order trap: mixed_logpdf takes (..., mixing_g, mixing_d) while
    mix / unmix take (..., mixing_d, mixing_g). They are passed correctly below.
    """
    #the QE norm is evaluated once at the fiducial cosmology and frozen, matching the sampler
    qe_frozen = data_set.quadratic_estimate.scalar_matrix

    #mix ONCE, at the fiducial cosmology; this pair is the "sample" held fixed
    _, _, d_fid, g_fid = _mixed_theta_matrices(param_ground, data_set, nside, pix_width,
                                               ell_grid, qe_frozen)
    mixed_field, mixed_phi = mix(data_set.unlensed_field, data_set.phi, d_fid, g_fid)

    values = {}
    for offset in stencil:
        params = dict(param_ground)
        for i, name in enumerate(names):
            params[name] = param_ground[name] + offset[i] * steps[i]
        cf_op, cphi_op, d_op, g_op = _mixed_theta_matrices(
            params, data_set, nside, pix_width, ell_grid, qe_frozen)
        values[offset] = float(mixed_logpdf(
            mixed_field, mixed_phi, data_set.data, data_set.noise_covariance,
            cphi_op, cf_op, data_set.mask, data_set.beam, g_op, d_op))

    return -hessian_from_stencil(values, steps)


def mixed_hessian_realization(nside, theta_pix, noise_level, is_sampled, param_ground,
                              map_seed, step_fracs = None, l_knee = 0, shared_cls = None):
    """One realization end to end: simulate, mix, finite-difference.

    This is what run_single_mixed_hessian.py calls - one slurm job, one realization, one
    npz - so that the 100-realization average is built in parallel instead of sequentially.
    Returns (hessian, names, steps).
    """
    names, steps, _ = sampled_names_and_steps(is_sampled, param_ground, step_fracs)
    stencil = stencil_offsets(len(names))

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    camb_kwargs = load_sim_cosmology(param_ground)

    #reuse the fiducial spectra rather than re-running CAMB inside load_sim (see
    #load_sim's precomputed_cls); every realization sits at the same cosmology
    if shared_cls is None:
        shared_cls = camb_cls_at_params(param_ground)
    data_set = load_sim(nside, theta_pix, "I", map_seed, **camb_kwargs,
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = shared_cls)

    hessian = mixed_logpdf_hessian_one_realization(
        data_set, names, param_ground, steps, stencil, nside, pix_width, ell_grid)
    return hessian, names, steps


def forecast_from_mixed_logpdf(nside, theta_pix, noise_level, is_sampled, param_ground,
                               n_realizations = 100, step_fracs = None, l_knee = 0,
                               map_seed = 20260908, hessian_dir = None, verbose = True):
    """The realization-averaged mixed-coordinate Hessian, computed here or read from disk.

    With `hessian_dir = None` this is the sequential driver - the local / small-N path. For
    the full 100 realizations that is slow (two lensing solves per stencil point per
    realization), which is what sampling_chains/mixed_hessian.sh is for: it fans the same
    per-realization work across slurm jobs, one hessian_<index>.npz each. Point
    `hessian_dir` at that output directory and every file in it is averaged INSTEAD of
    computing anything; `n_realizations` and `map_seed` are then ignored (the files decide
    both), and the files must have been produced for exactly the box, noise, l_knee and
    parameter set requested here, or a ValueError says which differs - a forecast must
    describe the configuration it is labelled with.

    Returns (fisher, names, per_realization).
    """
    names, _, _ = sampled_names_and_steps(is_sampled, param_ground, step_fracs)

    if hessian_dir is not None:
        requested = dict(names = names, nside = nside, theta_pix = theta_pix,
                         noise_level = noise_level, l_knee = l_knee)
        fisher, names, per_realization, metadata = load_hessian_directory(hessian_dir,
                                                                          verbose = verbose)
        _check_cached_config(hessian_dir, metadata, requested)
        return fisher, names, per_realization

    if verbose:
        print(f"Fisher forecast [mixed]: nside {nside}, theta_pix {theta_pix}', "
              f"{noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  sampled: {names}")
        print(f"  {n_realizations} realizations x {len(stencil_offsets(len(names)))} "
              f"mixed_logpdf evaluations, 2 lensing solves each")

    shared_cls = camb_cls_at_params(param_ground)
    per_realization = []
    for index in range(n_realizations):
        hessian, names, _ = mixed_hessian_realization(
            nside, theta_pix, noise_level, is_sampled, param_ground, map_seed + index,
            step_fracs = step_fracs, l_knee = l_knee, shared_cls = shared_cls)
        per_realization.append(hessian)
        if verbose and (index + 1) % 10 == 0:
            running = np.mean(per_realization, axis = 0)
            print(f"    {index + 1}/{n_realizations}  running diagonal "
                  f"{np.array2string(np.diag(running), precision = 4)}")

    per_realization = np.array(per_realization)
    return np.mean(per_realization, axis = 0), names, per_realization


def _check_cached_config(directory, metadata, requested):
    """Raise unless the cached Hessians were produced for the configuration requested."""
    mismatches = [f"{key}: files {metadata[key]!r}, requested {requested[key]!r}"
                  for key in requested if metadata[key] != requested[key]]
    if mismatches:
        raise ValueError(
            f"the Hessians in {directory} were not produced for the requested "
            f"configuration - " + "; ".join(mismatches) + ". Pass the box the files were "
            f"made for (hessian_directory_config reads it), or leave hessian_dir unset "
            f"to compute the realizations here")


def hessian_directory_config(directory):
    """The configuration every hessian_*.npz in `directory` was produced with.

    Reads only the metadata written by run_single_mixed_hessian.py and checks the files
    agree on it, so results from two different runs cannot be silently averaged together.
    Returns {"names", "nside", "theta_pix", "noise_level", "l_knee", "method",
    "n_realizations", "paths"}.
    """
    paths = sorted(glob.glob(os.path.join(directory, "hessian_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"no hessian_*.npz in {directory}. Run sampling_chains/mixed_hessian.sh first, "
            f"or point --hessian_dir at the out_dir that script writes to.")

    config = None
    for path in paths:
        data = np.load(path, allow_pickle = True)
        #npz round-trips strings as np.str_; they compare and hash like str but print as
        #np.str_('omch2'), so convert back for readable reports and clean npz round-trips
        current = dict(names = [str(name) for name in data["names"]],
                       nside = int(data["nside"]), theta_pix = float(data["theta_pix"]),
                       noise_level = float(data["noise_level"]),
                       l_knee = float(data["l_knee"]), method = str(data["method"]))
        if config is None:
            config = current
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; a forecast cannot mix configurations")

    return dict(config, n_realizations = len(paths), paths = paths)


def load_hessian_directory(directory, verbose = True):
    """Average the per-realization Hessian npz files written by run_single_mixed_hessian.py.

    The parallel counterpart to computing the realizations in forecast_from_mixed_logpdf
    (which calls this when given hessian_dir): slurm jobs each write one
    hessian_<index>.npz, and this collects them, refusing mixed configurations and
    duplicate seeds.

    Returns (fisher, names, per_realization, metadata) with metadata as
    hessian_directory_config gives it.
    """
    metadata = hessian_directory_config(directory)
    if metadata["method"] != METHOD:
        raise ValueError(f"{directory} holds '{metadata['method']}' Hessians, not "
                         f"'{METHOD}' ones")

    hessians, seeds = [], []
    for path in metadata["paths"]:
        data = np.load(path, allow_pickle = True)
        hessians.append(data["hessian"])
        seeds.append(int(data["map_seed"]))

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                         f"realizations would be double counted. Each slurm job must use "
                         f"a distinct seed - check the map_prefix loop in mixed_hessian.sh.")

    per_realization = np.array(hessians)
    if verbose:
        print(f"Averaging {len(hessians)} cached realizations from {directory}")
        print(f"  method {metadata['method']}, parameters {metadata['names']}, nside "
              f"{metadata['nside']}, {metadata['theta_pix']:g}', "
              f"{metadata['noise_level']:g} uK-arcmin, l_knee {metadata['l_knee']:g}")
        spread = np.std(per_realization, axis = 0) / np.sqrt(len(hessians))
        mean = np.mean(per_realization, axis = 0)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            relative = np.abs(spread / mean)
        print(f"  standard error on the mean, relative: max {np.nanmax(relative):.3f}")

    return np.mean(per_realization, axis = 0), metadata["names"], per_realization, metadata


def averaged_hessian_covariance(fisher, names, n_realizations):
    """Invert an averaged Hessian, blaming too few realizations rather than the stencil."""
    try:
        return covariance_from_fisher(fisher, names)
    except RuntimeError as error:
        raise RuntimeError(
            f"the averaged Hessian is not positive definite over {n_realizations} "
            f"realizations. For a realization-averaged method that usually means too few "
            f"realizations rather than a bad finite-difference step - add more jobs and "
            f"re-run the averaging. Underlying: {error}") from error


def main():
    parser = argparse.ArgumentParser(
        description = "Fisher forecast from the realization-averaged Hessian of "
                      "statistics.mixed_logpdf at a fixed mixed (f, phi) pair")
    add_box_arguments(parser)
    parser.add_argument("--realizations", type = int, default = 100,
                        help = "(data, f, phi) draws to average over, sequentially. "
                               "Ignored with --hessian_dir, where the files decide")
    parser.add_argument("--hessian_dir", type = str, default = "/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/mixed_hessian_output/",
                        help = "average per-realization hessian_*.npz files written by "
                               "sampling_chains/mixed_hessian.sh instead of computing them "
                               "here. This is the time-feasible route: 100 slurm jobs run "
                               "the realizations in parallel and this just collects them. "
                               "The box arguments (--nside, --theta_pix, --noise, "
                               "--l_knee, --params) are then taken from the files, not "
                               "from the command line. Pass an empty string to compute "
                               "the realizations here instead")
    args = parser.parse_args()
    hessian_dir = args.hessian_dir or None

    if hessian_dir is not None:
        #the files fix the configuration; the command line's box defaults would otherwise
        #describe a box the cached Hessians were never computed for
        cached = hessian_directory_config(hessian_dir)
        args.nside, args.theta_pix = cached["nside"], cached["theta_pix"]
        args.noise, args.l_knee = cached["noise_level"], cached["l_knee"]
        args.params = cached["names"]
        args.realizations = cached["n_realizations"]
    is_sampled = sampled_from_args(parser, args)

    fisher, names, per_realization = forecast_from_mixed_logpdf(
        args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
        n_realizations = args.realizations, l_knee = args.l_knee, hessian_dir = hessian_dir)
    covariance = averaged_hessian_covariance(fisher, names, len(per_realization))
    report_sigmas(covariance, names, SPECTRA, METHOD)

    source = "cached" if hessian_dir is not None else "sequential"
    subtitle = (f"{box_subtitle(SPECTRA, args)}  |  {len(per_realization)} realizations "
                f"({source})")
    config = run_config(SPECTRA, args, names, n_realizations = len(per_realization),
                        standard_error = np.std(per_realization, axis = 0)
                                         / np.sqrt(len(per_realization)))
    save_outputs(fisher, covariance, names, METHOD, SUFFIX, LABEL, subtitle, config)


if __name__ == "__main__":
    main()
