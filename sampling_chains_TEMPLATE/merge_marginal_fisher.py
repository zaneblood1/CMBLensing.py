"""Combine the per-map outputs of run_single_marginal_fisher.py into one forecast.

Reads every map_*_scores.npz written by the slurm jobs and reports the marginal Fisher by
both estimators, with a jackknife-over-maps error bar on each:

  Louis          the average over maps of I_obs = E[-Hessian] - Cov(score). Each map's
                 value is that map's OBSERVED information; averaging turns it into the
                 expected information, i.e. the forecast.
  score covar    Cov_d[ E[score | d] ] over maps, debiased by the half-split
                 cross-covariance. No subtraction of two large terms, but it needs many
                 maps where Louis needs many draws per map.

The two are independent contractions of the same stored scores and degrade in different
directions, so agreement within the jackknife errors is the end-to-end check. Compare the
result against fisher_forecast's ceiling and lensed bounds - it must sit between them -
and against the measured posterior covariance chain_analysis.py prints.

Usage:
    python merge_marginal_fisher.py --run_dir marginal_fisher_output
"""

import argparse
import glob
import os

import numpy as np

from cmb_lensing.fisher_forecast import (covariance_from_fisher, write_outputs,
                                         correlation_from_covariance, output_dir)
from cmb_lensing.marginal_fisher import score_covariance_information
from cmb_lensing.precompute_camb_1d import PARAM_SIGMA


def load_maps(run_dir):
    """Every per-map npz in run_dir, checked for a consistent configuration."""
    paths = sorted(glob.glob(os.path.join(run_dir, "map_*_scores.npz")))
    if not paths:
        raise FileNotFoundError(f"no map_*_scores.npz in {run_dir}")

    maps, config = [], None
    for path in paths:
        data = np.load(path, allow_pickle = True)
        current = (tuple(data["names"]), int(data["nside"]), float(data["theta_pix"]),
                   float(data["noise_level"]), int(data["n_draws"]))
        if config is None:
            config = current
        elif current != config:
            raise ValueError(f"{path} was run at {current}, but the earlier files used "
                             f"{config}; a forecast cannot mix configurations")
        maps.append(data)
    return maps, config


def jackknife(per_map_matrices, combine):
    """Delete-one jackknife error on `combine` applied across maps.

    combine takes a list of per-map entries and returns a matrix. The jackknife is the
    right tool here because both estimators are non-linear in the map sample (one is an
    average of inverses' inputs, the other a covariance across maps), so a naive
    standard error of the mean would understate the score-covariance estimator's spread.
    """
    n = len(per_map_matrices)
    full = combine(per_map_matrices)
    partials = np.array([combine([m for j, m in enumerate(per_map_matrices) if j != i])
                         for i in range(n)])
    variance = (n - 1) / n * np.sum((partials - partials.mean(axis = 0))**2, axis = 0)
    return full, np.sqrt(variance)


def report(information, error, names, label):
    covariance = covariance_from_fisher(information, names)
    sigmas = np.sqrt(np.diag(covariance))
    correlation = correlation_from_covariance(covariance)
    #a fractional error on the diagonal of F propagates to half that on sigma = 1/sqrt(F)
    sigma_error = 0.5 * sigmas * np.abs(np.diag(error) / np.diag(information))

    print(f"\n{label}")
    print(f"  {'parameter':<14s} {'sigma':>12s} {'+/-':>11s} {'sigma/PARAM_SIGMA':>20s}")
    for i, name in enumerate(names):
        print(f"  {name:<14s} {sigmas[i]:>12.4g} {sigma_error[i]:>11.2g} "
              f"{sigmas[i] / PARAM_SIGMA[name]:>20.3g}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"  r({names[i]}, {names[j]}) = {correlation[i, j]:+.3f}")
    return covariance


def main():
    parser = argparse.ArgumentParser(
        description = "combine per-map marginal-Fisher jobs into one forecast")
    parser.add_argument("--run_dir", type = str, required = True,
                        help = "directory holding the map_*_scores.npz files")
    parser.add_argument("--no_write", action = "store_true",
                        help = "skip the figures and npz")
    args = parser.parse_args()

    maps, config = load_maps(args.run_dir)
    names = list(config[0])
    nside, theta_pix, noise_level, n_draws = config[1], config[2], config[3], config[4]

    print(f"Marginal Fisher from {len(maps)} maps x {n_draws} draws "
          f"[nside {nside}, {theta_pix:g}', {noise_level:g} uK-arcmin]")
    print(f"  parameters: {names}")

    acceptance = np.array([float(data["phi_acceptance"]) for data in maps])
    autocorrelation = np.array([data["autocorrelation"] for data in maps])
    effective = n_draws / autocorrelation
    print(f"  phi acceptance   mean {acceptance.mean():.2f}  "
          f"range [{acceptance.min():.2f}, {acceptance.max():.2f}]")
    print(f"  autocorrelation  mean {np.array2string(autocorrelation.mean(axis = 0), precision = 1)}")
    print(f"  N_eff per map    mean {np.array2string(effective.mean(axis = 0), precision = 0)}   "
          f"-> ~{np.sqrt(2 / effective.mean()):.1%} error on Cov(score) per map")

    hessian = np.mean([data["hessian_term"] for data in maps], axis = 0)
    louis_per_map = [data["information"] for data in maps]
    louis, louis_error = jackknife(louis_per_map, lambda entries: np.mean(entries, axis = 0))
    survived = np.diag(louis) / np.diag(hessian)
    print(f"  Louis subtraction: {np.array2string(survived * 100, precision = 0)}% of "
          f"E[-H] survives Cov(score)")

    covariance = report(louis, louis_error, names, "LOUIS (averaged over maps)")

    scores = [data["scores"] for data in maps]
    if len(maps) > len(names) + 1:
        cross, cross_error = jackknife(
            scores, lambda entries: score_covariance_information(entries)[0])
        try:
            report(cross, cross_error, names, "SCORE COVARIANCE (half-split debiased)")
            #the two estimators are independent contractions of the same draws
            spread = np.abs(np.diag(cross) - np.diag(louis))
            combined = np.sqrt(np.diag(louis_error)**2 + np.diag(cross_error)**2)
            print(f"\n  estimator agreement (|difference| / combined jackknife error): "
                  f"{np.array2string(spread / combined, precision = 1)}")
            print("  values under ~2 mean the two estimators agree")
        except RuntimeError as error:
            print(f"\n  score-covariance estimator not positive definite: {error}")
    else:
        print(f"\n  score-covariance estimator skipped: needs more than {len(names) + 1} "
              f"maps, got {len(maps)}")

    if not args.no_write:
        directory = output_dir()
        os.makedirs(directory, exist_ok = True)
        subtitle = (f"marginal  |  nside {nside}, {theta_pix:g}', {noise_level:g} uK-arcmin"
                    f"  |  {len(maps)} maps x {n_draws} draws")
        written = write_outputs(
            louis, covariance, names, directory, "marginal", subtitle,
            dict(spectra = "marginal", nside = nside, theta_pix = theta_pix,
                 noise_level = noise_level, n_maps = len(maps), n_draws = n_draws,
                 hessian_term = hessian, jackknife_error = louis_error,
                 phi_acceptance = acceptance, autocorrelation = autocorrelation))
        print(f"\nWrote {', '.join(written)} to {directory}")


if __name__ == "__main__":
    main()
