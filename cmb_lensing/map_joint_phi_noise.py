"""The lensing reconstruction's EFFECTIVE noise N_phi, measured per mode with map_joint.

WHAT IS MEASURED. For one realization at the fiducial cosmology (GROUND_TRUTH): load_sim draws
the truth, map_joint reconstructs phi from the data, and per rfft2 mode k three moments are
stored,

    auto      A_k = |phi_hat_k|^2
    cross     B_k = Re phi_hat_k phi_k*
    true_auto D_k = |phi_k|^2

all divided by nside^2, i.e. in covar_matrix_from_cls units. Writing the least-squares split
phi_hat = rho phi + n (n uncorrelated with phi by construction of rho, so this holds for any
estimator), the correlation coefficient r^2 = <B>^2 / (<A><D>) is the Wiener weight
C_phi / (C_phi + N_phi) of the response-deconvolved reconstruction, so

    N_phi = C_phi (1 / r^2 - 1)

per mode. It is rescale invariant: multiplying phi_hat by a constant leaves r^2 untouched.

HOW IT IS AVERAGED. The moments are averaged over realizations FIRST and r^2 is formed from
the averages: one mode of one realization has r^2 = cos^2(phase difference), so per-mode
ratios of single realizations cannot be averaged into the answer. Modes whose mean cross power is not positive carry no measurement
and are stored as NaN (the forecast gives them no phi information, delensed_covariance's
load_stored_nphi gives them W = 0). The origin is 0.

N_phi describes map_joint AT the number of steps it was run for, on THIS box and data
polarization (`pol`: "I", "P" or "IP"; phi is a scalar field for all three), at the fiducial
cosmology; it is held fixed across the forecast's finite-difference stencil.

Fanned out one slurm job per realization by
measure_phi_noise/get_map_joint_phi_noise/get_map_joint_phi_noise.sh
-> get_map_joint_phi_noise_1_realization.sh/.py, merged by merge_map_joint_phi_noise.py into
map_joint_phi_noise.npz, whose `nphi` is read by fisher_forecast (--phi_noise) and by
delensed_covariance (phi_noise).
"""

import glob
import os
import numpy as np
import jax.numpy as jnp
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.fields import Basis, fourier
from cmb_lensing.map_joint import map_joint
from cmb_lensing.simulate import load_sim
from cmb_lensing.constants import PARAM_ORDER, GROUND_TRUTH
from cmb_lensing.camb_spectra import (camb_cls_at_params, load_sim_cosmology, covar_from_cl,
                                      DEFAULT_POL)

#the filename every per-realization job writes, and the merged product
REALIZATION_GLOB = "map_joint_phi_noise_*.npz"
MERGED_NAME = "map_joint_phi_noise.npz"

#metadata copied from the job files into the merged file
_CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "map_joint_steps")
#map_joint iterations of the phi reconstruction (map_joint's own default)
DEFAULT_MAP_JOINT_STEPS = 30


def stored_pol(job):
    """The data polarization a job file was run at; files written before 2026-11-11 have no
    `pol` and are temperature-only."""
    return str(job["pol"]) if "pol" in job.files else DEFAULT_POL


def cross_power_grid(first_fourier, second_fourier, nside):
    """Re(a conj(b)) / nside^2 on the rfft grid; pass the same field twice for an auto.

    field_from_covar_single_key draws irfft2(rfft2(white) * sqrt(C_grid)) from real white
    noise, so E|rfft2(f)|^2 = nside^2 C_grid: dividing by nside^2 puts the result in the units
    of covar_matrix_from_cls.
    """
    product = first_fourier * jnp.conj(second_fourier)
    return np.asarray(jnp.real(product)) / nside**2


#fields.fourier applies rfft2 unconditionally, so the basis is checked rather than assumed
def _scalar_matrix(field):
    """The FOURIER-basis rfft2 matrix of a FlatS0, whatever basis it arrives in."""
    in_fourier = field if field.basis == Basis.FOURIER else fourier(field)
    return in_fourier.scalar_matrix


# ── One realization ───────────────────────────────────────────────────────

def measure_phi_noise(nside, theta_pix, noise_level, map_seed, l_knee = 0.0,
                      map_joint_steps = DEFAULT_MAP_JOINT_STEPS, pol = DEFAULT_POL):
    """One realization at GROUND_TRUTH: simulate data of polarization `pol` ("I", "P" or
    "IP"), reconstruct with map_joint, store the moments.

    Returns per-mode rfft-grid matrices `auto`, `cross`, `true_auto` and `cphi`, all in
    covar_matrix_from_cls units. map_joint's phi_hat is used as it comes (no rescaling).
    """
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = camb_cls_at_params(GROUND_TRUTH)
    cphi = np.asarray(covar_from_cl(nside, pix_width, ell_grid, cls["phi"]))

    data_set = load_sim(nside, theta_pix, pol, map_seed,
                        **load_sim_cosmology(GROUND_TRUTH),
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = cls)
    _, phi_hat = map_joint(data_set, num_steps = map_joint_steps)

    true_fourier = _scalar_matrix(data_set.phi)
    hat_fourier = _scalar_matrix(phi_hat)

    auto = cross_power_grid(hat_fourier, hat_fourier, nside)
    cross = cross_power_grid(hat_fourier, true_fourier, nside)
    true_auto = cross_power_grid(true_fourier, true_fourier, nside)

    return dict(auto = auto, cross = cross, true_auto = true_auto, cphi = cphi)


# ── Merging the realizations ──────────────────────────────────────────────

def _noise_from_r_squared(r_squared, mean_cross_with_truth, cphi):
    """N_phi = C_phi (1 / r^2 - 1); NaN where the mean cross power with the truth is not
    positive (no correlation measured), 0 at the origin."""
    with np.errstate(divide = "ignore", invalid = "ignore"):
        nphi = np.where(mean_cross_with_truth > 0, cphi * (1.0 / r_squared - 1.0), np.nan)
    nphi[0, 0] = 0.0
    return nphi


def noise_from_correlation(mean_estimate_power, mean_cross_with_truth, mean_true_power, cphi):
    """Per-mode N_phi from the squared correlation of the estimate with the truth,

        r^2 = <Re(phi_estimate phi_true*)>^2 / (<|phi_estimate|^2> <|phi_true|^2>).

    Valid for any estimate (unchanged if the estimate is rescaled), but every source of
    scatter in the estimate counts as noise, including a sampler's Monte Carlo error in a
    finite chain average.
    """
    with np.errstate(divide = "ignore", invalid = "ignore"):
        r_squared = mean_cross_with_truth**2 / (mean_estimate_power * mean_true_power)
    return _noise_from_r_squared(r_squared, mean_cross_with_truth, cphi)


def noise_from_response(mean_cross_with_truth, mean_true_power, cphi):
    """Per-mode N_phi from the estimate's response to the truth,

        r^2 = <Re(phi_estimate phi_true*)> / <|phi_true|^2>.

    Valid only when the estimate is the exact posterior mean E[phi | d], for which
    <Re(phi_estimate phi_true*)> = <|phi_estimate|^2>, so this equals noise_from_correlation.
    It never uses <|phi_estimate|^2>, so a chain average's Monte Carlo error (independent of
    the truth) drops out; a shrunk estimate (e.g. a chain not yet burnt in) reads as noise.
    Not valid for a MAP estimate.
    """
    with np.errstate(divide = "ignore", invalid = "ignore"):
        r_squared = mean_cross_with_truth / mean_true_power
    return _noise_from_r_squared(r_squared, mean_cross_with_truth, cphi)

def _noise_from_means(estimator, mean_estimate_power, mean_cross_with_truth, mean_true_power, cphi):
    if estimator == "response":
        return noise_from_response(mean_cross_with_truth, mean_true_power, cphi)
    return noise_from_correlation(mean_estimate_power, mean_cross_with_truth, mean_true_power, cphi)


def noise_with_jackknife(autos, crosses, true_autos, cphi, estimator = "correlation"):
    """(nphi, nphi_error, nphi_leave_one_out, mean_auto, mean_cross, mean_true_auto) from the
    stacked (n_realizations, nside, nside // 2 + 1) per-realization moments: `autos`
    |phi_estimate|^2, `crosses` Re(phi_estimate phi_true*), `true_autos` |phi_true|^2.

    The moments are averaged over realizations before the `estimator` ("correlation":
    noise_from_correlation, "response": noise_from_response) takes the ratio; `nphi_error` is a
    delete-one-realization jackknife through the same ratio and `nphi_leave_one_out` the
    (n_realizations, nside, nside // 2 + 1) N_phi it is built from (both NaN with fewer than
    3 realizations). Shared with sampler_phi_noise.
    """
    n_realization = len(autos)
    mean_auto = np.mean(autos, axis = 0)
    mean_cross = np.mean(crosses, axis = 0)
    mean_true_auto = np.mean(true_autos, axis = 0)
    nphi = _noise_from_means(estimator, mean_auto, mean_cross, mean_true_auto, cphi)

    #delete-one jackknife
    nphi_error = np.full(cphi.shape, np.nan)
    nphi_leave_one_out = np.full((n_realization,) + cphi.shape, np.nan)
    if n_realization > 2:
        leave_one_out = []
        for index in range(n_realization):
            kept = np.arange(n_realization) != index
            leave_one_out.append(_noise_from_means(estimator,
                                                   np.mean(autos[kept], axis = 0),
                                                   np.mean(crosses[kept], axis = 0),
                                                   np.mean(true_autos[kept], axis = 0),
                                                   cphi))
        nphi_leave_one_out = np.array(leave_one_out)
        spread = nphi_leave_one_out - np.mean(nphi_leave_one_out, axis = 0)
        nphi_error = np.sqrt((n_realization - 1) / n_realization *
                             np.sum(spread**2, axis = 0))
    return nphi, nphi_error, nphi_leave_one_out, mean_auto, mean_cross, mean_true_auto


def merge_phi_noise(directory):
    """Average the per-realization moments and form the per-mode N_phi, with a jackknife error.

    The error `nphi_error` is a delete-one-realization jackknife, recomputed through the
    same ratio as the answer, and `nphi_leave_one_out` holds the N leave-one-out N_phi it comes
    from. Returns the dict written to map_joint_phi_noise.npz.
    """
    paths = sorted(glob.glob(os.path.join(directory, REALIZATION_GLOB)))
    n_realization = len(paths)

    autos = []
    crosses = []
    true_autos = []
    seeds = []
    for path in paths:
        data = np.load(path)
        autos.append(data["auto"])
        crosses.append(data["cross"])
        true_autos.append(data["true_auto"])
        seeds.append(int(data["map_seed"]))
    autos = np.array(autos)
    crosses = np.array(crosses)
    true_autos = np.array(true_autos)

    first = np.load(paths[0])
    cphi = np.asarray(first["cphi"])

    nphi, nphi_error, nphi_leave_one_out, mean_auto, mean_cross, mean_true_auto = noise_with_jackknife(
        autos, crosses, true_autos, cphi)

    return dict(nphi = nphi, nphi_error = nphi_error, nphi_leave_one_out = nphi_leave_one_out,
                cphi = cphi,
                mean_auto = mean_auto, mean_cross = mean_cross,
                mean_true_auto = mean_true_auto,
                n_realizations = n_realization, seeds = np.array(seeds),
                params = np.array([GROUND_TRUTH[name] for name in PARAM_ORDER]),
                param_names = np.array(PARAM_ORDER), pol = stored_pol(first),
                **{key: first[key] for key in _CONFIG_KEYS})
