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

N_phi describes map_joint AT the number of steps it was run for, on THIS box, at the
fiducial cosmology; it is held fixed across the forecast's finite-difference stencil.

Fanned out one slurm job per realization by get_map_joint_phi_noise/get_map_joint_phi_noise.sh
-> get_map_joint_phi_noise_1_realization.sh/.py, merged by merge_map_joint_phi_noise.py into
map_joint_phi_noise.npz, whose `nphi` is read by fisher_forecast (--nphi_source map_joint
--phi_noise) and by delensed_covariance (phi_noise).
"""

import glob
import os
import numpy as np
import jax.numpy as jnp
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.fields import Basis, fourier
from cmb_lensing.map_joint import map_joint
from cmb_lensing.simulate import load_sim, covar_matrix_from_cls
from cmb_lensing.constants import PARAM_ORDER, GROUND_TRUTH
from cmb_lensing.fisher_forecast import camb_cls_at_params, load_sim_cosmology

#the filename every per-realization job writes, and the merged product
REALIZATION_GLOB = "map_joint_phi_noise_*.npz"
MERGED_NAME = "map_joint_phi_noise.npz"

#metadata copied from the job files into the merged file
_CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "map_joint_steps")


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
                      map_joint_steps = 30, verbose = True):
    """One realization at GROUND_TRUTH: simulate, reconstruct with map_joint, store the moments.

    Returns per-mode rfft-grid matrices `auto`, `cross`, `true_auto` and `cphi`, all in
    covar_matrix_from_cls units. map_joint's phi_hat is used as it comes (no rescaling).
    """
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = camb_cls_at_params(GROUND_TRUTH)

    phi_ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)
    cphi = np.asarray(covar_matrix_from_cls(nside, pix_width, ell_grid, phi_ells, cls["phi"],
                                            origin_value = 0))

    data_set = load_sim(nside, theta_pix, "I", map_seed,
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

def noise_from_moments(auto, cross, true_auto, cphi):
    """N_phi = C_phi (1 / r^2 - 1) per mode, r^2 = cross^2 / (auto true_auto).

    NaN where the mean cross power is not positive (no correlation with the truth measured);
    0 at the origin.
    """
    with np.errstate(divide = "ignore", invalid = "ignore"):
        r_squared = cross**2 / (auto * true_auto)
        nphi = np.where(cross > 0, cphi * (1.0 / r_squared - 1.0), np.nan)
    nphi[0, 0] = 0.0
    return nphi


def noise_with_jackknife(autos, crosses, true_autos, cphi):
    """(nphi, nphi_error, mean_auto, mean_cross, mean_true_auto) from the stacked
    (n_realizations, nside, nside // 2 + 1) per-realization moments.

    The moments are averaged over realizations before noise_from_moments takes the ratio;
    `nphi_error` is a delete-one-realization jackknife through the same ratio (NaN with
    fewer than 3 realizations). Shared with sampler_phi_noise.
    """
    n_realization = len(autos)
    mean_auto = np.mean(autos, axis = 0)
    mean_cross = np.mean(crosses, axis = 0)
    mean_true_auto = np.mean(true_autos, axis = 0)
    nphi = noise_from_moments(mean_auto, mean_cross, mean_true_auto, cphi)

    #delete-one jackknife
    nphi_error = np.full(cphi.shape, np.nan)
    if n_realization > 2:
        leave_one_out = []
        for index in range(n_realization):
            kept = np.arange(n_realization) != index
            leave_one_out.append(noise_from_moments(np.mean(autos[kept], axis = 0),
                                                    np.mean(crosses[kept], axis = 0),
                                                    np.mean(true_autos[kept], axis = 0),
                                                    cphi))
        leave_one_out = np.array(leave_one_out)
        spread = leave_one_out - np.mean(leave_one_out, axis = 0)
        nphi_error = np.sqrt((n_realization - 1) / n_realization *
                             np.sum(spread**2, axis = 0))
    return nphi, nphi_error, mean_auto, mean_cross, mean_true_auto


def merge_phi_noise(directory):
    """Average the per-realization moments and form the per-mode N_phi, with a jackknife error.

    The error `nphi_error` is a delete-one-realization jackknife, recomputed through the
    same ratio as the answer. Returns the dict written to map_joint_phi_noise.npz.
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

    nphi, nphi_error, mean_auto, mean_cross, mean_true_auto = noise_with_jackknife(
        autos, crosses, true_autos, cphi)

    return dict(nphi = nphi, nphi_error = nphi_error, cphi = cphi,
                mean_auto = mean_auto, mean_cross = mean_cross,
                mean_true_auto = mean_true_auto,
                n_realizations = n_realization, seeds = np.array(seeds),
                params = np.array([GROUND_TRUTH[name] for name in PARAM_ORDER]),
                param_names = np.array(PARAM_ORDER),
                **{key: first[key] for key in _CONFIG_KEYS})
