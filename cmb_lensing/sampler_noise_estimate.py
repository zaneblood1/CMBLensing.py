"""The sampler's phi reconstruction noise, bounded from the Fisher information of p(d | phi).

    p(d | phi) = N(0, Sigma(phi)),   Sigma(phi) = M B L(phi) C_f L(phi)^T B^T M^T + C_n
    F_phi = Var_d[ score ],          score = d/dphi log p(d | phi)

WHY THIS EXISTS. Every forecast in this package writes the lensing block as C_phi + N_phi with
N_phi from a quadratic estimator, but sample_joint does not run a quadratic estimator: it
samples the exact posterior, whose posterior mean can only be as noisy as the information in
the full likelihood allows. The van Trees (Bayesian Cramer-Rao) inequality makes that precise
per mode,

    N_sampler(k) >= N_bound(k) = 1 / <F_phi(k)>_phi,

with the average over phi drawn from its prior. This module estimates N_bound on the box the
chains run on - exact LenseFlow lensing, the periodic square domain - without running the
sampler. Measured at nside 64 / 5' (2026-10-01, 5 phi x 100 draws): N_bound / N_QE-iterated is
0.90-0.94 at L 500-1100 and falls to ~0.8 at L 2500, at 2.5 / 5 / 10 uK alike.

HOW. By the Fisher identity the score is E_{f | d, phi}[d/dphi log p(d | f, phi)], and since
that is quadratic in f it equals grad_phi logpdf at the Wiener-filtered f plus a term with no
data dependence (Carron & Lewis 2017, eqs. 2.7-2.14: the quadratic piece minus the "mean
field"). A constant does not change a variance, so diag F_phi is the per-mode sample variance,
over data drawn at a FIXED phi, of grad_phi_logpdf(f_WF(d, phi), phi). The phi prior's gradient
is the same constant at every draw and drops out too. Each job takes one phi (load_sim's at its
seed) and draws `n_draws` fresh (f, n) data sets at it; the merge averages F over jobs (= over
phi) and inverts per mode.

NORMALIZATION. grad_phi_logpdf comes out in the field module's inner-product convention; the
per-mode factor a_k linking it to a C^-1 phi gradient is read off the prior's gradient
(g_prior = -a_k phi_k / C_k, from the same function at f = d = 0). It measures a_k = 1 to
1e-16 on this codebase's convention, but is stored per job and applied rather than assumed.
With E|phi_k|^2 = nside^2 C_k (field_from_covar_single_key) the noise in covar_matrix_from_cls
units is N_k = a_k^2 nside^2 / Var(score_k).



CALIBRATION. At phi = 0 the bound must equal the quadratic estimator's N0 built with the
UNLENSED filter and the UNLENSED response, exactly: the QE is the optimal estimator there
(Hirata & Seljak 2003). A calibration job (phi = 0) checks the whole chain - Wiener filter,
gradient, normalization - and the merge reports the ratio per band. Measured at nside 64 / 5':
0.99-1.03 over L 500-2900, but 1.07 at L 225 and 1.2-1.3 at L 75 (4-22 modes) - an unexplained
low-L excess, so treat the lowest bands of N_bound with suspicion.

SELF-CONJUGATE COLUMNS. kx = 0 and kx = Nyquist hold the anti-Hermitian gradient content that
a per-mode complex variance does not describe (CLAUDE.md, "The Phi Gradient in Fourier Space"),
so they are excluded from every band and their N in the merged matrix is the QE's own value
there times the band ratio N_bound / N_QE at their |L|.

IT IS A BOUND. The sampler's actual posterior-mean noise sits at or above N_bound; a forecast
built on it is slightly optimistic by construction.

Per-job files store running SUMS (sum of scores, sum of |score|^2) so a checkpoint is always a
valid partial measurement; the variance is formed at the merge. HPC path:
sampling_chains/get_sampler_noise_estimate.sh -> get_sampler_noise_estimate_1_phi_realization.sh
-> get_sampler_noise_estimate_1_phi_realization.py -> merge_sampler_noise_estimate.py. Locally:

    python -m cmb_lensing.sampler_noise_estimate --nside 64 --theta_pix 5 --noise_level 5 \
        --n_phi 5 --n_draws 100 --out_dir <dir>
"""
import argparse
import glob
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.constants import *
from cmb_lensing.fields import *
from cmb_lensing.lense_flow import *
from cmb_lensing.simulate import load_sim, field_from_covar_single_key, covar_matrix_from_cls
from cmb_lensing.gradients import grad_phi_logpdf
from cmb_lensing.wiener_filter import wiener_filter
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER
from cmb_lensing.delensed_spectrum import _to_fourier, _lense
from cmb_lensing.fisher_forecast import (camb_cls_at_params, load_sim_cosmology,
                                         qe_noise_matrix, measured_phi_noise_cl)

#Wiener-filter CG tolerance. Measured at nside 64 / 5': the score's rms error against
#tol = 1e-12 is 4e-3 at map_joint's 1e-1 and 1.4e-6 at 1e-8, for 0.57 vs 0.69 s per draw
DEFAULT_TOL = 1e-8
DEFAULT_N_DRAWS = 50
DEFAULT_DELTA_ELL = 100.0
#rewrite the job file every this many draws (atomically), so a job killed at the wall clock
#keeps every finished draw
DEFAULT_CHECKPOINT_EVERY = 5
#the draws' random stream, folded into the map seed's key. A large constant keeps the draws
#off load_sim's own split(PRNGKey(seed), 100) keys: under partitionable threefry
#fold_in(k, i) for a small i reproduces split(k, n)[i] (see CLAUDE.md, the Louis chains)
SCORE_DRAW_STREAM = 0x5C0E5
JOB_PREFIX = "sampler_noise"
CALIBRATION_NAME = "sampler_noise_calibration.npz"
MERGED_NAME = "sampler_noise_estimate.npz"
#everything two job files must agree on before their Fisher matrices may be averaged
CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "beam_fwhm", "tol")


# ── One phi realization ────────────────────────────────────────────────────

def simulate_base(nside, theta_pix, noise_level, seed, params, l_knee = 0.0):
    return load_sim(nside, theta_pix, "I", seed, **load_sim_cosmology(params),
                    uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                    precomputed_cls = camb_cls_at_params(params))


def draw_data(data_set, phi, key):
    """d = M B L(phi) f + n with fresh f ~ N(0, C_f) and n ~ N(0, C_n), phi held fixed."""
    nside = data_set.data.nside
    key_f, key_n = jax.random.split(key)
    template = _to_fourier(data_set.unlensed_field)
    field = template.replace(scalar_matrix = jnp.fft.rfft2(field_from_covar_single_key(
        nside, data_set.field_covariance.scalar_matrix, key_f)))
    noise = template.replace(scalar_matrix = jnp.fft.rfft2(field_from_covar_single_key(
        nside, data_set.noise_covariance.scalar_matrix, key_n)))
    lensed = _lense(field, phi, FORWARD_LENSE)
    return data_set.mask * data_set.beam * lensed + noise


def score(data_set, phi, data, tol = DEFAULT_TOL):
    """grad_phi logpdf at the Wiener-filtered f: the data-dependent part of the score."""
    f_wf = wiener_filter(0 * data, phi, data, data_set.field_covariance,
                         data_set.noise_covariance, data_set.mask, data_set.beam, tol = tol)
    return np.asarray(grad_phi_logpdf(f_wf, phi, data, data_set.noise_covariance,
                                      data_set.phi_covariance, data_set.field_covariance,
                                      data_set.mask, data_set.beam).scalar_matrix)


def prior_factor(data_set, phi):
    """a_k with g_prior = -a_k phi_k / C_k, from grad_phi_logpdf at f = d = 0 (no data term).
    `phi` must be nonzero (a calibration job passes the TRUE phi here, then zeroes it)."""
    zero = 0 * data_set.data
    g_prior = np.asarray(grad_phi_logpdf(zero, phi, zero, data_set.noise_covariance,
                                         data_set.phi_covariance, data_set.field_covariance,
                                         data_set.mask, data_set.beam).scalar_matrix)
    cphi = np.asarray(data_set.phi_covariance.scalar_matrix)
    phik = np.asarray(phi.scalar_matrix)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.real(-g_prior * cphi / phik)


def draw_key(map_seed, draw_index):
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(map_seed),
                                                 SCORE_DRAW_STREAM), draw_index)


def measure_score_sums(nside, theta_pix, noise_level, param_ground, map_seed,
                       n_draws = DEFAULT_N_DRAWS, l_knee = 0.0, tol = DEFAULT_TOL,
                       calibration = False, on_checkpoint = None,
                       checkpoint_every = DEFAULT_CHECKPOINT_EVERY, verbose = True):
    """One phi (load_sim's at `map_seed`, or zero for `calibration`), `n_draws` data draws at
    it, and the running per-mode sums the merge forms the score variance from.

    Returns a dict: `score_sum` (complex, rfft grid), `score_abs2_sum`, `n_done`,
    `prior_factor` (a_k), `phi_power` (|phi_k|^2 of the true phi, zero for calibration) and
    `cphi` (the phi covariance matrix the jobs ran with). `on_checkpoint(result)` is called
    every `checkpoint_every` draws and after the last one.
    """
    data_set = simulate_base(nside, theta_pix, noise_level, map_seed, param_ground,
                             l_knee = l_knee)
    phi = _to_fourier(data_set.phi)
    a = prior_factor(data_set, phi)
    if calibration:
        phi = phi.replace(scalar_matrix = 0 * phi.scalar_matrix)
    shape = np.asarray(phi.scalar_matrix).shape
    result = dict(score_sum = np.zeros(shape, dtype = complex),
                  score_abs2_sum = np.zeros(shape), n_done = 0, prior_factor = a,
                  phi_power = np.abs(np.asarray(phi.scalar_matrix))**2,
                  cphi = np.asarray(data_set.phi_covariance.scalar_matrix))

    start = time.time()
    for index in range(n_draws):
        s = score(data_set, phi, draw_data(data_set, phi, draw_key(map_seed, index)), tol)
        result["score_sum"] += s
        result["score_abs2_sum"] += np.abs(s)**2
        result["n_done"] = index + 1
        if verbose and (index == 0 or (index + 1) % 10 == 0 or index + 1 == n_draws):
            print(f"  draw {index + 1}/{n_draws}: {(time.time() - start) / (index + 1):.2f} "
                  f"s per draw", flush = True)
        if on_checkpoint is not None and ((index + 1) % checkpoint_every == 0
                                          or index + 1 == n_draws):
            on_checkpoint(result)
    return result


def job_file_name(realization_index):
    return f"{JOB_PREFIX}_{realization_index:04d}.npz"


def save_atomic(path, **arrays):
    """np.savez through a hidden temporary + os.replace, so a file is never half written."""
    directory, name = os.path.split(path)
    temporary = os.path.join(directory, f".{name}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


# ── Collecting the job files ───────────────────────────────────────────────

def _config(data):
    return {key: (int(data[key]) if key == "nside" else float(data[key]))
            for key in CONFIG_KEYS}


def load_score_directory(directory, verbose = True):
    """Every phi-realization job file in `directory` (calibration excluded), as dicts.

    Refuses mixed configurations, mixed cosmologies and duplicate seeds. A file with fewer
    than two draws carries no variance and is set aside; an unfinished file with two or more
    is used as the partial measurement it is (its variance is unbiased at any length).
    """
    paths = sorted(glob.glob(os.path.join(directory, f"{JOB_PREFIX}_[0-9]*.npz")))
    if not paths:
        raise FileNotFoundError(f"no {JOB_PREFIX}_NNNN.npz job files in {directory}")
    jobs, seeds, reference = [], set(), None
    for path in paths:
        data = dict(np.load(path, allow_pickle = True))
        config = (_config(data), tuple(np.round(np.asarray(data["params"]), 12)))
        if reference is None:
            reference = config
        elif config != reference:
            raise ValueError(f"{path} was run at {config}, but the first file at "
                             f"{reference}; refusing to average different configurations")
        seed = int(data["map_seed"])
        if seed in seeds:
            raise ValueError(f"duplicate map_seed {seed} ({path}): the same phi would be "
                             f"counted twice")
        seeds.add(seed)
        if int(data["n_done"]) < 2:
            if verbose:
                print(f"  setting aside {os.path.basename(path)}: {int(data['n_done'])} "
                      f"draw(s), no variance yet")
            continue
        if verbose and not bool(data.get("finished", True)):
            print(f"  {os.path.basename(path)} is unfinished: using its "
                  f"{int(data['n_done'])}/{int(data['n_draws'])} draws")
        jobs.append(data)
    if not jobs:
        raise ValueError(f"no job in {directory} has two or more draws yet")
    return jobs


def fisher_from_sums(data):
    """Per-mode Fisher information in 1 / (covar_matrix_from_cls units), from a job's sums:
    Var(score) / (a^2 nside^2), the unbiased (n - 1) variance."""
    n = int(data["n_done"])
    mean = np.asarray(data["score_sum"]) / n
    var = (np.asarray(data["score_abs2_sum"]) - n * np.abs(mean)**2) / (n - 1)
    a = np.asarray(data["prior_factor"])
    with np.errstate(divide = "ignore", invalid = "ignore"):
        return var / (a**2 * int(data["nside"])**2)


def used_modes(ell_grid):
    """The modes every band and the per-mode estimate use: no origin, no self-conjugate
    column (see the module docstring)."""
    use = np.asarray(ell_grid) > 0
    use[:, 0] = False
    use[:, -1] = False
    return use


def band_index(ell_grid, edges, use):
    ells = np.asarray(ell_grid)
    index = np.digitize(ells, edges) - 1
    return np.where(use & (index >= 0) & (index < len(edges) - 1), index, -1)


def band_harmonic(matrix, index, n_band):
    """The INFORMATION-weighted band value 1 / <1/N> over the band's modes - the mean that
    commutes with averaging Fisher information, used for every N in this module."""
    inside = (index >= 0) & np.isfinite(matrix) & (matrix > 0)
    total = np.bincount(index[inside], weights = 1.0 / matrix[inside], minlength = n_band)
    count = np.bincount(index[inside], minlength = n_band)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.where(count > 0, count / total, np.nan), count


def qe_load_sim_matrix(param_ground, nside, theta_pix, noise_level, l_knee = 0.0):
    """The 1st-principles N_phi: load_sim's own quadratic_estimate WITHOUT the NPHI_FAC
    preconditioning factor - scalar_quadratic_estimate with the unlensed response and the
    lensed filter, i.e. fisher_forecast.qe_noise_matrix at its defaults."""
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = camb_cls_at_params(param_ground)
    return np.asarray(qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                                      0.0, 10_000))


def qe_calibration_matrix(param_ground, nside, theta_pix, noise_level, l_knee = 0.0):
    """The QE N0 the phi = 0 bound must reproduce: UNLENSED filter and UNLENSED response."""
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = camb_cls_at_params(param_ground)
    return np.asarray(qe_noise_matrix(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                                      0.0, 10_000, filter_tt = cls["scalar_TT"],
                                      qe_response = "unlensed"))


def load_map_joint_noise(path, nside, theta_pix, noise_level):
    """map_joint's empirical N_eff on the rfft grid, from either product that measures it:

    - a merge_delensed_covariance.py npz: the per-mode `phi_noise_fid` (C_phi (1/r^2 - 1)),
      NaN where `phi_measured_fid` is False;
    - a merge_phi_noise.py npz: the band N_L^eff, put on the grid through
      measured_phi_noise_cl + covar_matrix_from_cls (isotropic).

    Refuses a file measured on a different box. Returns (matrix, label).
    """
    data = dict(np.load(path, allow_pickle = True))
    measured = (int(data["nside"]), float(data["theta_pix"]), float(data["noise_level"]))
    if measured != (int(nside), float(theta_pix), float(noise_level)):
        raise ValueError(f"{path} was measured at (nside, theta_pix, noise) = {measured}, "
                         f"not {(nside, theta_pix, noise_level)}")
    if "phi_noise_fid" in data:
        matrix = np.asarray(data["phi_noise_fid"], dtype = float).copy()
        if "phi_measured_fid" in data:
            matrix[~np.asarray(data["phi_measured_fid"], dtype = bool)] = np.nan
        return matrix, f"map_joint N_eff, per mode ({int(data['n_realizations'])} real.)"
    if "n_eff" in data:
        ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
        ells = np.arange(2, 2 + int(np.ceil(np.max(np.asarray(ell_grid)))) + 2,
                         dtype = np.float64)
        cl = measured_phi_noise_cl(data, ells)
        matrix = np.asarray(covar_matrix_from_cls(nside, pix_width, ell_grid,
                                                  jnp.asarray(ells), jnp.asarray(cl),
                                                  origin_value = 0))
        return matrix, f"map_joint N_eff, bands ({int(data['n_realizations'])} real.)"
    raise ValueError(f"{path} is neither a merge_delensed_covariance.py nor a "
                     f"merge_phi_noise.py product (no phi_noise_fid / n_eff)")


def merge_sampler_noise(directory, delta_ell = DEFAULT_DELTA_ELL, verbose = True):
    """Average the jobs' Fisher information over phi and invert it per mode.

    Returns a dict ready for np.savez: `nphi` (the full rfft-grid matrix the forecast loads -
    per-mode 1/<F> on the used modes, the QE-shape fill on the self-conjugate columns, 0 at
    the origin), `nphi_raw` (NaN off the used modes), `nphi_jackknife_error` (per mode,
    delete-one-phi), `nphi_qe` (the load_sim matrix without NPHI_FAC), band spectra of both
    with the jackknife error, the calibration ratio if a calibration file is present, and
    the configuration.
    """
    jobs = load_score_directory(directory, verbose = verbose)
    first = jobs[0]
    config = _config(first)
    nside, theta_pix = config["nside"], config["theta_pix"]
    noise_level, l_knee = config["noise_level"], config["l_knee"]
    params = np.asarray(first["params"])
    param_ground = {name: float(params[i]) for i, name in enumerate(PARAM_ORDER)}
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    ell_grid = np.asarray(ell_grid)
    use = used_modes(ell_grid)

    #each job's per-mode Fisher, weighted by its n - 1 (an unbiased variance at any n, so
    #the weights only set the noise, never the mean)
    fishers = np.array([fisher_from_sums(job) for job in jobs])
    weights = np.array([int(job["n_done"]) - 1 for job in jobs], dtype = float)
    fisher = np.tensordot(weights, fishers, axes = 1) / weights.sum()

    def invert(f):
        with np.errstate(divide = "ignore", invalid = "ignore"):
            return np.where(use & (f > 0), 1.0 / f, np.nan)

    nphi_raw = invert(fisher)
    #delete-one-phi jackknife: per mode, and (below) per band
    n_jobs = len(jobs)
    leave_one_out = []
    if n_jobs > 2:
        for j in range(n_jobs):
            keep = np.arange(n_jobs) != j
            leave_one_out.append(invert(np.tensordot(weights[keep], fishers[keep], axes = 1)
                                        / weights[keep].sum()))
        leave_one_out = np.array(leave_one_out)
        jackknife = np.sqrt((n_jobs - 1) / n_jobs * np.sum(
            (leave_one_out - leave_one_out.mean(axis = 0))**2, axis = 0))
    else:
        jackknife = np.full(nphi_raw.shape, np.nan)

    nphi_qe = qe_load_sim_matrix(param_ground, nside, theta_pix, noise_level, l_knee)
    edges = np.arange(0.0, float(ell_grid.max()) + delta_ell, delta_ell)
    index = band_index(ell_grid, edges, use)
    n_band = len(edges) - 1
    centres = 0.5 * (edges[1:] + edges[:-1])
    band_bound, band_count = band_harmonic(nphi_raw, index, n_band)
    band_qe, _ = band_harmonic(np.where(use, nphi_qe, np.nan), index, n_band)
    if n_jobs > 2:
        bands_loo = np.array([band_harmonic(m, index, n_band)[0] for m in leave_one_out])
        band_error = np.sqrt((n_jobs - 1) / n_jobs * np.sum(
            (bands_loo - np.nanmean(bands_loo, axis = 0))**2, axis = 0))
    else:
        band_error = np.full(n_band, np.nan)

    #the full matrix: per-mode where measured, and on the self-conjugate columns (and any
    #unmeasured mode) the QE's own value times the band ratio at that |L| - N_bound / N_QE is
    #smooth in L, the QE carries the box's anisotropy
    ratio = band_bound / band_qe
    good = np.isfinite(ratio)
    fill_ratio = np.interp(ell_grid, centres[good], ratio[good])
    nphi = np.where(np.isfinite(nphi_raw), nphi_raw, nphi_qe * fill_ratio)
    nphi[0, 0] = 0.0
    filled = int(np.sum(~np.isfinite(nphi_raw)) - 1)

    merged = dict(nphi = nphi, nphi_raw = nphi_raw, nphi_jackknife_error = jackknife,
                  nphi_qe = nphi_qe, fisher = fisher, cphi = np.asarray(first["cphi"]),
                  band_edges = edges, band_ells = centres, band_count = band_count,
                  band_nphi = band_bound, band_nphi_error = band_error,
                  band_nphi_qe = band_qe, n_realizations = n_jobs,
                  n_draws_total = int(sum(int(job["n_done"]) for job in jobs)),
                  seeds = np.array([int(job["map_seed"]) for job in jobs]),
                  params = params, param_names = np.array(PARAM_ORDER),
                  delta_ell = delta_ell, filled_modes = filled, **config)

    calibration_path = os.path.join(directory, CALIBRATION_NAME)
    if os.path.exists(calibration_path):
        calibration = dict(np.load(calibration_path, allow_pickle = True))
        if _config(calibration) != config:
            raise ValueError(f"{calibration_path} was run at {_config(calibration)}, not "
                             f"{config}")
        cal_matrix = invert(fisher_from_sums(calibration))
        cal_band, _ = band_harmonic(cal_matrix, index, n_band)
        cal_qe = qe_calibration_matrix(param_ground, nside, theta_pix, noise_level, l_knee)
        cal_qe_band, _ = band_harmonic(np.where(use, cal_qe, np.nan), index, n_band)
        merged.update(calibration_ratio = cal_band / cal_qe_band,
                      calibration_draws = int(calibration["n_done"]))

    if verbose:
        _report(merged)
    return merged


def _report(merged):
    print(f"\nsampler noise bound: nside {merged['nside']}, {merged['theta_pix']:g}', "
          f"{merged['noise_level']:g} uK-arcmin; {merged['n_realizations']} phi "
          f"realizations, {merged['n_draws_total']} draws; {merged['filled_modes']} "
          f"self-conjugate / unmeasured modes filled from the QE shape")
    has_cal = "calibration_ratio" in merged
    header = "  calibration (should be 1)" if has_cal else ""
    print(f"  {'L':>6s} {'modes':>5s}  N_bound / N_QE(load_sim, no NPHI_FAC){header}")
    for b, ell in enumerate(merged["band_ells"]):
        if not merged["band_count"][b] or not np.isfinite(merged["band_nphi"][b]):
            continue
        ratio = merged["band_nphi"][b] / merged["band_nphi_qe"][b]
        error = merged["band_nphi_error"][b] / merged["band_nphi_qe"][b]
        cal = (f"   {merged['calibration_ratio'][b]:.3f}" if has_cal else "")
        print(f"  {ell:6.0f} {merged['band_count'][b]:5d}  {ratio:.3f} +/- {error:.3f}"
              f"{cal}")


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_matrices(merged, map_joint, path):
    """N_bound, map_joint's N_eff and the 1st-principles QE on the rfft grid (shared log
    scale), and their pairwise ratios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    nside, theta_pix = int(merged["nside"]), float(merged["theta_pix"])
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    l_fund = 2 * np.pi / (nside * pix_width)
    extent = (-l_fund / 2, l_fund * (nside // 2 + 0.5),
              -l_fund * (nside // 2 + 0.5), l_fund * (nside // 2 - 0.5))
    valid = np.asarray(ell_grid) > 0

    matrices = {"sampler bound 1/<F>": np.asarray(merged["nphi"]),
                "QE (load_sim, no NPHI_FAC)": np.asarray(merged["nphi_qe"])}
    if map_joint is not None:
        matrices["map_joint N_eff"] = map_joint[0]
    names = list(matrices)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    n_cols = max(len(names), len(pairs))
    figure, axes = plt.subplots(2, n_cols, figsize = (5.2 * n_cols, 9), squeeze = False)

    def show(axis, matrix, norm, cmap, title, label):
        image = axis.imshow(np.fft.fftshift(np.where(valid, matrix, np.nan), axes = 0),
                            origin = "lower", extent = extent, aspect = "auto", norm = norm,
                            cmap = cmap, interpolation = "nearest")
        axis.set_title(title, fontsize = 9)
        axis.set_xlabel(r"$L_x$")
        axis.set_ylabel(r"$L_y$")
        figure.colorbar(image, ax = axis, fraction = 0.046, pad = 0.03).set_label(
            label, fontsize = 8)

    stacked = np.concatenate([m[valid & np.isfinite(m) & (m > 0)]
                              for m in matrices.values()])
    norm = LogNorm(vmin = np.percentile(stacked, 0.5), vmax = np.percentile(stacked, 99.5))
    for axis, name in zip(axes[0], names):
        show(axis, matrices[name], norm, "viridis", name, r"$N_\phi$ (covar units)")
    for axis in axes[0][len(names):]:
        axis.axis("off")
    for axis, (a, b) in zip(axes[1], pairs):
        ratio = matrices[a] / matrices[b]
        finite = ratio[valid & np.isfinite(ratio)]
        spread = min(max(np.percentile(np.abs(finite - 1), 99), 1e-3), 0.99)
        show(axis, ratio, TwoSlopeNorm(vcenter = 1.0, vmin = 1 - spread, vmax = 1 + spread),
             "RdBu_r", f"{a} / {b}", "ratio")
    for axis in axes[1][len(pairs):]:
        axis.axis("off")
    figure.suptitle(_title(merged, map_joint), fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.95))
    figure.savefig(path, dpi = 120)
    plt.close(figure)


def plot_spectra(merged, map_joint, path):
    """Band spectra (information-weighted, self-conjugate columns excluded) of the three N,
    C_phi for scale, and their ratios to the QE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ell_grid, _ = gen_ell_grid(int(merged["nside"]), float(merged["theta_pix"]))
    ell_grid = np.asarray(ell_grid)
    use = used_modes(ell_grid)
    edges = np.asarray(merged["band_edges"])
    index = band_index(ell_grid, edges, use)
    n_band = len(edges) - 1
    ells = np.asarray(merged["band_ells"])
    cphi_band, _ = band_harmonic(np.where(use, np.asarray(merged["cphi"]), np.nan), index,
                                 n_band)
    qe = np.asarray(merged["band_nphi_qe"])
    bound, bound_error = np.asarray(merged["band_nphi"]), np.asarray(merged["band_nphi_error"])
    keep = np.isfinite(bound) & np.isfinite(qe)

    figure, (top, bottom) = plt.subplots(2, 1, figsize = (9, 9), sharex = True,
                                         gridspec_kw = dict(height_ratios = [2, 1]))
    top.loglog(ells[keep], cphi_band[keep], color = "0.4", label = r"$C_L^{\phi\phi}$")
    top.loglog(ells[keep], qe[keep], color = "C3", linestyle = "--",
               label = "QE (load_sim, no NPHI_FAC)")
    top.errorbar(ells[keep], bound[keep], yerr = bound_error[keep], color = "C0", marker = "o",
                 markersize = 3, capsize = 2, linestyle = "none",
                 label = f"sampler bound 1/<F> ({int(merged['n_realizations'])} phi, "
                         f"{int(merged['n_draws_total'])} draws)")
    bottom.axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    bottom.errorbar(ells[keep], bound[keep] / qe[keep], yerr = bound_error[keep] / qe[keep],
                    color = "C0", marker = "o", markersize = 3, capsize = 2,
                    label = "sampler bound / QE")
    if map_joint is not None:
        mj_band, _ = band_harmonic(np.where(use, map_joint[0], np.nan), index, n_band)
        good = keep & np.isfinite(mj_band)
        top.loglog(ells[good], mj_band[good], color = "C2", marker = "s", markersize = 3,
                   linestyle = ":", label = map_joint[1])
        bottom.plot(ells[good], mj_band[good] / qe[good], color = "C2", marker = "s",
                    markersize = 3, linestyle = ":", label = "map_joint / QE")
    if "calibration_ratio" in merged:
        cal = np.asarray(merged["calibration_ratio"])
        good = keep & np.isfinite(cal)
        bottom.plot(ells[good], cal[good], color = "C4", marker = ".", linestyle = "-",
                    linewidth = 0.8, label = "calibration: phi = 0 bound / unlensed QE "
                                             "(should be 1)")
    top.set_ylabel(r"$N_L$ (covar units, information-weighted band mean)")
    top.legend(fontsize = 8)
    bottom.set_xscale("log")
    bottom.set_xlabel(r"$L$")
    bottom.set_ylabel("ratio to QE")
    bottom.legend(fontsize = 8)
    figure.suptitle(_title(merged, map_joint), fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.95))
    figure.savefig(path, dpi = 140)
    plt.close(figure)


def _title(merged, map_joint):
    return (f"phi reconstruction noise | nside {int(merged['nside'])}, "
            f"{float(merged['theta_pix']):g}', {float(merged['noise_level']):g} uK-arcmin"
            + ("" if map_joint is not None else " | (no map_joint file given)"))


# ── Local runner ───────────────────────────────────────────────────────────

def run_job(out_dir, realization_index, map_seed, nside, theta_pix, noise_level,
            param_ground = None, n_draws = DEFAULT_N_DRAWS, l_knee = 0.0, tol = DEFAULT_TOL,
            calibration = False, checkpoint_every = DEFAULT_CHECKPOINT_EVERY):
    """One job, checkpointed to `out_dir` - what the HPC python script calls."""
    param_ground = dict(GROUND_TRUTH) if param_ground is None else param_ground
    os.makedirs(out_dir, exist_ok = True)
    path = os.path.join(out_dir, CALIBRATION_NAME if calibration
                        else job_file_name(realization_index))
    metadata = dict(realization_index = realization_index, map_seed = map_seed,
                    nside = nside, theta_pix = theta_pix, noise_level = noise_level,
                    l_knee = l_knee, beam_fwhm = 0.0, tol = tol, n_draws = n_draws,
                    calibration = bool(calibration),
                    params = np.array([param_ground[name] for name in PARAM_ORDER]),
                    param_names = np.array(PARAM_ORDER))

    def checkpoint(result):
        save_atomic(path, finished = result["n_done"] == n_draws, **metadata, **result)

    print(f"{'calibration (phi = 0)' if calibration else f'phi realization {realization_index}'}"
          f": seed {map_seed}, nside {nside}, {theta_pix:g}', {noise_level:g} uK-arcmin, "
          f"{n_draws} draws", flush = True)
    measure_score_sums(nside, theta_pix, noise_level, param_ground, map_seed,
                       n_draws = n_draws, l_knee = l_knee, tol = tol,
                       calibration = calibration, on_checkpoint = checkpoint,
                       checkpoint_every = checkpoint_every)
    print(f"wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nside", type = int, default = 64)
    parser.add_argument("--theta_pix", type = float, default = 5.0)
    parser.add_argument("--noise_level", type = float, default = 5.0)
    parser.add_argument("--l_knee", type = float, default = 0.0)
    parser.add_argument("--n_phi", type = int, default = 5)
    parser.add_argument("--n_draws", type = int, default = DEFAULT_N_DRAWS)
    parser.add_argument("--tol", type = float, default = DEFAULT_TOL)
    parser.add_argument("--seed_prefix", type = int, default = 1235)
    parser.add_argument("--no_calibration", action = "store_true")
    parser.add_argument("--out_dir", type = str, required = True)
    args = parser.parse_args()

    common = dict(nside = args.nside, theta_pix = args.theta_pix,
                  noise_level = args.noise_level, n_draws = args.n_draws,
                  l_knee = args.l_knee, tol = args.tol)
    if not args.no_calibration:
        run_job(args.out_dir, 0, args.seed_prefix - 1, calibration = True, **common)
    for m in range(args.n_phi):
        run_job(args.out_dir, m, args.seed_prefix + m, **common)
    merged = merge_sampler_noise(args.out_dir)
    np.savez(os.path.join(args.out_dir, MERGED_NAME), **merged)


if __name__ == "__main__":
    main()
