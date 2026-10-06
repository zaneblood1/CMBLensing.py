"""The sampler's EFFECTIVE phi reconstruction noise, measured from its own posterior draws.

    phi_mean(d) = <phi>_{phi | d, theta_0}          (the chain average after burn-in)
    r_k^2       = <B>^2 / (<A> <D>),   A = |phi_mean|^2, B = Re(phi_mean phi_true*), D = |phi_true|^2
    N_k         = C_k (1 / r_k^2 - 1)

WHY THIS EXISTS. sampler_noise_estimate.py bounds the sampler's noise from the Fisher
information of p(d | phi), which turned out to be the "every other phi mode known" floor and
not the noise the sampler reaches. This module measures the noise itself: one chain per
(f, phi, d) realization draws (f, phi) | d with theta HELD at the ground truth, its phi
samples are averaged into the posterior mean, and that mean is cross correlated with the
realization's true phi. The moments are summed over REALIZATIONS per mode before the ratio is
taken - one mode of one realization has no correlation coefficient (it is the cosine of a
phase difference, and C tan^2 of that has an infinite mean).

THE CHAIN is fisher_forecast_from_mixed_logpdf.posterior_mixed_draws: sample_lcdm.sample_joint's
own Gibbs sweep (gibbs_sample_f, mix, one phi HMC step, unmix) minus its theta step, with C_f /
C_phi / D / G built at the ground truth the way the sampler builds them and the chain started
at phi = 0 (f needs no start: it is redrawn from its exact conditional every sweep). Every
phi sample is stored, un-thinned and with no burn-in cut; both are analysis-time choices.

FOUR ESTIMATES, which must agree if the chains are long enough and the sampler is exact:
    cross     r^2 = <B>^2 / (<A_x> <D>) with A_x the cross power between the means of
              disjoint BLOCKS of the chain. The headline. The Monte Carlo errors of two blocks
              are independent (once a block is much longer than the IAT), so A_x is free of
              the Var_post / n_eff that the plain |phi_mean|^2 carries.
    naive     the same with A = |phi_mean|^2: biased HIGH in N by ~(1 + N / C) / n_eff.
    b_only    r^2 = <B> / <D>. For an exact posterior mean <phi_mean phi_true*> =
              <|phi_mean|^2>, so this needs no A at all and is linear in the chain mean.
    variance  r^2 = 1 - <V> / C with V the chain's per-mode posterior variance, since
              <V> = C N / (C + N). Uses no truth at all, but is the most fragile where
              N >> C: r^2 is then a small difference, and a ~IAT / n error in V moves it.
The delete-one-REALIZATION jackknife gives the error of each. It does not see a bias common
to every realization, which is what the cross / naive / b_only / variance spread, the
first-half vs second-half split and the burn-in scan are for.

BANDS. Moments are whitened by C_k before they are summed over a band's modes, so the steep
C_L does not let the band's lowest L dominate, and the band's (1 / r^2 - 1) is quoted RELATIVE
to the QE's own under the same band functional (band_reference), times the information-weighted
(harmonic) band mean of N_QE that sampler_noise_estimate.py uses. A sampler whose noise is the
QE's therefore reads exactly 1 in every band; the plain C_band (1 / r^2 - 1) is ~10% off at
L < 300, where C_L and the box's anisotropic N_QE both vary across a band.

SELF-CONJUGATE COLUMNS (kx = 0 and kx = Nyquist) and the origin are excluded from every sum,
as in sampler_noise_estimate.py; in the merged matrix they, and any mode the realizations did
not measure, get the QE's own value times the band ratio N / N_QE at their |L|.

FILES. A job writes two files, both valid while it is still running:
    sampler_phi_NNNN_samples.npy   (n_sweeps, nside, nside // 2 + 1) complex64, preallocated
                                   and filled row by row (a numpy memmap)
    sampler_phi_NNNN.npz           the sidecar: `n_done` (rows that are complete AND flushed),
                                   phi_true, C_phi, the HMC acceptances and the configuration,
                                   rewritten atomically after every flush
The analysis trusts only the first `n_done` rows, so it can be run on a directory of running
jobs to gauge convergence; an unfinished chain is simply a shorter one.

HPC path: sampling_chains/get_sampler_effective_noise.sh -> get_sampler_effective_noise_1_chain.sh
-> get_sampler_effective_noise_1_chain.py -> merge_sampler_effective_noise.py. Locally:

    python -m cmb_lensing.sampler_effective_noise --nside 64 --theta_pix 5 --noise_level 5 \
        --n_maps 4 --n_sweeps 400 --burn_in 100 --out_dir <dir>
"""
import argparse
import glob
import os
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from cmb_lensing.matrix_operators import pinv
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER
from cmb_lensing.delensed_spectrum import _to_fourier
from cmb_lensing.sampler_noise_estimate import (simulate_base, save_atomic, used_modes,
                                                band_index, band_harmonic,
                                                qe_load_sim_matrix)

DEFAULT_N_SWEEPS = 3000
DEFAULT_BURN_IN = 200
DEFAULT_DELTA_ELL = 100.0
#flush the memmap and rewrite the sidecar every this many sweeps: the lag of a live analysis
#behind the job, and the most a job killed at the wall clock can lose
DEFAULT_FLUSH_EVERY = 10
#disjoint post-burn-in blocks whose means are cross multiplied for the unbiased |phi_mean|^2.
#Even, so the first and second half of a chain hold the same number of blocks
DEFAULT_N_BLOCKS = 4
#Sokal's window constant, as in chain_analysis.integrated_autocorrelation_time
IAT_WINDOW = 5.0
#modes whose autocorrelation is transformed at once (bounds the FFT's memory)
IAT_CHUNK = 2048
SAMPLE_DTYPE = np.complex64
JOB_PREFIX = "sampler_phi"
MERGED_NAME = "sampler_effective_noise.npz"
#everything two chains must agree on before their moments may be summed
CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "beam_fwhm")
#the estimates, in report order; the first is what `nphi` holds
ESTIMATES = ("cross", "naive", "b_only", "variance")
HALVES = ("half_1", "half_2")
#the per-realization, per-mode moments the merged npz keeps (un-whitened, rfft-grid units)
MOMENTS = ("auto_cross", "auto_naive", "cross_true", "true_auto", "variance")


# ── One chain ──────────────────────────────────────────────────────────────

def sidecar_name(realization_index):
    return f"{JOB_PREFIX}_{realization_index:04d}.npz"


def samples_name(realization_index):
    return f"{JOB_PREFIX}_{realization_index:04d}_samples.npy"


def run_job(out_dir, realization_index, map_seed, nside, theta_pix, noise_level,
            n_sweeps = DEFAULT_N_SWEEPS, l_knee = 0.0, param_ground = None,
            flush_every = DEFAULT_FLUSH_EVERY, verbose = True):
    """One chain of (f, phi) | d at the ground-truth theta, every phi sample written to disk.

    The realization (f, phi, d) is load_sim's at `map_seed`; the chain's randomness comes
    from chain_key(map_seed, 0), which is disjoint from load_sim's own keys.
    """
    #imported here: these pull in the whole sampler stack, which the analysis never needs
    from cmb_lensing.fisher_forecast_from_mixed_logpdf import (posterior_mixed_draws,
                                                               _mixed_theta_matrices,
                                                               chain_key)

    param_ground = dict(GROUND_TRUTH) if param_ground is None else param_ground
    os.makedirs(out_dir, exist_ok = True)
    sidecar_path = os.path.join(out_dir, sidecar_name(realization_index))
    samples_path = os.path.join(out_dir, samples_name(realization_index))

    data_set = simulate_base(nside, theta_pix, noise_level, map_seed, param_ground, l_knee)
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    #C_f / C_phi / D / G at the ground truth, built by the same constructors the sampler's
    #_recompute_cosmo_matrices uses, with load_sim's own QE norm
    cf_op, cphi_op, d_fid, g_fid = _mixed_theta_matrices(
        param_ground, data_set, nside, pix_width, ell_grid,
        data_set.quadratic_estimate.scalar_matrix)
    g_inverse = pinv(g_fid)

    shape = (n_sweeps, nside, nside // 2 + 1)
    samples = np.lib.format.open_memmap(samples_path, mode = "w+", dtype = SAMPLE_DTYPE,
                                        shape = shape)
    metadata = dict(realization_index = realization_index, map_seed = map_seed,
                    nside = nside, theta_pix = theta_pix, noise_level = noise_level,
                    l_knee = l_knee, beam_fwhm = 0.0, n_sweeps = n_sweeps,
                    phi_true = np.asarray(_to_fourier(data_set.phi).scalar_matrix),
                    cphi = np.asarray(cphi_op.scalar_matrix),
                    params = np.array([param_ground[name] for name in PARAM_ORDER]),
                    param_names = np.array(PARAM_ORDER))
    accepts = []

    def checkpoint(n_done):
        #the rows first, the counter second: a reader that trusts the counter never sees a
        #row that is not on disk
        samples.flush()
        save_atomic(sidecar_path, n_done = n_done, finished = n_done == n_sweeps,
                    phi_accepts = np.array(accepts, dtype = bool), **metadata)

    print(f"chain {realization_index}: seed {map_seed}, nside {nside}, {theta_pix:g}', "
          f"{noise_level:g} uK-arcmin, {n_sweeps} sweeps at the ground-truth theta, "
          f"phi start = 0", flush = True)
    checkpoint(0)
    start = time.time()

    def store(sweep, mixed_field, mixed_phi, accepted):
        phi = np.asarray((g_inverse * mixed_phi).scalar_matrix)
        if phi.shape != shape[1:]:
            raise ValueError(f"the sampler handed back a phi of shape {phi.shape}, not the "
                             f"rfft grid {shape[1:]}")
        samples[sweep - 1] = phi
        accepts.append(bool(accepted))
        if sweep % flush_every == 0 or sweep == n_sweeps:
            checkpoint(sweep)
        if verbose and (sweep == 1 or sweep % 100 == 0 or sweep == n_sweeps):
            print(f"  sweep {sweep}/{n_sweeps}: {(time.time() - start) / sweep:.2f} s per "
                  f"sweep, phi acceptance {np.mean(accepts):.3f}", flush = True)

    posterior_mixed_draws(data_set, cf_op, cphi_op, d_fid, g_fid, n_sweeps,
                          chain_key(map_seed, 0), on_draw = store, verbose = False)
    del samples
    print(f"wrote {samples_path}\nwrote {sidecar_path}")
    return sidecar_path


# ── Reading the chains ─────────────────────────────────────────────────────

def _config(data):
    return {key: (int(data[key]) if key == "nside" else float(data[key]))
            for key in CONFIG_KEYS}


def load_chain_directory(directory, burn_in, n_blocks = DEFAULT_N_BLOCKS, verbose = True):
    """The sidecars of every usable chain in `directory`, each with its `samples_path`.

    Running jobs are used as the prefixes they are (`n_done` sweeps). A chain with fewer
    than 10 sweeps per block after the burn-in is set aside. Refuses chains that disagree on
    the box, the noise or the cosmology, and duplicate seeds.
    """
    paths = sorted(glob.glob(os.path.join(directory, f"{JOB_PREFIX}_[0-9]*.npz")))
    if not paths:
        raise FileNotFoundError(f"no {JOB_PREFIX}_*.npz chain files in {directory}")
    jobs, short, unfinished = [], [], 0
    for path in paths:
        job = dict(np.load(path, allow_pickle = True))
        job["samples_path"] = path[:-len(".npz")] + "_samples.npy"
        job["n_done"] = int(job["n_done"])
        if job["n_done"] - burn_in < 10 * n_blocks or not os.path.exists(job["samples_path"]):
            short.append(os.path.basename(path))
            continue
        unfinished += not bool(job["finished"])
        jobs.append(job)
    if not jobs:
        raise ValueError(f"no chain in {directory} has {burn_in} burn-in + {10 * n_blocks} "
                         f"sweeps yet")
    config = _config(jobs[0])
    for job in jobs[1:]:
        if _config(job) != config or not np.allclose(job["params"], jobs[0]["params"]):
            raise ValueError(f"the chains in {directory} were run at different boxes, noise "
                             f"levels or cosmologies; refusing to combine them")
    seeds = [int(job["map_seed"]) for job in jobs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate map seeds in {directory}: the same realization would be "
                         f"counted twice")
    if verbose:
        print(f"{len(jobs)} chains in {directory} ({unfinished} still running or killed "
              f"early, used as far as they got)"
              + (f"; set aside as too short for burn-in {burn_in}: {short}" if short else ""))
    return jobs


def mode_iat(samples, window = IAT_WINDOW, chunk = IAT_CHUNK):
    """Integrated autocorrelation time of every column of the complex (n, n_modes) `samples`.

    The autocorrelation is Re <x_t conj(x_{t + lag})> / <|x|^2>, which pools a mode's real
    and imaginary parts (statistically identical), summed with Sokal's automatic window
    exactly as chain_analysis.integrated_autocorrelation_time does. NaN for a mode that
    never moves.
    """
    n, n_modes = samples.shape
    max_lag = n // 2
    size = 1 << int(np.ceil(np.log2(2 * n)))
    lags = np.arange(1, max_lag)[:, None]
    iat = np.full(n_modes, np.nan)
    for begin in range(0, n_modes, chunk):
        x = np.asarray(samples[:, begin:begin + chunk], dtype = np.complex128)
        x = x - x.mean(axis = 0)
        transform = np.fft.fft(x, n = size, axis = 0)
        covariance = np.fft.ifft(np.abs(transform)**2, axis = 0)[:max_lag].real
        with np.errstate(divide = "ignore", invalid = "ignore"):
            acf = covariance / covariance[0]
        taus = 1.0 + 2.0 * np.cumsum(acf[1:], axis = 0)
        stop = lags >= window * taus
        first = np.where(stop.any(axis = 0), stop.argmax(axis = 0), max_lag - 2)
        iat[begin:begin + chunk] = np.take_along_axis(taus, first[None, :], axis = 0)[0]
    return iat


def cross_power(block_means):
    """Mean over the distinct PAIRS of blocks of Re(m_i conj(m_j)): |mean|^2 without the
    blocks' own Monte Carlo noise."""
    count = len(block_means)
    total = np.abs(np.sum(block_means, axis = 0))**2 - np.sum(np.abs(block_means)**2, axis = 0)
    return total / (count * (count - 1))


def chain_moments(job, burn_in, n_blocks = DEFAULT_N_BLOCKS, with_iat = True):
    """Every per-mode quantity the merge needs from ONE chain, after cutting `burn_in`.

    Returns the five MOMENTS, the first / second half's (auto_cross, cross_true), and (with
    `with_iat`) the per-mode IAT, all on the rfft grid, plus the sweep count and the
    post-burn-in phi acceptance.
    """
    n_done = job["n_done"]
    stored = np.load(job["samples_path"], mmap_mode = "r")
    samples = np.asarray(stored[burn_in:n_done])
    n = len(samples)
    truth = np.asarray(job["phi_true"])

    block_means, abs2_sum = [], 0.0
    for block in np.array_split(np.arange(n), n_blocks):
        chunk = samples[block[0]:block[-1] + 1]
        block_means.append(chunk.mean(axis = 0, dtype = np.complex128))
        abs2_sum = abs2_sum + np.sum(chunk.real.astype(np.float64)**2
                                     + chunk.imag.astype(np.float64)**2, axis = 0)
    block_means = np.array(block_means)
    mean = samples.mean(axis = 0, dtype = np.complex128)
    half = n_blocks // 2

    def cross_true(estimate):
        return np.real(estimate * np.conj(truth))

    #a correlated chain's sample variance is low by ~IAT / n, which matters where the
    #"variance" estimate's r^2 = 1 - V / C is itself of that size (N >> C): divide by
    #n - IAT instead of n - 1 when the IAT is measured
    iat = mode_iat(samples.reshape(n, -1)).reshape(truth.shape) if with_iat else None
    lost = 1.0 if iat is None else np.where(np.isfinite(iat), np.clip(iat, 1.0, n / 2), 1.0)

    result = dict(auto_cross = cross_power(block_means), auto_naive = np.abs(mean)**2,
                  cross_true = cross_true(mean), true_auto = np.abs(truth)**2,
                  variance = (abs2_sum - n * np.abs(mean)**2) / (n - lost),
                  half_1_auto = cross_power(block_means[:half]),
                  half_1_cross = cross_true(block_means[:half].mean(axis = 0)),
                  half_2_auto = cross_power(block_means[half:]),
                  half_2_cross = cross_true(block_means[half:].mean(axis = 0)),
                  n_used = n,
                  phi_acceptance = float(np.mean(np.asarray(job["phi_accepts"])[burn_in:n_done])))
    if with_iat:
        result["iat"] = iat
    return result


# ── The estimates ──────────────────────────────────────────────────────────

def _r_squared(kind, auto, cross, true, variance):
    """r^2 of one estimate from REDUCED whitened moments (see the module docstring); NaN
    where the moments do not define a correlation in (0, 1)."""
    with np.errstate(divide = "ignore", invalid = "ignore"):
        if kind == "b_only":
            r2, positive = cross / true, (cross > 0) & (true > 0)
        elif kind == "variance":
            r2, positive = 1.0 - variance, np.isfinite(variance)
        else:
            r2, positive = cross**2 / (auto * true), (auto > 0) & (cross > 0) & (true > 0)
        return np.where(positive & (r2 > 0) & (r2 < 1), r2, np.nan)


#which whitened rows each estimate reads as its (auto, cross)
_ESTIMATE_ROWS = dict(cross = ("auto_cross", "cross_true"), naive = ("auto_naive", "cross_true"),
                      b_only = ("auto_cross", "cross_true"),
                      variance = ("auto_cross", "cross_true"),
                      half_1 = ("half_1_auto", "half_1_cross"),
                      half_2 = ("half_2_auto", "half_2_cross"))


def band_reference(nphi_qe, cphi, use, index, n_band):
    """The QE's own (1 / r^2 - 1) per band under the band functional noise_estimates uses:
    r^2 = the band mean of C / (C + N_QE), the whitened moments of a Wiener posterior."""
    inside = index >= 0
    with np.errstate(divide = "ignore", invalid = "ignore"):
        wiener = np.where(use, cphi / (cphi + nphi_qe), 0.0)
        r2 = (np.bincount(index[inside], weights = wiener[inside], minlength = n_band)
              / np.bincount(index[inside], minlength = n_band))
        return 1.0 / r2 - 1.0


def noise_estimates(rows, cphi, use, index, n_band, band_scale, keep = None):
    """({kind: per-mode N}, {kind: band N}) from the whitened per-realization rows.

    `rows` maps a moment name to its (n_realizations, nside, nside // 2 + 1) array, each
    already divided by nside^2 C_k (so `true_auto` averages to 1); `keep` selects the
    realizations summed (the jackknife passes all but one). Per mode N = C (1 / r^2 - 1)
    from the realization means; per band the means are first summed over the band's modes
    and N = `band_scale` (1 / r^2 - 1), see the module docstring.
    """
    means = {name: (value if keep is None else value[keep]).mean(axis = 0)
             for name, value in rows.items()}
    inside = index >= 0

    def band(matrix):
        return np.bincount(index[inside], weights = matrix[inside], minlength = n_band)

    banded = {name: band(value) for name, value in means.items()}
    #the variance estimate needs a band MEAN of V / C, the others only ratios of sums
    banded["variance"] = banded["variance"] / np.maximum(band(np.ones(index.shape)), 1)

    per_mode, per_band = {}, {}
    for kind in ESTIMATES + HALVES:
        auto, cross = _ESTIMATE_ROWS[kind]
        r2 = _r_squared(kind, means[auto], means[cross], means["true_auto"],
                        means["variance"])
        r2_band = _r_squared(kind, banded[auto], banded[cross], banded["true_auto"],
                             banded["variance"])
        with np.errstate(divide = "ignore", invalid = "ignore"):
            per_mode[kind] = np.where(use, cphi * (1.0 / r2 - 1.0), np.nan)
            per_band[kind] = band_scale * (1.0 / r2_band - 1.0)
    return per_mode, per_band


def _jackknife(leave_one_out):
    """Delete-one error from the stacked leave-one-out values (first axis); NaN entries of a
    delete-one estimate are left out of that entry's spread."""
    values = np.array(leave_one_out)
    count = np.sum(np.isfinite(values), axis = 0)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        centre = np.nansum(values, axis = 0) / count
        spread = np.nansum((values - centre)**2, axis = 0)
        return np.where(count > 1, np.sqrt((count - 1) / count * spread), np.nan)


def _whitened_rows(moments, cphi, nside, use):
    """Stack the per-chain moments into whitened rows: each divided by nside^2 C_k, and 0 off
    the used modes so they never enter a sum."""
    with np.errstate(divide = "ignore", invalid = "ignore"):
        scale = np.where(use, 1.0 / (nside**2 * cphi), 0.0)
    names = [name for name in moments[0] if np.ndim(moments[0][name]) == 2 and name != "iat"]
    return {name: np.array([m[name] for m in moments]) * scale for name in names}


def merge_effective_noise(directory, burn_in = DEFAULT_BURN_IN, delta_ell = DEFAULT_DELTA_ELL,
                          n_blocks = DEFAULT_N_BLOCKS, smooth = False, burn_in_scan = (),
                          verbose = True):
    """Burn in every chain, average it, and form the effective N_phi over realizations.

    Returns a dict ready for np.savez. `nphi` is the full rfft-grid matrix
    (fisher_forecast --nphi_source score --phi_noise <the npz> reads it): the per-mode
    "cross" estimate where the realizations measured the mode, the QE's value times the band
    ratio elsewhere - or, with `smooth`, that QE-shaped fill at EVERY mode (`nphi_smooth`
    always holds it). Also the raw per-mode matrix and jackknife error of every estimate,
    their band spectra and errors, the half-chain bands, the IAT summaries and the
    per-realization moments. `burn_in_scan` lists further burn-ins at which the headline band
    spectrum is recomputed (`band_scan`), to show that it has stopped moving.
    """
    if n_blocks < 4 or n_blocks % 2:
        raise ValueError(f"n_blocks must be even and at least 4 (two blocks per half-chain), "
                         f"got {n_blocks}")
    jobs = load_chain_directory(directory, burn_in, n_blocks, verbose = verbose)
    n_jobs = len(jobs)
    first = jobs[0]
    config = _config(first)
    nside, theta_pix = config["nside"], config["theta_pix"]
    params = np.asarray(first["params"])
    param_ground = {name: float(params[i]) for i, name in enumerate(PARAM_ORDER)}
    ell_grid = np.asarray(gen_ell_grid(nside, theta_pix)[0])
    cphi = np.asarray(first["cphi"], dtype = float)
    use = used_modes(ell_grid) & (cphi > 0)

    edges = np.arange(0.0, float(ell_grid.max()) + delta_ell, delta_ell)
    index = band_index(ell_grid, edges, use)
    n_band = len(edges) - 1
    centres = 0.5 * (edges[1:] + edges[:-1])
    cphi_band, band_count = band_harmonic(np.where(use, cphi, np.nan), index, n_band)
    nphi_qe = qe_load_sim_matrix(param_ground, nside, theta_pix, config["noise_level"],
                                 config["l_knee"])
    band_qe, _ = band_harmonic(np.where(use, nphi_qe, np.nan), index, n_band)

    moments = []
    for j, job in enumerate(jobs):
        moments.append(chain_moments(job, burn_in, n_blocks))
        if verbose:
            print(f"  chain {int(job['realization_index']):4d}: {moments[-1]['n_used']} "
                  f"sweeps after burn-in, phi acceptance "
                  f"{moments[-1]['phi_acceptance']:.3f}", flush = True)
    rows = _whitened_rows(moments, cphi, nside, use)
    #band N = N_QE's band value x (the estimate's 1 / r^2 - 1) / (the QE's own)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        band_scale = band_qe / band_reference(nphi_qe, cphi, use, index, n_band)
    arguments = (cphi, use, index, n_band, band_scale)
    per_mode, per_band = noise_estimates(rows, *arguments)

    #delete-one-realization jackknife of every estimate, per mode and per band
    if n_jobs > 2:
        loo = [noise_estimates(rows, *arguments, keep = np.arange(n_jobs) != j)
               for j in range(n_jobs)]
        mode_error = {kind: _jackknife([m[kind] for m, _ in loo]) for kind in per_mode}
        band_error = {kind: _jackknife([b[kind] for _, b in loo]) for kind in per_band}
    else:
        mode_error = {kind: np.full(cphi.shape, np.nan) for kind in per_mode}
        band_error = {kind: np.full(n_band, np.nan) for kind in per_band}

    #the full matrix: the QE carries the box's anisotropy, the band ratio N / N_QE is smooth
    ratio = per_band["cross"] / band_qe
    good = np.isfinite(ratio)
    if not good.any():
        raise ValueError("no band has a measured effective noise: the chains are too short "
                         "or too few to correlate phi_mean with the truth")
    nphi_smooth = nphi_qe * np.interp(ell_grid, centres[good], ratio[good])
    nphi_smooth[0, 0] = 0.0
    measured = np.isfinite(per_mode["cross"]) & (per_mode["cross"] > 0)
    nphi = nphi_smooth if smooth else np.where(measured, per_mode["cross"], nphi_smooth)
    nphi[0, 0] = 0.0

    #integrated autocorrelation times: per chain and band (median and max over the band's
    #modes), and the per-mode mean over chains
    iats = np.array([m["iat"] for m in moments])
    n_used = np.array([m["n_used"] for m in moments])
    band_iat_median = np.full((n_jobs, n_band), np.nan)
    band_iat_max = np.full((n_jobs, n_band), np.nan)
    for b in range(n_band):
        inside = index == b
        if inside.any():
            band_iat_median[:, b] = np.nanmedian(iats[:, inside], axis = 1)
            band_iat_max[:, b] = np.nanmax(iats[:, inside], axis = 1)

    merged = dict(nphi = nphi, nphi_smooth = nphi_smooth, nphi_qe = nphi_qe, cphi = cphi,
                  smoothed = bool(smooth), filled_modes = int(np.sum(~measured) - 1),
                  band_edges = edges, band_ells = centres, band_count = band_count,
                  band_cphi = cphi_band, band_nphi_qe = band_qe,
                  band_nphi = per_band["cross"], band_nphi_error = band_error["cross"],
                  iat = iats, iat_mean = np.where(use, np.nanmean(np.where(use, iats, 0.0),
                                                                  axis = 0), np.nan),
                  band_iat_median = band_iat_median, band_iat_max = band_iat_max,
                  n_used = n_used, burn_in = burn_in, n_blocks = n_blocks,
                  phi_acceptance = np.array([m["phi_acceptance"] for m in moments]),
                  finished = np.array([bool(job["finished"]) for job in jobs]),
                  realization_index = np.array([int(job["realization_index"])
                                                for job in jobs]),
                  seeds = np.array([int(job["map_seed"]) for job in jobs]),
                  n_realizations = n_jobs, params = params,
                  param_names = np.array(PARAM_ORDER), delta_ell = delta_ell, **config)
    for kind in ESTIMATES:
        merged[f"nphi_{kind}_raw"] = per_mode[kind]
        merged[f"nphi_{kind}_jackknife_error"] = mode_error[kind]
    for kind in ESTIMATES + HALVES:
        merged[f"band_nphi_{kind}"] = per_band[kind]
        merged[f"band_nphi_{kind}_error"] = band_error[kind]
    for name in MOMENTS:
        merged[f"moment_{name}"] = np.array([m[name] for m in moments])

    #the headline band spectrum at other burn-ins (no IAT, no jackknife: a stability check)
    scan = [b for b in burn_in_scan if b != burn_in]
    if scan:
        merged["burn_in_scan"] = np.array(scan)
        merged["band_scan"] = np.array([_scan_bands(jobs, b, n_blocks, cphi, nside, use,
                                                    arguments, verbose) for b in scan])
    if verbose:
        _report(merged)
    return merged


def _scan_bands(jobs, burn_in, n_blocks, cphi, nside, use, arguments, verbose):
    """The "cross" band spectrum at another burn-in, from the chains long enough for it."""
    usable = [job for job in jobs if job["n_done"] - burn_in >= 10 * n_blocks]
    if verbose:
        print(f"  burn-in scan: {burn_in} ({len(usable)} chains)", flush = True)
    if not usable:
        return np.full(arguments[3], np.nan)
    moments = [chain_moments(job, burn_in, n_blocks, with_iat = False) for job in usable]
    return noise_estimates(_whitened_rows(moments, cphi, nside, use), *arguments)[1]["cross"]


def _report(merged):
    n_used = merged["n_used"]
    print(f"\nsampler effective phi noise: nside {merged['nside']}, {merged['theta_pix']:g}', "
          f"{merged['noise_level']:g} uK-arcmin; {merged['n_realizations']} realizations "
          f"({int(np.sum(~merged['finished']))} unfinished), burn-in {merged['burn_in']}, "
          f"{n_used.min()}-{n_used.max()} sweeps used per chain, phi acceptance "
          f"{merged['phi_acceptance'].mean():.3f}; {merged['filled_modes']} self-conjugate / "
          f"unmeasured modes filled from the QE shape"
          + (" (smoothed: EVERY mode is the QE shape x the band ratio)"
             if merged["smoothed"] else ""))
    qe = merged["band_nphi_qe"]
    iat = np.nanmedian(merged["band_iat_median"], axis = 0)
    iat_max = np.nanmax(merged["band_iat_max"], axis = 0)
    print("  every N below is divided by N_QE (load_sim, no NPHI_FAC); +/- is the "
          "delete-one-realization jackknife")
    print(f"  {'L':>6s} {'modes':>5s}  {'cross':>15s} {'naive':>7s} {'b_only':>15s} "
          f"{'variance':>8s}  {'half 1':>7s} {'half 2':>7s}  {'IAT med':>7s} {'IAT max':>7s} "
          f"{'n_eff min':>9s}")
    for b, ell in enumerate(merged["band_ells"]):
        if not merged["band_count"][b]:
            continue

        def value(kind):
            return merged[f"band_nphi_{kind}"][b] / qe[b]

        def error(kind):
            return merged[f"band_nphi_{kind}_error"][b] / qe[b]

        print(f"  {ell:6.0f} {merged['band_count'][b]:5d}  {value('cross'):6.3f} +/- "
              f"{error('cross'):5.3f} {value('naive'):7.3f} {value('b_only'):6.3f} +/- "
              f"{error('b_only'):5.3f} {value('variance'):8.3f}  {value('half_1'):7.3f} "
              f"{value('half_2'):7.3f}  {iat[b]:7.1f} {iat_max[b]:7.1f} "
              f"{n_used.min() / iat_max[b]:9.1f}")
    if "band_scan" in merged:
        print(f"\n  burn-in scan of the cross estimate / N_QE (analysis burn-in "
              f"{merged['burn_in']}):")
        print(f"  {'L':>6s}  " + " ".join(f"{b:>7d}" for b in merged["burn_in_scan"]))
        for b, ell in enumerate(merged["band_ells"]):
            if merged["band_count"][b]:
                print(f"  {ell:6.0f}  " + " ".join(f"{row[b] / qe[b]:7.3f}"
                                                   for row in merged["band_scan"]))


# ── Plots ──────────────────────────────────────────────────────────────────

def _title(merged):
    return (f"sampler effective phi noise | nside {int(merged['nside'])}, "
            f"{float(merged['theta_pix']):g}', {float(merged['noise_level']):g} uK-arcmin | "
            f"{int(merged['n_realizations'])} realizations, burn-in {int(merged['burn_in'])}")


def _band_of_matrix(merged, matrix):
    ell_grid = np.asarray(gen_ell_grid(int(merged["nside"]), float(merged["theta_pix"]))[0])
    use = used_modes(ell_grid)
    edges = np.asarray(merged["band_edges"])
    return band_harmonic(np.where(use, matrix, np.nan), band_index(ell_grid, edges, use),
                         len(edges) - 1)[0]


def plot_spectra(merged, path, compare = ()):
    """Band spectra of the four estimates against C_phi and the QE, and their ratios to the
    QE. `compare` is a sequence of (label, rfft-grid N_phi matrix) drawn alongside."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ells, qe = np.asarray(merged["band_ells"]), np.asarray(merged["band_nphi_qe"])
    keep = np.isfinite(qe) & (np.asarray(merged["band_count"]) > 0)
    styles = dict(cross = ("C0", "o", "cross (block-mean cross power; headline)"),
                  naive = ("C1", "^", r"naive $|\bar\phi|^2$ (biased high)"),
                  b_only = ("C2", "s", r"b_only $\langle\bar\phi\,\phi^*\rangle"
                                       r"/\langle|\phi|^2\rangle$"),
                  variance = ("C4", "v", "posterior variance"))

    figure, (top, bottom) = plt.subplots(2, 1, figsize = (9, 9), sharex = True,
                                         gridspec_kw = dict(height_ratios = [2, 1]))
    top.loglog(ells[keep], np.asarray(merged["band_cphi"])[keep], color = "0.4",
               label = r"$C_L^{\phi\phi}$")
    top.loglog(ells[keep], qe[keep], color = "C3", linestyle = "--",
               label = "QE (load_sim, no NPHI_FAC)")
    bottom.axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    for kind in ESTIMATES:
        color, marker, label = styles[kind]
        band = np.asarray(merged[f"band_nphi_{kind}"])
        error = np.asarray(merged[f"band_nphi_{kind}_error"])
        good = keep & np.isfinite(band)
        for axis, scale in ((top, 1.0), (bottom, qe[good])):
            axis.errorbar(ells[good], band[good] / scale, yerr = error[good] / scale,
                          color = color, marker = marker, markersize = 3, capsize = 2,
                          linewidth = 0.8, label = label)
    for i, (label, matrix) in enumerate(compare):
        band = _band_of_matrix(merged, np.asarray(matrix))
        good = keep & np.isfinite(band)
        for axis, scale in ((top, 1.0), (bottom, qe[good])):
            axis.plot(ells[good], band[good] / scale, color = f"C{5 + i}", linestyle = ":",
                      marker = ".", label = label)
    top.set_ylabel(r"$N_L$ (covar units)")
    top.legend(fontsize = 8)
    bottom.set_xscale("log")
    bottom.set_xlabel(r"$L$")
    bottom.set_ylabel("ratio to QE")
    #one unmeasured high-L band's error bar would otherwise set the whole scale
    bottom.set_ylim(0.0, 2.5)
    bottom.legend(fontsize = 7, ncol = 2)
    figure.suptitle(_title(merged), fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.95))
    figure.savefig(path, dpi = 140)
    plt.close(figure)


def _iat_image(axis, figure, matrix, title):
    #fftshift the full axis so |L| grows away from the middle row
    image = axis.imshow(np.fft.fftshift(matrix, axes = 0), aspect = "auto", origin = "lower",
                        cmap = "viridis")
    axis.set_title(title, fontsize = 9)
    axis.set_xlabel("rfft column")
    axis.set_ylabel("row (fftshifted)")
    figure.colorbar(image, ax = axis, label = "IAT (sweeps)")


def plot_iat_summary(merged, path):
    """The per-mode IAT averaged over chains, and every chain's band-median IAT against L."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ells = np.asarray(merged["band_ells"])
    keep = np.asarray(merged["band_count"]) > 0
    figure, (left, right) = plt.subplots(1, 2, figsize = (14, 5.5))
    _iat_image(left, figure, np.asarray(merged["iat_mean"]),
               "IAT per phi mode, mean over chains")
    for row in np.asarray(merged["band_iat_median"]):
        right.plot(ells[keep], row[keep], color = "C0", linewidth = 0.6, alpha = 0.5)
    right.plot(ells[keep], np.nanmedian(merged["band_iat_median"], axis = 0)[keep],
               color = "k", linewidth = 1.5, label = "band median, median over chains")
    right.plot(ells[keep], np.nanmax(merged["band_iat_max"], axis = 0)[keep], color = "C3",
               linestyle = "--", label = "band max, max over chains")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel(r"$L$")
    right.set_ylabel("IAT (sweeps)")
    right.set_title("one thin line per chain: band-median IAT", fontsize = 9)
    right.legend(fontsize = 8)
    figure.suptitle(_title(merged), fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.95))
    figure.savefig(path, dpi = 140)
    plt.close(figure)


def plot_iat_per_chain(merged, directory):
    """One IAT-per-mode image per chain, into `directory`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok = True)
    for j, index in enumerate(np.asarray(merged["realization_index"])):
        figure, axis = plt.subplots(figsize = (7, 5.5))
        iat = np.asarray(merged["iat"][j])
        _iat_image(axis, figure, iat,
                   f"chain {int(index)} (seed {int(merged['seeds'][j])}): IAT per phi mode | "
                   f"{int(merged['n_used'][j])} sweeps after burn-in "
                   f"{int(merged['burn_in'])}, max {np.nanmax(iat):.0f}")
        figure.tight_layout()
        figure.savefig(os.path.join(directory, f"iat_chain_{int(index):04d}.png"), dpi = 110)
        plt.close(figure)


def write_outputs(merged, directory, compare = ()):
    """Save the merged npz and every plot into `directory`; returns the npz path."""
    out_path = os.path.join(directory, MERGED_NAME)
    np.savez(out_path, **merged)
    stem = os.path.splitext(out_path)[0]
    plot_spectra(merged, stem + "_spectra.png", compare = compare)
    plot_iat_summary(merged, stem + "_iat.png")
    plot_iat_per_chain(merged, os.path.join(directory, "iat_per_chain"))
    print(f"\nwrote {out_path}\nwrote {stem}_spectra.png\nwrote {stem}_iat.png\n"
          f"wrote {os.path.join(directory, 'iat_per_chain')}/iat_chain_*.png")
    return out_path


# ── Local runner ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nside", type = int, default = 64)
    parser.add_argument("--theta_pix", type = float, default = 5.0)
    parser.add_argument("--noise_level", type = float, default = 5.0)
    parser.add_argument("--l_knee", type = float, default = 0.0)
    parser.add_argument("--n_maps", type = int, default = 4)
    parser.add_argument("--n_sweeps", type = int, default = 400)
    parser.add_argument("--burn_in", type = int, default = 100)
    parser.add_argument("--seed_prefix", type = int, default = 1235)
    parser.add_argument("--out_dir", type = str, required = True)
    args = parser.parse_args()

    for m in range(args.n_maps):
        run_job(args.out_dir, m, args.seed_prefix + m, args.nside, args.theta_pix,
                args.noise_level, n_sweeps = args.n_sweeps, l_knee = args.l_knee)
    write_outputs(merge_effective_noise(args.out_dir, burn_in = args.burn_in), args.out_dir)


if __name__ == "__main__":
    main()
