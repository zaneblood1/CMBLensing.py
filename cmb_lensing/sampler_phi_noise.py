"""The sampler's EFFECTIVE phi reconstruction noise, measured from its own posterior draws.

    phi_mean(d) = <phi>_{phi | d, theta_0}          (the chain average after burn-in)
    r_k^2       = <B>^2 / (<A> <D>),   A = |phi_mean|^2, B = Re(phi_mean phi_true*), D = |phi_true|^2
    N_k         = C_k (1 / r_k^2 - 1)

One chain per (f, phi, d) realization draws (f, phi) | d with theta HELD at the ground truth,
its phi samples after the burn-in are averaged into the posterior mean, and that mean is cross
correlated with the realization's true phi. The moments are averaged over REALIZATIONS per mode
before the ratio is taken - one mode of one realization has no correlation coefficient (it is
the cosine of a phase difference). This is the estimator of map_joint_phi_noise, whose
noise_with_jackknife (mean, ratio and delete-one-realization jackknife) is used as is.

THE CHAIN is fixed_theta_chain.posterior_mixed_draws: sample_lcdm.sample_joint's
own Gibbs sweep (gibbs_sample_f, mix, one phi HMC step, unmix) minus its theta step, with C_f /
C_phi / D / G built at the ground truth the way the sampler builds them and the chain started
at phi = 0 (f needs no start: it is redrawn from its exact conditional every sweep). Every
phi sample is stored, un-thinned and with no burn-in cut; the burn-in is chosen at merge time.

SELF-CONJUGATE COLUMNS (kx = 0 and kx = Nyquist) are measured like every other mode. The phi
HMC runs on the Fourier phi, whose gradient carries anti-Hermitian content on exactly those two
columns (CLAUDE.md, "The Phi Gradient in Fourier Space"), so a sample can hold a part there that
the real-space phi (irfft2) does not see. phi_mean is therefore projected onto its physical,
Hermitian part rfft2(irfft2(phi_mean)) before the moments are taken; the projection is linear
(the same as projecting every sample) and the identity on every other column.
The origin (C_phi = 0, phi_true = 0) is stored as 0, and a mode whose mean cross power is not
positive as NaN (the forecast gives it no phi information, delensed_covariance W = 0).

FILES. A job writes two files, both valid while it is still running:
    sampler_phi_NNNN_samples.npy   (n_sweeps, nside, nside // 2 + 1) complex64, preallocated
                                   and filled row by row (a numpy memmap)
    sampler_phi_NNNN.npz           the sidecar: `n_done` (rows that are complete AND flushed),
                                   phi_true, C_phi, the HMC acceptances and the configuration,
                                   rewritten atomically after every flush
The merge reads only the first `n_done` rows, so it also runs on a directory of running jobs.
It writes sampler_phi_noise.npz, whose `nphi` is read by fisher_forecast (--nphi_source
sample_joint --phi_noise) and by delensed_covariance (phi_noise).

HPC path: get_sampler_phi_noise/get_sampler_phi_noise.sh -> get_sampler_phi_noise_1_chain.sh
-> get_sampler_phi_noise_1_chain.py, then merge_sampler_phi_noise.sh -> merge_sampler_phi_noise.py.
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
from cmb_lensing.constants import Basis, GROUND_TRUTH, PARAM_ORDER
from cmb_lensing.fields import fourier
from cmb_lensing.simulate import load_sim
from cmb_lensing.fisher_forecast import camb_cls_at_params, load_sim_cosmology
from cmb_lensing.map_joint_phi_noise import cross_power_grid, noise_with_jackknife

DEFAULT_N_SWEEPS = 6000
#burn-in of the local runner only; on the HPC it is set in merge_sampler_phi_noise.sh
DEFAULT_BURN_IN = 100
#flush the memmap and rewrite the sidecar every this many sweeps: the lag of a live merge
#behind the job, and the most a job killed at the wall clock can lose
DEFAULT_FLUSH_EVERY = 10
SAMPLE_DTYPE = np.complex64
JOB_PREFIX = "sampler_phi"
MERGED_NAME = "sampler_phi_noise.npz"
#metadata copied from the job files into the merged file
CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "beam_fwhm")

def save_atomic(path, **arrays):
    """np.savez through a hidden temporary + os.replace, so a file is never half written."""
    directory, name = os.path.split(path)
    temporary = os.path.join(directory, f".{name}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)

def simulate_base(nside, theta_pix, noise_level, seed, params, l_knee = 0.0):
    return load_sim(nside, theta_pix, "I", seed, **load_sim_cosmology(params),
                    uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                    precomputed_cls = camb_cls_at_params(params))

def _to_fourier(field):
    return field if field.basis == Basis.FOURIER else fourier(field)

def sidecar_name(realization_index):
    return f"{JOB_PREFIX}_{realization_index:04d}.npz"

def samples_name(realization_index):
    return f"{JOB_PREFIX}_{realization_index:04d}_samples.npy"

def run_job(out_dir, realization_index, map_seed, nside, theta_pix, noise_level,
            n_sweeps = DEFAULT_N_SWEEPS, l_knee = 0.0, param_ground = None,
            flush_every = DEFAULT_FLUSH_EVERY, verbose = True):
    """One chain of (f, phi) | d at the ground-truth theta, every phi sample written to disk.
    """
    #imported here: these pull in the whole sampler stack, which the merge never needs
    from cmb_lensing.fixed_theta_chain import (posterior_mixed_draws, mixed_theta_matrices,
                                               chain_key)

    param_ground = dict(GROUND_TRUTH) if param_ground is None else param_ground
    os.makedirs(out_dir, exist_ok = True)
    sidecar_path = os.path.join(out_dir, sidecar_name(realization_index))
    samples_path = os.path.join(out_dir, samples_name(realization_index))

    data_set = simulate_base(nside, theta_pix, noise_level, map_seed, param_ground, l_knee)
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    #C_f / C_phi / D / G at the ground truth, built by the same constructors the sampler's
    #_recompute_cosmo_matrices uses, with load_sim's own QE norm
    cf_op, cphi_op, d_fid, g_fid = mixed_theta_matrices(
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
                          chain_key(map_seed), on_draw = store, verbose = False)
    del samples
    print(f"wrote {samples_path}\nwrote {sidecar_path}")
    return sidecar_path


# ── Merging the chains ────────────────────────────────────────────────────

def merge_phi_noise(directory, burn_in):
    """Burn in every chain, average it into phi_mean, and form the per-mode N_phi.

    A chain is used up to the `n_done` sweeps its sidecar reports (so running jobs merge as
    shorter chains); one with no sweep past `burn_in` yet is left out. Returns the dict written
    to sampler_phi_noise.npz: `nphi` (rfft grid, covar_matrix_from_cls units, NaN where
    unmeasured, 0 at the origin), its jackknife error `nphi_error`, `cphi`, the mean moments
    and the box / cosmology metadata.
    """
    paths = sorted(glob.glob(os.path.join(directory, f"{JOB_PREFIX}_[0-9]*.npz")))

    autos = []
    crosses = []
    true_autos = []
    seeds = []
    n_used = []
    first = None
    for path in paths:
        job = np.load(path)
        n_done = int(job["n_done"])
        if n_done <= burn_in:
            print(f"{os.path.basename(path)}: {n_done} sweeps, not past burn-in {burn_in}; "
                  f"left out")
            continue
        if first is None:
            first = job
        nside = int(job["nside"])
        samples = np.load(path[:-len(".npz")] + "_samples.npy", mmap_mode = "r")
        phi_mean = np.mean(samples[burn_in:n_done], axis = 0, dtype = np.complex128)
        #the physical (Hermitian) part: drops the anti-Hermitian content the Fourier-space
        #HMC can leave on the self-conjugate columns (see the module docstring)
        phi_mean = np.fft.rfft2(np.fft.irfft2(phi_mean, s = (nside, nside)))
        phi_true = np.asarray(job["phi_true"])

        autos.append(cross_power_grid(phi_mean, phi_mean, nside))
        crosses.append(cross_power_grid(phi_mean, phi_true, nside))
        true_autos.append(cross_power_grid(phi_true, phi_true, nside))
        seeds.append(int(job["map_seed"]))
        n_used.append(n_done - burn_in)
    print(f"{len(seeds)} chains merged at burn-in {burn_in}, "
          f"{min(n_used)}-{max(n_used)} sweeps used per chain")

    cphi = np.asarray(first["cphi"])
    nphi, nphi_error, mean_auto, mean_cross, mean_true_auto = noise_with_jackknife(
        np.array(autos), np.array(crosses), np.array(true_autos), cphi)

    return dict(nphi = nphi, nphi_error = nphi_error, cphi = cphi,
                mean_auto = mean_auto, mean_cross = mean_cross,
                mean_true_auto = mean_true_auto,
                burn_in = burn_in, n_used = np.array(n_used),
                n_realizations = len(seeds), seeds = np.array(seeds),
                params = np.asarray(first["params"]), param_names = np.array(PARAM_ORDER),
                **{key: first[key] for key in CONFIG_KEYS})


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
    parser.add_argument("--burn_in", type = int, default = DEFAULT_BURN_IN)
    parser.add_argument("--seed_prefix", type = int, default = 1235)
    parser.add_argument("--out_dir", type = str, required = True)
    args = parser.parse_args()

    for m in range(args.n_maps):
        run_job(args.out_dir, m, args.seed_prefix + m, args.nside, args.theta_pix,
                args.noise_level, n_sweeps = args.n_sweeps, l_knee = args.l_knee)
    out_path = os.path.join(args.out_dir, MERGED_NAME)
    np.savez(out_path, **merge_phi_noise(args.out_dir, args.burn_in))
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
