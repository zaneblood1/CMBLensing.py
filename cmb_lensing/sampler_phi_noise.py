"""The sampler's EFFECTIVE phi reconstruction noise, measured from its own posterior draws.

    phi_mean(d) = <phi>_{phi | d, theta_0}          (all chains' samples after the burn-in)
    r_k^2       = <Re(phi_mean phi_true*)> / <|phi_true|^2>
    N_k         = C_k (1 / r_k^2 - 1)

Several chains per (f, phi, d) realization draw (f, phi) | d with theta HELD at the ground
truth; their phi samples after the burn-in are pooled into the posterior mean, which is
compared with the realization's true phi. The averages are taken over REALIZATIONS per mode
before the ratio (map_joint_phi_noise.noise_with_jackknife, with its delete-one-realization
jackknife). r_k^2 here is the posterior mean's RESPONSE to the truth
(map_joint_phi_noise.noise_from_response): for the exact posterior mean
<Re(phi_mean phi_true*)> = <|phi_mean|^2>, so it equals the squared correlation
<Re(phi_mean phi_true*)>^2 / (<|phi_mean|^2> <|phi_true|^2>), but unlike that it never uses
|phi_mean|^2, which a finite chain's Monte Carlo error inflates.

THE CHAIN is fixed_theta_chain.posterior_mixed_draws: sample_lcdm.sample_joint's
own Gibbs sweep (gibbs_sample_f, mix, one phi HMC step, unmix) minus its theta step, with C_f /
C_phi / D the ground-truth data set's own and G built the way the sampler builds it (f needs no start: it
is redrawn from its exact conditional every sweep). Chain 0 of a realization starts at phi = 0,
chain k > 0 at its own draw from the C_phi prior. A chain's random numbers come from
fixed_theta_chain.chain_key(map_seed, chain_seed): `chain_seed` is given by the caller or, by
default, drawn at random (as sample_lcdm.sample_joint does) and stored in the sidecar.
Every phi sample is stored, un-thinned and with no burn-in cut; the burn-in is chosen at merge
time, and the per-mode Gelman-Rubin R-hat across a realization's chains
(measure_phi_noise/get_sampler_phi_noise/plot_sampler_convergence.py) checks it.

SELF-CONJUGATE COLUMNS (kx = 0 and kx = Nyquist) are measured like every other mode. The phi
HMC runs on the Fourier phi, whose gradient carries anti-Hermitian content on exactly those two
columns (CLAUDE.md, "The Phi Gradient in Fourier Space"), so a sample can hold a part there that
the real-space phi (irfft2) does not see. phi_mean is therefore projected onto its physical,
Hermitian part rfft2(irfft2(phi_mean)) before the moments are taken; the projection is linear
(the same as projecting every sample) and the identity on every other column.
The origin (C_phi = 0, phi_true = 0) is stored as 0, and a mode whose mean cross power is not
positive as NaN (the forecast gives it no phi information, delensed_covariance W = 0).

FILES. A job (realization NNNN, chain K) writes two files, both valid while it is running:
    sampler_phi_NNNN_cK_samples.npy  (n_sweeps, nside, nside // 2 + 1) complex64, preallocated
                                   and filled row by row (a numpy memmap)
    sampler_phi_NNNN_cK.npz        the sidecar: `n_done` (rows that are complete AND flushed),
                                   phi_true, C_phi, the HMC acceptances and the configuration,
                                   rewritten atomically after every flush
The merge reads only the first `n_done` rows, so it also runs on a directory of running jobs.
It writes sampler_phi_noise.npz, whose `nphi` is read by fisher_forecast (--phi_noise) and by
delensed_covariance (phi_noise).

The data can be temperature ("I"), polarization ("P") or both ("IP") (`pol`); phi, and so
everything stored and merged here, is a scalar field for all three.

HPC path: measure_phi_noise/get_sampler_phi_noise/get_sampler_phi_noise.sh -> get_sampler_phi_noise_1_chain.sh
-> get_sampler_phi_noise_1_chain.py, then merge_sampler_phi_noise.sh -> merge_sampler_phi_noise.py.
"""
import argparse
import glob
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from cmb_lensing.matrix_operators import pinv
from cmb_lensing.constants import Basis, GROUND_TRUTH, PARAM_ORDER, DEFAULT_HMC_STEP_SIZE, DEFAULT_HMC_NUM_STEPS
from cmb_lensing.fields import fourier
from cmb_lensing.simulate import load_sim, field_from_covar_single_key
from cmb_lensing.camb_spectra import camb_cls_at_params, load_sim_cosmology, POLS, DEFAULT_POL
from cmb_lensing.map_joint_phi_noise import cross_power_grid, noise_with_jackknife, stored_pol
from cmb_lensing.mcmc_diagnostics import gelman_rubin_from_moments

DEFAULT_N_SWEEPS = 6000
#burn-in of the local runner only; on the HPC it is set in merge_sampler_phi_noise.sh
DEFAULT_BURN_IN = 100
#flush the memmap and rewrite the sidecar every this many sweeps: the lag of a live merge
#behind the job, and the most a job killed at the wall clock can lose
DEFAULT_FLUSH_EVERY = 10
SAMPLE_DTYPE = np.complex64
JOB_PREFIX = "sampler_phi"
#chains per realization in the local runner (the HPC driver sets its own num_chains)
DEFAULT_N_CHAINS = 2
#fold_in data deriving a chain's prior-draw starting phi from its root key, so the start
#never shares a key with the chain's sweeps
PRIOR_START_STREAM = 2_026_110_901
#a chain seed left unset is drawn uniformly from [0, MAX_CHAIN_SEED), as sample_lcdm does
MAX_CHAIN_SEED = 2**31
MERGED_NAME = "sampler_phi_noise.npz"
#metadata copied from the job files into the merged file
CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "beam_fwhm")

def save_atomic(path, **arrays):
    """np.savez through a hidden temporary + os.replace, so a file is never half written."""
    directory, name = os.path.split(path)
    temporary = os.path.join(directory, f".{name}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)

def simulate_base(nside, theta_pix, noise_level, seed, params, l_knee = 0.0, pol = DEFAULT_POL):
    return load_sim(nside, theta_pix, pol, seed, **load_sim_cosmology(params),
                    uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                    precomputed_cls = camb_cls_at_params(params))

def _to_fourier(field):
    return field if field.basis == Basis.FOURIER else fourier(field)

def sidecar_name(realization_index, chain_index):
    return f"{JOB_PREFIX}_{realization_index:04d}_c{chain_index}.npz"

def samples_name(realization_index, chain_index):
    return f"{JOB_PREFIX}_{realization_index:04d}_c{chain_index}_samples.npy"

def run_job(out_dir, realization_index, map_seed, nside, theta_pix, noise_level,
            n_sweeps = DEFAULT_N_SWEEPS, l_knee = 0.0, param_ground = None,
            hmc_step_size = DEFAULT_HMC_STEP_SIZE, hmc_num_steps = DEFAULT_HMC_NUM_STEPS,
            flush_every = DEFAULT_FLUSH_EVERY, chain_index = 0, chain_seed = None,
            verbose = True, pol = DEFAULT_POL):
    """One chain of (f, phi) | d at the ground-truth theta, every phi sample written to disk.
    `pol` ("I", "P" or "IP") is the polarization of the simulated data.

    Chain 0 of a realization starts at phi = 0, every other chain at its own draw from the
    C_phi prior, so the chains of one realization start far apart and their agreement after
    the burn-in tests convergence. `chain_seed` seeds the chain's random numbers
    (fixed_theta_chain.chain_key); None draws one at random. Either way it is stored in the
    sidecar, so the chain can be rerun exactly. Chains of one realization need different seeds.
    """
    #imported here: these pull in the whole sampler stack, which the merge never needs
    from cmb_lensing.fixed_theta_chain import (posterior_mixed_draws, mixed_theta_matrices,
                                               chain_key)

    param_ground = dict(GROUND_TRUTH) if param_ground is None else param_ground
    os.makedirs(out_dir, exist_ok = True)
    sidecar_path = os.path.join(out_dir, sidecar_name(realization_index, chain_index))
    samples_path = os.path.join(out_dir, samples_name(realization_index, chain_index))

    data_set = simulate_base(nside, theta_pix, noise_level, map_seed, param_ground, l_knee, pol)
    #C_f / C_phi / D of the ground-truth data set and the sampler's G from load_sim's QE norm
    cf_op, cphi_op, d_fid, g_fid = mixed_theta_matrices(data_set)
    g_inverse = pinv(g_fid)

    if chain_seed is None:
        chain_seed = int(np.random.randint(0, MAX_CHAIN_SEED))
    chain_root_key = chain_key(map_seed, chain_seed)
    phi_start = None
    start_name = "phi = 0"
    if chain_index > 0:
        prior_draw = field_from_covar_single_key(
            nside, cphi_op.scalar_matrix,
            jax.random.fold_in(chain_root_key, PRIOR_START_STREAM))
        phi_start = data_set.phi.replace(scalar_matrix = jnp.fft.rfft2(prior_draw))
        start_name = "a C_phi prior draw"

    shape = (n_sweeps, nside, nside // 2 + 1)
    samples = np.lib.format.open_memmap(samples_path, mode = "w+", dtype = SAMPLE_DTYPE,
                                        shape = shape)
    metadata = dict(realization_index = realization_index, map_seed = map_seed,
                    chain_index = chain_index, chain_seed = chain_seed,
                    nside = nside, theta_pix = theta_pix, noise_level = noise_level,
                    l_knee = l_knee, beam_fwhm = 0.0, pol = pol, n_sweeps = n_sweeps,
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
                          chain_root_key, on_draw = store, verbose = False,
                          hmc_step_size = hmc_step_size, hmc_num_steps = hmc_num_steps,
                          phi_start = phi_start)
    del samples
    print(f"wrote {samples_path}\nwrote {sidecar_path}")
    return sidecar_path


# ── Merging the chains ────────────────────────────────────────────────────

def hermitian_part(phi_fourier, nside):
    """The physical (Hermitian) part rfft2(irfft2(.)): drops the anti-Hermitian content the
    Fourier-space HMC can leave on the self-conjugate columns (see the module docstring)."""
    return np.fft.rfft2(np.fft.irfft2(phi_fourier, s = (nside, nside)))

def chain_moments(samples_path, burn_in, n_post_burn_in):
    """(mean, variance of the real parts, variance of the imaginary parts) per mode of one
    chain's `n_post_burn_in` samples after `burn_in`."""
    samples = np.load(samples_path, mmap_mode = "r")
    samples = np.asarray(samples[burn_in:burn_in + n_post_burn_in])
    mean = np.mean(samples, axis = 0, dtype = np.complex128)
    variance_real = np.var(samples.real, axis = 0, ddof = 1, dtype = np.float64)
    variance_imaginary = np.var(samples.imag, axis = 0, ddof = 1, dtype = np.float64)
    return mean, variance_real, variance_imaginary

def merge_phi_noise(directory, burn_in):
    """Burn in every chain, pool each realization's chains into its posterior mean phi_mean,
    and form the per-mode N_phi and the per-mode Gelman-Rubin R-hat across chains.

    A chain is used up to the `n_done` sweeps its sidecar reports (so running jobs merge as
    shorter chains); one with no sweep past `burn_in` yet is left out, and a realization's
    chains are trimmed to the shortest of them. Returns the dict written to
    sampler_phi_noise.npz:
        nphi, nphi_error, nphi_leave_one_out   N_phi from the RESPONSE of phi_mean to the truth
                                               (map_joint_phi_noise.noise_from_response; rfft
                                               grid, covar_matrix_from_cls units, NaN where
                                               unmeasured, 0 at the origin), its delete-one
                                               jackknife error and leave-one-out N_phi
        rhat_real, rhat_imaginary              (n_realizations, nside, nside // 2 + 1) R-hat of
                                               every mode's real / imaginary part across the
                                               realization's chains (NaN with one chain and at
                                               the origin)
    plus `cphi`, the mean moments, `n_chains`, `n_used` (post-burn-in sweeps per chain),
    `burn_in`, the seeds and the box / cosmology metadata.
    """
    paths = sorted(glob.glob(os.path.join(directory, f"{JOB_PREFIX}_[0-9]*.npz")))

    #post-burn-in chains grouped by realization (map seed), in path order
    jobs_by_seed = {}
    for path in paths:
        job = np.load(path)
        n_done = int(job["n_done"])
        if n_done <= burn_in:
            print(f"{os.path.basename(path)}: {n_done} sweeps, not past burn-in {burn_in}; "
                  f"left out")
            continue
        jobs_by_seed.setdefault(int(job["map_seed"]), []).append((path, job))

    autos = []
    crosses = []
    true_autos = []
    rhat_real = []
    rhat_imaginary = []
    n_chains = []
    n_used = []
    first = None
    for _, jobs in jobs_by_seed.items():
        first = jobs[0][1] if first is None else first
        nside = int(jobs[0][1]["nside"])
        n_post_burn_in = min(int(job["n_done"]) for _, job in jobs) - burn_in
        moments = [chain_moments(path[:-len(".npz")] + "_samples.npy", burn_in, n_post_burn_in)
                   for path, _ in jobs]
        means = np.array([mean for mean, _, _ in moments])
        real_variances = np.array([real_variance for _, real_variance, _ in moments])
        imaginary_variances = np.array([imaginary_variance for _, _, imaginary_variance in moments])

        if len(jobs) > 1:
            rhat_real.append(gelman_rubin_from_moments(means.real, real_variances, n_post_burn_in))
            rhat_imaginary.append(gelman_rubin_from_moments(means.imag, imaginary_variances, n_post_burn_in))
        else:
            rhat_real.append(np.full(means.shape[1:], np.nan))
            rhat_imaginary.append(np.full(means.shape[1:], np.nan))

        #equal-length chains, so the pooled mean is the mean of the chain means
        chain_means = np.array([hermitian_part(mean, nside) for mean in means])
        phi_mean = np.mean(chain_means, axis = 0)
        phi_true = np.asarray(jobs[0][1]["phi_true"])
        autos.append(cross_power_grid(phi_mean, phi_mean, nside))
        crosses.append(cross_power_grid(phi_mean, phi_true, nside))
        true_autos.append(cross_power_grid(phi_true, phi_true, nside))
        n_chains.append(len(jobs))
        n_used.append(n_post_burn_in)
    print(f"{len(n_chains)} realizations merged at burn-in {burn_in}: {min(n_chains)}-"
          f"{max(n_chains)} chains each, {min(n_used)}-{max(n_used)} sweeps used per chain")

    rhat_real = np.array(rhat_real)
    rhat_imaginary = np.array(rhat_imaginary)
    #the origin is identically 0
    rhat_real[:, 0, 0] = np.nan
    rhat_imaginary[:, 0, 0] = np.nan

    cphi = np.asarray(first["cphi"])
    nphi, nphi_error, nphi_leave_one_out, mean_auto, mean_cross, mean_true_auto = noise_with_jackknife(
                np.array(autos), np.array(crosses), np.array(true_autos), cphi, estimator = "response")

    return dict(nphi = nphi, nphi_error = nphi_error, nphi_leave_one_out = nphi_leave_one_out,
                rhat_real = rhat_real, rhat_imaginary = rhat_imaginary,
                cphi = cphi,
                mean_auto = mean_auto, mean_cross = mean_cross,
                mean_true_auto = mean_true_auto,
                burn_in = burn_in, n_chains = np.array(n_chains), n_used = np.array(n_used),
                n_realizations = len(n_chains), seeds = np.array(list(jobs_by_seed)),
                params = np.asarray(first["params"]), param_names = np.array(PARAM_ORDER),
                pol = stored_pol(first), **{key: first[key] for key in CONFIG_KEYS})


# ── Local runner ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description = __doc__,
             formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nside", type = int, default = 64)
    parser.add_argument("--theta_pix", type = float, default = 5.0)
    parser.add_argument("--noise_level", type = float, default = 5.0)
    parser.add_argument("--l_knee", type = float, default = 0.0)
    parser.add_argument("--pol", choices = POLS, default = DEFAULT_POL)
    parser.add_argument("--n_maps", type = int, default = 4)
    parser.add_argument("--n_sweeps", type = int, default = 400)
    parser.add_argument("--burn_in", type = int, default = DEFAULT_BURN_IN)
    parser.add_argument("--seed_prefix", type = int, default = 1235)
    parser.add_argument("--out_dir", type = str, required = True)
    parser.add_argument("--hmc_step_size", type = float, default = DEFAULT_HMC_STEP_SIZE)
    parser.add_argument("--hmc_num_steps", type = int, default = DEFAULT_HMC_NUM_STEPS)
    parser.add_argument("--n_chains", type = int, default = DEFAULT_N_CHAINS)
    parser.add_argument("--chain_seeds", type = int, nargs = "*", default = None,
                        help = "one seed per chain index, reused for every map (default: "
                               "random)")
    args = parser.parse_args()

    for m in range(args.n_maps):
        for chain_index in range(args.n_chains):
            chain_seed = None if args.chain_seeds is None else args.chain_seeds[chain_index]
            run_job(args.out_dir, m, args.seed_prefix + m, args.nside, args.theta_pix,
                    args.noise_level, n_sweeps = args.n_sweeps, l_knee = args.l_knee,
                    hmc_step_size = args.hmc_step_size, hmc_num_steps = args.hmc_num_steps,
                    chain_index = chain_index, chain_seed = chain_seed, pol = args.pol)
    out_path = os.path.join(args.out_dir, MERGED_NAME)
    np.savez(out_path, **merge_phi_noise(args.out_dir, args.burn_in))

if __name__ == "__main__":
    main()
