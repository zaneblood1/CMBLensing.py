"""The lensing reconstruction's EFFECTIVE noise spectrum N_L^eff, measured with map_joint.

WHY THIS EXISTS. Every Fisher forecast in this package writes the lensing block as
C_L^phiphi + N_L, and takes N_L from a quadratic estimator - either the box's own QE matrix
(simulate.scalar_quadratic_estimate) or the analytic Hu & Okamoto N^(0). But this codebase
does not reconstruct phi with a quadratic estimator. map_joint returns the joint MAP over
(f, phi), and sample_lcdm samples it. There is no reason the MAP's reconstruction noise
should equal the QE's, and a forecast built on the wrong N_L misweights the whole phi block
and - through Alens_L = N_L / (C_L + N_L) - the delensed temperature block as well.

WHAT IS MEASURED. For one realization at the fiducial cosmology: load_sim draws the truth,
map_joint reconstructs phi from the data, and the two are cross correlated in |L| annuli.
Writing the exact least-squares split of any reconstruction against the truth,

    phi_hat_L = rho_L phi_L + n_L,        rho_L = <phi_hat phi*> / <|phi|^2>

(n_L uncorrelated with phi BY CONSTRUCTION of rho, so this holds for any estimator, biased
or not), three band sums carry everything:

    auto      A = <|phi_hat|^2>
    cross     B = <Re phi_hat phi*>
    true_auto D = <|phi|^2>

    response   rho   = B / D          how much of the truth the estimate carries
    shrinkage  eps   = A / B          1 if and only if the estimator is optimally scaled
    r^2              = B^2 / (A D)    the cross-correlation coefficient, = rho / eps

and the effective noise of the RESPONSE-DECONVOLVED reconstruction phi_hat / rho is

    N_L^eff = C_L^phiphi (1 / r_L^2 - 1)        <=>        r_L^2 = C_L / (C_L + N_L^eff)

That identity is the point of the whole measurement: the correlation coefficient between
the reconstruction and the truth IS the Wiener weight the phi block's Fisher information is
built from. Deconvolving the response matters - a raw unfiltered estimate has rho -> 1 no
matter how noisy it is, so rho alone would report zero noise. Only when eps = 1 (an optimally
scaled, conditional-mean estimator) does rho determine N_L^eff on its own, as C_L (1 - rho) / rho.

N_L^eff is also rescale invariant: multiplying phi_hat by any constant moves rho and eps but
leaves r^2, and therefore N_L^eff, untouched. A normalization error in the reconstruction
cannot fake it.

WHAT IT IS A PROPERTY OF. N_L^eff describes map_joint AT the number of steps it was run for,
on THIS box, at the fiducial cosmology. It absorbs the MAP's convergence, the RK4 lensing
accuracy and the box's periodicity along with the physics, which is exactly what a forecast
of this pipeline should contract. It is measured once and held fixed across the
finite-difference stencil, the same freezing covariance_blocks already applies to N_phi:
N_L is a property of the estimator and the experiment, not of the model being constrained.

HOW IT IS AVERAGED. Every job stores the three band sums, NOT the ratios. The merge averages
A, B and D over realizations and forms rho, eps, r^2 and N_eff from the averages. A mean of
per-realization ratios carries a bias of order 1 / (band DOF) that does NOT shrink as
realizations are added - at ~20 DOF in the lowest band that is a 5% floor a hundred jobs
would never beat down - while a ratio of means is biased only as 1 / (DOF * realizations).

SELF-VALIDATION. Every realization reports two checks, so a broken run announces itself in
the slurm log rather than quietly biasing N_eff:
  rung 0  <|phi_true|^2> reproduces the input C_phi - pins the measurement's normalization
  rung 1  r^2 against the analytic QE's C / (C + N^(0)) - the comparison the whole exercise
          exists to make. r^2 above it means the MAP beats the quadratic estimator; equal
          means it does not, and no rescaling of N_phi will reconcile a forecast with the
          chains.

The measurement is fanned out one slurm job per realization by
sampling_chains_TEMPLATE/get_effective_phi_noise.sh -> get_phi_noise_1_realization.sh ->
get_phi_noise_1_realization.py, merged by merge_phi_noise.py, and consumed by
fisher_forecast.py's --phi_noise (nphi_source = "measured").
"""

import glob
import os

import numpy as np

import jax.numpy as jnp

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.fields import Basis, fourier
from cmb_lensing.map_joint import map_joint
from cmb_lensing.simulate import (load_sim, covar_matrix_from_cls,
                                  scalar_quadratic_estimate)
from cmb_lensing.precompute_camb_1d import PARAM_ORDER
from cmb_lensing.fisher_forecast import (camb_cls_at_params, cls_with_qe_response,
                                         qe_response_cl, load_sim_cosmology,
                                         _instrument_matrices, DEFAULT_QE_RESPONSE,
                                         QE_RESPONSE_SOURCES)


#default width of the |L| annuli the measurement is reported in. N_eff is smooth in L, so the
#bands exist to beat down Monte Carlo noise rather than to resolve structure
DEFAULT_DELTA_ELL = 100.0

#the filename every per-realization job writes, and the merged product
REALIZATION_GLOB = "phi_noise_*.npz"
MERGED_NAME = "effective_phi_noise.npz"

#everything that must agree between two realizations before they may be averaged. The
#cosmology is in here because N_eff is measured AT a cosmology, and map_joint_steps because
#N_eff describes the reconstruction as it was actually run
_CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "beam_fwhm", "delta_ell",
                "map_joint_steps", "qe_response")


# ── Band averaging on the rfft grid ───────────────────────────────────────

def band_edges(ell_grid, delta_ell = DEFAULT_DELTA_ELL):
    """Uniform |l| band edges spanning exactly the modes the rfft grid carries.

    The first edge is the grid's fundamental (its smallest nonzero |l|) and the last is its
    corner mode sqrt(2) pi / pix_width - outside that range the box holds no modes at all,
    so a band there would be empty.
    """
    ells = np.asarray(ell_grid).ravel()
    positive = ells[ells > 0]
    low = float(np.min(positive))
    high = float(np.max(positive))
    n_band = max(1, int(np.ceil((high - low) / delta_ell)))
    return low + delta_ell * np.arange(n_band + 1)


def band_average(grid_values, ell_grid, weights, edges):
    """DOF-weighted mean of an rfft-grid quantity inside each |l| band.

    Weighted by w_k, the real degrees of freedom per rfft entry (get_fourier_weights) - the
    same weighting _fisher_from_blocks uses, and the correct one here because the two
    self-conjugate columns carry half the freedom of the rest. It matters for the cross term
    in particular: a weighted auto against an unweighted cross would bias the correlation
    coefficient at the Nyquist column.

    Returns (centres, values, dof) over the NON-EMPTY bands only, with `centres` the
    DOF-weighted mean |l| in the band rather than the nominal bin midpoint.
    """
    ells = np.asarray(ell_grid).ravel()
    values = np.asarray(grid_values).ravel()
    dof = np.asarray(jnp.real(weights)).ravel()

    #the [0, 0] origin carries no |l| and is set to zero by every covar_matrix_from_cls call
    usable = (ells > 0) & np.isfinite(values)
    index = np.digitize(ells, edges) - 1
    usable = usable & (index >= 0) & (index < len(edges) - 1)

    n_band = len(edges) - 1
    total = np.bincount(index[usable], weights = (dof * values)[usable], minlength = n_band)
    norm = np.bincount(index[usable], weights = dof[usable], minlength = n_band)
    centre = np.bincount(index[usable], weights = (dof * ells)[usable], minlength = n_band)

    filled = norm > 0
    return centre[filled] / norm[filled], total[filled] / norm[filled], norm[filled]


def cross_power_grid(first_fourier, second_fourier, nside, pix_width):
    """Re(a conj(b)) as an honest C_l on the rfft grid; pass the same field twice for an auto.

    field_from_covar_single_key draws a field as irfft2(rfft2(white) * sqrt(C_grid)) from REAL
    white noise, so E|rfft2(f)|^2 = nside^2 * C_grid, and covar_matrix_from_cls's C_grid is
    itself C_l / pix_width^2. Both factors are undone here, so the result is directly
    comparable to a CAMB spectrum put on the grid.
    """
    product = first_fourier * jnp.conj(second_fourier)
    return np.asarray(jnp.real(product)) * pix_width**2 / nside**2


#fields.map and fields.fourier have no guardrails - they apply irfft2 / rfft2 unconditionally
#rather than looking at the field's basis, so calling fourier on an already-FOURIER field
#raises ("only real valued inputs supported for rfft"). load_sim stores FOURIER and map_joint
#returns whatever it was handed, so the basis is checked rather than assumed
def _scalar_matrix(field):
    """The FOURIER-basis rfft2 matrix of a FlatS0, whatever basis it arrives in."""
    in_fourier = field if field.basis == Basis.FOURIER else fourier(field)
    return in_fourier.scalar_matrix


# ── The analytic reference the measurement is judged against ──────────────

def analytic_qe_noise_grid(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                           l_cutoff, qe_response = DEFAULT_QE_RESPONSE):
    """The box's quadratic-estimator N^(0) on the rfft grid, in Cl units.

    Built from simulate.scalar_quadratic_estimate exactly as the sampler's G matrix is, but
    with NO division by any factor: this is the physical N^(0) = A_L, the thing r^2 would
    reproduce if map_joint were no better than a Wiener-filtered quadratic estimator. It is
    computed here rather than imported from fisher_forecast.qe_noise_matrix so that the
    reference the measurement is judged against cannot pick up a preconditioning or
    experimental factor from elsewhere.
    """
    ells = jnp.arange(2, 2 + cls["total_TT"].shape[0]).astype(jnp.float64)
    noise, mask, beam = _instrument_matrices(nside, pix_width, ell_grid, noise_level,
                                             l_knee, beam_fwhm, l_cutoff)

    def covar(cl):
        return covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl, origin_value = 0)

    matrix = scalar_quadratic_estimate(noise, covar(qe_response_cl(cls, qe_response)),
                                       covar(cls["total_TT"]), mask, beam, pix_width)
    #covar_matrix_from_cls divides every spectrum by pix_width**2; undo it so the reference
    #is on the same footing as the measured band powers
    return np.asarray(matrix) * pix_width**2


# ── One realization ───────────────────────────────────────────────────────

def measure_phi_noise(nside, theta_pix, noise_level, param_ground, map_seed,
                      l_knee = 0.0, beam_fwhm = 0.0, l_cutoff = 10_000,
                      delta_ell = DEFAULT_DELTA_ELL, map_joint_steps = 30,
                      qe_response = DEFAULT_QE_RESPONSE, shared_cls = None,
                      verbose = True):
    """One realization end to end: simulate, reconstruct, cross correlate, band average.

    This is what get_phi_noise_1_realization.py calls - one slurm job, one realization, one
    npz - mirroring mixed_hessian_realization's role for the mixed-Hessian forecast.

    The steps, in order:
      1. load_sim at `map_seed` and the fiducial cosmology gives the true phi and the noisy
         data d.
      2. map_joint(d) gives the MAP reconstruction phi_hat. It is used as it comes: no
         Wiener filter, no rescaling. Any mis-scaling shows up in `shrinkage` and cancels
         out of r^2 anyway.
      3. The three band sums A, B, D are formed in |L| annuli. The RATIOS are deliberately
         not averaged here - the merge forms them from the averaged sums, for the reason the
         module docstring gives.

    Returns a dict of the band sums, the references they are judged against, and the two
    self-validation rungs.
    """
    if qe_response not in QE_RESPONSE_SOURCES:
        raise ValueError(f"qe_response must be one of {QE_RESPONSE_SOURCES}, got "
                         f"{qe_response!r}")

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))
    edges = band_edges(ell_grid, delta_ell)

    if shared_cls is None:
        shared_cls = camb_cls_at_params(param_ground)
    #the QE reference may want the lensed temperature-gradient spectrum for its response;
    #cls_with_qe_response is a no-op on the default "unlensed"
    cls = cls_with_qe_response(param_ground, qe_response)

    #C_phi on the same grid and the same bands as the measurement, so the comparison never
    #comes down to two different interpolations of the multipole axis
    phi_ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)
    cphi_grid = covar_matrix_from_cls(nside, pix_width, ell_grid, phi_ells, cls["phi"],
                                      origin_value = 0) * pix_width**2
    qe_grid = analytic_qe_noise_grid(cls, nside, pix_width, ell_grid, noise_level, l_knee,
                                     beam_fwhm, l_cutoff, qe_response = qe_response)

    band_ells, cphi_band, band_dof = band_average(cphi_grid, ell_grid, weights, edges)
    _, qe_band, _ = band_average(qe_grid, ell_grid, weights, edges)

    if verbose:
        print(f"realization seed {map_seed}: nside {nside}, {theta_pix:g}', "
              f"{noise_level:g} uK-arcmin, l_knee {l_knee:g}")
        print(f"  {len(band_ells)} bands of {delta_ell:g} spanning "
              f"L = {band_ells[0]:.0f}..{band_ells[-1]:.0f}")

    # ── 1. simulate ──────────────────────────────────────────────────────
    data_set = load_sim(nside, theta_pix, "I", map_seed,
                        **load_sim_cosmology(param_ground),
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = shared_cls)

    # ── 2. reconstruct ───────────────────────────────────────────────────
    if verbose:
        print(f"  running map_joint ({map_joint_steps} steps)...")
    _, phi_hat = map_joint(data_set, num_steps = map_joint_steps)

    # ── 3. cross correlate ───────────────────────────────────────────────
    true_fourier = _scalar_matrix(data_set.phi)
    hat_fourier = _scalar_matrix(phi_hat)

    auto_grid = cross_power_grid(hat_fourier, hat_fourier, nside, pix_width)
    cross_grid = cross_power_grid(hat_fourier, true_fourier, nside, pix_width)
    true_auto_grid = cross_power_grid(true_fourier, true_fourier, nside, pix_width)

    _, auto, _ = band_average(auto_grid, ell_grid, weights, edges)
    _, cross, _ = band_average(cross_grid, ell_grid, weights, edges)
    _, true_auto, _ = band_average(true_auto_grid, ell_grid, weights, edges)

    # ── 4. the two validation rungs ──────────────────────────────────────
    #rung 0: the measured truth against the C_phi that generated it. Its per-realization
    #scatter is cosmic variance, not error - only its mean over realizations must be one
    rung_0 = true_auto / cphi_band
    #rung 1: this realization's correlation coefficient against the analytic QE's Wiener
    #weight. Above 1 means the MAP beats the quadratic estimator at that L
    this_r_squared = cross**2 / (auto * true_auto)
    rung_1 = this_r_squared / (cphi_band / (cphi_band + qe_band))

    if verbose:
        weight = band_dof * cphi_band / (cphi_band + qe_band)
        print(f"  rung 0 (measured |phi|^2 / C_phi): DOF-weighted mean "
              f"{np.sum(band_dof * rung_0) / np.sum(band_dof):.4f} "
              f"[per-realization scatter is cosmic variance, not error]")
        print(f"  rung 1 (r^2 / QE Wiener weight):   weighted mean "
              f"{np.sum(weight * rung_1) / np.sum(weight):.4f} "
              f"[> 1 means the MAP beats the quadratic estimator]")

    return dict(band_ells = band_ells, band_dof = band_dof,
                auto = auto, cross = cross, true_auto = true_auto,
                cphi_band = cphi_band, qe_noise_band = qe_band,
                rung_0 = rung_0, rung_1 = rung_1)


# ── Collecting the per-realization files ──────────────────────────────────

def phi_noise_directory_config(directory):
    """The configuration every phi_noise_*.npz in `directory` was produced with.

    Mirrors fisher_forecast_from_mixed_logpdf.hessian_directory_config: read only the
    metadata, refuse to average files that disagree on any of it. Returns the shared config
    plus {"n_realizations", "paths", "params"}.
    """
    paths = sorted(glob.glob(os.path.join(directory, REALIZATION_GLOB)))
    if not paths:
        raise FileNotFoundError(
            f"no {REALIZATION_GLOB} in {directory}. Run "
            f"sampling_chains/get_effective_phi_noise.sh first, or point the merge at the "
            f"out_dir that script writes to.")

    config = None
    params = None
    for path in paths:
        data = np.load(path, allow_pickle = True)
        current = {}
        for key in _CONFIG_KEYS:
            value = data[key]
            if value.dtype.kind in "US":
                current[key] = str(value)
            elif key in ("nside", "map_joint_steps"):
                current[key] = int(value)
            else:
                current[key] = float(value)

        current_params = {name: float(data["params"][i])
                          for i, name in enumerate(PARAM_ORDER)}

        if config is None:
            config = current
            params = current_params
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; an effective noise spectrum cannot mix "
                             f"configurations")
        elif current_params != params:
            raise ValueError(
                f"{path} was produced at cosmology {current_params} but earlier files used "
                f"{params}. N_eff is measured AT a cosmology and is held fixed across the "
                f"forecast's stencil, so averaging two of them would blur the very point "
                f"that is being frozen. Use a separate out_dir per cosmology.")

    return dict(config, params = params, n_realizations = len(paths), paths = paths)


def load_phi_noise_directory(directory, verbose = True):
    """Stack every per-realization file in `directory`, refusing duplicate seeds.

    Returns (stacked, metadata) where `stacked` holds one row per realization for each of
    the three band sums and the two validation rungs.
    """
    metadata = phi_noise_directory_config(directory)

    stacked = {}
    seeds = []
    band_ells = None
    row_names = ("auto", "cross", "true_auto", "rung_0", "rung_1")

    for path in metadata["paths"]:
        data = np.load(path, allow_pickle = True)
        if band_ells is None:
            band_ells = data["band_ells"]
        elif not np.allclose(band_ells, data["band_ells"]):
            raise ValueError(f"{path} uses different bands than earlier files; the band "
                             f"edges follow from nside, theta_pix and delta_ell, so this "
                             f"should be impossible unless the files were mixed by hand")
        for name in row_names:
            stacked.setdefault(name, []).append(data[name])
        seeds.append(int(data["map_seed"]))

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                         f"realizations would be double counted. Each slurm job must use a "
                         f"distinct seed - check the seed_prefix loop in "
                         f"get_effective_phi_noise.sh.")

    for name in row_names:
        stacked[name] = np.array(stacked[name])

    #deterministic in the configuration, so every file holds the same values
    reference = np.load(metadata["paths"][0], allow_pickle = True)
    stacked["band_ells"] = np.asarray(band_ells)
    stacked["band_dof"] = np.asarray(reference["band_dof"])
    stacked["cphi_band"] = np.asarray(reference["cphi_band"])
    stacked["qe_noise_band"] = np.asarray(reference["qe_noise_band"])
    metadata["seeds"] = seeds

    if verbose:
        print(f"Loaded {len(seeds)} realizations from {directory}")
        print(f"  nside {metadata['nside']}, {metadata['theta_pix']:g}', "
              f"{metadata['noise_level']:g} uK-arcmin, l_knee {metadata['l_knee']:g}, "
              f"bands of {metadata['delta_ell']:g}")
        print(f"  map_joint {metadata['map_joint_steps']} steps, QE reference response "
              f"{metadata['qe_response']}")

    return stacked, metadata


def derived_spectra(auto, cross, true_auto, cphi_band):
    """(response, shrinkage, r_squared, n_eff) from one set of band sums.

    Called on the averaged sums for the answer and on each delete-one average for the
    jackknife, so the error is propagated through exactly the nonlinearity the answer went
    through rather than through a linearization of it. See the module docstring for what
    each quantity is.
    """
    response = cross / true_auto
    shrinkage = auto / cross
    r_squared = cross**2 / (auto * true_auto)

    #r^2 is non-negative by construction, but it SQUARES the cross power, so a band whose
    #reconstruction has gone pure noise - mean cross at or below zero, i.e. anticorrelated
    #with the truth - comes back with a perfectly healthy looking r^2 and a finite N_eff.
    #Those bands carry no measurement at all, so they are marked here and merge_phi_noise
    #drops them from the product rather than letting them anchor an interpolation
    measured = (cross > 0) & (r_squared > 0)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        n_eff = np.where(measured, cphi_band * (1.0 / r_squared - 1.0), np.nan)

    return response, shrinkage, r_squared, n_eff


def merge_phi_noise(directory, verbose = True):
    """Average the per-realization files into one N_L^eff, with a jackknife error.

    The three band sums are averaged over realizations FIRST and the ratios formed from the
    averages - see the module docstring for why a mean of per-realization ratios carries a
    bias that adding realizations does not remove.

    The error is a delete-one jackknife over realizations rather than a standard error on
    the mean, because the per-realization band powers are not independent across bands: one
    unlucky phi realization moves a whole run of neighbouring bands together, and the
    jackknife sees that where a per-band standard error does not.

    Returns the dict that is written to effective_phi_noise.npz.
    """
    stacked, metadata = load_phi_noise_directory(directory, verbose = verbose)
    n_realization = metadata["n_realizations"]
    cphi_band = stacked["cphi_band"]

    mean_auto = np.mean(stacked["auto"], axis = 0)
    mean_cross = np.mean(stacked["cross"], axis = 0)
    mean_true_auto = np.mean(stacked["true_auto"], axis = 0)

    response, shrinkage, r_squared, n_eff = derived_spectra(mean_auto, mean_cross,
                                                            mean_true_auto, cphi_band)

    #delete-one jackknife: recompute the derived spectra from each leave-one-out average
    leave_one_out = []
    for index in range(n_realization):
        kept = np.arange(n_realization) != index
        leave_one_out.append(derived_spectra(np.mean(stacked["auto"][kept], axis = 0),
                                             np.mean(stacked["cross"][kept], axis = 0),
                                             np.mean(stacked["true_auto"][kept], axis = 0),
                                             cphi_band))

    errors = {}
    for position, name in enumerate(("response", "shrinkage", "r_squared", "n_eff")):
        if n_realization < 2:
            errors[name] = np.full(len(cphi_band), np.nan)
            continue
        samples = np.array([entry[position] for entry in leave_one_out])
        spread = samples - np.mean(samples, axis = 0)
        errors[name] = np.sqrt((n_realization - 1) / n_realization *
                               np.sum(spread**2, axis = 0))

    #a band with no correlation left has no N_eff to report and nothing to contribute; drop
    #it rather than writing a nan the forecast's interpolation would have to handle
    usable = np.isfinite(n_eff) & (n_eff > 0)
    if not np.all(usable):
        dropped = stacked["band_ells"][~usable]
        print(f"  dropping {int(np.sum(~usable))} band(s) with no measurable correlation "
              f"between the reconstruction and the truth: "
              f"L = {', '.join(f'{ell:.0f}' for ell in dropped)}")
    if not np.any(usable):
        raise RuntimeError(
            f"no band in {directory} has a positive mean cross power, so nothing correlates "
            f"the reconstruction with the truth anywhere. Check the map_joint step count and "
            f"the rung-0 normalization before adding realizations.")

    qe_band = stacked["qe_noise_band"]
    mean_rung_0 = np.mean(stacked["rung_0"], axis = 0)
    mean_rung_1 = np.mean(stacked["rung_1"], axis = 0)

    if verbose:
        print(f"\nEffective reconstruction noise [{n_realization} realizations]")
        print(f"  {'L':>7}{'DOF':>7}{'rho':>9}{'eps':>9}{'r^2':>9}"
              f"{'QE weight':>11}{'N_eff/N_0':>11}{'+/-':>9}")
        for index, ell in enumerate(stacked["band_ells"]):
            if not usable[index]:
                continue
            qe_weight = cphi_band[index] / (cphi_band[index] + qe_band[index])
            ratio = n_eff[index] / qe_band[index]
            print(f"  {ell:7.0f}{stacked['band_dof'][index]:7.0f}"
                  f"{response[index]:9.4f}{shrinkage[index]:9.4f}{r_squared[index]:9.4f}"
                  f"{qe_weight:11.4f}{ratio:11.4f}"
                  f"{errors['n_eff'][index] / qe_band[index]:9.4f}")

        #the single number the whole exercise is for, weighted the way the phi block's Fisher
        #information is: DOF times the Wiener weight the analytic QE would give
        weight = stacked["band_dof"][usable] * (cphi_band[usable] /
                                                (cphi_band[usable] + qe_band[usable]))
        ratio = n_eff[usable] / qe_band[usable]
        mean_ratio = np.sum(weight * ratio) / np.sum(weight)
        mean_error = (np.sqrt(np.sum((weight * errors["n_eff"][usable] /
                                      qe_band[usable])**2)) / np.sum(weight))
        print(f"\n  Fisher-weighted mean N_eff / N^(0) : {mean_ratio:.4f} "
              f"+/- {mean_error:.4f}")
        print(f"    1.0 means map_joint reconstructs exactly as well as the quadratic "
              f"estimator;\n    below 1 means it does better, above 1 worse.")
        print(f"  rung 0 (normalization): DOF-weighted mean "
              f"{np.sum(stacked['band_dof'] * mean_rung_0) / np.sum(stacked['band_dof']):.4f}"
              f"   (must be 1)")

    return dict(band_ells = stacked["band_ells"][usable],
                band_dof = stacked["band_dof"][usable],
                n_eff = n_eff[usable], n_eff_error = errors["n_eff"][usable],
                response = response[usable], response_error = errors["response"][usable],
                shrinkage = shrinkage[usable], shrinkage_error = errors["shrinkage"][usable],
                r_squared = r_squared[usable], r_squared_error = errors["r_squared"][usable],
                cphi_band = cphi_band[usable], qe_noise_band = qe_band[usable],
                rung_0 = mean_rung_0[usable], rung_1 = mean_rung_1[usable],
                n_realizations = n_realization,
                params = np.array([metadata["params"][name] for name in PARAM_ORDER]),
                param_names = np.array(PARAM_ORDER),
                **{key: metadata[key] for key in _CONFIG_KEYS})
