"""The delensed TT spectrum measured EMPIRICALLY, with the codebase's own lense_flow.

WHY THIS EXISTS. fisher_forecast.py's `--spectra delensed` mode gets its delensed temperature
spectrum from CAMB (`delensed_cls_at_params` -> `get_partially_lensed_cls`), which lenses with
a correlation-function method on the full sky and a per-multipole `Alens_L` scaling of
C_L^phiphi. That is NOT how this codebase lenses: `primal_lense_flow` integrates the LenseFlow
ODE on a periodic flat-sky box with a finite number of RK4 steps, and the reconstruction that
does the delensing is `map_joint`'s MAP estimate, not a Wiener-filtered quadratic estimator
with residual fraction N_L / (C_L + N_L). Two different lensing calculations and two different
reconstructions, so there is no reason for them to agree at the precision a Fisher forecast
cares about - and nothing in the repository has ever checked.

WHAT IS MEASURED. A transfer function

    R(l) = C_l^delensed[measured on the box] / C_l^delensed[CAMB, at the same frozen Alens_L]

at the FIDUCIAL cosmology only. fisher_forecast then uses R to carry CAMB's theta dependence
across to the box's calculation:

    C_l^delensed(theta) = R(l) * C_l^delensed,CAMB(theta)

so the stencil's derivative is R * dC_CAMB/dtheta. This is the standard simulation-calibrated
transfer-function construction, and it exists because the alternative - finite-differencing an
independent Monte Carlo measurement at every stencil point - divides the Monte Carlo noise by
2h with h = 0.05 sigma, which is hopeless. The price is the assumption that R is flat in theta
over the stencil width. That assumption is TESTABLE and must be tested:
compare_transfer_functions.py measures R at two cosmologies separated by far more than the
finite-difference step and checks their ratio is consistent with one.

THE VARIANCE-REDUCTION THAT MAKES IT AFFORDABLE. A single realization's binned spectrum carries
the full cosmic variance of the box, and R is a ratio of two such spectra, so the naive
estimator needs thousands of realizations. Instead every realization reports its delensed,
lensed and unlensed spectra measured from the SAME underlying (f, phi) draw. Their ratios have
the cosmic variance almost entirely cancelled - the three fields differ only by a remapping -
so the paired estimator

    R(l) = <C^delensed_MC / C^unlensed_MC>  *  C^unlensed_CAMB / C^delensed_CAMB

converges orders of magnitude faster than <C^delensed_MC> / C^delensed_CAMB. Both are computed
and reported; use the paired one, and read the naive one as a cross-check.

SELF-VALIDATION. Every realization reports three things that must hold independently of any
delensing question, so a broken measurement announces itself rather than quietly biasing R:
  rung 0  <|rfft2(f_unlensed)|^2> reproduces the input C_f - pins the normalization
          (C_l = |rfft2(f)|^2 * pix_width^2 / nside^2, verified to 1e-5 on 400 draws)
  rung 1  C^lensed_MC / C^unlensed_MC reproduces CAMB's C^lensed / C^unlensed - validates the
          forward lensing and the whole measurement chain on this box
  rung 2  lense_flow(lense_flow(f, phi, FORWARD), phi, INVERSE) returns f - LenseFlow is
          exactly invertible by integrating its ODE backwards, so this is machine precision
          up to the RK4 truncation, and it is the only direct test of the delensing operator

WHAT DELENSING MEANS HERE. The NOISELESS lensed field is inverse-lensed by the reconstructed
phi. Delensing the DATA instead would remap the noise into something anisotropic and
correlated, which is not what `covariance_blocks` means when it writes the block as
C_TT^delensed + C_n - there the noise is added back isotropically. So the object measured here
is a signal spectrum, matching that convention.

The measurement is fanned out one slurm job per realization by
sampling_chains_TEMPLATE/get_delensed_spectra.sh -> get_single_delensed_spectra.sh ->
get_single_delensed_spectra.py, merged by merge_delensed_spectra.py, and consumed by
fisher_forecast.py's --transfer_function.
"""

import glob
import os

import numpy as np

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft

from cmb_lensing.constants import FORWARD_LENSE, INVERSE_LENSE
from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.fields import Basis, map as to_map, fourier
from cmb_lensing.lense_flow import lense_flow
from cmb_lensing.map_joint import map_joint
from cmb_lensing.simulate import load_sim, covar_matrix_from_cls
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER, CAMB_LMAX
from cmb_lensing.fisher_forecast import (camb_cls_at_params, cls_with_qe_response,
                                         delensed_cls_at_params, delensing_alens,
                                         qe_noise_cl, delensing_efficiency,
                                         NPHI_SOURCES, DEFAULT_QE_RESPONSE,
                                         QE_RESPONSE_SOURCES)


#number of RK4 steps for BOTH the forward and the inverse lensing solve. load_sim lenses the
#data with n = 10, so the forward direction here matches it by construction; the inverse must
#use the same n or the round trip is not the identity even at perfect phi (rung 2 checks it)
LENSE_STEPS = 10

#default band width for the |l| annuli the spectra are reported in. R is smooth, so the bands
#exist to beat down Monte Carlo noise rather than to resolve structure - wider bands are
#cheaper in realizations and lose nothing, until they start averaging across the acoustic
#peaks that the ratio to CAMB does not fully divide out
DEFAULT_DELTA_ELL = 100.0

#the filename every per-realization job writes, and the merged product
REALIZATION_GLOB = "delensed_spectra_*.npz"
MERGED_NAME = "transfer_function.npz"


def band_edges(ell_grid, delta_ell = DEFAULT_DELTA_ELL):
    """Uniform |l| band edges spanning exactly the modes the rfft grid carries.

    The first edge is the grid's fundamental (its smallest nonzero |l|, 68 at nside 64 / 5')
    and the last is its corner mode (sqrt(2) pi / pix_width, 3055 there) - outside that range
    the box holds no modes at all, so a band there would be empty and R undefined.
    """
    ells = np.asarray(ell_grid).ravel()
    positive = ells[ells > 0]
    low, high = float(np.min(positive)), float(np.max(positive))
    n_band = max(1, int(np.ceil((high - low) / delta_ell)))
    return low + delta_ell * np.arange(n_band + 1)


def band_average(grid_values, ell_grid, weights, edges):
    """DOF-weighted mean of an rfft-grid quantity inside each |l| band.

    Weighted by w_k, the real degrees of freedom per rfft entry (get_fourier_weights) - the
    same weighting _fisher_from_blocks and _radial_cl_profile use, and the correct one here
    because the two self-conjugate columns carry half the freedom of the rest, so an
    unweighted mean would over-count them in the variance of the estimate.

    Returns (centres, values, dof) over the NON-EMPTY bands only, with `centres` the
    DOF-weighted mean |l| in the band rather than the nominal bin midpoint.
    """
    ells = np.asarray(ell_grid).ravel()
    values = np.asarray(grid_values).ravel()
    dof = np.asarray(jnp.real(weights)).ravel()

    #the [0, 0] origin carries no |l| and is set to zero by every covar_matrix_from_cls call
    usable = (ells > 0) & np.isfinite(values)
    index = np.digitize(ells, edges) - 1
    usable &= (index >= 0) & (index < len(edges) - 1)

    n_band = len(edges) - 1
    total = np.bincount(index[usable], weights = (dof * values)[usable], minlength = n_band)
    norm = np.bincount(index[usable], weights = dof[usable], minlength = n_band)
    centre = np.bincount(index[usable], weights = (dof * ells)[usable], minlength = n_band)

    filled = norm > 0
    return centre[filled] / norm[filled], total[filled] / norm[filled], norm[filled]


def grid_power_spectrum(field_fourier, nside, pix_width):
    """|rfft2(f)|^2 as an honest C_l on the rfft grid.

    field_from_covar_single_key draws f = irfft2(rfft2(white) * sqrt(C_grid)) from REAL white
    noise, so E|rfft2(f)|^2 = nside^2 * C_grid, and covar_matrix_from_cls's C_grid is itself
    C_l / pix_width^2. Both factors are undone here. MEASURED on 400 draws against a known
    input spectrum: the recovered ratio is 1.0000087, i.e. exact to the sampling error.
    """
    return np.asarray(jnp.abs(field_fourier)**2) * pix_width**2 / nside**2


#fields.map and fields.fourier carry their own "NOTE this has no guardrails" - they apply
#irfft2 / rfft2 unconditionally rather than looking at the field's basis, so calling fourier
#on a field that is already FOURIER raises ("only real valued inputs supported for rfft").
#Every field here arrives from a different place (load_sim stores FOURIER, map_joint returns
#what it was handed, lense_flow returns MAP), so the basis is checked rather than assumed
def _to_fourier(field):
    return field if field.basis == Basis.FOURIER else fourier(field)


def _to_map(field):
    return field if field.basis == Basis.MAP else to_map(field)


def _scalar_matrix(field):
    """The FOURIER-basis rfft2 matrix of a FlatS0, whatever basis it arrives in."""
    return _to_fourier(field).scalar_matrix


def _lense(field, phi, direction):
    """lense_flow in the MAP basis, returned in FOURIER - the pattern simulate._lens_fields
    and sample_lcdm.gibbs_sample_f both use. T-only: a FlatS0 needs no eb2qu round trip."""
    return _to_fourier(lense_flow(_to_map(field), _to_map(phi), n = LENSE_STEPS,
                                  direction = direction, adjoint = False))


def frozen_alens(param_ground, nside, theta_pix, noise_level, l_knee, beam_fwhm, l_cutoff,
                 nphi_source, qe_response):
    """(Alens_L, N_L, mean delensing efficiency) at the fiducial cosmology.

    Exactly what fisher_forecast.frozen_reconstruction computes for `--spectra delensed`, and
    recomputed here rather than passed in so each slurm job is self-contained. It is
    deterministic in its arguments, so every job gets the same Alens_L - and because Alens_L
    is what CAMB's reference spectrum is built from, nphi_source and qe_response are recorded
    in every output file and the merge refuses to average across different ones.
    """
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = cls_with_qe_response(param_ground, qe_response)
    nphi_cl = qe_noise_cl(cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm,
                          l_cutoff, nphi_source, qe_response = qe_response)
    alens = delensing_alens(cls, nphi_cl)
    return alens, nphi_cl, delensing_efficiency(cls["phi"], nphi_cl)


def measure_delensed_spectrum(nside, theta_pix, noise_level, param_ground, map_seed,
                              l_knee = 0.0, beam_fwhm = 0.0, l_cutoff = 10_000,
                              delta_ell = DEFAULT_DELTA_ELL, map_joint_steps = 30,
                              nphi_source = "covariance",
                              qe_response = DEFAULT_QE_RESPONSE,
                              shared_cls = None, verbose = True):
    """One realization end to end: simulate, reconstruct, delens, measure, ratio to CAMB.

    This is what get_single_delensed_spectra.py calls - one slurm job, one realization, one
    npz - mirroring mixed_hessian_realization's role for the mixed-Hessian forecast.

    The steps, in order:
      1. load_sim at `map_seed` and the fiducial cosmology gives the unlensed field f, the
         true phi, the NOISELESS lensed field L(phi) f, and the noisy data d.
      2. map_joint(d) gives the MAP reconstruction phi_hat. It is used as it comes: a MAP
         estimate is already a regularized optimum, NOT a raw quadratic estimate, so it must
         not be Wiener filtered a second time.
      3. The noiseless lensed field is inverse-lensed by phi_hat. LenseFlow integrates its
         ODE backwards for this, so the operation is exact rather than a Taylor remapping.
      4. All four fields are band-averaged into |l| annuli, and compared against CAMB's own
         unlensed / lensed / partially-lensed spectra on the same bands.

    Returns a dict of everything the merge needs, plus the three self-validation rungs.
    """
    if nphi_source not in NPHI_SOURCES:
        raise ValueError(f"nphi_source must be one of {NPHI_SOURCES}, got {nphi_source!r}")
    if qe_response not in QE_RESPONSE_SOURCES:
        raise ValueError(f"qe_response must be one of {QE_RESPONSE_SOURCES}, got "
                         f"{qe_response!r}")

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))
    edges = band_edges(ell_grid, delta_ell)

    #the reference Alens_L, and CAMB's spectra at it. Deterministic, so identical in every job
    alens, nphi_cl, efficiency = frozen_alens(param_ground, nside, theta_pix, noise_level,
                                              l_knee, beam_fwhm, l_cutoff, nphi_source,
                                              qe_response)
    cls_camb = delensed_cls_at_params(param_ground, alens)
    camb_ells = jnp.arange(2, 2 + cls_camb["scalar_TT"].shape[0]).astype(jnp.float64)

    def on_grid(cl):
        return covar_matrix_from_cls(nside, pix_width, ell_grid, camb_ells, cl,
                                     origin_value = 0) * pix_width**2

    #CAMB's spectra put on the SAME grid and the SAME bands as the measurement, so the ratio
    #never compares two different interpolations of the multipole axis
    camb_bands = {}
    for key, source in (("unlensed", "scalar_TT"), ("lensed", "total_TT"),
                        ("delensed", "delensed_TT")):
        centres, values, dof = band_average(on_grid(cls_camb[source]), ell_grid, weights, edges)
        camb_bands[key] = values
    band_ells, _, band_dof = band_average(on_grid(cls_camb["scalar_TT"]), ell_grid, weights,
                                          edges)

    if verbose:
        print(f"realization seed {map_seed}: nside {nside}, {theta_pix:g}', "
              f"{noise_level:g} uK-arcmin, l_knee {l_knee:g}")
        print(f"  {len(band_ells)} bands of {delta_ell:g} spanning "
              f"l = {band_ells[0]:.0f}..{band_ells[-1]:.0f}")
        print(f"  frozen reconstruction [{nphi_source} N_L, {qe_response} response]: "
              f"C_phi-weighted mean delensing efficiency {efficiency:.4f}")

    #── 1. simulate ──────────────────────────────────────────────────────
    if shared_cls is None:
        shared_cls = camb_cls_at_params(param_ground)
    camb_kwargs = dict(param_ground)
    camb_kwargs["cosmomc_theta"] = camb_kwargs.pop("theta_MC_100") / 100
    camb_kwargs["As"] = float(np.exp(camb_kwargs.pop("logA")) * 1e-10)
    data_set = load_sim(nside, theta_pix, "I", map_seed, **camb_kwargs,
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = shared_cls)

    unlensed = data_set.unlensed_field
    lensed = data_set.lensed_field
    phi_true = data_set.phi

    #── 2. reconstruct ───────────────────────────────────────────────────
    if verbose:
        print(f"  running map_joint ({map_joint_steps} steps)...")
    _, phi_hat = map_joint(data_set, num_steps = map_joint_steps)

    #── 3. delens ────────────────────────────────────────────────────────
    delensed = _lense(lensed, phi_hat, INVERSE_LENSE)

    #rung 2: the inverse must undo the forward at the TRUE phi. This is the only direct test
    #of the delensing operator, and it is the cheapest possible one - no reconstruction, no
    #spectra, just a round trip that has to come back to where it started
    round_trip = _lense(_lense(unlensed, phi_true, FORWARD_LENSE), phi_true, INVERSE_LENSE)
    reference = np.asarray(jnp.abs(_scalar_matrix(unlensed)))
    scale = float(np.max(reference))
    inverse_error = float(np.max(np.abs(np.asarray(_scalar_matrix(round_trip)) -
                                        np.asarray(_scalar_matrix(unlensed))))) / scale

    #── 4. measure ───────────────────────────────────────────────────────
    measured = {}
    for key, field in (("unlensed", unlensed), ("lensed", lensed), ("delensed", delensed)):
        power = grid_power_spectrum(_scalar_matrix(field), nside, pix_width)
        _, measured[key], _ = band_average(power, ell_grid, weights, edges)

    #rung 0: the measured unlensed spectrum against the C_f that generated it
    rung_0 = measured["unlensed"] / camb_bands["unlensed"]
    #rung 1: the measured lensing correction against CAMB's, cosmic variance cancelled by
    #taking both from the same realization
    rung_1 = ((measured["lensed"] / measured["unlensed"]) /
              (camb_bands["lensed"] / camb_bands["unlensed"]))

    #the two R estimators. `paired` divides by the same realization's unlensed spectrum first,
    #so the field's cosmic variance cancels before CAMB is brought in; `naive` does not and is
    #kept only as a cross-check that the pairing is not introducing something of its own
    naive = measured["delensed"] / camb_bands["delensed"]
    paired = ((measured["delensed"] / measured["unlensed"]) *
              (camb_bands["unlensed"] / camb_bands["delensed"]))

    if verbose:
        print(f"  rung 0 (measured unlensed / C_f):        mean {np.mean(rung_0):.4f} "
              f"[per-realization scatter is cosmic variance, not error]")
        print(f"  rung 1 (measured lensing / CAMB lensing): mean {np.mean(rung_1):.4f}")
        print(f"  rung 2 (inverse-lensing round trip):      max relative error "
              f"{inverse_error:.3e}")
        print(f"  R paired: mean {np.mean(paired):.4f}, range "
              f"{np.min(paired):.4f}..{np.max(paired):.4f}")

    return dict(band_ells = band_ells, band_dof = band_dof, edges = edges,
                measured_unlensed = measured["unlensed"],
                measured_lensed = measured["lensed"],
                measured_delensed = measured["delensed"],
                camb_unlensed = camb_bands["unlensed"],
                camb_lensed = camb_bands["lensed"],
                camb_delensed = camb_bands["delensed"],
                transfer_naive = naive, transfer_paired = paired,
                rung_0 = rung_0, rung_1 = rung_1, inverse_error = inverse_error,
                alens = alens, nphi_cl = np.asarray(nphi_cl),
                delensing_efficiency = efficiency)


# ── Collecting the per-realization files ──────────────────────────────────

#everything that must agree between two realizations before they may be averaged. The
#cosmology is in here because R is measured AT a cosmology - averaging two of them would
#silently destroy the very theta dependence compare_transfer_functions.py exists to test
_CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "beam_fwhm", "delta_ell",
                "map_joint_steps", "nphi_source", "qe_response", "lense_steps")


def transfer_directory_config(directory):
    """The configuration every delensed_spectra_*.npz in `directory` was produced with.

    Mirrors fisher_forecast_from_mixed_logpdf.hessian_directory_config: read only the
    metadata, refuse to average files that disagree on any of it. Returns the shared config
    plus {"n_realizations", "paths", "params"}.
    """
    paths = sorted(glob.glob(os.path.join(directory, REALIZATION_GLOB)))
    if not paths:
        raise FileNotFoundError(
            f"no {REALIZATION_GLOB} in {directory}. Run "
            f"sampling_chains/get_delensed_spectra.sh first, or point the merge at the "
            f"out_dir that script writes to.")

    config, params = None, None
    for path in paths:
        data = np.load(path, allow_pickle = True)
        current = {}
        for key in _CONFIG_KEYS:
            value = data[key]
            current[key] = (str(value) if value.dtype.kind in "US"
                            else (int(value) if key in ("nside", "map_joint_steps",
                                                        "lense_steps") else float(value)))
        current_params = {name: float(data["params"][i])
                          for i, name in enumerate(PARAM_ORDER)}
        if config is None:
            config, params = current, current_params
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; a transfer function cannot mix configurations")
        elif current_params != params:
            raise ValueError(
                f"{path} was produced at cosmology {current_params} but earlier files used "
                f"{params}. R is measured AT a cosmology - averaging across two of them "
                f"destroys exactly the theta dependence compare_transfer_functions.py "
                f"tests for. Use a separate out_dir per cosmology.")

    return dict(config, params = params, n_realizations = len(paths), paths = paths)


def load_transfer_directory(directory, verbose = True):
    """Stack every per-realization file in `directory`, refusing duplicate seeds.

    Returns (stacked, metadata) where `stacked` holds one row per realization for each of the
    measured and CAMB band spectra and the two R estimators.
    """
    metadata = transfer_directory_config(directory)

    stacked, seeds, band_ells = {}, [], None
    fields = ("measured_unlensed", "measured_lensed", "measured_delensed",
              "camb_unlensed", "camb_lensed", "camb_delensed",
              "transfer_naive", "transfer_paired", "rung_0", "rung_1")
    for path in metadata["paths"]:
        data = np.load(path, allow_pickle = True)
        if band_ells is None:
            band_ells = data["band_ells"]
        elif not np.allclose(band_ells, data["band_ells"]):
            raise ValueError(f"{path} uses different bands than earlier files; the band "
                             f"edges follow from nside, theta_pix and delta_ell, so this "
                             f"should be impossible unless the files were mixed by hand")
        for name in fields:
            stacked.setdefault(name, []).append(data[name])
        stacked.setdefault("inverse_error", []).append(float(data["inverse_error"]))
        seeds.append(int(data["map_seed"]))

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                         f"realizations would be double counted. Each slurm job must use a "
                         f"distinct seed - check the seed_prefix loop in "
                         f"get_delensed_spectra.sh.")

    stacked = {name: np.array(rows) for name, rows in stacked.items()}
    stacked["band_ells"] = np.asarray(band_ells)
    metadata["seeds"] = seeds

    if verbose:
        print(f"Loaded {len(seeds)} realizations from {directory}")
        print(f"  nside {metadata['nside']}, {metadata['theta_pix']:g}', "
              f"{metadata['noise_level']:g} uK-arcmin, l_knee {metadata['l_knee']:g}, "
              f"bands of {metadata['delta_ell']:g}")
        print(f"  N_phi source {metadata['nphi_source']}, QE response "
              f"{metadata['qe_response']}, map_joint {metadata['map_joint_steps']} steps")
        print(f"  worst inverse-lensing round trip over all realizations: "
              f"{np.max(stacked['inverse_error']):.3e}")

    return stacked, metadata


def merge_transfer_function(directory, estimator = "paired", verbose = True):
    """Average the per-realization files into one R(l), with a jackknife error.

    `estimator` picks "paired" (the cosmic-variance-cancelled ratio, the default and the one
    to use) or "naive". Both are averaged either way and both are stored; the choice only
    sets which one the npz advertises as `transfer`.

    The error is a delete-one jackknife over realizations rather than a standard error on the
    mean, because the per-realization ratios are not independent across bands - one unlucky
    phi realization moves a whole run of neighbouring bands together, and the jackknife sees
    that where a per-band standard error does not.

    Returns the dict that is written to transfer_function.npz.
    """
    if estimator not in ("paired", "naive"):
        raise ValueError(f"estimator must be 'paired' or 'naive', got {estimator!r}")

    stacked, metadata = load_transfer_directory(directory, verbose = verbose)
    n = metadata["n_realizations"]

    means, jackknife = {}, {}
    for name in ("transfer_paired", "transfer_naive", "rung_0", "rung_1"):
        rows = stacked[name]
        means[name] = np.mean(rows, axis = 0)
        if n > 1:
            #delete-one means; the jackknife variance is (n-1)/n * sum (x_i - x_bar)^2
            leave_one_out = (np.sum(rows, axis = 0) - rows) / (n - 1)
            jackknife[name] = np.sqrt((n - 1) / n *
                                      np.sum((leave_one_out -
                                              np.mean(leave_one_out, axis = 0))**2, axis = 0))
        else:
            jackknife[name] = np.full(rows.shape[1], np.nan)

    transfer = means[f"transfer_{estimator}"]
    error = jackknife[f"transfer_{estimator}"]

    if verbose:
        print(f"\nTransfer function R(l) [{estimator} estimator, {n} realizations]")
        print(f"  {'l':>8}{'R':>10}{'jackknife':>12}{'sigma from 1':>14}")
        for i, ell in enumerate(stacked["band_ells"]):
            deviation = ((transfer[i] - 1) / error[i]) if error[i] > 0 else np.nan
            print(f"  {ell:8.0f}{transfer[i]:10.4f}{error[i]:12.4f}{deviation:14.1f}")
        #the paired and naive estimators must agree in the mean; they differ only in variance,
        #so a real disagreement means the pairing is doing something other than cancelling
        offset = np.max(np.abs(means["transfer_paired"] / means["transfer_naive"] - 1))
        print(f"  paired vs naive, max fractional difference: {offset:.4f}")
        print(f"  rung 0 (normalization): mean over bands {np.mean(means['rung_0']):.4f}")
        print(f"  rung 1 (forward lensing vs CAMB): mean over bands "
              f"{np.mean(means['rung_1']):.4f}")
        flat = np.abs(transfer - 1) <= 3 * error
        print(f"  bands consistent with R = 1 at 3 sigma: {int(np.sum(flat))} of {len(flat)}")

    return dict(band_ells = stacked["band_ells"], transfer = transfer,
                transfer_error = error, estimator = estimator,
                transfer_paired = means["transfer_paired"],
                transfer_paired_error = jackknife["transfer_paired"],
                transfer_naive = means["transfer_naive"],
                transfer_naive_error = jackknife["transfer_naive"],
                rung_0 = means["rung_0"], rung_0_error = jackknife["rung_0"],
                rung_1 = means["rung_1"], rung_1_error = jackknife["rung_1"],
                worst_inverse_error = float(np.max(stacked["inverse_error"])),
                n_realizations = n,
                params = np.array([metadata["params"][name] for name in PARAM_ORDER]),
                param_names = np.array(PARAM_ORDER),
                **{key: metadata[key] for key in _CONFIG_KEYS})
