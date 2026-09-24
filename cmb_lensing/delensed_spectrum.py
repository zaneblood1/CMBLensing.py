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

DROPPING THE FLATNESS ASSUMPTION. merge_transfer_derivatives measures dR/dtheta_i instead of
assuming it is zero, from extra runs at theta_0 +/- Delta_i on the SAME seeds as the reference
run (common random numbers). load_sim draws every field as white noise scaled by sqrt(C(theta)),
so a seed's realizations at neighbouring cosmologies are nearly the same maps and the
per-realization difference R_s(theta_0 + Delta) - R_s(theta_0 - Delta) carries almost none of
the cosmic variance. The Monte Carlo noise therefore cancels in the difference instead of being
amplified by 1 / (2 Delta), and Delta can be chosen far wider than the forecast's step (a few
tenths of a sigma) because the derivative is then applied as a linear model:

    R(l; theta) = R_0(l) + sum_i (theta_i - theta_0,i) dR/dtheta_i(l)

so the stencil's derivative becomes R_0 dC_CAMB/dtheta + C_CAMB dR/dtheta. With both signs of
Delta the merge also reports the second difference, the check that Delta is still linear.

WHAT IS HELD FIXED ACROSS COSMOLOGIES. The forecast freezes the reconstruction at the fiducial
point (Alens_L is computed once and re-applied at every stencil point), so a shifted
measurement feeding a derivative must freeze it too: `reconstruction_params` gives map_joint
the fiducial C_f, C_phi, D and QE norm while the DATA are simulated at the shifted cosmology,
and the CAMB denominator uses the fiducial Alens_L. dR/dtheta then absorbs, among the rest,
the difference between CAMB's frozen-Alens_L residual and what a fixed reconstruction filter
really leaves when C_phi moves. Files record the reconstruction cosmology and the derivative
merge refuses runs whose reconstruction tracked their own shifted cosmology.

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
from cmb_lensing.precompute_camb_1d import GROUND_TRUTH, PARAM_ORDER, PARAM_SIGMA, CAMB_LMAX
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

#relative tolerance for "these two parameter vectors are the same cosmology". The values
#round-trip through np.savez as float64, so anything looser than exact only has to absorb the
#last bit of a GROUND_TRUTH + shift - shift
PARAM_RTOL = 1e-12


def _same_cosmology(a, b):
    return all(np.isclose(a[name], b[name], rtol = PARAM_RTOL, atol = 0) for name in PARAM_ORDER)


def jackknife_mean(rows):
    """(mean, delete-one jackknife error) over the first axis of `rows`.

    The jackknife variance is (n-1)/n * sum_i (x_i - x_bar)^2 over the delete-one means; with a
    single row there is no spread to measure and the error is NaN.
    """
    rows = np.asarray(rows)
    n = rows.shape[0]
    mean = np.mean(rows, axis = 0)
    if n < 2:
        return mean, np.full(rows.shape[1:], np.nan)
    leave_one_out = (np.sum(rows, axis = 0) - rows) / (n - 1)
    error = np.sqrt((n - 1) / n *
                    np.sum((leave_one_out - np.mean(leave_one_out, axis = 0))**2, axis = 0))
    return mean, error


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
                              shared_cls = None, reconstruction_params = None,
                              verbose = True):
    """One realization end to end: simulate, reconstruct, delens, measure, ratio to CAMB.

    This is what get_single_delensed_spectra.py calls - one slurm job, one realization, one
    npz - mirroring mixed_hessian_realization's role for the mixed-Hessian forecast.

    The steps, in order:
      1. load_sim at `map_seed` and the cosmology `param_ground` gives the unlensed field f,
         the true phi, the NOISELESS lensed field L(phi) f, and the noisy data d.
      2. map_joint(d) gives the MAP reconstruction phi_hat. It is used as it comes: a MAP
         estimate is already a regularized optimum, NOT a raw quadratic estimate, so it must
         not be Wiener filtered a second time.
      3. The noiseless lensed field is inverse-lensed by phi_hat. LenseFlow integrates its
         ODE backwards for this, so the operation is exact rather than a Taylor remapping.
      4. All four fields are band-averaged into |l| annuli, and compared against CAMB's own
         unlensed / lensed / partially-lensed spectra on the same bands.

    `reconstruction_params` is the cosmology the RECONSTRUCTION is built at; None means
    `param_ground` itself. When it differs, map_joint gets that cosmology's C_f, C_phi, D and
    QE norm (from a second load_sim at the same seed, used for nothing else) while the data
    stay at `param_ground`, and the CAMB delensed reference uses that cosmology's Alens_L -
    the frozen-estimator convention the forecast's stencil uses, and the one
    merge_transfer_derivatives requires of every shifted run.

    Returns a dict of everything the merge needs, plus the three self-validation rungs.
    """
    if nphi_source not in NPHI_SOURCES:
        raise ValueError(f"nphi_source must be one of {NPHI_SOURCES}, got {nphi_source!r}")
    if qe_response not in QE_RESPONSE_SOURCES:
        raise ValueError(f"qe_response must be one of {QE_RESPONSE_SOURCES}, got "
                         f"{qe_response!r}")
    if reconstruction_params is None:
        reconstruction_params = param_ground
    frozen = not _same_cosmology(reconstruction_params, param_ground)

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))
    edges = band_edges(ell_grid, delta_ell)

    #the reference Alens_L, and CAMB's spectra at it. Deterministic, so identical in every job.
    #Alens_L belongs to the reconstruction, so it comes from reconstruction_params; the
    #spectra it delenses belong to the sky, so they come from param_ground
    alens, nphi_cl, efficiency = frozen_alens(reconstruction_params, nside, theta_pix,
                                              noise_level, l_knee, beam_fwhm, l_cutoff,
                                              nphi_source, qe_response)
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
        if frozen:
            moved = ", ".join(f"{name} {reconstruction_params[name]:.6g} -> "
                              f"{param_ground[name]:.6g}" for name in PARAM_ORDER
                              if not np.isclose(reconstruction_params[name],
                                                param_ground[name], rtol = PARAM_RTOL,
                                                atol = 0))
            print(f"  data simulated at a shifted cosmology ({moved}); map_joint and "
                  f"Alens_L held at the reconstruction cosmology")

    #── 1. simulate ──────────────────────────────────────────────────────
    def simulate(params, cls):
        camb_kwargs = dict(params)
        camb_kwargs["cosmomc_theta"] = camb_kwargs.pop("theta_MC_100") / 100
        camb_kwargs["As"] = float(np.exp(camb_kwargs.pop("logA")) * 1e-10)
        return load_sim(nside, theta_pix, "I", map_seed, **camb_kwargs,
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = cls)

    if shared_cls is None:
        shared_cls = camb_cls_at_params(param_ground)
    data_set = simulate(param_ground, shared_cls)

    if frozen:
        #the operators map_joint reads, rebuilt at the reconstruction cosmology. load_sim
        #derives them from the cosmology alone (the seed only draws the fields, which are
        #discarded), so this is exactly the estimator a fiducial-cosmology run would apply -
        #the same four set_initial_ds_conditions swaps in for the sampler
        reference_set = simulate(reconstruction_params,
                                 camb_cls_at_params(reconstruction_params))
        data_set = data_set.replace(field_covariance = reference_set.field_covariance,
                                    phi_covariance = reference_set.phi_covariance,
                                    mixing_d = reference_set.mixing_d,
                                    quadratic_estimate = reference_set.quadratic_estimate)

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
                delensing_efficiency = efficiency,
                reconstruction_params = np.array([reconstruction_params[name]
                                                  for name in PARAM_ORDER]))


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
    plus {"n_realizations", "paths", "params", "reconstruction_params"}.

    Files written before the reconstruction cosmology was recorded carry no
    `reconstruction_params`; their reconstruction was always built at their own cosmology, so
    that is what they are read as.
    """
    paths = sorted(glob.glob(os.path.join(directory, REALIZATION_GLOB)))
    if not paths:
        raise FileNotFoundError(
            f"no {REALIZATION_GLOB} in {directory}. Run "
            f"sampling_chains/get_delensed_spectra.sh first, or point the merge at the "
            f"out_dir that script writes to.")

    config, params, reconstruction = None, None, None
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
        source = (data["reconstruction_params"] if "reconstruction_params" in data.files
                  else data["params"])
        current_reconstruction = {name: float(source[i]) for i, name in enumerate(PARAM_ORDER)}
        if config is None:
            config, params, reconstruction = current, current_params, current_reconstruction
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; a transfer function cannot mix configurations")
        elif current_params != params:
            raise ValueError(
                f"{path} was produced at cosmology {current_params} but earlier files used "
                f"{params}. R is measured AT a cosmology - averaging across two of them "
                f"destroys exactly the theta dependence compare_transfer_functions.py "
                f"tests for. Use a separate out_dir per cosmology.")
        elif current_reconstruction != reconstruction:
            raise ValueError(
                f"{path} reconstructed phi at {current_reconstruction} but earlier files used "
                f"{reconstruction}. The reconstruction cosmology sets the estimator whose "
                f"residual R measures, so files built with two different ones cannot be "
                f"averaged. Use a separate out_dir per reconstruction convention.")

    return dict(config, params = params, reconstruction_params = reconstruction,
                n_realizations = len(paths), paths = paths)


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
    #D(l) per realization: the delensed spectrum against the SAME draw's unlensed one, the
    #CAMB-free delensing measured_delensing_ratio averages and
    #merge_delensing_ratio_derivatives differentiates. Derived here so it is one more row set
    #alongside the two R estimators and every consumer sees the same per-realization pairing
    stacked["delensing_ratio"] = stacked["measured_delensed"] / stacked["measured_unlensed"]
    stacked["band_ells"] = np.asarray(band_ells)
    metadata["seeds"] = seeds

    if verbose:
        print(f"Loaded {len(seeds)} realizations from {directory}")
        print(f"  nside {metadata['nside']}, {metadata['theta_pix']:g}', "
              f"{metadata['noise_level']:g} uK-arcmin, l_knee {metadata['l_knee']:g}, "
              f"bands of {metadata['delta_ell']:g}")
        print(f"  N_phi source {metadata['nphi_source']}, QE response "
              f"{metadata['qe_response']}, map_joint {metadata['map_joint_steps']} steps")
        if not _same_cosmology(metadata["reconstruction_params"], metadata["params"]):
            print(f"  reconstruction held at {metadata['reconstruction_params']} while the "
                  f"data sit at {metadata['params']}")
        print(f"  worst inverse-lensing round trip over all realizations: "
              f"{np.max(stacked['inverse_error']):.3e}")

    return stacked, metadata


def measured_delensing_ratio(directory, shifted_dirs = None, verbose = True):
    """D(l) = < C_l^delensed / C_l^unlensed >, the box's OWN delensing, CAMB-free.

    R(l) (merge_transfer_function) is a ratio to CAMB's partially lensed spectrum at a frozen
    Alens_L, so using it still requires CAMB's delensing algorithm and a delensing fraction.
    D(l) instead compares the measured delensed spectrum to the measured UNLENSED one from
    the SAME realization, so nothing CAMB computes enters and there is no Alens_L anywhere:

        C_l^delensed(theta)  =  D(l) * C_l^unlensed_CAMB(theta)

    Dividing per realization before averaging is the same cosmic-variance cancellation that
    makes the "paired" R estimator affordable - both spectra come from one draw of f, so the
    sample variance divides out and what is left is the delensing itself. The error is the
    same delete-one jackknife over realizations.

    WHAT IS ASSUMED. D is measured at ONE cosmology and then applied at every stencil point,
    so the forecast's dC^delensed/dtheta is D * dC^unlensed/dtheta - the entire residual
    lensing correction is frozen in theta. That is a STRONGER assumption than the transfer
    function's: R is a ratio between two delensed spectra and is near one by construction,
    while D carries the whole residual lensing, which tracks C_phi(theta). Nothing here tests
    it; run two directories at separated cosmologies and compare, the way
    compare_transfer_functions.py does for R.

    Returns a dict shaped like the merged npz - band_ells, ratio, ratio_error and the
    configuration - so it can be checked against a forecast's box the same way.
    """
    stacked, metadata = load_transfer_directory(directory, verbose = verbose)
    rows = stacked["measured_delensed"] / stacked["measured_unlensed"]
    ratio, error = jackknife_mean(rows)

    if verbose:
        print(f"  measured delensing ratio D(l) = C^delensed / C^unlensed over "
              f"{len(stacked['band_ells'])} bands, "
              f"l = {stacked['band_ells'][0]:.0f}..{stacked['band_ells'][-1]:.0f}")
        print(f"    D in {np.min(ratio):.4f}..{np.max(ratio):.4f} "
              f"({metadata['n_realizations']} realizations); D = 1 would be perfect "
              f"delensing, D = C^lensed/C^unlensed none at all")

    result = dict(band_ells = np.asarray(stacked["band_ells"]), ratio = ratio,
                  ratio_error = error,
                  **{key: metadata[key] for key in _CONFIG_KEYS},
                  params = metadata["params"],
                  reconstruction_params = metadata["reconstruction_params"],
                  n_realizations = metadata["n_realizations"])

    #dD/dtheta turns D * C^unlensed from a restatement of the unlensed forecast into a real
    #one - see merge_delensing_ratio_derivatives for why the first term alone cancels
    if shifted_dirs:
        result.update(merge_delensing_ratio_derivatives(directory, shifted_dirs,
                                                        verbose = verbose))
    return result


def has_ratio_derivatives(measured):
    """Whether a measured_delensing_ratio dict carries dD/dtheta."""
    return ("ratio_derivative" in measured and
            len(np.asarray(measured["ratio_derivative"])) > 0)


def ratio_at_params(measured, params = None):
    """D(l) on its measured bands at `params`: D_0 + sum_i (theta_i - theta_0,i) dD/dtheta_i.

    Mirrors fisher_forecast.transfer_at_params. Without derivatives, or with params = None,
    this is D_0 - the frozen construction, which makes the forecast collapse onto the
    unlensed one. Parameters with no measured derivative contribute nothing.
    """
    ratio = np.asarray(measured["ratio"], dtype = np.float64)
    if params is None or not has_ratio_derivatives(measured):
        return ratio
    reference = measured["params"]
    for name, derivative in zip(measured["derivative_names"],
                                measured["ratio_derivative"]):
        name = str(name)
        ratio = ratio + (float(params[name]) - float(reference[name])) * np.asarray(derivative)
    return ratio


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
        means[name], jackknife[name] = jackknife_mean(stacked[name])

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
                reconstruction_params = np.array([metadata["reconstruction_params"][name]
                                                  for name in PARAM_ORDER]),
                param_names = np.array(PARAM_ORDER),
                **{key: metadata[key] for key in _CONFIG_KEYS})


# ── dR/dtheta from common-random-number runs ──────────────────────────────

def _rows_by_seed(stacked, metadata, key, seeds):
    """The per-realization rows of `key`, reordered to follow `seeds`."""
    position = {seed: i for i, seed in enumerate(metadata["seeds"])}
    return np.asarray(stacked[key])[[position[seed] for seed in seeds]]


def merge_delensing_ratio_derivatives(reference_dir, shifted_dirs, verbose = True):
    """dD/dtheta_i for the empirical delensing ratio D(l) = <C^delensed / C^unlensed>.

    Exactly merge_transfer_derivatives, differencing the per-realization D rows instead of
    the per-realization R rows: same common-random-number requirement, same one-parameter-per
    -directory rule, same frozen-reconstruction rule, same per-realization differencing and
    jackknife. See that function for all of it.

    Why this exists: without dD/dtheta the empirical delensed spectrum D * C^unlensed(theta)
    has a derivative D * dC^unlensed/dtheta, and D then CANCELS out of the Fisher integrand
    wherever the noise is small - measured at nside 128 / 2.5' / 5 uK the forecast came out
    equal to --spectra unlensed to seven digits. The whole content of the delensing sits in
    the second term of the product rule,

        d(D C_u)/dtheta = D dC_u/dtheta + C_u dD/dtheta

    so measuring dD/dtheta is what makes the mode a forecast rather than a restatement of the
    unlensed one.

    Returns `ratio_derivative`, `ratio_derivative_error`, `derivative_names` and the rest,
    named like merge_transfer_derivatives' output with "transfer" -> "ratio".
    """
    return _merge_row_derivatives(reference_dir, shifted_dirs, key = "delensing_ratio",
                                  prefix = "ratio", label = "dD/dtheta",
                                  camb_key = "camb_unlensed", verbose = verbose)


def merge_transfer_derivatives(reference_dir, shifted_dirs, estimator = "paired",
                               verbose = True):
    """dR/dtheta_i for every parameter displaced in `shifted_dirs`, relative to `reference_dir`.

    `reference_dir` is the production run at the fiducial cosmology. Each shifted directory is
    a run of get_delensed_spectra.sh with exactly ONE parameter displaced (by any amount, of
    either sign), the reconstruction held at the reference cosmology, and the same seeds as
    the reference - the common random numbers are what make this affordable, so a directory
    that shares no seeds with the reference is refused rather than differenced
    independently. A parameter may have one shifted directory (a one-sided difference against
    the reference, error O(Delta)) or two of opposite sign (a central difference, error
    O(Delta^2)); two on the same side are refused.

    Per parameter, only the seeds present in the reference AND every one of its shifted
    directories are used, so a failed job costs that realization and nothing else. The
    derivative is formed PER REALIZATION and then jackknifed - the rows at the different
    cosmologies are strongly correlated, which is the whole point, and propagating their
    errors as independent would miss that.

    Returns a dict of the derivative arrays, shaped (n_derivative, n_band) and ordered like
    `derivative_names`, ready to be written into the merged transfer_function.npz next to the
    output of merge_transfer_function(reference_dir).
    """
    if estimator not in ("paired", "naive"):
        raise ValueError(f"estimator must be 'paired' or 'naive', got {estimator!r}")
    return _merge_row_derivatives(reference_dir, shifted_dirs,
                                  key = f"transfer_{estimator}", prefix = "transfer",
                                  label = f"dR/dtheta [{estimator} estimator]",
                                  camb_key = "camb_delensed", verbose = verbose)


def _merge_row_derivatives(reference_dir, shifted_dirs, key, prefix, label, camb_key,
                           verbose = True):
    """The shared machinery behind merge_transfer_derivatives and
    merge_delensing_ratio_derivatives.

    `key` names the per-realization row set in load_transfer_directory's `stacked` to
    differentiate, `prefix` the output field names ("transfer" -> transfer_derivative, ...),
    `camb_key` the deterministic CAMB band spectrum the report compares the new term against
    (C^delensed for R, since the forecast forms R * C^delensed; C^unlensed for D, since it
    forms D * C^unlensed).
    """
    stack_ref, meta_ref = load_transfer_directory(reference_dir, verbose = False)
    reference = meta_ref["params"]
    if not _same_cosmology(meta_ref["reconstruction_params"], reference):
        raise ValueError(f"{reference_dir} reconstructed at a different cosmology than its "
                         f"data; the reference run must be an unshifted production run")

    #collect every shifted run by (parameter, sign of the displacement)
    runs = {}
    for directory in shifted_dirs:
        stack, meta = load_transfer_directory(directory, verbose = False)
        for config_key in _CONFIG_KEYS:
            if meta[config_key] != meta_ref[config_key]:
                raise ValueError(f"{directory} disagrees with the reference on {config_key} "
                                 f"({meta[config_key]!r} vs {meta_ref[config_key]!r}); only "
                                 f"the cosmology may differ")
        if not np.allclose(stack["band_ells"], stack_ref["band_ells"]):
            raise ValueError(f"{directory} uses different bands than the reference")

        displaced = [name for name in PARAM_ORDER
                     if not np.isclose(meta["params"][name], reference[name],
                                       rtol = PARAM_RTOL, atol = 0)]
        if len(displaced) != 1:
            raise ValueError(f"{directory} displaces {displaced or 'nothing'} relative to "
                             f"the reference; each derivative run must move exactly one "
                             f"parameter")
        name = displaced[0]

        if not _same_cosmology(meta["reconstruction_params"], reference):
            raise ValueError(
                f"{directory} reconstructed phi at its own shifted cosmology rather than the "
                f"reference's. The forecast freezes the reconstruction (Alens_L) across its "
                f"stencil, so a derivative taken with a reconstruction that tracks theta "
                f"measures a different quantity. Re-run it with the reconstruction frozen "
                f"(get_delensed_spectra.sh's freeze_reconstruction=1, the default).")

        delta = meta["params"][name] - reference[name]
        sign = int(np.sign(delta))
        if (name, sign) in runs:
            raise ValueError(f"two runs displace {name} to the same side "
                             f"({runs[(name, sign)][2]} and {directory}); pass one per sign")
        runs[(name, sign)] = (stack, meta, directory, delta)

    names = [name for name in PARAM_ORDER if (name, 1) in runs or (name, -1) in runs]
    if not names:
        raise ValueError("no shifted directories were given, so there is nothing to "
                         "differentiate")

    #dC_CAMB/dtheta on the same bands, for the report that says how much of the stencil's
    #derivative the new term actually is. CAMB's band spectra are deterministic, so any one
    #row of a run carries them exactly
    camb_ref = np.asarray(stack_ref[camb_key])[0]
    transfer_ref, _ = jackknife_mean(stack_ref[key])

    result = {"derivative_names": [], f"{prefix}_derivative": [],
              f"{prefix}_derivative_error": [], "derivative_deltas": [],
              "derivative_schemes": [], "derivative_n_realizations": [],
              f"{prefix}_second_difference": [], f"{prefix}_second_difference_error": []}

    if verbose:
        print(f"\n{label} from common-random-number runs, reference {reference_dir}")

    for name in names:
        plus, minus = runs.get((name, 1)), runs.get((name, -1))
        present = [run for run in (plus, minus) if run is not None]

        #the seeds every run of this parameter AND the reference share
        seeds = set(meta_ref["seeds"])
        for _, meta, _, _ in present:
            seeds &= set(meta["seeds"])
        seeds = sorted(seeds)
        if len(seeds) < 2:
            raise ValueError(
                f"the runs for {name} share {len(seeds)} seed(s) with the reference. The "
                f"derivative is only affordable with common random numbers - re-run the "
                f"shifted cosmology with the reference's seed_prefix and num_realizations.")

        rows_ref = _rows_by_seed(stack_ref, meta_ref, key, seeds)
        rows = {sign: _rows_by_seed(run[0], run[1], key, seeds)
                for sign, run in ((1, plus), (-1, minus)) if run is not None}

        if plus is not None and minus is not None:
            scheme = "central"
            deltas = (plus[3], minus[3])
            derivative_rows = (rows[1] - rows[-1]) / (plus[3] - minus[3])
            #the second difference, normalized to the size of the first-order change it would
            #contaminate: R(+) + R(-) - 2 R(0) for a symmetric Delta, generalised to unequal
            #ones through the divided difference
            curvature_rows = 2 * ((rows[1] - rows_ref) / plus[3] -
                                  (rows_ref - rows[-1]) / (-minus[3])) / (plus[3] - minus[3])
            camb_step = (np.asarray(plus[0][camb_key])[0] -
                         np.asarray(minus[0][camb_key])[0]) / (plus[3] - minus[3])
        else:
            (stack, meta, _, delta), = present
            scheme = "forward" if delta > 0 else "backward"
            deltas = (delta,)
            derivative_rows = (rows[int(np.sign(delta))] - rows_ref) / delta
            curvature_rows = None
            camb_step = (np.asarray(stack[camb_key])[0] - camb_ref) / delta

        derivative, error = jackknife_mean(derivative_rows)
        if curvature_rows is not None:
            curvature, curvature_error = jackknife_mean(curvature_rows)
        else:
            curvature = curvature_error = np.full_like(derivative, np.nan)

        result["derivative_names"].append(name)
        result[f"{prefix}_derivative"].append(derivative)
        result[f"{prefix}_derivative_error"].append(error)
        result["derivative_deltas"].append(np.array(deltas + (np.nan,) * (2 - len(deltas))))
        result["derivative_schemes"].append(scheme)
        result["derivative_n_realizations"].append(len(seeds))
        result[f"{prefix}_second_difference"].append(curvature)
        result[f"{prefix}_second_difference_error"].append(curvature_error)

        if verbose:
            _report_derivative(name, scheme, deltas, len(seeds), stack_ref["band_ells"],
                               derivative, error, curvature, transfer_ref, camb_ref,
                               camb_step, symbol = "R" if prefix == "transfer" else "D")

    return {name: np.array(value) for name, value in result.items()}


def _report_derivative(name, scheme, deltas, n, ells, derivative, error, curvature,
                       transfer_ref, camb_ref, camb_step, symbol = "R"):
    """Print one parameter's dR/dtheta (or dD/dtheta) and the numbers that say if it matters."""
    sigma = PARAM_SIGMA[name]
    shifts = ", ".join(f"{delta:+.4g} ({delta / sigma:+.2f} sigma)" for delta in deltas)
    print(f"\n  {name}: {scheme} difference over {shifts}, {n} shared realizations")
    print(f"  {'l':>8}{f'd{symbol}/dtheta':>13}{'jackknife':>12}{'sigma':>8}{f'C d{symbol} / {symbol} dC':>14}")
    #the stencil's derivative is X dC_CAMB + C_CAMB dX; the last column is the second term
    #against the first, i.e. the fractional error holding R flat would have made per band
    safe_step = np.where(camb_step != 0, camb_step, np.nan)
    relative = camb_ref * derivative / (transfer_ref * safe_step)
    for i, ell in enumerate(ells):
        significance = derivative[i] / error[i] if error[i] > 0 else np.nan
        print(f"  {ell:8.0f}{derivative[i]:13.4e}{error[i]:12.3e}{significance:8.1f}"
              f"{relative[i]:14.4f}")

    good = error > 0
    chi_squared = float(np.sum((derivative[good] / error[good])**2))
    print(f"  chi^2 against d{symbol}/dtheta = 0: {chi_squared:.1f} for {int(np.sum(good))} bands")
    print(f"  |C d{symbol} / {symbol} dC|: max {np.nanmax(np.abs(relative)):.3e}, median "
          f"{np.nanmedian(np.abs(relative)):.3e} -> the fractional correction to "
          f"dC^delensed/d{name} that holding {symbol} flat would have dropped")
    #how much R moves over one sigma, the scale the Fisher contour lives on
    print(f"  {symbol} drift over one sigma: max {np.max(np.abs(derivative)) * sigma:.3e}")
    if scheme == "central":
        #over the half-width of the measured displacement, how big the quadratic term is next
        #to the linear one. The linear model R_0 + (theta - theta_0) dR is only safe where
        #this is small; it is irrelevant at the forecast's own step, which is far narrower
        half_width = 0.5 * (deltas[0] - deltas[1])
        ratio = np.abs(0.5 * curvature * half_width**2) / np.maximum(
            np.abs(derivative * half_width), np.finfo(float).tiny)
        print(f"  second-order / first-order term at the measured Delta: median "
              f"{np.median(ratio):.3f}, max {np.max(ratio):.3f}"
              + ("  <- Delta may be too wide for a linear model" if np.median(ratio) > 0.3
                 else ""))
