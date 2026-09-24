"""The delensed TT covariance measured EMPIRICALLY at every finite-difference stencil point.

WHY THIS EXISTS. delensed_spectrum.py measures R(l) = C^delensed[box] / C^delensed[CAMB] and
fisher_forecast's --transfer_function rescales CAMB's partially lensed spectrum by it, so the
forecast's delensed f block is still CAMB's spectrum with an empirical correction on top - and
its theta dependence is still CAMB's frozen-Alens_L theta dependence plus a linear dR/dtheta.
That has not brought the forecast onto the chains. This module drops CAMB from the delensed
block altogether: the block's value AND its derivative both come from what this codebase's
lense_flow and map_joint actually produce.

WHAT IS MEASURED. For one seed (one set of common random numbers) and every point of the
central-difference stencil {theta_0, theta_0 +/- h_i e_i}, i over the sampled parameters:

    1. load_sim(theta) at the SAME seed. load_sim draws every field as white noise scaled by
       sqrt(C(theta)), and the instrument noise from its own key, so across the stencil the
       realizations differ ONLY through the cosmology - the Monte Carlo noise cancels in
       C(theta_0 + h) - C(theta_0 - h) instead of being amplified by 1 / (2h).
    2. phi_hat(theta) = map_joint(d(theta)).
    3. f_delensed(theta) = lense_flow(L(phi) f [NOISELESS], phi_hat, INVERSE).
    4. C_delensed(theta)[k] = F[k] conj(F[k]) / nside^2, F = rfft2(f_delensed), on the rfft
       grid, with the [0, 0] origin set to zero.

The normalization in 4 is covar_matrix_from_cls's grid convention (C_l / pix_width^2), since
field_from_covar_single_key gives E|rfft2(f)|^2 = nside^2 * C_grid - so the averaged matrix
drops straight into covariance_blocks next to the analytic C_n and C_phi + N_phi. No Fourier
weight enters a PER-MODE covariance: w_k counts how many real degrees of freedom an rfft entry
carries (2 in the bulk, 1 on the two self-conjugate columns), which is how many independent
draws of that one variance the entry holds, not a rescaling of the variance itself; E|F|^2 =
nside^2 C holds on the self-conjugate columns too, because the fields are drawn from REAL
white noise. The weights enter the Fisher contraction (_fisher_from_blocks) and the band
averages below, exactly where they always have.

The unlensed and (noiseless) lensed fields go through step 4 too. They are free - the fields
are already in hand - and they are the validation: with common random numbers the unlensed
per-mode ratio C(theta + h) / C(theta - h) is EXACTLY Cf(theta + h) / Cf(theta - h) for every
realization, so the empirical unlensed dlnC/dtheta must reproduce CAMB's to machine precision.
If it does not, the stencil points were not on common random numbers and nothing else in the
merge can be trusted.

WHAT IS HELD FIXED ACROSS THE STENCIL. `reconstruction = "fiducial"` (the default) builds
map_joint's C_f, C_phi, D and QE norm at theta_0 at EVERY stencil point while the DATA move
with theta, so the delensing is a fixed estimator applied to data from different cosmologies -
the same frozen-estimator convention covariance_stencil uses for N_phi / Alens_L and
delensed_spectrum.py uses for dR/dtheta. `reconstruction = "shifted"` rebuilds them at each
point's own cosmology, so dC/dtheta additionally carries how the estimator itself moves.

WHAT DELENSING MEANS HERE. As in delensed_spectrum.py: the NOISELESS lensed field is
inverse-lensed, and the forecast adds the isotropic C_n back on - covariance_blocks's
"C_TT^delensed + C_n" convention.

The measurement is fanned out one slurm job per seed by
sampling_chains_TEMPLATE/get_delensed_covariance.sh -> run_single_delensed_covariance.sh ->
run_single_delensed_covariance.py, merged locally by merge_delensed_covariance.py, and
consumed by fisher_forecast.py's --delensed_covariance.
"""

import glob
import os

import numpy as np

import jax.numpy as jnp

from cmb_lensing.constants import FORWARD_LENSE, INVERSE_LENSE
from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.map_joint import map_joint
from cmb_lensing.simulate import load_sim, covar_matrix_from_cls
from cmb_lensing.precompute_camb_1d import PARAM_ORDER, PARAM_SIGMA
from cmb_lensing.fisher_forecast import camb_cls_at_params, load_sim_cosmology
from cmb_lensing.delensed_spectrum import (LENSE_STEPS, DEFAULT_DELTA_ELL, PARAM_RTOL,
                                           band_edges, band_average, jackknife_mean,
                                           _lense, _scalar_matrix, _same_cosmology)


#the displacement h_i of the stencil, in units of PARAM_SIGMA. NOT fisher_forecast's 0.05
#sigma: every stencil point runs its own map_joint, and a BFGS optimum is only as smooth in
#theta as its convergence, so a very narrow step would difference convergence noise. Common
#random numbers keep the difference quiet at any h, so make it wide enough to clear that
#noise and narrow enough that C(theta) is still linear - the merge prints the second-order
#term as the check. The forecast takes its step from the merged file, whatever this is
DEFAULT_STEP_SIGMA = 0.5

#which cosmology map_joint's operators are built at - see the module docstring
RECONSTRUCTIONS = ("fiducial", "shifted")
DEFAULT_RECONSTRUCTION = "fiducial"

#the three fields step 4 is applied to; "delensed" is the product, the other two validate
FIELD_KINDS = ("unlensed", "lensed", "delensed")

REALIZATION_GLOB = "delensed_covariance_*.npz"
MERGED_NAME = "delensed_covariance.npz"


def realization_file_name(realization_index):
    return f"delensed_covariance_{realization_index:04d}.npz"


def stencil_points(param_ground, names, step_sigma = DEFAULT_STEP_SIGMA):
    """(offsets, points, steps): the 1 + 2k central-difference points in a fixed order.

    `offsets` is (1 + 2k, k) in units of h - the centre, then +h_i and -h_i for each name in
    `names` order, the first-derivative subset of fisher_forecast.stencil_offsets. `points`
    are the matching full five-parameter dicts, `steps` the h_i = step_sigma * PARAM_SIGMA.
    """
    steps = [step_sigma * PARAM_SIGMA[name] for name in names]
    offsets, points = [[0] * len(names)], [dict(param_ground)]
    for i, (name, step) in enumerate(zip(names, steps)):
        for sign in (+1, -1):
            offset = [0] * len(names)
            offset[i] = sign
            point = dict(param_ground)
            point[name] = param_ground[name] + sign * step
            offsets.append(offset)
            points.append(point)
    return np.array(offsets, dtype = int), points, np.array(steps)


def grid_covariance(field_fourier, nside):
    """F conj(F) / nside^2 on the rfft grid: one realization's per-mode covariance, in
    covar_matrix_from_cls's C_l / pix_width^2 units, with the [0, 0] origin zeroed as every
    covar_matrix_from_cls(..., origin_value = 0) block has it."""
    power = jnp.real(field_fourier * jnp.conj(field_fourier)) / nside**2
    return np.asarray(power.at[0, 0].set(0.0))


def measure_delensed_covariance(nside, theta_pix, noise_level, param_ground, map_seed,
                                names, step_sigma = DEFAULT_STEP_SIGMA, l_knee = 0.0,
                                map_joint_steps = 30,
                                reconstruction = DEFAULT_RECONSTRUCTION,
                                on_point = None, verbose = True):
    """One seed through the whole stencil: simulate, reconstruct, delens, square.

    `names` are the parameters to differentiate (any order; stored as given). `on_point`,
    if given, is called with the partial result dict after every stencil point, so the job
    can checkpoint - a job killed at the wall clock keeps every finished point.

    Returns a dict with, for each kind in FIELD_KINDS, `{kind}` of shape
    (1 + 2k, nside, nside // 2 + 1), plus the stencil (`offsets`, `point_params` in
    PARAM_ORDER, `steps`), the inverse-lensing round-trip error at the centre and
    `n_done`, the number of stencil points actually measured.
    """
    if reconstruction not in RECONSTRUCTIONS:
        raise ValueError(f"reconstruction must be one of {RECONSTRUCTIONS}, got "
                         f"{reconstruction!r}")
    unknown = [name for name in names if name not in PARAM_ORDER]
    if unknown or not names:
        raise ValueError(f"names must be a non-empty subset of {PARAM_ORDER}, got {names}")

    offsets, points, steps = stencil_points(param_ground, names, step_sigma)
    shape = (len(points), nside, nside // 2 + 1)
    result = dict(offsets = offsets, steps = steps,
                  point_params = np.array([[point[name] for name in PARAM_ORDER]
                                           for point in points]),
                  inverse_error = np.nan, n_done = 0,
                  **{kind: np.full(shape, np.nan) for kind in FIELD_KINDS})

    def simulate(params):
        return load_sim(nside, theta_pix, "I", map_seed, **load_sim_cosmology(params),
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = camb_cls_at_params(params))

    if verbose:
        print(f"seed {map_seed}: nside {nside}, {theta_pix:g}', {noise_level:g} uK-arcmin, "
              f"l_knee {l_knee:g}; {len(points)} stencil points over {list(names)} at "
              f"+/- {step_sigma:g} sigma, reconstruction at the {reconstruction} cosmology")

    fiducial_set = None
    for index, (offset, params) in enumerate(zip(offsets, points)):
        data_set = simulate(params)
        if index == 0:
            fiducial_set = data_set
        elif reconstruction == "fiducial":
            #the operators map_joint reads, at theta_0 - the same four fields
            #delensed_spectrum.measure_delensed_spectrum swaps in for a frozen estimator.
            #load_sim builds them from the cosmology alone, so this is exactly the estimator
            #the centre point applies
            data_set = data_set.replace(field_covariance = fiducial_set.field_covariance,
                                        phi_covariance = fiducial_set.phi_covariance,
                                        mixing_d = fiducial_set.mixing_d,
                                        quadratic_estimate = fiducial_set.quadratic_estimate)

        _, phi_hat = map_joint(data_set, num_steps = map_joint_steps)
        fields = dict(unlensed = data_set.unlensed_field, lensed = data_set.lensed_field,
                      delensed = _lense(data_set.lensed_field, phi_hat, INVERSE_LENSE))
        for kind in FIELD_KINDS:
            result[kind][index] = grid_covariance(_scalar_matrix(fields[kind]), nside)

        if index == 0:
            #the inverse must undo the forward at the TRUE phi - delensed_spectrum's rung 2,
            #once per job since the operator does not depend on the stencil point
            unlensed = data_set.unlensed_field
            round_trip = _lense(_lense(unlensed, data_set.phi, FORWARD_LENSE),
                                data_set.phi, INVERSE_LENSE)
            scale = float(np.max(np.abs(np.asarray(_scalar_matrix(unlensed)))))
            result["inverse_error"] = float(np.max(np.abs(
                np.asarray(_scalar_matrix(round_trip)) -
                np.asarray(_scalar_matrix(unlensed))))) / scale

        result["n_done"] = index + 1
        if verbose:
            label = ("centre" if index == 0 else
                     f"{names[int(np.argmax(np.abs(offset)))]} "
                     f"{'+' if offset.sum() > 0 else '-'}h")
            ratio = (np.sum(result["delensed"][index]) / np.sum(result["unlensed"][index]))
            print(f"  point {index + 1}/{len(points)} ({label}): "
                  f"sum C_delensed / sum C_unlensed = {ratio:.4f}", flush = True)
        if on_point is not None:
            on_point(result)

    if verbose:
        print(f"  inverse-lensing round trip at the true phi: max relative error "
              f"{result['inverse_error']:.3e}")
    return result


# ── Collecting the per-realization files ──────────────────────────────────

#everything that must agree between two files before they may be averaged
_CONFIG_KEYS = ("nside", "theta_pix", "noise_level", "l_knee", "map_joint_steps",
                "lense_steps", "reconstruction", "step_sigma")
_INT_KEYS = ("nside", "map_joint_steps", "lense_steps")


def _read_config(data):
    config = {}
    for key in _CONFIG_KEYS:
        value = data[key]
        config[key] = (str(value) if value.dtype.kind in "US"
                       else (int(value) if key in _INT_KEYS else float(value)))
    config["names"] = tuple(str(name) for name in data["names"])
    return config


def load_covariance_directory(directory, verbose = True):
    """Stack every finished delensed_covariance_*.npz in `directory`.

    Refuses files that disagree on the box, the reconstruction, the stencil (names, step,
    fiducial cosmology) or that repeat a seed. Files a job has not finished (checkpoints of a
    job still running or killed at the wall clock) are set aside with a warning rather than
    averaged in, since they are missing stencil points.

    Returns (stacked, metadata): `stacked[kind]` is (n_realizations, 1 + 2k, nside,
    nside // 2 + 1) per FIELD_KINDS entry, plus `inverse_error` per realization.
    """
    paths = sorted(glob.glob(os.path.join(directory, REALIZATION_GLOB)))
    if not paths:
        raise FileNotFoundError(
            f"no {REALIZATION_GLOB} in {directory}. Run "
            f"sampling_chains/get_delensed_covariance.sh first and scp its out_dir here.")

    config, offsets, point_params, seeds, unfinished = None, None, None, [], []
    stacked = {kind: [] for kind in FIELD_KINDS}
    stacked["inverse_error"] = []
    for path in paths:
        data = np.load(path, allow_pickle = True)
        if not bool(data["finished"]):
            unfinished.append((os.path.basename(path), int(data["n_done"]),
                               len(data["offsets"])))
            continue
        current = _read_config(data)
        if config is None:
            config, offsets, point_params = current, data["offsets"], data["point_params"]
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; an empirical covariance cannot mix configurations")
        elif not np.allclose(data["point_params"], point_params, rtol = PARAM_RTOL, atol = 0):
            raise ValueError(f"{path} sits on a different stencil (cosmologies) than earlier "
                             f"files; averaging them would blur exactly the theta "
                             f"dependence being measured. Use one out_dir per stencil.")
        for kind in FIELD_KINDS:
            stacked[kind].append(np.asarray(data[kind]))
        stacked["inverse_error"].append(float(data["inverse_error"]))
        seeds.append(int(data["map_seed"]))

    if unfinished and verbose:
        print(f"  WARNING: setting aside {len(unfinished)} unfinished file(s): "
              + ", ".join(f"{name} ({done}/{total} points)"
                          for name, done, total in unfinished))
    if not seeds:
        raise ValueError(f"no finished files in {directory}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                         f"realizations would be double counted - check the seed_prefix "
                         f"loop in get_delensed_covariance.sh.")

    stacked = {key: np.array(rows) for key, rows in stacked.items()}
    metadata = dict(config, offsets = offsets, point_params = point_params, seeds = seeds,
                    n_realizations = len(seeds), n_unfinished = len(unfinished))
    if verbose:
        print(f"Loaded {len(seeds)} realizations x {len(offsets)} stencil points from "
              f"{directory}")
        print(f"  nside {config['nside']}, {config['theta_pix']:g}', "
              f"{config['noise_level']:g} uK-arcmin, l_knee {config['l_knee']:g}, "
              f"map_joint {config['map_joint_steps']} steps, reconstruction at the "
              f"{config['reconstruction']} cosmology")
        print(f"  stencil: {list(config['names'])} at +/- {config['step_sigma']:g} sigma")
        print(f"  worst inverse-lensing round trip: "
              f"{np.max(stacked['inverse_error']):.3e}")
    return stacked, metadata


def _point_index(offsets, i, sign):
    target = np.zeros(offsets.shape[1], dtype = int)
    target[i] = sign
    return int(np.flatnonzero(np.all(offsets == target, axis = 1))[0])


def _camb_grid_covariances(point_params, nside, pix_width, ell_grid):
    """CAMB's unlensed and lensed TT on the grid at every stencil point, same units."""
    result = {"unlensed": [], "lensed": []}
    for row in point_params:
        cls = camb_cls_at_params({name: float(row[i]) for i, name in enumerate(PARAM_ORDER)})
        ells = jnp.arange(2, 2 + cls["scalar_TT"].shape[0]).astype(jnp.float64)
        for kind, source in (("unlensed", "scalar_TT"), ("lensed", "total_TT")):
            result[kind].append(np.asarray(covar_matrix_from_cls(
                nside, pix_width, ell_grid, ells, cls[source], origin_value = 0)))
    return {kind: np.array(rows) for kind, rows in result.items()}


def merge_delensed_covariance(directory, delta_ell = DEFAULT_DELTA_ELL, verbose = True):
    """Average the per-realization matrices into the 1 + 2k empirical covariances.

    The product is the realization MEAN of each kind at each stencil point - what the
    forecast differentiates. Alongside it, band-averaged diagnostics (never used by the
    forecast, only printed and plotted):
      * dlnC/dtheta_i per band for every kind, formed PER REALIZATION as
        band(C+ - C-) / band(C0) / 2h and jackknifed - the common random numbers make these
        rows strongly correlated across the stencil, which is the point, so their errors
        must not be propagated as independent
      * CAMB's dlnC/dtheta_i per band, unlensed and lensed, on the same grid and bands. The
        EMPIRICAL UNLENSED one must match CAMB's to machine precision (see the module
        docstring) - that is the common-random-number check
      * the second-order term |C+ + C- - 2 C0| / |C+ - C-| per band, the linearity check on h
      * D(l) = C_delensed / C_unlensed at the centre, how much lensing is left

    Returns the dict written to delensed_covariance.npz.
    """
    stacked, metadata = load_covariance_directory(directory, verbose = verbose)
    nside, names = metadata["nside"], list(metadata["names"])
    offsets, point_params = metadata["offsets"], metadata["point_params"]
    steps = np.array([metadata["step_sigma"] * PARAM_SIGMA[name] for name in names])

    ell_grid, pix_width = gen_ell_grid(nside, metadata["theta_pix"])
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))
    edges = band_edges(ell_grid, delta_ell)

    def band(matrix):
        centres, values, _ = band_average(matrix, ell_grid, weights, edges)
        return centres, values

    band_ells, _ = band(stacked["unlensed"][0, 0])
    #stencil_points puts the centre first
    centre = 0
    assert not np.any(offsets[centre])
    plus = [_point_index(offsets, i, +1) for i in range(len(names))]
    minus = [_point_index(offsets, i, -1) for i in range(len(names))]

    merged = dict(names = np.array(names), steps = steps, offsets = offsets,
                  point_params = point_params, param_names = np.array(PARAM_ORDER),
                  params = point_params[centre], n_realizations = metadata["n_realizations"],
                  seeds = np.array(metadata["seeds"]),
                  worst_inverse_error = float(np.max(stacked["inverse_error"])),
                  band_ells = band_ells, delta_ell = delta_ell,
                  **{key: metadata[key] for key in _CONFIG_KEYS})

    camb = _camb_grid_covariances(point_params, nside, pix_width, ell_grid)

    for kind in FIELD_KINDS:
        mean = np.mean(stacked[kind], axis = 0)
        merged[f"{kind}_fid"] = mean[centre]
        merged[f"{kind}_plus"] = mean[plus]
        merged[f"{kind}_minus"] = mean[minus]

        #band-level per-realization log-derivatives, jackknifed
        derivative, error, curvature = [], [], []
        for i in range(len(names)):
            rows = []
            for realization in stacked[kind]:
                _, up = band(realization[plus[i]])
                _, down = band(realization[minus[i]])
                _, mid = band(realization[centre])
                rows.append((up - down) / (2 * steps[i] * mid))
            value, value_error = jackknife_mean(np.array(rows))
            derivative.append(value)
            error.append(value_error)
            _, up = band(mean[plus[i]])
            _, down = band(mean[minus[i]])
            _, mid = band(mean[centre])
            curvature.append(np.abs(up + down - 2 * mid) /
                             np.maximum(np.abs(up - down), np.finfo(float).tiny))
        merged[f"band_dlnc_{kind}"] = np.array(derivative)
        merged[f"band_dlnc_{kind}_error"] = np.array(error)
        merged[f"band_curvature_{kind}"] = np.array(curvature)
        merged[f"band_c_{kind}"] = band(mean[centre])[1]

    for kind in ("unlensed", "lensed"):
        _, mid = band(camb[kind][centre])
        merged[f"band_c_camb_{kind}"] = mid
        merged[f"band_dlnc_camb_{kind}"] = np.array([
            (band(camb[kind][plus[i]])[1] - band(camb[kind][minus[i]])[1]) /
            (2 * steps[i] * mid) for i in range(len(names))])

    #the common-random-number check, per MODE rather than per band: with the same white
    #noise at every point, C_emp(+)/C_emp(-) = C_camb(+)/C_camb(-) exactly on every entry
    good = camb["unlensed"][centre] > 0
    crn_error = 0.0
    for i in range(len(names)):
        empirical = (merged["unlensed_plus"][i] - merged["unlensed_minus"][i])[good] / \
            merged["unlensed_fid"][good]
        model = (camb["unlensed"][plus[i]] - camb["unlensed"][minus[i]])[good] / \
            camb["unlensed"][centre][good]
        crn_error = max(crn_error, float(np.max(np.abs(empirical - model))) /
                        max(float(np.max(np.abs(model))), np.finfo(float).tiny))
    merged["crn_unlensed_error"] = crn_error

    if verbose:
        _report(merged, names)
    return merged


def _report(merged, names):
    ells = merged["band_ells"]
    print(f"\nCommon-random-number check: empirical vs CAMB unlensed dC/C per mode, max "
          f"relative error {merged['crn_unlensed_error']:.2e}"
          + ("  <-- NOT common random numbers?" if merged["crn_unlensed_error"] > 1e-6
             else " (exact, as it must be)"))
    residual = merged["band_c_delensed"] / merged["band_c_unlensed"]
    print(f"Residual lensing D(l) = C_delensed / C_unlensed at the centre: "
          f"{np.min(residual):.4f}..{np.max(residual):.4f} "
          f"(CAMB lensed / unlensed spans "
          f"{np.min(merged['band_c_camb_lensed'] / merged['band_c_camb_unlensed']):.4f}.."
          f"{np.max(merged['band_c_camb_lensed'] / merged['band_c_camb_unlensed']):.4f})")

    for i, name in enumerate(names):
        sigma = PARAM_SIGMA[name]
        print(f"\n  {name}: h = {merged['steps'][i]:.4g} ({merged['step_sigma']:g} sigma), "
              f"{merged['n_realizations']} realizations; sigma * dlnC/dtheta per band")
        print(f"  {'l':>7}{'delensed':>11}{'jackknife':>11}{'unl emp':>10}{'unl CAMB':>10}"
              f"{'len emp':>10}{'len CAMB':>10}{'2nd/1st':>9}")
        for b, ell in enumerate(ells):
            print(f"  {ell:7.0f}"
                  f"{merged['band_dlnc_delensed'][i, b] * sigma:11.4f}"
                  f"{merged['band_dlnc_delensed_error'][i, b] * sigma:11.4f}"
                  f"{merged['band_dlnc_unlensed'][i, b] * sigma:10.4f}"
                  f"{merged['band_dlnc_camb_unlensed'][i, b] * sigma:10.4f}"
                  f"{merged['band_dlnc_lensed'][i, b] * sigma:10.4f}"
                  f"{merged['band_dlnc_camb_lensed'][i, b] * sigma:10.4f}"
                  f"{merged['band_curvature_delensed'][i, b]:9.3f}")
        curvature = merged["band_curvature_delensed"][i]
        print(f"  second-order / first-order term at h: median {np.median(curvature):.3f}"
              + ("  <- h may be too wide for a central difference"
                 if np.median(curvature) > 0.1 else ""))
