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
Under "shifted", `constant_nphi` (default True, the forecast's own default) still holds the QE
norm N_phi at theta_0 while C_f, C_phi and D move; False rebuilds it too (what every "shifted"
run before the flag existed did). See DEFAULT_CONSTANT_NPHI.

THE EMPIRICAL PHI NOISE (the same jobs, no extra map_joint calls). At every stencil point the
job also stores three per-mode moments of the reconstruction against the truth,
A = |phi_hat|^2, B = Re(phi_hat conj(phi)), D = |phi|^2 (PHI_MOMENTS). Writing
phi_hat = rho phi + n with n uncorrelated with phi, the phi block C_phi + N needs the noise
of the response-deconvolved estimate phi_hat / rho, i.e. N_eff = C_phi (1/r^2 - 1) with
r^2 = <B>^2 / (<A><D>) - phi_noise.py's estimator, but per rfft MODE instead of per |L|
annulus (the square box's reconstruction noise is anisotropic) and at every stencil point
(so its theta dependence is measured, on the same common random numbers). The squared
difference |phi_hat - phi|^2 = A - 2B + D is NOT that N: it is (1 - rho)^2 C + Var(n), which
for a Wiener-like MAP is C N / (C + N) - the leftover lensing that sets the DELENSED block,
bounded by C even for a mode the data say nothing about. Ratios are formed from realization
MEANS, as in phi_noise.py. Modes with <B> <= 0 carry no measurement and are flagged, not
given an N. r^2 and N_eff are stored per mode at every stencil point, so the forecast can
hold N_eff at theta_0 or use each point's own.

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
from cmb_lensing.fisher_forecast import (camb_cls_at_params, load_sim_cosmology,
                                         qe_noise_matrix)
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

#whether map_joint's N_phi (the QE norm, data_set.quadratic_estimate) is held at theta_0 when
#reconstruction = "shifted" - the measurement-side twin of fisher_forecast's constant_nphi, so
#the empirical f block and the forecast's phi block can be run on the same convention. In
#map_joint N_phi only enters the phi step's preconditioner pinv(C_phi^-1 + N_phi^-1), so it
#moves phi_hat only through incomplete convergence of the map_joint_steps iterations - the
#MAP optimum itself does not depend on it. True matches the forecast's default
DEFAULT_CONSTANT_NPHI = True

#the three fields step 4 is applied to; "delensed" is the product, the other two validate
FIELD_KINDS = ("unlensed", "lensed", "delensed")

#the per-mode second moments of the reconstruction against the truth, stored at every
#stencil point: A = |phi_hat|^2, B = Re(phi_hat conj(phi)), D = |phi|^2 (same units as the
#covariances). The merge forms r^2 = <B>^2 / (<A> <D>) PER MODE from the realization means,
#and the phi block C_phi / r^2 = C_phi + N_eff - see the module docstring for why the squared
#difference |phi_hat - phi|^2 = A - 2B + D is NOT the N_phi that block needs
PHI_MOMENTS = ("phi_auto", "phi_cross", "phi_true")

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


def grid_cross(first_fourier, second_fourier, nside):
    """Re(a conj(b)) / nside^2 on the rfft grid: one realization's per-mode (cross) covariance,
    in covar_matrix_from_cls's C_l / pix_width^2 units, with the [0, 0] origin zeroed as every
    covar_matrix_from_cls(..., origin_value = 0) block has it. Pass one field twice for an
    auto-covariance."""
    power = jnp.real(first_fourier * jnp.conj(second_fourier)) / nside**2
    return np.asarray(power.at[0, 0].set(0.0))


def grid_covariance(field_fourier, nside):
    """F conj(F) / nside^2 on the rfft grid - grid_cross of a field with itself."""
    return grid_cross(field_fourier, field_fourier, nside)


def measure_delensed_covariance(nside, theta_pix, noise_level, param_ground, map_seed,
                                names, step_sigma = DEFAULT_STEP_SIGMA, l_knee = 0.0,
                                map_joint_steps = 30,
                                reconstruction = DEFAULT_RECONSTRUCTION,
                                constant_nphi = DEFAULT_CONSTANT_NPHI,
                                on_point = None, verbose = True):
    """One seed through the whole stencil: simulate, reconstruct, delens, square.

    `names` are the parameters to differentiate (any order; stored as given). `constant_nphi`
    only matters for reconstruction = "shifted" ("fiducial" freezes N_phi along with
    everything else): True keeps map_joint's QE norm at theta_0 at every point, False rebuilds
    it at each point's cosmology. See DEFAULT_CONSTANT_NPHI. `on_point`,
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
                  **{kind: np.full(shape, np.nan) for kind in FIELD_KINDS + PHI_MOMENTS})

    def simulate(params):
        return load_sim(nside, theta_pix, "I", map_seed, **load_sim_cosmology(params),
                        uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                        precomputed_cls = camb_cls_at_params(params))

    if verbose:
        print(f"seed {map_seed}: nside {nside}, {theta_pix:g}', {noise_level:g} uK-arcmin, "
              f"l_knee {l_knee:g}; {len(points)} stencil points over {list(names)} at "
              f"+/- {step_sigma:g} sigma, reconstruction at the {reconstruction} cosmology"
              + (f", N_phi {'frozen at theta_0' if constant_nphi else 'rebuilt per point'}"
                 if reconstruction == "shifted" else ""))

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
        elif constant_nphi:
            #"shifted" with N_phi frozen: C_f, C_phi and D follow this point's cosmology but
            #the QE norm stays at theta_0, mirroring the forecast's constant_nphi
            data_set = data_set.replace(quadratic_estimate = fiducial_set.quadratic_estimate)

        _, phi_hat = map_joint(data_set, num_steps = map_joint_steps)
        fields = dict(unlensed = data_set.unlensed_field, lensed = data_set.lensed_field,
                      delensed = _lense(data_set.lensed_field, phi_hat, INVERSE_LENSE))
        for kind in FIELD_KINDS:
            result[kind][index] = grid_covariance(_scalar_matrix(fields[kind]), nside)

        #the reconstruction against the truth, per mode - the empirical phi noise at this
        #point. Common random numbers make phi_true exactly sqrt(C_phi(theta)) times the same
        #white noise at every point, which the merge checks
        hat, true = _scalar_matrix(phi_hat), _scalar_matrix(data_set.phi)
        result["phi_auto"][index] = grid_cross(hat, hat, nside)
        result["phi_cross"][index] = grid_cross(hat, true, nside)
        result["phi_true"][index] = grid_cross(true, true, nside)

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
    config["constant_nphi"] = effective_constant_nphi(
        config["reconstruction"],
        bool(data["constant_nphi"]) if "constant_nphi" in data.files else None)
    return config


def effective_constant_nphi(reconstruction, constant_nphi):
    """Whether N_phi really was held at theta_0. A "fiducial" reconstruction freezes it along
    with everything else whatever the flag says; files written before the flag existed
    (constant_nphi None) rebuilt it under "shifted", so that is what they are read as."""
    if reconstruction == "fiducial":
        return True
    return False if constant_nphi is None else bool(constant_nphi)


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
        #files written before the phi moments existed have none; a directory may not mix the
        #two, or the phi average would silently cover only some of the realizations
        current["has_phi_moments"] = all(moment in data.files for moment in PHI_MOMENTS)
        if config is None:
            config, offsets, point_params = current, data["offsets"], data["point_params"]
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; an empirical covariance cannot mix configurations")
        elif not np.allclose(data["point_params"], point_params, rtol = PARAM_RTOL, atol = 0):
            raise ValueError(f"{path} sits on a different stencil (cosmologies) than earlier "
                             f"files; averaging them would blur exactly the theta "
                             f"dependence being measured. Use one out_dir per stencil.")
        for kind in FIELD_KINDS + (PHI_MOMENTS if current["has_phi_moments"] else ()):
            stacked.setdefault(kind, []).append(np.asarray(data[kind]))
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
              f"{config['reconstruction']} cosmology, N_phi "
              f"{'frozen at theta_0' if config['constant_nphi'] else 'rebuilt per point'}")
        print(f"  stencil: {list(config['names'])} at +/- {config['step_sigma']:g} sigma")
        if not config["has_phi_moments"]:
            print(f"  no per-mode phi moments (files predate them): no empirical phi noise")
        print(f"  worst inverse-lensing round trip: "
              f"{np.max(stacked['inverse_error']):.3e}")
    return stacked, metadata


def _point_index(offsets, i, sign):
    target = np.zeros(offsets.shape[1], dtype = int)
    target[i] = sign
    return int(np.flatnonzero(np.all(offsets == target, axis = 1))[0])


def _camb_grid_covariances(point_params, nside, pix_width, ell_grid):
    """CAMB's unlensed and lensed TT, and C_phi, on the grid at every stencil point, in the
    same units as the measured covariances."""
    result = {"unlensed": [], "lensed": [], "phi": []}
    for row in point_params:
        cls = camb_cls_at_params({name: float(row[i]) for i, name in enumerate(PARAM_ORDER)})
        for kind, source in (("unlensed", "scalar_TT"), ("lensed", "total_TT"),
                             ("phi", "phi")):
            ells = jnp.arange(2, 2 + cls[source].shape[0]).astype(jnp.float64)
            result[kind].append(np.asarray(covar_matrix_from_cls(
                nside, pix_width, ell_grid, ells, cls[source], origin_value = 0)))
    return {kind: np.array(rows) for kind, rows in result.items()}


def _band_index(ell_grid, edges):
    """Each rfft entry's |L| band (-1 outside the edges or at the origin)."""
    ells = np.asarray(ell_grid)
    index = np.digitize(ells, edges) - 1
    return np.where((ells > 0) & (index < len(edges) - 1), index, -1)


def _band_smooth(matrix, index, weights):
    """Replace every entry by the DOF-weighted mean of its |L| band (entries outside every
    band are left alone). The isotropic counterpart of a per-mode moment."""
    matrix = np.asarray(matrix)
    dof = np.asarray(weights)
    inside = index >= 0
    n_band = int(np.max(index)) + 1
    total = np.bincount(index[inside], weights = (dof * matrix)[inside], minlength = n_band)
    norm = np.bincount(index[inside], weights = dof[inside], minlength = n_band)
    smoothed = matrix.copy()
    smoothed[inside] = (total / np.where(norm > 0, norm, 1.0))[index[inside]]
    return smoothed


def phi_r_squared(auto, cross, true):
    """(r^2, measured) per entry from per-realization moment ROWS (first axis = realization):
    r^2 = <B>^2 / (<A><D>) from the realization means, as phi_noise.derived_spectra forms it.

    `measured` is False where the mean cross power is not positive (squaring B would give a
    healthy-looking r^2 to a mode uncorrelated or ANTI-correlated with the truth -
    phi_noise.derived_spectra's rule) and at the zeroed origin. Such a mode has no measured
    noise; the forecast gives it no phi information rather than an invented N.
    """
    mean_a, mean_b, mean_d = (np.mean(np.asarray(rows, dtype = float), axis = 0)
                              for rows in (auto, cross, true))
    with np.errstate(divide = "ignore", invalid = "ignore"):
        r_squared = mean_b**2 / (mean_a * mean_d)
    measured = (mean_b > 0) & np.isfinite(r_squared) & (r_squared > 0)
    return np.where(measured, r_squared, np.nan), measured


def _merge_phi(stacked, metadata, merged, camb, centre, plus, minus, steps, band,
               band_count, ell_grid, weights, edges, pix_width, smooth_delta_ell):
    """The empirical phi-noise half of the merge; adds its keys to `merged` in place.

    Per mode and per stencil point: r^2 (phi_r_squared) from the realization means of the
    moments (optionally smoothed in |L| annuli first) and N_eff = C_phi (1/r^2 - 1), with
    C_phi CAMB's at that point. The forecast reads `phi_r2_*` / `phi_measured_*` at the centre
    (frozen N_eff) or at every stencil point (theta-dependent N_eff). The band quantities
    below are printed / plotted diagnostics only.
    """
    nside = metadata["nside"]
    n_real = len(stacked["phi_auto"])

    #common random numbers for phi: <D>(+) / <D>(-) = C_phi(+) / C_phi(-) exactly, per mode.
    #Checked on the RAW means - smoothing mixes modes, so it would fail by construction
    raw_true = np.mean(stacked["phi_true"], axis = 0)
    good = camb["phi"][centre] > 0
    crn_error = 0.0
    for i in range(len(plus)):
        empirical = (raw_true[plus[i]] - raw_true[minus[i]])[good] / raw_true[centre][good]
        model = (camb["phi"][plus[i]] - camb["phi"][minus[i]])[good] / camb["phi"][centre][good]
        crn_error = max(crn_error, float(np.max(np.abs(empirical - model))) /
                        max(float(np.max(np.abs(model))), np.finfo(float).tiny))
    merged["phi_crn_error"] = crn_error

    rows = {moment: np.asarray(stacked[moment]) for moment in PHI_MOMENTS}
    if smooth_delta_ell > 0:
        smooth_index = _band_index(ell_grid, band_edges(ell_grid, smooth_delta_ell))
        rows = {moment: np.array([[_band_smooth(point, smooth_index, weights)
                                   for point in realization] for realization in value])
                for moment, value in rows.items()}

    r_squared, measured = phi_r_squared(rows["phi_auto"], rows["phi_cross"],
                                        rows["phi_true"])
    with np.errstate(invalid = "ignore", divide = "ignore"):
        noise = np.where(measured, camb["phi"] * (1.0 / r_squared - 1.0), np.nan)

    for key, value in (("phi_r2", r_squared), ("phi_measured", measured),
                       ("phi_noise", noise)):
        merged[f"{key}_fid"] = value[centre]
        merged[f"{key}_plus"] = value[plus]
        merged[f"{key}_minus"] = value[minus]
    merged["phi_smooth_delta_ell"] = float(smooth_delta_ell)
    merged["has_phi_moments"] = True

    #band N_eff from per-realization band sums (always of the RAW moments), jackknifed
    def band_sums(values):
        return np.array([[band(point)[1] for point in realization] for realization in values])

    sums = {moment: band_sums(stacked[moment]) for moment in PHI_MOMENTS}
    cphi_band = np.array([band(point)[1] for point in camb["phi"]])

    def band_noise(kept):
        r2, ok = phi_r_squared(*(sums[m][kept] for m in PHI_MOMENTS))
        with np.errstate(invalid = "ignore", divide = "ignore"):
            return np.where(ok, cphi_band * (1.0 / r2 - 1.0), np.nan)

    def log_derivatives(band_n):
        return np.array([(band_n[plus[i]] - band_n[minus[i]]) / (2 * steps[i] * band_n[centre])
                         for i in range(len(plus))])

    everything = np.ones(n_real, dtype = bool)
    central = band_noise(everything)
    derivative = log_derivatives(central)
    if n_real >= 3:
        leave_one_out = [band_noise(np.arange(n_real) != drop) for drop in range(n_real)]
        centres = np.array([entry[centre] for entry in leave_one_out])
        derivs = np.array([log_derivatives(entry) for entry in leave_one_out])
        factor = (n_real - 1) / n_real
        centre_error = np.sqrt(factor * np.sum((centres - centres.mean(axis = 0))**2, axis = 0))
        deriv_error = np.sqrt(factor * np.sum((derivs - derivs.mean(axis = 0))**2, axis = 0))
    else:
        centre_error = np.full(band_count, np.nan)
        deriv_error = np.full((len(plus), band_count), np.nan)

    #band_average reports the NON-EMPTY bands only, so the scatter below walks that subset
    index = _band_index(ell_grid, edges)
    filled = np.flatnonzero(np.bincount(index[index >= 0], minlength = len(edges) - 1) > 0)
    assert len(filled) == band_count

    #the box's quadratic-estimator N^(0) at the centre, on the same grid and bands - the
    #reference the measurement is judged against (fisher_forecast.qe_noise_matrix, the
    #physical N0 with no NPHI_FAC)
    cls_fid = camb_cls_at_params({name: float(value) for name, value in
                                  zip(PARAM_ORDER, metadata["point_params"][centre])})
    qe_grid = np.asarray(qe_noise_matrix(cls_fid, nside, pix_width, ell_grid,
                                         metadata["noise_level"], metadata["l_knee"], 0.0,
                                         10_000))
    merged["phi_noise_qe_fid"] = qe_grid

    #within-annulus anisotropy: DOF-weighted std of ln N inside each band, measured and QE
    dof = np.asarray(weights)

    def log_scatter(matrix):
        values = np.full(band_count, np.nan)
        for place, b in enumerate(filled):
            inside = (index == b) & np.isfinite(matrix) & (matrix > 0)
            if np.sum(inside) > 1:
                logs = np.log(matrix[inside])
                w = dof[inside]
                mean = np.sum(w * logs) / np.sum(w)
                values[place] = np.sqrt(np.sum(w * (logs - mean)**2) / np.sum(w))
        return values

    merged.update(band_phi_noise = central[centre], band_phi_noise_error = centre_error,
                  band_phi_noise_qe = band(qe_grid)[1], band_c_camb_phi = cphi_band[centre],
                  band_dlnn_phi = derivative, band_dlnn_phi_error = deriv_error,
                  band_phi_scatter = log_scatter(noise[centre]),
                  band_phi_scatter_qe = log_scatter(qe_grid),
                  #over the modes that exist, i.e. without the origin
                  phi_measured_fraction = float(np.sum((dof * measured[centre])[index >= 0]) /
                                                np.sum(dof[index >= 0])))


def merge_delensed_covariance(directory, delta_ell = DEFAULT_DELTA_ELL, smooth_delta_ell = 0.0,
                              verbose = True):
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

    When the files carry the phi moments (PHI_MOMENTS), also the empirical phi noise
    (_merge_phi): per-mode r^2 and N_eff at every stencil point, which fisher_forecast's
    --empirical_phi_noise contracts. `smooth_delta_ell` > 0 averages the three moments in
    |L| annuli of that width before forming r^2 (isotropic, quieter); 0 keeps every mode.

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
                  constant_nphi = metadata["constant_nphi"],
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

    merged["has_phi_moments"] = bool(metadata["has_phi_moments"])
    if metadata["has_phi_moments"]:
        _merge_phi(stacked, metadata, merged, camb, centre, plus, minus, steps, band,
                   len(band_ells), ell_grid, weights, edges, pix_width, smooth_delta_ell)

    if verbose:
        _report(merged, names)
        if merged["has_phi_moments"]:
            _report_phi(merged, names)
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


def _report_phi(merged, names):
    ells = merged["band_ells"]
    smoothing = merged["phi_smooth_delta_ell"]
    print(f"\nEmpirical phi noise, per mode"
          + (f" (moments smoothed in |L| annuli of {smoothing:g})" if smoothing > 0 else "")
          + f": {merged['phi_measured_fraction']:.1%} of the DOF measured at the centre "
          f"(the rest have <Re phi_hat phi*> <= 0 and carry no phi information)")
    print(f"Common-random-number check: empirical vs CAMB C_phi, dC/C per mode, max relative "
          f"error {merged['phi_crn_error']:.2e}"
          + ("  <-- NOT common random numbers?" if merged["phi_crn_error"] > 1e-6
             else " (exact, as it must be)"))
    print(f"  {'L':>7}{'N_eff/N_QE':>12}{'+/-':>9}{'N_eff/C':>10}"
          f"{'ln-scatter':>12}{'QE scatter':>12}"
          + "".join(f"{'s dlnN/d' + name[:6]:>16}" for name in names))
    for b, ell in enumerate(ells):
        qe = merged["band_phi_noise_qe"][b]
        row = (f"  {ell:7.0f}{merged['band_phi_noise'][b] / qe:12.4f}"
               f"{merged['band_phi_noise_error'][b] / qe:9.4f}"
               f"{merged['band_phi_noise'][b] / merged['band_c_camb_phi'][b]:10.3g}"
               f"{merged['band_phi_scatter'][b]:12.3f}{merged['band_phi_scatter_qe'][b]:12.3f}")
        for i, name in enumerate(names):
            sigma = PARAM_SIGMA[name]
            row += (f"{merged['band_dlnn_phi'][i, b] * sigma:9.4f}"
                    f"+/-{merged['band_dlnn_phi_error'][i, b] * sigma:<5.3f}")
        print(row)
    print("  N_eff/N_QE < 1: map_joint beats the box's quadratic estimator in that band. "
          "ln-scatter: DOF-weighted std of ln N_eff inside the annulus (anisotropy plus MC "
          "noise), next to the QE matrix's own. s dlnN/dtheta: how far N_eff moves per "
          "sigma, the theta dependence --vary_nphi contracts")
