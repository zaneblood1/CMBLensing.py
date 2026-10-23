"""The delensed TT covariance measured EMPIRICALLY at every finite-difference stencil point.

WHY THIS EXISTS. fisher_forecast's default delensed f block is CAMB's partially lensed
spectrum at a frozen Alens_L, so its value and its theta dependence are CAMB's. This module
replaces it: the block's value AND its derivative both come from what this codebase's
lense_flow actually produces (fisher_forecast --delensed_covariance).

WHAT IS MEASURED. For one seed (one set of common random numbers) and every point of the
central-difference stencil {theta_0, theta_0 +/- h_i e_i}, i over the sampled parameters:

    1. load_sim(theta) at the SAME seed. load_sim draws every field as white noise scaled by
       sqrt(C(theta)), so across the stencil the realizations differ ONLY through the
       cosmology - the Monte Carlo noise cancels in C(theta_0 + h) - C(theta_0 - h) instead
       of being amplified by 1 / (2h).
    2. f_delensed(theta) = lense_flow(L(phi) f [NOISELESS], W phi(theta), INVERSE), with the
       per-mode Wiener weight W = C_phi / (C_phi + N_phi) built ONCE at theta_0 and applied
       to the TRUE phi at every point. N_phi is held at theta_0 - it never depends on theta.
       It is either the box QE matrix (fisher_forecast.qe_reconstruction, with `qe_filter`
       and `qe_response` picking the forecast's qe_delensed / qe_unlensed bounds) or a
       per-mode rfft2 matrix stored under the key "nphi" in an npz (`phi_noise`).
    3. C_delensed(theta)[k] = |F[k]|^2 / nside^2, F = rfft2(f_delensed), on the rfft grid,
       with the [0, 0] origin set to zero.

The normalization in 3 is covar_matrix_from_cls's grid convention (C_l / pix_width^2), since
field_from_covar_single_key gives E|rfft2(f)|^2 = nside^2 * C_grid - so the averaged matrix
drops straight into covariance_blocks next to the analytic C_n and C_phi + N_phi. No Fourier
weight enters a PER-MODE covariance: w_k counts how many real degrees of freedom an rfft entry
carries (2 in the bulk, 1 on the two self-conjugate columns), which is how many independent
draws of that one variance the entry holds, not a rescaling of the variance itself. The
weights enter the Fisher contraction.

The NOISELESS lensed field is inverse-lensed, and the forecast adds the isotropic C_n back
on - covariance_blocks's "C_TT^delensed + C_n" convention.

The reconstruction noise of map_joint is NOT measured here; that is get_map_joint_phi_noise/.

The measurement is fanned out one slurm job per seed by
get_delensed_field_covariance/get_delensed_covariance.sh -> run_single_delensed_covariance.sh ->
run_single_delensed_covariance.py, merged locally by merge_delensed_covariance.py, and
consumed by fisher_forecast.py's --delensed_covariance.
"""

import glob
import os

import numpy as np

import jax.numpy as jnp

from cmb_lensing.constants import INVERSE_LENSE
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.fields import Basis, map as to_map, fourier
from cmb_lensing.lense_flow import lense_flow
from cmb_lensing.simulate import load_sim, covar_matrix_from_cls
from cmb_lensing.constants import PARAM_ORDER, PARAM_SIGMA, DEFAULT_ELL_CUTOFF
from cmb_lensing.fisher_forecast import (camb_cls_at_params, load_sim_cosmology,
                                         qe_reconstruction, check_box,
                                         DEFAULT_QE_RESPONSE, PARAM_RTOL)


#number of RK4 steps for BOTH the forward and the inverse lensing solve. load_sim lenses the
#data with n = 10, so the forward direction matches it by construction; the inverse must use
#the same n or the round trip is not the identity even at the true phi
LENSE_STEPS = 10

#default width of the |l| bands of band_edges (used by chain_analysis.py's spectra plot)
DEFAULT_DELTA_ELL = 100.0

#the displacement h_i of the stencil, in units of PARAM_SIGMA. Common random numbers keep the
#difference quiet at any h, so the only constraint is that C(theta) is still linear over
#+/- h. 0.5 is what every existing run used. The forecast takes its step from the merged
#file, whatever this is
DEFAULT_STEP_SIGMA = 0.5

#the TT spectrum in the box QE's filter when W's N_phi is the QE matrix
#(fisher_forecast.QE_FILTERS): "lensed" one QE pass on the lensed map, "delensed" iterated to
#the partially delensed fixed point (fisher_forecast's qe_delensed, the lower bound),
#"unlensed" the unlensed C_TT (fisher_forecast's qe_unlensed, the upper bound)
DEFAULT_QE_FILTER = "lensed"

#the key a stored N_phi file holds its per-mode rfft2 noise matrix under
STORED_NPHI_KEY = "nphi"

REALIZATION_GLOB = "delensed_covariance_*.npz"
MERGED_NAME = "delensed_covariance.npz"

def realization_file_name(realization_index):
    return f"delensed_covariance_{realization_index:04d}.npz"

def band_edges(ell_grid, delta_ell = DEFAULT_DELTA_ELL):
    """Uniform |l| band edges from the rfft grid's fundamental to its corner mode - outside
    that range the box holds no modes, so a band there would be empty."""
    ells = np.asarray(ell_grid)
    low = float(np.min(ells[ells > 0]))
    high = float(np.max(ells))
    n_band = max(1, int(np.ceil((high - low) / delta_ell)))
    return low + delta_ell * np.arange(n_band + 1)


def band_average(grid_values, ell_grid, weights, edges):
    """DOF-weighted (w_k) mean of an rfft-grid quantity inside each |l| band.

    Returns (centres, values, dof) over the NON-EMPTY bands only, with `centres` the
    DOF-weighted mean |l| in the band rather than the nominal bin midpoint.
    """
    ells = np.asarray(ell_grid)
    values = np.asarray(grid_values)
    dof = np.broadcast_to(np.real(np.asarray(weights)), ells.shape)

    centres, averages, dofs = [], [], []
    for b in range(len(edges) - 1):
        #the [0, 0] origin carries no |l| and is set to zero by every covar_matrix_from_cls call
        inside = (ells >= edges[b]) & (ells < edges[b + 1]) & (ells > 0) & np.isfinite(values)
        band_dof = np.sum(dof[inside])
        if band_dof > 0:
            centres.append(np.sum(dof[inside] * ells[inside]) / band_dof)
            averages.append(np.sum(dof[inside] * values[inside]) / band_dof)
            dofs.append(band_dof)
    return np.array(centres), np.array(averages), np.array(dofs)


#fields.map and fields.fourier apply irfft2 / rfft2 unconditionally rather than looking at the
#field's basis, and the fields here arrive in both (load_sim stores FOURIER, lense_flow returns
#MAP), so the basis is checked rather than assumed
def _to_fourier(field):
    if field.basis == Basis.FOURIER:
        return field
    return fourier(field)


def _to_map(field):
    if field.basis == Basis.MAP:
        return field
    return to_map(field)


def _scalar_matrix(field):
    """The FOURIER-basis rfft2 matrix of a FlatS0, whatever basis it arrives in."""
    return _to_fourier(field).scalar_matrix


def _lense(field, phi, direction):
    """lense_flow in the MAP basis, returned in FOURIER (T-only: no eb2qu round trip)."""
    lensed = lense_flow(_to_map(field), _to_map(phi), n = LENSE_STEPS, direction = direction, adjoint = False)
    return _to_fourier(lensed)


def stencil_points(param_ground, names, step_sigma = DEFAULT_STEP_SIGMA):
    """(offsets, points, steps): the 1 + 2k central-difference points in a fixed order.

    The centre comes first, then +h_i and -h_i for each name in `names` order. `offsets` is
    (1 + 2k, k) in units of h, `points` the matching full five-parameter dicts, `steps` the
    h_i = step_sigma * PARAM_SIGMA.
    """
    steps = np.array([step_sigma * PARAM_SIGMA[name] for name in names])
    offsets = [[0] * len(names)]
    points = [dict(param_ground)]
    for i, name in enumerate(names):
        for sign in (+1, -1):
            offset = [0] * len(names)
            offset[i] = sign
            point = dict(param_ground)
            point[name] = param_ground[name] + sign * steps[i]
            offsets.append(offset)
            points.append(point)
    return np.array(offsets, dtype = int), points, steps


def _stencil_indices(n_names):
    """(centre, plus, minus) point indices in stencil_points's order: the centre is 0, and
    parameter i sits at 1 + 2i (+h) and 2 + 2i (-h)."""
    plus = [1 + 2 * i for i in range(n_names)]
    minus = [2 + 2 * i for i in range(n_names)]
    return 0, plus, minus


def grid_covariance(field_fourier, nside):
    """|F|^2 / nside^2 on the rfft grid: one realization's per-mode covariance, in
    covar_matrix_from_cls's C_l / pix_width^2 units, with the [0, 0] origin zeroed as every
    covar_matrix_from_cls(..., origin_value = 0) block has it."""
    power = np.abs(np.asarray(field_fourier))**2 / nside**2
    power[0, 0] = 0.0
    return power


def load_stored_nphi(path, nside, theta_pix, noise_level, l_knee, param_ground):
    """The per-mode N_phi matrix stored under "nphi" in the npz at `path`.

    The file must be on THIS box (nside / theta_pix / noise_level / l_knee keys) at this
    fiducial cosmology (param_names / params keys), and `nphi` must be an
    (nside, nside // 2 + 1) rfft2 grid in covar_matrix_from_cls units. Returned as stored:
    NaN marks a mode the file has no estimate for.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no phi noise file at {path}")
    stored = np.load(path, allow_pickle = True)
    if STORED_NPHI_KEY not in stored.files:
        raise ValueError(f"{path} has no `{STORED_NPHI_KEY}` matrix")
    check_box(stored, nside, theta_pix, noise_level, l_knee, path = path, what = "phi noise")
    nphi = np.asarray(stored[STORED_NPHI_KEY], dtype = float)
    if nphi.shape != (nside, nside // 2 + 1):
        raise ValueError(f"{path}: `{STORED_NPHI_KEY}` has shape {nphi.shape}, expected the "
                         f"rfft grid {(nside, nside // 2 + 1)}")
    return nphi


def measure_delensed_covariance(nside, theta_pix, noise_level, param_ground, map_seed,
                                names, step_sigma = DEFAULT_STEP_SIGMA, l_knee = 0.0,
                                qe_filter = DEFAULT_QE_FILTER,
                                qe_response = DEFAULT_QE_RESPONSE,
                                phi_noise = None, verbose = True):
    """One seed through the whole stencil: simulate, delens by W phi_true, square.

    W's N_phi is the box QE matrix built with `qe_filter` and `qe_response`, unless
    `phi_noise` is the path of an npz holding a per-mode "nphi" matrix (load_stored_nphi),
    in which case that matrix is used and the two QE settings do not apply. Either way N_phi
    is built once at theta_0.

    `names` are the parameters to differentiate (any order; stored as given).

    Returns a dict with `delensed` of shape (1 + 2k, nside, nside // 2 + 1), plus the stencil (`offsets`, `point_params` in
    PARAM_ORDER, `steps`) and how W was built.
    """

    offsets, points, steps = stencil_points(param_ground, names, step_sigma)
    n_points = len(points)
    shape = (n_points, nside, nside // 2 + 1)

    #the per-mode Wiener weight W = C_phi / (C_phi + N_phi) at theta_0, held there for every
    #stencil point: CAMB's C_phi on the grid and N_phi from the QE or from a file, both in
    #covar_matrix_from_cls units
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls_fid = camb_cls_at_params(param_ground)
    phi_ells = jnp.arange(2, 2 + cls_fid["phi"].shape[0]).astype(jnp.float64)
    cphi_fid = np.asarray(covar_matrix_from_cls(nside, pix_width, ell_grid, phi_ells,
                                                cls_fid["phi"], origin_value = 0))
    if phi_noise:
        nphi_fid = load_stored_nphi(phi_noise, nside, theta_pix, noise_level, l_knee, param_ground)
        wiener_source = "file"
        #a mode the file has no estimate for gets N_phi = inf, i.e. W = 0: no delensing by
        #a mode with no estimate
        unmeasured = ~np.isfinite(nphi_fid) | (nphi_fid < 0)
        nphi_fid = np.where(unmeasured, np.inf, nphi_fid)
    else:
        #CAMB_LMAX, not the forecast's camb_lmax_for_grid: the W every existing file used
        nphi_fid, _ = qe_reconstruction(param_ground, nside, pix_width, ell_grid,
                                        noise_level, l_knee, 0.0, DEFAULT_ELL_CUTOFF, qe_filter,
                                        qe_response = qe_response, verbose = verbose)
        nphi_fid = np.asarray(nphi_fid)
        wiener_source = "qe"
    total = cphi_fid + nphi_fid
    wiener = np.zeros_like(cphi_fid)
    positive = total > 0
    wiener[positive] = cphi_fid[positive] / total[positive]

    result = {}
    result["offsets"] = offsets
    result["steps"] = steps
    result["point_params"] = np.array([[point[name] for name in PARAM_ORDER]
                                       for point in points])
    result["qe_filter"] = qe_filter
    result["qe_response"] = qe_response
    result["wiener_nphi_source"] = wiener_source
    result["wiener_nphi_file"] = str(phi_noise or "")

    result["delensed"] = np.full(shape, np.nan)

    for index in range(n_points):
        params = points[index]
        data_set = load_sim(nside, theta_pix, "I", map_seed, **load_sim_cosmology(params),
                            uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                            precomputed_cls = camb_cls_at_params(params))

        true_phi = _to_fourier(data_set.phi)
        wiener_phi = true_phi.replace(scalar_matrix = wiener * true_phi.scalar_matrix)
        delensed = _lense(data_set.lensed_field, wiener_phi, INVERSE_LENSE)

        result["delensed"][index] = grid_covariance(_scalar_matrix(delensed), nside)
    return result


# ── Collecting the per-realization files ──────────────────────────────────

def _read_config(data):
    """Everything that must agree between two job files before they may be averaged."""
    config = {}
    config["nside"] = int(data["nside"])
    config["theta_pix"] = float(data["theta_pix"])
    config["noise_level"] = float(data["noise_level"])
    config["l_knee"] = float(data["l_knee"])
    config["lense_steps"] = int(data["lense_steps"])
    config["step_sigma"] = float(data["step_sigma"])
    config["names"] = tuple(str(name) for name in data["names"])

    config["qe_filter"] = str(data["qe_filter"])
    config["qe_response"] = str(data["qe_response"])
    config["wiener_nphi_source"] = str(data["wiener_nphi_source"])
    return config


def load_covariance_directory(directory, verbose = True):
    """Stack every delensed_covariance_*.npz in `directory`.

    Refuses files that disagree on the box, W, the stencil (names, step, fiducial cosmology)
    or that repeat a seed.

    Returns (stacked, metadata): `stacked` is the delensed covariances, (n_realizations,
    1 + 2k, nside, nside // 2 + 1).
    """
    paths = sorted(glob.glob(os.path.join(directory, REALIZATION_GLOB)))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"no {REALIZATION_GLOB} in {directory}. Run "
            f"get_delensed_field_covariance/get_delensed_covariance.sh first and scp its "
            f"out_dir here.")

    config = None
    offsets = None
    point_params = None
    seeds = []
    stacked = []
    for path in paths:
        data = np.load(path, allow_pickle = True)
        current = _read_config(data)
        if config is None:
            config = current
            offsets = data["offsets"]
            point_params = data["point_params"]
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; an empirical covariance cannot mix configurations")
        elif not np.allclose(data["point_params"], point_params, rtol = PARAM_RTOL, atol = 0):
            raise ValueError(f"{path} sits on a different stencil (cosmologies) than earlier "
                             f"files; averaging them would blur exactly the theta "
                             f"dependence being measured. Use one out_dir per stencil.")
        stacked.append(np.asarray(data["delensed"]))
        seeds.append(int(data["map_seed"]))

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                         f"realizations would be double counted - check the seed_prefix "
                         f"loop in get_delensed_covariance.sh.")

    stacked = np.array(stacked)
    metadata = dict(config)
    metadata["offsets"] = offsets
    metadata["point_params"] = point_params
    metadata["seeds"] = seeds
    metadata["n_realizations"] = len(seeds)
    return stacked, metadata


def forecast_grids(mean, n_names):
    """The per-mode grids fisher_forecast reads, from the realization-MEAN delensed
    covariance (1 + 2k, nside, nside // 2 + 1): `delensed_fid` (the centre), `delensed_plus`
    and `delensed_minus` (stacked over the k parameters).

    The ONE place these grids are formed: merge_delensed_covariance calls it on the full
    mean, leave_one_out_grids on each delete-one mean.
    """
    centre, plus, minus = _stencil_indices(n_names)
    grids = {}
    grids["delensed_fid"] = mean[centre]
    grids["delensed_plus"] = mean[plus]
    grids["delensed_minus"] = mean[minus]
    return grids


def leave_one_out_grids(directory, verbose = True):
    """(seeds, generator): the forecast grids with each realization of `directory` left out.

    The generator yields forecast_grids(...) of the delete-one means, one realization at a
    time and in `seeds` order, as (total - row_i) / (n - 1).
    """
    stacked, metadata = load_covariance_directory(directory, verbose = verbose)
    n_real = metadata["n_realizations"]
    if n_real < 3:
        raise ValueError(f"{directory} holds {n_real} realization(s); a delete-one "
                         f"jackknife needs at least 3")
    n_names = len(metadata["names"])
    total = np.sum(stacked, axis = 0)

    def generate():
        for drop in range(n_real):
            mean = (total - stacked[drop]) / (n_real - 1)
            yield forecast_grids(mean, n_names)

    return list(metadata["seeds"]), generate()


def merge_delensed_covariance(directory, verbose = True):
    """Average the per-realization delensed matrices into the 1 + 2k empirical covariances:
    the realization MEAN at each stencil point, which is what the forecast differentiates.

    Returns the dict written to delensed_covariance.npz.
    """
    stacked, metadata = load_covariance_directory(directory, verbose = verbose)
    names = list(metadata["names"])
    point_params = metadata["point_params"]
    steps = np.array([metadata["step_sigma"] * PARAM_SIGMA[name] for name in names])
    centre, _, _ = _stencil_indices(len(names))
    merged = {}
    merged["names"] = np.array(names)
    merged["steps"] = steps
    merged["offsets"] = metadata["offsets"]
    merged["point_params"] = point_params
    merged["param_names"] = np.array(PARAM_ORDER)
    merged["params"] = point_params[centre]
    merged["n_realizations"] = metadata["n_realizations"]
    merged["seeds"] = np.array(metadata["seeds"])
    for key in ("nside", "theta_pix", "noise_level", "l_knee", "lense_steps", "step_sigma",
                "qe_filter", "qe_response", "wiener_nphi_source"):
        merged[key] = metadata[key]

    #the grids the forecast reads - through forecast_grids, as the jackknife does
    mean = np.mean(stacked, axis = 0)
    merged.update(forecast_grids(mean, len(names)))
    return merged
