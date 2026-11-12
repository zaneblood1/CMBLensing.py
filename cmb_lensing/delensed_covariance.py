"""The delensed field covariance measured EMPIRICALLY at every finite-difference stencil point.

WHY THIS EXISTS. The Fisher forecast's field blocks are the per-mode covariance of the
delensed field, and both their value AND their theta derivative come from what this
codebase's lense_flow actually produces (fisher_forecast's `delensed_covariance`).

WHAT IS MEASURED. For one seed (one set of common random numbers) and every point of the
central-difference stencil {theta_0, theta_0 +/- h_i e_i}, i over the sampled parameters:

    1. load_sim(theta) at the SAME seed and `pol`. load_sim draws every field as white noise
       scaled by sqrt(C(theta)), so across the stencil the realizations differ ONLY through
       the cosmology - the Monte Carlo noise cancels in C(theta_0 + h) - C(theta_0 - h)
       instead of being amplified by 1 / (2h).
    2. f_delensed(theta) = lense_flow(L(phi) f [NOISELESS], W phi(theta), INVERSE), with the
       per-mode Wiener weight W = C_phi / (C_phi + N_phi) built ONCE at theta_0 and applied
       to the TRUE phi at every point. N_phi is held at theta_0 - it never depends on theta.
       It is either the box QE N0 (camb_spectra.qe_noise_matrix with `qe_filter`) or a
       per-mode rfft2 matrix stored under the key "nphi" in an npz (`phi_noise`).
    3. For every component XY of the data's polarization (POL_COMPONENTS: TT for "I",
       EE and BB for "P", TT, TE, EE and BB for "IP"),
       C_delensed^XY(theta)[k] = Re(X[k] conj(Y[k])) / nside^2 on the rfft grid, with the
       [0, 0] origin set to zero.

The normalization in 3 is covar_matrix_from_cls's grid convention (C_l / pix_width^2), since
field_from_covar_single_key gives E|rfft2(f)|^2 = nside^2 * C_grid - so the averaged matrix
drops straight into the forecast's blocks next to the analytic C_n and C_phi + N_phi. No
Fourier weight enters a PER-MODE covariance: w_k counts how many real degrees of freedom an
rfft entry carries, not a rescaling of the variance itself. The weights enter the Fisher
contraction.

The NOISELESS lensed field is inverse-lensed, and the forecast adds the isotropic C_n back on.

The measurement is fanned out one slurm job per seed by
get_delensed_field_covariance/get_delensed_field_covariance.sh ->
run_single_delensed_field_covariance.sh -> run_single_delensed_field_covariance.py, merged
locally by merge_delensed_field_covariance.py, and consumed by fisher_forecast.py.
"""

import glob
import os

import numpy as np

from cmb_lensing.constants import (INVERSE_LENSE, PARAM_ORDER, PARAM_SIGMA, DEFAULT_ELL_CUTOFF,
                                   Basis)
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.fields import map as to_map, fourier, eb2qu, qu2eb
from cmb_lensing.lense_flow import lense_flow
from cmb_lensing.simulate import load_sim
from cmb_lensing.camb_spectra import (camb_cls_at_params, load_sim_cosmology, covar_from_cl,
                                      qe_noise_matrix, DEFAULT_QE_FILTER, DEFAULT_POL)

#number of RK4 steps for BOTH the forward and the inverse lensing solve. load_sim lenses the
#data with n = 10, so the forward direction matches it by construction; the inverse must use
#the same n or the round trip is not the identity even at the true phi
LENSE_STEPS = 10

#default width of the |l| bands of band_edges
DEFAULT_DELTA_ELL = 100.0

#the displacement h_i of the stencil, in units of PARAM_SIGMA (the driver's default too).
#Common random numbers keep the difference quiet at any h, so the constraint is that
#C(theta) is still linear over +/- h; too large a step under-reads dC/dtheta where the
#acoustic peaks shift (theta_MC_100 at high l first). 0.05 is accurate for the configuration
#tested (nside 128, theta_pix 2.5', 2.5 / 5 / 10 uK-arcmin). For any other nside, theta_pix
#or noise level CHECK IT BY HAND: rerun with the step halved and confirm the forecast does not
#change substantially. The forecast takes its step from the merged file, whatever this is
DEFAULT_STEP_SIGMA = 0.05

#the components stored for each polarization of the data, in this order
POL_COMPONENTS = {"I": ("TT",), "P": ("EE", "BB"), "IP": ("TT", "TE", "EE", "BB")}
#the field matrix holding each of T, E and B (FlatS0 / FlatS2 / FlatS02, EB parametrization)
FIELD_MATRIX_NAMES = {"T": "scalar_matrix", "E": "polar_matrix_1", "B": "polar_matrix_2"}

#the key a stored N_phi file holds its per-mode rfft2 noise matrix under
STORED_NPHI_KEY = "nphi"
#relative tolerance for "these two files sit on the same stencil": the parameter values
#round-trip through np.savez as float64
PARAM_RTOL = 1e-12

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


#fields.map and fields.fourier apply irfft2 / rfft2 unconditionally rather than looking at the
#field's basis, and the fields here arrive in both, so the basis is checked rather than assumed
def _to_fourier(field):
    if field.basis == Basis.FOURIER:
        return field
    return fourier(field)


def _to_map(field):
    if field.basis == Basis.MAP:
        return field
    return to_map(field)


def delens(lensed_field, phi):
    """The lensed field (T, EB or TEB, Fourier) inverse-lensed by phi, returned in Fourier.
    lense_flow acts on Q / U maps, so E / B are converted around it as mixing.unmix does."""
    lensed_map = _to_map(eb2qu(_to_fourier(lensed_field)))
    delensed = lense_flow(lensed_map, _to_map(phi), n = LENSE_STEPS, direction = INVERSE_LENSE,
                          adjoint = False)
    return qu2eb(_to_fourier(delensed))


def stencil_points(param_ground, names, step_sigma = DEFAULT_STEP_SIGMA):
    """(names, offsets, points, steps): the 1 + 2k central-difference points in a fixed order.

    `names` is returned sorted into PARAM_ORDER. The centre comes first, then +h_i and -h_i
    for each name. `offsets` is (1 + 2k, k) in units of h, `points` the matching full
    five-parameter dicts, `steps` the h_i = step_sigma * PARAM_SIGMA.
    """
    names = [name for name in PARAM_ORDER if name in names]
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
    return names, np.array(offsets, dtype = int), points, steps


def _stencil_indices(n_names):
    """(centre, plus, minus) point indices in stencil_points's order: the centre is 0, and
    parameter i sits at 1 + 2i (+h) and 2 + 2i (-h)."""
    plus = [1 + 2 * i for i in range(n_names)]
    minus = [2 + 2 * i for i in range(n_names)]
    return 0, plus, minus


def grid_covariance(first_fourier, second_fourier, nside):
    """Re(X conj(Y)) / nside^2 on the rfft grid: one realization's per-mode covariance of two
    field components, in covar_matrix_from_cls's C_l / pix_width^2 units, with the [0, 0]
    origin zeroed as every covar_matrix_from_cls(..., origin_value = 0) block has it."""
    first = np.asarray(first_fourier)
    if second_fourier is first_fourier:
        power = np.abs(first)**2 / nside**2
    else:
        power = np.real(first * np.conj(np.asarray(second_fourier))) / nside**2
    power[0, 0] = 0.0
    return power


def component_covariances(field_fourier, nside, pol):
    """The per-mode covariance of every POL_COMPONENTS[pol] pair of one Fourier field,
    stacked: (n_components, nside, nside // 2 + 1)."""
    matrices = {letter: getattr(field_fourier, name, None)
                for letter, name in FIELD_MATRIX_NAMES.items()}
    return np.array([grid_covariance(matrices[component[0]], matrices[component[1]], nside)
                     for component in POL_COMPONENTS[pol]])


def check_box(merged, nside, theta_pix, noise_level, l_knee, path = "", what = "file"):
    """Refuse a measurement taken on a different box (nside / theta_pix / noise / l_knee)."""
    measured = dict(nside = int(merged["nside"]), theta_pix = float(merged["theta_pix"]),
                    noise_level = float(merged["noise_level"]),
                    l_knee = float(merged["l_knee"]))
    wanted = dict(nside = int(nside), theta_pix = float(theta_pix),
                  noise_level = float(noise_level), l_knee = float(l_knee))
    if measured != wanted:
        raise ValueError(f"the {what} {path} was measured at {measured} but this run is at "
                         f"{wanted}; re-measure it at this box")


def load_stored_nphi(path, nside, theta_pix, noise_level, l_knee):
    """The per-mode N_phi matrix stored under "nphi" in the npz at `path`.

    The file must be on THIS box (nside / theta_pix / noise_level / l_knee keys), and `nphi`
    must be an (nside, nside // 2 + 1) rfft2 grid in covar_matrix_from_cls units. Returned as
    stored: NaN marks a mode the file has no estimate for.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no phi noise file at {path}")
    stored = np.load(path)
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
                                qe_filter = DEFAULT_QE_FILTER, phi_noise = None,
                                pol = DEFAULT_POL):
    """One seed through the whole stencil: simulate, delens by W phi_true, square.

    W's N_phi is the box QE N0 for `pol` built with `qe_filter` ("unlensed" or "lensed"
    spectra in its filter), unless `phi_noise` is the path of an npz holding a per-mode "nphi"
    matrix (load_stored_nphi), in which case that matrix is used. Either way N_phi is built
    once at theta_0.

    `names` are the parameters to differentiate (stored sorted into PARAM_ORDER).

    Returns a dict with `delensed` of shape (1 + 2k, n_components, nside, nside // 2 + 1),
    its `components` (POL_COMPONENTS[pol]), the stencil (`names`, `offsets`, `point_params`
    in PARAM_ORDER, `steps`) and how W was built.
    """
    names, offsets, points, steps = stencil_points(param_ground, names, step_sigma)

    #the per-mode Wiener weight W = C_phi / (C_phi + N_phi) at theta_0, held there for every
    #stencil point: CAMB's C_phi on the grid and N_phi from the QE or from a file, both in
    #covar_matrix_from_cls units
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls_fid = camb_cls_at_params(param_ground)
    cphi_fid = np.asarray(covar_from_cl(nside, pix_width, ell_grid, cls_fid["phi"]))
    if phi_noise:
        nphi_fid = load_stored_nphi(phi_noise, nside, theta_pix, noise_level, l_knee)
        wiener_source = "file"
        #a mode the file has no estimate for gets N_phi = inf, i.e. W = 0: no delensing by
        #a mode with no estimate
        unmeasured = ~np.isfinite(nphi_fid) | (nphi_fid < 0)
        nphi_fid = np.where(unmeasured, np.inf, nphi_fid)
    else:
        nphi_fid = np.asarray(qe_noise_matrix(cls_fid, nside, pix_width, ell_grid, noise_level,
                                              l_knee, 0.0, DEFAULT_ELL_CUTOFF, pol = pol,
                                              qe_filter = qe_filter))
        wiener_source = "qe"
    total = cphi_fid + nphi_fid
    wiener = np.zeros_like(cphi_fid)
    positive = total > 0
    wiener[positive] = cphi_fid[positive] / total[positive]

    result = {}
    result["names"] = names
    result["offsets"] = offsets
    result["steps"] = steps
    result["point_params"] = np.array([[point[name] for name in PARAM_ORDER]
                                       for point in points])
    result["pol"] = pol
    result["components"] = POL_COMPONENTS[pol]
    result["qe_filter"] = qe_filter
    result["wiener_nphi_source"] = wiener_source
    result["wiener_nphi_file"] = str(phi_noise or "")

    delensed_covariances = []
    for params in points:
        data_set = load_sim(nside, theta_pix, pol, map_seed, **load_sim_cosmology(params),
                            uk_arcmin_t = noise_level, r = 0, nt = 0, l_knee = l_knee,
                            precomputed_cls = camb_cls_at_params(params))
        true_phi = _to_fourier(data_set.phi)
        wiener_phi = true_phi.replace(scalar_matrix = wiener * true_phi.scalar_matrix)
        delensed = delens(data_set.lensed_field, wiener_phi)
        delensed_covariances.append(component_covariances(delensed, nside, pol))
    result["delensed"] = np.array(delensed_covariances)
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
    config["pol"] = str(data["pol"])
    config["components"] = tuple(str(component) for component in data["components"])
    config["qe_filter"] = str(data["qe_filter"])
    config["wiener_nphi_source"] = str(data["wiener_nphi_source"])
    return config


def load_covariance_directory(directory):
    """Stack every delensed_covariance_*.npz in `directory`.

    Refuses files that disagree on the box, pol, W, the stencil (names, step, fiducial
    cosmology) or that repeat a seed.

    Returns (stacked, metadata): `stacked` is the delensed covariances, (n_realizations,
    1 + 2k, n_components, nside, nside // 2 + 1).
    """
    paths = sorted(glob.glob(os.path.join(directory, REALIZATION_GLOB)))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"no {REALIZATION_GLOB} in {directory}. Run "
            f"get_delensed_field_covariance/get_delensed_field_covariance.sh first and scp "
            f"its out_dir here.")

    config = None
    metadata = {}
    seeds = []
    stacked = []
    for path in paths:
        data = np.load(path)
        current = _read_config(data)
        if config is None:
            config = current
            metadata = {key: data[key] for key in ("offsets", "point_params", "steps")}
        elif current != config:
            raise ValueError(f"{path} was produced at {current} but earlier files used "
                             f"{config}; an empirical covariance cannot mix configurations")
        elif not np.allclose(data["point_params"], metadata["point_params"],
                             rtol = PARAM_RTOL, atol = 0):
            raise ValueError(f"{path} sits on a different stencil (cosmologies) than earlier "
                             f"files; averaging them would blur exactly the theta "
                             f"dependence being measured. Use one out_dir per stencil.")
        stacked.append(np.asarray(data["delensed"]))
        seeds.append(int(data["map_seed"]))

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory} contains duplicate map_seed values, so some "
                         f"realizations would be double counted - check the seed_prefix "
                         f"loop in get_delensed_field_covariance.sh.")

    metadata.update(config)
    metadata["seeds"] = seeds
    metadata["n_realizations"] = len(seeds)
    return np.array(stacked), metadata


def merge_delensed_covariance(directory):
    """Average the per-realization delensed covariances into the 1 + 2k empirical ones: the
    realization MEAN at each stencil point, which is what the forecast differentiates.

    Returns the dict written to delensed_covariance.npz: `delensed_fid` (n_components, nside,
    nside // 2 + 1) at theta_0, `delensed_plus` / `delensed_minus` (k, n_components, ...) at
    theta_0 +/- h_i, with `components`, `pol`, the stencil and the box.
    """
    stacked, metadata = load_covariance_directory(directory)
    names = list(metadata["names"])
    centre, plus, minus = _stencil_indices(len(names))
    mean = np.mean(stacked, axis = 0)

    merged = {}
    merged["names"] = np.array(names)
    merged["steps"] = metadata["steps"]
    merged["offsets"] = metadata["offsets"]
    merged["point_params"] = metadata["point_params"]
    merged["param_names"] = np.array(PARAM_ORDER)
    merged["params"] = metadata["point_params"][centre]
    merged["n_realizations"] = metadata["n_realizations"]
    merged["seeds"] = np.array(metadata["seeds"])
    merged["components"] = np.array(metadata["components"])
    for key in ("nside", "theta_pix", "noise_level", "l_knee", "lense_steps", "step_sigma",
                "pol", "qe_filter", "wiener_nphi_source"):
        merged[key] = metadata[key]
    merged["delensed_fid"] = mean[centre]
    merged["delensed_plus"] = mean[plus]
    merged["delensed_minus"] = mean[minus]
    return merged


def load_merged_covariance(path):
    """A merged delensed_covariance.npz as the forecast reads it: `names` (list), `steps`,
    `components` (list) and the `delensed_fid` (n_components, nside, nside // 2 + 1) /
    `delensed_plus` / `delensed_minus` (k, n_components, ...) grids.

    A file with no `components` is a temperature-only one written before 2026-11-11, whose
    grids have no component axis.
    """
    data = np.load(path)
    merged = {"names": [str(name) for name in data["names"]], "steps": data["steps"]}
    grids = {key: data[key] for key in ("delensed_fid", "delensed_plus", "delensed_minus")}
    if "components" in data.files:
        merged["components"] = [str(component) for component in data["components"]]
    else:
        merged["components"] = list(POL_COMPONENTS["I"])
        grids["delensed_fid"] = grids["delensed_fid"][None]
        grids["delensed_plus"] = grids["delensed_plus"][:, None]
        grids["delensed_minus"] = grids["delensed_minus"][:, None]
    merged.update(grids)
    return merged
