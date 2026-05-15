import jax.numpy as jnp
import os
from cmb_lensing.constants import *
from cmb_lensing.fields import *
from cmb_lensing.matrix_operators import *
from cmb_lensing.lense_flow import *
from cmb_lensing.util import *
from cmb_lensing.statistics import *
from cmb_lensing.dataset import *
from cmb_lensing.gradients import *
from cmb_lensing.wiener_filter import *
from cmb_lensing.map_joint import *
from cmb_lensing.simulate import *

#constants relating to the ground truth data
GROUND_TRUTH = os.getcwd() + "/tests/ground_truth_data/"
FIGURE_PATH = os.getcwd() + "/tests/test_generated_figures/"
with open(GROUND_TRUTH + "n_side.txt", "r") as file:
    N_SIDE = int(file.read().strip())
with open(GROUND_TRUTH + "theta_pix.txt", "r") as file:
    THETA_PIX = float(file.read().strip())
PIX_WIDTH = float(jnp.deg2rad(THETA_PIX / ARCMIN_PER_DEGREE))

#threshold values for unit tests
MAX_NORM_DIFF = 1
MIN_AVG_CORRELATION = 0

#noise covariance, beam, mask, field covariance, quadratic estimate

_DATASET_CACHE = {}

def _get_dataset(pol):
    if pol not in _DATASET_CACHE:
        _DATASET_CACHE[pol] = load_sim(N_SIDE, THETA_PIX, pol, master_seed=1)
    return _DATASET_CACHE[pol]

def _plot_matrix_comparison(ground, predict, title, folder):
    os.makedirs(folder, exist_ok=True)
    ground_abs = jnp.abs(ground)
    predict_abs = jnp.abs(predict)
    diff_abs = jnp.abs(ground - predict)

    plt.figure()
    plt.imshow(ground_abs, cmap="coolwarm")
    plt.title(f"{title} Ground")
    plt.colorbar()
    plt.savefig(os.path.join(folder, f"{title} Ground.png"))
    plt.close()

    plt.figure()
    plt.imshow(predict_abs, cmap="coolwarm")
    plt.title(f"{title} Prediction")
    plt.colorbar()
    plt.savefig(os.path.join(folder, f"{title} Prediction.png"))
    plt.close()

    plt.figure()
    plt.imshow(diff_abs, cmap="coolwarm")
    plt.title(f"{title} Absolute Difference")
    plt.colorbar()
    plt.savefig(os.path.join(folder, f"{title} Absolute Difference.png"))
    plt.close()

# ── Noise Covariance ────────────────────────────────────────────────────

def test_noise_covariance_intensity_only():
    data_set = _get_dataset("I")
    cn_ground = precision_load(GROUND_TRUTH + "cn_I.npz")
    cn_predict = data_set.noise_covariance.scalar_matrix
    perc_diff = primal_percent_diff_2d(cn_ground, cn_predict)
    folder = FIGURE_PATH + "covariance_matrices/intensity_only/noise_covariance/"
    _plot_matrix_comparison(cn_ground, cn_predict, "Cn (I)", folder)
    assert perc_diff <= MAX_NORM_DIFF

def test_noise_covariance_polarization_only():
    data_set = _get_dataset("P")
    cn_ee_ground = precision_load(GROUND_TRUTH + "cn_ee_P.npz")
    cn_bb_ground = precision_load(GROUND_TRUTH + "cn_bb_P.npz")
    cn_ee_predict = data_set.noise_covariance.matrix_EE
    cn_bb_predict = data_set.noise_covariance.matrix_BB
    folder = FIGURE_PATH + "covariance_matrices/polarization_only/noise_covariance/"
    _plot_matrix_comparison(cn_ee_ground, cn_ee_predict, "Cn EE (P)", folder)
    _plot_matrix_comparison(cn_bb_ground, cn_bb_predict, "Cn BB (P)", folder)
    perc_diff_ee = primal_percent_diff_2d(cn_ee_ground, cn_ee_predict)
    perc_diff_bb = primal_percent_diff_2d(cn_bb_ground, cn_bb_predict)
    assert perc_diff_ee <= MAX_NORM_DIFF
    assert perc_diff_bb <= MAX_NORM_DIFF

def test_noise_covariance_intensity_and_polarization():
    data_set = _get_dataset("IP")
    cn_tt_ground = precision_load(GROUND_TRUTH + "cn_tt_IP.npz")
    cn_te_ground = precision_load(GROUND_TRUTH + "cn_te_IP.npz")
    cn_ee_ground = precision_load(GROUND_TRUTH + "cn_ee_IP.npz")
    cn_bb_ground = precision_load(GROUND_TRUTH + "cn_bb_IP.npz")
    cn_tt_predict = data_set.noise_covariance.matrix_TT
    cn_te_predict = data_set.noise_covariance.matrix_TE
    cn_ee_predict = data_set.noise_covariance.matrix_EE
    cn_bb_predict = data_set.noise_covariance.matrix_BB
    folder = FIGURE_PATH + "covariance_matrices/intensity_and_polarization/noise_covariance/"
    _plot_matrix_comparison(cn_tt_ground, cn_tt_predict, "Cn TT (IP)", folder)
    _plot_matrix_comparison(cn_te_ground, cn_te_predict, "Cn TE (IP)", folder)
    _plot_matrix_comparison(cn_ee_ground, cn_ee_predict, "Cn EE (IP)", folder)
    _plot_matrix_comparison(cn_bb_ground, cn_bb_predict, "Cn BB (IP)", folder)
    assert primal_percent_diff_2d(cn_tt_ground, cn_tt_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(cn_te_ground, cn_te_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(cn_ee_ground, cn_ee_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(cn_bb_ground, cn_bb_predict) <= MAX_NORM_DIFF

# ── Beam ────────────────────────────────────────────────────────────────

def test_beam_intensity_only():
    data_set = _get_dataset("I")
    b_ground = precision_load(GROUND_TRUTH + "b_I.npz")
    b_predict = data_set.beam.scalar_matrix
    perc_diff = primal_percent_diff_2d(b_ground, b_predict)
    folder = FIGURE_PATH + "covariance_matrices/intensity_only/beam/"
    _plot_matrix_comparison(b_ground, b_predict, "Beam (I)", folder)
    assert perc_diff <= MAX_NORM_DIFF

def test_beam_polarization_only():
    data_set = _get_dataset("P")
    b_ee_ground = precision_load(GROUND_TRUTH + "b_ee_P.npz")
    b_bb_ground = precision_load(GROUND_TRUTH + "b_bb_P.npz")
    b_ee_predict = data_set.beam.matrix_EE
    b_bb_predict = data_set.beam.matrix_BB
    folder = FIGURE_PATH + "covariance_matrices/polarization_only/beam/"
    _plot_matrix_comparison(b_ee_ground, b_ee_predict, "Beam EE (P)", folder)
    _plot_matrix_comparison(b_bb_ground, b_bb_predict, "Beam BB (P)", folder)
    assert primal_percent_diff_2d(b_ee_ground, b_ee_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(b_bb_ground, b_bb_predict) <= MAX_NORM_DIFF

def test_beam_intensity_and_polarization():
    data_set = _get_dataset("IP")
    b_tt_ground = precision_load(GROUND_TRUTH + "b_tt_IP.npz")
    b_te_ground = precision_load(GROUND_TRUTH + "b_te_IP.npz")
    b_ee_ground = precision_load(GROUND_TRUTH + "b_ee_IP.npz")
    b_bb_ground = precision_load(GROUND_TRUTH + "b_bb_IP.npz")
    b_tt_predict = data_set.beam.matrix_TT
    b_te_predict = data_set.beam.matrix_TE
    b_ee_predict = data_set.beam.matrix_EE
    b_bb_predict = data_set.beam.matrix_BB
    folder = FIGURE_PATH + "covariance_matrices/intensity_and_polarization/beam/"
    _plot_matrix_comparison(b_tt_ground, b_tt_predict, "Beam TT (IP)", folder)
    _plot_matrix_comparison(b_te_ground, b_te_predict, "Beam TE (IP)", folder)
    _plot_matrix_comparison(b_ee_ground, b_ee_predict, "Beam EE (IP)", folder)
    _plot_matrix_comparison(b_bb_ground, b_bb_predict, "Beam BB (IP)", folder)
    assert primal_percent_diff_2d(b_tt_ground, b_tt_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(b_te_ground, b_te_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(b_ee_ground, b_ee_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(b_bb_ground, b_bb_predict) <= MAX_NORM_DIFF

# ── Mask ────────────────────────────────────────────────────────────────

def test_mask_intensity_only():
    data_set = _get_dataset("I")
    m_ground = precision_load(GROUND_TRUTH + "m_I.npz")
    m_predict = data_set.mask.scalar_matrix
    perc_diff = primal_percent_diff_2d(m_ground, m_predict)
    folder = FIGURE_PATH + "covariance_matrices/intensity_only/mask/"
    _plot_matrix_comparison(m_ground, m_predict, "Mask (I)", folder)
    assert perc_diff <= MAX_NORM_DIFF

def test_mask_polarization_only():
    data_set = _get_dataset("P")
    m_ee_ground = precision_load(GROUND_TRUTH + "m_ee_P.npz")
    m_bb_ground = precision_load(GROUND_TRUTH + "m_bb_P.npz")
    m_ee_predict = data_set.mask.matrix_EE
    m_bb_predict = data_set.mask.matrix_BB
    folder = FIGURE_PATH + "covariance_matrices/polarization_only/mask/"
    _plot_matrix_comparison(m_ee_ground, m_ee_predict, "Mask EE (P)", folder)
    _plot_matrix_comparison(m_bb_ground, m_bb_predict, "Mask BB (P)", folder)
    assert primal_percent_diff_2d(m_ee_ground, m_ee_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(m_bb_ground, m_bb_predict) <= MAX_NORM_DIFF

def test_mask_intensity_and_polarization():
    data_set = _get_dataset("IP")
    m_tt_ground = precision_load(GROUND_TRUTH + "m_tt_IP.npz")
    m_te_ground = precision_load(GROUND_TRUTH + "m_te_IP.npz")
    m_ee_ground = precision_load(GROUND_TRUTH + "m_ee_IP.npz")
    m_bb_ground = precision_load(GROUND_TRUTH + "m_bb_IP.npz")
    m_tt_predict = data_set.mask.matrix_TT
    m_te_predict = data_set.mask.matrix_TE
    m_ee_predict = data_set.mask.matrix_EE
    m_bb_predict = data_set.mask.matrix_BB
    folder = FIGURE_PATH + "covariance_matrices/intensity_and_polarization/mask/"
    _plot_matrix_comparison(m_tt_ground, m_tt_predict, "Mask TT (IP)", folder)
    _plot_matrix_comparison(m_te_ground, m_te_predict, "Mask TE (IP)", folder)
    _plot_matrix_comparison(m_ee_ground, m_ee_predict, "Mask EE (IP)", folder)
    _plot_matrix_comparison(m_bb_ground, m_bb_predict, "Mask BB (IP)", folder)
    assert primal_percent_diff_2d(m_tt_ground, m_tt_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(m_te_ground, m_te_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(m_ee_ground, m_ee_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(m_bb_ground, m_bb_predict) <= MAX_NORM_DIFF

# ── Field Covariance ────────────────────────────────────────────────────

def test_field_covariance_intensity_only():
    data_set = _get_dataset("I")
    cf_ground = precision_load(GROUND_TRUTH + "cf_I.npz")
    cf_predict = data_set.field_covariance.scalar_matrix
    perc_diff = primal_percent_diff_2d(cf_ground, cf_predict)
    folder = FIGURE_PATH + "covariance_matrices/intensity_only/field_covariance/"
    _plot_matrix_comparison(cf_ground, cf_predict, "Cf (I)", folder)
    assert perc_diff <= MAX_NORM_DIFF

def test_field_covariance_polarization_only():
    data_set = _get_dataset("P")
    cf_ee_ground = precision_load(GROUND_TRUTH + "cf_ee_P.npz")
    cf_bb_ground = precision_load(GROUND_TRUTH + "cf_bb_P.npz")
    cf_ee_predict = data_set.field_covariance.matrix_EE
    cf_bb_predict = data_set.field_covariance.matrix_BB
    folder = FIGURE_PATH + "covariance_matrices/polarization_only/field_covariance/"
    _plot_matrix_comparison(cf_ee_ground, cf_ee_predict, "Cf EE (P)", folder)
    _plot_matrix_comparison(cf_bb_ground, cf_bb_predict, "Cf BB (P)", folder)
    assert primal_percent_diff_2d(cf_ee_ground, cf_ee_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(cf_bb_ground, cf_bb_predict) <= MAX_NORM_DIFF

def test_field_covariance_intensity_and_polarization():
    data_set = _get_dataset("IP")
    cf_tt_ground = precision_load(GROUND_TRUTH + "cf_tt_IP.npz")
    cf_te_ground = precision_load(GROUND_TRUTH + "cf_te_IP.npz")
    cf_ee_ground = precision_load(GROUND_TRUTH + "cf_ee_IP.npz")
    cf_bb_ground = precision_load(GROUND_TRUTH + "cf_bb_IP.npz")
    cf_tt_predict = data_set.field_covariance.matrix_TT
    cf_te_predict = data_set.field_covariance.matrix_TE
    cf_ee_predict = data_set.field_covariance.matrix_EE
    cf_bb_predict = data_set.field_covariance.matrix_BB
    folder = FIGURE_PATH + "covariance_matrices/intensity_and_polarization/field_covariance/"
    _plot_matrix_comparison(cf_tt_ground, cf_tt_predict, "Cf TT (IP)", folder)
    _plot_matrix_comparison(cf_te_ground, cf_te_predict, "Cf TE (IP)", folder)
    _plot_matrix_comparison(cf_ee_ground, cf_ee_predict, "Cf EE (IP)", folder)
    _plot_matrix_comparison(cf_bb_ground, cf_bb_predict, "Cf BB (IP)", folder)
    assert primal_percent_diff_2d(cf_tt_ground, cf_tt_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(cf_te_ground, cf_te_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(cf_ee_ground, cf_ee_predict) <= MAX_NORM_DIFF
    assert primal_percent_diff_2d(cf_bb_ground, cf_bb_predict) <= MAX_NORM_DIFF

# ── Quadratic Estimate ──────────────────────────────────────────────────

def test_quadratic_estimate_intensity_only():
    data_set = _get_dataset("I")
    nphi_ground = reciprocal_matrix(precision_load(GROUND_TRUTH + "nphi_I.npz"))
    qe_predict = reciprocal_matrix(data_set.quadratic_estimate.scalar_matrix)
    perc_diff = primal_percent_diff_2d(nphi_ground, qe_predict)
    folder = FIGURE_PATH + "covariance_matrices/intensity_only/quadratic_estimate/"
    _plot_matrix_comparison(nphi_ground, qe_predict, "QE (I)", folder)
    assert perc_diff <= MAX_NORM_DIFF

def test_quadratic_estimate_polarization_only():
    data_set = _get_dataset("P")
    nphi_ground = reciprocal_matrix(precision_load(GROUND_TRUTH + "nphi_P.npz"))
    qe_predict = reciprocal_matrix(data_set.quadratic_estimate.scalar_matrix)
    perc_diff = primal_percent_diff_2d(nphi_ground, qe_predict)
    folder = FIGURE_PATH + "covariance_matrices/polarization_only/quadratic_estimate/"
    _plot_matrix_comparison(nphi_ground, qe_predict, "QE (P)", folder)
    assert perc_diff <= MAX_NORM_DIFF

def test_quadratic_estimate_intensity_and_polarization():
    data_set = _get_dataset("IP")
    nphi_ground = reciprocal_matrix(precision_load(GROUND_TRUTH + "nphi_IP.npz"))
    qe_predict = reciprocal_matrix(data_set.quadratic_estimate.scalar_matrix)
    perc_diff = primal_percent_diff_2d(nphi_ground, qe_predict)
    folder = FIGURE_PATH + "covariance_matrices/intensity_and_polarization/quadratic_estimate/"
    _plot_matrix_comparison(nphi_ground, qe_predict, "QE (IP)", folder)
    assert perc_diff <= MAX_NORM_DIFF
