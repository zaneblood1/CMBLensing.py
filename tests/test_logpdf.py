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

def test_logpdf_intensity_only():

    #load ground truth unlensed intensity field, data field, and phi field
    phi_matrix = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_I.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_I.npz")
    noise_covariance_matrix = precision_load(GROUND_TRUTH + "cn_I.npz")
    phi_covariance_matrix = precision_load(GROUND_TRUTH + "cphi_I.npz")
    field_covariance_matrix = precision_load(GROUND_TRUTH + "cf_I.npz")
    mask_matrix = precision_load(GROUND_TRUTH + "m_I.npz")
    beam_matrix = precision_load(GROUND_TRUTH + "b_I.npz")

    #convert the matrices into fields and operators
    phi = FlatS0(scalar_matrix = phi_matrix)
    unlensed_field = FlatS0(scalar_matrix = unlensed_t_matrix)
    data = FlatS0(scalar_matrix = data_t_matrix)
    noise_covariance = DiagonalScalar(scalar_matrix = noise_covariance_matrix)
    phi_covariance = DiagonalScalar(scalar_matrix = phi_covariance_matrix)
    field_covariance = DiagonalScalar(scalar_matrix = field_covariance_matrix)
    mask = DiagonalScalar(scalar_matrix = mask_matrix)
    beam = DiagonalScalar(scalar_matrix = beam_matrix)
    mixing_g = 0*phi_covariance
    logpdf_t_predict = logpdf(unlensed_field, phi, data,
                              noise_covariance, phi_covariance,
                              field_covariance, mask, beam, mixing_g)
    with open(GROUND_TRUTH + "logpdf_I.txt", "r") as file:
        logpdf_t_ground = float(file.read().strip())
    assert abs(logpdf_t_predict - logpdf_t_ground) < 1

def test_logpdf_polarization_only():
    #load ground truth unlensed polarization fields, data fields, and phi field
    phi_matrix = precision_load(GROUND_TRUTH + "phi_P.npz")
    unlensed_e_matrix = precision_load(GROUND_TRUTH + "e_field_P.npz")
    unlensed_b_matrix = precision_load(GROUND_TRUTH + "b_field_P.npz")
    data_e_matrix = precision_load(GROUND_TRUTH + "data_e_field_P.npz")
    data_b_matrix = precision_load(GROUND_TRUTH + "data_b_field_P.npz")
    noise_covariance_ee_matrix = precision_load(GROUND_TRUTH + "cn_ee_P.npz")
    noise_covariance_bb_matrix = precision_load(GROUND_TRUTH + "cn_bb_P.npz")
    phi_covariance_matrix = precision_load(GROUND_TRUTH + "cphi_P.npz")
    field_covariance_ee_matrix = precision_load(GROUND_TRUTH + "cf_ee_P.npz")
    field_covariance_bb_matrix = precision_load(GROUND_TRUTH + "cf_bb_P.npz")
    mask_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_P.npz")
    mask_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_P.npz")
    beam_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_P.npz")
    beam_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_P.npz")

    #convert the matrices into fields and operators
    phi = FlatS0(scalar_matrix = phi_matrix)
    unlensed_eb_field = FlatS2(
        polar_matrix_1 = unlensed_e_matrix,
        polar_matrix_2 = unlensed_b_matrix,
    )
    data_eb_field = FlatS2(
        polar_matrix_1 = data_e_matrix,
        polar_matrix_2 = data_b_matrix,
    )
    noise_covariance = DiagonalEB(
        matrix_EE = noise_covariance_ee_matrix,
        matrix_BB = noise_covariance_bb_matrix,
    )
    phi_covariance = DiagonalScalar(scalar_matrix = phi_covariance_matrix)
    field_covariance = DiagonalEB(
        matrix_EE = field_covariance_ee_matrix,
        matrix_BB = field_covariance_bb_matrix,
    )
    mask = DiagonalEB(
        matrix_EE = mask_ee_matrix,
        matrix_BB = mask_bb_matrix,
    )
    beam = DiagonalEB(
        matrix_EE = beam_ee_matrix,
        matrix_BB = beam_bb_matrix,
    )
    mixing_g = 0*phi_covariance
    logpdf_eb_predict = logpdf(unlensed_eb_field, phi, data_eb_field,
                              noise_covariance, phi_covariance,
                              field_covariance, mask, beam, mixing_g)
    with open(GROUND_TRUTH + "logpdf_P.txt", "r") as file:
        logpdf_eb_ground = float(file.read().strip())
    assert abs(logpdf_eb_predict - logpdf_eb_ground) < 1

def test_logpdf_intensity_and_polarization():
    #load ground truth unlensed intensity and polarization fields, data fields, and phi field
    phi_matrix = precision_load(GROUND_TRUTH + "phi_IP.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_IP.npz")
    unlensed_e_matrix = precision_load(GROUND_TRUTH + "e_field_IP.npz")
    unlensed_b_matrix = precision_load(GROUND_TRUTH + "b_field_IP.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_IP.npz")
    data_e_matrix = precision_load(GROUND_TRUTH + "data_e_field_IP.npz")
    data_b_matrix = precision_load(GROUND_TRUTH + "data_b_field_IP.npz")
    noise_covariance_tt_matrix = precision_load(GROUND_TRUTH + "cn_tt_IP.npz")
    noise_covariance_te_matrix = precision_load(GROUND_TRUTH + "cn_te_IP.npz")
    noise_covariance_ee_matrix = precision_load(GROUND_TRUTH + "cn_ee_IP.npz")
    noise_covariance_bb_matrix = precision_load(GROUND_TRUTH + "cn_bb_IP.npz")
    phi_covariance_matrix = precision_load(GROUND_TRUTH + "cphi_IP.npz")
    field_covariance_tt_matrix = precision_load(GROUND_TRUTH + "cf_tt_IP.npz")
    field_covariance_te_matrix = precision_load(GROUND_TRUTH + "cf_te_IP.npz")
    field_covariance_ee_matrix = precision_load(GROUND_TRUTH + "cf_ee_IP.npz")
    field_covariance_bb_matrix = precision_load(GROUND_TRUTH + "cf_bb_IP.npz")
    mask_tt_matrix = precision_load(GROUND_TRUTH + "m_tt_IP.npz")
    mask_te_matrix = precision_load(GROUND_TRUTH + "m_te_IP.npz")
    mask_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_IP.npz")
    mask_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_IP.npz")
    beam_tt_matrix = precision_load(GROUND_TRUTH + "b_tt_IP.npz")
    beam_te_matrix = precision_load(GROUND_TRUTH + "b_te_IP.npz")
    beam_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_IP.npz")
    beam_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_IP.npz")

    #convert the matrices into fields and operators
    phi = FlatS0(scalar_matrix = phi_matrix)
    unlensed_teb_field = FlatS02(
        scalar_matrix = unlensed_t_matrix,
        polar_matrix_1 = unlensed_e_matrix,
        polar_matrix_2 = unlensed_b_matrix,
    )
    data_teb_field = FlatS02(
        scalar_matrix = data_t_matrix,
        polar_matrix_1 = data_e_matrix,
        polar_matrix_2 = data_b_matrix,
    )
    noise_covariance = BlockTEB(
        matrix_TT = noise_covariance_tt_matrix,
        matrix_TE = noise_covariance_te_matrix,
        matrix_ET = noise_covariance_te_matrix,
        matrix_EE = noise_covariance_ee_matrix,
        matrix_BB = noise_covariance_bb_matrix,
    )
    phi_covariance = DiagonalScalar(scalar_matrix = phi_covariance_matrix)
    field_covariance = BlockTEB(
        matrix_TT = field_covariance_tt_matrix,
        matrix_TE = field_covariance_te_matrix,
        matrix_ET = field_covariance_te_matrix,
        matrix_EE = field_covariance_ee_matrix,
        matrix_BB = field_covariance_bb_matrix,
    )
    mask = BlockTEB(
        matrix_TT = mask_tt_matrix,
        matrix_TE = mask_te_matrix,
        matrix_ET = mask_te_matrix,
        matrix_EE = mask_ee_matrix,
        matrix_BB = mask_bb_matrix,
    )
    beam = BlockTEB(
        matrix_TT = beam_tt_matrix,
        matrix_TE = beam_te_matrix,
        matrix_ET = beam_te_matrix,
        matrix_EE = beam_ee_matrix,
        matrix_BB = beam_bb_matrix,
    )
    mixing_g = 0*phi_covariance
    logpdf_teb_predict = logpdf(unlensed_teb_field, phi, data_teb_field,
                                noise_covariance, phi_covariance,
                                field_covariance, mask, beam, mixing_g)
    with open(GROUND_TRUTH + "logpdf_IP.txt", "r") as file:
        logpdf_teb_ground = float(file.read().strip())
    assert abs(logpdf_teb_predict - logpdf_teb_ground) < 1
