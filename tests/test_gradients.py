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

def test_gradf_intensity_only():

    #load ground truth data
    phi_matrix = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_I.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_I.npz")
    cn_matrix = precision_load(GROUND_TRUTH + "cn_I.npz")
    cf_matrix = precision_load(GROUND_TRUTH + "cf_I.npz")
    m_matrix = precision_load(GROUND_TRUTH + "m_I.npz")
    b_matrix = precision_load(GROUND_TRUTH + "b_I.npz")
    gradf_ground_matrix = precision_load(GROUND_TRUTH + "gradf_I.npz")

    data = FlatS0(scalar_matrix = data_t_matrix)
    field = FlatS0(scalar_matrix = unlensed_t_matrix)
    phi = FlatS0(scalar_matrix = phi_matrix)
    gradf_ground = FlatS0(scalar_matrix = gradf_ground_matrix)

    noise_covariance = DiagonalScalar(scalar_matrix = cn_matrix)
    field_covariance = DiagonalScalar(scalar_matrix = cf_matrix)
    mask = DiagonalScalar(scalar_matrix = m_matrix)
    beam = DiagonalScalar(scalar_matrix = b_matrix)

    gradf_predict = gradf_logpdf(field, phi, data, field_covariance, noise_covariance, mask, beam)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "gradients/intensity_only/grad_f_logpdf/"

    _plot_field = gradf_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad F Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = gradf_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad F Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = gradf_predict - gradf_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad F Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Absolute Difference.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = fourier(gradf_ground) if gradf_ground.basis != Basis.FOURIER else gradf_ground
    _cc_f2 = fourier(gradf_predict) if gradf_predict.basis != Basis.FOURIER else gradf_predict
    ell, gf_cross_gf = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, gf_cross_gf, label = "TT")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Grad F Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    perc_diff = jnp.max(percent_diff_2d(gradf_ground, gradf_predict))
    assert perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation = jnp.mean(gf_cross_gf)
    assert avg_correlation > MIN_AVG_CORRELATION

def test_gradf_polarization_only():
    #load ground truth data
    phi_matrix = precision_load(GROUND_TRUTH + "phi_P.npz")
    unlensed_e_matrix = precision_load(GROUND_TRUTH + "e_field_P.npz")
    unlensed_b_matrix = precision_load(GROUND_TRUTH + "b_field_P.npz")
    data_e_matrix = precision_load(GROUND_TRUTH + "data_e_field_P.npz")
    data_b_matrix = precision_load(GROUND_TRUTH + "data_b_field_P.npz")
    cn_ee_matrix = precision_load(GROUND_TRUTH + "cn_ee_P.npz")
    cn_bb_matrix = precision_load(GROUND_TRUTH + "cn_bb_P.npz")
    cf_ee_matrix = precision_load(GROUND_TRUTH + "cf_ee_P.npz")
    cf_bb_matrix = precision_load(GROUND_TRUTH + "cf_bb_P.npz")
    m_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_P.npz")
    m_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_P.npz")
    b_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_P.npz")
    b_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_P.npz")
    gradf_e_ground_matrix = precision_load(GROUND_TRUTH + "gradf_e_P.npz")
    gradf_b_ground_matrix = precision_load(GROUND_TRUTH + "gradf_b_P.npz")

    data_eb = FlatS2(polar_matrix_1 = data_e_matrix,
                     polar_matrix_2 = data_b_matrix)
    field_eb = FlatS2(polar_matrix_1 = unlensed_e_matrix,
                      polar_matrix_2 = unlensed_b_matrix)
    phi = FlatS0(scalar_matrix = phi_matrix)
    gradf_eb_ground = FlatS2(polar_matrix_1 = gradf_e_ground_matrix,
                             polar_matrix_2 = gradf_b_ground_matrix)

    noise_covariance = DiagonalEB(matrix_EE = cn_ee_matrix,
                                 matrix_BB = cn_bb_matrix)
    field_covariance = DiagonalEB(matrix_EE = cf_ee_matrix,
                                  matrix_BB = cf_bb_matrix)
    mask = DiagonalEB(matrix_EE = m_ee_matrix,
                      matrix_BB = m_bb_matrix)
    beam = DiagonalEB(matrix_EE = b_ee_matrix,
                      matrix_BB = b_bb_matrix)

    gradf_eb_predict = gradf_logpdf(field_eb, phi, data_eb, field_covariance, noise_covariance, mask, beam)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "gradients/polarization_only/grad_f_logpdf/"

    _plot_field = gradf_eb_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Grad F Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Grad F Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = gradf_eb_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Grad F Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Grad F Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = gradf_eb_predict - gradf_eb_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Grad F Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Grad F Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between ground and predict
    _cc_f1 = fourier(gradf_eb_ground) if gradf_eb_ground.basis != Basis.FOURIER else gradf_eb_ground
    _cc_f2 = fourier(gradf_eb_predict) if gradf_eb_predict.basis != Basis.FOURIER else gradf_eb_predict
    ell, gfe_cross_gfe = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, gfb_cross_gfb = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, gfe_cross_gfe, label = "EE")
    plt.plot(ell, gfb_cross_gfb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Grad F Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    perc_diff = jnp.max(percent_diff_2d(gradf_eb_ground, gradf_eb_predict))
    assert perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_e_correlation = jnp.mean(gfe_cross_gfe)
    assert avg_e_correlation > MIN_AVG_CORRELATION
    avg_b_correlation = jnp.mean(gfb_cross_gfb)
    assert avg_b_correlation > MIN_AVG_CORRELATION

def test_gradf_intensity_and_polarization():
    #load ground truth data
    phi_matrix = precision_load(GROUND_TRUTH + "phi_IP.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_IP.npz")
    unlensed_e_matrix = precision_load(GROUND_TRUTH + "e_field_IP.npz")
    unlensed_b_matrix = precision_load(GROUND_TRUTH + "b_field_IP.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_IP.npz")
    data_e_matrix = precision_load(GROUND_TRUTH + "data_e_field_IP.npz")
    data_b_matrix = precision_load(GROUND_TRUTH + "data_b_field_IP.npz")
    cn_ee_matrix = precision_load(GROUND_TRUTH + "cn_ee_IP.npz")
    cn_bb_matrix = precision_load(GROUND_TRUTH + "cn_bb_IP.npz")
    cn_te_matrix = precision_load(GROUND_TRUTH + "cn_te_IP.npz")
    cn_tt_matrix = precision_load(GROUND_TRUTH + "cn_tt_IP.npz")
    cf_ee_matrix = precision_load(GROUND_TRUTH + "cf_ee_IP.npz")
    cf_bb_matrix = precision_load(GROUND_TRUTH + "cf_bb_IP.npz")
    cf_te_matrix = precision_load(GROUND_TRUTH + "cf_te_IP.npz")
    cf_tt_matrix = precision_load(GROUND_TRUTH + "cf_tt_IP.npz")
    m_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_IP.npz")
    m_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_IP.npz")
    m_te_matrix = precision_load(GROUND_TRUTH + "m_te_IP.npz")
    m_tt_matrix = precision_load(GROUND_TRUTH + "m_tt_IP.npz")
    b_te_matrix = precision_load(GROUND_TRUTH + "b_te_IP.npz")
    b_tt_matrix = precision_load(GROUND_TRUTH + "b_tt_IP.npz")
    b_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_IP.npz")
    b_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_IP.npz")
    gradf_t_ground_matrix = precision_load(GROUND_TRUTH + "gradf_t_IP.npz")
    gradf_e_ground_matrix = precision_load(GROUND_TRUTH + "gradf_e_IP.npz")
    gradf_b_ground_matrix = precision_load(GROUND_TRUTH + "gradf_b_IP.npz")

    data_teb = FlatS02(scalar_matrix = data_t_matrix,
                        polar_matrix_1 = data_e_matrix,
                        polar_matrix_2 = data_b_matrix)
    field_teb = FlatS02(scalar_matrix = unlensed_t_matrix,
                         polar_matrix_1 = unlensed_e_matrix,
                         polar_matrix_2 = unlensed_b_matrix)
    phi = FlatS0(scalar_matrix = phi_matrix)
    gradf_teb_ground = FlatS02(scalar_matrix = gradf_t_ground_matrix,
                                polar_matrix_1 = gradf_e_ground_matrix,
                                polar_matrix_2 = gradf_b_ground_matrix)
    noise_covariance = BlockTEB(matrix_TT = cn_tt_matrix,
                                matrix_TE = cn_te_matrix,
                                matrix_ET = cn_te_matrix,
                                matrix_EE = cn_ee_matrix,
                                matrix_BB = cn_bb_matrix)
    field_covariance = BlockTEB(matrix_TT = cf_tt_matrix,
                                matrix_TE = cf_te_matrix,
                                matrix_ET = cf_te_matrix,
                                matrix_EE = cf_ee_matrix,
                                matrix_BB = cf_bb_matrix)
    mask = BlockTEB(matrix_TT = m_tt_matrix,
                    matrix_TE = m_te_matrix,
                    matrix_ET = m_te_matrix,
                    matrix_EE = m_ee_matrix,
                    matrix_BB = m_bb_matrix)
    beam = BlockTEB(matrix_TT = b_tt_matrix,
                    matrix_TE = b_te_matrix,
                    matrix_ET = b_te_matrix,
                    matrix_EE = b_ee_matrix,
                    matrix_BB = b_bb_matrix)

    gradf_teb_predict = gradf_logpdf(field_teb, phi, data_teb, field_covariance, noise_covariance, mask, beam)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "gradients/intensity_and_polarization/grad_f_logpdf/"

    _plot_field = gradf_teb_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad F Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Grad F Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Grad F Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = gradf_teb_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad F Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Grad F Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Grad F Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = gradf_teb_predict - gradf_teb_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad F Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Grad F Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Grad F Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = fourier(gradf_teb_ground) if gradf_teb_ground.basis != Basis.FOURIER else gradf_teb_ground
    _cc_f2 = fourier(gradf_teb_predict) if gradf_teb_predict.basis != Basis.FOURIER else gradf_teb_predict
    ell, gft_cross_gft = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _, gfe_cross_gfe = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, gfb_cross_gfb = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, gft_cross_gft, label = "TT")
    plt.plot(ell, gfe_cross_gfe, label = "EE")
    plt.plot(ell, gfb_cross_gfb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Grad F Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad F Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    perc_diff = jnp.max(percent_diff_2d(gradf_teb_ground, gradf_teb_predict))
    assert perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_t_correlation = jnp.mean(gft_cross_gft)
    assert avg_t_correlation > MIN_AVG_CORRELATION
    avg_e_correlation = jnp.mean(gfe_cross_gfe)
    assert avg_e_correlation > MIN_AVG_CORRELATION
    avg_b_correlation = jnp.mean(gfb_cross_gfb)
    assert avg_b_correlation > MIN_AVG_CORRELATION

def test_grad_phi_intensity_only():

    #load ground truth data
    phi_matrix = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_I.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_I.npz")
    cn_matrix = precision_load(GROUND_TRUTH + "cn_I.npz")
    cphi_matrix = precision_load(GROUND_TRUTH + "cphi_I.npz")
    cf_matrix = precision_load(GROUND_TRUTH + "cf_I.npz")
    m_matrix = precision_load(GROUND_TRUTH + "m_I.npz")
    b_matrix = precision_load(GROUND_TRUTH + "b_I.npz")
    grad_phi_ground_matrix = precision_load(GROUND_TRUTH + "grad_phi_I.npz")
    data = FlatS0(scalar_matrix = data_t_matrix)
    field = FlatS0(scalar_matrix = unlensed_t_matrix)
    phi = FlatS0(scalar_matrix = phi_matrix)
    phi_covariance = DiagonalScalar(scalar_matrix = cphi_matrix)
    noise_covariance = DiagonalScalar(scalar_matrix = cn_matrix)
    field_covariance = DiagonalScalar(scalar_matrix = cf_matrix)
    mask = DiagonalScalar(scalar_matrix = m_matrix)
    beam = DiagonalScalar(scalar_matrix = b_matrix)
    grad_phi_ground = FlatS0(scalar_matrix = grad_phi_ground_matrix)

    #Call the python predicted phi gradient of logpdf
    grad_phi_predict = grad_phi_logpdf(field, phi, data, noise_covariance,
                                   phi_covariance, field_covariance, mask, beam)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "gradients/intensity_only/grad_phi_logpdf/"

    _plot_field = grad_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = grad_phi_predict - grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Absolute Difference.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = fourier(grad_phi_ground) if grad_phi_ground.basis != Basis.FOURIER else grad_phi_ground
    _cc_f2 = fourier(grad_phi_predict) if grad_phi_predict.basis != Basis.FOURIER else grad_phi_predict
    ell, gphi_cross_gphi = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, gphi_cross_gphi, label = "PP")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Grad Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    perc_diff = jnp.max(percent_diff_2d(grad_phi_ground, grad_phi_predict))
    assert perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation = jnp.mean(gphi_cross_gphi)
    assert avg_correlation > MIN_AVG_CORRELATION

def test_grad_phi_polarization_only():

    #load ground truth data
    phi_matrix = precision_load(GROUND_TRUTH + "phi_P.npz")
    unlensed_e_matrix = precision_load(GROUND_TRUTH + "e_field_P.npz")
    unlensed_b_matrix = precision_load(GROUND_TRUTH + "b_field_P.npz")
    data_e_matrix = precision_load(GROUND_TRUTH + "data_e_field_P.npz")
    data_b_matrix = precision_load(GROUND_TRUTH + "data_b_field_P.npz")
    cn_ee_matrix = precision_load(GROUND_TRUTH + "cn_ee_P.npz")
    cn_bb_matrix = precision_load(GROUND_TRUTH + "cn_bb_P.npz")
    cphi_matrix = precision_load(GROUND_TRUTH + "cphi_P.npz")
    cf_ee_matrix = precision_load(GROUND_TRUTH + "cf_ee_P.npz")
    cf_bb_matrix = precision_load(GROUND_TRUTH + "cf_bb_P.npz")
    m_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_P.npz")
    m_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_P.npz")
    b_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_P.npz")
    b_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_P.npz")
    grad_phi_ground_matrix = precision_load(GROUND_TRUTH + "grad_phi_t_P.npz")
    data = FlatS2(polar_matrix_1 = data_e_matrix,
                  polar_matrix_2 = data_b_matrix)
    field = FlatS2(polar_matrix_1 = unlensed_e_matrix,
                   polar_matrix_2 = unlensed_b_matrix)
    phi = FlatS0(scalar_matrix = phi_matrix)
    phi_covariance = DiagonalScalar(scalar_matrix = cphi_matrix)
    noise_covariance = DiagonalEB(matrix_EE = cn_ee_matrix,
                                  matrix_BB = cn_bb_matrix)
    field_covariance = DiagonalEB(matrix_EE = cf_ee_matrix,
                                  matrix_BB = cf_bb_matrix)
    mask = DiagonalEB(matrix_EE = m_ee_matrix,
                      matrix_BB = m_bb_matrix)
    beam = DiagonalEB(matrix_EE = b_ee_matrix,
                      matrix_BB = b_bb_matrix)
    grad_phi_ground = FlatS0(scalar_matrix = grad_phi_ground_matrix)

    #Call the python predicted phi gradient of logpdf
    grad_phi_predict = grad_phi_logpdf(field, phi, data, noise_covariance,
                                   phi_covariance, field_covariance, mask, beam)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "gradients/polarization_only/grad_phi_logpdf/"

    _plot_field = grad_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = grad_phi_predict - grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Absolute Difference.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = fourier(grad_phi_ground) if grad_phi_ground.basis != Basis.FOURIER else grad_phi_ground
    _cc_f2 = fourier(grad_phi_predict) if grad_phi_predict.basis != Basis.FOURIER else grad_phi_predict
    ell, gphi_cross_gphi = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, gphi_cross_gphi, label = "PP")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Grad Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    perc_diff = jnp.max(percent_diff_2d(grad_phi_ground, grad_phi_predict))
    assert perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation = jnp.mean(gphi_cross_gphi)
    assert avg_correlation > MIN_AVG_CORRELATION

def test_grad_phi_intensity_and_polarization():

    #load ground truth data
    phi_matrix = precision_load(GROUND_TRUTH + "phi_IP.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_IP.npz")
    unlensed_e_matrix = precision_load(GROUND_TRUTH + "e_field_IP.npz")
    unlensed_b_matrix = precision_load(GROUND_TRUTH + "b_field_IP.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_IP.npz")
    data_e_matrix = precision_load(GROUND_TRUTH + "data_e_field_IP.npz")
    data_b_matrix = precision_load(GROUND_TRUTH + "data_b_field_IP.npz")
    cn_tt_matrix = precision_load(GROUND_TRUTH + "cn_tt_IP.npz")
    cn_ee_matrix = precision_load(GROUND_TRUTH + "cn_ee_IP.npz")
    cn_bb_matrix = precision_load(GROUND_TRUTH + "cn_bb_IP.npz")
    cn_te_matrix = precision_load(GROUND_TRUTH + "cn_te_IP.npz")
    cphi_matrix = precision_load(GROUND_TRUTH + "cphi_IP.npz")
    cf_tt_matrix = precision_load(GROUND_TRUTH + "cf_tt_IP.npz")
    cf_ee_matrix = precision_load(GROUND_TRUTH + "cf_ee_IP.npz")
    cf_bb_matrix = precision_load(GROUND_TRUTH + "cf_bb_IP.npz")
    cf_te_matrix = precision_load(GROUND_TRUTH + "cf_te_IP.npz")
    m_tt_matrix = precision_load(GROUND_TRUTH + "m_tt_IP.npz")
    m_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_IP.npz")
    m_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_IP.npz")
    m_te_matrix = precision_load(GROUND_TRUTH + "m_te_IP.npz")
    b_tt_matrix = precision_load(GROUND_TRUTH + "b_tt_IP.npz")
    b_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_IP.npz")
    b_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_IP.npz")
    b_te_matrix = precision_load(GROUND_TRUTH + "b_te_IP.npz")
    grad_phi_ground_matrix = precision_load(GROUND_TRUTH + "grad_phi_t_IP.npz")
    data = FlatS02(scalar_matrix = data_t_matrix,
                   polar_matrix_1 = data_e_matrix,
                   polar_matrix_2 = data_b_matrix)
    field = FlatS02(scalar_matrix = unlensed_t_matrix,
                    polar_matrix_1 = unlensed_e_matrix,
                    polar_matrix_2 = unlensed_b_matrix)
    phi = FlatS0(scalar_matrix = phi_matrix)
    phi_covariance = DiagonalScalar(scalar_matrix = cphi_matrix)
    noise_covariance = BlockTEB(matrix_TT = cn_tt_matrix,
                                matrix_TE = cn_te_matrix,
                                matrix_ET = cn_te_matrix,
                                matrix_EE = cn_ee_matrix,
                                matrix_BB = cn_bb_matrix)
    field_covariance = BlockTEB(matrix_TT = cf_tt_matrix,
                                matrix_TE = cf_te_matrix,
                                matrix_ET = cf_te_matrix,
                                matrix_EE = cf_ee_matrix,
                                matrix_BB = cf_bb_matrix)
    mask = BlockTEB(matrix_TT = m_tt_matrix,
                    matrix_TE = m_te_matrix,
                    matrix_ET = m_te_matrix,
                    matrix_EE = m_ee_matrix,
                    matrix_BB = m_bb_matrix)
    beam = BlockTEB(matrix_TT = b_tt_matrix,
                    matrix_TE = b_te_matrix,
                    matrix_ET = b_te_matrix,
                    matrix_EE = b_ee_matrix,
                    matrix_BB = b_bb_matrix)
    grad_phi_ground = FlatS0(scalar_matrix = grad_phi_ground_matrix)

    #Call the python predicted phi gradient of logpdf
    grad_phi_predict = grad_phi_logpdf(field, phi, data, noise_covariance,
                                   phi_covariance, field_covariance, mask, beam)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "gradients/intensity_and_polarization/grad_phi_logpdf/"

    _plot_field = grad_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = grad_phi_predict - grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Grad Phi Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Absolute Difference.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = fourier(grad_phi_ground) if grad_phi_ground.basis != Basis.FOURIER else grad_phi_ground
    _cc_f2 = fourier(grad_phi_predict) if grad_phi_predict.basis != Basis.FOURIER else grad_phi_predict
    ell, gphi_cross_gphi = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, gphi_cross_gphi, label = "PP")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Grad Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Grad Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    perc_diff = jnp.max(percent_diff_2d(grad_phi_ground, grad_phi_predict))
    assert perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation = jnp.mean(gphi_cross_gphi)
    assert avg_correlation > MIN_AVG_CORRELATION

def test_mixed_grad_phi_intensity_only():
    #load ground truth data matrices
    phi_matrix = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_I.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_I.npz")
    cn_matrix = precision_load(GROUND_TRUTH + "cn_I.npz")
    cf_matrix = precision_load(GROUND_TRUTH + "cf_I.npz")
    cphi_matrix = precision_load(GROUND_TRUTH + "cphi_I.npz")
    m_matrix = precision_load(GROUND_TRUTH + "m_I.npz")
    b_matrix = precision_load(GROUND_TRUTH + "b_I.npz")
    mixing_d_matrix = precision_load(GROUND_TRUTH + "d_I.npz")
    mixing_g_matrix = precision_load(GROUND_TRUTH + "g_I.npz")
    mixed_grad_phi_matrix = precision_load(GROUND_TRUTH + "mixed_grad_phi_I.npz")

    #convert ground truth data matrices into field and operator objects
    data = FlatS0(scalar_matrix = data_t_matrix)
    field = FlatS0(scalar_matrix = unlensed_t_matrix)
    phi = FlatS0(scalar_matrix = phi_matrix)
    mixed_grad_phi_ground = FlatS0(scalar_matrix = mixed_grad_phi_matrix)

    noise_covariance = DiagonalScalar(scalar_matrix = cn_matrix)
    phi_covariance = DiagonalScalar(scalar_matrix = cphi_matrix)
    field_covariance = DiagonalScalar(scalar_matrix = cf_matrix)
    mask = DiagonalScalar(scalar_matrix = m_matrix)
    beam = DiagonalScalar(scalar_matrix = b_matrix)
    mixing_d = DiagonalScalar(scalar_matrix = mixing_d_matrix)
    mixing_g = DiagonalScalar(scalar_matrix = mixing_g_matrix)

    mixed_field, mixed_phi = mix(field, phi, mixing_d, mixing_g)
    mixed_grad_phi_predict = mixed_grad_phi_logpdf(mixed_field, mixed_phi, data, noise_covariance,
                                                    phi_covariance, field_covariance, mask, beam, mixing_d, mixing_g)

    SUB_FOLDER = "gradients/intensity_only/mixed_grad_phi_logpdf/"

    _plot_field = mixed_grad_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Predict")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Predict.png", bbox_inches="tight")
    plt.close()

    _plot_field = mixed_grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = mixed_grad_phi_predict - mixed_grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Diff")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Diff.png", bbox_inches="tight")
    plt.close()

    _cc_f1 = fourier(mixed_grad_phi_ground) if mixed_grad_phi_ground.basis != Basis.FOURIER else mixed_grad_phi_ground
    _cc_f2 = fourier(mixed_grad_phi_predict) if mixed_grad_phi_predict.basis != Basis.FOURIER else mixed_grad_phi_predict
    ell, gphi_cross_gphi = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, gphi_cross_gphi, label = "PP")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Mixed Grad Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(mixed_grad_phi_ground, mixed_grad_phi_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    assert jnp.mean(gphi_cross_gphi) > MIN_AVG_CORRELATION

def test_mixed_grad_phi_polarization_only():
    #load ground truth data matrices
    phi_matrix = precision_load(GROUND_TRUTH + "phi_P.npz")
    unlensed_e_matrix = precision_load(GROUND_TRUTH + "e_field_P.npz")
    unlensed_b_matrix = precision_load(GROUND_TRUTH + "b_field_P.npz")
    data_e_matrix = precision_load(GROUND_TRUTH + "data_e_field_P.npz")
    data_b_matrix = precision_load(GROUND_TRUTH + "data_b_field_P.npz")
    cn_ee_matrix = precision_load(GROUND_TRUTH + "cn_ee_P.npz")
    cn_bb_matrix = precision_load(GROUND_TRUTH + "cn_bb_P.npz")
    cf_ee_matrix = precision_load(GROUND_TRUTH + "cf_ee_P.npz")
    cf_bb_matrix = precision_load(GROUND_TRUTH + "cf_bb_P.npz")
    cphi_matrix = precision_load(GROUND_TRUTH + "cphi_P.npz")
    m_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_P.npz")
    m_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_P.npz")
    b_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_P.npz")
    b_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_P.npz")
    mixing_d_ee_matrix = precision_load(GROUND_TRUTH + "d_ee_P.npz")
    mixing_d_bb_matrix = precision_load(GROUND_TRUTH + "d_bb_P.npz")
    mixing_g_matrix = precision_load(GROUND_TRUTH + "g_I.npz")
    mixed_grad_phi_matrix = precision_load(GROUND_TRUTH + "mixed_grad_phi_P.npz")

    #convert ground truth data matrices into field and operator objects
    data = FlatS2(
        polar_matrix_1 = data_e_matrix,
        polar_matrix_2 = data_b_matrix,
    )
    field = FlatS2(
        polar_matrix_1 = unlensed_e_matrix,
        polar_matrix_2 = unlensed_b_matrix,
    )
    phi = FlatS0(scalar_matrix = phi_matrix)
    mixed_grad_phi_ground = FlatS0(scalar_matrix = mixed_grad_phi_matrix)

    noise_covariance = DiagonalEB(
        matrix_EE = cn_ee_matrix,
        matrix_BB = cn_bb_matrix,
    )
    phi_covariance = DiagonalScalar(scalar_matrix = cphi_matrix)
    field_covariance = DiagonalEB(
        matrix_EE = cf_ee_matrix,
        matrix_BB = cf_bb_matrix,
    )
    mask = DiagonalEB(
        matrix_EE = m_ee_matrix,
        matrix_BB = m_bb_matrix,
    )
    beam = DiagonalEB(
        matrix_EE = b_ee_matrix,
        matrix_BB = b_bb_matrix,
    )
    mixing_d = DiagonalEB(
        matrix_EE = mixing_d_ee_matrix,
        matrix_BB = mixing_d_bb_matrix,
    )
    mixing_g = DiagonalScalar(scalar_matrix = mixing_g_matrix)

    mixed_field, mixed_phi = mix(field, phi, mixing_d, mixing_g)
    mixed_grad_phi_predict = mixed_grad_phi_logpdf(mixed_field, mixed_phi, data, noise_covariance,
                                                    phi_covariance, field_covariance, mask, beam, mixing_d, mixing_g)

    SUB_FOLDER = "gradients/polarization_only/mixed_grad_phi_logpdf/"

    _plot_field = mixed_grad_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Predict")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Predict.png", bbox_inches="tight")
    plt.close()

    _plot_field = mixed_grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = mixed_grad_phi_predict - mixed_grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Diff")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Diff.png", bbox_inches="tight")
    plt.close()

    _cc_f1 = fourier(mixed_grad_phi_ground) if mixed_grad_phi_ground.basis != Basis.FOURIER else mixed_grad_phi_ground
    _cc_f2 = fourier(mixed_grad_phi_predict) if mixed_grad_phi_predict.basis != Basis.FOURIER else mixed_grad_phi_predict
    ell, gphi_cross_gphi = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, gphi_cross_gphi, label = "PP")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Mixed Grad Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(mixed_grad_phi_ground, mixed_grad_phi_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    assert jnp.mean(gphi_cross_gphi) > MIN_AVG_CORRELATION

def test_mixed_grad_phi_intensity_and_polarization():
    #load ground truth data matrices
    phi_matrix = precision_load(GROUND_TRUTH + "phi_IP.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_IP.npz")
    unlensed_e_matrix = precision_load(GROUND_TRUTH + "e_field_IP.npz")
    unlensed_b_matrix = precision_load(GROUND_TRUTH + "b_field_IP.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_IP.npz")
    data_e_matrix = precision_load(GROUND_TRUTH + "data_e_field_IP.npz")
    data_b_matrix = precision_load(GROUND_TRUTH + "data_b_field_IP.npz")
    cn_tt_matrix = precision_load(GROUND_TRUTH + "cn_tt_IP.npz")
    cn_te_matrix = precision_load(GROUND_TRUTH + "cn_te_IP.npz")
    cn_ee_matrix = precision_load(GROUND_TRUTH + "cn_ee_IP.npz")
    cn_bb_matrix = precision_load(GROUND_TRUTH + "cn_bb_IP.npz")
    cf_tt_matrix = precision_load(GROUND_TRUTH + "cf_tt_IP.npz")
    cf_te_matrix = precision_load(GROUND_TRUTH + "cf_te_IP.npz")
    cf_ee_matrix = precision_load(GROUND_TRUTH + "cf_ee_IP.npz")
    cf_bb_matrix = precision_load(GROUND_TRUTH + "cf_bb_IP.npz")
    cphi_matrix = precision_load(GROUND_TRUTH + "cphi_IP.npz")
    m_tt_matrix = precision_load(GROUND_TRUTH + "m_tt_IP.npz")
    m_te_matrix = precision_load(GROUND_TRUTH + "m_te_IP.npz")
    m_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_IP.npz")
    m_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_IP.npz")
    b_tt_matrix = precision_load(GROUND_TRUTH + "b_tt_IP.npz")
    b_te_matrix = precision_load(GROUND_TRUTH + "b_te_IP.npz")
    b_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_IP.npz")
    b_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_IP.npz")
    mixing_d_tt_matrix = precision_load(GROUND_TRUTH + "d_tt_IP.npz")
    mixing_d_te_matrix = precision_load(GROUND_TRUTH + "d_te_IP.npz")
    mixing_d_ee_matrix = precision_load(GROUND_TRUTH + "d_ee_IP.npz")
    mixing_d_bb_matrix = precision_load(GROUND_TRUTH + "d_bb_IP.npz")
    mixing_g_matrix = precision_load(GROUND_TRUTH + "g_I.npz")
    mixed_grad_phi_matrix = precision_load(GROUND_TRUTH + "mixed_grad_phi_IP.npz")

    #convert ground truth data matrices into field and operator objects
    data = FlatS02(
        scalar_matrix = data_t_matrix,
        polar_matrix_1 = data_e_matrix,
        polar_matrix_2 = data_b_matrix,
    )
    field = FlatS02(
        scalar_matrix = unlensed_t_matrix,
        polar_matrix_1 = unlensed_e_matrix,
        polar_matrix_2 = unlensed_b_matrix,
    )
    phi = FlatS0(scalar_matrix = phi_matrix)
    mixed_grad_phi_ground = FlatS0(scalar_matrix = mixed_grad_phi_matrix)

    noise_covariance = BlockTEB(
        matrix_TT = cn_tt_matrix,
        matrix_TE = cn_te_matrix,
        matrix_ET = cn_te_matrix,
        matrix_EE = cn_ee_matrix,
        matrix_BB = cn_bb_matrix,
    )
    phi_covariance = DiagonalScalar(scalar_matrix = cphi_matrix)
    field_covariance = BlockTEB(
        matrix_TT = cf_tt_matrix,
        matrix_TE = cf_te_matrix,
        matrix_ET = cf_te_matrix,
        matrix_EE = cf_ee_matrix,
        matrix_BB = cf_bb_matrix,
    )
    mask = BlockTEB(
        matrix_TT = m_tt_matrix,
        matrix_TE = m_te_matrix,
        matrix_ET = m_te_matrix,
        matrix_EE = m_ee_matrix,
        matrix_BB = m_bb_matrix,
    )
    beam = BlockTEB(
        matrix_TT = b_tt_matrix,
        matrix_TE = b_te_matrix,
        matrix_ET = b_te_matrix,
        matrix_EE = b_ee_matrix,
        matrix_BB = b_bb_matrix,
    )
    mixing_d = BlockTEB(
        matrix_TT = mixing_d_tt_matrix,
        matrix_TE = mixing_d_te_matrix,
        matrix_ET = mixing_d_te_matrix,
        matrix_EE = mixing_d_ee_matrix,
        matrix_BB = mixing_d_bb_matrix,
    )
    mixing_g = DiagonalScalar(scalar_matrix = mixing_g_matrix)

    mixed_field, mixed_phi = mix(field, phi, mixing_d, mixing_g)
    mixed_grad_phi_predict = mixed_grad_phi_logpdf(mixed_field, mixed_phi, data, noise_covariance,
                                                    phi_covariance, field_covariance, mask, beam, mixing_d, mixing_g)

    SUB_FOLDER = "gradients/intensity_and_polarization/mixed_grad_phi_logpdf/"

    _plot_field = mixed_grad_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Predict")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Predict.png", bbox_inches="tight")
    plt.close()

    _plot_field = mixed_grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = mixed_grad_phi_predict - mixed_grad_phi_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Mixed Grad Phi Diff")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Diff.png", bbox_inches="tight")
    plt.close()

    _cc_f1 = fourier(mixed_grad_phi_ground) if mixed_grad_phi_ground.basis != Basis.FOURIER else mixed_grad_phi_ground
    _cc_f2 = fourier(mixed_grad_phi_predict) if mixed_grad_phi_predict.basis != Basis.FOURIER else mixed_grad_phi_predict
    ell, gphi_cross_gphi = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, gphi_cross_gphi, label = "PP")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Mixed Grad Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Mixed Grad Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(mixed_grad_phi_ground, mixed_grad_phi_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    assert jnp.mean(gphi_cross_gphi) > MIN_AVG_CORRELATION
