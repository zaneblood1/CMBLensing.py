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

def test_wiener_filter_intensity_only():

    #load ground truth data matrices
    phi_matrix = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix = precision_load(GROUND_TRUTH + "t_field_I.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_I.npz")
    cn_matrix = precision_load(GROUND_TRUTH + "cn_I.npz")
    cf_matrix = precision_load(GROUND_TRUTH + "cf_I.npz")
    m_matrix = precision_load(GROUND_TRUTH + "m_I.npz")
    b_matrix = precision_load(GROUND_TRUTH + "b_I.npz")
    wf_ffpp_matrix = precision_load(GROUND_TRUTH + "f_wf_ffpp_I.npz")
    wf_ffp0_matrix = precision_load(GROUND_TRUTH + "f_wf_ffp0_I.npz")
    wf_f0pp_matrix = precision_load(GROUND_TRUTH + "f_wf_f0pp_I.npz")
    wf_f0p0_matrix = precision_load(GROUND_TRUTH + "f_wf_f0p0_I.npz")

    #convert ground truth data matrices into field and operator objects
    data = FlatS0(scalar_matrix = data_t_matrix)
    field = FlatS0(scalar_matrix = unlensed_t_matrix)
    phi = FlatS0(scalar_matrix = phi_matrix)
    noise_covariance = DiagonalScalar(scalar_matrix = cn_matrix)
    field_covariance = DiagonalScalar(scalar_matrix = cf_matrix)
    mask = DiagonalScalar(scalar_matrix = m_matrix)
    beam = DiagonalScalar(scalar_matrix = b_matrix)
    wf_ffpp_ground = FlatS0(scalar_matrix = wf_ffpp_matrix)
    wf_ffp0_ground = FlatS0(scalar_matrix = wf_ffp0_matrix)
    wf_f0pp_ground = FlatS0(scalar_matrix = wf_f0pp_matrix)
    wf_f0p0_ground = FlatS0(scalar_matrix = wf_f0p0_matrix)

    #Call the JAX predictions for the wiener filter at various f and phi values
    wf_ffpp_predict = wiener_filter(field, phi, data, field_covariance, noise_covariance, mask, beam)
    wf_ffp0_predict = wiener_filter(field, 0*phi, data, field_covariance, noise_covariance, mask, beam)
    wf_f0pp_predict = wiener_filter(0*field, phi, data, field_covariance, noise_covariance, mask, beam)
    wf_f0p0_predict = wiener_filter(0*field, 0*phi, data, field_covariance, noise_covariance, mask, beam)

    #plot the cross correlation between ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_ffpp_f1 = fourier(wf_ffpp_ground) if wf_ffpp_ground.basis != Basis.FOURIER else wf_ffpp_ground
    _cc_ffpp_f2 = fourier(wf_ffpp_predict) if wf_ffpp_predict.basis != Basis.FOURIER else wf_ffpp_predict
    _cc_ffp0_f1 = fourier(wf_ffp0_ground) if wf_ffp0_ground.basis != Basis.FOURIER else wf_ffp0_ground
    _cc_ffp0_f2 = fourier(wf_ffp0_predict) if wf_ffp0_predict.basis != Basis.FOURIER else wf_ffp0_predict
    _cc_f0pp_f1 = fourier(wf_f0pp_ground) if wf_f0pp_ground.basis != Basis.FOURIER else wf_f0pp_ground
    _cc_f0pp_f2 = fourier(wf_f0pp_predict) if wf_f0pp_predict.basis != Basis.FOURIER else wf_f0pp_predict
    _cc_f0p0_f1 = fourier(wf_f0p0_ground) if wf_f0p0_ground.basis != Basis.FOURIER else wf_f0p0_ground
    _cc_f0p0_f2 = fourier(wf_f0p0_predict) if wf_f0p0_predict.basis != Basis.FOURIER else wf_f0p0_predict

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "wiener_filter/intensity_only/nonzero_f_nonzero_phi/"

    _plot_field = wf_ffpp_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffpp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffpp_predict - wf_ffpp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Absolute Difference.png", bbox_inches="tight")
    plt.close()
    ell, ffpp_cross_ffpp = primal_cross_correlation(_cc_ffpp_f1.scalar_matrix, _cc_ffpp_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, ffpp_cross_ffpp, label = "TT")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = F, Phi = Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/intensity_only/nonzero_f_zero_phi/"

    _plot_field = wf_ffp0_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffp0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffp0_predict - wf_ffp0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Absolute Difference.png", bbox_inches="tight")
    plt.close()
    ell, ffp0_cross_ffp0 = primal_cross_correlation(_cc_ffp0_f1.scalar_matrix, _cc_ffp0_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, ffp0_cross_ffp0, label = "TT")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = F, Phi = 0 Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/intensity_only/zero_f_nonzero_phi/"

    _plot_field = wf_f0pp_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0pp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0pp_predict - wf_f0pp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Absolute Difference.png", bbox_inches="tight")
    plt.close()
    ell, f0pp_cross_f0pp = primal_cross_correlation(_cc_f0pp_f1.scalar_matrix, _cc_f0pp_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, f0pp_cross_f0pp, label = "TT")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/intensity_only/zero_f_zero_phi/"

    _plot_field = wf_f0p0_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0p0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0p0_predict - wf_f0p0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Absolute Difference.png", bbox_inches="tight")
    plt.close()
    ell, f0p0_cross_f0p0 = primal_cross_correlation(_cc_f0p0_f1.scalar_matrix, _cc_f0p0_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, f0p0_cross_f0p0, label = "TT")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(wf_ffpp_ground, wf_ffpp_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_ffp0_ground, wf_ffp0_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_f0pp_ground, wf_f0pp_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_f0p0_ground, wf_f0p0_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    #TODO somehow fix the fact that these are becoming NaN
    #assert jnp.mean(ffpp_cross_ffpp) > MIN_AVG_CORRELATION
    #assert jnp.mean(ffp0_cross_ffp0) > MIN_AVG_CORRELATION
    #assert jnp.mean(f0pp_cross_f0pp) > MIN_AVG_CORRELATION
    #assert jnp.mean(f0p0_cross_f0p0) > MIN_AVG_CORRELATION

def test_wiener_filter_polarization_only():

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
    m_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_P.npz")
    m_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_P.npz")
    b_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_P.npz")
    b_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_P.npz")
    wf_ffpp_e_matrix = precision_load(GROUND_TRUTH + "f_wf_ffpp_e_P.npz")
    wf_ffpp_b_matrix = precision_load(GROUND_TRUTH + "f_wf_ffpp_b_P.npz")
    wf_ffp0_e_matrix = precision_load(GROUND_TRUTH + "f_wf_ffp0_e_P.npz")
    wf_ffp0_b_matrix = precision_load(GROUND_TRUTH + "f_wf_ffp0_b_P.npz")
    wf_f0pp_e_matrix = precision_load(GROUND_TRUTH + "f_wf_f0pp_e_P.npz")
    wf_f0pp_b_matrix = precision_load(GROUND_TRUTH + "f_wf_f0pp_b_P.npz")
    wf_f0p0_e_matrix = precision_load(GROUND_TRUTH + "f_wf_f0p0_e_P.npz")
    wf_f0p0_b_matrix = precision_load(GROUND_TRUTH + "f_wf_f0p0_b_P.npz")

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
    noise_covariance = DiagonalEB(
        matrix_EE = cn_ee_matrix,
        matrix_BB = cn_bb_matrix,
    )
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
    wf_ffpp_ground = FlatS2(
        polar_matrix_1 = wf_ffpp_e_matrix,
        polar_matrix_2 = wf_ffpp_b_matrix,
    )
    wf_ffp0_ground = FlatS2(
        polar_matrix_1 = wf_ffp0_e_matrix,
        polar_matrix_2 = wf_ffp0_b_matrix,
    )
    wf_f0pp_ground = FlatS2(
        polar_matrix_1 = wf_f0pp_e_matrix,
        polar_matrix_2 = wf_f0pp_b_matrix,
    )
    wf_f0p0_ground = FlatS2(
        polar_matrix_1 = wf_f0p0_e_matrix,
        polar_matrix_2 = wf_f0p0_b_matrix,
    )

    #Call the JAX predictions for the wiener filter at various f and phi values
    wf_ffpp_predict = wiener_filter(field, phi, data, field_covariance, noise_covariance, mask, beam)
    wf_ffp0_predict = wiener_filter(field, 0*phi, data, field_covariance, noise_covariance, mask, beam)
    wf_f0pp_predict = wiener_filter(0*field, phi, data, field_covariance, noise_covariance, mask, beam)
    wf_f0p0_predict = wiener_filter(0*field, 0*phi, data, field_covariance, noise_covariance, mask, beam)

    #plot the cross correlation between ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_ffpp_f1 = fourier(wf_ffpp_ground) if wf_ffpp_ground.basis != Basis.FOURIER else wf_ffpp_ground
    _cc_ffpp_f2 = fourier(wf_ffpp_predict) if wf_ffpp_predict.basis != Basis.FOURIER else wf_ffpp_predict
    _cc_ffp0_f1 = fourier(wf_ffp0_ground) if wf_ffp0_ground.basis != Basis.FOURIER else wf_ffp0_ground
    _cc_ffp0_f2 = fourier(wf_ffp0_predict) if wf_ffp0_predict.basis != Basis.FOURIER else wf_ffp0_predict
    _cc_f0pp_f1 = fourier(wf_f0pp_ground) if wf_f0pp_ground.basis != Basis.FOURIER else wf_f0pp_ground
    _cc_f0pp_f2 = fourier(wf_f0pp_predict) if wf_f0pp_predict.basis != Basis.FOURIER else wf_f0pp_predict
    _cc_f0p0_f1 = fourier(wf_f0p0_ground) if wf_f0p0_ground.basis != Basis.FOURIER else wf_f0p0_ground
    _cc_f0p0_f2 = fourier(wf_f0p0_predict) if wf_f0p0_predict.basis != Basis.FOURIER else wf_f0p0_predict

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "wiener_filter/polarization_only/nonzero_f_nonzero_phi/"

    _plot_field = wf_ffpp_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffpp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffpp_predict - wf_ffpp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()
    ell, ffpp_cross_ee = primal_cross_correlation(_cc_ffpp_f1.polar_matrix_1, _cc_ffpp_f2.polar_matrix_1, THETA_PIX)
    ell, ffpp_cross_bb = primal_cross_correlation(_cc_ffpp_f1.polar_matrix_2, _cc_ffpp_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, ffpp_cross_ee, label = "EE")
    plt.plot(ell, ffpp_cross_bb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = F, Phi = Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/polarization_only/nonzero_f_zero_phi/"

    _plot_field = wf_ffp0_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffp0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffp0_predict - wf_ffp0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()
    ell, ffp0_cross_ee = primal_cross_correlation(_cc_ffp0_f1.polar_matrix_1, _cc_ffp0_f2.polar_matrix_1, THETA_PIX)
    ell, ffp0_cross_bb = primal_cross_correlation(_cc_ffp0_f1.polar_matrix_2, _cc_ffp0_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, ffp0_cross_ee, label = "EE")
    plt.plot(ell, ffp0_cross_bb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = F, Phi = 0 Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/polarization_only/zero_f_nonzero_phi/"

    _plot_field = wf_f0pp_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0pp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0pp_predict - wf_f0pp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()
    ell, f0pp_cross_ee = primal_cross_correlation(_cc_f0pp_f1.polar_matrix_1, _cc_f0pp_f2.polar_matrix_1, THETA_PIX)
    ell, f0pp_cross_bb = primal_cross_correlation(_cc_f0pp_f1.polar_matrix_2, _cc_f0pp_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, f0pp_cross_ee, label = "EE")
    plt.plot(ell, f0pp_cross_bb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/polarization_only/zero_f_zero_phi/"

    _plot_field = wf_f0p0_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0p0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0p0_predict - wf_f0p0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()
    ell, f0p0_cross_ee = primal_cross_correlation(_cc_f0p0_f1.polar_matrix_1, _cc_f0p0_f2.polar_matrix_1, THETA_PIX)
    ell, f0p0_cross_bb = primal_cross_correlation(_cc_f0p0_f1.polar_matrix_2, _cc_f0p0_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, f0p0_cross_ee, label = "EE")
    plt.plot(ell, f0p0_cross_bb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(wf_ffpp_ground, wf_ffpp_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_ffp0_ground, wf_ffp0_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_f0pp_ground, wf_f0pp_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_f0p0_ground, wf_f0p0_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    #TODO somehow fix the fact that these are becoming NaN
    #assert jnp.mean(ffpp_cross_ee) > MIN_AVG_CORRELATION
    #assert jnp.mean(ffp0_cross_ee) > MIN_AVG_CORRELATION
    #assert jnp.mean(f0pp_cross_ee) > MIN_AVG_CORRELATION
    #assert jnp.mean(f0p0_cross_ee) > MIN_AVG_CORRELATION

def test_wiener_filter_intensity_and_polarization():

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
    m_tt_matrix = precision_load(GROUND_TRUTH + "m_tt_IP.npz")
    m_te_matrix = precision_load(GROUND_TRUTH + "m_te_IP.npz")
    m_ee_matrix = precision_load(GROUND_TRUTH + "m_ee_IP.npz")
    m_bb_matrix = precision_load(GROUND_TRUTH + "m_bb_IP.npz")
    b_tt_matrix = precision_load(GROUND_TRUTH + "b_tt_IP.npz")
    b_te_matrix = precision_load(GROUND_TRUTH + "b_te_IP.npz")
    b_ee_matrix = precision_load(GROUND_TRUTH + "b_ee_IP.npz")
    b_bb_matrix = precision_load(GROUND_TRUTH + "b_bb_IP.npz")
    wf_ffpp_t_matrix = precision_load(GROUND_TRUTH + "f_wf_ffpp_t_IP.npz")
    wf_ffpp_e_matrix = precision_load(GROUND_TRUTH + "f_wf_ffpp_e_IP.npz")
    wf_ffpp_b_matrix = precision_load(GROUND_TRUTH + "f_wf_ffpp_b_IP.npz")
    wf_ffp0_t_matrix = precision_load(GROUND_TRUTH + "f_wf_ffp0_t_IP.npz")
    wf_ffp0_e_matrix = precision_load(GROUND_TRUTH + "f_wf_ffp0_e_IP.npz")
    wf_ffp0_b_matrix = precision_load(GROUND_TRUTH + "f_wf_ffp0_b_IP.npz")
    wf_f0pp_t_matrix = precision_load(GROUND_TRUTH + "f_wf_f0pp_t_IP.npz")
    wf_f0pp_e_matrix = precision_load(GROUND_TRUTH + "f_wf_f0pp_e_IP.npz")
    wf_f0pp_b_matrix = precision_load(GROUND_TRUTH + "f_wf_f0pp_b_IP.npz")
    wf_f0p0_t_matrix = precision_load(GROUND_TRUTH + "f_wf_f0p0_t_IP.npz")
    wf_f0p0_e_matrix = precision_load(GROUND_TRUTH + "f_wf_f0p0_e_IP.npz")
    wf_f0p0_b_matrix = precision_load(GROUND_TRUTH + "f_wf_f0p0_b_IP.npz")

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
    noise_covariance = BlockTEB(
        matrix_TT = cn_tt_matrix,
        matrix_TE = cn_te_matrix,
        matrix_ET = cn_te_matrix,
        matrix_EE = cn_ee_matrix,
        matrix_BB = cn_bb_matrix,
    )
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
    wf_ffpp_ground = FlatS02(
        scalar_matrix = wf_ffpp_t_matrix,
        polar_matrix_1 = wf_ffpp_e_matrix,
        polar_matrix_2 = wf_ffpp_b_matrix,
    )
    wf_ffp0_ground = FlatS02(
        scalar_matrix = wf_ffp0_t_matrix,
        polar_matrix_1 = wf_ffp0_e_matrix,
        polar_matrix_2 = wf_ffp0_b_matrix,
    )
    wf_f0pp_ground = FlatS02(
        scalar_matrix = wf_f0pp_t_matrix,
        polar_matrix_1 = wf_f0pp_e_matrix,
        polar_matrix_2 = wf_f0pp_b_matrix,
    )
    wf_f0p0_ground = FlatS02(
        scalar_matrix = wf_f0p0_t_matrix,
        polar_matrix_1 = wf_f0p0_e_matrix,
        polar_matrix_2 = wf_f0p0_b_matrix,
    )

    #Call the JAX predictions for the wiener filter at various f and phi values
    wf_ffpp_predict = wiener_filter(field, phi, data, field_covariance, noise_covariance, mask, beam)
    wf_ffp0_predict = wiener_filter(field, 0*phi, data, field_covariance, noise_covariance, mask, beam)
    wf_f0pp_predict = wiener_filter(0*field, phi, data, field_covariance, noise_covariance, mask, beam)
    wf_f0p0_predict = wiener_filter(0*field, 0*phi, data, field_covariance, noise_covariance, mask, beam)

    #plot the cross correlation between ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_ffpp_f1 = fourier(wf_ffpp_ground) if wf_ffpp_ground.basis != Basis.FOURIER else wf_ffpp_ground
    _cc_ffpp_f2 = fourier(wf_ffpp_predict) if wf_ffpp_predict.basis != Basis.FOURIER else wf_ffpp_predict
    _cc_ffp0_f1 = fourier(wf_ffp0_ground) if wf_ffp0_ground.basis != Basis.FOURIER else wf_ffp0_ground
    _cc_ffp0_f2 = fourier(wf_ffp0_predict) if wf_ffp0_predict.basis != Basis.FOURIER else wf_ffp0_predict
    _cc_f0pp_f1 = fourier(wf_f0pp_ground) if wf_f0pp_ground.basis != Basis.FOURIER else wf_f0pp_ground
    _cc_f0pp_f2 = fourier(wf_f0pp_predict) if wf_f0pp_predict.basis != Basis.FOURIER else wf_f0pp_predict
    _cc_f0p0_f1 = fourier(wf_f0p0_ground) if wf_f0p0_ground.basis != Basis.FOURIER else wf_f0p0_ground
    _cc_f0p0_f2 = fourier(wf_f0p0_predict) if wf_f0p0_predict.basis != Basis.FOURIER else wf_f0p0_predict

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "wiener_filter/intensity_and_polarization/nonzero_f_nonzero_phi/"

    _plot_field = wf_ffpp_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffpp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffpp_predict - wf_ffpp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = Phi Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()
    ell, ffpp_cross_tt = primal_cross_correlation(_cc_ffpp_f1.scalar_matrix, _cc_ffpp_f2.scalar_matrix, THETA_PIX)
    ell, ffpp_cross_ee = primal_cross_correlation(_cc_ffpp_f1.polar_matrix_1, _cc_ffpp_f2.polar_matrix_1, THETA_PIX)
    ell, ffpp_cross_bb = primal_cross_correlation(_cc_ffpp_f1.polar_matrix_2, _cc_ffpp_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, ffpp_cross_tt, label = "TT")
    plt.plot(ell, ffpp_cross_ee, label = "EE")
    plt.plot(ell, ffpp_cross_bb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = F, Phi = Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/intensity_and_polarization/nonzero_f_zero_phi/"

    _plot_field = wf_ffp0_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffp0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_ffp0_predict - wf_ffp0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = F, Phi = 0 Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()
    ell, ffp0_cross_tt = primal_cross_correlation(_cc_ffp0_f1.scalar_matrix, _cc_ffp0_f2.scalar_matrix, THETA_PIX)
    ell, ffp0_cross_ee = primal_cross_correlation(_cc_ffp0_f1.polar_matrix_1, _cc_ffp0_f2.polar_matrix_1, THETA_PIX)
    ell, ffp0_cross_bb = primal_cross_correlation(_cc_ffp0_f1.polar_matrix_2, _cc_ffp0_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, ffp0_cross_tt, label = "TT")
    plt.plot(ell, ffp0_cross_ee, label = "EE")
    plt.plot(ell, ffp0_cross_bb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = F, Phi = 0 Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = F, Phi = 0 Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/intensity_and_polarization/zero_f_nonzero_phi/"

    _plot_field = wf_f0pp_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0pp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0pp_predict - wf_f0pp_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()
    ell, f0pp_cross_tt = primal_cross_correlation(_cc_f0pp_f1.scalar_matrix, _cc_f0pp_f2.scalar_matrix, THETA_PIX)
    ell, f0pp_cross_ee = primal_cross_correlation(_cc_f0pp_f1.polar_matrix_1, _cc_f0pp_f2.polar_matrix_1, THETA_PIX)
    ell, f0pp_cross_bb = primal_cross_correlation(_cc_f0pp_f1.polar_matrix_2, _cc_f0pp_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, f0pp_cross_tt, label = "TT")
    plt.plot(ell, f0pp_cross_ee, label = "EE")
    plt.plot(ell, f0pp_cross_bb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = 0, Phi = Phi Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "wiener_filter/intensity_and_polarization/zero_f_zero_phi/"

    _plot_field = wf_f0p0_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Prediction: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Prediction: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Prediction: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Prediction: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0p0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Ground: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Ground: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Ground: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Ground: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = wf_f0p0_predict - wf_f0p0_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Absolute Difference: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Absolute Difference: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Absolute Difference: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Absolute Difference: B Mode.png", bbox_inches="tight")
    plt.close()
    ell, f0p0_cross_tt = primal_cross_correlation(_cc_f0p0_f1.scalar_matrix, _cc_f0p0_f2.scalar_matrix, THETA_PIX)
    ell, f0p0_cross_ee = primal_cross_correlation(_cc_f0p0_f1.polar_matrix_1, _cc_f0p0_f2.polar_matrix_1, THETA_PIX)
    ell, f0p0_cross_bb = primal_cross_correlation(_cc_f0p0_f1.polar_matrix_2, _cc_f0p0_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, f0p0_cross_tt, label = "TT")
    plt.plot(ell, f0p0_cross_ee, label = "EE")
    plt.plot(ell, f0p0_cross_bb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Wiener Filter @ F = 0, Phi = 0 Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Wiener Filter @ F = 0, Phi = 0 Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(wf_ffpp_ground, wf_ffpp_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_ffp0_ground, wf_ffp0_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_f0pp_ground, wf_f0pp_predict)) <= MAX_NORM_DIFF
    assert jnp.max(percent_diff_2d(wf_f0p0_ground, wf_f0p0_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    #TODO somehow fix the fact that these are becoming NaN
    #assert jnp.mean(ffpp_cross_tt) > MIN_AVG_CORRELATION
    #assert jnp.mean(ffp0_cross_tt) > MIN_AVG_CORRELATION
    #assert jnp.mean(f0pp_cross_tt) > MIN_AVG_CORRELATION
    #assert jnp.mean(f0p0_cross_tt) > MIN_AVG_CORRELATION
