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

#constants relating to the ground truth data
GROUND_TRUTH = os.getcwd() + "/tests/ground_truth_data/"
FIGURE_PATH = os.getcwd() + "/tests/test_generated_figures/"
with open(GROUND_TRUTH + "n_side.txt", "r") as file:
    N_SIDE = int(file.read().strip())
with open(GROUND_TRUTH + "theta_pix.txt", "r") as file:
    THETA_PIX = int(file.read().strip())
PIX_WIDTH = float(jnp.deg2rad(THETA_PIX / ARCMIN_PER_DEGREE))

#threshold values for unit tests
MAX_NORM_DIFF = 1 
MIN_AVG_CORRELATION = 0

def test_forward_lensing_intensity_only():

    #load ground truth unlensed intensity field and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_field_I.npz")
    #load the ground truth lensed intensity field
    lensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_lensed_field_I.npz")

    #convert the matrices into fields
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    unlensed_t = map(FlatS0(scalar_matrix = unlensed_t_matrix_ground))
    lensed_t_ground = map(FlatS0(scalar_matrix = lensed_t_matrix_ground))

    #run the unlensed field through the jax lenseflow
    #NOTE lensing MUST be done in Map space / real space
    lensed_t_predict = lense_flow(unlensed_t, phi)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/intensity_only/forward_lensing/"

    _plot_field = lensed_t_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Lensed Field Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = lensed_t_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Lensed Field Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = lensed_t_predict - lensed_t_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Lensed Field Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Absolute Difference.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between lensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = fourier(lensed_t_ground)
    _cc_f2 = fourier(lensed_t_predict)
    ell, lt_cross_lt = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, lt_cross_lt, label = "TT")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    lensed_t_perc_diff = percent_diff_2d(lensed_t_ground, lensed_t_predict)
    assert lensed_t_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_lt = jnp.mean(lt_cross_lt)
    assert avg_correlation_lt > MIN_AVG_CORRELATION

def test_inverse_lensing_intensity_only():
    #load ground truth unlensed intensity field and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_field_I.npz")
    #load the ground truth lensed intensity field
    lensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_lensed_field_I.npz")

    #convert the matrices into fields
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    unlensed_t_ground = map(FlatS0(scalar_matrix = unlensed_t_matrix_ground))
    lensed_t_ground = map(FlatS0(scalar_matrix = lensed_t_matrix_ground))

    #run the lensed field through the jax inverse lenseflow
    unlensed_t_predict = lense_flow(lensed_t_ground, phi, direction = INVERSE_LENSE)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/intensity_only/inverse_lensing/"

    _plot_field = unlensed_t_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Lensed Field Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_t_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Lensed Field Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_t_predict - unlensed_t_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Lensed Field Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Absolute Difference.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between unlensed ground and predict
    _cc_f1 = fourier(unlensed_t_ground)
    _cc_f2 = fourier(unlensed_t_predict)
    ell, t_cross_t = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, t_cross_t, label = "TT")
    plt.title("Inverse Lensed Field Cross Correlation")
    plt.legend()
    plt.xlabel("ell")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    unlensed_t_perc_diff = percent_diff_2d(unlensed_t_ground, unlensed_t_predict)
    assert unlensed_t_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_t = jnp.mean(t_cross_t)
    assert avg_correlation_t > MIN_AVG_CORRELATION

def test_adjoint_lensing_intensity_only():
    #load ground truth unlensed intensity field and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_field_I.npz")
    #load the ground truth adjoint lensed intensity field
    adjoint_lensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_adjoint_lensed_field_I.npz")

    #convert the matrices into fields
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    unlensed_t = map(FlatS0(scalar_matrix = unlensed_t_matrix_ground))
    adjoint_lensed_t_ground = map(FlatS0(scalar_matrix = adjoint_lensed_t_matrix_ground))

    #run the unlensed field through the jax forward adjoint lenseflow
    adjoint_lensed_t_predict = lense_flow(unlensed_t, phi, adjoint = True)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/intensity_only/adjoint_lensing/"

    _plot_field = adjoint_lensed_t_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = adjoint_lensed_t_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Ground.png", bbox_inches="tight")
    plt.close()
    
    _plot_field = adjoint_lensed_t_predict - adjoint_lensed_t_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Absolute Difference.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between adjoint lensed ground and predict
    _cc_f1 = fourier(adjoint_lensed_t_ground)
    _cc_f2 = fourier(adjoint_lensed_t_predict)
    ell, alt_cross_alt = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, alt_cross_alt, label = "TT")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Adjoint Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    adjoint_lensed_t_perc_diff = percent_diff_2d(adjoint_lensed_t_ground, adjoint_lensed_t_predict)
    assert adjoint_lensed_t_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_alt = jnp.mean(alt_cross_alt)
    assert avg_correlation_alt > MIN_AVG_CORRELATION

def test_inverse_adjoint_lensing_intensity_only():
    #load ground truth unlensed intensity field and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_field_I.npz")
    #load the ground truth adjoint lensed intensity field
    adjoint_lensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_adjoint_lensed_field_I.npz")

    #convert the matrices into fields
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    unlensed_t_ground = map(FlatS0(scalar_matrix = unlensed_t_matrix_ground))
    adjoint_lensed_t_ground = map(FlatS0(scalar_matrix = adjoint_lensed_t_matrix_ground))

    #run the adjoint lensed field through the jax inverse adjoint lenseflow
    unlensed_t_predict = lense_flow(adjoint_lensed_t_ground, phi, direction = INVERSE_LENSE, adjoint = True)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/intensity_only/inverse_adjoint_lensing/"

    _plot_field = unlensed_t_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Prediction")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Prediction.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_t_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Ground")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Ground.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_t_predict - unlensed_t_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Absolute Difference")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Absolute Difference.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between unlensed ground and prediction
    _cc_f1 = fourier(unlensed_t_ground)
    _cc_f2 = fourier(unlensed_t_predict)
    ell, t_cross_t = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, t_cross_t, label = "TT")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Inverse Adjoint Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    unlensed_t_perc_diff = percent_diff_2d(unlensed_t_ground, unlensed_t_predict)
    assert unlensed_t_perc_diff <= MAX_NORM_DIFF

def test_forward_lensing_polarization_only():

    #load ground truth unlensed polarization fields and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_P.npz")
    unlensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_field_P.npz")
    unlensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_field_P.npz")
    #load the ground truth lensed polarization fields
    lensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_lensed_field_P.npz")
    lensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_lensed_field_P.npz")

    #convert the matrices into fields and convert EB -> QU -> map for lensing
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    unlensed_qu = map(eb2qu(FlatS2(
        polar_matrix_1 = unlensed_e_matrix_ground,
        polar_matrix_2 = unlensed_b_matrix_ground,
    )))
    lensed_qu_ground = map(eb2qu(FlatS2(
        polar_matrix_1 = lensed_e_matrix_ground,
        polar_matrix_2 = lensed_b_matrix_ground,
    )))

    #run the unlensed fields through the jax lenseflow
    #NOTE lensing MUST be done in Map space / real space
    lensed_qu_predict = lense_flow(unlensed_qu, phi)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/polarization_only/forward_lensing/"

    _plot_field = lensed_qu_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Lensed Field Prediction: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Prediction: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Lensed Field Prediction: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Prediction: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = lensed_qu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Lensed Field Ground: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Ground: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Lensed Field Ground: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Ground: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = lensed_qu_predict - lensed_qu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Lensed Field Absolute Difference: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Absolute Difference: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Lensed Field Absolute Difference: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Absolute Difference: U Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between lensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = qu2eb(fourier(lensed_qu_ground))
    _cc_f2 = qu2eb(fourier(lensed_qu_predict))
    ell, le_cross_le = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, lb_cross_lb = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, le_cross_le, label = "EE")
    plt.plot(ell, lb_cross_lb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    lensed_perc_diff = jnp.max(percent_diff_2d(lensed_qu_ground, lensed_qu_predict))
    assert lensed_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_le = jnp.mean(le_cross_le)
    assert avg_correlation_le > MIN_AVG_CORRELATION
    avg_correlation_lb = jnp.mean(lb_cross_lb)
    assert avg_correlation_lb > MIN_AVG_CORRELATION

def test_inverse_lensing_polarization_only():
    #load ground truth unlensed polarization fields and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_P.npz")
    unlensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_field_P.npz")
    unlensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_field_P.npz")
    #load the ground truth lensed polarization fields
    lensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_lensed_field_P.npz")
    lensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_lensed_field_P.npz")

    #convert the matrices into fields and convert EB -> QU -> map for lensing
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    lensed_qu = map(eb2qu(FlatS2(
        polar_matrix_1 = lensed_e_matrix_ground,
        polar_matrix_2 = lensed_b_matrix_ground,
    )))
    unlensed_qu_ground = map(eb2qu(FlatS2(
        polar_matrix_1 = unlensed_e_matrix_ground,
        polar_matrix_2 = unlensed_b_matrix_ground,
    )))

    #run the lensed field through the jax inverse lenseflow
    unlensed_qu_predict = lense_flow(lensed_qu, phi, direction = INVERSE_LENSE)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/polarization_only/inverse_lensing/"

    _plot_field = unlensed_qu_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Lensed Field Prediction: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Prediction: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Lensed Field Prediction: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Prediction: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_qu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Lensed Field Ground: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Ground: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Lensed Field Ground: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Ground: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_qu_predict - unlensed_qu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Lensed Field Absolute Difference: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Absolute Difference: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Lensed Field Absolute Difference: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Absolute Difference: U Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between unlensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = qu2eb(fourier(unlensed_qu_ground))
    _cc_f2 = qu2eb(fourier(unlensed_qu_predict))
    ell, e_cross_e = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, b_cross_b = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, e_cross_e, label = "EE")
    plt.plot(ell, b_cross_b, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Inverse Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    unlensed_perc_diff = jnp.max(percent_diff_2d(unlensed_qu_ground, unlensed_qu_predict))
    assert unlensed_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_e = jnp.mean(e_cross_e)
    assert avg_correlation_e > MIN_AVG_CORRELATION
    avg_correlation_b = jnp.mean(b_cross_b)
    assert avg_correlation_b > MIN_AVG_CORRELATION

def test_adjoint_lensing_polarization_only():
    #load ground truth unlensed polarization fields and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_P.npz")
    unlensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_field_P.npz")
    unlensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_field_P.npz")
    #load the ground truth adjoint lensed polarization fields
    adjoint_lensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_adjoint_lensed_field_P.npz")
    adjoint_lensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_adjoint_lensed_field_P.npz")

    #convert the matrices into fields and convert EB -> QU -> map for lensing
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    unlensed_qu = map(eb2qu(FlatS2(
        polar_matrix_1 = unlensed_e_matrix_ground,
        polar_matrix_2 = unlensed_b_matrix_ground,
    )))
    adjoint_lensed_qu_ground = map(eb2qu(FlatS2(
        polar_matrix_1 = adjoint_lensed_e_matrix_ground,
        polar_matrix_2 = adjoint_lensed_b_matrix_ground,
    )))

    #run the unlensed field through the jax forward adjoint lenseflow
    adjoint_lensed_qu_predict = lense_flow(unlensed_qu, phi, adjoint = True)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/polarization_only/adjoint_lensing/"

    _plot_field = adjoint_lensed_qu_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Prediction: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Prediction: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Prediction: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Prediction: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = adjoint_lensed_qu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Ground: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Ground: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Ground: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Ground: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = adjoint_lensed_qu_predict - adjoint_lensed_qu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Absolute Difference: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Absolute Difference: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Absolute Difference: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Absolute Difference: U Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between adjoint lensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = qu2eb(fourier(adjoint_lensed_qu_ground))
    _cc_f2 = qu2eb(fourier(adjoint_lensed_qu_predict))
    ell, ale_cross_ale = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, alb_cross_alb = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, ale_cross_ale, label = "EE")
    plt.plot(ell, alb_cross_alb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Adjoint Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    adjoint_lensed_perc_diff = jnp.max(percent_diff_2d(adjoint_lensed_qu_ground, adjoint_lensed_qu_predict))
    assert adjoint_lensed_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_ale = jnp.mean(ale_cross_ale)
    assert avg_correlation_ale > MIN_AVG_CORRELATION
    avg_correlation_alb = jnp.mean(alb_cross_alb)
    assert avg_correlation_alb > MIN_AVG_CORRELATION

def test_inverse_adjoint_lensing_polarization_only():
    #load ground truth unlensed polarization fields and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_P.npz")
    unlensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_field_P.npz")
    unlensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_field_P.npz")
    #load the ground truth adjoint lensed polarization fields
    adjoint_lensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_adjoint_lensed_field_P.npz")
    adjoint_lensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_adjoint_lensed_field_P.npz")

    #convert the matrices into fields and convert EB -> QU -> map for lensing
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    adjoint_lensed_qu = map(eb2qu(FlatS2(
        polar_matrix_1 = adjoint_lensed_e_matrix_ground,
        polar_matrix_2 = adjoint_lensed_b_matrix_ground,
    )))
    unlensed_qu_ground = map(eb2qu(FlatS2(
        polar_matrix_1 = unlensed_e_matrix_ground,
        polar_matrix_2 = unlensed_b_matrix_ground,
    )))

    #run the lensed field through the jax inverse adjoint lenseflow
    unlensed_qu_predict = lense_flow(adjoint_lensed_qu, phi, direction = INVERSE_LENSE, adjoint = True)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/polarization_only/inverse_adjoint_lensing/"

    _plot_field = unlensed_qu_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Prediction: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Prediction: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Prediction: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Prediction: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_qu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Ground: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Ground: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Ground: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Ground: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_qu_predict - unlensed_qu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Absolute Difference: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Absolute Difference: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Absolute Difference: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Absolute Difference: U Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between unlensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = qu2eb(fourier(unlensed_qu_ground))
    _cc_f2 = qu2eb(fourier(unlensed_qu_predict))
    ell, e_cross_e = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, b_cross_b = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, e_cross_e, label = "EE")
    plt.plot(ell, b_cross_b, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Inverse Adjoint Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    unlensed_perc_diff = jnp.max(percent_diff_2d(unlensed_qu_ground, unlensed_qu_predict))
    assert unlensed_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_e = jnp.mean(e_cross_e)
    assert avg_correlation_e > MIN_AVG_CORRELATION
    avg_correlation_b = jnp.mean(b_cross_b)
    assert avg_correlation_b > MIN_AVG_CORRELATION

def test_forward_lensing_intensity_and_polarization():

    #load ground truth unlensed intensity and polarization fields and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_IP.npz")
    unlensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_field_IP.npz")
    unlensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_field_IP.npz")
    unlensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_field_IP.npz")
    #load the ground truth lensed intensity and polarization fields
    lensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_lensed_field_IP.npz")
    lensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_lensed_field_IP.npz")
    lensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_lensed_field_IP.npz")

    #convert the matrices into fields and convert EB -> QU -> map for lensing
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    unlensed_tqu = map(eb2qu(FlatS02(
        scalar_matrix = unlensed_t_matrix_ground,
        polar_matrix_1 = unlensed_e_matrix_ground,
        polar_matrix_2 = unlensed_b_matrix_ground,
    )))
    lensed_tqu_ground = map(eb2qu(FlatS02(
        scalar_matrix = lensed_t_matrix_ground,
        polar_matrix_1 = lensed_e_matrix_ground,
        polar_matrix_2 = lensed_b_matrix_ground,
    )))

    #run the unlensed field through the jax lenseflow
    #NOTE lensing MUST be done in Map space / real space
    lensed_tqu_predict = lense_flow(unlensed_tqu, phi)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/intensity_and_polarization/forward_lensing/"

    _plot_field = lensed_tqu_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Lensed Field Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Lensed Field Prediction: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Prediction: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Lensed Field Prediction: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Prediction: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = lensed_tqu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Lensed Field Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Lensed Field Ground: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Ground: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Lensed Field Ground: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Ground: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = lensed_tqu_predict - lensed_tqu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Lensed Field Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Lensed Field Absolute Difference: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Absolute Difference: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Lensed Field Absolute Difference: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Absolute Difference: U Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between lensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = qu2eb(fourier(lensed_tqu_ground))
    _cc_f2 = qu2eb(fourier(lensed_tqu_predict))
    ell, lt_cross_lt = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _, le_cross_le = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, lb_cross_lb = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, lt_cross_lt, label = "TT")
    plt.plot(ell, le_cross_le, label = "EE")
    plt.plot(ell, lb_cross_lb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    lensed_tqu_perc_diff = jnp.max(percent_diff_2d(lensed_tqu_ground, lensed_tqu_predict))
    assert lensed_tqu_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_lt = jnp.mean(lt_cross_lt)
    assert avg_correlation_lt > MIN_AVG_CORRELATION
    avg_correlation_le = jnp.mean(le_cross_le)
    assert avg_correlation_le > MIN_AVG_CORRELATION
    avg_correlation_lb = jnp.mean(lb_cross_lb)
    assert avg_correlation_lb > MIN_AVG_CORRELATION

def test_inverse_lensing_intensity_and_polarization():
    #load ground truth unlensed intensity and polarization fields and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_IP.npz")
    unlensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_field_IP.npz")
    unlensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_field_IP.npz")
    unlensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_field_IP.npz")
    #load the ground truth lensed intensity and polarization fields
    lensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_lensed_field_IP.npz")
    lensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_lensed_field_IP.npz")
    lensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_lensed_field_IP.npz")

    #convert the matrices into fields and convert EB -> QU -> map for lensing
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    lensed_tqu = map(eb2qu(FlatS02(
        scalar_matrix = lensed_t_matrix_ground,
        polar_matrix_1 = lensed_e_matrix_ground,
        polar_matrix_2 = lensed_b_matrix_ground,
    )))
    unlensed_tqu_ground = map(eb2qu(FlatS02(
        scalar_matrix = unlensed_t_matrix_ground,
        polar_matrix_1 = unlensed_e_matrix_ground,
        polar_matrix_2 = unlensed_b_matrix_ground,
    )))

    #run the lensed field through the jax inverse lenseflow
    unlensed_tqu_predict = lense_flow(lensed_tqu, phi, direction = INVERSE_LENSE)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/intensity_and_polarization/inverse_lensing/"

    _plot_field = unlensed_tqu_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Lensed Field Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Lensed Field Prediction: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Prediction: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Lensed Field Prediction: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Prediction: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_tqu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Lensed Field Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Lensed Field Ground: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Ground: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Lensed Field Ground: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Ground: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_tqu_predict - unlensed_tqu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Lensed Field Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Lensed Field Absolute Difference: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Absolute Difference: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Lensed Field Absolute Difference: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Absolute Difference: U Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between unlensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = qu2eb(fourier(unlensed_tqu_ground))
    _cc_f2 = qu2eb(fourier(unlensed_tqu_predict))
    ell, t_cross_t = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _, e_cross_e = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, b_cross_b = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, t_cross_t, label = "TT")
    plt.plot(ell, e_cross_e, label = "EE")
    plt.plot(ell, b_cross_b, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Inverse Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    unlensed_tqu_perc_diff = jnp.max(percent_diff_2d(unlensed_tqu_ground, unlensed_tqu_predict))
    assert unlensed_tqu_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_t = jnp.mean(t_cross_t)
    assert avg_correlation_t > MIN_AVG_CORRELATION
    avg_correlation_e = jnp.mean(e_cross_e)
    assert avg_correlation_e > MIN_AVG_CORRELATION
    avg_correlation_b = jnp.mean(b_cross_b)
    assert avg_correlation_b > MIN_AVG_CORRELATION

def test_adjoint_lensing_intensity_and_polarization():
    #load ground truth unlensed intensity and polarization fields and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_IP.npz")
    unlensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_field_IP.npz")
    unlensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_field_IP.npz")
    unlensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_field_IP.npz")
    #load the ground truth adjoint lensed intensity and polarization fields
    adjoint_lensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_adjoint_lensed_field_IP.npz")
    adjoint_lensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_adjoint_lensed_field_IP.npz")
    adjoint_lensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_adjoint_lensed_field_IP.npz")

    #convert the matrices into fields and convert EB -> QU -> map for lensing
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    unlensed_tqu = map(eb2qu(FlatS02(
        scalar_matrix = unlensed_t_matrix_ground,
        polar_matrix_1 = unlensed_e_matrix_ground,
        polar_matrix_2 = unlensed_b_matrix_ground,
    )))
    adjoint_lensed_tqu_ground = map(eb2qu(FlatS02(
        scalar_matrix = adjoint_lensed_t_matrix_ground,
        polar_matrix_1 = adjoint_lensed_e_matrix_ground,
        polar_matrix_2 = adjoint_lensed_b_matrix_ground,
    )))

    #run the unlensed field through the jax forward adjoint lenseflow
    adjoint_lensed_tqu_predict = lense_flow(unlensed_tqu, phi, adjoint = True)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/intensity_and_polarization/adjoint_lensing/"

    _plot_field = adjoint_lensed_tqu_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Prediction: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Prediction: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Prediction: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Prediction: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = adjoint_lensed_tqu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Ground: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Ground: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Ground: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Ground: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = adjoint_lensed_tqu_predict - adjoint_lensed_tqu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Absolute Difference: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Absolute Difference: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Adjoint Lensed Field Absolute Difference: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Absolute Difference: U Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between adjoint lensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = qu2eb(fourier(adjoint_lensed_tqu_ground))
    _cc_f2 = qu2eb(fourier(adjoint_lensed_tqu_predict))
    ell, alt_cross_alt = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _, ale_cross_ale = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, alb_cross_alb = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, alt_cross_alt, label = "TT")
    plt.plot(ell, ale_cross_ale, label = "EE")
    plt.plot(ell, alb_cross_alb, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Adjoint Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Adjoint Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    adjoint_lensed_tqu_perc_diff = jnp.max(percent_diff_2d(adjoint_lensed_tqu_ground, adjoint_lensed_tqu_predict))
    assert adjoint_lensed_tqu_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_alt = jnp.mean(alt_cross_alt)
    assert avg_correlation_alt > MIN_AVG_CORRELATION
    avg_correlation_ale = jnp.mean(ale_cross_ale)
    assert avg_correlation_ale > MIN_AVG_CORRELATION
    avg_correlation_alb = jnp.mean(alb_cross_alb)
    assert avg_correlation_alb > MIN_AVG_CORRELATION

def test_inverse_adjoint_lensing_intensity_and_polarization():
    #load ground truth unlensed intensity and polarization fields and phi field
    phi_matrix_ground = precision_load(GROUND_TRUTH + "phi_IP.npz")
    unlensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_field_IP.npz")
    unlensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_field_IP.npz")
    unlensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_field_IP.npz")
    #load the ground truth adjoint lensed intensity and polarization fields
    adjoint_lensed_t_matrix_ground = precision_load(GROUND_TRUTH + "t_adjoint_lensed_field_IP.npz")
    adjoint_lensed_e_matrix_ground = precision_load(GROUND_TRUTH + "e_adjoint_lensed_field_IP.npz")
    adjoint_lensed_b_matrix_ground = precision_load(GROUND_TRUTH + "b_adjoint_lensed_field_IP.npz")

    #convert the matrices into fields and convert EB -> QU -> map for lensing
    phi = map(FlatS0(scalar_matrix = phi_matrix_ground))
    adjoint_lensed_tqu = map(eb2qu(FlatS02(
        scalar_matrix = adjoint_lensed_t_matrix_ground,
        polar_matrix_1 = adjoint_lensed_e_matrix_ground,
        polar_matrix_2 = adjoint_lensed_b_matrix_ground,
    )))
    unlensed_tqu_ground = map(eb2qu(FlatS02(
        scalar_matrix = unlensed_t_matrix_ground,
        polar_matrix_1 = unlensed_e_matrix_ground,
        polar_matrix_2 = unlensed_b_matrix_ground,
    )))

    #run the lensed field through the jax inverse adjoint lenseflow
    unlensed_tqu_predict = lense_flow(adjoint_lensed_tqu, phi, direction = INVERSE_LENSE, adjoint = True)

    #plot the absolute difference and cross correlation
    SUB_FOLDER = "lensing/intensity_and_polarization/inverse_adjoint_lensing/"

    _plot_field = unlensed_tqu_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Prediction: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Prediction: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Prediction: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Prediction: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Prediction: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Prediction: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_tqu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Ground: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Ground: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Ground: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Ground: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Ground: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Ground: U Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = unlensed_tqu_predict - unlensed_tqu_ground
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Absolute Difference: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Absolute Difference: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Absolute Difference: Q Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Absolute Difference: Q Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Inverse Adjoint Lensed Field Absolute Difference: U Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Absolute Difference: U Mode.png", bbox_inches="tight")
    plt.close()

    #plot the cross correlation between unlensed ground and predict
    #NOTE cross correlation is computed in fourier space
    _cc_f1 = qu2eb(fourier(unlensed_tqu_ground))
    _cc_f2 = qu2eb(fourier(unlensed_tqu_predict))
    ell, t_cross_t = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _, e_cross_e = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _, b_cross_b = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, t_cross_t, label = "TT")
    plt.plot(ell, e_cross_e, label = "EE")
    plt.plot(ell, b_cross_b, label = "BB")
    plt.legend()
    plt.xlabel("ell")
    plt.title("Inverse Adjoint Lensed Field Cross Correlation")
    plt.ylim([-2, 2])
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Inverse Adjoint Lensed Field Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    unlensed_tqu_perc_diff = jnp.max(percent_diff_2d(unlensed_tqu_ground, unlensed_tqu_predict))
    assert unlensed_tqu_perc_diff <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    avg_correlation_t = jnp.mean(t_cross_t)
    assert avg_correlation_t > MIN_AVG_CORRELATION
    avg_correlation_e = jnp.mean(e_cross_e)
    assert avg_correlation_e > MIN_AVG_CORRELATION
    avg_correlation_b = jnp.mean(b_cross_b)
    assert avg_correlation_b > MIN_AVG_CORRELATION

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

    logpdf_t_predict = logpdf(unlensed_field, phi, data,
                              noise_covariance, phi_covariance,
                              field_covariance, mask, beam)
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

    logpdf_eb_predict = logpdf(unlensed_eb_field, phi, data_eb_field,
                              noise_covariance, phi_covariance,
                              field_covariance, mask, beam)
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

    logpdf_teb_predict = logpdf(unlensed_teb_field, phi, data_teb_field,
                                noise_covariance, phi_covariance,
                                field_covariance, mask, beam)
    with open(GROUND_TRUTH + "logpdf_IP.txt", "r") as file:
        logpdf_teb_ground = float(file.read().strip())
    assert abs(logpdf_teb_predict - logpdf_teb_ground) < 1

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

def test_map_joint_intensity_only():

    #load ground truth data matrices
    phi_matrix = precision_load(GROUND_TRUTH + "phi_I.npz")
    unlensed_field_matrix = precision_load(GROUND_TRUTH + "t_field_I.npz")
    data_t_matrix = precision_load(GROUND_TRUTH + "data_t_field_I.npz")
    cn_matrix = precision_load(GROUND_TRUTH + "cn_I.npz")
    cf_matrix = precision_load(GROUND_TRUTH + "cf_I.npz")
    cphi_matrix = precision_load(GROUND_TRUTH + "cphi_I.npz")
    m_matrix = precision_load(GROUND_TRUTH + "m_I.npz")
    b_matrix = precision_load(GROUND_TRUTH + "b_I.npz")
    mixing_d_matrix = precision_load(GROUND_TRUTH + "d_I.npz")
    quadratic_estimate_matrix = precision_load(GROUND_TRUTH + "nphi_I.npz")
    julia_phi_predict_matrix = precision_load(GROUND_TRUTH + "phiJ_I.npz")
    julia_field_predict_matrix = precision_load(GROUND_TRUTH + "fJ_I.npz")

    #convert ground truth data matrices into field and operator objects
    data = FlatS0(scalar_matrix = data_t_matrix)
    simulated_field = FlatS0(scalar_matrix = unlensed_field_matrix)
    simulated_phi = FlatS0(scalar_matrix = phi_matrix)
    julia_phi_predict = FlatS0(scalar_matrix = julia_phi_predict_matrix)
    julia_field_predict = FlatS0(scalar_matrix = julia_field_predict_matrix)
    noise_covariance = DiagonalScalar(scalar_matrix = cn_matrix)
    phi_covariance = DiagonalScalar(scalar_matrix = cphi_matrix)
    field_covariance = DiagonalScalar(scalar_matrix = cf_matrix)
    mask = DiagonalScalar(scalar_matrix = m_matrix)
    beam = DiagonalScalar(scalar_matrix = b_matrix)
    mixing_d = DiagonalScalar(scalar_matrix = mixing_d_matrix)
    quadratic_estimate = DiagonalScalar(scalar_matrix = quadratic_estimate_matrix)

    #store fields and matrices inside of the dataset
    data_set = DataSetT(
        data = data,
        noise_covariance = noise_covariance,
        mixing_d = mixing_d,
        field_covariance = field_covariance,
        phi_covariance = phi_covariance,
        mask = mask,
        beam = beam,
        quadratic_estimate = quadratic_estimate
    )

    #Call the python version of the map_joint algorithm
    python_field_predict, python_phi_predict = map_joint(data_set, num_steps = 30)

    SUB_FOLDER = "map_joint/intensity_only/phi/"

    _plot_field = python_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Python MLE Phi")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Phi.png", bbox_inches="tight")
    plt.close()

    _plot_field = julia_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Julia MLE Phi")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Phi.png", bbox_inches="tight")
    plt.close()

    _plot_field = python_phi_predict - julia_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("MLE Phi Diff")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Phi Diff.png", bbox_inches="tight")
    plt.close()
    _cc_f1 = fourier(julia_phi_predict) if julia_phi_predict.basis != Basis.FOURIER else julia_phi_predict
    _cc_f2 = fourier(simulated_phi) if simulated_phi.basis != Basis.FOURIER else simulated_phi
    ell, phi_correlation_js = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(python_phi_predict) if python_phi_predict.basis != Basis.FOURIER else python_phi_predict
    _cc_f2 = fourier(simulated_phi) if simulated_phi.basis != Basis.FOURIER else simulated_phi
    _, phi_correlation_ps = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(julia_phi_predict) if julia_phi_predict.basis != Basis.FOURIER else julia_phi_predict
    _cc_f2 = fourier(python_phi_predict) if python_phi_predict.basis != Basis.FOURIER else python_phi_predict
    _, phi_correlation_jp = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, phi_correlation_js, label = "Julia x Sim")
    plt.plot(ell, phi_correlation_ps, label = "Python x Sim")
    plt.plot(ell, phi_correlation_jp, label = "Julia x Python")
    plt.title("Phi Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()
    
    SUB_FOLDER = "map_joint/intensity_only/fields/"

    _plot_field = python_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Python MLE Field: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Field: Temperature.png", bbox_inches="tight")
    plt.close()

    _plot_field = julia_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Julia MLE Field: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Field: Temperature.png", bbox_inches="tight")
    plt.close()

    _plot_field = python_field_predict - julia_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("MLE Field Diff: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Field Diff: Temperature.png", bbox_inches="tight")
    plt.close()
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    ell, t_correlation_js = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, t_correlation_ps = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _, t_correlation_jp = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, t_correlation_js, label = "Julia x Sim")
    plt.plot(ell, t_correlation_ps, label = "Python x Sim")
    plt.plot(ell, t_correlation_jp, label = "Julia x Python")
    plt.title("Temperature Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Temperature Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(julia_field_predict, python_field_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    #TODO handle NaN case here...
    # assert jnp.mean(t_correlation_jp) > MIN_AVG_CORRELATION

def test_map_joint_polarization_only():

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
    quadratic_estimate_matrix = precision_load(GROUND_TRUTH + "nphi_P.npz")
    julia_phi_predict_matrix = precision_load(GROUND_TRUTH + "phiJ_P.npz")
    julia_field_predict_e_matrix = precision_load(GROUND_TRUTH + "fJ_e_P.npz")
    julia_field_predict_b_matrix = precision_load(GROUND_TRUTH + "fJ_b_P.npz")

    #convert ground truth data matrices into field and operator objects
    data = FlatS2(
        polar_matrix_1 = data_e_matrix,
        polar_matrix_2 = data_b_matrix,
    )
    simulated_field = FlatS2(
        polar_matrix_1 = unlensed_e_matrix,
        polar_matrix_2 = unlensed_b_matrix,
    )
    simulated_phi = FlatS0(scalar_matrix = phi_matrix)
    julia_phi_predict = FlatS0(scalar_matrix = julia_phi_predict_matrix)
    julia_field_predict = FlatS2(
        polar_matrix_1 = julia_field_predict_e_matrix,
        polar_matrix_2 = julia_field_predict_b_matrix,
    )
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
    quadratic_estimate = DiagonalScalar(scalar_matrix = quadratic_estimate_matrix)

    #store fields and matrices inside of the dataset
    data_set = DataSetEB(
        data = data,
        noise_covariance = noise_covariance,
        mixing_d = mixing_d,
        field_covariance = field_covariance,
        phi_covariance = phi_covariance,
        mask = mask,
        beam = beam,
        quadratic_estimate = quadratic_estimate
    )

    #Call the python version of the map_joint algorithm
    python_field_predict, python_phi_predict = map_joint(data_set, num_steps = 30)

    SUB_FOLDER = "map_joint/polarization_only/phi/"

    _plot_field = python_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Python MLE Phi")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Phi.png", bbox_inches="tight")
    plt.close()

    _plot_field = julia_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Julia MLE Phi")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Phi.png", bbox_inches="tight")
    plt.close()

    _plot_field = python_phi_predict - julia_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("MLE Phi Diff")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Phi Diff.png", bbox_inches="tight")
    plt.close()
    _cc_f1 = fourier(julia_phi_predict) if julia_phi_predict.basis != Basis.FOURIER else julia_phi_predict
    _cc_f2 = fourier(simulated_phi) if simulated_phi.basis != Basis.FOURIER else simulated_phi
    ell, phi_correlation_js = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(python_phi_predict) if python_phi_predict.basis != Basis.FOURIER else python_phi_predict
    _cc_f2 = fourier(simulated_phi) if simulated_phi.basis != Basis.FOURIER else simulated_phi
    _, phi_correlation_ps = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(julia_phi_predict) if julia_phi_predict.basis != Basis.FOURIER else julia_phi_predict
    _cc_f2 = fourier(python_phi_predict) if python_phi_predict.basis != Basis.FOURIER else python_phi_predict
    _, phi_correlation_jp = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, phi_correlation_js, label = "Julia x Sim")
    plt.plot(ell, phi_correlation_ps, label = "Python x Sim")
    plt.plot(ell, phi_correlation_jp, label = "Julia x Python")
    plt.title("Phi Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "map_joint/polarization_only/fields/"

    _plot_field = python_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Python MLE Field: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Field: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Python MLE Field: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Field: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = julia_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Julia MLE Field: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Field: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Julia MLE Field: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Field: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = python_field_predict - julia_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("MLE Field Diff: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Field Diff: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("MLE Field Diff: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Field Diff: B Mode.png", bbox_inches="tight")
    plt.close()
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    ell, ee_correlation_js = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _cc_f1 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, ee_correlation_ps = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _, ee_correlation_jp = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, bb_correlation_js = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    _cc_f1 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, bb_correlation_ps = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _, bb_correlation_jp = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    plt.figure()
    plt.plot(ell, ee_correlation_js, label = "Julia x Sim EE")
    plt.plot(ell, ee_correlation_ps, label = "Python x Sim EE")
    plt.plot(ell, ee_correlation_jp, label = "Julia x Python EE")
    plt.title("E Mode Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "E Mode Cross Correlation.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(ell, bb_correlation_js, label = "Julia x Sim BB")
    plt.plot(ell, bb_correlation_ps, label = "Python x Sim BB")
    plt.plot(ell, bb_correlation_jp, label = "Julia x Python BB")
    plt.title("B Mode Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "B Mode Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(julia_field_predict, python_field_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    #TODO handle NaN case here...
    # assert jnp.mean(ee_correlation_jp) > MIN_AVG_CORRELATION
    # assert jnp.mean(bb_correlation_jp) > MIN_AVG_CORRELATION

def test_map_joint_intensity_and_polarization():

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
    quadratic_estimate_matrix = precision_load(GROUND_TRUTH + "nphi_IP.npz")
    julia_phi_predict_matrix = precision_load(GROUND_TRUTH + "phiJ_IP.npz")
    julia_field_predict_t_matrix = precision_load(GROUND_TRUTH + "fJ_t_IP.npz")
    julia_field_predict_e_matrix = precision_load(GROUND_TRUTH + "fJ_e_IP.npz")
    julia_field_predict_b_matrix = precision_load(GROUND_TRUTH + "fJ_b_IP.npz")

    #convert ground truth data matrices into field and operator objects
    data = FlatS02(
        scalar_matrix = data_t_matrix,
        polar_matrix_1 = data_e_matrix,
        polar_matrix_2 = data_b_matrix,
    )
    simulated_field = FlatS02(
        scalar_matrix = unlensed_t_matrix,
        polar_matrix_1 = unlensed_e_matrix,
        polar_matrix_2 = unlensed_b_matrix,
    )
    simulated_phi = FlatS0(scalar_matrix = phi_matrix)
    julia_phi_predict = FlatS0(scalar_matrix = julia_phi_predict_matrix)
    julia_field_predict = FlatS02(
        scalar_matrix = julia_field_predict_t_matrix,
        polar_matrix_1 = julia_field_predict_e_matrix,
        polar_matrix_2 = julia_field_predict_b_matrix,
    )
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
    quadratic_estimate = DiagonalScalar(scalar_matrix = quadratic_estimate_matrix)

    #store fields and matrices inside of the dataset
    data_set = DataSetTEB(
        data = data,
        noise_covariance = noise_covariance,
        mixing_d = mixing_d,
        field_covariance = field_covariance,
        phi_covariance = phi_covariance,
        mask = mask,
        beam = beam,
        quadratic_estimate = quadratic_estimate
    )

    #Call the python version of the map_joint algorithm
    python_field_predict, python_phi_predict = map_joint(data_set, num_steps = 30)

    SUB_FOLDER = "map_joint/intensity_and_polarization/phi/"

    _plot_field = python_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Python MLE Phi")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Phi.png", bbox_inches="tight")
    plt.close()

    _plot_field = julia_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Julia MLE Phi")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Phi.png", bbox_inches="tight")
    plt.close()

    _plot_field = python_phi_predict - julia_phi_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("MLE Phi Diff")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Phi Diff.png", bbox_inches="tight")
    plt.close()
    _cc_f1 = fourier(julia_phi_predict) if julia_phi_predict.basis != Basis.FOURIER else julia_phi_predict
    _cc_f2 = fourier(simulated_phi) if simulated_phi.basis != Basis.FOURIER else simulated_phi
    ell, phi_correlation_js = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(python_phi_predict) if python_phi_predict.basis != Basis.FOURIER else python_phi_predict
    _cc_f2 = fourier(simulated_phi) if simulated_phi.basis != Basis.FOURIER else simulated_phi
    _, phi_correlation_ps = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(julia_phi_predict) if julia_phi_predict.basis != Basis.FOURIER else julia_phi_predict
    _cc_f2 = fourier(python_phi_predict) if python_phi_predict.basis != Basis.FOURIER else python_phi_predict
    _, phi_correlation_jp = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    plt.figure()
    plt.plot(ell, phi_correlation_js, label = "Julia x Sim")
    plt.plot(ell, phi_correlation_ps, label = "Python x Sim")
    plt.plot(ell, phi_correlation_jp, label = "Julia x Python")
    plt.title("Phi Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Phi Cross Correlation.png", bbox_inches="tight")
    plt.close()

    SUB_FOLDER = "map_joint/intensity_and_polarization/fields/"

    _plot_field = python_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Python MLE Field: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Field: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Python MLE Field: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Field: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Python MLE Field: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Python MLE Field: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = julia_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("Julia MLE Field: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Field: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("Julia MLE Field: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Field: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("Julia MLE Field: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Julia MLE Field: B Mode.png", bbox_inches="tight")
    plt.close()

    _plot_field = python_field_predict - julia_field_predict
    _plot_field = map(_plot_field) if _plot_field.basis != Basis.MAP else _plot_field
    plt.figure()
    plt.imshow(_plot_field.scalar_matrix, cmap="coolwarm")
    plt.title("MLE Field Diff: Temperature")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Field Diff: Temperature.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_1, cmap="coolwarm")
    plt.title("MLE Field Diff: E Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Field Diff: E Mode.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(_plot_field.polar_matrix_2, cmap="coolwarm")
    plt.title("MLE Field Diff: B Mode")
    plt.colorbar()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "MLE Field Diff: B Mode.png", bbox_inches="tight")
    plt.close()
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    ell, tt_correlation_js = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, tt_correlation_ps = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _, tt_correlation_jp = primal_cross_correlation(_cc_f1.scalar_matrix, _cc_f2.scalar_matrix, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, ee_correlation_js = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _cc_f1 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, ee_correlation_ps = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _, ee_correlation_jp = primal_cross_correlation(_cc_f1.polar_matrix_1, _cc_f2.polar_matrix_1, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, bb_correlation_js = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    _cc_f1 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _cc_f2 = fourier(simulated_field) if simulated_field.basis != Basis.FOURIER else simulated_field
    _, bb_correlation_ps = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)
    _cc_f1 = fourier(julia_field_predict) if julia_field_predict.basis != Basis.FOURIER else julia_field_predict
    _cc_f2 = fourier(python_field_predict) if python_field_predict.basis != Basis.FOURIER else python_field_predict
    _, bb_correlation_jp = primal_cross_correlation(_cc_f1.polar_matrix_2, _cc_f2.polar_matrix_2, THETA_PIX)

    plt.figure()
    plt.plot(ell, tt_correlation_js, label = "Julia x Sim TT")
    plt.plot(ell, tt_correlation_ps, label = "Python x Sim TT")
    plt.plot(ell, tt_correlation_jp, label = "Julia x Python TT")
    plt.title("Temperature Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "Temperature Cross Correlation.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(ell, ee_correlation_js, label = "Julia x Sim EE")
    plt.plot(ell, ee_correlation_ps, label = "Python x Sim EE")
    plt.plot(ell, ee_correlation_jp, label = "Julia x Python EE")
    plt.title("E Mode Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "E Mode Cross Correlation.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(ell, bb_correlation_js, label = "Julia x Sim BB")
    plt.plot(ell, bb_correlation_ps, label = "Python x Sim BB")
    plt.plot(ell, bb_correlation_jp, label = "Julia x Python BB")
    plt.title("B Mode Cross Correlation")
    plt.legend()
    plt.savefig(FIGURE_PATH + SUB_FOLDER + "B Mode Cross Correlation.png", bbox_inches="tight")
    plt.close()

    #compare percent difference between prediction and ground truth
    assert jnp.max(percent_diff_2d(julia_field_predict, python_field_predict)) <= MAX_NORM_DIFF

    #also assert that the average cross correlation is greater than our threshold
    #TODO handle NaN case here...
    # assert jnp.mean(tt_correlation_jp) > MIN_AVG_CORRELATION
    # assert jnp.mean(ee_correlation_jp) > MIN_AVG_CORRELATION
    # assert jnp.mean(bb_correlation_jp) > MIN_AVG_CORRELATION