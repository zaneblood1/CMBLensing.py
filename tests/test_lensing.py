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
