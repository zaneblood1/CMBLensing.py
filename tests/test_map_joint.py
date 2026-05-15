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
