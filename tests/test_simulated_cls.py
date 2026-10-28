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
from figure_utils import save_figure

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
CAMB_LMAX_PRIME = 5000
CAMB_LMAX = 17_000

def test_simulated_cls_intensity_only():

    #compute cls for temperature only
    batched_results = get_avg_cls(theta_pix = THETA_PIX_DEFAULT, num_trials = 100, pol = "I", nside = NSIDE_DEFAULT,
                                  uk_arcmin_t = 10, delta_l = 50, lmax = CAMB_LMAX, r = 0.2, 
                                  tau = 0.055, nt = -0.025, k_pivot = 0.002, max_l_tensor = 2*CAMB_LMAX_PRIME,
                                  max_eta_k_tensor = 4*CAMB_LMAX_PRIME, use_mask = True, begin_ell_interp = CAMB_LMAX_PRIME)

    #load ground truth ell and cls
    ell_I = precision_load(GROUND_TRUTH + "ell_I.npz")
    cl_pp_avg_I = precision_load(GROUND_TRUTH + "cl_pp_avg_I.npz")
    cl_tt_avg_I = precision_load(GROUND_TRUTH + "cl_tt_avg_I.npz")
    cl_ll_avg_I = precision_load(GROUND_TRUTH + "cl_ll_avg_I.npz")
    cl_dd_avg_I = precision_load(GROUND_TRUTH + "cl_dd_avg_I.npz")

    #plot the log(cls) v.s. ell against each other for julia v.s. jax
    SUB_FOLDER = "simulated_cls/intensity_only/"
    plt.figure()
    plt.plot(ell_I, jnp.log(cl_pp_avg_I), label = "Julia")
    plt.plot(batched_results["phi"][0], jnp.log(batched_results["phi"][1]), label = "Python")
    plt.title("Simulated Cl_PP Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_pp_julia_v_s_jax_i.png")
    plt.close()

    plt.figure()
    plt.plot(ell_I, jnp.log(cl_tt_avg_I), label = "Julia")
    plt.plot(batched_results["unlensed_t"][0], jnp.log(batched_results["unlensed_t"][1]), label = "Python")
    plt.title("Simulated Cl_TT Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_tt_julia_v_s_jax_i.png")
    plt.close()

    plt.figure()
    plt.plot(ell_I, jnp.log(cl_ll_avg_I), label = "Julia")
    plt.plot(batched_results["lensed_t"][0], jnp.log(batched_results["lensed_t"][1]), label = "Python")
    plt.title("Simulated Cl_LL Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_ll_julia_v_s_jax_i.png")
    plt.close()

    plt.figure()
    plt.plot(ell_I, jnp.log(cl_dd_avg_I), label = "Julia")
    plt.plot(batched_results["data_t"][0], jnp.log(batched_results["data_t"][1]), label = "Python")
    plt.title("Simulated Cl_DD Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_dd_julia_v_s_jax_i.png")
    plt.close()

    #there is no 'test' in this case we simply just inspect the plots by eye
    assert 1 == 1

def test_simulated_cls_polarization_only():

    #compute cls for temperature only
    batched_results = get_avg_cls(theta_pix = THETA_PIX_DEFAULT, num_trials = 100, pol = "P", nside = NSIDE_DEFAULT,
                                  uk_arcmin_t = 10, delta_l = 50, lmax = CAMB_LMAX, r = 0.2, 
                                  tau = 0.055, nt = -0.025, k_pivot = 0.002, max_l_tensor = 2*CAMB_LMAX_PRIME,
                                  max_eta_k_tensor = 4*CAMB_LMAX_PRIME, use_mask = True, begin_ell_interp = CAMB_LMAX_PRIME)

    #load ground truth ell and cls
    ell_P = precision_load(GROUND_TRUTH + "ell_P.npz")
    cl_pp_avg_P = precision_load(GROUND_TRUTH + "cl_pp_avg_P.npz")
    cl_ee_avg_P = precision_load(GROUND_TRUTH + "cl_tt_e_avg_P.npz")
    cl_bb_avg_P = precision_load(GROUND_TRUTH + "cl_tt_b_avg_P.npz")
    cl_elel_avg_P = precision_load(GROUND_TRUTH + "cl_ll_e_avg_P.npz")
    cl_blbl_avg_P = precision_load(GROUND_TRUTH + "cl_ll_b_avg_P.npz")
    cl_dede_avg_P = precision_load(GROUND_TRUTH + "cl_dd_e_avg_P.npz")
    cl_dbdb_avg_P = precision_load(GROUND_TRUTH + "cl_dd_b_avg_P.npz")

    #plot the log(cls) v.s. ell against each other for julia v.s. jax
    SUB_FOLDER = "simulated_cls/polarization_only/"
    plt.figure()
    plt.plot(ell_P, jnp.log(cl_pp_avg_P), label = "Julia")
    plt.plot(batched_results["phi"][0], jnp.log(batched_results["phi"][1]), label = "Python")
    plt.title("Simulated Cl_PP Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_pp_julia_v_s_jax_p.png")
    plt.close()

    plt.figure()
    plt.plot(ell_P, jnp.log(cl_ee_avg_P), label = "Julia")
    plt.plot(batched_results["unlensed_e"][0], jnp.log(batched_results["unlensed_e"][1]), label = "Python")
    plt.title("Simulated Cl_EE Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_ee_julia_v_s_jax_p.png")
    plt.close()

    plt.figure()
    plt.plot(ell_P, jnp.log(cl_elel_avg_P), label = "Julia")
    plt.plot(batched_results["lensed_e"][0], jnp.log(batched_results["lensed_e"][1]), label = "Python")
    plt.title("Simulated Lensed Cl_EE Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_lensed_cl_ee_julia_v_s_jax_p.png")
    plt.close()

    plt.figure()
    plt.plot(ell_P, jnp.log(cl_bb_avg_P), label = "Julia")
    plt.plot(batched_results["unlensed_b"][0], jnp.log(batched_results["unlensed_b"][1]), label = "Python")
    plt.title("Simulated Cl_BB Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_bb_julia_v_s_jax_p.png")
    plt.close()

    plt.figure()
    plt.plot(ell_P, jnp.log(cl_blbl_avg_P), label = "Julia")
    plt.plot(batched_results["lensed_b"][0], jnp.log(batched_results["lensed_b"][1]), label = "Python")
    plt.title("Simulated Lensed Cl_BB Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_lensed_cl_bb_julia_v_s_jax_p.png")
    plt.close()

    plt.figure()
    plt.plot(ell_P, jnp.log(cl_dede_avg_P), label = "Julia")
    plt.plot(batched_results["data_e"][0], jnp.log(batched_results["data_e"][1]), label = "Python")
    plt.title("Simulated E mode Cl_DD Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_e_mode_cl_dd_julia_v_s_jax_p.png")
    plt.close()

    plt.figure()
    plt.plot(ell_P, jnp.log(cl_dbdb_avg_P), label = "Julia")
    plt.plot(batched_results["data_b"][0], jnp.log(batched_results["data_b"][1]), label = "Python")
    plt.title("Simulated B mode Cl_DD Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_b_mode_cl_dd_julia_v_s_jax_p.png")
    plt.close()

    #there is no 'test' in this case we simply just inspect the plots by eye
    assert 1 == 1

def test_simulated_cls_intensity_and_polarization():

    #compute cls for temperature only
    batched_results = get_avg_cls(theta_pix = THETA_PIX_DEFAULT, num_trials = 100, pol = "IP", nside = NSIDE_DEFAULT,
                                  uk_arcmin_t = 10, delta_l = 50, lmax = CAMB_LMAX, r = 0.2, 
                                  tau = 0.055, nt = -0.025, k_pivot = 0.002, max_l_tensor = 2*CAMB_LMAX_PRIME,
                                  max_eta_k_tensor = 4*CAMB_LMAX_PRIME, use_mask = True, begin_ell_interp = CAMB_LMAX_PRIME)

    #load ground truth ell and cls
    ell_IP = precision_load(GROUND_TRUTH + "ell_IP.npz")
    cl_pp_avg_IP = precision_load(GROUND_TRUTH + "cl_pp_avg_IP.npz")
    cl_tt_avg_IP = precision_load(GROUND_TRUTH + "cl_tt_i_avg_IP.npz")
    cl_ee_avg_IP = precision_load(GROUND_TRUTH + "cl_tt_e_avg_IP.npz")
    cl_bb_avg_IP = precision_load(GROUND_TRUTH + "cl_tt_b_avg_IP.npz")
    cl_tltl_avg_IP = precision_load(GROUND_TRUTH + "cl_ll_i_avg_IP.npz")
    cl_elel_avg_IP = precision_load(GROUND_TRUTH + "cl_ll_e_avg_IP.npz")
    cl_blbl_avg_IP = precision_load(GROUND_TRUTH + "cl_ll_b_avg_IP.npz")
    cl_dtdt_avg_IP = precision_load(GROUND_TRUTH + "cl_dd_i_avg_IP.npz")
    cl_dede_avg_IP = precision_load(GROUND_TRUTH + "cl_dd_e_avg_IP.npz")
    cl_dbdb_avg_IP = precision_load(GROUND_TRUTH + "cl_dd_b_avg_IP.npz")

    #plot the log(cls) v.s. ell against each other for julia v.s. jax
    SUB_FOLDER = "simulated_cls/intensity_and_polarization/"
    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_pp_avg_IP), label = "Julia")
    plt.plot(batched_results["phi"][0], jnp.log(batched_results["phi"][1]), label = "Python")
    plt.title("Simulated Cl_PP Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_pp_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_tt_avg_IP), label = "Julia")
    plt.plot(batched_results["unlensed_t"][0], jnp.log(batched_results["unlensed_t"][1]), label = "Python")
    plt.title("Simulated Cl_TT Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_tt_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_tltl_avg_IP), label = "Julia")
    plt.plot(batched_results["lensed_t"][0], jnp.log(batched_results["lensed_t"][1]), label = "Python")
    plt.title("Simulated Lensed Cl_TT Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_lensed_cl_tt_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_ee_avg_IP), label = "Julia")
    plt.plot(batched_results["unlensed_e"][0], jnp.log(batched_results["unlensed_e"][1]), label = "Python")
    plt.title("Simulated Cl_EE Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_ee_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_elel_avg_IP), label = "Julia")
    plt.plot(batched_results["lensed_e"][0], jnp.log(batched_results["lensed_e"][1]), label = "Python")
    plt.title("Simulated Lensed Cl_EE Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_lensed_cl_ee_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_bb_avg_IP), label = "Julia")
    plt.plot(batched_results["unlensed_b"][0], jnp.log(batched_results["unlensed_b"][1]), label = "Python")
    plt.title("Simulated Cl_BB Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_cl_bb_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_blbl_avg_IP), label = "Julia")
    plt.plot(batched_results["lensed_b"][0], jnp.log(batched_results["lensed_b"][1]), label = "Python")
    plt.title("Simulated Lensed Cl_BB Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_lensed_cl_bb_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_dtdt_avg_IP), label = "Julia")
    plt.plot(batched_results["data_t"][0], jnp.log(batched_results["data_t"][1]), label = "Python")
    plt.title("Simulated Temperature Cl_DD Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_temperature_cl_dd_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_dede_avg_IP), label = "Julia")
    plt.plot(batched_results["data_e"][0], jnp.log(batched_results["data_e"][1]), label = "Python")
    plt.title("Simulated E mode Cl_DD Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_e_mode_cl_dd_julia_v_s_jax_ip.png")
    plt.close()

    plt.figure()
    plt.plot(ell_IP, jnp.log(cl_dbdb_avg_IP), label = "Julia")
    plt.plot(batched_results["data_b"][0], jnp.log(batched_results["data_b"][1]), label = "Python")
    plt.title("Simulated B mode Cl_DD Julia v.s. JAX")
    plt.legend()
    save_figure(FIGURE_PATH + SUB_FOLDER + "simulated_b_mode_cl_dd_julia_v_s_jax_ip.png")
    plt.close()

    #there is no 'test' in this case we simply just inspect the plots by eye
    assert 1 == 1
