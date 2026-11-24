import jax.numpy as jnp
import numpy as np
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
import matplotlib
matplotlib.use("Agg")
from figure_utils import save_figure
from regression_thresholds import max_allowed

#constants relating to the ground truth data
GROUND_TRUTH = os.getcwd() + "/tests/ground_truth_data/"
FIGURE_PATH = os.getcwd() + "/tests/test_generated_figures/"
with open(GROUND_TRUTH + "n_side.txt", "r") as file:
    N_SIDE = int(file.read().strip())
with open(GROUND_TRUTH + "theta_pix.txt", "r") as file:
    THETA_PIX = float(file.read().strip())
PIX_WIDTH = float(jnp.deg2rad(THETA_PIX / ARCMIN_PER_DEGREE))

CAMB_LMAX_PRIME = 5000
CAMB_LMAX = 17_000
NUM_TRIALS = 100
NOISE_UK_ARCMIN = 10
DELTA_L = 50

#seeds numpy's global generator, which get_avg_cls draws its map seeds from, so the test is reproducible
SIMULATION_SEED = 0

#The Julia and Python spectra are averages over two independent sets of NUM_TRIALS simulations, so they
#differ by sample variance. The difference is the rms over the ell bins of ln(Cl_python / Cl_julia);
#this is the largest value over all spectra and polarizations, measured at SIMULATION_SEED
MEASURED_RMS_LOG_CL_DIFF = 0.035

def _compare_simulated_cls(pol, sub_folder, spectra):
    """Plot and compare the averaged simulated spectra of one polarization with Julia's.

    spectra holds one (python key, julia file stem, plot title, figure file name) per spectrum.
    """
    np.random.seed(SIMULATION_SEED)
    batched_results = get_avg_cls(theta_pix = THETA_PIX_DEFAULT, num_trials = NUM_TRIALS, pol = pol, nside = NSIDE_DEFAULT,
                                  uk_arcmin_t = NOISE_UK_ARCMIN, delta_l = DELTA_L, lmax = CAMB_LMAX, r = 0.2,
                                  tau = 0.055, nt = -0.025, k_pivot = 0.002, max_l_tensor = 2*CAMB_LMAX_PRIME,
                                  max_eta_k_tensor = 4*CAMB_LMAX_PRIME, use_mask = True, begin_ell_interp = CAMB_LMAX_PRIME)
    ell_julia = precision_load(GROUND_TRUTH + f"ell_{pol}.npz")

    rms_log_cl_diffs = {}
    for python_key, julia_file, title, figure_name in spectra:
        cl_julia = precision_load(GROUND_TRUTH + julia_file + ".npz")
        ell_python, cl_python = batched_results[python_key]

        #plot the log(cls) v.s. ell against each other for julia v.s. jax
        plt.figure()
        plt.plot(ell_julia, jnp.log(cl_julia), label = "Julia")
        plt.plot(ell_python, jnp.log(cl_python), label = "Python")
        plt.title(title)
        plt.legend()
        save_figure(FIGURE_PATH + sub_folder + figure_name + ".png")
        plt.close()

        log_cl_diff = jnp.log(cl_python / cl_julia)
        rms_log_cl_diffs[python_key] = float(jnp.sqrt(jnp.mean(log_cl_diff**2)))

    for python_key, rms_log_cl_diff in rms_log_cl_diffs.items():
        assert rms_log_cl_diff <= max_allowed(MEASURED_RMS_LOG_CL_DIFF), python_key

def test_simulated_cls_intensity_only():
    _compare_simulated_cls("I", "simulated_cls/intensity_only/", [
        ("phi", "cl_pp_avg_I", "Simulated Cl_PP Julia v.s. JAX", "simulated_cl_pp_julia_v_s_jax_i"),
        ("unlensed_t", "cl_tt_avg_I", "Simulated Cl_TT Julia v.s. JAX", "simulated_cl_tt_julia_v_s_jax_i"),
        ("lensed_t", "cl_ll_avg_I", "Simulated Cl_LL Julia v.s. JAX", "simulated_cl_ll_julia_v_s_jax_i"),
        ("data_t", "cl_dd_avg_I", "Simulated Cl_DD Julia v.s. JAX", "simulated_cl_dd_julia_v_s_jax_i"),
    ])

def test_simulated_cls_polarization_only():
    _compare_simulated_cls("P", "simulated_cls/polarization_only/", [
        ("phi", "cl_pp_avg_P", "Simulated Cl_PP Julia v.s. JAX", "simulated_cl_pp_julia_v_s_jax_p"),
        ("unlensed_e", "cl_tt_e_avg_P", "Simulated Cl_EE Julia v.s. JAX", "simulated_cl_ee_julia_v_s_jax_p"),
        ("lensed_e", "cl_ll_e_avg_P", "Simulated Lensed Cl_EE Julia v.s. JAX", "simulated_lensed_cl_ee_julia_v_s_jax_p"),
        ("unlensed_b", "cl_tt_b_avg_P", "Simulated Cl_BB Julia v.s. JAX", "simulated_cl_bb_julia_v_s_jax_p"),
        ("lensed_b", "cl_ll_b_avg_P", "Simulated Lensed Cl_BB Julia v.s. JAX", "simulated_lensed_cl_bb_julia_v_s_jax_p"),
        ("data_e", "cl_dd_e_avg_P", "Simulated E mode Cl_DD Julia v.s. JAX", "simulated_e_mode_cl_dd_julia_v_s_jax_p"),
        ("data_b", "cl_dd_b_avg_P", "Simulated B mode Cl_DD Julia v.s. JAX", "simulated_b_mode_cl_dd_julia_v_s_jax_p"),
    ])

def test_simulated_cls_intensity_and_polarization():
    _compare_simulated_cls("IP", "simulated_cls/intensity_and_polarization/", [
        ("phi", "cl_pp_avg_IP", "Simulated Cl_PP Julia v.s. JAX", "simulated_cl_pp_julia_v_s_jax_ip"),
        ("unlensed_t", "cl_tt_i_avg_IP", "Simulated Cl_TT Julia v.s. JAX", "simulated_cl_tt_julia_v_s_jax_ip"),
        ("lensed_t", "cl_ll_i_avg_IP", "Simulated Lensed Cl_TT Julia v.s. JAX", "simulated_lensed_cl_tt_julia_v_s_jax_ip"),
        ("unlensed_e", "cl_tt_e_avg_IP", "Simulated Cl_EE Julia v.s. JAX", "simulated_cl_ee_julia_v_s_jax_ip"),
        ("lensed_e", "cl_ll_e_avg_IP", "Simulated Lensed Cl_EE Julia v.s. JAX", "simulated_lensed_cl_ee_julia_v_s_jax_ip"),
        ("unlensed_b", "cl_tt_b_avg_IP", "Simulated Cl_BB Julia v.s. JAX", "simulated_cl_bb_julia_v_s_jax_ip"),
        ("lensed_b", "cl_ll_b_avg_IP", "Simulated Lensed Cl_BB Julia v.s. JAX", "simulated_lensed_cl_bb_julia_v_s_jax_ip"),
        ("data_t", "cl_dd_i_avg_IP", "Simulated Temperature Cl_DD Julia v.s. JAX", "simulated_temperature_cl_dd_julia_v_s_jax_ip"),
        ("data_e", "cl_dd_e_avg_IP", "Simulated E mode Cl_DD Julia v.s. JAX", "simulated_e_mode_cl_dd_julia_v_s_jax_ip"),
        ("data_b", "cl_dd_b_avg_IP", "Simulated B mode Cl_DD Julia v.s. JAX", "simulated_b_mode_cl_dd_julia_v_s_jax_ip"),
    ])
