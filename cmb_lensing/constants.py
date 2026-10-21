import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

DEFAULT_HMC_STEP_SIZE = 0.05
DEFAULT_HMC_NUM_STEPS = 10
NPHI_FAC = 2
ARCMIN_PER_DEGREE = 60
INVERSE_LENSE = -1
FORWARD_LENSE = +1
BEAM_TRANSFER_SCALAR = 8*jnp.log(2)
DEFAULT_MAX_ELL = 4000
DEFAULT_MNU = 0.06
DEFAULT_TAUREIO = 0.05
DEFAULT_K_PIVOT = 0.05
DEFAULT_A_LENSE = 1
MAX_NUM_SAMPLED = 5
DEFAULT_ELL_CUTOFF = 10_000

#Bracket CAMB searches when it has to solve for H0 from cosmomc_theta. CAMB's own
#default is (10, 100), which has no solution for theta_MC_100 above ~1.117 - that sits
#inside PARAM_BOUNDS, whose top edge of 1.1452 needs H0 ~ 102, so proposals near the top
#of the theta box raised CAMBParamRangeError and were silently turned into NaN spectra.
#150 covers the whole box (theta_MC_100 reaches 1.22-1.27 at H0 = 150 across the omch2
#range). CAMB ignores this whenever H0 is passed directly
DEFAULT_THETA_H0_RANGE = [10, 150]

# ── The LCDM parameters ───────────────────────────────────────────────────

#the sampled parameters and the order every (5,) parameter array is laid out in (tau is
#pinned at DEFAULT_TAUREIO, mnu at DEFAULT_MNU)
PARAM_ORDER = ["theta_MC_100", "logA", "ns", "ombh2", "omch2"]
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_ORDER)}

#the fiducial point every simulated data set, forecast and measurement is built at
GROUND_TRUTH = {
    "theta_MC_100": 1.031732,
    "logA": 3.218387,
    "ns": 0.959814,
    "ombh2": 0.022386,
    "omch2": 0.109381,
}

#the sampler's search box for each parameter, roughly GROUND_TRUTH +/- 5 PARAM_SIGMA
PARAM_BOUNDS = {
    "theta_MC_100": (0.9328, 1.1452),
    "logA": (2.661635, 3.782861),
    "ns": (0.867143, 1.042186),
    "ombh2": (0.020413, 0.024389),
    "omch2": (0.085, 0.155541),
}

#one sigma of each parameter: normalizes the sampler's progress plot and sets the Fisher
#forecast's and the empirical stencils' finite-difference steps
PARAM_SIGMA = {
    "theta_MC_100": 0.0262018,
    "logA": 0.1128948,
    "ns": 0.0164744,
    "ombh2": 0.0004006,
    "omch2": 0.0059354,
}

#lmax = lmax_prime = DEFAULT_MAX_ELL makes dl2cl's ell grid arange(2, lmax) land on
#ells 2..DEFAULT_MAX_ELL-1 with identity interpolation - pure CAMB, no extrapolation.
#IMPORTANT: this must stay at DEFAULT_MAX_ELL (not +1) so the spline caches share the exact
#ell support of load_sim's CAMB data-map path (also 2..DEFAULT_MAX_ELL-1). The model and the
#data then log-log extrapolate the 2D grid tail from the SAME anchor interval; a
#one-multipole support mismatch was measured to shift the clamped ombh2 conditional peak by
#~+5e-5
CAMB_LMAX = DEFAULT_MAX_ELL

#where the CAMB spline caches live: camb_splines/ at the repository root (the five 1D
#camb_<param>_grid.npz of camb_grid_generation/precompute_camb_1d.py and the gitignored
#~8 GB 5D camb_grid_spline.npz the sampler reads by default)
CAMB_SPLINES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "camb_splines")
CAMB_GRID_PATH = os.path.join(CAMB_SPLINES_DIR, "camb_grid_spline.npz")

#The default field and operator constants are defined for a square 256 x 256 map
#with 2 arcminute resolution...
NSIDE_DEFAULT = 256
THETA_PIX_DEFAULT = 2.5
PIX_WIDTH_DEFAULT = float(jnp.deg2rad(THETA_PIX_DEFAULT / ARCMIN_PER_DEGREE))
FOURIER_WEIGHTS_DEFAULT = 2 * jnp.ones(NSIDE_DEFAULT//2+1, dtype =jnp.complex128).at[0].set(0.5).at[-1].set(0.5)
FOURIER_MATRIX_DEFAULT = jnp.zeros((NSIDE_DEFAULT, NSIDE_DEFAULT // 2 + 1), dtype=jnp.complex128)
MAP_MATRIX_DEFAULT = jnp.zeros((NSIDE_DEFAULT, NSIDE_DEFAULT), dtype=jnp.float64)

#Enum for different possible parametrizations (i.e. Stokes' parameters)
class Parametrization:
    T = 0
    QU = 1
    EB = 2

#Enum for different possible bases (i.e. real space v.s. Fourier space)
class Basis:
    MAP = 0
    FOURIER = 1

class Polarity:
    I = "I"
    P = "P"
    IP = "IP"
