import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

NPHI_FAC = 2
ARCMIN_PER_DEGREE = 60
INVERSE_LENSE = -1
FORWARD_LENSE = +1
PID_CONTROLLER_RTOL = 1e-6 #Relative tolerance for Diffrax integrator
PID_CONTROLLER_ATOL = 1e-6 #Absolute tolerance for Diffrax integrator
MAX_DIFFRAX_STEPS = 100_000
BEAM_TRANSFER_SCALAR = 8*jnp.log(2)
PARAM_KEYS = {"ombh2": 0, "omch2": 1, "tau": 2, "ns": 3, "As": 4, "H0": 5}
AR_KEYS = {"r": 0, "a_phi": 1}

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
