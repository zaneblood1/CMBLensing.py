import jax
import flax
jax.config.update("jax_enable_x64", True)
from flax import struct
from cmb_lensing.fields import *
from cmb_lensing.matrix_operators import *
from cmb_lensing.constants import *

##############################################################
#The data set classes below are used as containers to 
#efficiently pass around commonly used fields and data
##############################################################

@struct.dataclass
class DataSet:
    #various constants / metadata needed for some calculations throughout the code
    fourier_weights: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_WEIGHTS_DEFAULT.copy())
    nside: int = flax.struct.field(pytree_node = False, default = NSIDE_DEFAULT)
    theta_pix: float = flax.struct.field(pytree_node = False, default = THETA_PIX_DEFAULT)
    pix_width: float = flax.struct.field(pytree_node = False, default = PIX_WIDTH_DEFAULT)

#Temperature only data set
@struct.dataclass
class DataSetT(DataSet):

    #diagonal matrix operators
    noise_covariance: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())
    mixing_d: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())
    mixing_g: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())
    field_covariance: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())
    lensed_field_covariance: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())
    phi_covariance: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())
    mask: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())
    beam: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())
    quadratic_estimate: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())

    #spin-0 fields
    data: FlatS0 = flax.struct.field(default_factory = lambda: FlatS0())
    unlensed_field: FlatS0 = flax.struct.field(default_factory = lambda: FlatS0())
    lensed_field: FlatS0 = flax.struct.field(default_factory = lambda: FlatS0())
    phi: FlatS0 = flax.struct.field(default_factory = lambda: FlatS0())

#Polarization only data set
@struct.dataclass
class DataSetEB(DataSet):

    #block diagonal matrix operators
    noise_covariance: DiagonalEB = flax.struct.field(default_factory = lambda: DiagonalEB())
    mixing_d: DiagonalEB = flax.struct.field(default_factory = lambda: DiagonalEB())
    mixing_g: DiagonalEB = flax.struct.field(default_factory = lambda: DiagonalEB())
    field_covariance: DiagonalEB = flax.struct.field(default_factory = lambda: DiagonalEB())
    lensed_field_covariance: DiagonalEB = flax.struct.field(default_factory = lambda: DiagonalEB())
    phi_covariance: DiagonalEB = flax.struct.field(default_factory = lambda: DiagonalEB())
    mask: DiagonalEB = flax.struct.field(default_factory = lambda: DiagonalEB())
    beam: DiagonalEB = flax.struct.field(default_factory = lambda: DiagonalEB())
    quadratic_estimate: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())

    #spin-2 fields
    data: FlatS2 = flax.struct.field(default_factory = lambda: FlatS2())
    unlensed_field: FlatS2 = flax.struct.field(default_factory = lambda: FlatS2())
    lensed_field: FlatS2 = flax.struct.field(default_factory = lambda: FlatS2())
    phi: FlatS2 = flax.struct.field(default_factory = lambda: FlatS2())

#Full temperature and polarization parametrization data set
@struct.dataclass
class DataSetTEB(DataSet):

    noise_covariance: BlockTEB = flax.struct.field(default_factory = lambda: BlockTEB())
    mixing_d: BlockTEB = flax.struct.field(default_factory = lambda: BlockTEB())
    mixing_g: BlockTEB = flax.struct.field(default_factory = lambda: BlockTEB())
    field_covariance: BlockTEB = flax.struct.field(default_factory = lambda: BlockTEB())
    lensed_field_covariance: BlockTEB = flax.struct.field(default_factory = lambda: BlockTEB())
    phi_covariance: BlockTEB = flax.struct.field(default_factory = lambda: BlockTEB())
    mask: BlockTEB = flax.struct.field(default_factory = lambda: BlockTEB())
    beam: BlockTEB = flax.struct.field(default_factory = lambda: BlockTEB())
    quadratic_estimate: DiagonalScalar = flax.struct.field(default_factory = lambda: DiagonalScalar())

    #fields
    data: FlatS02 = flax.struct.field(default_factory = lambda: FlatS02())
    unlensed_field: FlatS02 = flax.struct.field(default_factory = lambda: FlatS02())
    lensed_field: FlatS02 = flax.struct.field(default_factory = lambda: FlatS02())
    phi: FlatS02 = flax.struct.field(default_factory = lambda: FlatS02())