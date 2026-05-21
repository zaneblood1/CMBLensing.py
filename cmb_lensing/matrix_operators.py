import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import flax
from flax import struct
from cmb_lensing.constants import *
from cmb_lensing.util import *
from functools import singledispatch
from cmb_lensing.fields import *

@struct.dataclass
class FieldOperator:

    fourier_weights: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_WEIGHTS_DEFAULT.copy())
    nside: int = flax.struct.field(pytree_node = False, default = NSIDE_DEFAULT)
    theta_pix: float = flax.struct.field(pytree_node = False, default = THETA_PIX_DEFAULT)
    pix_width: float = flax.struct.field(pytree_node = False, default = PIX_WIDTH_DEFAULT)

    def _matrix_names(self):
        raise NotImplementedError
    
    #addition and subtraction work the same for all types of operators
    def __add__(self, other):
        if isinstance(other, self.__class__):
            updates = {name: getattr(self, name) + getattr(other, name) for name in self._matrix_names()}
        else:
            updates = {name: getattr(self, name) + other for name in self._matrix_names()}
        return self.replace(**updates)
    
    def __sub__(self, other):
        if isinstance(other, self.__class__):
            updates = {name: getattr(self, name) - getattr(other, name) for name in self._matrix_names()}
        else:
            updates = {name: getattr(self, name) - other for name in self._matrix_names()}
        return self.replace(**updates)
    
    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            updates = {name: getattr(other, name) - getattr(self, name) for name in self._matrix_names()}
        else:
            updates = {name: other - getattr(self, name) for name in self._matrix_names()}
        return self.replace(**updates)
    
    
@struct.dataclass
class DiagonalScalar(FieldOperator):

    scalar_matrix: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())

    def _matrix_names(self):
        return {"scalar_matrix"}
    
    #NOTE we must define operator-operator, operator-field, and operator scalar multiplication
    def __mul__(self, other):
        if isinstance(other, DiagonalScalar):
            return self.replace(scalar_matrix = self.scalar_matrix * other.scalar_matrix)
        elif isinstance(other, FlatS0):
            return other.replace(scalar_matrix = self.scalar_matrix * other.scalar_matrix)
        else:
            return self.replace(scalar_matrix = self.scalar_matrix * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)

@struct.dataclass
class DiagonalEB(FieldOperator):

    #NOTE these matrices have the following 2 x 2 block diagonal format
    #[EE 0 ]
    #[0  BB]

    matrix_EE: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    matrix_BB: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())

    def _matrix_names(self):
        return {"matrix_EE", "matrix_BB"}
    
    def __mul__(self, other):
        if isinstance(other, DiagonalEB):
            return self.replace(matrix_EE = self.matrix_EE * other.matrix_EE,
                                matrix_BB = self.matrix_BB * other.matrix_BB)
        elif isinstance(other, FlatS2):
            return other.replace(polar_matrix_1 = self.matrix_EE * other.polar_matrix_1,
                                 polar_matrix_2 = self.matrix_BB * other.polar_matrix_2)
        else:
            return self.replace(matrix_EE = self.matrix_EE * other,
                                matrix_BB = self.matrix_BB * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
@struct.dataclass
class BlockTEB(FieldOperator):

    #NOTE these matrices have the following 3 x 3 block diagonal format
    #[TT TE 0 ]
    #[ET EE 0 ]
    #[0  0  BB]

    matrix_TT: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    matrix_TE: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    matrix_ET: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    matrix_EE: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    matrix_BB: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())

    def _matrix_names(self):
        return {"matrix_TT", "matrix_TE", "matrix_ET", "matrix_EE", "matrix_BB"}
    
    #NOTE non-diagonal matrix multiplication is NOT commutative hence we need distinction
    #between multiplying on the left or right for BlockTEB matrices specifically
    def __mul__(self, other):
          
        #matrix-matrix multiplication, matrix-field multiplication, and matrix-scalar multiplication                                                                                                                                                               
        if isinstance(other, BlockTEB):
            #compute a non-diagonal matrix-matrix product                   
            return self.replace(matrix_TT = self.matrix_TT * other.matrix_TT + self.matrix_TE * other.matrix_ET,
                                matrix_ET = self.matrix_ET * other.matrix_TT + self.matrix_EE * other.matrix_ET,
                                matrix_TE = self.matrix_TT * other.matrix_TE + self.matrix_TE * other.matrix_EE,
                                matrix_EE = self.matrix_ET * other.matrix_TE + self.matrix_EE * other.matrix_EE,
                                matrix_BB = self.matrix_BB * other.matrix_BB)
        #matrix-field multiplication
        elif isinstance(other, FlatS02):
            return other.replace(scalar_matrix = self.matrix_TT * other.scalar_matrix + self.matrix_TE * other.polar_matrix_1,
                                 polar_matrix_1 = self.matrix_ET * other.scalar_matrix + self.matrix_EE * other.polar_matrix_1,
                                 polar_matrix_2 = self.matrix_BB * other.polar_matrix_2)
        #matrix-scalar multiplication
        return self.replace(matrix_TT = self.matrix_TT * other,
                            matrix_TE = self.matrix_TE * other,
                            matrix_ET = self.matrix_ET * other,
                            matrix_EE = self.matrix_EE * other,
                            matrix_BB = self.matrix_BB * other)                                                                                                                                         
    
    #NOTE we only expect to multiply a BlockTEB on the right by another BlockTEB or a scalar
    def __rmul__(self, other): 
                                                                                                                                                                       
        if isinstance(other, BlockTEB):
            #compute a non-diagonal matrix-matrix product                   
            return self.replace(matrix_TT = other.matrix_TT * self.matrix_TT + other.matrix_TE * self.matrix_ET,
                                matrix_ET = other.matrix_ET * self.matrix_TT + other.matrix_EE * self.matrix_ET,
                                matrix_TE = other.matrix_TT * self.matrix_TE + other.matrix_TE * self.matrix_EE,
                                matrix_EE = other.matrix_ET * self.matrix_TE + other.matrix_EE * self.matrix_EE,
                                matrix_BB = other.matrix_BB * self.matrix_BB)
        #matrix-scalar multiplication
        return self.replace(matrix_TT = self.matrix_TT * other,
                            matrix_TE = self.matrix_TE * other,
                            matrix_ET = self.matrix_ET * other,
                            matrix_EE = self.matrix_EE * other,
                            matrix_BB = self.matrix_BB * other) 
    
#get an identity operator shaped like a specific operator
@jax.jit
def get_identity_like(matrix_operator):
    updates = {name: jnp.ones(getattr(matrix_operator, name).shape, dtype = jnp.float64) 
               for name in matrix_operator._matrix_names()}
    return matrix_operator.replace(**updates)

#get a DiagonalScalar() identity operator with specific nside and theta_pix values
@jax.jit
def get_scalar_identity(matrix_operator):

    nside = matrix_operator.nside
    shape = (nside, nside//2+1)

    return DiagonalScalar(
        fourier_weights = get_fourier_weights(shape),
        nside = nside,
        theta_pix = matrix_operator.theta_pix,
        pix_width = matrix_operator.pix_width,
        scalar_matrix = jnp.ones(shape, dtype = jnp.complex128)
    )

#custom log-determinant method    
@singledispatch
@jax.jit                                                                                                                                                                             
def log_det(matrix_operator):                                     
    raise TypeError(f"Unsupported type: {jax.typeof(matrix_operator)}")

#Temperature only                                                                                                                                                                                            
@log_det.register(DiagonalScalar)
@jax.jit
def _(matrix_operator):                                                                                                                                                                            
    return primal_log_det(matrix_operator.scalar_matrix, matrix_operator.fourier_weights)

#EB Polarization only                                                                                                                                                                                            
@log_det.register(DiagonalEB)
@jax.jit
def _(matrix_operator):
    log_det_EE = primal_log_det(matrix_operator.matrix_EE, matrix_operator.fourier_weights)
    log_det_BB = primal_log_det(matrix_operator.matrix_BB, matrix_operator.fourier_weights)                                                                                                                                                                              
    return log_det_EE + log_det_BB

#Full TEB Polarization method                                                                                                                                                                                            
@log_det.register(BlockTEB)
@jax.jit
def _(matrix_operator):
    log_det_TE_block = block_matrix_logdet(matrix_operator.matrix_TT, matrix_operator.matrix_TE,
                                           matrix_operator.matrix_ET, matrix_operator.matrix_EE,
                                           matrix_operator.fourier_weights)
    log_det_BB_block = primal_log_det(matrix_operator.matrix_BB, matrix_operator.fourier_weights)                                                                                                                                                                              
    return log_det_TE_block + log_det_BB_block

#custom pseudo-inverse methods
@singledispatch
@jax.jit                                                                                                                                                                             
def pinv(matrix_operator):                                     
    raise TypeError(f"Unsupported type: {jax.typeof(matrix_operator)}")

#Temperature only (inverse of a diagonal matrix is just the reciprocal matrix)                                                                                                                                                                                               
@pinv.register(DiagonalScalar)
@jax.jit
def _(matrix_operator):                                                                                                                                                                   
    return matrix_operator.replace(scalar_matrix = reciprocal_matrix(matrix_operator.scalar_matrix))             

#EB Polarization only (inverse of a block diagonal matrix is just the reciprocal matrices)  
@pinv.register(DiagonalEB)
@jax.jit
def _(matrix_operator):                                                                                                                                                                       
    return matrix_operator.replace(matrix_EE = reciprocal_matrix(matrix_operator.matrix_EE),
                                   matrix_BB = reciprocal_matrix(matrix_operator.matrix_BB)) 

#Full TEB Polarization method (inverse requires computing the Schur components)   
@pinv.register(BlockTEB)
@jax.jit
def _(matrix_operator):
    pinv_matrix_TT, pinv_matrix_TE, pinv_matrix_ET, pinv_matrix_EE = invert_block_matrix(matrix_operator.matrix_TT, 
                                                                                         matrix_operator.matrix_TE, 
                                                                                         matrix_operator.matrix_ET, 
                                                                                         matrix_operator.matrix_EE)
    pinv_matrix_BB = reciprocal_matrix(matrix_operator.matrix_BB)                                                                                                                                                                        
    return matrix_operator.replace(matrix_TT = pinv_matrix_TT,
                                   matrix_TE = pinv_matrix_TE,
                                   matrix_ET = pinv_matrix_ET,
                                   matrix_EE = pinv_matrix_EE,
                                   matrix_BB = pinv_matrix_BB)