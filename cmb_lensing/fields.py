import jax
import jax.numpy as jnp
import flax
from flax import struct
jax.config.update("jax_enable_x64", True)
from cmb_lensing.util import *
from cmb_lensing.constants import (NSIDE_DEFAULT, THETA_PIX_DEFAULT, PIX_WIDTH_DEFAULT, 
                                   FOURIER_WEIGHTS_DEFAULT, FOURIER_MATRIX_DEFAULT)
from functools import singledispatch

##############################################################
#BASE PARENT FIELD WHICH S0, S2, AND S02 ALL INHERIT FROM
##############################################################

@struct.dataclass
class Field:

    fourier_weights: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_WEIGHTS_DEFAULT.copy())
    nside: int = flax.struct.field(pytree_node = False, default = NSIDE_DEFAULT)
    theta_pix: float = flax.struct.field(pytree_node = False, default = THETA_PIX_DEFAULT)
    pix_width: float = flax.struct.field(pytree_node = False, default = PIX_WIDTH_DEFAULT)
    basis: int = flax.struct.field(default_factory = lambda: Basis.FOURIER)

    def _matrix_names(self):
        raise NotImplementedError
    
    #Custom overides for field-on-field and field-on-scalar
    #add (+), subtract(-), multiply (*), and divide (/)
    def __add__(self, other):                                                                                                                    
      if isinstance(other, self.__class__):
          updates = {name: getattr(self, name) + getattr(other, name) for name in self._matrix_names()}
      else:                                                                                                                                    
          updates = {name: getattr(self, name) + other for name in self._matrix_names()}
      return self.replace(**updates)

    #relative order of operands matters for subtraction and division but not multiplication or addition
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
    
    def __truediv__(self, other):                                                                                                                    
      if isinstance(other, self.__class__):
          updates = {name: getattr(self, name) / getattr(other, name) for name in self._matrix_names()}
      else:                                                                                                                                    
          updates = {name: getattr(self, name) / other for name in self._matrix_names()}
      return self.replace(**updates)
    
    def __rtruediv__(self, other):                                                                                                                    
      if isinstance(other, self.__class__):
          updates = {name: getattr(other, name) / getattr(self, name) for name in self._matrix_names()}
      else:                                                                                                                                    
          updates = {name: other / getattr(self, name) for name in self._matrix_names()}
      return self.replace(**updates)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            updates = {name: getattr(self, name) * getattr(other, name) for name in self._matrix_names()}
        else:
            updates = {name: getattr(self, name) * other for name in self._matrix_names()}
        return self.replace(**updates)

    def __rmul__(self, other):
        return self.__mul__(other)
     

###################################
#TEMPERATURE ONLY FIELDS
###################################

@struct.dataclass
class FlatS0(Field):

    scalar_matrix: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    parametrization: int = flax.struct.field(default_factory = lambda: Parametrization.T)

    def _matrix_names(self):
        return ["scalar_matrix"]

###################################
#POLARIZATION ONLY FIELDS
###################################

@struct.dataclass
class FlatS2(Field):

    polar_matrix_1: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    polar_matrix_2: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    parametrization: int = flax.struct.field(default_factory = lambda: Parametrization.EB)

    def _matrix_names(self):
        return ["polar_matrix_1", "polar_matrix_2"]

########################################
#TEMPERATURE AND POLARIZATION FIELDS
########################################

@struct.dataclass
class FlatS02(Field):

    scalar_matrix: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    polar_matrix_1: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    polar_matrix_2: jnp.ndarray = flax.struct.field(default_factory = lambda: FOURIER_MATRIX_DEFAULT.copy())
    parametrization: int = flax.struct.field(default_factory = lambda: Parametrization.EB)

    def _matrix_names(self):
        return ["scalar_matrix", "polar_matrix_1", "polar_matrix_2"]
    
##############################################
#FIELD METHODS
##############################################

@jax.jit
def zero_scalar_field_like(field):
    nside = field.nside
    field_shape = (nside, nside//2+1)
    zeros = FlatS0(
        fourier_weights = get_fourier_weights(field_shape),
        nside = nside,
        theta_pix = field.theta_pix,
        pix_width = field.pix_width,
        scalar_matrix = jnp.zeros(field_shape, dtype = jnp.complex128)
    )
    return zeros
    
#dot product implementation for field objects
@jax.jit
def dot(field_1, field_2):
    dot_sum = 0
    for name in field_1._matrix_names():
        dot_sum += primal_dot(getattr(field_1, name), getattr(field_2, name), 
                              field_1.fourier_weights, field_1.nside**2)
    return dot_sum

#The following block of code is necessary to undo the mathematical operations of taking
#an inner product on a field to get the correct physical gradient.
#NOTE this should only be called while the field is in MAP space but there are
#currently no guard-rails implemented
@jax.jit
def undo_inner_product(field):
    updates = {name: jfft.irfft2(jnp.conj(jfft.rfft2(getattr(field, name)) / field.fourier_weights) * field.nside**2) 
               for name in field._matrix_names()}
    return field.replace(**updates)

#Transform a field's matrices to fourier space
#NOTE this has no guardrails at the moment
@jax.jit
def fourier(field):
    updates = {name: jfft.rfft2(getattr(field, name)) for name in field._matrix_names()}
    return field.replace(
            basis = Basis.FOURIER,
            **updates)

#Transform a field's matrices to map space
#NOTE this has no guardrails at the moment
@jax.jit
def map(field):
    updates = {name: jfft.irfft2(getattr(field, name)) for name in field._matrix_names()}
    return field.replace(
            basis = Basis.MAP,
            **updates)

#Transform an S02 or S2 field from EB to QU basis
#TODO add guard-rails later on but for the time being assume caller is smart
@singledispatch
@jax.jit                                                                                                                                                                             
def eb2qu(field):                                     
    raise TypeError(f"Unsupported type: {jax.typeof(field)}")

#Temperature only                                                                                                                                                                                            
@eb2qu.register(FlatS0)
@jax.jit
def _(field):    
    return field

#Polarization only and full IP params                                                                                                                                                                                          
@eb2qu.register(FlatS2)
@eb2qu.register(FlatS02)
@jax.jit
def _(field):
    e_matrix = field.polar_matrix_1
    b_matrix = field.polar_matrix_2
    q_matrix, u_matrix = primal_eb2qu(e_matrix, b_matrix, field.nside, field.theta_pix)    
    return field.replace(
            polar_matrix_1 = q_matrix,
            polar_matrix_2 = u_matrix,
            parametrization = Parametrization.QU
        )

#Transform an S02 or S2 field from QU to EB basis
#TODO add guard-rails later on but for the time being assume caller is smart
@singledispatch
@jax.jit                                                                                                                                                                             
def qu2eb(field):                                     
    raise TypeError(f"Unsupported type: {jax.typeof(field)}")

#Temperature only                                                                                                                                                                                            
@qu2eb.register(FlatS0)
@jax.jit
def _(field):    
    return field

#Polarization only and full IP params                                                                                                                                                                                       
@qu2eb.register(FlatS2)
@qu2eb.register(FlatS02)
@jax.jit
def _(field):
    q_matrix = field.polar_matrix_1
    u_matrix = field.polar_matrix_2
    e_matrix, b_matrix = primal_qu2eb(q_matrix, u_matrix, field.nside, field.theta_pix)    
    return field.replace(
            polar_matrix_1 = e_matrix,
            polar_matrix_2 = b_matrix,
            parametrization = Parametrization.EB
        )

