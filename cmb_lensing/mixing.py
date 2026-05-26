import jax
from cmb_lensing.fields import *
from cmb_lensing.matrix_operators import *
from cmb_lensing.lense_flow import *

#Move fields into mixed parametrization which is better for convergence of MAP joint algorithm
@jax.jit
def mix(field, phi, mixing_d, mixing_g):

    #Multiply f by D in fourier EB space then convert to QU Map basis
    #before applying lense flow
    df = map(eb2qu(mixing_d * field))
    #fall back to EB fourier space after lensing is done
    mixed_field = qu2eb(fourier(lense_flow(df, map(phi))))
    
    #phi_mixed = G * phi
    mixed_phi = mixing_g * phi

    #return the mixed tuple in fourier space
    return mixed_field, mixed_phi

#Move fields OUT of the mixed parametrization
@jax.jit
def unmix(mixed_field, mixed_phi, mixing_d, mixing_g):
    
    #phi = G^-1 * mixed_phi
    phi = pinv(mixing_g) * mixed_phi

    #f = D^-1 * L^-1 * mixed_f
    #First move from EB Fourier space into QU Map space to apply inverse lense flow
    mixed_field = map(eb2qu(mixed_field))
    field = qu2eb(fourier(lense_flow(mixed_field, map(phi), direction = INVERSE_LENSE)))
    field = pinv(mixing_d) * field

    #return the mixed tuple in fourier space
    return field, phi