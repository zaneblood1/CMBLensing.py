import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from cmb_lensing.fields import *
from cmb_lensing.lense_flow import *
from cmb_lensing.matrix_operators import *
from cmb_lensing.statistics import logpdf
from cmb_lensing.mixing import *

#TODO it would be nice to refactor this so that we do not have to constantly
#manually convert between QU, EB and Fourier, Map...
@jax.jit
def gradf_logpdf(field, phi, data, field_covariance, noise_covariance, mask, beam):
    #gradf is analytically equal to L^dagger * B^dagger * M^dagger * Cn^-1 * (d - M * B * L * f) - Cf^-1 * f
    #First convert to QU Map basis and lense the field, then fall back to EB Fourier basis
    field = map(eb2qu(field))
    phi = map(phi)
    lensed_field = qu2eb(fourier(lense_flow(field, phi)))
    #Compute B * M * Cn^-1 * (d - M * B * L * f)
    grad_f = beam * mask * pinv(noise_covariance) * (data - mask * beam * lensed_field)
    #Now convert this to QU Map basis again, adjoint lense the field, and fall back to EB fourier basis again
    grad_f = map(eb2qu(grad_f))
    grad_f = lense_flow(grad_f, phi, adjoint = True)
    #Finally subtract off Cf^-1
    grad_f = qu2eb(fourier(grad_f)) - pinv(field_covariance) * qu2eb(fourier(field))
    return grad_f

#Safe container for the phi gradient
@jax.jit
def grad_phi_logpdf(unlensed_field, phi, data, noise_covariance, 
                    phi_covariance, field_covariance, mask, beam):
    grad_phi = jax.grad(logpdf, argnums = 1, allow_int = True)(unlensed_field, phi, data, noise_covariance, 
                                                               phi_covariance, field_covariance, mask, beam)
    #TODO figure out why parametrization, weights, and basis fields are being destroyed here...
    return grad_phi.replace(fourier_weights = phi.fourier_weights,
                            basis = Basis.FOURIER, 
                            parametrization = Parametrization.T)

#Safe container for the MIXED phi gradient
@jax.jit
def mixed_grad_phi_logpdf(mixed_field, mixed_phi, data, noise_covariance, 
                            phi_covariance, field_covariance, 
                            mask, beam, mixing_d, mixing_g):
    
    #first unmix the fields
    field, phi = unmix(mixed_field, mixed_phi, mixing_d, mixing_g)

    #compute the phi gradient in the unlensed / unmixed parametrization
    grad_phi = grad_phi_logpdf(field, phi, data, noise_covariance, 
                                phi_covariance, field_covariance, mask, beam)

    #compute the f gradient in the unlensed / unmixed parametrization
    grad_f = gradf_logpdf(field, phi, data, field_covariance, 
                          noise_covariance, mask, beam)

    #add these terms together using the Jacobian factors from the chain rule:
    #d(logpdf)/d(phi_m) = d(logpdf)/d(phi) * d(phi)/d(phi_m) 
    #                   + d(logpdf)/d(f) * d(f)/d(phi) * d(phi)/d(phi_m)
    #                   = G^-1 * (grad_phi + jacobian * grad_f)
    chain_rule = mixing_jacobian_phi_component(mixed_field, phi, grad_f, mixing_d)
    grad_phi_mixed = pinv(mixing_g) * (grad_phi + chain_rule)
    return grad_phi_mixed

#Mathematically this function computes d/d_phi (unmix(f, phi).f) which is an (nside x nside) x (nside x nside)
#Jacobian term... Materializing this Jacobian would cause an OOM error so we use the jax.vjp method which
#is equivalent to the Julia Zygote.pullback method
@jax.jit
def mixing_jacobian_phi_component(mixed_field, phi, grad_f, mixing_d):

    #we need to use a Fourier --to--> Map method here since we require phi to be in Fourier
    #space to preserve the physically meaningful imaginary components of the DC and Nyquist columns
    def unmix_partial(phi):
        field = map(eb2qu(mixed_field))
        #since we are using AD here we need to use the custom wrapper that tells the 
        #AD graph how to function properly
        field = lense_flow_wrapper(field, phi, n = 10, direction = INVERSE_LENSE)
        field = qu2eb(fourier(field))
        field = pinv(mixing_d) * field
        return map(field)
    
    #call the vjp with phi in Fourier space and apply to grad_f in real space. The
    #cotangent is already a fourier field so no fourier() conversion is needed
    _, pullback = jax.vjp(unmix_partial, phi)
    (differential,) = pullback(map(grad_f))
    return differential