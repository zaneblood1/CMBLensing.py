import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from cmb_lensing.fields import *
from cmb_lensing.matrix_operators import *
from cmb_lensing.constants import *
from cmb_lensing.gradients import *
from functools import partial

@jax.jit
def wiener_filter(field, phi, data, field_covariance, noise_covariance, mask, beam):

    #Compute the b vector which is the gradient w.r.t. f of the logpdf 
    #function evaluated at f = 0, d = d, phi = phi, etc...
    b_vector = get_b_vector(phi, data, field_covariance, noise_covariance, mask, beam)

    #use a preconditioner to speed up the calculation
    preconditioner = hessian_logpdf_preconditioner(field_covariance, beam, mask, noise_covariance)

    #take the conjugate gradient of A @ f = b and solve for f assuming
    f_wiener_filtered = conjugate_gradient(field, data, phi, mask, beam, noise_covariance, field_covariance, 
                                           b_vector, preconditioner, maxiter = 500, tol = 1e-1)

    #return the value of f found via conjugate gradient. This is the
    #wiener filtered version of f...
    return f_wiener_filtered

@jax.jit
def get_b_vector(phi, data, field_covariance, noise_covariance, mask, beam):
    #The b vector is the gradient of logpdf w.r.t. f evaluated at f = 0 and other inputs at their current values
    field = 0*data
    grad_f = gradf_logpdf(field, phi, data, field_covariance, noise_covariance, mask, beam)
    return -1*grad_f

@jax.jit
def hessian_logpdf_preconditioner(field_covariance, beam, mask, noise_covariance):
    #find the inverse covariance matrices needed
    cf_inv = pinv(field_covariance)
    cn_inv = pinv(noise_covariance)
    #NOTE we assume that the Daggers of M and B are the same as the original fields M and B...
    #the preconditioner is then equal to Cf^-1 + B^Dagger * M^Dagger * Cn^-1 * M * B
    preconditioner = cf_inv + beam * mask * cn_inv * mask * beam
    return preconditioner

@jax.jit
def A_matrix_operator(field, phi, data, field_covariance, noise_covariance, mask, beam):
    #The A matrix operator is the gradient of logpdf w.r.t. f evaluated at d = 0 and other inputs at their current values
    data = 0*field
    return gradf_logpdf(field, phi, data, field_covariance, noise_covariance, mask, beam)

@partial(jax.jit, static_argnames = ["maxiter", "tol"])
def conjugate_gradient(x0, data, phi, mask, beam, noise_covariance, field_covariance, 
                        b_vector, preconditioner, maxiter = 500, tol = 1e-1):

    #Compute the A matrix which is the gradient w.r.t. f of the logpdf
    #function evaluated at f = f, d = 0, phi = phi, etc..
    def linear_operator(field):
        return A_matrix_operator(field, phi, data, field_covariance, 
                                 noise_covariance, mask, beam)

    Af = linear_operator(x0) 
    r = b_vector - Af #compute the 1st residual base on the initial guess x0
    preconditioner_inv = pinv(preconditioner)
    #compute z = (M^-1 @ r) = (M \ r) since M is diagonal 
    z = preconditioner_inv * r
    #copy the values of z into p
    p = z
    #compute the dot products in fourier space between r and z
    res = dot(r, z)
    #define the initial state
    initial_state = (x0, res, res, p, r, 0)

    #only stop if max iterations hit or residual value drops below
    #the user sepcified tolerance
    def loop_condition(state):
        _, _, res_curr, _, _, step_idx = state
        return jnp.logical_and(res_curr >= tol, step_idx < maxiter)
    
    def main_loop(state):
        x, res, res_curr, p, r, step_idx = state
        #compute Ap = A(p)
        Ap = linear_operator(p)
        #compute alpha = res / dot(p, Ap)
        alpha = res / dot(p, Ap)
        #compute x = x + alpha * p
        x = x + alpha * p
        #compute r = r - alpha * Ap
        r = r - alpha * Ap
        #compute z = (M \ r) = (M^-1 @ r)
        z = preconditioner_inv * r
        #current value of the residual
        res_curr = dot(r, z)
        #update the p value
        res_ratio = (res_curr / res)
        p = z + res_ratio * p
        res = res_curr
        #update the step index
        step_idx += 1
        return (x, res, res_curr, p, r, step_idx)
    
    #perform a jitted while-loop
    final_state = jax.lax.while_loop(loop_condition, main_loop, initial_state)

    #after max iterations have been looped through return the final value for x
    return final_state[0]