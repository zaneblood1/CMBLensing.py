import math
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from cmb_lensing.util import *
from cmb_lensing.fields import *
from cmb_lensing.matrix_operators import *
from cmb_lensing.lense_flow import *
from cmb_lensing.dataset import *
from functools import partial

#wrapper function to return an array of percent differences 
#for each of the matrices in the field objects
def percent_diff_2d(ground, predict):
    diffs = []
    for name in ground._matrix_names():
        diffs.append(primal_percent_diff_2d(getattr(ground, name), getattr(predict, name)))
    return jnp.asarray(diffs)

#This function is mostly used in unit testing benchmarks
def primal_percent_diff_2d(ground, predict):
    if jnp.linalg.norm(ground) != 0:
        percent_diff = jnp.linalg.norm(predict - ground) / jnp.linalg.norm(ground)
    else: #avoid divide by zero errors
        percent_diff = jnp.linalg.norm(predict - ground)
    return percent_diff

#NOTE this does not really need to be JIT compiled because it is mostly used in analysis not
#the core gradient descent code
def primal_cross_correlation(field_1, field_2, theta_pix):
    ell_1, cl_1 = primal_power_spectra(field_1, theta_pix)
    _, cl_2 = primal_power_spectra(field_2, theta_pix)
    _, cl_cross = primal_power_spectra(field_1, theta_pix, field_2)
    rho = cl_cross / jnp.sqrt(cl_1 * cl_2)
    return ell_1, rho

#NOTE this does not really need to be JIT compiled because it is mostly used in analysis not
#the core gradient descent code...
#TODO refactor and digest...
def primal_power_spectra(field_1, theta_pix, field_2 = None, delta_l = 50, lmax = 17_000):

    #create our binned ell values
    l_edges = np.arange(0, lmax, delta_l)
    nside, _ = field_1.shape
    if field_2 is None:
        field_2 = field_1

    #convert rfft2 to full fft2
    field_1 = real_fourier_2_full_plane(field_1)
    field_2 = real_fourier_2_full_plane(field_2)

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    ell_grid = real_fourier_2_full_plane(ell_grid)

    scale_factor = nside**2/pix_width**2

    mask_1 = jnp.where(ell_grid.flatten() > jnp.min(l_edges), True, False)
    mask_2 = jnp.where(ell_grid.flatten() < jnp.max(l_edges), True, False)
    total_mask = mask_1 * mask_2
    ell_grid_masked = jnp.real(ell_grid.flatten()[total_mask])
    field_1_masked = field_1.flatten()[total_mask]
    field_2_masked = field_2.flatten()[total_mask]
    
    cl_obs =  jnp.real(field_1_masked * jnp.conj(field_2_masked)) / scale_factor 
    weights = jnp.real(jnp.nan_to_num(1 / (2 / (2*ell_grid_masked + 1)), nan = 0))
    normalization, _ = np.histogram(ell_grid_masked, bins = l_edges, weights = weights)

    cl, _ = np.histogram(ell_grid_masked, bins = l_edges, weights = weights * cl_obs)
    ell, _ = np.histogram(ell_grid_masked, bins = l_edges, weights = weights * ell_grid_masked)
    cl_normalized = cl / normalization
    ell_normalized = ell / normalization

    #now we need to filter off the NaNs
    cl_nan_mask = jnp.where(jnp.isnan(cl_normalized), False, True)
    ell_nan_mask = jnp.where(jnp.isnan(cl_normalized), False, True)
    cl_normalized = cl_normalized[cl_nan_mask]
    ell_normalized = ell_normalized[ell_nan_mask]

    return ell_normalized, cl_normalized

# We mark box_size_deg as static so JAX can use it to determine array shapes
@partial(jax.jit, static_argnums=(1,))
def calculate_unbinned_cl(field, box_size_deg):
    """
    Calculates the unbinned 1D power spectrum Cl from a square 2D realization.
    
    Parameters:
    -----------
    field : jnp.ndarray
        A square (N, N) real-space map realization.
    box_size_deg : float
        The physical side length of the map in degrees (Compile-time static).
    """
    N = field.shape[0]
    
    box_size_rad = jnp.radians(box_size_deg)
    
    # 1. Compute 2D Real FFT
    fft_field = jnp.fft.rfft2(field)
    
    # 2. Compute 2D Power Spectrum P(k)
    p_2d = (box_size_rad**2 / (N**4)) * jnp.abs(fft_field)**2
    
    # 3. Generate the 2D angular frequencies (ell)
    kx = jnp.fft.fftfreq(N, d=box_size_rad/N) * 2 * jnp.pi
    ky = jnp.fft.rfftfreq(N, d=box_size_rad/N) * 2 * jnp.pi
    
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    ell_2d = jnp.sqrt(KX**2 + KY**2)
    ell_2d_int = jnp.round(ell_2d).astype(jnp.int32)
    
    # 4. Calculate max_ell analytically using static Python math.
    # The maximum frequency component is at the corner of the FFT grid (Nyquist * sqrt(2))
    # Using python builtins/numpy here ensures it evaluates to a static int at compile time
    delta_theta_static = math.radians(box_size_deg) / N
    ly_nyquist = math.pi / delta_theta_static
    max_ell = int(math.ceil(ly_nyquist * math.sqrt(2)))
    
    # 5. Flat-sky azimuthal averaging using segment operations
    flat_ell = ell_2d_int.ravel()
    flat_p2d = p_2d.ravel()
    
    # num_segments is now a static Python integer!
    cl_sum = jax.ops.segment_sum(flat_p2d, flat_ell, num_segments=max_ell + 1)
    mode_counts = jax.ops.segment_sum(jnp.ones_like(flat_p2d), flat_ell, num_segments=max_ell + 1)
    
    Cl = jnp.where(mode_counts > 0, cl_sum / mode_counts, 0.0)
    ell = jnp.arange(max_ell + 1)
    
    return ell, Cl

#The objective function we are minimizing during MLE gradient descent
#NOTE there is actually a lot going on under the hood here due to the field overrides
#of plus, minus, (*), (/), and pseudo-inverse... 
@jax.jit
def logpdf(field, phi, data, noise_covariance, 
           phi_covariance, field_covariance, mask, beam, a_phi = 1):

    #rescale phi covariance by the appropriate band power
    phi_covariance = a_phi * phi_covariance

    #Compute the 3 log-determinants
    phi_logdet = log_det(phi_covariance)
    noise_logdet = log_det(noise_covariance)
    field_logdet = log_det(field_covariance)

    #Calculate the f^2 and phi^2 contributions
    field_product = dot(field, pinv(field_covariance) * field)
    phi_product = phi_dot_wrapper(phi, phi_covariance)

    #convert fields to MAP, TQU basis and parametrization to apply lenseflow
    field = map(eb2qu(field))
    phi = map(phi)
    #Once field is lensed fall back to fourier EB basis
    lensed_field = qu2eb(fourier(lense_flow_wrapper(field, phi)))

    #Finally find the inner product term (d - M * B * L * f)^2
    difference = data - mask * beam * lensed_field
    data_product = dot(difference, pinv(noise_covariance) * difference)

    #Return negative one half times the sum of tall six terms
    return -jnp.real(data_product + field_product + phi_product 
                   + noise_logdet + phi_logdet + field_logdet)/2

@jax.custom_vjp
@jax.jit
def phi_dot_wrapper(phi, phi_covar):
    return dot(phi, pinv(phi_covar) * phi)

@jax.jit
def dot_forward(phi, phi_covar):
    #Compute primal output
    dot_value = dot(phi, pinv(phi_covar) * phi)
    #return the value of the dot product and also store any data needed by backward pass.
    return dot_value, (phi, phi_covar)

@jax.jit
def dot_backwards(residuals, cotangent):
    #unpack the necessary data
    (phi, phi_covar) = residuals
    #gradient w.r.t. first field is 2 times the 2nd field and vice versa
    grad_phi = 2 * pinv(phi_covar) * phi

    def dot_only_others(*others):
        return dot(phi, *others)
    _, vjp_function = jax.vjp(dot_only_others, phi_covar)
    other_gradients = vjp_function(cotangent)

    #Return a gradient for EVERY input (either via a custom analytical gradient or via AutoDiff)
    return (cotangent * grad_phi, *other_gradients)

# ----------------------------------------------------------
# Register the custom vjp for the field dot product wrapper
# ----------------------------------------------------------
phi_dot_wrapper.defvjp(dot_forward, dot_backwards)

