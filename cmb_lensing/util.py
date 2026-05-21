import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy.fft as jfft
from cmb_lensing.constants import *
from functools import partial
import math

@jax.jit
def teb_matrix_mult(tt1, te1, et1, ee1, bb1, 
                    tt2, te2, et2, ee2, bb2):
    # [TT_1 TE_1   0  ]    [TT_2 TE_2   0  ]
    # [ET_1 EE_1   0  ] x  [ET_2 EE_2   0  ]
    # [0    0    BB_1 ]    [0    0    BB_2 ]
    result_tt = tt1 * tt2 + te1 * et2
    result_et = et1 * tt2 + ee1 * et2
    result_te = tt1 * te2 + te1 * ee2
    result_ee = et1 * te2 + ee1 * ee2
    result_bb = bb1 * bb2 
    return result_tt, result_te, result_et, result_ee, result_bb

#NOTE The square root of a block diagonal matrix is NOT equivalent to
#the square root of the individual elements. This was found out the painful way...
@jax.jit
def block_matrix_sqrt(a, b, c, d):
    # [ a b ]
    # [ c d ]
    s = jnp.sqrt(a * d - b * c)
    t = jnp.sqrt(a + (d + 2*s))
    t = jnp.where(t != 0, 1/t, 0)
    result_11 = t * (a + s)
    result_12 = t * b
    result_21 = t * c
    result_22 = t * (d + s)
    return result_11, result_12, result_21, result_22

#TODO try to figure out a way to JIT all of these to improve performance...
#compute the log(determinant) of a diagonal matrix
def primal_log_det(matrix, fourier_weights):
    log_det_value = jnp.sum(jnp.where(matrix != 0, jnp.log(jnp.abs(matrix)), 0) * fourier_weights)
    log_det_sign = jnp.prod(jnp.sign(jnp.where(matrix != 0, matrix, 1)))   
    return log_det_sign * log_det_value

#compute the inverse of a diagonal matrix by simply inverting the diagonal entries
def reciprocal_matrix(matrix):
    inv = jnp.where(matrix != 0, 1.0/matrix, 0.0) 
    return inv

#compute the log(determinant) of a 2 x 2 block matrix
#of the form [A & B \\ C & D] where each sub-matrix is assumed to be diagonal
def block_matrix_logdet(a, b, c, d, fourier_weights):
    #mathematically we return ln(det(A)) + ln(det(D - C * A^-1 * B))
    return primal_log_det(a, fourier_weights) + primal_log_det(d - c * reciprocal_matrix(a) * b, fourier_weights)

#compute the inverse of a 2 x 2 block matrix of the form
#[A & B \\ C & D] where each sub-matrix is assumed to be diagonal
#(i.e. implying that their inverses are their reciprocals...)
#these formulas are known as Schur's terms / decomposition
def invert_block_matrix(a, b, c, d):
    #(A - B * D^-1 * C)^-1
    inv_11 = reciprocal_matrix(a - b * reciprocal_matrix(d) * c)
    #D^-1 + D^-1 * C * (A - B * D^-1 * C)^-1 * B * D^-1
    inv_22 = reciprocal_matrix(d) + \
        reciprocal_matrix(d) * c * reciprocal_matrix(a - b * reciprocal_matrix(d) * c) * b * reciprocal_matrix(d)
    #-(A - B * D^-1 * C)^-1 * B * D^-1
    inv_12 = -reciprocal_matrix(a - b * reciprocal_matrix(d) * c) * b * reciprocal_matrix(d)
    #-D^-1 * C * (A - B * D^-1 * C)^-1
    inv_21 = - reciprocal_matrix(d) * c * reciprocal_matrix(a - b * reciprocal_matrix(d) * c)
    return inv_11, inv_12, inv_21, inv_22 

#load files saved on disk with default 64 bit precision
def precision_load(file_path):
    return jnp.array(np.load(file_path))

#convert a flattend matrix of real fourier coefficients back into
#a 2N x N shaped matrix
def reshape_diagonal(diagonal_matrix):
    #apparently ndim is known ahead of time by jax so we do not
    #need to use a jax.lax.cond statement here
    if diagonal_matrix.ndim == 2:
        return diagonal_matrix #avoid re-shaping rectangular matrices
    diagonal_length = diagonal_matrix.shape[0]
    #for some reason, using numpy operations here and casting to
    #int does not throw an error whereas using jnp.astype does...
    num_rows = int(np.sqrt(2*diagonal_length))
    num_cols = num_rows // 2 + 1
    return diagonal_matrix.reshape((num_rows, num_cols))

#for a given field shape, get the default fourier weights
def get_fourier_weights(field_shape):
    rows, _ = field_shape
    length = rows // 2 + 1
    weights = 2 * jnp.ones((length,), dtype = jnp.complex128) 
    weights = weights.at[0].set(1)
    weights = weights.at[-1].set(1)
    return weights

#compute a dot product between two fields in fourier space accounting for the fact
#that points with different fourier weights contribute un-equally
def primal_dot(field_1, field_2, fourier_weights, num_pixels):
    return jnp.real(jnp.sum(jnp.conj(field_1) * field_2 * fourier_weights * (1/num_pixels)))

#get all 1st and 2nd order derivatives of a field in real space
def get_primal_derivatives(field, pix_width):
    N, _ = field.shape
    kx = 2 * jnp.pi * jfft.fftfreq(N, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(N, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    F = jfft.rfft2(field)
    Fx  = jfft.irfft2(1j * KX * F)
    Fy  = jfft.irfft2(1j * KY * F)
    Fxx = jfft.irfft2(-(KX**2) * F)
    Fyy = jfft.irfft2(-(KY**2) * F)
    Fxy = jfft.irfft2(-(KX * KY) * F)
    #return the real valued derivatives
    return Fx, Fy, Fxx, Fxy, Fyy

#Convert EB to QU stokes polarization
def primal_eb2qu(e_field, b_field, nside, theta_pix):
    #generate mesh grid of fourier modes of size (N, N // 2 + 1)
    lx, ly, _ = gen_mesh_grid(nside, theta_pix)
    #NOTE the transformation below mimics the error that Marius' code makes
    #so we can more accurately compare apples to apples :)
    lx = lx.at[:nside//2+1, -1].set(-1*lx[:nside//2+1, -1])
    #compute the arctangent at each point in the meshgrid
    l_angle = jnp.arctan2(lx, ly)
    #Q and U are then given in terms of E and B by a rotation in fourier space
    q_field = -e_field * jnp.cos(2*l_angle) + b_field * jnp.sin(2*l_angle)
    u_field = -e_field * jnp.sin(2*l_angle) - b_field * jnp.cos(2*l_angle)
    return q_field, u_field

#Convert QU to EB stokes polarization
def primal_qu2eb(q_field, u_field, nside, theta_pix):
    #generate mesh grid of fourier modes of size (N, N // 2 + 1)
    lx, ly, _ = gen_mesh_grid(nside, theta_pix)
    #NOTE the transformation below mimics the error that Marius' code makes
    #so we can more accurately compare apples to apples :)
    lx = lx.at[:nside//2+1, -1].set(-1*lx[:nside//2+1, -1])
    #compute the arctangent at each point in the meshgrid
    l_angle = jnp.arctan2(lx, ly)
    #E and B are then given in terms of U and Q by a rotation in fourier space
    e_field = -q_field * jnp.cos(2*l_angle) - u_field * jnp.sin(2*l_angle)
    b_field = q_field * jnp.sin(2*l_angle) - u_field * jnp.cos(2*l_angle)
    return e_field, b_field

def gen_mesh_grid(nside, theta_pix):
    #1 deg = 60' so convert arcmins per pixel to deg per pixel and then deg per pixel to rad per pixel
    d = math.radians(theta_pix / ARCMIN_PER_DEGREE)
    #create an array of real frequencies in the x-direction lx[] using fft.rfreq
    lx = 2 * jnp.pi * jfft.rfftfreq(nside, d) #d is the spacing in radians per pixel
    #create an array of fully complex frequencies in the y-direction ly[] using fft.freq
    ly = 2 * jnp.pi * jfft.fftfreq(nside, d)
    #create a mesh-grid of these frequencies
    lx, ly = jnp.meshgrid(lx, ly) #i.e. repeat lx for length of ly and vice versa
    return lx, ly, d

def gen_ell_grid(nside, theta_pix):
    lx, ly, d = gen_mesh_grid(nside, theta_pix)
    #find the magnitude of total l at each point 
    #in this meshgrid l = sqrt{lx^2+ly^2}
    ls =  jnp.sqrt(lx**2 + ly**2)
    return ls, d

#just do something stupid like this because reflections
#make my head hurt...
def real_fourier_2_full_plane(real_fourier_field):
    full_fourier_field = jfft.fft2(jfft.irfft2(real_fourier_field))
    return full_fourier_field

#JAX-JIT compatible version of a loess smoothing method which 
#creates a smooth distribution of y-value outputs with x-value inputs
#given the corresponding discrete input vectors x and y
@partial(jax.jit, static_argnums=(2, 3))
def loess(x, y, frac=0.3, degree=1):
    n = x.shape[0]
    h = max(math.ceil(frac * n), degree + 1)

    def fit_point(x0):
        dists = jnp.abs(x - x0)
        sorted_dists = jnp.sort(dists)
        max_dist = sorted_dists[h - 1]
        u = jnp.clip(dists / jnp.maximum(max_dist, 1e-10), 0, 1)
        w = (1 - u**3)**3

        V = jnp.stack([x**p for p in range(degree + 1)], axis=-1)
        W = jnp.diag(w)
        VtW = V.T @ W
        beta = jnp.linalg.solve(VtW @ V, VtW @ y)
        x0_vec = jnp.array([x0**p for p in range(degree + 1)])
        return x0_vec @ beta

    return jax.vmap(fit_point)(x)
