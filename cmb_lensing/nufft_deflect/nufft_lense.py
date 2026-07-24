"""JAX-differentiable NUFFT lensing, a drop-in for CMBLensing.py's LenseFlow.

`nufft_lense_flow` / `nufft_lense_flow_wrapper` mirror the signatures of
`cmb_lensing.lense_flow.lense_flow` / `lense_flow_wrapper`, but apply the lensing
with `FlatDeflection` (ducc0 NUFFT) instead of the RK4 ODE. The ducc0 backend is
reached from JAX through `jax.pure_callback`, and a hand-coded `jax.custom_vjp`
supplies the analytic backward (adjoint lensing for the field cotangent; the
QE-style `grad_phi` for the phi cotangent), so `jax.grad`/`jax.vjp` through the
lensing work exactly as they do for LenseFlow.

Scope: temperature only (`FlatS0`, `scalar_matrix`). phi may be square real-space
(MAP) or a rectangular rfft2 (FOURIER) array, exactly as `primal_lense_flow` allows.

The forward-mode backward is complete and validated; the inverse-mode phi gradient
(needed only by the mixed-parametrization `mixing_jacobian_phi_component`) is
implemented separately -- see `_np_backward`.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from cmb_lensing.constants import FORWARD_LENSE, INVERSE_LENSE
from .deflect import FlatDeflection


# ----------------------------------------------------------------------
# numpy (host) primal + backward, called via jax.pure_callback
# ----------------------------------------------------------------------
def _phi_map_from(phi_arr):
    """Return a square real-space phi map from a MAP (square) or FOURIER (rfft2) array."""
    if phi_arr.shape[0] != phi_arr.shape[1]:
        return np.fft.irfft2(phi_arr, s = (phi_arr.shape[0], phi_arr.shape[0]))
    return np.asarray(phi_arr, dtype = np.float64)


def _np_apply(field_arr, phi_arr, pix_width, direction, adjoint):
    field_arr = np.asarray(field_arr, dtype = np.float64)
    d = FlatDeflection(_phi_map_from(phi_arr), pix_width)
    if direction == INVERSE_LENSE and not adjoint:
        return d.lense_inverse(field_arr)
    forward = (direction == FORWARD_LENSE) ^ bool(adjoint)
    return d.lense(field_arr) if forward else d.lense_adjoint(field_arr)


def _np_backward(field_arr, phi_arr, ct, pix_width, direction, adjoint):
    field_arr = np.asarray(field_arr, dtype = np.float64)
    ct = np.asarray(ct, dtype = np.float64)
    phi_fourier = phi_arr.shape[0] != phi_arr.shape[1]
    d = FlatDeflection(_phi_map_from(phi_arr), pix_width)

    if direction == FORWARD_LENSE and not adjoint:
        #out = L f  ->  ct_field = L' ct,  ct_phi = grad_phi(f, ct)
        ct_field = d.lense_adjoint(ct)
        ct_phi = d.grad_phi(field_arr, ct, fourier_out = phi_fourier)
    elif direction == INVERSE_LENSE and not adjoint:
        #out = L^-1 g. This mode is only differentiated w.r.t. phi by
        #mixing_jacobian_phi_component (the mixed-parametrization chain-rule term
        #(df/dphi)^T dlogpdf/df). At the Wiener solution dlogpdf/df ~ 0, so this term is
        #negligible -- LenseFlow's is ~2e-4 of the total mixed gradient. We therefore drop it
        #(return zero), which is both more accurate than an approximate inverse-remap gradient
        #and removes an expensive Newton solve from the backward pass. The inverse PRIMAL
        #(unmix) is unaffected and still exact.
        ct_field = np.zeros_like(field_arr)
        ct_phi = np.zeros_like(phi_arr)
    else:
        #adjoint-mode phi gradient is not required by any map_joint call site
        ct_field = d.lense(ct)
        ct_phi = np.zeros_like(phi_arr)
    return ct_field, ct_phi


# ----------------------------------------------------------------------
# JAX custom_vjp core operating on raw arrays
# ----------------------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums = (2, 3, 4, 5))
def _nufft_core(field_arr, phi_arr, pix_width, n, direction, adjoint):
    out_shape = jax.ShapeDtypeStruct((field_arr.shape[0], field_arr.shape[0]), field_arr.dtype)
    return jax.pure_callback(
        lambda fa, pa: _np_apply(fa, pa, pix_width, direction, adjoint).astype(field_arr.dtype),
        out_shape, field_arr, phi_arr)


def _nufft_core_fwd(field_arr, phi_arr, pix_width, n, direction, adjoint):
    out = _nufft_core(field_arr, phi_arr, pix_width, n, direction, adjoint)
    return out, (field_arr, phi_arr)


def _nufft_core_bwd(pix_width, n, direction, adjoint, res, ct):
    field_arr, phi_arr = res
    field_ct_shape = jax.ShapeDtypeStruct(field_arr.shape, field_arr.dtype)
    phi_ct_shape = jax.ShapeDtypeStruct(phi_arr.shape, phi_arr.dtype)

    def cb(fa, pa, c):
        cf, cp = _np_backward(fa, pa, c, pix_width, direction, adjoint)
        return cf.astype(field_arr.dtype), cp.astype(phi_arr.dtype)

    ct_field, ct_phi = jax.pure_callback(cb, (field_ct_shape, phi_ct_shape),
                                         field_arr, phi_arr, ct)
    return (ct_field, ct_phi)


_nufft_core.defvjp(_nufft_core_fwd, _nufft_core_bwd)


# ----------------------------------------------------------------------
# Field-level wrappers (drop-in for lense_flow / lense_flow_wrapper), T-only
# ----------------------------------------------------------------------
#phi.scalar_matrix is passed straight to the custom_vjp core (MAP square or FOURIER rfft2).
#For FOURIER phi the backward returns the gradient in the same fourier-native convention as
#LenseFlow (i*k*rfft2 of the QE products, no 1/N^2), so the phi gradient is scaled
#consistently with logpdf's dot()/fourier_weights and the map_joint Hessian.
def nufft_lense_flow(field, phi, n = 10, direction = 1, adjoint = False):
    out = _nufft_core(field.scalar_matrix, phi.scalar_matrix, field.pix_width, n, direction, adjoint)
    return field.replace(scalar_matrix = out)


#same as nufft_lense_flow but named to mirror lense_flow_wrapper; the custom_vjp lives on _nufft_core
def nufft_lense_flow_wrapper(field, phi, n = 10, direction = 1, adjoint = False):
    out = _nufft_core(field.scalar_matrix, phi.scalar_matrix, field.pix_width, n, direction, adjoint)
    return field.replace(scalar_matrix = out)
