"""JAX-differentiable NUFFT lensing, a drop-in for CMBLensing.py's LenseFlow.

`nufft_lense_flow` / `nufft_lense_flow_wrapper` mirror the signatures of
`cmb_lensing.lense_flow.lense_flow` / `lense_flow_wrapper`, but apply the lensing
with `FlatDeflection` (ducc0 NUFFT) instead of the RK4 ODE. The ducc0 backend is
reached from JAX through `jax.pure_callback`, and a hand-coded `jax.custom_vjp`
supplies the analytic backward (adjoint lensing for the field cotangent; the
QE-style `grad_phi` for the phi cotangent), so `jax.grad`/`jax.vjp` through the
lensing work exactly as they do for LenseFlow.

Scope: temperature (`FlatS0`, `scalar_matrix`) and polarization (`FlatS2` QU pair,
`FlatS02` T+QU) via `singledispatch`-style shape dispatch in the wrappers. Spin-2 lensing
is two independent scalar remaps of Q and U (no polarization rotation; matches LenseFlow
on the flat sky) with the phi gradient summed over the Q and U legs (`grad_phi_pol`). For
`FlatS02` the T scalar core and the QU pol core share the same phi array, so autodiff sums
the T + Q + U phi-gradient legs automatically. phi may be square real-space (MAP) or a
rectangular rfft2 (FOURIER) array, exactly as `primal_lense_flow` allows.

The forward-mode backward is complete and validated; the inverse-mode phi gradient
(needed only by the mixed-parametrization `mixing_jacobian_phi_component`) is
implemented separately -- see `_np_backward`.
"""
import os
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


#FlatDeflection.__init__ recomputes grad(phi) by FFT and rebuilds the (N^2, 2) coordinate array,
#which at 256^2 is ~10 ms -- about a quarter of the per-call host time. phi is *constant* across
#every CG iteration of a Wiener solve (and across the phi-gradient that follows it), so the same
#deflection is rebuilt dozens of times per outer step. Cache it, keyed on the phi array itself.
#A few slots rather than one because the mixed path evaluates logpdf at more than one phi within a
#step (trial phi in the line search vs the fixed template), which would thrash a 1-slot cache.
_DEFL_CACHE = []
_DEFL_CACHE_MAX = int(os.environ.get("NUFFT_DEFL_CACHE", "4"))   #0 disables (A/B testing)


def _get_deflection(phi_arr, pix_width):
    phi_arr = np.asarray(phi_arr)
    for entry in _DEFL_CACHE:
        p_cached, pw_cached, defl = entry
        if (pw_cached == pix_width and p_cached.shape == phi_arr.shape
                and p_cached.dtype == phi_arr.dtype and np.array_equal(p_cached, phi_arr)):
            return defl
    defl = FlatDeflection(_phi_map_from(phi_arr), pix_width)
    _DEFL_CACHE.append((phi_arr.copy(), pix_width, defl))
    if len(_DEFL_CACHE) > _DEFL_CACHE_MAX:
        _DEFL_CACHE.pop(0)
    return defl


def _np_apply(field_arr, phi_arr, pix_width, direction, adjoint):
    field_arr = np.asarray(field_arr, dtype = np.float64)
    d = _get_deflection(phi_arr, pix_width)
    if direction == INVERSE_LENSE and not adjoint:
        return d.lense_inverse(field_arr)
    forward = (direction == FORWARD_LENSE) ^ bool(adjoint)
    return d.lense(field_arr) if forward else d.lense_adjoint(field_arr)


def _np_backward(field_arr, phi_arr, ct, pix_width, direction, adjoint):
    field_arr = np.asarray(field_arr, dtype = np.float64)
    ct = np.asarray(ct, dtype = np.float64)
    phi_fourier = phi_arr.shape[0] != phi_arr.shape[1]
    d = _get_deflection(phi_arr, pix_width)

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


def _np_apply_pol(q_arr, u_arr, phi_arr, pix_width, direction, adjoint):
    q = np.asarray(q_arr, dtype = np.float64)
    u = np.asarray(u_arr, dtype = np.float64)
    d = _get_deflection(phi_arr, pix_width)
    if direction == INVERSE_LENSE and not adjoint:
        return d.lense_inverse(q), d.lense_inverse(u)
    forward = (direction == FORWARD_LENSE) ^ bool(adjoint)
    return d.lense_pol(q, u) if forward else d.lense_adjoint_pol(q, u)


def _np_backward_pol(q_arr, u_arr, phi_arr, ctq, ctu, pix_width, direction, adjoint):
    q = np.asarray(q_arr, dtype = np.float64)
    u = np.asarray(u_arr, dtype = np.float64)
    ctq = np.asarray(ctq, dtype = np.float64)
    ctu = np.asarray(ctu, dtype = np.float64)
    phi_fourier = phi_arr.shape[0] != phi_arr.shape[1]
    d = _get_deflection(phi_arr, pix_width)

    if direction == FORWARD_LENSE and not adjoint:
        #out = L (q,u) -> ct_fields = L'(ctq,ctu),  ct_phi = summed Q+U QE legs
        cfq, cfu = d.lense_adjoint_pol(ctq, ctu)
        cp = d.grad_phi_pol(q, u, ctq, ctu, fourier_out = phi_fourier)
    elif direction == INVERSE_LENSE and not adjoint:
        #inverse phi-gradient dropped (see _np_backward); inverse primal unaffected
        cfq = np.zeros_like(q); cfu = np.zeros_like(u); cp = np.zeros_like(phi_arr)
    else:
        cfq, cfu = d.lense_pol(ctq, ctu)
        cp = np.zeros_like(phi_arr)
    return cfq, cfu, cp


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
# spin-2 (QU) custom_vjp core: two field arrays through one pure_callback
# ----------------------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums = (3, 4, 5, 6))
def _nufft_core_pol(q_arr, u_arr, phi_arr, pix_width, n, direction, adjoint):
    N = q_arr.shape[0]
    out_shape = (jax.ShapeDtypeStruct((N, N), q_arr.dtype), jax.ShapeDtypeStruct((N, N), u_arr.dtype))
    def cb(qa, ua, pa):
        qo, uo = _np_apply_pol(qa, ua, pa, pix_width, direction, adjoint)
        return qo.astype(q_arr.dtype), uo.astype(u_arr.dtype)
    return jax.pure_callback(cb, out_shape, q_arr, u_arr, phi_arr)


def _nufft_core_pol_fwd(q_arr, u_arr, phi_arr, pix_width, n, direction, adjoint):
    out = _nufft_core_pol(q_arr, u_arr, phi_arr, pix_width, n, direction, adjoint)
    return out, (q_arr, u_arr, phi_arr)


def _nufft_core_pol_bwd(pix_width, n, direction, adjoint, res, ct):
    q_arr, u_arr, phi_arr = res
    ctq, ctu = ct
    shapes = (jax.ShapeDtypeStruct(q_arr.shape, q_arr.dtype),
              jax.ShapeDtypeStruct(u_arr.shape, u_arr.dtype),
              jax.ShapeDtypeStruct(phi_arr.shape, phi_arr.dtype))
    def cb(qa, ua, pa, cq, cu):
        cfq, cfu, cp = _np_backward_pol(qa, ua, pa, cq, cu, pix_width, direction, adjoint)
        return cfq.astype(q_arr.dtype), cfu.astype(u_arr.dtype), cp.astype(phi_arr.dtype)
    cfq, cfu, cp = jax.pure_callback(cb, shapes, q_arr, u_arr, phi_arr, ctq, ctu)
    return (cfq, cfu, cp)


_nufft_core_pol.defvjp(_nufft_core_pol_fwd, _nufft_core_pol_bwd)


# ----------------------------------------------------------------------
# Field-level wrappers (drop-in for lense_flow / lense_flow_wrapper)
# ----------------------------------------------------------------------
#phi.scalar_matrix is passed straight to the custom_vjp core(s) (MAP square or FOURIER rfft2).
#For FOURIER phi the backward returns the gradient in the same fourier-native convention as
#LenseFlow (i*k*rfft2 of the QE products, no 1/N^2), so the phi gradient is scaled
#consistently with logpdf's dot()/fourier_weights and the map_joint Hessian.
#Dispatch on the field's matrices: scalar_matrix -> T scalar core; polar_matrix_1/2 -> QU pol
#core. For FlatS02 both run on the SAME phi array, so autodiff sums the T + Q + U phi-grad legs.
def _nufft_dispatch(field, phi, n, direction, adjoint):
    names = field._matrix_names()
    phi_arr = phi.scalar_matrix
    pw = field.pix_width
    updates = {}
    if "scalar_matrix" in names:
        updates["scalar_matrix"] = _nufft_core(field.scalar_matrix, phi_arr, pw, n, direction, adjoint)
    if "polar_matrix_1" in names:
        qo, uo = _nufft_core_pol(field.polar_matrix_1, field.polar_matrix_2, phi_arr, pw, n, direction, adjoint)
        updates["polar_matrix_1"] = qo
        updates["polar_matrix_2"] = uo
    return field.replace(**updates)


def nufft_lense_flow(field, phi, n = 10, direction = 1, adjoint = False):
    return _nufft_dispatch(field, phi, n, direction, adjoint)


#same as nufft_lense_flow but named to mirror lense_flow_wrapper; the custom_vjp lives on the cores
def nufft_lense_flow_wrapper(field, phi, n = 10, direction = 1, adjoint = False):
    return _nufft_dispatch(field, phi, n, direction, adjoint)
