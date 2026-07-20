#Step 1 mass matrix: the diagonal of the true mixed-phi posterior precision, estimated by
#Hessian-vector products, replacing the analytic approximation in get_mass_matrix
#(M = pinv(G)^2 (pinv(Cphi) + pinv(Nphi))).
#
#Why HVP-diagonal beats the analytic M: the analytic M uses Nphi (the quadratic-estimate /
#N0 noise), a fixed, field-averaged, isotropic stand-in for the lensing information. The
#TRUE curvature d^2(-logpdf)/d(phi_mixed)^2 depends on the actual field realization and the
#mask, and where the analytic M under-estimates it, those modes are stiff and the leapfrog
#is unstable there (the "smaller eps / more steps didn't help" symptom). This module measures
#the real per-mode curvature directly.
#
#Why finite-difference HVP (not jax.jvp): mixed_grad_phi_logpdf runs through
#lense_flow_wrapper, which defines a custom_vjp (reverse mode only). Forward-mode jvp -- the
#usual HVP primitive -- is not defined through a custom_vjp and raises. A central finite
#difference of the (validated) analytic gradient avoids higher-order autodiff entirely:
#    H v ~= (grad(phi + eps v) - grad(phi - eps v)) / (2 eps).
#The posterior is ~quadratic near the mode, so the FD is accurate over a wide eps range;
#build_hvp_mass_matrix exposes eps and there is an eps-scan helper to confirm stability.

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
import matplotlib.pyplot as plt

from cmb_lensing.gradients import mixed_grad_phi_logpdf
from cmb_lensing.util import get_k_meshgrid
from cmb_lensing.mode_diagnostics import radial_profile


def _grad_matrix_closure(mixed_phi, mixed_temp, data_field, args):
    #the mixed-phi gradient as a pure function of the rfft2 array, all else held fixed.
    #this is exactly the force the HMC leapfrog integrates.
    def grad_fn(phi_mat):
        mp = mixed_phi.replace(scalar_matrix = phi_mat)
        g = mixed_grad_phi_logpdf(mixed_temp, mp, data_field,
                                  args["noise_covariance"], args["phi_covariance"],
                                  args["field_covariance"], args["mask"], args["beam"],
                                  args["mixing_d"], args["mixing_g"])
        return g.scalar_matrix
    return grad_fn


def _hutchinson_fd_diag(grad_fn, phi0, nside, rng_key, num_probes, eps, probe_scale):
    #core estimator: averaged Rayleigh quotient Re(conj(v).Hv)/|v|^2 with a central FD HVP
    #of grad_fn, over Hermitian (real-field rfft2) probes. Returns diag(H), a real array.
    def body(i, carry):
        num, den, key = carry
        key, sub = jax.random.split(key)
        #Rademacher real-space probe -> Hermitian rfft2, rescaled to probe_scale amplitude
        v_real = jnp.sign(jax.random.normal(sub, (nside, nside)))
        v = jfft.rfft2(v_real)
        v = v * (probe_scale / jnp.sqrt(jnp.mean(jnp.abs(v) ** 2)))

        Hv = (grad_fn(phi0 + eps * v) - grad_fn(phi0 - eps * v)) / (2.0 * eps)
        num = num + jnp.real(jnp.conj(v) * Hv)
        den = den + jnp.abs(v) ** 2
        return num, den, key

    num0 = jnp.zeros(phi0.shape, dtype = jnp.float64)
    den0 = jnp.zeros(phi0.shape, dtype = jnp.float64)
    num, den, _ = jax.lax.fori_loop(0, num_probes, body, (num0, den0, rng_key))
    return num / jnp.where(den > 0, den, 1.0)


def diagonal_hvp_precision(mixed_phi, mixed_temp, data_field, args, rng_key,
                           num_probes = 16, eps = 1e-2, probe_scale = None):
    #Hutchinson estimate of the per-mode diagonal of the mixed-phi posterior precision
    #(-Hessian of logpdf) at mixed_phi. Returns a real (N, N//2+1) array.
    #
    #probes are drawn like the HMC momentum -- rfft2 of a real Rademacher field -- so they
    #live in the same Hermitian subspace the sampler moves in.
    phi0 = mixed_phi.scalar_matrix
    nside = mixed_phi.nside

    grad_fn = _grad_matrix_closure(mixed_phi, mixed_temp, data_field, args)

    #reference scale so eps is a fractional perturbation even if phi0 is near zero
    if probe_scale is None:
        rms_phi = jnp.sqrt(jnp.mean(jnp.abs(phi0) ** 2))
        probe_scale = jnp.maximum(rms_phi, 1e-12)

    diag_H = _hutchinson_fd_diag(grad_fn, phi0, nside, rng_key, num_probes, eps, probe_scale)
    #precision = -Hessian (logpdf is concave at the mode, so diag_H is negative)
    return -diag_H


def _radial_smooth(values, ell, nbins = 60):
    #replace each mode by the mean over its |ell| annulus (isotropy denoising). Keeps the
    #estimate stable at modest probe counts; drop it if you suspect real anisotropy.
    v = np.asarray(values)
    l = np.asarray(ell)
    edges = np.geomspace(max(l[l > 0].min(), 1e-6), l.max(), nbins + 1)
    idx = np.clip(np.digitize(l.ravel(), edges) - 1, 0, nbins - 1)
    out = v.ravel().copy()
    for b in range(nbins):
        sel = idx == b
        if sel.any():
            out[sel] = np.mean(v.ravel()[sel])
    return out.reshape(v.shape)


def build_hvp_mass_matrix(analytic_mass, mixed_phi, mixed_temp, data_field, args, rng_key,
                          num_probes = 16, eps = 1e-2, floor_frac = 0.1,
                          radial_smooth = True):
    #Returns a drop-in replacement for get_mass_matrix's output: the analytic mass operator
    #with its scalar_matrix swapped for the HVP-diagonal precision.
    #
    #The analytic M is used as a stability FLOOR (M >= floor_frac * analytic): where the HVP
    #is noisy or a mode is data-unconstrained (e.g. masked -> prior-limited), we never let the
    #mass collapse below a fraction of the prior+QE precision the analytic M already encodes.
    precision = diagonal_hvp_precision(mixed_phi, mixed_temp, data_field, args, rng_key,
                                       num_probes = num_probes, eps = eps)

    analytic = analytic_mass.scalar_matrix
    precision = jnp.real(precision)

    if radial_smooth:
        KX, KY = (np.asarray(a) for a in get_k_meshgrid(mixed_phi.nside, args["pix_width"]))
        ell = np.sqrt(KX ** 2 + KY ** 2)
        precision = jnp.asarray(_radial_smooth(np.asarray(precision), ell))

    floor = floor_frac * jnp.real(analytic)
    m_diag = jnp.maximum(precision, floor)
    #keep the exact dtype/shape the momentum draw and kinetic term expect
    m_diag = m_diag.astype(analytic.dtype)
    return analytic_mass.replace(scalar_matrix = m_diag)


#------------------ frozen-R correction (for joint theta + phi sampling) -----------------

def build_hvp_correction_factor(analytic_mass, mixed_phi, mixed_temp, data_field, args, rng_key,
                                num_probes = 16, eps = 1e-2, floor_frac = 0.1, cap = 1e3,
                                radial_smooth = True):
    #R = HVP_precision / analytic_M, the expensive per-mode "shape" correction the analytic
    #M misses. R is far more theta-stable than either factor alone (both scale together with
    #theta), so it can be frozen while the cheap analytic M(theta) is refreshed every sweep.
    #Returned as a real (N, N//2+1) array, clipped to [floor_frac, cap] for stability.
    precision = jnp.real(diagonal_hvp_precision(mixed_phi, mixed_temp, data_field, args,
                                                rng_key, num_probes = num_probes, eps = eps))
    precision = jnp.maximum(precision, 0.0)

    if radial_smooth:
        KX, KY = (np.asarray(a) for a in get_k_meshgrid(mixed_phi.nside, args["pix_width"]))
        ell = np.sqrt(KX ** 2 + KY ** 2)
        precision = jnp.asarray(_radial_smooth(np.asarray(precision), ell))

    analytic = jnp.real(analytic_mass.scalar_matrix)
    R = precision / jnp.where(analytic > 0, analytic, 1.0)
    return jnp.clip(R, floor_frac, cap)


def apply_correction(analytic_mass, R):
    #M(theta) = R (frozen) elementwise-times the current-theta analytic mass operator.
    a = analytic_mass.scalar_matrix
    m = (R * jnp.real(a)).astype(a.dtype)
    return analytic_mass.replace(scalar_matrix = m)


#------------------ inspection helpers (run before committing to the new M) ---------------

def compare_to_analytic(analytic_mass, mixed_phi, mixed_temp, data_field, args, rng_key,
                        out_dir, num_probes = 16, eps = 1e-2, tag = "hvp"):
    #Plot the HVP-diagonal precision against the analytic M per |ell|. Ratio > 1 marks modes
    #the analytic M UNDER-preconditions (too soft -> stiff/unstable in the leapfrog) -- exactly
    #the modes a better mass matrix must stiffen.
    precision = np.asarray(diagonal_hvp_precision(mixed_phi, mixed_temp, data_field, args,
                                                  rng_key, num_probes = num_probes, eps = eps))
    analytic = np.real(np.asarray(analytic_mass.scalar_matrix))

    KX, KY = (np.asarray(a) for a in get_k_meshgrid(mixed_phi.nside, args["pix_width"]))
    ell = np.sqrt(KX ** 2 + KY ** 2)

    ratio = precision / np.where(analytic != 0, analytic, np.nan)
    c, ratio_prof, counts = radial_profile(ratio, ell, nbins = 30)
    _, hvp_prof, _ = radial_profile(precision, ell, nbins = 30)
    _, an_prof, _ = radial_profile(analytic, ell, nbins = 30)

    ok = counts > 0
    print("=" * 70)
    print(f"[hvp_mass_matrix] HVP precision / analytic M  vs |ell|  (>1 = analytic too soft)")
    for cc, rr, ct in zip(c[ok], ratio_prof[ok], counts[ok]):
        if np.isfinite(rr):
            print(f"    ell~{cc:8.1f}  ratio={rr:8.3f}  n={ct:5d}")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    axes[0].plot(c[ok], hvp_prof[ok], marker = "o", label = "HVP diagonal precision")
    axes[0].plot(c[ok], an_prof[ok], marker = "s", label = "analytic M")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("|ell|"); axes[0].set_ylabel("per-mode mass (precision)")
    axes[0].set_title(f"{tag}: HVP vs analytic mass"); axes[0].legend()

    axes[1].plot(c[ok], ratio_prof[ok], marker = "o")
    axes[1].axhline(1.0, color = "grey", ls = "--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("|ell|"); axes[1].set_ylabel("HVP / analytic")
    axes[1].set_title(f"{tag}: ratio (>1 = analytic under-preconditions)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{tag}_mass_matrix_compare.png", dpi = 120)
    plt.close()
    return {"precision": precision, "analytic": analytic, "ell": ell,
            "ratio_profile": (c, ratio_prof, counts)}


def eps_scan(mixed_phi, mixed_temp, data_field, args, rng_key,
             eps_values = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3), num_probes = 4):
    #FD-HVP should be insensitive to eps where it is accurate. Prints the median per-mode
    #precision at each eps; pick eps in the flat region (too large -> nonlinearity bias,
    #too small -> roundoff).
    print("[hvp_mass_matrix] eps scan (median precision should be ~flat in the good range):")
    for e in eps_values:
        p = diagonal_hvp_precision(mixed_phi, mixed_temp, data_field, args, rng_key,
                                   num_probes = num_probes, eps = e)
        p = np.asarray(p)
        med = np.median(p[np.isfinite(p)])
        print(f"    eps={e:8.1e}  median_precision={med:12.5e}")
