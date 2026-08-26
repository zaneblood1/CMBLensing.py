import camb
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
jax.config.update("jax_enable_x64", True)
from functools import partial
from itertools import combinations
import multiprocessing
import time
from cmb_lensing.util import *
from cmb_lensing.lense_flow import *
from cmb_lensing.dataset import *
from cmb_lensing.statistics import *

_CAMB_COLS = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}


# ── Power Spectrum Conversion ─────────────────────────────────────────────

def dl2cl(dl_xx, lmax, lmax_prime, is_phi=False):
    ell = jnp.arange(2, lmax).astype(jnp.float64)
    ell_prime = jnp.arange(2, lmax_prime).astype(jnp.float64)

    if is_phi:
        cl_xx = dl_xx[2:] * 2 * jnp.pi / ell_prime**4
    else:
        cl_xx = dl_xx[2:] * 2 * jnp.pi / (ell_prime * (ell_prime + 1))

    def exponential_interpolate(cl_xx):
        return jnp.exp(jnp.interp(
            jnp.log(ell), jnp.log(ell_prime), jnp.log(cl_xx),
            left="extrapolate", right="extrapolate"
        ))

    def linear_interpolate(cl_xx):
        return jnp.interp(ell, ell_prime, cl_xx, left=0.0, right=0.0)

    cl_xx = jax.lax.cond(
        jnp.all(cl_xx > 0),
        exponential_interpolate,
        linear_interpolate,
        cl_xx
    )

    return cl_xx


def beam_cls(beam_fwhm, ell):
    return jnp.exp(-ell**2 * jnp.deg2rad(beam_fwhm / ARCMIN_PER_DEGREE)**2 / BEAM_TRANSFER_SCALAR)


def noise_cls(lmax_prime, uk_arcmin_t, beam_fwhm=0, l_knee=100, alpha_knee=3):
    ell_prime = jnp.arange(2, lmax_prime)
    bls = beam_cls(beam_fwhm, ell_prime)
    #NOTE temporarily remove 1/f noise for LCDM parameter inference
    nls = 1 + (l_knee / ell_prime)**alpha_knee
    cn_tt = jnp.nan_to_num(
        jnp.deg2rad(uk_arcmin_t / ARCMIN_PER_DEGREE)**2 * nls / bls,
        nan=0.0, posinf=0.0, neginf=0.0
    )
    cn_te = jnp.zeros(cn_tt.shape)
    cn_ee = 2 * cn_tt
    cn_bb = 2 * cn_tt
    return cn_tt, cn_te, cn_ee, cn_bb


# ── Covariance and Field Generation ──────────────────────────────────────

@partial(jax.jit, static_argnames=["nside", "rescale"])
def covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cls, origin_value=None,
                          rescale=True, use_linear_interpolation=False, fill_value=None):
    shape = (nside, nside // 2 + 1)

    def exponential_interpolate(ell_grid, ells, cls):
        ls_safe = jnp.where(ell_grid.flatten() == 0, 1e-10, ell_grid.flatten())
        return jnp.exp(jnp.interp(
            jnp.log(ls_safe), jnp.log(ells), jnp.log(cls),
            left="extrapolate", right="extrapolate"
        )).reshape(shape)

    result = exponential_interpolate(ell_grid, ells, cls)

    if origin_value is not None:
        result = result.at[0, 0].set(origin_value)
    if rescale:
        result = result / pix_width**2
    return result


#covar_matrix_from_cls interpolates in log(Cl), so an identically-zero spectrum (unlensed
#scalar BB at r = 0, the TE noise Cls) would come out NaN, not zero. This wrapper maps
#all-zero spectra to an exactly zero covariance instead. The value-level check forces
#eager evaluation, so this must only be called OUTSIDE jit (load_sim is eager)
def _covar_or_zeros(nside, pix_width, ell_grid, ells, cls, origin_value=None):
    if bool(jnp.all(cls == 0)):
        return jnp.zeros((nside, nside // 2 + 1))
    return covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cls,
                                 origin_value=origin_value)

#TE crosses zero, so it cannot go through the log-interpolating covariance builder. It is
#carried instead as the correlation ratio rho = Cl_TE / sqrt(Cl_TT * Cl_EE): bounded in
#(-1, 1), smooth, and interpolated LINEARLY onto the 2D ell grid (flat beyond the last
#ell - the log-log extrapolation used for TT/EE has no analogue for a sign-changing
#quantity, and those scales are noise-dominated anyway). Scaling by sqrt(cf_tt * cf_ee)
#reconstructs the TE covariance from the SAME TT/EE the model uses everywhere else, so
#|rho| <= 1 makes the 2x2 T/E block positive semi-definite BY CONSTRUCTION - an
#independently interpolated TE could overshoot sqrt(TT * EE) and silently break
#invert_block_matrix / block_matrix_logdet
@jax.jit
def te_covar_from_rho(rho, ells, ell_grid, cf_tt, cf_ee):
    rho_grid = jnp.interp(ell_grid.flatten(), ells, rho).reshape(cf_tt.shape)
    cf_te = rho_grid * jnp.sqrt(cf_tt * cf_ee)
    return cf_te.at[0, 0].set(0.0)


def te_rho_from_cls(cl_te, cl_tt, cl_ee):
    """Correlation ratio rho(ell) from raw Cls, zero where TT or EE has no power."""
    denominator = jnp.sqrt(cl_tt * cl_ee)
    return jnp.where(denominator > 0, cl_te / jnp.where(denominator > 0, denominator, 1.0),
                     0.0)


def field_from_covar(nside, covar_matrix, rng_keys, key_counter):
    seed = rng_keys[key_counter]
    key_counter = key_counter + 1
    field = field_from_covar_single_key(nside, covar_matrix, seed)
    return field, key_counter

def field_from_covar_single_key(nside, covar_matrix, seed):
    #draw a real Gaussian field with power spectrum covar_matrix. building it as the rfft2 of
    #REAL white noise gives the correct Hermitian symmetry for free -- including the two
    #self-conjugate columns kx=0 and kx=Nyquist. the previous version drew every half-plane
    #mode as an independent complex Gaussian; irfft2 then symmetrized those two columns and
    #halved their variance (bulk modes = C, kx=0 / kx=Nyquist columns = C/2), which biased
    #every sampled field, noise realization, and HMC momentum. white-noise rfft2 has
    #E|F_k|^2 = nside^2 uniformly over all modes, so the effective per-mode variance downstream
    #(callers rfft2 the returned map) is nside^2 * covar_matrix everywhere -- same bulk
    #normalization as before, with the self-conjugate columns now at full power.
    white = jax.random.normal(seed, shape = (nside, nside))
    field = jfft.irfft2(jfft.rfft2(white) * jnp.sqrt(covar_matrix))
    return field

# ── Instrument Response ───────────────────────────────────────────────────

def cos_ramp_up(length):
    return (jnp.array([jnp.cos(x) for x in jnp.linspace(jnp.pi, 0, length)]) + 1) / 2

def cos_ramp_down(length):
    return 1 - cos_ramp_up(length)

def low_pass(l_cutoff, delta_l=50):
    return jnp.concatenate([jnp.ones(l_cutoff - delta_l + 1), cos_ramp_down(delta_l)])

def get_mask(l_cutoff, nside, pix_width, ell_grid):
    screen_cls = low_pass(l_cutoff)
    ell = jnp.arange(2, len(screen_cls)).astype(jnp.float64)
    return covar_matrix_from_cls(nside, pix_width, ell_grid, ell,
                                 screen_cls[2:], origin_value=1, rescale=False)

def get_beam(nside, pix_width, ell_grid, lmax_prime, beam_fwhm=0):
    ell_prime = jnp.arange(2, lmax_prime).astype(jnp.float64)
    b = jnp.sqrt(beam_cls(beam_fwhm, ell_prime))
    return covar_matrix_from_cls(nside, pix_width, ell_grid, ell_prime, b,
                                 origin_value=0, rescale=False,
                                 use_linear_interpolation=True, fill_value=1.0)


# ── D Matrix ──────────────────────────────────────────────────────────────

def get_d_matrix(cf_tt, cf_te, cf_ee, cf_bb, cn_tt, cn_te, cn_ee, cn_bb):
    pre_factor = jnp.deg2rad(5 / ARCMIN_PER_DEGREE)**2
    identity = jnp.ones(cn_tt.shape)

    cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee = invert_block_matrix(cf_tt, cf_te, cf_te, cf_ee)
    cf_inv_bb = reciprocal_matrix(cf_bb)
    cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee, cf_inv_bb = [
        jnp.nan_to_num(x, nan=0, posinf=0, neginf=0)
        for x in (cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee, cf_inv_bb)
    ]

    sum_tt = cf_tt + pre_factor * identity + 2 * cn_tt
    sum_te = cf_te + 2 * cn_te
    sum_ee = cf_ee + pre_factor * identity + 2 * cn_ee
    sum_bb = cf_bb + pre_factor * identity + 2 * cn_bb

    d_tt, d_te, d_et, d_ee, d_bb = teb_matrix_mult(
        sum_tt, sum_te, sum_te, sum_ee, sum_bb,
        cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee, cf_inv_bb
    )

    # Uses d_et for both off-diagonal entries — empirically matches Julia results
    d_tt, d_te, _, d_ee = block_matrix_sqrt(d_tt, d_et, d_et, d_ee)
    d_bb = jnp.sqrt(d_bb)

    return d_tt, d_te, d_ee, d_bb

@jax.jit
def get_d_tt_matrix(cf_curr, cn_tt):
    pre_factor = jnp.deg2rad(5 / ARCMIN_PER_DEGREE)**2
    identity = jnp.ones(cn_tt.shape)
    d_tt = cf_curr + pre_factor * identity + 2 * cn_tt
    d_tt = jnp.sqrt(d_tt * reciprocal_matrix(cf_curr))
    return d_tt

#D mixing matrix for T + P sampling with a full (nonzero-TE) 2x2 T/E block, following
#get_d_matrix's validated algorithm: D^2 = (Cf + sigma_len^2 I + 2 Cn) * Cf^-1 on the
#coupled T/E block (Schur inverse + block square root, with the same d_et choice for
#both off-diagonals that empirically matches Julia). The unlensed scalar B field has
#zero power at r = 0, so its block is set to the IDENTITY where cf_bb = 0: the B prior
#is a delta at zero and the field carries no B power to whiten, any invertible choice is
#valid there, and the identity keeps mix/unmix invertible while contributing nothing to
#logdet(D). (The naive formula would give d_bb = 0 via the pinv-style reciprocal, which
#zeroes the mixed B sector.) Nonzero cf_bb entries (e.g. a future tensor-power run)
#whiten normally. With cf_te = 0 this reduces exactly to the previous diagonal-block
#version (block_matrix_sqrt of a diagonal block is the elementwise sqrt)
@jax.jit
def get_d_teb_matrix(cf_tt, cf_te, cf_ee, cf_bb, cn_tt, cn_te, cn_ee, cn_bb):
    pre_factor = jnp.deg2rad(5 / ARCMIN_PER_DEGREE)**2
    identity = jnp.ones(cn_tt.shape)

    cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee = invert_block_matrix(cf_tt, cf_te,
                                                                     cf_te, cf_ee)
    cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee = [
        jnp.nan_to_num(x, nan=0, posinf=0, neginf=0)
        for x in (cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee)
    ]

    sum_tt = cf_tt + pre_factor * identity + 2 * cn_tt
    sum_te = cf_te + 2 * cn_te
    sum_ee = cf_ee + pre_factor * identity + 2 * cn_ee

    ones = jnp.ones(cn_tt.shape)
    d_tt, d_te, d_et, d_ee, _ = teb_matrix_mult(
        sum_tt, sum_te, sum_te, sum_ee, ones,
        cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee, ones
    )
    d_tt, d_te, _, d_ee = block_matrix_sqrt(d_tt, d_et, d_et, d_ee)

    d_bb = jnp.where(cf_bb != 0,
                     jnp.sqrt((cf_bb + pre_factor * identity + 2 * cn_bb)
                              * reciprocal_matrix(cf_bb)),
                     1.0)
    return d_tt, d_te, d_ee, d_bb

#D mixing matrix for POLARIZATION-ONLY (pol = "P", FlatS2 / DiagonalEB) sampling. The
#E and B blocks are diagonal (no T, hence no TE coupling), so this is get_d_teb_matrix
#restricted to its E/B sector - identical to that function's d_ee / d_bb when cf_te = 0
#(block_matrix_sqrt of a diagonal block is the elementwise sqrt), including the IDENTITY
#B block where cf_bb = 0 (unlensed B has zero power at r = 0; see get_d_teb_matrix)
@jax.jit
def get_d_eb_matrix(cf_ee, cf_bb, cn_ee, cn_bb):
    pre_factor = jnp.deg2rad(5 / ARCMIN_PER_DEGREE)**2
    identity = jnp.ones(cn_ee.shape)
    d_ee = jnp.sqrt((cf_ee + pre_factor * identity + 2 * cn_ee)
                    * reciprocal_matrix(cf_ee))
    d_bb = jnp.where(cf_bb != 0,
                     jnp.sqrt((cf_bb + pre_factor * identity + 2 * cn_bb)
                              * reciprocal_matrix(cf_bb)),
                     1.0)
    return d_ee, d_bb

# ── G Matrix ──────────────────────────────────────────────────────────────

@jax.jit
def get_g_matrix(cphi_fid, nphi, a_phi_fid, a_phi = 1):
    g0 = jnp.sqrt(1 + 2 * nphi * reciprocal_matrix(cphi_fid))
    g = jnp.sqrt(1 + 2 * nphi * reciprocal_matrix((a_phi/a_phi_fid) * cphi_fid))
    g = reciprocal_matrix(g0) * g
    return g

@jax.jit
def get_g_matrix_lcdm(cphi_curr, nphi):
    g = jnp.sqrt(1 + 2 * nphi * reciprocal_matrix(cphi_curr))
    return g

# ── Quadratic Estimate ────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(0,))
def get_indices(length):
    parts = [unique_permutations(num_0s, length - num_0s) for num_0s in range(length + 1)]
    return jnp.concatenate(parts, axis=0)


@partial(jax.jit, static_argnums=(0, 1))
def unique_permutations(n0, n1):
    total = n0 + n1
    res = []
    for indices in combinations(range(total), n0):
        p = [1] * total
        for i in indices:
            p[i] = 0
        res.append(p)
    return jnp.array(res, dtype=jnp.int32).reshape(-1, total)


@jax.jit
def get_fourier_derivatives(f, pix_width):
    Nx, _ = f.shape
    kx = 2 * jnp.pi * jfft.fftfreq(Nx, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(Nx, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    Fx = 1j * KX * f
    Fy = 1j * KY * f
    Fxx = -(KX**2) * f
    Fyy = -(KY**2) * f
    Fxy = -(KX * KY) * f
    return Fx, Fy, Fxx, Fxy, Fyy


@jax.jit
def get_specific_derivative(field, pix_width, type_1=None, type_2=None):
    if type_1 is None and type_2 is None:
        return field

    x, y, xx, xy, yy = get_fourier_derivatives(field, pix_width)
    first_derivatives = jnp.array([x, y])
    second_derivatives = jnp.array([xx, yy, xy])

    if type_2 is None:
        return first_derivatives[type_1]

    def return_double(second_derivatives, type_1):
        return second_derivatives[type_1]

    def return_mixed(second_derivatives, type_1):
        _ = type_1
        return second_derivatives[2]

    return jax.lax.cond(
        jnp.equal(type_1, type_2),
        return_double, return_mixed,
        second_derivatives, type_1
    )


@jax.jit
def laplacian_2d(field, pix_width):
    Nx, _ = field.shape
    kx = 2 * jnp.pi * jfft.fftfreq(Nx, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(Nx, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    return KX**2 + KY**2


@jax.jit
def levi_civita_3d():
    i, j, k = jnp.meshgrid(jnp.arange(3), jnp.arange(3), jnp.arange(3), indexing='ij')
    return (i - j) * (j - k) * (k - i) / 2


@jax.jit
def qe_leg(field, pix_width, indices):
    num_derivatives = len(indices["isolated"])
    squashed = jnp.concatenate((indices["encapsulated"], indices["isolated"]), axis=0)
    num_y = jnp.sum(squashed)
    num_x = len(squashed) - num_y
    return qe_leg_internal(field, pix_width, num_derivatives, num_x, num_y)


@jax.jit
def qe_leg_internal(field, pix_width, num_derivatives, num_x, num_y):
    field = jnp.nan_to_num(field, nan=0)

    def identity_branch(field, pix_width, num_derivatives, num_x, num_y):
        _ = pix_width, num_derivatives, num_x, num_y
        return field * (1.0 + 0.0j)

    def derivative_branch(field, pix_width, num_derivatives, num_x, num_y):
        laplacian_power = jnp.sqrt(laplacian_2d(field, pix_width)**num_derivatives)
        N, _ = field.shape
        kx = 2 * jnp.pi * jfft.fftfreq(N, pix_width)
        ky = 2 * jnp.pi * jfft.rfftfreq(N, pix_width)
        KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
        result = (1j * KX)**num_x * (1j * KY)**num_y * field / laplacian_power
        return jnp.nan_to_num(result, nan=0)

    return jax.lax.cond(
        jnp.logical_and(jnp.equal(num_x, 0), jnp.equal(num_y, 0)),
        identity_branch, derivative_branch,
        field, pix_width, num_derivatives, num_x, num_y
    )


@jax.jit
def scalar_quadratic_estimate(cn_tt, cf_tt, cfl_tt, m_tt, b_tt, pix_width):
    tf = m_tt * b_tt
    sigma = tf**2 * cfl_tt + cn_tt
    qe_sum = np.zeros(cf_tt.shape, dtype=jnp.float64)

    for (i, j) in get_indices(2):
        internal = get_scalar_qe_norm_at_position(tf, cf_tt, sigma, pix_width, i, j)
        qe_sum += jnp.abs(get_specific_derivative(jfft.rfft2(internal), pix_width, i, j))

    return jnp.nan_to_num(1 / qe_sum, nan=0, posinf=0, neginf=0)


@jax.jit
def polar_quadratic_estimate(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb,
                             mask_ee, mask_bb, beam_ee, beam_bb, pix_width):
    qe_sum = jnp.zeros(cf_ee.shape, dtype=jnp.complex128)

    for (i, j) in get_indices(2):
        internal = polar_quadratic_estimate_internal(
            cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb,
            mask_ee, mask_bb, beam_ee, beam_bb, pix_width, i, j
        )
        qe_sum += jnp.abs(get_specific_derivative(jfft.rfft2(internal), pix_width, i, j))

    return jnp.nan_to_num(1 / qe_sum, nan=0, posinf=0, neginf=0)


@jax.jit
def polar_quadratic_estimate_internal(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb,
                                      mask_ee, mask_bb, beam_ee, beam_bb, pix_width, i, j):
    epsilon = levi_civita_3d()
    tf2e = (mask_ee * beam_ee)**2
    tf2b = (mask_bb * beam_bb)**2
    sigma_e = tf2e * cfl_ee + cn_ee
    sigma_b = tf2b * cfl_bb + cn_bb
    N, _ = cf_ee.shape
    qe_sum = np.zeros((N, N), dtype=jnp.float64)

    for (k, l, m, n, p, q) in get_indices(6):
        qe_sum += 4 * epsilon[m, p, 2] * epsilon[n, q, 2] \
                  * get_polar_qe_norm_at_position(
                      tf2e, cf_ee, sigma_e, tf2b, cf_bb, sigma_b,
                      pix_width, i, j, k, l, m, n, p, q
                  )

    return qe_sum


def get_scalar_qe_norm_at_position(tf, ct, sigma, pix_width, i, j):
    enc_ij = {"encapsulated": jnp.array([i, j]), "isolated": jnp.array([])}
    enc_i = {"encapsulated": jnp.array([i]), "isolated": jnp.array([])}
    enc_j = {"encapsulated": jnp.array([j]), "isolated": jnp.array([])}
    enc_none = {"encapsulated": jnp.array([]), "isolated": jnp.array([])}

    return (jfft.irfft2(qe_leg(tf**2 * ct**2 / sigma, pix_width, enc_ij))
            * jfft.irfft2(qe_leg(tf**2 / sigma, pix_width, enc_none))
            + jfft.irfft2(qe_leg(tf**2 * ct / sigma, pix_width, enc_i))
            * jfft.irfft2(qe_leg(tf**2 * ct / sigma, pix_width, enc_j)))


def get_polar_qe_norm_at_position(tf2e, cf_ee, sigma_e, tf2b, cf_bb, sigma_b,
                                  pix_width, i, j, k, l, m, n, p, q):
    enc_ij_klmn = {"encapsulated": jnp.array([i, j]), "isolated": jnp.array([k, l, m, n])}
    enc_none_klpq = {"encapsulated": jnp.array([]), "isolated": jnp.array([k, l, p, q])}
    enc_i_klmn = {"encapsulated": jnp.array([i]), "isolated": jnp.array([k, l, m, n])}
    enc_j_klpq = {"encapsulated": jnp.array([j]), "isolated": jnp.array([k, l, p, q])}
    enc_none_klmn = {"encapsulated": jnp.array([]), "isolated": jnp.array([k, l, m, n])}
    enc_ij_klpq = {"encapsulated": jnp.array([i, j]), "isolated": jnp.array([k, l, p, q])}

    return (jfft.irfft2(qe_leg(tf2e * cf_ee**2 / sigma_e, pix_width, enc_ij_klmn))
            * jfft.irfft2(qe_leg(tf2b / sigma_b, pix_width, enc_none_klpq))
            - 2 * jfft.irfft2(qe_leg(tf2e * cf_ee / sigma_e, pix_width, enc_i_klmn))
            * jfft.irfft2(qe_leg(tf2b * cf_bb / sigma_b, pix_width, enc_j_klpq))
            + jfft.irfft2(qe_leg(tf2e / sigma_e, pix_width, enc_none_klmn))
            * jfft.irfft2(qe_leg(tf2b * cf_bb**2 / sigma_b, pix_width, enc_ij_klpq)))


# ── CAMB Interface ────────────────────────────────────────────────────────

def _run_camb(H0, ombh2, omch2, cosmomc_theta, r, mnu, tau, As, nt, ns,
              lmax_prime, k_pivot, Alens):
    
    pars = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2, cosmomc_theta=cosmomc_theta,
        r=r, mnu=mnu, As=As, nt=nt, ns=ns, lmax=lmax_prime,
        tau=tau, pivot_scalar=k_pivot, pivot_tensor=k_pivot, Alens=Alens
    )
    pars.max_l_tensor = 600 #NOTE switching to Yuuki's values from 2 * lmax_prime
    pars.max_eta_k_tensor = 1200 #NOTE switching to Yuuki's values from 4 * lmax_prime
    pars.WantScalars = True
    pars.WantTensors = True #DEBUG should this be set to False?
    pars.DoLensing = True
    pars.set_nonlinear_lensing(True)

    results = camb.get_results(pars)
    power_spectra = results.get_cmb_power_spectra(pars, lmax=lmax_prime - 1, CMB_unit="muK")
    lens_potential = jnp.asarray(results.get_lens_potential_cls(lmax=lmax_prime - 1)[:, 0])
    return power_spectra, lens_potential


def _camb_callback_fn(H0, ombh2, omch2, cosmomc_theta, r, mnu, tau, As, nt, ns,
                      lmax_prime, k_pivot, Alens):
    """Eagerly runs CAMB and returns flat arrays for use inside JIT via pure_callback."""
    lmax_prime = int(lmax_prime)
    try:
        power_spectra, lens_potential = _run_camb(
            H0, float(ombh2), float(omch2), float(cosmomc_theta),
            float(r), float(mnu), float(tau), float(As), float(nt), float(ns),
            lmax_prime, float(k_pivot), float(Alens)
        )
        unlensed_scalar = jnp.asarray(power_spectra["unlensed_scalar"], dtype=jnp.float64)
        tensor = jnp.asarray(power_spectra["tensor"], dtype=jnp.float64)
        total = jnp.asarray(power_spectra["total"], dtype=jnp.float64)
    except Exception:
        #unphysical parameters — return NaN arrays so downstream logpdf becomes -inf
        unlensed_scalar = jnp.full((lmax_prime, 4), jnp.nan)
        tensor = jnp.full((lmax_prime, 4), jnp.nan)
        total = jnp.full((lmax_prime, 4), jnp.nan)
        lens_potential = jnp.full((lmax_prime,), jnp.nan)
    return unlensed_scalar, tensor, total, lens_potential


def _camb_via_callback(H0, ombh2, omch2, cosmomc_theta, r, mnu, tau, As, nt, ns,
                       lmax_prime, k_pivot, Alens):
    """Calls CAMB through jax.pure_callback so the surrounding function can be JIT-ted."""
    result_shapes = (
        jax.ShapeDtypeStruct((lmax_prime, 4), jnp.float64),
        jax.ShapeDtypeStruct((lmax_prime, 4), jnp.float64),
        jax.ShapeDtypeStruct((lmax_prime, 4), jnp.float64),
        jax.ShapeDtypeStruct((lmax_prime,), jnp.float64),
    )
    return jax.pure_callback(
        _camb_callback_fn, result_shapes,
        H0, ombh2, omch2, cosmomc_theta, r, mnu, tau, As, nt, ns,
        lmax_prime, k_pivot, Alens,
        vmap_method = 'sequential'
    )

def _extract_all_cls(unlensed_scalar, tensor, total, lens_potential, lmax, lmax_prime):
    cls = {}
    for stokes, col in _CAMB_COLS.items():
        cls[f"scalar_{stokes}"] = dl2cl(unlensed_scalar[:, col], lmax, lmax_prime)
        cls[f"tensor_{stokes}"] = dl2cl(tensor[:, col], lmax, lmax_prime)
    for stokes in ("TT", "TE", "EE", "BB"):
        cls[f"total_{stokes}"] = dl2cl(total[:, _CAMB_COLS[stokes]], lmax, lmax_prime)
    cls["phi"] = dl2cl(lens_potential, lmax, lmax_prime, is_phi=True)
    return cls


# ── Lensing ───────────────────────────────────────────────────────────────

def _lens_fields(field_t, field_e, field_b, phi, pix_width, nside, theta_pix):
    lensed_t = jfft.rfft2(primal_lense_flow(
        field_t, phi, pix_width, n=10, direction=FORWARD_LENSE, adjoint=False
    ))
    field_q, field_u = primal_eb2qu(jfft.rfft2(field_e), jfft.rfft2(field_b), nside, theta_pix)
    lensed_q = primal_lense_flow(
        jfft.irfft2(field_q), phi, pix_width, n=10, direction=FORWARD_LENSE, adjoint=False
    )
    lensed_u = primal_lense_flow(
        jfft.irfft2(field_u), phi, pix_width, n=10, direction=FORWARD_LENSE, adjoint=False
    )
    lensed_e, lensed_b = primal_qu2eb(
        jfft.rfft2(lensed_q), jfft.rfft2(lensed_u), nside, theta_pix
    )
    return lensed_t, lensed_e, lensed_b


# ── Dataset Construction ─────────────────────────────────────────────────

def build_phi_template(nside, theta_pix, pix_width, cphi, qe, phi_real):
    fourier_weights = get_fourier_weights((nside, nside // 2 + 1))
    phi_cov = DiagonalScalar(
        fourier_weights=fourier_weights, nside=nside,
        theta_pix=theta_pix, pix_width=pix_width, scalar_matrix=cphi
    )
    qe_op = phi_cov.replace(scalar_matrix=qe)
    phi = FlatS0(
        fourier_weights=fourier_weights, nside=nside,
        theta_pix=theta_pix, pix_width=pix_width,
        basis=Basis.FOURIER, parametrization=Parametrization.T,
        scalar_matrix=phi_real
    )
    return fourier_weights, phi_cov, qe_op, phi


def _build_dataset_t(nside, theta_pix, pix_width, cphi, unscaled_cphi, qe, phi_real,
                     cf, cf_scalar, cf_tensor, cfl, cn, d_tt, g, mask, beam,
                     field_t, lensed_t, data_t, r, a_phi):
    _, phi_cov, qe_op, phi = build_phi_template(nside, theta_pix, pix_width, cphi, qe, phi_real)
    return DataSetT(
        noise_covariance=phi_cov.replace(scalar_matrix=cn["TT"]),
        field_covariance=phi_cov.replace(scalar_matrix=cf["TT"]),
        lensed_field_covariance=phi_cov.replace(scalar_matrix=cfl["TT"]),
        scalar_field_covariance=phi_cov.replace(scalar_matrix=cf_scalar["TT"]),
        tensor_field_covariance=phi_cov.replace(scalar_matrix=cf_tensor["TT"]),
        mixing_d=phi_cov.replace(scalar_matrix=d_tt),
        mixing_g=phi_cov.replace(scalar_matrix=g),
        phi_covariance=phi_cov,
        unscaled_phi_covariance = phi_cov.replace(scalar_matrix = unscaled_cphi),
        mask=phi_cov.replace(scalar_matrix=mask),
        beam=phi_cov.replace(scalar_matrix=beam),
        quadratic_estimate=qe_op,
        data=phi.replace(scalar_matrix=data_t),
        unlensed_field=phi.replace(scalar_matrix=field_t),
        lensed_field=phi.replace(scalar_matrix=lensed_t),
        phi=phi,
        fid_r = r,
        fid_a_phi = a_phi,
        nside = nside,
        theta_pix = theta_pix,
        pix_width = pix_width, 
        fourier_weights = get_fourier_weights((nside, nside))
    )


def _build_dataset_eb(nside, theta_pix, pix_width, cphi, unscaled_cphi, qe, phi_real,
                      cf, cf_scalar, cf_tensor, cfl, cn, d_ee, d_bb, g, mask, beam,
                      field_e, field_b, lensed_e, lensed_b, data_e, data_b, r, a_phi):
    fw, phi_cov, qe_op, phi = build_phi_template(nside, theta_pix, pix_width, cphi, qe, phi_real)
    noise_cov = DiagonalEB(
        fourier_weights=fw, nside=nside,
        theta_pix=theta_pix, pix_width=pix_width,
        matrix_EE=cn["EE"], matrix_BB=cn["BB"]
    )
    data = FlatS2(
        fourier_weights=fw, nside=nside,
        theta_pix=theta_pix, pix_width=pix_width,
        basis=Basis.FOURIER, parametrization=Parametrization.EB,
        polar_matrix_1=data_e, polar_matrix_2=data_b
    )
    return DataSetEB(
        noise_covariance=noise_cov,
        lensed_field_covariance=noise_cov.replace(matrix_EE=cfl["EE"], matrix_BB=cfl["BB"]),
        field_covariance=noise_cov.replace(matrix_EE=cf["EE"], matrix_BB=cf["BB"]),
        scalar_field_covariance=noise_cov.replace(matrix_EE=cf_scalar["EE"], matrix_BB=cf_scalar["BB"]),
        tensor_field_covariance=noise_cov.replace(matrix_EE=cf_tensor["EE"], matrix_BB=cf_tensor["BB"]),
        mixing_d=noise_cov.replace(matrix_EE=d_ee, matrix_BB=d_bb),
        mixing_g=phi_cov.replace(scalar_matrix = g),
        phi_covariance=phi_cov,
        unscaled_phi_covariance=phi_cov.replace(scalar_matrix = unscaled_cphi),
        mask=noise_cov.replace(matrix_EE=mask, matrix_BB=mask),
        beam=noise_cov.replace(matrix_EE=beam, matrix_BB=beam),
        quadratic_estimate=qe_op,
        data=data,
        unlensed_field=data.replace(polar_matrix_1=field_e, polar_matrix_2=field_b),
        lensed_field=data.replace(polar_matrix_1=lensed_e, polar_matrix_2=lensed_b),
        phi=phi,
        fid_r = r,
        fid_a_phi = a_phi,
        nside = nside,
        theta_pix = theta_pix,
        pix_width = pix_width, 
        fourier_weights = get_fourier_weights((nside, nside))
    )


def _build_dataset_teb(nside, theta_pix, pix_width, cphi, unscaled_cphi, qe, phi_real,
                       cf, cf_scalar, cf_tensor, cfl, cn, d_tt, d_te, d_ee, d_bb, g, mask, beam,
                       field_t, field_e, field_b,
                       lensed_t, lensed_e, lensed_b,
                       data_t, data_e, data_b, r, a_phi):
    fw, phi_cov, qe_op, phi = build_phi_template(nside, theta_pix, pix_width, cphi, qe, phi_real)
    zero = jnp.zeros(cn["TT"].shape)
    noise_cov = BlockTEB(
        fourier_weights=fw, nside=nside,
        theta_pix=theta_pix, pix_width=pix_width,
        matrix_TT=cn["TT"], matrix_TE=cn["TE"], matrix_ET=cn["TE"],
        matrix_EE=cn["EE"], matrix_BB=cn["BB"]
    )
    data = FlatS02(
        fourier_weights=fw, nside=nside,
        theta_pix=theta_pix, pix_width=pix_width,
        basis=Basis.FOURIER, parametrization=Parametrization.EB,
        scalar_matrix=data_t, polar_matrix_1=data_e, polar_matrix_2=data_b
    )
    return DataSetTEB(
        noise_covariance=noise_cov,
        field_covariance=noise_cov.replace(
            matrix_TT=cf["TT"], matrix_TE=cf["TE"], matrix_ET=cf["TE"],
            matrix_EE=cf["EE"], matrix_BB=cf["BB"]
        ),
        lensed_field_covariance=noise_cov.replace(
            matrix_TT=cfl["TT"], matrix_TE=cfl["TE"], matrix_ET=cfl["TE"],
            matrix_EE=cfl["EE"], matrix_BB=cfl["BB"]
        ),
        scalar_field_covariance=noise_cov.replace(
            matrix_TT=cf_scalar["TT"], matrix_TE=cf_scalar["TE"], matrix_ET=cf_scalar["TE"],
            matrix_EE=cf_scalar["EE"], matrix_BB=cf_scalar["BB"]
        ),
        tensor_field_covariance=noise_cov.replace(
            matrix_TT=cf_tensor["TT"], matrix_TE=cf_tensor["TE"], matrix_ET=cf_tensor["TE"],
            matrix_EE=cf_tensor["EE"], matrix_BB=cf_tensor["BB"]
        ),
        mixing_d=noise_cov.replace(
            matrix_TT=d_tt, matrix_TE=d_te, matrix_ET=d_te,
            matrix_EE=d_ee, matrix_BB=d_bb
        ),
        mixing_g = phi_cov.replace(scalar_matrix = g),
        phi_covariance=phi_cov,
        unscaled_phi_covariance=phi_cov.replace(scalar_matrix = unscaled_cphi),
        mask=noise_cov.replace(
            matrix_TT=mask, matrix_TE=zero, matrix_ET=zero,
            matrix_EE=mask, matrix_BB=mask
        ),
        beam=noise_cov.replace(
            matrix_TT=beam, matrix_TE=zero, matrix_ET=zero,
            matrix_EE=beam, matrix_BB=beam
        ),
        quadratic_estimate=qe_op,
        data=data,
        unlensed_field=data.replace(
            scalar_matrix=field_t, polar_matrix_1=field_e, polar_matrix_2=field_b
        ),
        lensed_field=data.replace(
            scalar_matrix=lensed_t, polar_matrix_1=lensed_e, polar_matrix_2=lensed_b
        ),
        phi=phi,
        fid_r = r,
        fid_a_phi = a_phi,
        nside = nside,
        theta_pix = theta_pix,
        pix_width = pix_width, 
        fourier_weights = get_fourier_weights((nside, nside))
    )

@partial(jax.jit, static_argnames = ["lmax", "lmax_prime"])
def interpolate_cls(cls, lmax, lmax_prime):

    ell = jnp.arange(2, lmax + 1).astype(jnp.float64)
    ell_prime = jnp.arange(2, lmax_prime + 1).astype(jnp.float64)

    def exponential_interpolate(cls):
        return jnp.exp(jnp.interp(
            jnp.log(ell), jnp.log(ell_prime), jnp.log(cls),
            left="extrapolate", right="extrapolate"
        ))

    def linear_interpolate(cls):
        return jnp.interp(ell, ell_prime, cls, left=0.0, right=0.0)

    cls = jax.lax.cond(
        jnp.all(cls > 0),
        exponential_interpolate,
        linear_interpolate,
        cls
    )

    return cls


# ── Main Simulation Entry Point ──────────────────────────────────────────

def load_sim(nside, theta_pix, pol, master_seed, uk_arcmin_t=3, H0=None,
             ombh2=0.0224567, omch2=0.118489, cosmomc_theta=0.0104098,
             r=0.0, mnu=0.06, tau=0.05, As=jnp.exp(3.043) * 1e-10,
             nt=0, ns=0.968602, lmax=4000, l_knee = 100,
             k_pivot = 0.05, Alens=1, nphi_fac=2, a_phi = 1):
    
    #NOTE changing k_pivot from Marius' choice to match Yuuki's emulator
    lmax_prime = min(lmax, DEFAULT_MAX_ELL)
    unlensed_scalar, tensor, total, lens_potential = _camb_via_callback(
        H0, ombh2, omch2, cosmomc_theta, r, mnu, tau, As, nt, ns,
        lmax_prime, k_pivot, Alens
    )
    cls = _extract_all_cls(unlensed_scalar, tensor, total, lens_potential, lmax, lmax_prime)

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    keys = jax.random.split(jax.random.PRNGKey(master_seed), 100)
    ells = jnp.arange(2, lmax).astype(jnp.float64)

    #ells arrays matched to the cls lengths: the emulator override returns ells
    #2..lmax inclusive while the CAMB dl2cl path returns 2..lmax-1
    phi_ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)

    #Lensing potential
    unscaled_cphi = covar_matrix_from_cls(nside, pix_width, ell_grid,
                                          phi_ells,
                                          cls["phi"], origin_value=0)
    cphi = a_phi * unscaled_cphi
    phi, kc = field_from_covar(nside, cphi, keys, 0)

    #Unlensed field covariances (scalar + tensor). Unlensed scalar BB (identically zero
    #at r = 0) cannot go through the log-interpolating covariance builder (log(0) -> NaN)
    #so _covar_or_zeros maps it to an exactly zero matrix. TE crosses zero, so it is
    #built from the correlation ratio rho = Cl_TE / sqrt(Cl_TT * Cl_EE) instead
    #(te_covar_from_rho): linear interpolation of rho onto the ell grid, scaled by
    #sqrt(cf_tt * cf_ee), which keeps the T/E block positive semi-definite by
    #construction and matched to the sampler's grid-spline TE path
    cf = {}
    cf_scalar = {}
    cf_tensor = {}
    pol_comps = ("TT",) if pol == "I" else ("TT", "EE", "BB")
    for comp in pol_comps:
        scalar_ells = jnp.arange(2, 2 + cls[f"scalar_{comp}"].shape[0]).astype(jnp.float64)
        cf_scalar[comp] = _covar_or_zeros(nside, pix_width, ell_grid,
                                          scalar_ells,
                                          cls[f"scalar_{comp}"], origin_value=0)
        cf_tensor[comp] = _covar_or_zeros(nside, pix_width, ell_grid, ells,
                                          cls[f"tensor_{comp}"], origin_value=0)
        cf[comp] = cf_scalar[comp] + cf_tensor[comp]
    if pol != "I":
        scalar_ells = jnp.arange(2, 2 + cls["scalar_TE"].shape[0]).astype(jnp.float64)
        rho_te = te_rho_from_cls(cls["scalar_TE"], cls["scalar_TT"], cls["scalar_EE"])
        cf["TE"] = cf_scalar["TE"] = te_covar_from_rho(rho_te, scalar_ells, ell_grid,
                                                       cf["TT"], cf["EE"])
        cf_tensor["TE"] = jnp.zeros((nside, nside // 2 + 1))

    #Lensed field covariances (TE via the same correlation-ratio route)
    cfl = {comp: covar_matrix_from_cls(nside, pix_width, ell_grid, ells,
                                       cls[f"total_{comp}"], origin_value=0)
           for comp in ("TT", "EE", "BB")}
    rho_te_lensed = te_rho_from_cls(cls["total_TE"], cls["total_TT"], cls["total_EE"])
    cfl["TE"] = te_covar_from_rho(rho_te_lensed, ells, ell_grid, cfl["TT"], cfl["EE"])

    #Unlensed random fields. T and E are drawn CORRELATED to match the model's TE block:
    #E ~ N(0, C_EE), then T = (C_TE / C_EE) E + t_indep with
    #t_indep ~ N(0, C_TT - C_TE^2 / C_EE), which gives exactly
    #Cov = [[C_TT, C_TE], [C_TE, C_EE]] (the conditional variance is
    #C_TT (1 - rho^2) >= 0 by the |rho| <= 1 construction; the maximum(0) guards
    #round-off only). The B draw against a zero covariance gives an exactly zero field
    if pol == "I":
        field_t, kc = field_from_covar(nside, cf["TT"], keys, kc)
    else:
        conditional_tt = jnp.maximum(
            cf["TT"] - cf["TE"]**2 * reciprocal_matrix(cf["EE"]), 0.0)
        t_indep, kc = field_from_covar(nside, conditional_tt, keys, kc)
        field_e, kc = field_from_covar(nside, cf["EE"], keys, kc)
        field_b, kc = field_from_covar(nside, cf["BB"], keys, kc)
        te_ratio = cf["TE"] * reciprocal_matrix(cf["EE"])
        field_t = jfft.irfft2(te_ratio * jfft.rfft2(field_e)) + t_indep

    #Lensing
    if pol == "I":
        lensed_t = jfft.rfft2(primal_lense_flow(
            field_t, phi, pix_width, n=10, direction=FORWARD_LENSE, adjoint=False
        ))
    else:
        lensed_t, lensed_e, lensed_b = _lens_fields(
            field_t, field_e, field_b, phi, pix_width, nside, theta_pix
        )

    #NOTE we have removed the mask for LCDM inference
    mask = jnp.ones_like(get_mask(4000, nside, pix_width, ell_grid))
    beam = get_beam(nside, pix_width, ell_grid, lmax_prime)

    #Noise covariance and white noise (the TE noise Cls are identically zero, so
    #_covar_or_zeros keeps that block an exact zero matrix instead of NaN)
    n_tt, n_te, n_ee, n_bb = noise_cls(lmax_prime, uk_arcmin_t, l_knee = l_knee)
    ell_prime = jnp.arange(2, lmax_prime)
    cn = {}
    for comp, ncl in zip(("TT", "TE", "EE", "BB"), (n_tt, n_te, n_ee, n_bb)):
        cn[comp] = _covar_or_zeros(nside, pix_width, ell_grid, ell_prime, ncl, origin_value=0)

    wn_t, kc = field_from_covar(nside, cn["TT"], keys, kc)
    if pol != "I":
        wn_e, kc = field_from_covar(nside, cn["EE"], keys, kc)
        wn_b, kc = field_from_covar(nside, cn["BB"], keys, kc)

    #Data = Mask * Beam * Lensed + Noise
    data_t = mask * beam * lensed_t + jfft.rfft2(wn_t)
    if pol != "I":
        data_e = mask * beam * lensed_e + jfft.rfft2(wn_e)
        data_b = mask * beam * lensed_b + jfft.rfft2(wn_b)

    #Convert unlensed fields to Fourier space
    field_t = jfft.rfft2(field_t)
    if pol != "I":
        field_e, field_b = jfft.rfft2(field_e), jfft.rfft2(field_b)

    #D matrix (full T/E block including TE; see get_d_teb_matrix for the identity B
    #block at r = 0)
    d_tt = get_d_tt_matrix(cf["TT"], cn["TT"])
    if pol != "I":
        d_tt, d_te, d_ee, d_bb = get_d_teb_matrix(cf["TT"], cf["TE"], cf["EE"], cf["BB"],
                                                  cn["TT"], cn["TE"], cn["EE"], cn["BB"])

    #Quadratic estimate
    if pol == "I":
        qe = scalar_quadratic_estimate(cn["TT"], cf["TT"], cfl["TT"],
                                       mask, beam, pix_width) / nphi_fac
    else:
        qe = polar_quadratic_estimate(cf["EE"], cf["BB"], cfl["EE"], cfl["BB"],
                                      cn["EE"], cn["BB"], mask, mask,
                                      beam, beam, pix_width) / nphi_fac

    #The G mixing matrix    
    g = get_g_matrix(cphi, qe, a_phi, a_phi = a_phi)

    #convert phi back to fourier space
    phi = jfft.rfft2(phi)
    #Build dataset for the requested polarization
    if pol == "I":
        return _build_dataset_t(
            nside, theta_pix, pix_width, cphi, unscaled_cphi, qe, phi,
            cf, cf_scalar, cf_tensor, cfl, cn, d_tt, g, mask, beam,
            field_t, lensed_t, data_t, r, a_phi
        )
    elif pol == "P":
        return _build_dataset_eb(
            nside, theta_pix, pix_width, cphi, unscaled_cphi, qe, phi,
            cf, cf_scalar, cf_tensor, cfl, cn, d_ee, d_bb, g, mask, beam,
            field_e, field_b, lensed_e, lensed_b, data_e, data_b, r, a_phi
        )
    else:
        return _build_dataset_teb(
            nside, theta_pix, pix_width, cphi, unscaled_cphi, qe, phi,
            cf, cf_scalar, cf_tensor, cfl, cn, d_tt, d_te, d_ee, d_bb, g, mask, beam,
            field_t, field_e, field_b,
            lensed_t, lensed_e, lensed_b,
            data_t, data_e, data_b, r, a_phi
        )

def batch_simulated_trials(num_trials=10, nside=256, theta_pix=2,
                           uk_arcmin_t=10, lmax=17_000, pol="I"):
    def parallel_sim(seed):
        return load_sim(nside=nside, theta_pix=theta_pix, pol=pol,
                        master_seed=seed, uk_arcmin_t=uk_arcmin_t, lmax=lmax)

    seeds = jnp.asarray([np.random.randint(0, 2**31) for _ in range(num_trials)])
    return jax.vmap(parallel_sim)(seeds)

def get_field_specs(pol):
    specs = []
    if pol in ("I", "IP"):
        for ds_attr, prefix in [("data", "data"), ("unlensed_field", "unlensed"),
                                ("lensed_field", "lensed")]:
            specs.append((f"{prefix}_t", ds_attr, "scalar_matrix"))
    if pol in ("P", "IP"):
        for ds_attr, prefix in [("data", "data"), ("unlensed_field", "unlensed"),
                                ("lensed_field", "lensed")]:
            specs.append((f"{prefix}_e", ds_attr, "polar_matrix_1"))
            specs.append((f"{prefix}_b", ds_attr, "polar_matrix_2"))
    specs.append(("phi", "phi", "scalar_matrix"))
    return specs

#Compute avg Cl^TT, Cl^DD, Cl^PP, Cl^LL over a large number of trials
#to gauge whether our GRF maps are indeed following the correct statistics...
def get_avg_cls(theta_pix, num_trials=100, pol="I", nside=256,
                uk_arcmin_t=10, lmax=17000, delta_l=50):
    trial_results = batch_simulated_trials(
        num_trials=num_trials, nside=nside, theta_pix=theta_pix,
        uk_arcmin_t=uk_arcmin_t, lmax=lmax, pol=pol
    )

    specs = get_field_specs(pol)
    results = {}

    for key, ds_attr, component in specs:
        batched = getattr(getattr(trial_results, ds_attr), component)
        cls_sum = None
        for trial in range(num_trials):
            ell, cls = primal_power_spectra(batched[trial], theta_pix, delta_l=delta_l)
            cls_sum = cls if cls_sum is None else cls_sum + cls
        results[key] = (ell, cls_sum / num_trials)

    return results

#The fraction of the total sky covered by the flat sky approximation
def f_sky(nside, theta_pix):
    dx = np.deg2rad(theta_pix / ARCMIN_PER_DEGREE)
    map_area = nside**2 * dx**2 * (180 / np.pi)**2
    return map_area / 40_000

#The variance of log(Cls) is actually independent of Cl to first order...
def log_c_ell_variance(ell, f_sky, delta_l):
    return 2 / (delta_l * (2 * ell + 1) * f_sky)

if __name__ == "__main__":
    start_time = time.time()
    batch_trials = batch_simulated_trials(num_trials=100, nside=256, theta_pix=2,
                                          uk_arcmin_t=10, lmax=17_000, pol="IP")
    end_time = time.time()
    print(f"Total run time = {end_time - start_time}")
