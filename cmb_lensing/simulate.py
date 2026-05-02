import camb
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_disable_jit", True)
from cmb_lensing.util import *
from cmb_lensing.lense_flow import *
from cmb_lensing.dataset import *
from cmb_lensing.util import *
from functools import partial
from itertools import combinations
import time

#NOTE for the time being I would only recommend running this with the default params...
#noise level, theta_pix, nside, master_seed, and lmax could all be changed comfortably but more
#validation needs to be done before other params can be set
#TODO this file is super messy and ugly so it should be cleaned up in the future
def load_sim(nside, theta_pix, pol, master_seed = None, uk_arcmin_t = 3, H0 = None, 
            ombh2 = 0.0224567, omch2 = 0.118489, cosmomc_theta = 0.0104098,
            r = 0.2, mnu = 0.06, tau = 0.055, As = jnp.exp(3.043) * 1e-10, 
            nt = -0.2/8, ns = 0.968602,lmax = 17_000,
            k_pivot = 0.002, Alens = 1, nphi_fac = 2):
    
    #the clamped lmax_prime is used to get Dl's and Cl's from
    #camb and then we linearly extrapolate in log-log space higer ell
    #Dl's and Cl's using the higher original lmax value
    lmax_prime = min(lmax, 5000)
    #first generate the camb parameters object
    pars = camb.set_params(
        H0 = H0, 
        ombh2 = ombh2, 
        omch2 = omch2, 
        cosmomc_theta = cosmomc_theta,
        r = r,
        mnu = mnu, 
        As = As,
        nt = nt, 
        ns = ns,
        lmax = lmax_prime,
        tau = tau, 
        pivot_scalar = k_pivot,
        pivot_tensor = k_pivot,
        Alens = Alens)
    
    pars.max_l_tensor = 2*lmax_prime
    pars.max_eta_k_tensor = 4*lmax_prime
    pars.WantScalars = True
    pars.WantTensors = True
    pars.DoLensing = True
    pars.set_nonlinear_lensing(True)

    #calculate results for these parameters
    results = camb.get_results(pars)

    #get the Dl's from camb
    power_spectra = results.get_cmb_power_spectra(pars, lmax = lmax_prime - 1, CMB_unit = "muK")

    #Split the RNG key into 100 independent sub-keys
    if master_seed is None:
        master_seed = np.random.randint(0, 2**31)
    KEYS = jax.random.split(jax.random.PRNGKey(master_seed), 100)

    #Get the TT, TE, EE, and BB Cl's for unlensed, tensor, and total fields
    dl_tt_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,0])
    dl_te_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,3])
    dl_ee_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,1])
    dl_bb_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,2])
    dl_tt_tensor = jnp.asarray(power_spectra["tensor"][:,0])
    dl_te_tensor = jnp.asarray(power_spectra["tensor"][:,3])
    dl_ee_tensor = jnp.asarray(power_spectra["tensor"][:,1])
    dl_bb_tensor = jnp.asarray(power_spectra["tensor"][:,2])
    dl_tt_total = jnp.asarray(power_spectra["total"][:,0])
    dl_ee_total = jnp.asarray(power_spectra["total"][:,1])
    dl_bb_total = jnp.asarray(power_spectra["total"][:,2])

    #Convert the Dl's to Cl's
    cl_tt_scalar = dl2cl(dl_tt_scalar, lmax, lmax_prime)
    cl_te_scalar = dl2cl(dl_te_scalar, lmax, lmax_prime)
    cl_ee_scalar = dl2cl(dl_ee_scalar, lmax, lmax_prime)
    cl_bb_scalar = dl2cl(dl_bb_scalar, lmax, lmax_prime)
    cl_tt_tensor = dl2cl(dl_tt_tensor, lmax, lmax_prime)
    cl_te_tensor = dl2cl(dl_te_tensor, lmax, lmax_prime)
    cl_ee_tensor = dl2cl(dl_ee_tensor, lmax, lmax_prime)
    cl_bb_tensor = dl2cl(dl_bb_tensor, lmax, lmax_prime)
    cl_tt_total = dl2cl(dl_tt_total, lmax, lmax_prime)
    cl_ee_total = dl2cl(dl_ee_total, lmax, lmax_prime)
    cl_bb_total = dl2cl(dl_bb_total, lmax, lmax_prime)

    #lensing potential Dl's
    dl_pp = jnp.asarray(results.get_lens_potential_cls(lmax = lmax_prime - 1)[:,0])
    cl_pp = dl2cl(dl_pp, lmax, lmax_prime, is_phi = True)

    #compute a meshgrid of fourier modes and alsoreturn the pixel width in radians
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)

    #calculate the noise Cl's
    noise_cls_tt, noise_cls_te, noise_cls_ee, noise_cls_bb = noise_cls(lmax_prime, uk_arcmin_t)

    #given the Cl's from each type of field generate the corresponding Gaussian random fields
    ells = jnp.arange(2, lmax).astype(jnp.float64)
    cphi = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_pp, origin_value = 0)
    #NOTE key_counter automatically updates to poll new RNG key from KEYS
    phi, key_counter = field_from_covar(nside, cphi, rng_keys = KEYS, key_counter = 0)

    #the cf covariance matrix is the sum of the tensor and scalar matrices
    cf_tt_scalar = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_tt_scalar, origin_value = 0)
    cf_tt_tensor = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_tt_tensor, origin_value = 0)
    cf_te_scalar = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_te_scalar, origin_value = 0)
    cf_te_tensor = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_te_tensor, origin_value = 0)
    cf_ee_scalar = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_ee_scalar, origin_value = 0)
    cf_ee_tensor = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_ee_tensor, origin_value = 0)
    cf_bb_scalar = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_bb_scalar, origin_value = 0)
    cf_bb_tensor = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_bb_tensor, origin_value = 0)
    cf_tt = cf_tt_scalar + cf_tt_tensor
    cf_te = cf_te_scalar + cf_te_tensor
    cf_ee = cf_ee_scalar + cf_ee_tensor
    cf_bb = cf_bb_scalar + cf_bb_tensor

    #the lensed cf i.e. "cfl" is also needed for the quadratic estimate
    cfl_tt = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_tt_total, origin_value = 0)
    cfl_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_ee_total, origin_value = 0)
    cfl_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl_bb_total, origin_value = 0)
    field_t, key_counter = field_from_covar(nside, cf_tt, KEYS, key_counter)
    field_e, key_counter = field_from_covar(nside, cf_ee, KEYS, key_counter)
    field_b, key_counter = field_from_covar(nside, cf_bb, KEYS, key_counter)

    #the lensed field is just found by lensing the unlensed field
    lensed_field_t = jfft.rfft2(primal_lense_flow(field_t, phi, pix_width, n = 10, 
                                       direction = FORWARD_LENSE, adjoint = False))
    field_q, field_u = primal_eb2qu(jfft.rfft2(field_e), jfft.rfft2(field_b), nside, theta_pix)
    lensed_field_q = primal_lense_flow(jfft.irfft2(field_q), phi, pix_width, n = 10, 
                                       direction = FORWARD_LENSE, adjoint = False)
    lensed_field_u = primal_lense_flow(jfft.irfft2(field_u), phi, pix_width, n = 10, 
                                       direction = FORWARD_LENSE, adjoint = False)
    lensed_field_e, lensed_field_b = primal_qu2eb(jfft.rfft2(lensed_field_q), 
                                                  jfft.rfft2(lensed_field_u), nside, theta_pix)

    #compute the mask and beam which are needed to simulate the data field
    l_cutoff = 3000
    m_tt = get_mask(l_cutoff, nside, pix_width, ell_grid)
    m_te = jnp.zeros(m_tt.shape)
    m_ee = m_tt
    m_bb = m_tt
    b_tt = get_beam(nside, pix_width, ell_grid, lmax_prime)
    b_te = jnp.zeros(b_tt.shape)
    b_ee = b_tt
    b_bb = b_tt

    #the data field is M * B * L * f + n where n ~ N(0, Cn) i.e. "white noise"
    ell_prime = jnp.arange(2, lmax_prime)
    cn_tt = covar_matrix_from_cls(nside, pix_width, ell_grid, ell_prime, noise_cls_tt, origin_value = 0)
    cn_te = covar_matrix_from_cls(nside, pix_width, ell_grid, ell_prime, noise_cls_te, origin_value = 0)
    cn_ee = covar_matrix_from_cls(nside, pix_width, ell_grid, ell_prime, noise_cls_ee, origin_value = 0)
    cn_bb = covar_matrix_from_cls(nside, pix_width, ell_grid, ell_prime, noise_cls_bb, origin_value = 0)
    white_noise_t, key_counter = field_from_covar(nside, cn_tt, KEYS, key_counter)
    white_noise_e, key_counter = field_from_covar(nside, cn_ee, KEYS, key_counter)
    white_noise_b, key_counter = field_from_covar(nside, cn_bb, KEYS, key_counter)
    data_t = m_tt * b_tt * lensed_field_t + jfft.rfft2(white_noise_t)
    data_e = m_ee * b_ee * lensed_field_e + jfft.rfft2(white_noise_e)
    data_b = m_bb * b_bb * lensed_field_b + jfft.rfft2(white_noise_b)

    #convert unlensed fields back to fourier space before storing them inside of the data_sets
    field_t = jfft.rfft2(field_t)
    field_e = jfft.rfft2(field_e)
    field_b = jfft.rfft2(field_b)

    #the D matrix is used in mixing and map estimation...
    d_tt, d_te, d_ee, d_bb = get_d_matrix(cf_tt, cf_te, cf_ee, cf_bb, cn_tt, cn_te, cn_ee, cn_bb)

    #The value of the quadratic estimate varies depending on whether or not polarization fields are included
    if pol == "I":
        quadratic_estimate = scalar_quadratic_estimate(cn_tt, cf_tt, cfl_tt, m_tt, b_tt, pix_width) / nphi_fac 
    else:
        quadratic_estimate = polar_quadratic_estimate(cf_ee, cf_bb, cfl_ee, cfl_bb, 
                                                      cn_ee, cn_bb, m_ee, m_bb, 
                                                      b_ee, b_bb, pix_width) / nphi_fac

    #Create a template DiagonalScalar covariance matrix for phi
    fourier_weights = get_fourier_weights((nside, nside//2+1))
    phi_covariance = DiagonalScalar(
        fourier_weights = fourier_weights,
        nside = nside,
        theta_pix = theta_pix,
        pix_width = pix_width,
        scalar_matrix = cphi
    )
    quadratic_estimate = phi_covariance.replace(scalar_matrix = quadratic_estimate)

    #Create a template FlatS0 field for phi
    phi = FlatS0(
        fourier_weights = fourier_weights,
        nside = nside,
        theta_pix = theta_pix,
        pix_width = pix_width,
        basis = Basis.FOURIER,
        parametrization = Parametrization.T,
        scalar_matrix = jfft.rfft2(phi)
    )

    #Create a temperature only data set object
    if pol == "I":
        data_set = DataSetT(
            #Covariance matrices
            noise_covariance = phi_covariance.replace(scalar_matrix = cn_tt),
            field_covariance = phi_covariance.replace(scalar_matrix =  cf_tt),
            mixing_d = phi_covariance.replace(scalar_matrix = d_tt),
            phi_covariance = phi_covariance,
            mask = phi_covariance.replace(scalar_matrix = m_tt),
            beam = phi_covariance.replace(scalar_matrix = b_tt),
            quadratic_estimate = quadratic_estimate,
            
            #fields
            data = phi.replace(scalar_matrix = data_t),
            unlensed_field = phi.replace(scalar_matrix = field_t),
            lensed_field = phi.replace(scalar_matrix = lensed_field_t),
            phi = phi,
        )

    #Create a polarization only data set object
    elif pol == "P":

        #Create a template DiagonalEB covariance matrix for the noise term
        noise_covariance = DiagonalEB(
            fourier_weights = fourier_weights,
            nside = nside,
            theta_pix = theta_pix,
            pix_width = pix_width,
            matrix_EE = cn_ee,
            matrix_BB = cn_bb,
        )

        #Create a template FlatS2 field for the data term
        data = FlatS2(
            fourier_weights = fourier_weights,
            nside = nside,
            theta_pix = theta_pix,
            pix_width = pix_width,
            basis = Basis.FOURIER,
            parametrization = Parametrization.EB,
            polar_matrix_1 = data_e,
            polar_matrix_2 = data_b
        )

        data_set = DataSetEB(
            #Covariance matrices
            noise_covariance = noise_covariance,
            field_covariance = noise_covariance.replace(matrix_EE = cf_ee,
                                                        matrix_BB = cf_bb),
            mixing_d = noise_covariance.replace(matrix_EE = d_ee,
                                                matrix_BB = d_bb),
            phi_covariance = phi_covariance,
            mask = noise_covariance.replace(matrix_EE = m_ee,
                                            matrix_BB = m_bb),
            beam = noise_covariance.replace(matrix_EE = b_ee,
                                            matrix_BB = b_bb),
            quadratic_estimate = quadratic_estimate,
            
            #fields
            data = data,
            unlensed_field = data.replace(polar_matrix_1 = field_e,
                                 polar_matrix_2 = field_b),
            lensed_field = data.replace(polar_matrix_1 = lensed_field_e,
                                        polar_matrix_2 = lensed_field_b),
            phi = phi,
        )
    
    #Create a full intensity and polarization data set object
    else:
        #Create a template DiagonalEB covariance matrix for the noise term
        noise_covariance = BlockTEB(
            fourier_weights = fourier_weights,
            nside = nside,
            theta_pix = theta_pix,
            pix_width = pix_width,
            matrix_TT = cn_tt,
            matrix_TE = cn_te,
            matrix_ET = cn_te,
            matrix_EE = cn_ee,
            matrix_BB = cn_bb,
        )

        #Create a template FlatS02 field for the data term
        data = FlatS02(
            fourier_weights = fourier_weights,
            nside = nside,
            theta_pix = theta_pix,
            pix_width = pix_width,
            basis = Basis.FOURIER,
            parametrization = Parametrization.EB,
            scalar_matrix = data_t,
            polar_matrix_1 = data_e,
            polar_matrix_2 = data_b
        )

        data_set = DataSetTEB(
            #Covariance matrices
            noise_covariance = noise_covariance,
            field_covariance = noise_covariance.replace(matrix_TT = cf_tt,
                                                        matrix_TE = cf_te,
                                                        matrix_ET = cf_te,
                                                        matrix_EE = cf_ee,
                                                        matrix_BB = cf_bb),
            mixing_d = noise_covariance.replace(matrix_TT = d_tt,
                                                matrix_TE = d_te,
                                                matrix_ET = d_te,
                                                matrix_EE = d_ee,
                                                matrix_BB = d_bb),
            phi_covariance = phi_covariance,
            mask = noise_covariance.replace(matrix_TT = m_tt,
                                            matrix_TE = m_te,
                                            matrix_ET = m_te,
                                            matrix_EE = m_ee,
                                            matrix_BB = m_bb),
            beam = noise_covariance.replace(matrix_TT = b_tt,
                                            matrix_TE = b_te,
                                            matrix_ET = b_te,
                                            matrix_EE = b_ee,
                                            matrix_BB = b_bb),
            quadratic_estimate = quadratic_estimate,
            
            #fields
            data = data,
            unlensed_field = data.replace(scalar_matrix = field_t,
                                 polar_matrix_1 = field_e,
                                 polar_matrix_2 = field_b),
            lensed_field = data.replace(scalar_matrix = lensed_field_t,
                                        polar_matrix_1 = lensed_field_e,
                                        polar_matrix_2 = lensed_field_b),
            phi = phi,
        )

    return data_set

@partial(jax.jit, static_argnums = (0,))
def get_indices(length):
    parts = [unique_permutations(num_0s, length - num_0s) for num_0s in range(length + 1)]
    return jnp.concatenate(parts, axis = 0)

@partial(jax.jit, static_argnums = (0, 1))
def unique_permutations(n0, n1):
    total_length = n0 + n1
    res = []
    for indices in combinations(range(total_length), n0):
        p = [1] * total_length
        for i in indices:
            p[i] = 0
        res.append(p)
    return jnp.array(res, dtype = jnp.int32).reshape(-1, total_length)

#Main entry point for the scalar QE normalization factor
@jax.jit
def scalar_quadratic_estimate(cn_tt, cf_tt, cfl_tt, m_tt, b_tt, pix_width):

    tf = m_tt * b_tt
    sigma = tf**2 * cfl_tt + cn_tt
    ct = cf_tt
    qe_sum = np.zeros(ct.shape, dtype = jnp.float64)

    for (i, j) in get_indices(2):

        internal = get_scalar_qe_norm_at_position(tf, ct, sigma, pix_width, i, j)
        qe_sum += jnp.abs(get_specific_derivative(jfft.rfft2(internal), pix_width, i, j))

    return jnp.nan_to_num(1 / qe_sum, nan = 0, posinf = 0, neginf = 0)

#Main entry point for the polar QE normalization factor
@jax.jit
def polar_quadratic_estimate(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb, mask_ee, mask_bb, beam_ee, beam_bb, pix_width):

    qe_sum = jnp.zeros(cf_ee.shape, dtype = jnp.complex128)

    for (i, j) in get_indices(2):

        internal = polar_quadratic_estimate_internal(cf_ee, cf_bb, cfl_ee, cfl_bb, 
                                               cn_ee, cn_bb, mask_ee, mask_bb, 
                                               beam_ee, beam_bb, pix_width, i, j)
        qe_sum += jnp.abs(get_specific_derivative(jfft.rfft2(internal), pix_width, i, j))

    return jnp.nan_to_num(1 / qe_sum, nan = 0, posinf = 0, neginf = 0)

@jax.jit
def get_fourier_derivatives(f, pix_width):
    Nx, _ = f.shape
    Ny = Nx
    kx = 2 * jnp.pi * jfft.fftfreq(Nx, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(Ny, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    Fx  = 1j * KX * f
    Fy  = 1j * KY * f
    Fxx = -(KX**2) * f
    Fyy = -(KY**2) * f
    Fxy = -(KX * KY) * f

    #return the complex valued fourier fields
    return Fx, Fy, Fxx, Fxy, Fyy

@jax.jit
def get_specific_derivative(field, pix_width, type_1 = None, type_2 = None):

    #0th order derivative of a field is the field
    if type_1 is None and type_2 is None:
        return field
    
    #derivative calculations should be done in fourier space
    #for this specific QE Lef calculation
    x, y, xx, xy, yy = get_fourier_derivatives(field, pix_width)
    first_derivatives = jnp.array([x, y])
    second_derivatives = jnp.array([xx, yy, xy])

    #1st order derivatives
    if type_2 is None:
        return first_derivatives[type_1]

    def return_double_2nd_order(second_derivatives, type_1):
        return second_derivatives[type_1]
    
    def return_mixed_2nd_order(second_derivatives, type_1):
        _ = type_1
        return second_derivatives[2]

    operands = second_derivatives, type_1
    result = jax.lax.cond(
        jnp.equal(type_1, type_2),
        return_double_2nd_order,
        return_mixed_2nd_order,
        *operands
    )

    return result

@jax.jit
def levi_civita_3d():
    i, j, k = jnp.meshgrid(jnp.arange(3), jnp.arange(3), jnp.arange(3), indexing='ij')
    return (i - j) * (j - k) * (k - i) / 2

@jax.jit
def polar_quadratic_estimate_internal(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb, mask_ee, mask_bb, beam_ee, beam_bb, pix_width, i, j):

    epsilon = levi_civita_3d()
    tf2e = (mask_ee * beam_ee)**2
    tf2b = (mask_bb * beam_bb)**2
    sigma_e_tot = tf2e * cfl_ee + cn_ee
    sigma_b_tot = tf2b * cfl_bb + cn_bb
    N, _ = cf_ee.shape
    qe_sum = np.zeros((N, N), dtype = jnp.float64)

    for (k, l, m, n, p, q) in get_indices(6):

        qe_sum += 4 * epsilon[m, p, 2] * epsilon[n, q, 2] \
                    * get_polar_qe_norm_at_position(tf2e, cf_ee, sigma_e_tot, 
                                                 tf2b, cf_bb, sigma_b_tot, 
                                                 pix_width, i, j, k, l, m, n, p, q)
    return qe_sum

#QE_leg and QE_leg_internal are helper functions that make the logic easier when computing 
#different types of quadratic estimators
@jax.jit
def qe_leg(field, pix_width, indices):
    num_derivatives = len(indices["isolated"])
    squashed_indices = jnp.concatenate((indices["encapsulated"], indices["isolated"]), axis = 0)
    num_y = jnp.sum(squashed_indices)
    num_x = len(squashed_indices) - num_y
    return qe_leg_internal(field, pix_width, num_derivatives, num_x, num_y)

@jax.jit
def qe_leg_internal(field, pix_width, num_derivatives, num_x, num_y):

    #start by converting nan to 0 in case we quit early and 
    #just return the field
    field = jnp.nan_to_num(field, nan = 0)

    #for zero derivatives just return the field
    def quit_early(field, pix_width, num_derivatives, num_x, num_y):
        _ = pix_width #quiet the linter...
        _ = num_derivatives
        _ = num_x
        _ = num_y
        return field * (1.0 + 0.0j)
    
    def take_partials(field, pix_width, num_derivatives, num_x, num_y):
        #compute the laplacian raised to the n/2 power
        laplacian_power = jnp.sqrt(laplacian_2d(field, pix_width)**num_derivatives)
        
        #compute the specific derivative term
        N, _ = field.shape
        kx = 2 * jnp.pi * jfft.fftfreq(N, pix_width)
        ky = 2 * jnp.pi * jfft.rfftfreq(N, pix_width)
        KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
        x_deriv  = 1j * KX
        y_deriv  = 1j * KY
        qe_leg_value = (x_deriv**num_x * y_deriv**num_y) * field
        qe_leg_value = qe_leg_value / laplacian_power

        #gracefully handle any NaNs
        return jnp.nan_to_num(qe_leg_value, nan = 0)
    
    operands = field, pix_width, num_derivatives, num_x, num_y
    result = jax.lax.cond(
        jnp.logical_and(jnp.equal(num_x, 0), jnp.equal(num_y, 0)),
        quit_early,
        take_partials,
        *operands
    )

    return result

#2-Dimensional version of the Del^2 operator in fourier space
@jax.jit
def laplacian_2d(field, pix_width):
    Nx, _ = field.shape
    Ny = Nx
    kx = 2 * jnp.pi * jfft.fftfreq(Nx, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(Ny, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    laplacian = KX**2 + KY**2
    return laplacian

#The term we sum together and differentiate when computing the SCALAR version of the quadratic estimate
def get_scalar_qe_norm_at_position(tf, ct, sigma, pix_width, i, j):
    result = jfft.irfft2(qe_leg(tf**2*ct**2/sigma, pix_width, {"encapsulated": jnp.array([i, j]), "isolated": jnp.array([])})) \
             * jfft.irfft2(qe_leg(tf**2/sigma, pix_width, {"encapsulated": jnp.array([]), "isolated": jnp.array([])})) \
             + jfft.irfft2(qe_leg(tf**2*ct/sigma, pix_width, {"encapsulated": jnp.array([i]), "isolated": jnp.array([])})) \
             * jfft.irfft2(qe_leg(tf**2*ct/sigma, pix_width, {"encapsulated": jnp.array([j]), "isolated": jnp.array([])}))
    return result

#The term we sum together and differentiate when computing the POLAR version of the quadratic estimate
def get_polar_qe_norm_at_position(tf2e, cf_ee, sigma_e_tot, tf2b, cf_bb, sigma_b_tot, pix_width, i, j, k, l, m, n, p, q):
    result = jfft.irfft2(qe_leg(tf2e * cf_ee**2 / sigma_e_tot, pix_width, {"encapsulated": jnp.array([i, j]), "isolated": jnp.array([k, l, m, n])})) \
             * jfft.irfft2(qe_leg(tf2b / sigma_b_tot, pix_width, {"encapsulated": jnp.array([]), "isolated": jnp.array([k, l, p, q])})) \
             - 2*jfft.irfft2(qe_leg(tf2e * cf_ee / sigma_e_tot, pix_width, {"encapsulated": jnp.array([i]), "isolated": jnp.array([k, l, m, n])})) \
             * jfft.irfft2(qe_leg(tf2b * cf_bb / sigma_b_tot, pix_width, {"encapsulated": jnp.array([j]), "isolated": jnp.array([k, l, p, q])})) \
             + jfft.irfft2(qe_leg(tf2e / sigma_e_tot, pix_width, {"encapsulated": jnp.array([]), "isolated": jnp.array([k, l, m, n])})) \
             * jfft.irfft2(qe_leg(tf2b * cf_bb**2 / sigma_b_tot, pix_width, {"encapsulated": jnp.array([i, j]), "isolated": jnp.array([k, l, p, q])}))
    return result

#Pass an array of JAX RNG Keys along with the current index in the Keys array
#in order to get a fresh key and update the key counter...
def get_fresh_key(keys, key_counter):
    if key_counter >= len(keys):
        raise RuntimeError("Out of Keys Error. Please split on more keys.")
    else:
        return keys[key_counter], key_counter+1

#Get the mixing D matrix operator used in the mixed parametrization in terms
#of the field and noise covariance matrices
#TODO it would be nice to have all the arithmetic done here with the pre-defined
#matrix operator math but it's already working and time is valuable
def get_d_matrix(cf_tt, cf_te, cf_ee, cf_bb, 
                 cn_tt, cn_te, cn_ee, cn_bb):

    pre_factor = jnp.deg2rad(5/ARCMIN_PER_DEGREE)**2
    identity = jnp.ones(cn_tt.shape)

    cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee = invert_block_matrix(cf_tt, cf_te, cf_te, cf_ee)
    cf_inv_bb = reciprocal_matrix(cf_bb)

    #Avoid possible flop errors after inversion
    cf_inv_tt = jnp.nan_to_num(cf_inv_tt, nan = 0, posinf = 0, neginf = 0)
    cf_inv_te = jnp.nan_to_num(cf_inv_te, nan = 0, posinf = 0, neginf = 0)
    cf_inv_et = jnp.nan_to_num(cf_inv_et, nan = 0, posinf = 0, neginf = 0)
    cf_inv_ee = jnp.nan_to_num(cf_inv_ee, nan = 0, posinf = 0, neginf = 0)
    cf_inv_bb = jnp.nan_to_num(cf_inv_bb, nan = 0, posinf = 0, neginf = 0)

    sum_tt = (cf_tt + pre_factor * identity + 2*cn_tt)
    sum_te = (cf_te + 2*cn_te)
    sum_ee = (cf_ee + pre_factor * identity + 2*cn_ee)
    sum_bb = (cf_bb + pre_factor * identity + 2*cn_bb)

    d_matrix_tt, d_matrix_te, d_matrix_et, d_matrix_ee, d_matrix_bb = teb_matrix_mult(sum_tt, sum_te, sum_te, sum_ee, sum_bb, 
                                                                        cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee, cf_inv_bb)

    #NOTE I do not know why but the distinction between d_et and d_te seems to matter
    #a lot here... The current usage of just d_et seems to give results that
    #match Julia much better for some reason than actually doing the correct math...
    d_matrix_tt, d_matrix_te, d_matrix_et, d_matrix_ee = block_matrix_sqrt(d_matrix_tt, d_matrix_et, 
                                                                           d_matrix_et, d_matrix_ee)
    d_matrix_bb = jnp.sqrt(d_matrix_bb)

    return d_matrix_tt, d_matrix_te, d_matrix_ee, d_matrix_bb

#Given a fourier space covariance matrix, generate a Gaussian Random Field
def field_from_covar(nside, covar_matrix, rng_keys = None, key_counter = None):
    if rng_keys is None or key_counter is None:
        seed = jax.random.key(np.random.randint(0, 2**31))
    else:
        seed, key_counter = get_fresh_key(rng_keys, key_counter)
    #real and imaginary Gaussian fields both with mean 0, variance 1
    real_dist = jax.random.normal(seed, shape=(nside, nside//2 + 1))
    imag_dist = 1j * jax.random.normal(seed, shape=(nside, nside//2 + 1))
    #Rescale the variance using the fact that Var(c*X) = c^2 * Var(X)
    field = jnp.sqrt(covar_matrix/2) * (real_dist + imag_dist)
    #now transform back to real space
    field = jfft.irfft2(field, norm = "ortho")
    return field, key_counter

#Convert Dl's to Cl's
def dl2cl(dl_xx, lmax, lmax_prime, is_phi = False):

    #create lists of ell and ell primed values
    ell = jnp.asarray(list(range(2, lmax))).astype(jnp.float64) #2, 3, 4, ..., lmax - 1
    ell_prime = jnp.asarray(list(range(2, lmax_prime))).astype(jnp.float64) #2, 3, 4, ..., lmax_prime - 1

    #convert Dl's to Cl's with special logic for Phi Cl's
    if is_phi:
        #cl_pp is rescaled by ~ 1/l^4 specifically whereas the other
        #fields are rescaled by ~ 1/l^2
        cl_xx = dl_xx[2:] * 2 * jnp.pi / ell_prime**4
    else:
        #NOTE we disregard the l = 0 and l = 1 modes
        cl_xx = dl_xx[2:] * 2 * jnp.pi / (ell_prime*(ell_prime + 1))

    #interpolate and extrapolate the Cl's to higher ell values
    if jnp.all(cl_xx > 0): #if we can safely do it, use log-log interpolation
        cl_xx = jnp.interp(jnp.log(ell), jnp.log(ell_prime), jnp.log(cl_xx), left="extrapolate", right="extrapolate")
        cl_xx = jnp.exp(cl_xx)
    else: #otherwise use standard linear interpolation
        cl_xx = jnp.interp(ell, ell_prime, cl_xx, left=0.0, right=0.0)

    #return the resulting Cl's
    return cl_xx

#Given a mesh grid of fourier modes and Cl's, interpolate the Cl's on
#the mesh grid
def covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cls, origin_value = None, 
                          rescale = True, use_linear_interpolation = False, 
                          fill_value = None):
    #interpolate the cls on the grid of fourier modes.
    #use log-log interpolation if all cls greater than zero and override not specified
    if jnp.all(cls > 0) and use_linear_interpolation == False:
        ls_safe = ell_grid.flatten()
        #avoid divide by zero errors using a small-epsilon
        ls_safe = jnp.where(ls_safe == 0, 1e-10, ls_safe)
        result = jnp.interp(jnp.log(ls_safe), jnp.log(ells), jnp.log(cls), left="extrapolate", right="extrapolate")
        result = jnp.exp(result).reshape((nside, nside//2+1))
    else: #otherwise use linear-linear interpolation with a default fill_value of 0.0
        fill_value = fill_value if fill_value is not None else 0.0
        result = jnp.interp(ell_grid.flatten(), ells, cls, left = fill_value, right = fill_value)
        result = result.reshape((nside, nside//2+1))
    #replace value at origin with custom value if specified otherwise use numerical value
    if origin_value is not None:
        result = result.at[0, 0].set(origin_value)
    #return the interpolated result rescaled by radians per pixel squared if rescale is specified True
    if rescale:
        result = result / pix_width**2 
    #otherwise, return the unscaled value
    return result

#Cl's associated with atmospheric noise (i.e. so-called 1/f noise) and the beam transfer function
def noise_cls(lmax_prime, uk_arcmin_t, beam_fwhm = 0, l_knee = 100, alpha_knee = 3):
    ell_prime = jnp.asarray(list(range(2, lmax_prime))) #2, 3, 4, ..., lmax_prime - 1
    bls = beam_cls(beam_fwhm, ell_prime)
    nls = 1 + (l_knee/ell_prime)**alpha_knee
    cn_tt = jnp.deg2rad(uk_arcmin_t/ARCMIN_PER_DEGREE)**2 * nls / bls
    cn_tt = jnp.nan_to_num(cn_tt, nan = 0.0, posinf = 0.0, neginf = 0.0)
    cn_te = jnp.zeros(cn_tt.shape)
    cn_ee = 2 * cn_tt
    cn_bb = 2 * cn_tt
    return cn_tt, cn_te, cn_ee, cn_bb

#so-called beam transfer function which essentially filters off high-ell
def beam_cls(beam_fwhm, ell):
    bls = jnp.exp(-ell**2 * jnp.deg2rad(beam_fwhm/ARCMIN_PER_DEGREE)**2 / BEAM_TRANSFER_SCALAR)
    return bls

#Return the beam "B" matrix operator for a given max ell and grid of fourier modes
def get_beam(nside, pix_width, ell_grid, lmax_prime, beam_fwhm = 0):
    ell_prime = jnp.arange(2, lmax_prime).astype(jnp.float64)
    b = jnp.sqrt(beam_cls(beam_fwhm, ell_prime))
    beam = covar_matrix_from_cls(nside, pix_width, ell_grid, ell_prime, b, origin_value = 0, 
                                 rescale = False, use_linear_interpolation = True, fill_value = 1.0)
    return beam

#Define a cutoff mask for highest ell in fourier space allowed
def get_mask(l_cutoff, nside, pix_width, ell_grid):
    screen_cls = low_pass(l_cutoff)
    ell = jnp.arange(2, len(screen_cls)).astype(jnp.float64)
    #note that we always disregard the first two Cl's
    mask = covar_matrix_from_cls(nside, pix_width, ell_grid, ell, 
                                 screen_cls[2:], origin_value = 1, rescale = False)
    return mask

#create a jax array from 0 to 1 in the range of length which increases 
#in the same functional form as a cosine wave
def cos_ramp_up(length):
    result = (jnp.array([jnp.cos(x) for x in jnp.linspace(jnp.pi, 0, length)]) + 1)/2
    return result

#create a jax array from 1 to 0 in the range of length which decreases 
#in the same functional form as a cosine wave
def cos_ramp_down(length):
    result = 1 - cos_ramp_up(length)
    return result

#create a low pass filter of length lmax + 1 that is 1 all the way 
#up until the last delta_l entries where it decreases down to 0
#in the same functional form as a cosine
def low_pass(l_cutoff, delta_l = 50):
    low_ell_pass = jnp.ones(l_cutoff - delta_l + 1)
    high_ell_filter = cos_ramp_down(delta_l)
    screen = jnp.concatenate([low_ell_pass, high_ell_filter], axis = 0)
    return screen

#Run many simulations in parallel
def batch_simulated_trials(num_trials = 10, nside = 256, theta_pix = 2,
                           uk_arcmin_t = 10, lmax = 17_000, pol = "I"):
    
    #frozen wrapper function whose only input is seed to be run
    #in parallel using jax.vmap
    def parallel_sim(seed):
        return load_sim(nside = nside, theta_pix = theta_pix,
                    pol = pol, master_seed = seed, 
                    uk_arcmin_t = uk_arcmin_t, lmax = lmax)
    
    #built a list of seeds to run in parallel
    seeds = []
    for _ in range(num_trials):
        seeds.append(np.random.randint(0, 2**31))
    seeds = jnp.asarray(seeds)

    #run the load_sim() method in parallel for num_trials
    #total number of randomly generated seeds
    parallelized_load_sim = jax.vmap(parallel_sim)
    trial_results = parallelized_load_sim(seeds)
    return trial_results

if __name__ == "__main__":
    #NOTE approximately 2 minutes and 30 seconds for 100 IP trials
    start_time = time.time()
    batch_trials = batch_simulated_trials(num_trials = 100, nside = 256, theta_pix = 2,
                                          uk_arcmin_t = 10, lmax = 17_000, pol = "IP")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total run time = {total_time}")