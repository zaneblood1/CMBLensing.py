import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from cmb_lensing.fields import *
from cmb_lensing.constants import *
from cmb_lensing.util import rk4_solve, get_primal_derivatives, \
get_primal_derivatives_from_fourier, get_primal_derivatives_to_fourier
from functools import singledispatch

@jax.custom_vjp
@jax.jit
def lense_flow_wrapper(field, phi, n = 10, direction = 1, adjoint = False):
    return lense_flow(field, phi, n, direction, adjoint)

@jax.jit
def lense_flow_forward(field, phi, n, direction, adjoint):
    #compute primal outputs
    transformed_field = lense_flow(field, phi, n, direction, adjoint)
    #return the value of the transformed fields and also store any
    #data needed by the backwards pass
    return transformed_field, (transformed_field, field, phi, n, direction, adjoint)

@jax.jit
def lense_flow_backwards(residuals, cotangent):
    #unpack the necessary data
    (transformed_field, field, phi, n, direction, adjoint) = residuals
    
    #get the value of the gradients of the lensing operator w.r.t. the fields and phi
    #NOTE we need to use the transformed fields as the inputs here and integrate in the opposite direction
    _,  delta_field, delta_phi = get_lensing_operator_gradients(phi, transformed_field, cotangent, -direction, n)

    #we also need to undo the multiplication by the fourier weights and conjugation that is done
    #when taking the logpdf inner products
    delta_field = undo_inner_product(delta_field)
    delta_phi = undo_inner_product(delta_phi)

    #Use autodiff for the rest of the inputs.
    def lense_only_others(*other_inputs):
        return lense_flow(field, phi, *other_inputs)

    #Now we get their VJPs in one call:
    _, vjp_fun = jax.vjp(lense_only_others, n, direction, adjoint)
    other_gradients = vjp_fun(cotangent) 

    #Return a gradient for EVERY input (either via a custom analytical gradient or via AutoDiff)
    return (delta_field, delta_phi, *other_gradients)

# -----------------------------------------------------------
# Register the two-pass rule for the lense flow wrapper
# -----------------------------------------------------------
lense_flow_wrapper.defvjp(lense_flow_forward, lense_flow_backwards)

@jax.jit
def get_lensing_operator_gradients(phi, tqu_field, tqu_cotangent, direction, n):

    #default is forward integration operations
    def forward(_):
        t0, t1 = 0.0, 1.0
        dt0 = 1.0/n
        return t0, t1, dt0

    #set to negative 1 for inverse integration operations
    def inverse(_):
        t0, t1 = 1.0, 0.0
        dt0 = -1.0/n
        return t0, t1, dt0

    #jax trace requires if statements to be written like this
    t0, t1, dt0 = jax.lax.cond(
        jnp.equal(direction, INVERSE_LENSE),
        inverse,
        forward,
        operand = None
    )

    #precompute phi partials. phi may be a square real-space (MAP) matrix or a
    #rectangular rfft2 (FOURIER) matrix. When FOURIER the gradient w.r.t. phi
    #(delta_phi) is assembled/accumulated in fourier space so the i*k adjoint's
    #anti-Hermitian content on the self-conjugate Nyquist row survives, matching
    #CMBLensing.jl's negδvelocityᴴ (which accumulates δϕ via -∇'*Ð(...) in fourier space)
    pix_width = phi.pix_width
    shape_x = phi.scalar_matrix.shape[0]
    shape = (shape_x, shape_x)
    phi_fourier = (phi.scalar_matrix.shape[0] != phi.scalar_matrix.shape[1])
    if phi_fourier:
        phi_x, phi_y, phi_xx, phi_xy, phi_yy = get_primal_derivatives_from_fourier(phi.scalar_matrix, pix_width)
        delta_phi_shape = phi.scalar_matrix.shape
        delta_phi_init = jnp.zeros(delta_phi_shape, dtype = jnp.complex128)
    else:
        phi_x, phi_y, phi_xx, phi_xy, phi_yy = get_primal_derivatives(phi.scalar_matrix, pix_width)
        delta_phi_shape = shape
        delta_phi_init = jnp.zeros(delta_phi_shape)

    #unpack transformed_field and cotangent into T, QU, or TQU matrices (3 returned)
    t_field, q_field, u_field = spin_vector_matrix_data(tqu_field)
    delta_t_init, delta_q_init, delta_u_init = spin_vector_matrix_data(tqu_cotangent)

    y0 = (t_field.ravel(), q_field.ravel(), u_field.ravel(),
          delta_t_init.ravel(), delta_q_init.ravel(), delta_u_init.ravel(),
          delta_phi_init.ravel())

    #store extra arguments in a single tuple
    args = (phi_x, phi_y, phi_xx, phi_xy, phi_yy, pix_width, phi_fourier)

    #solve with RK4 to match Julia's CMBLensing.jl ODE solver
    result = rk4_solve(lensing_gradients_integration_step, y0, t0, t1, n, args)

    #pull out the primal output and keep the field in MAP space
    t_field = result[0].reshape(shape)
    q_field = result[1].reshape(shape)
    u_field = result[2].reshape(shape)
    delta_t = result[3].reshape(shape)
    delta_q = result[4].reshape(shape)
    delta_u = result[5].reshape(shape)
    delta_phi = result[6].reshape(delta_phi_shape)

    #NOTE the following "output_like_input" method is a hacky way of making sure
    #T --> T, QU --> QU, and TQU --> TQU even though all three fields were
    #integrated in tandem. This makes it so that we do not have to define a custom
    #lensing_operator_gradient for each style of input since we always convert inputs
    #{T, QU, TQU}  --> {TQU} and then convert our outpus {TQU}  --> {T, QU, TQU} depending
    #on what the input shape originally was...
    tqu_field_prime = output_like_input(tqu_field, t_field, q_field, u_field)
    delta_tqu_field = output_like_input(tqu_field, delta_t, delta_q, delta_u)
    delta_phi = phi.replace(scalar_matrix = delta_phi)
    return tqu_field_prime, delta_tqu_field, delta_phi

@singledispatch
@jax.jit                                                                                                                                                                             
def output_like_input(input_field, t_matrix, q_matrix, u_matrix):                              
    raise TypeError(f"Unsupported type: {jax.typeof(input_field)}")

#Temperature only                                                                                                                                                                                                
@output_like_input.register(FlatS0)
@jax.jit
def _(input_field, t_matrix, q_matrix, u_matrix):
    return input_field.replace(scalar_matrix = t_matrix)

#Polarization only                                                                                                                                                                                                
@output_like_input.register(FlatS2)
@jax.jit
def _(input_field, t_matrix, q_matrix, u_matrix):
    return input_field.replace(polar_matrix_1 = u_matrix, 
                               polar_matrix_2 = q_matrix)

#Temperature and Polarization                                                                                                                                                                                                
@output_like_input.register(FlatS02)
@jax.jit
def _(input_field, t_matrix, q_matrix, u_matrix):
    return input_field.replace(scalar_matrix = t_matrix, 
                               polar_matrix_1 = u_matrix, 
                               polar_matrix_2 = q_matrix)

#NOTE not @jax.jit'd: always called inside the jitted get_lensing_operator_gradients,
#so the static phi_fourier flag (selecting array dtype/shape) stays a Python bool
def lensing_gradients_integration_step(time, y, args):
    #unpack the args array
    phi_x, phi_y, phi_xx, phi_xy, phi_yy, pix_width, phi_fourier = args

    #reshape 1D coupled raveled fields into 2D fields.
    t, q, u, delta_t, delta_q, delta_u, _ = y
    shape = phi_x.shape

    t = t.reshape(shape)
    q = q.reshape(shape)
    u = u.reshape(shape)

    #delta_tqu is a small parameter we will integrate in tandem with delta_phi
    delta_t = delta_t.reshape(shape)
    delta_q = delta_q.reshape(shape)
    delta_u = delta_u.reshape(shape)

    #calculate the three rates of change for f, delta_phi and delta_f:
    #1. The df/dt term is just the normal lensing term applied to the full field "f"
    dtemp_dt = get_standard_lensing_term(t, phi_x, phi_y, phi_xx, phi_xy, phi_yy, time, pix_width)
    dq_dt = get_standard_lensing_term(q, phi_x, phi_y, phi_xx, phi_xy, phi_yy, time, pix_width)
    du_dt = get_standard_lensing_term(u, phi_x, phi_y, phi_xx, phi_xy, phi_yy, time, pix_width)

    #2. The d_delta_f/dt term is the adjoint lensing term applied to delta_f
    d_delta_temp_dt = get_adjoint_lensing_term(delta_t, phi_x, phi_y, phi_xx, phi_xy, phi_yy, time, pix_width)
    d_delta_q_dt = get_adjoint_lensing_term(delta_q, phi_x, phi_y, phi_xx, phi_xy, phi_yy, time, pix_width)
    d_delta_u_dt = get_adjoint_lensing_term(delta_u, phi_x, phi_y, phi_xx, phi_xy, phi_yy, time, pix_width)

    #3. The d_delta_phi_dt term is a little more involved... See the function definition for the inner workings
    d_delta_phi_dt = get_delta_phi_tqu_roc(t, q, u, delta_t, delta_q, delta_u,
                                           phi_x, phi_y, phi_xx, phi_xy, phi_yy,
                                           time, pix_width, phi_fourier)
    
    #return the three coupled ODE dynamics raveled up into 1D arrays
    return (dtemp_dt.ravel(), dq_dt.ravel(), du_dt.ravel(), d_delta_temp_dt.ravel(), 
           d_delta_q_dt.ravel(), d_delta_u_dt.ravel(), d_delta_phi_dt.ravel())

#NOTE not @jax.jit'd: called inside the jitted get_lensing_operator_gradients with a
#static (Python bool) phi_fourier flag selecting the real-space vs fourier assembly
def get_delta_phi_tqu_roc(t, q, u, delta_t, delta_q, delta_u,
                          phi_x, phi_y, phi_xx, phi_xy, phi_yy,
                          time, pix_width, phi_fourier = False):

    #We need to begin by taking the point-wise product of
    #the delta_f field and the x and y components of the gradient of f
    dt_dx, dt_dy, _, _, _ = get_primal_derivatives(t, pix_width)
    dq_dx, dq_dy, _, _, _ = get_primal_derivatives(q, pix_width)
    du_dx, du_dy, _, _, _ = get_primal_derivatives(u, pix_width)
    tdt_product_x = delta_t * dt_dx
    tdt_product_y = delta_t * dt_dy
    qdq_product_x = delta_q * dq_dx
    qdq_product_y = delta_q * dq_dy
    udu_product_x = delta_u * du_dx
    udu_product_y = delta_u * du_dy

    #Compute the spin adjoint product SAP = I^2 + Q^2 + U^2...?
    fdf_product_x = tdt_product_x + qdq_product_x + udu_product_x
    fdf_product_y = tdt_product_y + qdq_product_y + udu_product_y

    #now we take the components of this "fdf product" vector and multiply
    #by the inverse magnification matrix on the left...
    #Let's first get the inverse magnification matrix components
    m_xx, m_xy, _, m_yy = get_m_matrix_components_at_time(time, phi_xx, phi_xy, phi_yy) 
    m_inv_xx, m_inv_xy, m_inv_yy = get_inverse_matrix_components(m_xx, m_xy, m_yy)
    #now we can compute the components we want
    m_inv_fdf_x = m_inv_xx*fdf_product_x + m_inv_xy*fdf_product_y
    m_inv_fdf_y = m_inv_xy*fdf_product_x + m_inv_yy*fdf_product_y
    #we are almost there... Now we need to compute the tensor product between the m_inv_fdf vector and the standard p vector
    #then apply a "laplacian" style operator to these components
    p_x, p_y = get_p_vector_components(phi_x, phi_y, phi_xx, phi_xy, phi_yy, time)
    w_xx = p_x * m_inv_fdf_x
    w_xy = p_x * m_inv_fdf_y
    w_yx = p_y * m_inv_fdf_x
    w_yy = p_y * m_inv_fdf_y
    #the divergence + laplacian below are the FINAL derivatives building d_delta_phi_dt.
    #When phi_fourier specified we keep their output in rfft2 space (no final irfft2) so the i*k
    #content on the self-conjugate Nyquist row survives (matching CMBLensing.jl)
    if phi_fourier:
        delta_phi_div_term = (get_primal_derivatives_to_fourier(m_inv_fdf_x, pix_width)[0]
                              + get_primal_derivatives_to_fourier(m_inv_fdf_y, pix_width)[1])
        laplacian_xx = get_primal_derivatives_to_fourier(time * w_xx, pix_width)[2]
        laplacian_xy = get_primal_derivatives_to_fourier(time * w_xy, pix_width)[3]
        laplacian_yx = get_primal_derivatives_to_fourier(time * w_yx, pix_width)[3]
        laplacian_yy = get_primal_derivatives_to_fourier(time * w_yy, pix_width)[4]
    else:
        #next we take the divergence of this peculiar vector
        delta_phi_div_term_x, _, _, _, _ = get_primal_derivatives(m_inv_fdf_x, pix_width)
        _, delta_phi_div_term_y, _, _, _ = get_primal_derivatives(m_inv_fdf_y, pix_width)
        delta_phi_div_term = delta_phi_div_term_x + delta_phi_div_term_y
        _, _, laplacian_xx, _, _  = get_primal_derivatives(time * w_xx, pix_width)
        _, _, _, laplacian_xy, _ = get_primal_derivatives(time * w_xy, pix_width)
        _, _, _, laplacian_yx, _ = get_primal_derivatives(time * w_yx, pix_width)
        _, _, _, _, laplacian_yy = get_primal_derivatives(time * w_yy, pix_width)
    laplacian_sum = laplacian_xx + laplacian_xy + laplacian_yx + laplacian_yy
    #the final form of the time rate of change of delta_phi is the laplacian sum minus the divergence term
    d_delta_phi_dt = laplacian_sum + delta_phi_div_term
    return d_delta_phi_dt

@singledispatch
@jax.jit                                                                                                                                                                             
def spin_vector_matrix_data(field):                                     
    raise TypeError(f"Unsupported type: {jax.typeof(field)}")

#Temperature only                                                                                                                                                                                                
@spin_vector_matrix_data.register(FlatS0)
@jax.jit
def _(field):
    T_data = field.scalar_matrix
    Q_data = jnp.zeros_like(T_data)
    U_data = jnp.zeros_like(T_data)
    return T_data, Q_data, U_data

#QU only                                                                                                                                                                                                
@spin_vector_matrix_data.register(FlatS2)
@jax.jit
def _(field):
    Q_data = field.polar_matrix_1
    U_data = field.polar_matrix_2
    T_data = jnp.zeros_like(Q_data)
    return T_data, Q_data, U_data

#Full TQU parametrization                                                                                                                                                                                               
@spin_vector_matrix_data.register(FlatS02)
@jax.jit
def _(field):
    T_data = field.scalar_matrix
    Q_data = field.polar_matrix_1
    U_data = field.polar_matrix_2
    return T_data, Q_data, U_data

#Lense flow is just applied sequentially to each data matrix in the field
#NOTE lense flow only works in the MAP basis and EB must be converted to QU
#before the function is applied
def lense_flow(field, phi, n = 10, direction = 1, adjoint = False):
    updates = {name: primal_lense_flow(getattr(field, name), phi.scalar_matrix,
               field.pix_width, n, direction, adjoint) for name in field._matrix_names()}
    return field.replace(**updates)

#Given a lensing potential phi and a field to be lensed, compute and return the lensed field
def primal_lense_flow(primal_field, phi, pix_width, n, direction, adjoint):

    #precompute phi partials. phi may be a square real-space (MAP) matrix or a
    #rectangular rfft2 (FOURIER) matrix; same forward value either way
    if phi.shape[0] != phi.shape[1]:
        phi_x, phi_y, phi_xx, phi_xy, phi_yy = get_primal_derivatives_from_fourier(phi, pix_width)
    else:
        phi_x, phi_y, phi_xx, phi_xy, phi_yy = get_primal_derivatives(phi, pix_width)

    #default is forward lensing operations
    def forward(_):
        t0, t1 = 0.0, 1.0
        dt0 = 1.0/n
        return t0, t1, dt0

    #set to negative 1 for inverse lensing operations
    def inverse(_):
        t0, t1 = 1.0, 0.0
        dt0 = -1.0/n
        return t0, t1, dt0

    #jax trace requires if statements to be written like this
    t0, t1, dt0 = jax.lax.cond(
        jnp.logical_xor(jnp.equal(direction, FORWARD_LENSE), adjoint),
        forward,
        inverse,
        operand = None
    )

    #ravel up 2D array into 1D array for the ODE solver
    y0 = primal_field.ravel()
    #store extra arguments in a single tuple
    args = (phi_x, phi_y, phi_xx, phi_xy, phi_yy, pix_width, adjoint)

    #solve with RK4 to match Julia's CMBLensing.jl ODE solver
    y_final = rk4_solve(single_lense_flow_step, y0, t0, t1, n, args)

    #reshape the flattened form into the 2D format
    return y_final.reshape(primal_field.shape)

@jax.jit
def single_lense_flow_step(t, y, args):
    #unpack the args array
    phi_x, phi_y, phi_xx, phi_xy, phi_yy, pix_width, adjoint = args
    #reshape y into 2D field
    shape = phi_x.shape
    field = y.reshape(shape)
    #the rate of change per pixel (i.e. the form of df/dt) has a different form 
    #depending on whether we are taking the adjoint or not
    operands = field, phi_x, phi_y, phi_xx, phi_xy, phi_yy, t, pix_width
    rate = jax.lax.cond(
        adjoint,
        get_adjoint_lensing_term, #the adjoint rate of change is given by -1*divergence(p_vec*f)
        get_standard_lensing_term, #the non-adjoint rate of change is given by (grad_phi * v) <--> (grad_phi * M^-1 * grad_f)
        *operands
    )
    #return the flattened vector
    return rate.ravel()

@jax.jit
def get_adjoint_lensing_term(field, phi_x, phi_y, phi_xx, phi_xy, phi_yy, t, pix_width):
    p_vec_x, p_vec_y = get_p_vector_components(phi_x, phi_y, phi_xx, phi_xy, phi_yy, t)
    #equivalent to <--> d/dx (f * (grad_phi * M^-1)_x)
    divergence_x, _, _, _, _ = get_primal_derivatives(p_vec_x * field, pix_width)
    #equivalent to <--> d/dy (f * (grad_phi * M^-1)_y)
    _, divergence_y, _, _, _ = get_primal_derivatives(p_vec_y * field, pix_width)     
    adjoint_term = divergence_x + divergence_y
    return adjoint_term

@jax.jit
def get_standard_lensing_term(field, phi_x, phi_y, phi_xx, phi_xy, phi_yy, t, pix_width):
    #compute the vector components of v = M^{-1} * grad_f
    vx, vy = get_v_vector_components(t, field, phi_xx, phi_xy, phi_yy, pix_width)
    #return df/dt = grad_phi^T * M^{-1} * grad_f
    standard_lensing_term = phi_x * vx + phi_y * vy
    return standard_lensing_term

@jax.jit
def get_p_vector_components(phi_x, phi_y, phi_xx, phi_xy, phi_yy, time):
    #begin by getting the inverse magnification matrix components
    m_xx, m_xy, _, m_yy = get_m_matrix_components_at_time(time, phi_xx, phi_xy, phi_yy) 
    #m_xy = m_yx so only one off-diagonal component is needed
    m_inv_xx, m_inv_xy, m_inv_yy = get_inverse_matrix_components(m_xx, m_xy, m_yy)
    #given the gradients and inverse magnification matrix components compute and return the p-vector components
    p_x = phi_x * m_inv_xx + phi_y * m_inv_xy
    p_y = phi_x * m_inv_xy + phi_y * m_inv_yy
    return p_x, p_y

@jax.jit
def get_v_vector_components(time, field, phi_xx, phi_xy, phi_yy, pix_width):
    #the derivatives of f change throughout time so we need to compute them at each call
    df_dx, df_dy, _, _, _ = get_primal_derivatives(field, pix_width)
    #get the inverse magnification matrix components
    m_xx, m_xy, _, m_yy = get_m_matrix_components_at_time(time, phi_xx, phi_xy, phi_yy) 
    #m_xy = m_yx via symmetry so only one off-diagonal component is needed
    m_inv_xx, m_inv_xy, m_inv_yy = get_inverse_matrix_components(m_xx, m_xy, m_yy)
    #compute the v vector components given the input data
    v_x = df_dx * m_inv_xx + df_dy * m_inv_xy
    v_y = df_dy * m_inv_yy + df_dx * m_inv_xy
    #return the calculated values
    return v_x, v_y

@jax.jit
def get_m_matrix_components_at_time(time, phi_xx, phi_xy, phi_yy):
    #the derivatives of phi do not change with time so we can reuse their values
    m_xx = 1 + time * phi_xx
    m_xy = time * phi_xy
    m_yy = 1 + time * phi_yy
    m_yx = m_xy #NOTE this matrix is symmetric
    #return the components of the magnification matrix
    return m_xx, m_xy, m_yx, m_yy

@jax.jit
def get_inverse_matrix_components(m_xx, m_xy, m_yy, eps = 1e-12):
    det = m_xx * m_yy - m_xy * m_xy
    #avoid divide by zero errors by supplying a small epsilon
    inv_det = 1.0 / (det + eps) 
    m_inv_xx =  m_yy * inv_det
    m_inv_xy = -m_xy * inv_det
    m_inv_yy =  m_xx * inv_det
    return m_inv_xx, m_inv_xy, m_inv_yy
