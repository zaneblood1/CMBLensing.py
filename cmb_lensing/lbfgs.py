import jax
jax.config.update("jax_enable_x64", True)

#delensalot-style L-BFGS two-loop recursion (delensalot bfgs.py::get_mHkgk), ported to
#CMBLensing field objects. This is a line-search-free quasi-Newton search direction:
#delensalot never evaluates a scalar objective to pick a step, it applies an L-BFGS
#inverse-Hessian (built from a fixed H0 and stored curvature pairs) to the gradient and
#takes a fixed step. Its H0 = (N0^-1 + Cphi^-1)^-1 is identical in form to CMBLensing's
#hessian = pinv(Cphi^-1 + QE^-1).
#
#We MAXIMIZE logpdf, so the minimized objective is (-logpdf) with gradient g = -grad_logpdf,
#and H0 = hessian is SPD (approximates the inverse of the pos-def Hessian of -logpdf). The
#recursion returns -H_k g_k = +H_k * grad_logpdf, an ASCENT direction on logpdf. With an
#empty history it reduces exactly to apply_H0(grad_logpdf) = hessian * grad_logpdf (the unit
#Fisher / Wiener-filtered QE step).
#
#Everything is elementwise field arithmetic (+, -, scalar *) plus the injected apply_H0
#(DiagonalScalar * field) and dot_op (fields.dot). Nothing routes through irfft2, so the
#physically meaningful anti-Hermitian DC/Nyquist imaginary content of a FOURIER phi gradient
#is preserved (see cmb_lensing/CLAUDE.md, "The Phi Gradient in Fourier Space").
#
#s_list, y_list are plain python lists of FOURIER FlatS0 fields (most recent last), with
#s_i = phi_{i+1} - phi_i and y_i = g_{i+1} - g_i = -(grad_{i+1} - grad_i). Memory truncation
#to L pairs is done by the caller (pop the oldest), so looping over the whole list here is
#equivalent to delensalot's windowed range(k-1, k-L-1, -1).
def lbfgs_direction(grad_logpdf, s_list, y_list, apply_H0, dot_op):
    g = -1 * grad_logpdf
    q = g
    k = len(s_list)
    alphas = [None] * k
    #dot_op(s_i, y_i) > 0 is guaranteed by the caller's curvature guard
    rho = [1.0 / dot_op(s_list[i], y_list[i]) for i in range(k)]
    for i in range(k - 1, -1, -1):
        alphas[i] = rho[i] * dot_op(s_list[i], q)
        q = q - alphas[i] * y_list[i]
    r = apply_H0(q)
    for i in range(k):
        beta = rho[i] * dot_op(y_list[i], r)
        r = r + (alphas[i] - beta) * s_list[i]
    return -1 * r
