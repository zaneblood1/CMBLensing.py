"""Flat-sky, lenspyx-style CMB lensing deflection via non-uniform FFTs (ducc0).

This is the flat-sky analogue of lenspyx's `deflection_028` remap: instead of
solving the LenseFlow ODE, the lensed field is the band-limited trigonometric
interpolant of the unlensed field evaluated directly at the deflected positions
x + grad(phi), using ducc0's type-2 NUFFT (`u2nu`). The exact transpose is the
type-1 NUFFT (`nu2u`), and the inverse deflection is a fixed-point solve on the
angles -- mirroring how delensalot/lenspyx obtain the same operators.

Conventions match CMBLensing.py (`cmb_lensing/util.py:get_primal_derivatives`):
  - a field is a real (N, N) array, axis 0 = x, axis 1 = y
  - wavenumbers k = 2*pi*fftfreq(N, pix_width) (pix_width in radians)
  - the deflection is d = grad(phi) in radians; lensed(x) = f(x + grad phi(x))

All operators are pure numpy + ducc0 (no autodiff); this is deliberate -- the
point is to compare against LenseFlow's ODE, and to expose the hand-coded
forward/adjoint/gradient that a future jax.custom_vjp would wrap.
"""
import numpy as np
import ducc0.nufft as nufft


class FlatDeflection:
    """Lensing deflection operator on a square flat-sky grid.

    Parameters
    ----------
    phi_map : (N, N) real array
        the lensing potential in real space (radians^2 convention such that
        grad(phi) is a deflection angle in radians).
    pix_width : float
        pixel width in radians (CMBLensing's `field.pix_width`).
    epsilon : float
        target NUFFT accuracy (double precision must be > 2e-13).
    nthreads : int
        ducc0 thread count (0 = all hardware threads).
    """

    def __init__(self, phi_map, pix_width, epsilon = 1e-10, nthreads = 0):
        phi_map = np.asarray(phi_map, dtype = np.float64)
        assert phi_map.ndim == 2 and phi_map.shape[0] == phi_map.shape[1], \
            "phi_map must be a square (N, N) real array"
        self.N = phi_map.shape[0]
        self.pix_width = float(pix_width)
        self.epsilon = float(epsilon)
        self.nthreads = int(nthreads)

        #wavenumber grids (rad^-1); axis 0 = x, axis 1 = y
        k = 2 * np.pi * np.fft.fftfreq(self.N, d = self.pix_width)
        self.KX = k[:, None]
        self.KY = k[None, :]

        #deflection field d = grad(phi), in radians, then in pixel units
        self.phi_x, self.phi_y = self._grad(phi_map)
        self._dx_pix = self.phi_x / self.pix_width
        self._dy_pix = self.phi_y / self.pix_width

        #forward deflected coordinates (periodicity 2*pi over N samples)
        self._coord = self._coords_from_pix(self._dx_pix, self._dy_pix)

    # ------------------------------------------------------------------
    # low-level helpers
    # ------------------------------------------------------------------
    def _grad(self, field):
        """First derivatives d/dx, d/dy via FFT (same convention as CMBLensing)."""
        F = np.fft.fft2(field)
        fx = np.fft.ifft2(1j * self.KX * F).real
        fy = np.fft.ifft2(1j * self.KY * F).real
        return fx, fy

    def _div(self, wx, wy):
        """Divergence d/dx wx + d/dy wy via FFT."""
        dwx = np.fft.ifft2(1j * self.KX * np.fft.fft2(wx)).real
        dwy = np.fft.ifft2(1j * self.KY * np.fft.fft2(wy)).real
        return dwx + dwy

    def _coords_from_pix(self, dx_pix, dy_pix):
        """Build the (N*N, 2) NUFFT coordinate array for x -> x + d."""
        idx = np.arange(self.N)
        ix, iy = np.meshgrid(idx, idx, indexing = "ij")
        px = (ix + dx_pix) * (2 * np.pi / self.N)
        py = (iy + dy_pix) * (2 * np.pi / self.N)
        return np.stack([px.ravel(), py.ravel()], axis = 1).astype(np.float64)

    def _synthesis(self, grid_complex, coord):
        """Type-2 NUFFT: evaluate the trig interpolant of `grid_complex` at coord."""
        return nufft.u2nu(grid = grid_complex, coord = coord, forward = False,
                          epsilon = self.epsilon, nthreads = self.nthreads,
                          fft_order = True)

    def _adjoint_synthesis(self, vals_complex, coord):
        """Type-1 NUFFT: exact adjoint of `_synthesis`, spreading onto an (N, N) grid."""
        out = np.zeros((self.N, self.N), dtype = np.complex128)
        return nufft.nu2u(points = vals_complex, coord = coord, forward = False,
                          epsilon = self.epsilon, nthreads = self.nthreads,
                          fft_order = True, out = out)

    def _remap_at(self, f, coord):
        """Evaluate the band-limited f at the deflected positions given by coord."""
        F = np.fft.fft2(np.asarray(f, dtype = np.float64))
        vals = self._synthesis(F, coord)
        return (vals.real / self.N ** 2).reshape(self.N, self.N)

    # ------------------------------------------------------------------
    # public operator interface (mirrors LenseFlow's L, L', L^-1)
    # ------------------------------------------------------------------
    def lense(self, f):
        """Forward lensing  L f (x) = f(x + grad phi(x))."""
        return self._remap_at(f, self._coord)

    def lense_adjoint(self, g):
        """Transpose  L' g  (exact adjoint of `lense` in the real pixel inner product).

        Derivation (u2nu/nu2u are a *bilinear* transpose pair, same forward flag):
            <L f, g> = (1/N^2) Re sum(fft2(f) * nu2u(g)) = <f, (1/N^2) Re fft2(nu2u(g))>.
        """
        g = np.asarray(g, dtype = np.float64).ravel().astype(np.complex128)
        grid = self._adjoint_synthesis(g, self._coord)
        return (np.fft.fft2(grid).real) / self.N ** 2

    def lense_inverse(self, g, n_iter = 10, tol = 1e-12, return_info = False):
        """Approximate inverse  L^-1 g (x) = g(x + d_inv(x)).

        d_inv is the inverse deflection, found by the fixed-point iteration
        s_{k+1} = x - grad phi(x + s_k) (i.e. Newton on the displacement).
        Only needed if the mixed parametrization is retained (see plan).
        """
        idx = np.arange(self.N)
        ix, iy = np.meshgrid(idx, idx, indexing = "ij")
        #precompute Fourier grids of the deflection components for evaluation
        Fdx = np.fft.fft2(self._dx_pix)
        Fdy = np.fft.fft2(self._dy_pix)

        sx = np.zeros((self.N, self.N))
        sy = np.zeros((self.N, self.N))
        info = []
        for _ in range(n_iter):
            coord = np.stack([((ix + sx) * (2 * np.pi / self.N)).ravel(),
                              ((iy + sy) * (2 * np.pi / self.N)).ravel()], axis = 1)
            #deflection evaluated at the current guessed source position
            dx_here = self._synthesis(Fdx, coord).real.reshape(self.N, self.N) / self.N ** 2
            dy_here = self._synthesis(Fdy, coord).real.reshape(self.N, self.N) / self.N ** 2
            sx_new = -dx_here
            sy_new = -dy_here
            step = max(np.max(np.abs(sx_new - sx)), np.max(np.abs(sy_new - sy)))
            sx, sy = sx_new, sy_new
            info.append(step)
            if step < tol:
                break
        coord = np.stack([((ix + sx) * (2 * np.pi / self.N)).ravel(),
                          ((iy + sy) * (2 * np.pi / self.N)).ravel()], axis = 1)
        out = self._remap_at(g, coord)
        if return_info:
            return out, info
        return out

    def grad_phi(self, f, cotangent, fourier_out = False):
        """Gradient of the deflection w.r.t. phi contracted with `cotangent`.

        For J = f -> some scalar with dJ/d(Lf) = cotangent, this returns dJ/dphi:
            dJ/dphi = -div( cotangent * (grad f)(x + grad phi) ).
        This is the flat-sky analogue of delensalot's QE gradient leg
        (`MAP_opfilt_iso_p.get_qlms`) and the backward a custom_vjp needs.

        If `fourier_out`, the divergence is returned directly in rfft2 space
        (shape (N, N//2+1), complex) without a closing irfft2, so the i*k content
        on the self-conjugate DC/Nyquist columns survives -- required when phi is
        carried in the FOURIER basis (see CMBLensing.py CLAUDE.md).
        """
        f = np.asarray(f, dtype = np.float64)
        cotangent = np.asarray(cotangent, dtype = np.float64)
        fx, fy = self._grad(f)
        #(grad f) evaluated at the deflected positions
        gfx = self._remap_at(fx, self._coord)
        gfy = self._remap_at(fy, self._coord)
        wx = cotangent * gfx
        wy = cotangent * gfy
        if not fourier_out:
            return -self._div(wx, wy)
        #rfft2-space divergence, matching CMBLensing's get_k_meshgrid convention
        #(axis 0 = full fftfreq, axis 1 = rfftfreq with negated Nyquist)
        kx = 2 * np.pi * np.fft.fftfreq(self.N, d = self.pix_width)[:, None]
        ky = 2 * np.pi * np.fft.rfftfreq(self.N, d = self.pix_width)
        ky = ky.copy(); ky[-1] = -ky[-1]
        ky = ky[None, :]
        return -(1j * kx * np.fft.rfft2(wx) + 1j * ky * np.fft.rfft2(wy))
