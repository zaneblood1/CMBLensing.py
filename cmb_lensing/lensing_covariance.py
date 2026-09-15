"""Non-Gaussian bandpower covariance of (TT, phi-phi) power estimates on the flat-sky box.

fisher_forecast's "cls" path contracts derivative vectors against the covariance of the
per-mode power estimates,  F_ij = dmu_i^T Cov(mu_hat)^-1 dmu_j.  With a covariance that is
diagonal in the modes that is identical to the trace formula ("blocks"): every mode adds a
non-negative rank-one piece of information and nothing can ever subtract any. This module
supplies the part of the covariance that is NOT diagonal - the coupling between different
multipoles that gravitational lensing induces - so that the "cls" path can describe
information being SHARED between the temperature two-point function and the lensing
four-point function rather than counted twice.

THE PHYSICS, at first order in C_phi (Hu 2000; Benoit-Levy, Smith & Hu 2012). On the flat
sky the lensed temperature spectrum is a convolution of the unlensed one with the lensing
potential spectrum,

    C~(l) = C(l) [1 - l^2 R] + (1/A) sum_L [(l - L).L]^2 C_phi(L) C(|l - L|),
    R     = (1/A) sum_L L^2 C_phi(L) / 2,        A = (nside * pix_width)^2,

so a fluctuation of the REALIZATION's lensing power at wavevector L, dP(L), moves the
observed TT power at every l through the response kernel

    K(l, L) = dC~(l) / dC_phi(L) = (1/A) { [(l - L).L]^2 C(|l - L|) - l^2 L^2 C(l) / 2 }.

That single realization-level fact produces all three non-Gaussian terms:

  TT-TT     Cov(C^_l, C^_l')  +=  sum_L K(l, L) K(l', L) Var(P_L),   Var(P_L) = 2 C_phi(L)^2 / w_L
            the lensing sample variance of BLSH12 (their dominant term; the second term of
            that paper, the unlensed field's own sample variance propagated through the
            lensing, is nearly diagonal and is what the ordinary Gaussian diagonal of the
            lensed spectrum already approximates - so it is not added a second time)
  TT-phiphi Cov(C^_l, C^_L)   =   K(l, L) Cov(P_L, P^_L)
            the same realization's phi power appears in both estimates (Schmittfull et al.
            2013; Peloton et al. 2017's "signal" cross term)
  phiphi    stays Gaussian, 2 (C_phi + N_phi)^2 / w_L, as in covariance_blocks

with w_L the real degrees of freedom of each rfft entry (util.get_fourier_weights).

WHAT IT MEANS FOR EACH SPECTRA MODE. Write the realization's potential as the part the
reconstruction recovers plus the part it does not, phi = phi_rec + phi_res. For a Wiener
reconstruction the two are uncorrelated and Var(phi_res) = C_phi N_phi / (C_phi + N_phi).
  "lensed"    nothing is delensed: the TT power responds to the FULL phi (C_eff = C_phi) and
              the phi-phi estimate |phi + n|^2 shares it, so the cross term is on.
  "delensed"  the delensed TT power responds only to the RESIDUAL (C_eff = C_phi N_phi /
              (C_phi + N_phi), per mode), which is uncorrelated with the reconstructed
              |phi_rec|^2 - so the TT-phiphi cross term vanishes at this order and the
              TT-TT term shrinks to the residual's sample variance.
  "unlensed" / "ceiling"  no residual lensing in the field block at all, so no
              non-Gaussian term: the covariance is the Gaussian diagonal and "cls"
              reproduces "blocks" exactly - the identity fisher_forecast documents.

IMPLEMENTATION. Nothing here is a matrix. The kernel is applied by FFT convolutions - the
polynomial [(l - L).L]^2 splits into three ordinary convolutions of l_a l_b C(l) against
L_a L_b v(L) - so Cov(mu_hat) is a matrix-free linear operator on the stacked (TT, phiphi)
per-mode vector, and F_ij = dmu_i . x_j with x_j = Cov^-1 dmu_j solved by preconditioned
conjugate gradients (scipy.sparse.linalg.cg). That keeps the contraction exact per mode -
no banding, no dense 16 000 x 16 000 matrix - and costs a few dozen 2D FFTs per iteration.

The convolutions are CIRCULAR on the nside x nside grid, deliberately: the simulated box
is periodic and lense_flow forms the lensed map from products of periodic grid functions,
so its Fourier-space mode coupling aliases in exactly this way. A real sky would call for
zero padding; the difference lives at the corners of the box where the noise dominates.

BOOKKEEPING. Everything is carried in the codebase's own units (covar_matrix_from_cls's
C_l / pix_width^2) - the Fisher is invariant to a consistent rescaling of the estimates -
which turns the 1/A of the continuum formula into 1/nside^2. Per-mode vectors live on the
INDEPENDENT modes of the rfft half plane (independent_modes: one entry per conjugate
pair, since the half plane's two self-conjugate columns otherwise hold each pair twice)
restricted to the modes with positive power; the FFTs live on the full plane. An even
function of the wavevector is the same thing in both, and every quantity the kernel
touches is even (spectra, derivatives, and the outputs of K and K^T acting on even
inputs), so independent_to_full / full_to_half move between them without loss. The one
place the real-DOF weights enter the non-Gaussian terms is the identity
    sum_{L, full plane} K(l, L) v(L) = sum_{L, independent} w_L K_sym(l, L) v(L)
for even v, which is what lets a full-plane FFT sum stand in for the covariance sum; the
1/w on the way in and the w inside Var(P_L) cancel to leave
    Cov_NG(TT, TT) = 2 K diag(C_eff^2) K^T,   Cov(TT, phiphi) = 2 K diag(C_eff^2),
applied through K and K^T on evenly-extended vectors divided by w. The kernel itself is
symmetrized under joint negation of its two wavevectors (see LensingKernel), which the
periodic grid's Nyquist convention would otherwise break.

VALIDATION. first_order_lensing_check applies the kernel to the fiducial C_phi and
compares the resulting first-order lensing correction against CAMB's own
C_lensed - C_unlensed on the grid, annulus by annulus. That pins the 1/nside^2, the
wavevector conventions and the FFT conventions in one number per annulus - the ratio must
sit near 1 where first-order lensing is accurate (ell below ~1500) and drift away above,
where the first-order approximation is known to break down. tests/test_lensing_covariance.py
checks the operator against a brute-force kernel matrix on a tiny grid.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def full_wavevectors(nside, pix_width):
    """(KX, KY) on the FULL nside x nside plane, fftfreq convention on both axes.

    The Nyquist entry is negative on both axes, which agrees with util.get_k_meshgrid's
    convention (it flips the rfft axis's Nyquist to negative by hand) and with the sign
    the lensing code uses for its derivative operators.
    """
    k = 2 * np.pi * np.fft.fftfreq(nside, pix_width)
    return np.meshgrid(k, k, indexing = "ij")


def half_to_full(half):
    """Even extension of an rfft-half-plane array (nside, nside//2+1) to the full plane.

    Column j > nside//2 of the full plane is the mode -(i, nside - j), so it is read from
    the half plane at row (-i) mod nside, column nside - j. Only valid for even functions
    of the wavevector - which is every spectrum, derivative and kernel output used here.
    """
    half = np.asarray(half)
    nside = half.shape[0]
    full = np.zeros((nside, nside), dtype = half.dtype)
    full[:, :nside // 2 + 1] = half
    rows = (-np.arange(nside)) % nside
    for j in range(nside // 2 + 1, nside):
        full[:, j] = half[rows, nside - j]
    return full


def full_to_half(full):
    nside = full.shape[0]
    return np.asarray(full)[:, :nside // 2 + 1]


def independent_modes(nside):
    """(mask, weights) selecting ONE representative per conjugate pair on the rfft half plane.

    The rfft half plane is not a set of independent modes: in its two self-conjugate
    columns (ky = 0 and ky = Nyquist) the entries (i, j) and (nside - i, j) are complex
    conjugates of each other, and util.get_fourier_weights gives each of them weight 1 so
    that a SUM over the half plane counts the pair once with weight 2. For a sum that is
    all that matters, and it is why fisher_forecast's trace formula and its Gaussian
    bandpower formula agree. A COVARIANCE of the per-mode power estimates cannot carry a
    duplicated entry - the two copies are perfectly correlated and the matrix is singular -
    so the estimator vector here lives on the set returned by this function:

        every entry of the bulk columns 0 < j < nside // 2         weight 2 (complex mode)
        rows 0 < i < nside // 2 of the columns j = 0, nside // 2    weight 2 (complex mode;
                                                                   the partner row is dropped)
        the four real modes (0, 0), (0, N/2), (N/2, 0), (N/2, N/2)  weight 1 (one real DOF)

    The weights sum to nside^2 exactly as get_fourier_weights' do, so every per-mode sum
    is unchanged; only the duplicates are gone. The origin is in the mask and is removed
    by the positivity of the fiducial block, as everywhere else in fisher_forecast.
    """
    half = nside // 2
    mask = np.ones((nside, half + 1), dtype = bool)
    weights = np.full((nside, half + 1), 2.0)
    for j in (0, half):
        mask[half + 1:, j] = False
        weights[half + 1:, j] = 0.0
        weights[0, j] = 1.0
        weights[half, j] = 1.0
    return mask, weights


def negate_modes(full):
    """x(-l) for a full-plane array: index i -> (-i) mod nside on both axes."""
    nside = full.shape[0]
    rows = (-np.arange(nside)) % nside
    return full[np.ix_(rows, rows)]


def independent_to_full(half):
    """Even extension to the full plane of an array holding values ONLY on the independent
    modes of independent_modes (zeros on the dropped duplicates).

    Each conjugate pair {L, -L} ends up holding the same value at both positions; the four
    self-conjugate real modes, which the negation maps onto themselves, are halved after
    the fold so they hold their value once rather than twice.
    """
    half = np.asarray(half, dtype = np.float64)
    nside = half.shape[0]
    n = nside // 2
    full = np.zeros((nside, nside))
    full[:, :n + 1] = half
    full = full + negate_modes(full)
    for i, j in ((0, 0), (0, n), (n, 0), (n, n)):
        full[i, j] *= 0.5
    return full


class LensingKernel:
    """K(l, L) = dC~_TT(l) / dC_phi(L) at first order, applied by circular FFT convolution.

    `cf_unlensed_half` is the unlensed TT covariance on the rfft half plane in the
    codebase's units (covar_matrix_from_cls output); the origin must already be zero.
    apply / apply_transpose act on FULL-plane real arrays and return full-plane arrays.

    THE NYQUIST SYMMETRIZATION. A covariance of power estimates only ever sees the part of
    the kernel that is even under negating BOTH wavevectors, K(-l, -L) = K(l, L); the
    continuum formula has that symmetry exactly. On the periodic grid it does not: the
    Nyquist mode is its own negative, so its wavevector keeps its sign when every other
    component flips, and [(l - L).L]^2 changes whenever L, l or their difference carries a
    Nyquist component. That is the same ambiguity util.get_k_meshgrid resolves by fixing
    the Nyquist sign by hand for the derivative operators. Here the kernel is symmetrized
    explicitly - averaged with itself evaluated in the opposite Nyquist convention - which
    is exact for the covariance and only touches Nyquist-coupled entries, where the box
    carries no signal anyway. Without it the operator is not symmetric and CG fails.
    """

    def __init__(self, cf_unlensed_half, nside, pix_width):
        self.nside = nside
        self.kx, self.ky = full_wavevectors(nside, pix_width)
        self.cf = half_to_full(cf_unlensed_half)
        self.l_squared = self.kx**2 + self.ky**2
        #the three quadratic weights the polynomial [(l - L).L]^2 splits into
        self.pairs = [(self.kx, self.kx, 1.0), (self.kx, self.ky, 2.0), (self.ky, self.ky, 1.0)]
        #h_ab(l') = l'_a l'_b C(l'), Fourier transformed once for every convolution to come
        self.h_fft = [np.fft.fft2(a * b * self.cf) for a, b, _ in self.pairs]
        #the rank-one "1 - l^2 R" piece: l^2 C(l) on the left, L^2 / 2 on the right
        self.l2c = self.l_squared * self.cf

    def _apply_raw(self, v):
        total = np.zeros((self.nside, self.nside))
        for (a, b, factor), h_fft in zip(self.pairs, self.h_fft):
            q_fft = np.fft.fft2(a * b * v)
            total += factor * np.real(np.fft.ifft2(h_fft * q_fft))
        total -= 0.5 * self.l2c * np.sum(self.l_squared * v)
        return total / self.nside**2

    def _apply_transpose_raw(self, g):
        g_fft = np.fft.fft2(g)
        total = np.zeros((self.nside, self.nside))
        for (a, b, factor), h_fft in zip(self.pairs, self.h_fft):
            #sum_l g(l) h(l - L) is the correlation of g with h, ifft(fft(g) conj(fft(h)))
            total += factor * a * b * np.real(np.fft.ifft2(g_fft * np.conj(h_fft)))
        total -= 0.5 * self.l_squared * np.sum(self.l2c * g)
        return total / self.nside**2

    def apply(self, v):
        """y(l) = sum_L K(l, L) v(L), full-plane sum, with K symmetrized under joint
        negation: (K + P K P) / 2 with P the mode-negation permutation."""
        v = np.asarray(v, dtype = np.float64)
        return 0.5 * (self._apply_raw(v) + negate_modes(self._apply_raw(negate_modes(v))))

    def apply_transpose(self, g):
        """u(L) = sum_l K(l, L) g(l), full-plane sum, same symmetrization."""
        g = np.asarray(g, dtype = np.float64)
        return 0.5 * (self._apply_transpose_raw(g)
                      + negate_modes(self._apply_transpose_raw(negate_modes(g))))

    def dense(self):
        """The full (nside^2 x nside^2) symmetrized kernel matrix by brute force - tests only.

        K[l, L] with both indices flattened over the full plane, l - L taken with circular
        wrap, i.e. read from the grid mode the difference aliases onto, then averaged with
        its joint-negation image exactly as apply / apply_transpose do.
        """
        n = self.nside
        idx = np.arange(n)
        matrix = np.zeros((n * n, n * n))
        for i in range(n):
            for j in range(n):
                #l - L for every L at once, as wrapped grid indices
                di = (i - idx[:, None]) % n
                dj = (j - idx[None, :]) % n
                l_minus_L_x = self.kx[di, dj]
                l_minus_L_y = self.ky[di, dj]
                c_diff = self.cf[di, dj]
                dot = l_minus_L_x * self.kx + l_minus_L_y * self.ky
                row = dot**2 * c_diff - 0.5 * self.l_squared[i, j] * self.l_squared * self.cf[i, j]
                matrix[i * n + j] = row.ravel()
        matrix /= n**2
        negated = ((-idx) % n)[:, None] * n + ((-idx) % n)[None, :]
        permutation = negated.ravel()
        return 0.5 * (matrix + matrix[np.ix_(permutation, permutation)])


def first_order_lensing_check(kernel, cphi_half, cf_half, cfl_half, ell_grid, weights,
                              edges = (100, 300, 600, 1000, 1500, 2000, 3000)):
    """Ratio of the kernel's first-order lensing correction to CAMB's, per annulus.

    Returns a list of (ell_low, ell_high, ratio). The ratio is the DOF-weighted mean of
    K C_phi over the annulus divided by the same mean of C_lensed - C_unlensed, both on
    the grid. Near 1 where first-order lensing holds; the drift above ell ~ 1500 is the
    known breakdown of the first-order expansion, not a bug in the kernel.
    """
    correction = full_to_half(kernel.apply(half_to_full(cphi_half)))
    camb = np.asarray(cfl_half) - np.asarray(cf_half)
    ells = np.asarray(ell_grid)
    dof = np.asarray(weights, dtype = np.float64)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (ells >= low) & (ells < high)
        if not mask.any():
            continue
        ratio = np.sum(dof[mask] * correction[mask]) / np.sum(dof[mask] * camb[mask])
        rows.append((low, high, float(ratio)))
    return rows


class LensingBandpowerCovariance:
    """Cov(mu_hat) for the stacked per-mode (TT, phiphi) power estimates, matrix-free.

    Args:
        tt_fid, pp_fid:   the fiducial TT and phiphi blocks on the rfft half plane, with
                          their noise (covariance_blocks output, codebase units)
        cf_unlensed:      unlensed TT covariance on the half plane (the kernel's C)
        cphi_eff:         the phi power whose sample variance leaks into TT: C_phi for
                          "lensed", C_phi N_phi / (C_phi + N_phi) for "delensed"
        cross:            include the TT-phiphi cross term (True for "lensed")
        nside, pix_width: the box
        non_gaussian:     False gives the plain Gaussian diagonal (the "blocks" identity)

    Vectors are laid out as [TT modes with positive power, then phiphi modes with positive
    power], over the INDEPENDENT modes of independent_modes in row-major order - one entry
    per conjugate pair, which is what a covariance of power estimates requires.
    """

    def __init__(self, tt_fid, pp_fid, cf_unlensed, cphi_eff, cross, nside, pix_width,
                 non_gaussian = True):
        self.nside = nside
        self.tt_fid = np.asarray(tt_fid, dtype = np.float64)
        self.pp_fid = np.asarray(pp_fid, dtype = np.float64)
        independent, self.weights = independent_modes(nside)
        self.good_tt = (self.tt_fid > 0) & independent
        self.good_pp = (self.pp_fid > 0) & independent
        self.n_tt = int(self.good_tt.sum())
        self.n_pp = int(self.good_pp.sum())
        self.size = self.n_tt + self.n_pp
        self.non_gaussian = non_gaussian
        self.cross = cross

        #Gaussian diagonal: Var(mu_k) = 2 C_k^2 / w_k
        self.diag_tt = 2 * self.tt_fid[self.good_tt]**2 / self.weights[self.good_tt]
        self.diag_pp = 2 * self.pp_fid[self.good_pp]**2 / self.weights[self.good_pp]
        self.diagonal = np.concatenate([self.diag_tt, self.diag_pp])

        if non_gaussian:
            self.kernel = LensingKernel(cf_unlensed, nside, pix_width)
            self.cphi_eff_squared_full = half_to_full(np.asarray(cphi_eff, dtype = np.float64))**2

    #── half-plane vectors <-> full-plane arrays ──────────────────────────
    def _tt_to_full(self, x):
        half = np.zeros(self.tt_fid.shape)
        half[self.good_tt] = x / self.weights[self.good_tt]
        return independent_to_full(half)

    def _pp_to_full(self, x):
        half = np.zeros(self.pp_fid.shape)
        half[self.good_pp] = x / self.weights[self.good_pp]
        return independent_to_full(half)

    def _full_to_tt(self, full):
        return full_to_half(full)[self.good_tt]

    def _full_to_pp(self, full):
        return full_to_half(full)[self.good_pp]

    #── the operator ─────────────────────────────────────────────────────
    def matvec(self, x):
        x = np.asarray(x, dtype = np.float64)
        x_tt, x_pp = x[:self.n_tt], x[self.n_tt:]
        y_tt = self.diag_tt * x_tt
        y_pp = self.diag_pp * x_pp
        if not self.non_gaussian:
            return np.concatenate([y_tt, y_pp])

        #u(L) = sum_l K(l, L) x~_tt(l) is needed by both the TT-TT term and the cross term
        u = self.kernel.apply_transpose(self._tt_to_full(x_tt))
        #TT-TT: 2 K diag(C_eff^2) K^T
        y_tt = y_tt + 2 * self._full_to_tt(self.kernel.apply(self.cphi_eff_squared_full * u))
        if self.cross:
            #TT <- phiphi: 2 K diag(C_eff^2) x_pp ; phiphi <- TT: 2 diag(C_eff^2) K^T x_tt
            y_tt = y_tt + 2 * self._full_to_tt(
                self.kernel.apply(self.cphi_eff_squared_full * self._pp_to_full(x_pp)))
            y_pp = y_pp + 2 * self._full_to_pp(self.cphi_eff_squared_full * u)
        return np.concatenate([y_tt, y_pp])

    def as_linear_operator(self):
        return LinearOperator((self.size, self.size), matvec = self.matvec,
                              dtype = np.float64)

    def stack(self, tt_half, pp_half):
        """Pack a pair of half-plane arrays into the operator's vector layout."""
        return np.concatenate([np.asarray(tt_half)[self.good_tt],
                               np.asarray(pp_half)[self.good_pp]])

    def solve(self, rhs, rtol = 1e-10, maxiter = 5000):
        """x = Cov^-1 rhs by conjugate gradients, preconditioned by the Gaussian diagonal."""
        if not self.non_gaussian:
            return rhs / self.diagonal
        preconditioner = LinearOperator((self.size, self.size),
                                        matvec = lambda v: v / self.diagonal,
                                        dtype = np.float64)
        x, info = cg(self.as_linear_operator(), rhs, M = preconditioner, rtol = rtol,
                     maxiter = maxiter)
        if info != 0:
            raise RuntimeError(f"conjugate gradients did not converge (info = {info}) "
                               f"solving the non-Gaussian bandpower covariance; the "
                               f"operator should be positive definite, so check the "
                               f"fiducial blocks for non-positive modes")
        return x

    def dense(self):
        """The explicit covariance matrix by applying matvec to every unit vector - tests
        only, and only sensible for tiny grids."""
        columns = [self.matvec(np.eye(self.size)[:, i]) for i in range(self.size)]
        return np.stack(columns, axis = 1)


def fisher_from_bandpower_covariance(covariance, derivatives):
    """F_ij = dmu_i . Cov^-1 dmu_j for a list of stacked derivative vectors.

    Returns (fisher, asymmetry): the matrix is symmetrized, and the largest relative
    asymmetry before symmetrization is reported as a check on the CG tolerance - it must
    be tiny, since the exact contraction is symmetric.
    """
    n = len(derivatives)
    solved = [covariance.solve(d) for d in derivatives]
    raw = np.array([[float(np.dot(derivatives[i], solved[j])) for j in range(n)]
                    for i in range(n)])
    scale = np.sqrt(np.outer(np.abs(np.diag(raw)), np.abs(np.diag(raw))))
    asymmetry = float(np.max(np.abs(raw - raw.T) / scale)) if n > 1 else 0.0
    return 0.5 * (raw + raw.T), asymmetry
