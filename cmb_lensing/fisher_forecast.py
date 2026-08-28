"""Gaussian Fisher forecast for the LCDM parameters on sample_lcdm.py's flat-sky box.

This answers "what is the best a power-spectrum analysis of this data set could do?" -
the baseline that sample_joint should beat, since the Gibbs sampler works with the full
hierarchical likelihood (phi ~ N(0, Cphi), f ~ N(0, Cf), d = M B L(phi) f + n) and
therefore also sees the lensing-induced non-Gaussianity that a 2-point analysis discards.

The forecast is computed on the SAME flat-sky rfft grid the sampler runs on (gen_ell_grid
at the given nside / theta_pix), not with a full-sky (2l+1) sum, so the box geometry, the
finite ell_min = 2*pi/L and the mode count all match the real run.

    F_ij = 1/2 * sum_k w_k * Tr[C_k^-1 dC_k/dtheta_i C_k^-1 dC_k/dtheta_j]

which for the temperature-only case used here collapses to

    F_ij = 1/2 * sum_k w_k * dln(C_k)/dtheta_i * dln(C_k)/dtheta_j

This is the standard zero-mean Gaussian Fisher F = 1/2 Tr[C^-1 C,i C^-1 C,j] (see e.g.
Tegmark, Taylor & Heavens 1997 for the general form), specialized to a covariance that is
diagonal in the Fourier basis. The sum-over-rfft-entries form above is NOT quoted from
anywhere - the literature writes either the bare trace or the full-sky f_sky (2l+1) / 2
sum; it is a transcription of the same result onto this codebase's array layout, and it
is derived from scratch in the "Derivation" section below. The weights w_k are
util.get_fourier_weights - the number of INDEPENDENT REAL degrees of freedom carried by
each rfft entry (2 for bulk columns, 1 for the two self-conjugate columns kx = 0 and
kx = Nyquist, whose entries are constrained by F[-n] = conj(F[n]) within the column).
They sum to nside**2, the true DOF count of the real map, and they are the same weights
util.primal_dot / util.primal_log_det use to build statistics.logpdf - so this Fisher is
the second moment of the derivative of the very likelihood the codebase evaluates.

Derivation (scalar case, in this codebase's normalization). statistics.logpdf is built
from primal_dot and primal_log_det, i.e. schematically

    log L = -1/2 sum_k w_k [ |x_k|^2 / (nside^2 C_k) + ln C_k ] + const

with E|x_k|^2 = nside^2 C_k (see the field_from_covar_single_key comment). Then
E[d log L / dtheta_i] = 0, and with Var(|x_k|^2) = (nside^2 C_k)^2 * (2 / w_k) - variance
v^2 for a free complex mode, 2 v^2 for a real one - the outer product E[d_i log L d_j log L]
collapses to

    F_ij = 1/4 sum_k w_k^2 (2 / w_k) dln(C_k)/dtheta_i dln(C_k)/dtheta_j
         = 1/2 sum_k w_k dln(C_k)/dtheta_i dln(C_k)/dtheta_j

VALIDATION. Unlensed scalar TT is exactly proportional to As = exp(logA) * 1e-10, so in
"ceiling" mode dln(C)/dlogA is identically 1 and the f block must return exactly
1/2 * sum_k w_k = (nside^2 - 1) / 2 (the [0, 0] origin is excluded). At nside 64 that is
2047.5; the code returns 2047.63. That single number pins the weights, the factor of 1/2,
the origin handling and the finite difference simultaneously - re-run it after any change
to this module.

Three spectra modes, bracketing the answer:

  "lensed"   (DEFAULT) C = beam^2 * Cl_lensed + N. The achievable baseline: the Fisher of
             a surrogate Gaussian model matched to the second moment of the observed map.
             The data is NOT actually a GRF draw of Cl_lensed (load_sim draws unlensed f
             and phi and lenses them), so this surrogate is wrong above second order - it
             discards the trispectrum, i.e. all lensing-reconstruction information. That
             omission is exactly the gap sample_joint is meant to close.
  "unlensed" C = beam^2 * Cl_unlensed + N. The perfect-delensing heuristic. NOT a
             Cramer-Rao bound on the real data - no estimator acting on d is guaranteed
             to reach it, and delensing correlates the noise (L^-1 n is not isotropic).
             Quote it as a marker, not a bound.
  "ceiling"  Complete-data Fisher: f and phi known exactly. Because noise_cls / mask /
             beam carry no cosmology, log p(d | f, phi) is theta-INDEPENDENT, so given the
             fields the data says nothing about theta and all the information sits in the
             two priors. Hence NOISELESS unlensed Cl plus a Cphi term:
                 F = F[Cf_unlensed] + F[Cphi]
             This is a loose ceiling (noiseless mode counting), not a forecast. Its value
             is as a validation target: it is the ensemble average of Louis's first term,
             so a Monte-Carlo estimate of -<d2/dtheta2 log p(f, phi | theta)> over prior
             draws must reproduce it.

WHAT THIS IS NOT. None of these is the marginal Fisher of p(d | theta) = the integral of
p(d, f, phi | theta) over the fields, which is what sample_joint actually targets. That
one cannot be evaluated pointwise; it needs Louis's identity

    I_marg = E[-d2 log p(d, x | theta)] - Cov[d log p(d, x | theta)]

with BOTH expectations over the conditional p(f, phi | d, theta) - i.e. over draws from
the Gibbs sampler at fixed theta, NOT over prior draws. Prior draws give "ceiling" above.
The ratio F_ceiling / F_lensed printed by __main__ is a useful advance warning of how
severe the cancellation in that subtraction will be.

Finite differences. Cl derivatives are central differences with step
h_p = FD_STEP_FRAC[p] * PARAM_SIGMA[p], and the per-parameter constants in FD_STEP_FRAC
are the tuning knob. The Cls come from DIRECT CAMB calls, never from the 5D grid spline:
the spline's ~1e-3 lnCl interpolation error is a DETERMINISTIC function of theta, so it
does not average away, and differencing it over the grid's coarse node spacing can
produce curvature comparable to the signal. Only 2 * n_sampled + 1 CAMB runs are needed.
Always confirm the answer is step-independent - "--stability" recomputes at 2h and reports
the fractional change in the forecast sigmas.

Parameters with is_sampled[name] = False are held FIXED, not marginalized over; F is
n_sampled x n_sampled.

Usage:
    python -m cmb_lensing.fisher_forecast                       #nside 64, T-only, 5 uK-arcmin
    python -m cmb_lensing.fisher_forecast --spectra ceiling
    python -m cmb_lensing.fisher_forecast --nside 128 --noise 1.0 --stability

Writes cmb_lensing/fisher_output/{fisher_matrix.png, covariance_matrix.png, fisher.npz}.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cmb_lensing.util import gen_ell_grid, get_fourier_weights
from cmb_lensing.simulate import (_camb_via_callback, _extract_all_cls,
                                  covar_matrix_from_cls, noise_cls, get_beam, get_mask)
from cmb_lensing.constants import DEFAULT_MAX_ELL, DEFAULT_A_LENSE, DEFAULT_K_PIVOT, DEFAULT_MNU, DEFAULT_TAUREIO
from cmb_lensing.precompute_camb_1d import (PARAM_ORDER, GROUND_TRUTH, PARAM_SIGMA,
                                            CAMB_LMAX)

#finite-difference step for each parameter, as a FRACTION of PARAM_SIGMA. these are the
#tuning knobs: raise them if the forecast is noisy (CAMB's own ~1e-3 lnCl accuracy floor
#leaking into the difference), lower them if it drifts with step size (genuine curvature
#of Cl(theta) breaking the linear approximation). always verify with --stability
FD_STEP_FRAC = {
    "theta_MC_100": 0.05,
    "logA": 0.05,
    "ns": 0.05,
    "ombh2": 0.05,
    "omch2": 0.05,
}

SPECTRA_MODES = ("lensed", "unlensed", "ceiling")

#CAMB is deterministic in the parameters, and the stencil re-queries the fiducial point
#for every parameter, so memoize whole cls dicts across calls
_CAMB_CACHE = {}


def output_dir():
    #mirrors sample_lcdm.py's sample_lcdm_output convention: a directory next to the
    #package modules, so the same relative layout works on the laptop and the cluster
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisher_output")


# ── CAMB ──────────────────────────────────────────────────────────────────

def camb_cls_at_params(params):
    """Full cls dict from one CAMB run at an arbitrary 5-parameter point.

    precompute_camb_1d.camb_cls_at only moves a single parameter off GROUND_TRUTH and
    only returns (TT, PP); the Fisher stencil needs an arbitrary point and the lensed
    spectra too. Fixed (non-sampled) CAMB parameters match load_sim's defaults:
    H0 = None (solved from cosmomc_theta), r = 0, mnu = 0.06, tau = 0.05, nt = 0,
    k_pivot = 0.05, Alens = 1. Cls land on ells 2..CAMB_LMAX-1, pure CAMB values with no
    extrapolation - the same support as load_sim's data-map path.
    """
    key = tuple(round(float(params[name]), 12) for name in PARAM_ORDER)
    if key in _CAMB_CACHE:
        return _CAMB_CACHE[key]

    cosmomc_theta = params["theta_MC_100"] / 100
    As = np.exp(params["logA"]) * 1e-10
    unlensed_scalar, tensor, total, lens_potential = _camb_via_callback(
        None, params["ombh2"], params["omch2"], cosmomc_theta, 0.0, DEFAULT_MNU, DEFAULT_TAUREIO,
        As, 0, params["ns"], CAMB_LMAX, DEFAULT_K_PIVOT, DEFAULT_A_LENSE
    )
    cls = _extract_all_cls(unlensed_scalar, tensor, total, lens_potential,
                           CAMB_LMAX, CAMB_LMAX)

    #_camb_callback_fn swallows CAMB failures and returns NaN arrays (so a sampler
    #proposal rejects instead of crashing). here that silently poisons the whole matrix,
    #so fail loudly with the offending point
    if not bool(jnp.all(jnp.isfinite(cls["total_TT"]))):
        raise RuntimeError(
            f"CAMB returned non-finite Cls at {dict((k, float(params[k])) for k in PARAM_ORDER)}. "
            f"theta_MC_100 has no H0 < 100 solution above ~1.117 - check the stencil "
            f"point is reachable, or shrink FD_STEP_FRAC for that parameter."
        )

    _CAMB_CACHE[key] = cls
    return cls


# ── Covariance blocks on the flat-sky grid ────────────────────────────────

def covariance_blocks(cls, spectra, nside, pix_width, ell_grid,
                      noise_level, l_knee, beam_fwhm, l_cutoff):
    """The theta-dependent covariance block(s) whose Fisher information we are counting.

    Every block is built through the same covar_matrix_from_cls the sampler uses, so the
    signal and noise share a normalization (the 1/pix_width**2 rescale). The trace formula
    is invariant under any theta-independent rescaling of C, so only that relative
    normalization matters.
    """
    ells = jnp.arange(2, 2 + cls["total_TT"].shape[0]).astype(jnp.float64)
    phi_ells = jnp.arange(2, 2 + cls["phi"].shape[0]).astype(jnp.float64)

    def covar(cl, ell_axis):
        return covar_matrix_from_cls(nside, pix_width, ell_grid, ell_axis, cl,
                                     origin_value = 0)

    if spectra == "ceiling":
        #conditional on (f, phi) the data term carries no theta dependence at all, so the
        #complete-data information is the two priors, noiseless, with no beam or mask
        return {"f_unlensed": covar(cls["scalar_TT"], ells),
                "phi": covar(cls["phi"], phi_ells)}

    lmax_prime = min(DEFAULT_MAX_ELL, CAMB_LMAX)
    n_tt, _, _, _ = noise_cls(lmax_prime, noise_level, beam_fwhm = beam_fwhm,
                              l_knee = l_knee)
    noise = covar(n_tt, jnp.arange(2, lmax_prime).astype(jnp.float64))
    #load_sim forms data = mask * beam * lensed + noise with mask identically one, so the
    #observed covariance carries beam**2 on the signal only
    beam = get_beam(nside, pix_width, ell_grid, lmax_prime, beam_fwhm = beam_fwhm)
    mask = get_mask(l_cutoff, nside, pix_width, ell_grid)

    source = "total_TT" if spectra == "lensed" else "scalar_TT"
    return {"TT": (mask * beam)**2 * covar(cls[source], ells) + noise}



def _fisher_from_blocks(blocks_plus, blocks_minus, blocks_fid, steps, weights):
    """F_ij = 1/2 sum_blocks sum_k w_k dln(C)/dtheta_i dln(C)/dtheta_j.

    blocks_plus/minus are lists (one entry per sampled parameter) of block dicts at
    theta +/- h; blocks_fid is the block dict at the fiducial point.
    """
    n_param = len(steps)
    fisher = np.zeros((n_param, n_param))

    for name in blocks_fid:
        c_fid = blocks_fid[name]
        #the [0, 0] origin is set to zero by construction (origin_value = 0) and carries
        #no information; guard the division rather than letting it produce NaN
        good = c_fid > 0
        safe = jnp.where(good, c_fid, 1.0)

        dlog = []
        for i in range(n_param):
            derivative = (blocks_plus[i][name] - blocks_minus[i][name]) / (2 * steps[i])
            dlog.append(jnp.where(good, derivative / safe, 0.0))

        for i in range(n_param):
            for j in range(i, n_param):
                value = 0.5 * float(jnp.sum(weights * dlog[i] * dlog[j]))
                fisher[i, j] += value
                if i != j:
                    fisher[j, i] += value

    return fisher


# ── Main entry point ──────────────────────────────────────────────────────

def forecast(nside, theta_pix, noise_level, is_sampled, param_ground,
             spectra = "lensed", step_fracs = None, l_knee = 0, beam_fwhm = 0,
             l_cutoff = 10_000, verbose = True):
    """Gaussian Fisher matrix for the sampled LCDM parameters on an nside x nside box.

    Args:
        nside:        pixels per side (the sampler's nside)
        theta_pix:    pixel width in arcmin
        noise_level:  white-noise level in uK-arcmin
        is_sampled:   {param_name: bool}. False parameters are held FIXED (not
                      marginalized), matching the sampler's should_sample semantics
        param_ground: {param_name: float} fiducial point. All five must be present,
                      since the four unsampled ones still set the cosmology
        spectra:      "lensed" (default), "unlensed" or "ceiling" - see the module
                      docstring for what each one does and does not bound
        step_fracs:   override for FD_STEP_FRAC (fractions of PARAM_SIGMA)
        l_knee:       1/f noise knee. Defaults to 0 (pure white noise), matching every
                      sampler entry point; load_sim's own default of 100 does not apply
        beam_fwhm:    beam FWHM in arcmin, default 0 as in load_sim

    Returns:
        (fisher, names) - the n_sampled x n_sampled matrix and the parameter names in
        PARAM_ORDER order.
    """
    if spectra not in SPECTRA_MODES:
        raise ValueError(f"spectra must be one of {SPECTRA_MODES}, got {spectra!r}")

    missing = [name for name in PARAM_ORDER if name not in param_ground]
    if missing:
        raise ValueError(f"param_ground is missing {missing}; all five of {PARAM_ORDER} "
                         f"are needed because the unsampled ones still fix the cosmology")

    names = [name for name in PARAM_ORDER if is_sampled.get(name, False)]
    if not names:
        raise ValueError("is_sampled selects no parameters - nothing to forecast")

    fracs = dict(FD_STEP_FRAC)
    if step_fracs is not None:
        fracs.update(step_fracs)
    steps = [fracs[name] * PARAM_SIGMA[name] for name in names]

    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    #w_k = independent real DOF per rfft entry; get_fourier_weights indexes the half-axis
    #(columns), so it broadcasts across rows
    weights = jnp.broadcast_to(jnp.real(get_fourier_weights((nside, nside // 2 + 1))),
                               (nside, nside // 2 + 1))

    def blocks_at(params):
        return covariance_blocks(camb_cls_at_params(params), spectra, nside,
                                 pix_width, ell_grid, noise_level, l_knee, 
                                 beam_fwhm, l_cutoff)

    if verbose:
        ells_on_grid = ell_grid[ell_grid > 0]
        print(f"Fisher forecast [{spectra}]: nside {nside}, theta_pix {theta_pix}', "
              f"{noise_level} uK-arcmin, l_knee {l_knee}")
        print(f"  box {nside * theta_pix / 60:.2f} deg, ell in "
              f"[{float(jnp.min(ells_on_grid)):.0f}, {float(jnp.max(ells_on_grid)):.0f}], "
              f"{int(jnp.sum(weights))} real DOF")
        print(f"  sampled: {names}")
        print(f"  running {2 * len(names) + 1} CAMB calls...")

    blocks_fid = blocks_at(param_ground)

    blocks_plus, blocks_minus = [], []
    for name, step in zip(names, steps):
        up = dict(param_ground)
        up[name] = param_ground[name] + step
        down = dict(param_ground)
        down[name] = param_ground[name] - step
        blocks_plus.append(blocks_at(up))
        blocks_minus.append(blocks_at(down))
        if verbose:
            print(f"    {name}: h = {step:.6g} ({fracs[name]:g} sigma)")

    fisher = _fisher_from_blocks(blocks_plus, blocks_minus, blocks_fid, steps, weights)
    return fisher, names


def covariance_from_fisher(fisher, names):
    """Invert the Fisher matrix, failing loudly on a singular / non-PSD result."""
    eigenvalues = np.linalg.eigvalsh(fisher)
    if np.any(eigenvalues <= 0):
        raise RuntimeError(
            f"Fisher matrix is not positive definite (eigenvalues {eigenvalues}). "
            f"Usually a degenerate parameter pair at this box size, or a finite-difference "
            f"step that is too small - try raising FD_STEP_FRAC and re-running --stability."
        )
    condition = eigenvalues[-1] / eigenvalues[0]
    if condition > 1e12:
        print(f"WARNING: Fisher condition number {condition:.3g} - the inverse is "
              f"numerically unreliable, so at least one direction is near-degenerate "
              f"over {names}")
    return np.linalg.inv(fisher)


def step_stability(nside, theta_pix, noise_level, is_sampled, param_ground,
                   spectra = "lensed", factor = 2.0, **kwargs):
    """Recompute the forecast at `factor` x the step size and report the drift.

    A well-converged Fisher is step-independent. Drift means either the step is too large
    (real curvature in Cl(theta)) or too small (CAMB's own accuracy floor amplified by
    1/h). Second derivatives would amplify this by 1/h**2; here the log-derivative is
    first order, so a few percent drift is already worth chasing.
    """
    base = {name: FD_STEP_FRAC[name] for name in PARAM_ORDER}
    doubled = {name: factor * value for name, value in base.items()}

    fisher_a, names = forecast(nside, theta_pix, noise_level, is_sampled, param_ground,
                               spectra = spectra, step_fracs = base, **kwargs)
    fisher_b, _ = forecast(nside, theta_pix, noise_level, is_sampled, param_ground,
                           spectra = spectra, step_fracs = doubled, verbose = False,
                           **{k: v for k, v in kwargs.items() if k != "verbose"})

    sigma_a = np.sqrt(np.diag(covariance_from_fisher(fisher_a, names)))
    sigma_b = np.sqrt(np.diag(covariance_from_fisher(fisher_b, names)))
    drift = np.abs(sigma_b / sigma_a - 1)

    print(f"\nStep stability (h vs {factor:g}h), fractional change in forecast sigma:")
    for name, value in zip(names, drift):
        flag = "  <-- unstable" if value > 0.01 else ""
        print(f"  {name:<14s} {value:.2%}{flag}")
    if np.max(drift) > 0.01:
        print("  tune FD_STEP_FRAC until every entry is well under 1%")
    return drift


# ── Plotting ──────────────────────────────────────────────────────────────

def _annotated_heatmap(axis, matrix, names, title, colorbar_label):
    """Heatmap coloured by the correlation-normalized matrix (bounded, readable) with the
    raw entries annotated - the raw values span many orders of magnitude across
    parameters with wildly different units, so colouring by them directly is useless."""
    diagonal = np.sqrt(np.abs(np.diag(matrix)))
    normalized = matrix / np.outer(diagonal, diagonal)

    image = axis.imshow(normalized, cmap = "RdBu_r", vmin = -1, vmax = 1)
    axis.set_xticks(range(len(names)))
    axis.set_yticks(range(len(names)))
    axis.set_xticklabels(names, rotation = 45, ha = "right")
    axis.set_yticklabels(names)
    axis.set_title(title)

    for i in range(len(names)):
        for j in range(len(names)):
            shade = "white" if abs(normalized[i, j]) > 0.6 else "black"
            axis.text(j, i, f"{matrix[i, j]:.2e}", ha = "center", va = "center",
                      color = shade, fontsize = 7)

    colorbar = plt.colorbar(image, ax = axis, fraction = 0.046, pad = 0.04)
    colorbar.set_label(colorbar_label)
    return image


def plot_fisher_matrix(fisher, names, path, subtitle = ""):
    figure, axis = plt.subplots(figsize = (7.5, 6.5))
    _annotated_heatmap(axis, fisher, names,
                       "Fisher matrix $F_{ij}$" + (f"\n{subtitle}" if subtitle else ""),
                       r"$F_{ij}\,/\,\sqrt{F_{ii}F_{jj}}$")
    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


def plot_covariance_matrix(covariance, names, path, subtitle = ""):
    sigmas = np.sqrt(np.diag(covariance))
    ratios = np.array([sigmas[i] / PARAM_SIGMA[name] for i, name in enumerate(names)])

    figure, (left, right) = plt.subplots(
        1, 2, figsize = (12.5, 6.0), gridspec_kw = {"width_ratios": [1.35, 1]})

    _annotated_heatmap(left, covariance, names,
                       "Covariance $F^{-1}$" + (f"\n{subtitle}" if subtitle else ""),
                       "correlation coefficient")

    positions = np.arange(len(names))
    right.barh(positions, ratios, color = "#4C72B0")
    right.axvline(1.0, color = "k", linestyle = "--", linewidth = 1)
    right.set_yticks(positions)
    right.set_yticklabels(names)
    right.invert_yaxis()
    right.set_xscale("log")
    #the default log locator crowds this narrow range with overlapping minor labels
    right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    right.xaxis.set_minor_formatter(ticker.NullFormatter())
    right.set_xlim(0.5 * float(np.min(ratios)), 3.0 * float(np.max(ratios)))
    right.set_xlabel(r"forecast $\sigma$ / PARAM_SIGMA")
    right.set_title("Marginalized errors\n(dashed = PARAM_SIGMA; < 1 means informative)")
    for position, ratio, sigma in zip(positions, ratios, sigmas):
        right.text(ratio, position, f"  {sigma:.3g}", va = "center", fontsize = 8)

    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    plt.close(figure)


# ── CLI ───────────────────────────────────────────────────────────────────

def _report(covariance, names, spectra):
    sigmas = np.sqrt(np.diag(covariance))
    print(f"\nForecast 1-sigma errors [{spectra}]:")
    print(f"  {'parameter':<14s} {'sigma':>12s} {'sigma/PARAM_SIGMA':>20s}")
    for i, name in enumerate(names):
        print(f"  {name:<14s} {sigmas[i]:>12.4g} {sigmas[i] / PARAM_SIGMA[name]:>20.3g}")

    if len(names) > 1:
        correlation = covariance / np.outer(sigmas, sigmas)
        pairs = [(abs(correlation[i, j]), correlation[i, j], names[i], names[j])
                 for i in range(len(names)) for j in range(i + 1, len(names))]
        _, value, first, second = max(pairs)
        print(f"  strongest degeneracy: {first} - {second}  (r = {value:+.2f})")


def main():
    parser = argparse.ArgumentParser(
        description = "Gaussian Fisher forecast for the LCDM parameters on the "
                      "sample_lcdm.py flat-sky box")
    parser.add_argument("--nside", type = int, default = 64)
    parser.add_argument("--theta_pix", type = float, default = 5.0,
                        help = "pixel width in arcmin")
    parser.add_argument("--noise", type = float, default = 5.0,
                        help = "white noise level in uK-arcmin")
    parser.add_argument("--l_knee", type = float, default = 0.0,
                        help = "1/f knee; 0 (default) matches every sampler entry point")
    parser.add_argument("--beam_fwhm", type = float, default = 0.0)
    parser.add_argument("--spectra", choices = SPECTRA_MODES, default = "ceiling")
    parser.add_argument("--params", nargs = "*", default = None,
                        help = f"subset to sample; default all of {PARAM_ORDER}")
    parser.add_argument("--stability", action = "store_true",
                        help = "also recompute at 2x the step size and report the drift")
    args = parser.parse_args()

    is_sampled = {name: (args.params is None or name in args.params)
                  for name in PARAM_ORDER}
    if args.params is not None:
        unknown = [name for name in args.params if name not in PARAM_ORDER]
        if unknown:
            parser.error(f"unknown parameters {unknown}; choose from {PARAM_ORDER}")

    fisher, names = forecast(args.nside, args.theta_pix, args.noise, is_sampled,
                             GROUND_TRUTH, spectra = args.spectra,
                             l_knee = args.l_knee, beam_fwhm = args.beam_fwhm)
    covariance = covariance_from_fisher(fisher, names)
    _report(covariance, names, args.spectra)

    if args.spectra == "lensed":
        #the ceiling / baseline ratio sets how badly Louis's identity will cancel when
        #the marginal Fisher is estimated from Gibbs draws - worth knowing in advance
        ceiling, _ = forecast(args.nside, args.theta_pix, args.noise, is_sampled,
                              GROUND_TRUTH, spectra = "ceiling", l_knee = args.l_knee,
                              beam_fwhm = args.beam_fwhm, verbose = False)
        ratio = np.sqrt(np.diag(ceiling)) / np.sqrt(np.diag(fisher))
        print(f"\n  F_ceiling / F_lensed per parameter (diagonal, sqrt): "
              f"{np.array2string(ratio, precision = 1)}")
        print("  the marginal Fisher sits between these two; a large ratio means Louis's "
              "identity\n  will subtract two nearly-equal numbers and needs many Gibbs draws")

    if args.stability:
        step_stability(args.nside, args.theta_pix, args.noise, is_sampled, GROUND_TRUTH,
                       spectra = args.spectra, l_knee = args.l_knee,
                       beam_fwhm = args.beam_fwhm)

    directory = output_dir()
    os.makedirs(directory, exist_ok = True)
    subtitle = (f"{args.spectra}  |  nside {args.nside}, {args.theta_pix:g}', "
                f"{args.noise:g} uK-arcmin")
    plot_fisher_matrix(fisher, names, os.path.join(directory, "fisher_matrix.png"),
                       subtitle)
    plot_covariance_matrix(covariance, names,
                           os.path.join(directory, "covariance_matrix.png"), subtitle)
    np.savez(os.path.join(directory, "fisher.npz"),
             fisher = fisher, covariance = covariance, names = np.array(names),
             sigmas = np.sqrt(np.diag(covariance)), spectra = args.spectra,
             nside = args.nside, theta_pix = args.theta_pix, noise_level = args.noise,
             l_knee = args.l_knee, beam_fwhm = args.beam_fwhm,
             step_fracs = np.array([FD_STEP_FRAC[name] for name in names]))
    print(f"\nWrote fisher_matrix.png, covariance_matrix.png and fisher.npz to {directory}")


if __name__ == "__main__":
    main()
