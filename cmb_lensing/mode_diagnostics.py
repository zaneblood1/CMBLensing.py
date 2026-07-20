#Step 0 debugging: per-mode mixing diagnostics for the phi HMC.
#
#The question this answers: when the chain refuses to forget its initialization, WHICH
#Fourier modes of phi are failing to decorrelate between Gibbs sweeps? A short per-mode
#autocorrelation time everywhere means phi is mixing (so a persistent theta bias is NOT
#a mass-matrix problem). A cluster of long-autocorrelation modes localizes exactly what
#a better mass matrix (or reparametrization) must fix, and where it sits in ell tells you
#whether it is the mask coupling (low-ell / mask scales) or the realization-dependent
#lensing curvature (broadband).
#
#Usage:
#  1. During the chain, record the (physical, unmixed) phi each sweep:
#         recorder = PhiModeRecorder()
#         ...inside the loop, after unmix...:  recorder.record(phi)
#  2. After the chain:
#         analyze_phi_modes(recorder.stack(), nside, pix_width, out_dir, burn_in = 50)
#
#sample_joint() in sample_lcdm_legacy.py is already wired to do this when
#record_phi_modes = True (it dumps the stack and calls analyze_phi_modes at the end).

import numpy as np
import matplotlib.pyplot as plt

from cmb_lensing.util import get_k_meshgrid

#------------------ recording ------------------------------------------------------------

class PhiModeRecorder:
    #lightweight collector: appends a host-side copy of each sweep's phi rfft2 array
    def __init__(self):
        self.frames = []

    def record(self, phi_field):
        #phi_field is a FlatS0-like object whose scalar_matrix is the rfft2 (N, N//2+1) array
        self.frames.append(np.asarray(phi_field.scalar_matrix))

    def stack(self):
        #returns (num_sweeps, N, N//2+1) complex array
        return np.stack(self.frames, axis = 0)


#------------------ per-mode autocorrelation ---------------------------------------------

def per_mode_acf(stack, max_lag = None):
    #normalized per-mode autocorrelation along the sweep axis.
    #stack: (T, ...) complex. returns (L, ...) real, with acf[0] == 1 for live modes and
    #nan for structurally-dead modes (masked-out / zero-variance).
    T = stack.shape[0]
    if max_lag is None:
        max_lag = T // 2
    max_lag = int(min(max_lag, T))

    w = stack - stack.mean(axis = 0, keepdims = True)

    #FFT-based linear autocovariance: zero-pad to >= 2T so the wrap-around is discarded
    n = 1
    while n < 2 * T:
        n *= 2
    W = np.fft.fft(w, n = n, axis = 0)
    acov = np.fft.ifft(np.abs(W) ** 2, axis = 0)[:T]

    #unbiased normalization by the number of overlapping sweeps at each lag
    counts = (T - np.arange(T)).reshape((T,) + (1,) * (w.ndim - 1))
    acov = acov / counts

    var = np.real(acov[0])
    live = var > (np.max(var) * 1e-12)

    acf = np.real(acov) / np.where(var == 0, 1.0, var)[None]
    acf = np.where(live[None], acf, np.nan)
    return acf[:max_lag]


def integrated_act(acf, c = 5.0):
    #integrated autocorrelation time per mode via Sokal's adaptive window.
    #convention: tau_int = 1 + 2*sum_{lag>=1} rho(lag); iid -> 1.
    #acf: (L, ...) with acf[0] == 1. returns (...,) with nan for dead modes.
    L = acf.shape[0]
    dead = np.isnan(acf[0])

    tail = np.nan_to_num(acf[1:], nan = 0.0)
    running = 1.0 + 2.0 * np.cumsum(tail, axis = 0)          #running[m-1] uses window M = m
    M = np.arange(1, L).reshape((L - 1,) + (1,) * (acf.ndim - 1))

    crossed = M >= (c * running)
    any_cross = crossed.any(axis = 0)
    first = np.argmax(crossed, axis = 0)                     #first True, or 0 if never
    tau = np.take_along_axis(running, first[None], axis = 0)[0]
    tau = np.where(any_cross, tau, running[-1])              #fall back to full window
    tau = np.maximum(tau, 1.0)
    return np.where(dead, np.nan, tau)


#------------------ radial (ell) binning -------------------------------------------------

def radial_profile(values, ell, nbins = 30, log_bins = True):
    #mean of a per-mode 2D map within concentric |ell| annuli, ignoring nan modes.
    #returns (bin_centers, bin_means, bin_counts).
    v = values.ravel()
    l = ell.ravel()
    good = np.isfinite(v) & (l > 0)
    v = v[good]
    l = l[good]

    if log_bins:
        edges = np.geomspace(l.min(), l.max(), nbins + 1)
    else:
        edges = np.linspace(l.min(), l.max(), nbins + 1)

    idx = np.clip(np.digitize(l, edges) - 1, 0, nbins - 1)
    means = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype = int)
    for b in range(nbins):
        sel = idx == b
        counts[b] = sel.sum()
        if counts[b] > 0:
            means[b] = np.mean(v[sel])
    centers = np.sqrt(edges[:-1] * edges[1:]) if log_bins else 0.5 * (edges[:-1] + edges[1:])
    return centers, means, counts


#------------------ top-level analysis ---------------------------------------------------

def analyze_phi_modes(stack, nside, pix_width, out_dir, burn_in = 0,
                      max_lag = None, num_traces = 6, tag = "phi"):
    #stack: (T, N, N//2+1) complex history of the (unmixed) phi rfft2 arrays.
    #writes plots + a printed summary that localize the stuck modes.
    stack = np.asarray(stack)
    if burn_in > 0:
        stack = stack[burn_in:]
    T = stack.shape[0]
    if T < 8:
        print(f"[mode_diagnostics] only {T} sweeps after burn-in; need more for a reliable ACF")

    KX, KY = (np.asarray(a) for a in get_k_meshgrid(nside, pix_width))
    ell = np.sqrt(KX ** 2 + KY ** 2)

    acf = per_mode_acf(stack, max_lag = max_lag)
    tau = integrated_act(acf)
    variance = stack.var(axis = 0)                           #per-mode variance across sweeps

    #----- summary print -----
    finite_tau = tau[np.isfinite(tau)]
    print("=" * 70)
    print(f"[mode_diagnostics] {T} sweeps, {finite_tau.size} live modes")
    if finite_tau.size:
        print(f"  tau_int  median={np.median(finite_tau):.2f}  "
              f"90th={np.percentile(finite_tau, 90):.2f}  max={finite_tau.max():.2f}")
        print(f"  (iid -> 1.0; tau_int ~ T means the mode never moved)")
    centers, tau_prof, counts = radial_profile(tau, ell, nbins = 25, log_bins = True)
    print("  tau_int vs |ell| (stuck bins have tau_int >> 1):")
    for c, tp, ct in zip(centers, tau_prof, counts):
        if ct > 0 and np.isfinite(tp):
            bar = "#" * int(np.clip(tp, 0, 60))
            print(f"    ell~{c:8.1f}  tau_int={tp:7.2f}  n={ct:5d}  {bar}")
    print("=" * 70)

    #----- plot 1: tau_int map over (kx, ky) -----
    tau_disp = np.fft.fftshift(tau, axes = 0)
    plt.figure(figsize = (8, 7))
    plt.imshow(np.log10(tau_disp.T), origin = "lower", aspect = "auto", cmap = "inferno")
    plt.colorbar(label = "log10(tau_int)")
    plt.title(f"{tag}: per-mode integrated autocorrelation time")
    plt.xlabel("kx index (fftshifted)")
    plt.ylabel("ky index (rfft half-axis)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{tag}_tau_int_map.png", dpi = 120)
    plt.close()

    #----- plot 2: tau_int radial profile -----
    plt.figure(figsize = (9, 6))
    ok = counts > 0
    plt.plot(centers[ok], tau_prof[ok], marker = "o")
    plt.axhline(1.0, color = "grey", ls = "--", label = "iid (tau_int = 1)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("|ell|")
    plt.ylabel("mean tau_int in annulus")
    plt.title(f"{tag}: mixing vs scale  (peaks = stuck modes a better M must fix)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{tag}_tau_int_vs_ell.png", dpi = 120)
    plt.close()

    #----- plot 3: traces of the stuckest vs best-mixed live modes -----
    flat_tau = tau.ravel()
    order = np.argsort(np.where(np.isfinite(flat_tau), flat_tau, -np.inf))
    live_order = order[np.isfinite(flat_tau[order])]
    if live_order.size:
        stuck = live_order[-num_traces:][::-1]
        mixed = live_order[:num_traces]
        flat_stack = stack.reshape(T, -1)
        flat_ell = ell.ravel()

        fig, axes = plt.subplots(2, 1, figsize = (11, 9), sharex = True)
        for m in stuck:
            axes[0].plot(np.real(flat_stack[:, m]),
                         label = f"ell~{flat_ell[m]:.0f}, tau={flat_tau[m]:.1f}")
        axes[0].set_title(f"{tag}: Re of the {num_traces} STUCKEST modes (should wander if mixing)")
        axes[0].set_ylabel("Re(phi_mode)")
        axes[0].legend(fontsize = 7, ncol = 2)
        for m in mixed:
            axes[1].plot(np.real(flat_stack[:, m]),
                         label = f"ell~{flat_ell[m]:.0f}, tau={flat_tau[m]:.1f}")
        axes[1].set_title(f"{tag}: Re of the {num_traces} best-mixed modes (reference)")
        axes[1].set_ylabel("Re(phi_mode)")
        axes[1].set_xlabel("sweep")
        axes[1].legend(fontsize = 7, ncol = 2)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{tag}_mode_traces.png", dpi = 120)
        plt.close()

    return {"tau_int": tau, "acf": acf, "variance": variance,
            "ell": ell, "tau_profile": (centers, tau_prof, counts)}
