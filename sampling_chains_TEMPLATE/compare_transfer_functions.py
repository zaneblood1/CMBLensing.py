"""Test that the transfer function R(l) is FLAT in cosmology - the assumption that licenses it.

fisher_forecast's --transfer_function carries the empirical delensing correction across the
whole finite-difference stencil:

    C_l^delensed(theta) = R(l) * C_l^delensed,CAMB(theta)

so the derivative it contracts is R * dC_CAMB/dtheta. That is only legitimate if R itself does
not move with theta over the stencil width. R captures the difference between two LENSING
CALCULATIONS (LenseFlow on a periodic box with a MAP reconstruction, versus CAMB's full-sky
correlation-function lensing at a Wiener Alens_L), and a difference between algorithms should
depend on cosmology only at second order - but "should" is not a measurement, and the
alternative to assuming it is finite-differencing an independent Monte Carlo at every stencil
point, which divides the Monte Carlo noise by 2h with h = 0.05 sigma and is hopeless.

So measure R twice, at two cosmologies separated by MUCH MORE than the finite-difference step,
and check the ratio is consistent with one:

    python merge_delensed_spectra.py --spectra_dir <reference_dir>
    python merge_delensed_spectra.py --spectra_dir <shifted_dir>
    python compare_transfer_functions.py --reference <reference_dir> --shifted <shifted_dir>

The test is deliberately conservative in the useful direction: a shift of several sigma that
leaves R unmoved certainly implies R is unmoved over 0.05 sigma. The script reports the
ratio per band, its combined jackknife error, a chi-squared against the null R_a = R_b, and -
the number that actually matters - the ratio's departure from one rescaled to the stencil
step, which is the systematic error the flatness assumption would introduce into dC/dtheta.

This file is byte-identical between sampling_chains_TEMPLATE/ and sampling_chains/.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cmb_lensing.delensed_spectrum import (MERGED_NAME, _CONFIG_KEYS,
                                           load_transfer_directory)
from cmb_lensing.precompute_camb_1d import PARAM_ORDER, PARAM_SIGMA
from cmb_lensing.fisher_forecast import FD_STEP_FRAC


def load_merged(directory, name = MERGED_NAME):
    path = directory if directory.endswith(".npz") else os.path.join(directory, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {path}. Run merge_delensed_spectra.py on that "
                                f"directory first.")
    return path, dict(np.load(path, allow_pickle = True))


def ratio_and_error(reference_dir, shifted_dir, a, b):
    """R_shifted / R_reference per band, with an error that knows whether the seeds are shared.

    get_delensed_spectra.sh uses the same seed_prefix for every run, so the two cosmologies
    are normally measured on the SAME realizations - common random numbers. That is the right
    way to run it: the field draws are then nearly identical between the two, the cosmic
    variance cancels in the per-realization ratio, and the ratio is far more precise than
    either R is on its own. But it also means the two jackknife errors are strongly
    CORRELATED, so propagating them as if independent (adding in quadrature) badly
    OVERESTIMATES the error on the ratio - which would make the flatness test pass on a
    ratio that had genuinely moved.

    So when the seeds match, the ratio is formed per realization and jackknifed directly.
    When they do not, the runs really are independent and quadrature is correct. The mode
    actually used is returned so the report can say which.
    """
    try:
        stack_a, meta_a = load_transfer_directory(reference_dir, verbose = False)
        stack_b, meta_b = load_transfer_directory(shifted_dir, verbose = False)
    except (FileNotFoundError, ValueError):
        #a caller may point at bare npz files with no per-realization directory behind them
        stack_a = stack_b = None

    if stack_a is not None and meta_a["seeds"] == meta_b["seeds"]:
        key = f"transfer_{str(a['estimator'])}"
        rows = stack_b[key] / stack_a[key]
        n = len(meta_a["seeds"])
        ratio = np.mean(rows, axis = 0)
        if n > 1:
            leave_one_out = (np.sum(rows, axis = 0) - rows) / (n - 1)
            error = np.sqrt((n - 1) / n *
                            np.sum((leave_one_out - np.mean(leave_one_out, axis = 0))**2,
                                   axis = 0))
        else:
            error = np.full(rows.shape[1], np.nan)
        return ratio, error, "paired (shared seeds, jackknifed per realization)"

    ratio = b["transfer"] / a["transfer"]
    error = np.abs(ratio) * np.sqrt((a["transfer_error"] / a["transfer"])**2 +
                                    (b["transfer_error"] / b["transfer"])**2)
    return ratio, error, "independent (different seeds, errors added in quadrature)"


def check_comparable(a, b):
    """Both runs must share everything EXCEPT the cosmology, which is the point of the test."""
    for key in _CONFIG_KEYS:
        left = a[key].item() if a[key].ndim == 0 else a[key]
        right = b[key].item() if b[key].ndim == 0 else b[key]
        if left != right:
            raise ValueError(f"the two runs disagree on {key} ({left!r} vs {right!r}); "
                             f"only the cosmology may differ, or the comparison measures "
                             f"the configuration change rather than the theta dependence")
    if not np.allclose(a["band_ells"], b["band_ells"]):
        raise ValueError("the two runs use different bands")

    displaced = [(name, float(x), float(y))
                 for name, x, y in zip(PARAM_ORDER, a["params"], b["params"]) if x != y]
    if not displaced:
        raise ValueError("the two runs are at the SAME cosmology, so this comparison tests "
                         "nothing. Re-run get_delensed_spectra.sh with shift_param and "
                         "shift_value set, into a different out_dir.")
    return displaced


def main():
    parser = argparse.ArgumentParser(
        description = "Check the delensing transfer function is flat in cosmology")
    parser.add_argument("--reference", type = str, required = True,
                        help = "directory (or npz) holding R at the fiducial cosmology")
    parser.add_argument("--shifted", type = str, required = True,
                        help = "directory (or npz) holding R at the displaced cosmology")
    parser.add_argument("--out_dir", type = str, default = None,
                        help = "where to write the comparison plot (default: --shifted)")
    parser.add_argument("--tolerance", type = float, default = 1e-2,
                        help = "largest fractional drift of R over ONE finite-difference "
                               "step that still counts as negligible (default 1e-2). This, "
                               "not the chi-squared, is what decides whether R may be held "
                               "fixed: with shared seeds the ratio becomes precise enough to "
                               "resolve drifts far too small to move any forecast")
    args = parser.parse_args()

    path_a, a = load_merged(args.reference)
    path_b, b = load_merged(args.shifted)
    displaced = check_comparable(a, b)

    ells = a["band_ells"]
    ratio, error, error_mode = ratio_and_error(args.reference, args.shifted, a, b)

    good = error > 0
    chi_squared = float(np.sum(((ratio[good] - 1) / error[good])**2))
    dof = int(np.sum(good))

    print(f"Transfer-function flatness test")
    print(f"  reference: {path_a}  ({int(a['n_realizations'])} realizations)")
    print(f"  shifted:   {path_b}  ({int(b['n_realizations'])} realizations)")
    for name, reference_value, shifted_value in displaced:
        delta = shifted_value - reference_value
        sigma = delta / PARAM_SIGMA[name]
        step = FD_STEP_FRAC[name] * PARAM_SIGMA[name]
        print(f"  displaced {name}: {reference_value:.6g} -> {shifted_value:.6g} "
              f"(delta {delta:+.4g} = {sigma:+.2f} sigma = {delta / step:+.1f} "
              f"finite-difference steps)")
    print(f"  error mode: {error_mode}")
    #merged files from before 2026-09-18 carry no reconstruction_params; their reconstruction
    #always tracked their own cosmology
    reconstruction = b.get("reconstruction_params", b["params"])
    frozen = not np.allclose(reconstruction, b["params"], rtol = 1e-12, atol = 0)
    print(f"  shifted run's reconstruction: "
          + ("frozen at the reference cosmology (the forecast's convention)" if frozen
             else "built at its own shifted cosmology (NOT the forecast's frozen-Alens_L "
                  "convention - rerun with freeze_reconstruction=1 to test what the "
                  "forecast actually assumes)"))

    print(f"\n  {'l':>8}{'R_shift/R_ref':>16}{'error':>10}{'sigma':>9}")
    for i, ell in enumerate(ells):
        deviation = ((ratio[i] - 1) / error[i]) if error[i] > 0 else np.nan
        print(f"  {ell:8.0f}{ratio[i]:16.4f}{error[i]:10.4f}{deviation:9.1f}")

    print(f"\n  chi^2 = {chi_squared:.1f} for {dof} bands "
          f"({chi_squared / max(dof, 1):.2f} per band) against the null 'R does not move'")

    #the number that decides whether the assumption is safe. R's drift ACROSS THE DISPLACEMENT
    #is what was measured; what matters is its drift across one finite-difference step, which
    #is smaller by the ratio of the two. That is the fractional systematic the flatness
    #assumption puts into dC^delensed/dtheta
    worst_band = int(np.argmax(np.abs(ratio - 1)))
    worst_drift = 0.0
    for name, reference_value, shifted_value in displaced:
        delta = shifted_value - reference_value
        step = FD_STEP_FRAC[name] * PARAM_SIGMA[name]
        drift = np.abs(ratio - 1) * np.abs(step / delta)
        worst_drift = max(worst_drift, float(np.max(drift)))
        print(f"\n  rescaled to ONE finite-difference step in {name} "
              f"(h = {step:.4g}):")
        print(f"    worst band l = {ells[np.argmax(drift)]:.0f}, R drifts "
              f"{np.max(drift):.2e} fractionally")
        print(f"    mean over bands: {np.mean(drift):.2e}")
        print(f"    -> this is the fractional systematic in dC^delensed/dtheta from holding "
              f"R fixed")

    print(f"\n  largest raw departure: band l = {ells[worst_band]:.0f}, "
          f"R_shift/R_ref = {ratio[worst_band]:.4f} "
          f"({(ratio[worst_band] - 1) / error[worst_band]:+.1f} sigma)")

    #TWO separate questions, and only the second one decides anything. With enough
    #realizations and shared seeds the ratio gets precise enough to resolve a drift that is
    #far too small to matter - "statistically distinguishable from flat" and "big enough to
    #corrupt the forecast" are not the same statement, so they are reported apart
    moves = chi_squared / max(dof, 1) >= 2.0
    print(f"\n  1. is R's motion DETECTABLE at this displacement? "
          f"{'yes' if moves else 'no'} (chi^2/band {chi_squared / max(dof, 1):.2f})")
    print(f"  2. is it NEGLIGIBLE over one finite-difference step? "
          f"{'no' if worst_drift > args.tolerance else 'yes'} "
          f"(worst {worst_drift:.2e} vs tolerance {args.tolerance:.0e})")

    if worst_drift <= args.tolerance:
        print(f"  VERDICT: R may be held fixed across the stencil. Its drift over one step "
              f"is {worst_drift:.2e} fractionally"
              + (" - detectable with this many realizations, but far below anything that "
                 "moves a forecast" if moves else ""))
    else:
        print(f"  VERDICT: R drifts {worst_drift:.2e} over one finite-difference step, "
              f"above the {args.tolerance:.0e} tolerance. Compare that against the "
              f"fractional change in C^delensed itself over one step before trusting "
              f"--transfer_function; if it is a real fraction of the signal, measure "
              f"dR/dtheta instead of holding R fixed: get_delensed_spectra.sh's "
              f"derivative_params mode, then merge_delensed_spectra.py --shifted_dirs")

    out_dir = args.out_dir or (args.shifted if os.path.isdir(args.shifted)
                               else os.path.dirname(path_b))
    figure, axis = plt.subplots(figsize = (9, 5))
    axis.axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    axis.errorbar(ells, ratio, yerr = error, marker = "o", markersize = 4, capsize = 3,
                  color = "C0")
    axis.set_xlabel(r"$\ell$")
    axis.set_ylabel(r"$R_{\rm shifted}(\ell)\ /\ R_{\rm reference}(\ell)$")
    shift_label = ", ".join(f"{name} {x:.5g} -> {y:.5g}" for name, x, y in displaced)
    axis.set_title(f"Is the transfer function flat in cosmology?  |  {shift_label}\n"
                   f"chi2/band {chi_squared / max(dof, 1):.2f}", fontsize = 10)
    figure.tight_layout()
    plot_path = os.path.join(out_dir, "transfer_function_flatness.png")
    figure.savefig(plot_path, dpi = 150)
    plt.close(figure)
    print(f"\nwrote {plot_path}")


if __name__ == "__main__":
    main()
