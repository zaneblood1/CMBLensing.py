"""MCMC plots for the LCDM chains of sample_lcdm.sample_joint, and their comparison with one
Fisher forecast.

Input: a directory of the history files sample_joint writes under `hpc_path`,
`<param>_map_<map seed>_chain_<chain index>_history.txt` (one value per sweep). The sampled
parameters, the maps and their sub-chains are read off the file names. Nothing here depends
on which data the chains saw, so temperature ("I"), polarization ("P") and joint ("IP") runs
are analysed alike; `pol` labels the plots and tells the forecast which data to model.

Output (default `<chain_dir>/chain_analysis_output/`):
    <param>/               per sampled parameter: autocorrelations, integrated autocorrelation
                           times, traces, per-map marginals, the histogram of per-map means
                           and the product distribution
    joint_information/     z_scores.png, the chains' covariance / correlation matrices and
                           triangle_plot.png; with the forecast also its matrices and the
                           fisher-over-empirical ratios

The Fisher forecast (fisher_forecast.forecast) is off by default (`run_fisher`). When it
runs, its field blocks are ALWAYS the empirical delensed covariance of a merged
delensed_covariance.npz measured at the chains' `pol` (`delensed_covariance`, required), and
its N_phi is the box QE N0 with the unlensed response and the unlensed (default) or lensed
spectra in the filter (`qe_filter`), or the measured `nphi` of a merged
map_joint_phi_noise.npz / sampler_phi_noise.npz (`phi_noise`). For "P" / "IP" chains
`include_bb = False` leaves the delensed BB block out of the forecast.

    python -m cmb_lensing.chain_analysis <chain_dir> --nside 128 --theta_pix 2.5 --noise 2.5
    python -m cmb_lensing.chain_analysis <chain_dir> --noise 2.5 --run_fisher \\
        --delensed_covariance <delensed_covariance.npz> [--qe_filter lensed | --phi_noise <npz>]
"""

import argparse
import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde
from cmb_lensing.constants import GROUND_TRUTH, CAMB_LMAX
from cmb_lensing.mcmc_diagnostics import gelman_rubin, autocorrelation, integrated_autocorrelation_time
from cmb_lensing.camb_spectra import QE_FILTERS, DEFAULT_QE_FILTER, DEFAULT_POL
from cmb_lensing.matrix_plots import annotated_heatmap, plot_annotated_matrix
from cmb_lensing.fisher_forecast import (correlation_from_covariance, covariance_from_fisher,
                                         forecast)

#sample_joint's history files: <param>_map_<map seed>_chain_<chain index>_history.txt
HISTORY_FILE_PATTERN = re.compile(r"^(?P<param>.+)_map_(?P<map_seed>\d+)_chain_(?P<chain_index>\d+)_history\.txt$")
#the plots go under <chain_dir>/OUTPUT_DIR_NAME unless out_dir is given: one subfolder per
#sampled parameter and JOINT_DIR_NAME for the joint plots
OUTPUT_DIR_NAME = "chain_analysis_output"
JOINT_DIR_NAME = "joint_information"

#the box and the post-processing burn-in (sweeps cut from the start of every sub-chain)
DEFAULT_NSIDE = 128
DEFAULT_THETA_PIX = 2.5
DEFAULT_NOISE = 2.5
DEFAULT_BURN_IN = 1000

#the kde grids: points, and the padding past the samples as a fraction of their range
GRID_SIZE = 1000
GRID_PAD_FRACTION = 0.1
#the multiplicative-mean plot's x range ends where the pdf drops below this fraction of its peak
PDF_XLIM_THRESHOLD = 1e-3
#bootstrap replicates (and their seed) of the product distribution's Monte Carlo errors
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 0
HISTOGRAM_BINS = 20
#decimals of the numbers printed in the mean histogram's legend and title
LEGEND_DECIMALS = 7
FIGURE_DPI = 150

#the labels of the measured (chain) posterior and of the forecast in file names
EMPIRICAL_LABEL = "empirical"
FISHER_LABEL = "fisher"

#Triangle plot styling: the chains always blue, the forecast always red, each contour
#filled translucently under an opaque outline so the overlap stays legible
MEASURED_COLOUR = "blue"
FORECAST_COLOUR = "red"
FILL_ALPHA = 0.18
CONTOUR_LINEWIDTH = 1.6
#square panels (inches), so the ellipses are not stretched
PANEL_WIDTH = 3.3
PANEL_HEIGHT = 3.3
#the axes reach this many sigma past the widest contour
CONTOUR_PAD = 0.8
GAUSSIAN_GRID_SIZE = 500
ANNOTATION_FONTSIZE = 12
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10
#the parameters' symbols (mathtext, without the enclosing $)
PARAM_SYMBOLS = {"theta_MC_100": "100*\\theta_{\\rm MC}",
                 "logA": "\\ln(A_s*10^{\\rm 10})",
                 "ns": "n_s",
                 "ombh2": "\\Omega_b h^2",
                 "omch2": "\\Omega_c h^2"}
#the data each pol setting of the sampler sees, for the triangle plot's subtitle
POL_DESCRIPTIONS = {"I": "Temperature Data",
                    "P": "Polarization Data",
                    "IP": "Temperature + Polarization Data"}


# ── Loading ───────────────────────────────────────────────────────────────

def find_history_files(chain_dir):
    """{param: {(map_seed, chain_index): path}} for every history file in chain_dir."""
    files = {}
    for path in sorted(glob.glob(os.path.join(chain_dir, "*_history.txt"))):
        match = HISTORY_FILE_PATTERN.match(os.path.basename(path))
        if match is not None:
            key = (int(match["map_seed"]), int(match["chain_index"]))
            files.setdefault(match["param"], {})[key] = path
    return files

def load_chains(chain_dir, burn_in = DEFAULT_BURN_IN):
    """Read every post-burn-in sub-chain in a directory of sample_joint history files.

    A sub-chain is kept when every parameter has its file. Its files are cut to their common
    length (they differ by a sweep when copied from a running job) BEFORE the burn-in is
    removed, so row k of every parameter is the same sweep; a sub-chain no longer than the
    burn-in is left out, and so is a map with no sub-chain left.

    Args:
        chain_dir: directory holding `<param>_map_<seed>_chain_<index>_history.txt` files
        burn_in:   sweeps removed from the start of every sub-chain

    Returns:
        (chains, map_seeds) - `chains[param][map][sub_chain]` is a 1D array for every
        GROUND_TRUTH parameter with files, in GROUND_TRUTH's order, with maps and sub-chains
        in the same order for every parameter; map_seeds lists the kept maps.
    """
    files = find_history_files(chain_dir)
    #every sampled parameter, in GROUND_TRUTH's (= PARAM_ORDER's) order
    params = [name for name in GROUND_TRUTH if name in files]
    shared_keys = sorted(set.intersection(*[set(files[name]) for name in params]))

    chains = {name: [] for name in params}
    map_seeds = []
    num_discarded = 0
    for map_seed in sorted({key[0] for key in shared_keys}):
        sub_chains = {name: [] for name in params}
        for key in [key for key in shared_keys if key[0] == map_seed]:
            histories = [np.atleast_1d(np.loadtxt(files[name][key])) for name in params]
            common_length = min(len(history) for history in histories)
            if common_length <= burn_in:
                num_discarded += 1
                continue
            for name, history in zip(params, histories):
                sub_chains[name].append(history[burn_in:common_length])
        if len(sub_chains[params[0]]) > 0:
            map_seeds.append(map_seed)
            for name in params:
                chains[name].append(sub_chains[name])
    print(f"Loaded {sum(len(sub_chains) for sub_chains in chains[params[0]])} sub-chains of "
          f"{len(map_seeds)} maps for {params}; discarded {num_discarded} no longer than the "
          f"burn-in of {burn_in}")
    return chains, map_seeds


# ── Per-parameter diagnostics and plots ───────────────────────────────────

def save_figure(output_dir, file_name):
    plt.savefig(os.path.join(output_dir, file_name), dpi = FIGURE_DPI, bbox_inches = "tight")
    plt.close()
    return

def gelman_rubin_per_map(chains_per_map):
    """The Gelman-Rubin R-hat across each map's sub-chains (NaN for a map with one)."""
    return np.array([gelman_rubin(chains) if len(chains) > 1 else np.nan
                     for chains in chains_per_map])

def metropolis_acceptance_rate(chains_per_map):
    """The fraction of sweeps in which the parameter moved, averaged over every sub-chain.

    A rejected proposal repeats the previous value bit for bit, so this is the Metropolis
    acceptance rate for one move per sweep (sample_joint's default). It needs
    the UNTHINNED post-burn-in chains."""
    rates = [np.mean(chain[1:] != chain[:-1]) for chains in chains_per_map for chain in chains]
    return np.mean(rates)

def plot_autocorrelation_chains(chains_per_map, output_dir, param_name):
    acfs_per_map = []
    plt.figure()
    for chains in chains_per_map:
        acfs = []
        for chain in chains:
            #a chain that never moved has no autocorrelation: NaN
            with np.errstate(divide = "ignore", invalid = "ignore"):
                acf = autocorrelation(chain)
            acfs.append(acf)
            plt.plot(acf)
        acfs_per_map.append(acfs)
    plt.xlabel("Lag"); plt.ylabel("Auto-Correlation")
    plt.axhline(0, color = "black")
    plt.title(f"{param_name} Auto-Correlation")
    save_figure(output_dir, f"{param_name.lower()}_auto_correlation.png")
    return acfs_per_map

def plot_iat_per_chain(acfs_per_map, output_dir, param_name):
    iats_per_chain = [[integrated_autocorrelation_time(acf) for acf in acfs]
                      for acfs in acfs_per_map]
    flat_iats = [iat for iats in iats_per_chain for iat in iats]
    plt.figure()
    plt.barh(range(len(flat_iats)), flat_iats)
    plt.ylabel("Chain"); plt.xlabel("Integrated Auto-Correlation Time")
    mean_iat = np.nanmean(flat_iats)
    plt.axvline(mean_iat, label = f"Mean IAT = {mean_iat}", color = "black")
    plt.legend()
    plt.title(f"{param_name} Integrated Auto Correlation Time per Chain")
    save_figure(output_dir, f"{param_name.lower()}_integrated_autocorrelation_times.png")
    return iats_per_chain

def thin_chains(chains_per_map, iats_per_chain, should_thin = True):
    """One array per map: its sub-chains, each thinned by its own integrated autocorrelation
    time (when should_thin) so the retained points are close to independent, concatenated."""
    thinned = []
    for chains, iats in zip(chains_per_map, iats_per_chain):
        processed = []
        for chain, iat in zip(chains, iats):
            #no thinning for an IAT below 1 or undefined (a chain that never moved)
            stride = max(1, int(iat)) if should_thin and np.isfinite(iat) else 1
            processed.append(chain[::stride])
        thinned.append(np.concatenate(processed))
    return thinned

def plot_traces(chains, output_dir, param_name):
    """One trace per map, of its (thinned, concatenated) samples."""
    plt.figure(figsize = (16, 5))
    for chain in chains:
        plt.plot(range(len(chain)), chain, lw = 0.7)
    lengths = np.array([len(chain) for chain in chains])
    for label, value in (("Mean", np.mean(lengths)), ("Std.", np.std(lengths)),
                         ("Min.", np.min(lengths)), ("Max.", np.max(lengths))):
        plt.plot([], [], label = f"{label} Num. Samples = {value}", color = "white")
    plt.xlabel("Sample"); plt.ylabel("Value")
    plt.title(f"{param_name} Markov Chain Traces")
    plt.legend(loc = "upper left", fontsize = 6, ncol = 2); plt.grid(alpha = 0.2)
    save_figure(output_dir, f"{param_name.lower()}_all_chain_trace.png")
    return

def density_grid(chains):
    """A uniform grid covering every sample, padded by GRID_PAD_FRACTION of their range."""
    low = min(chain.min() for chain in chains)
    high = max(chain.max() for chain in chains)
    pad = GRID_PAD_FRACTION * (high - low)
    return np.linspace(low - pad, high + pad, GRID_SIZE)

def plot_marginals(chains, output_dir, ground_truth, param_name):
    grid = density_grid(chains)
    plt.figure()
    for chain in chains:
        plt.plot(grid, gaussian_kde(chain)(grid))
    plt.axvline(ground_truth, color = "black", label = f"Ground Truth = {ground_truth}")
    plt.legend()
    plt.xlabel(f"{param_name} Value"); plt.ylabel("Density")
    plt.title(f"{param_name} Marginals"); plt.grid(alpha = 0.2)
    save_figure(output_dir, f"{param_name.lower()}_all_chain_marginals.png")
    return

def plot_mean_histogram(chains, output_dir, ground_truth, param_name):
    """Histogram of the per-map posterior means. Returns (std of the means, average
    single-map posterior std, std of the single-map stds); the first two should roughly
    agree for a calibrated posterior."""
    means = np.array([np.mean(chain) for chain in chains])
    widths = np.array([np.std(chain, ddof = 1) for chain in chains])
    means_mean = np.mean(means)
    means_std = np.std(means, ddof = 1)
    widths_std = np.std(widths, ddof = 1)
    avg_width = np.mean(widths)

    plt.figure()
    plt.hist(means, bins = HISTOGRAM_BINS, alpha = 0.6, edgecolor = "white")
    plt.axvline(ground_truth, color = "green", label = f"Ground Truth = {ground_truth}")
    plt.axvline(means_mean, color = "indianred",
                label = f"Mean of Means = {round(means_mean, LEGEND_DECIMALS)}")
    plt.axvline(means_mean + means_std, color = "black",
                label = f"+/- 1 Std of Means = {round(means_std, LEGEND_DECIMALS)}")
    plt.axvline(means_mean - means_std, color = "black")
    plt.xlabel("Mean Value"); plt.ylabel("Count")
    plt.title(f"{param_name} Histogram of Marginal Means\n"
              f"std(Means) = {round(means_std, LEGEND_DECIMALS)} vs avg single-marginal std = "
              f"{round(avg_width, LEGEND_DECIMALS)}\n"
              f"std in single-marginal std = {round(widths_std, LEGEND_DECIMALS)}")
    plt.legend(); plt.grid(alpha = 0.2)
    save_figure(output_dir, f"{param_name.lower()}_mean_histogram.png")
    return means_std, avg_width, widths_std

def product_pdf(densities, grid):
    """The product of the per-map densities on one grid, renormalized after every factor so
    it never underflows."""
    pdf = np.ones_like(grid)
    for density in densities:
        pdf = pdf * density
        pdf = pdf / trapezoid(pdf, grid)
    return pdf

def product_moments(pdf, grid):
    """Mean and standard deviation of a normalized pdf on a grid."""
    mean = trapezoid(pdf * grid, grid)
    std = np.sqrt(trapezoid(pdf * (grid - mean)**2, grid))
    return mean, std

def bootstrap_product_errors(chains, grid, ground_truth):
    """Monte Carlo errors on the product mean, std and z-score (truth - mean) / std from the
    finite chains. Every replicate resamples EVERY map's thinned chain with replacement at its
    own length (the maps themselves are never resampled: that would be the scatter between
    maps, not chain noise), rebuilds each map's KDE and the product. The z-score is formed per
    replicate, so its error carries the mean-std correlation. Valid because the chains are
    thinned by their IAT, so their points are close to independent."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_means = np.zeros(N_BOOTSTRAP)
    boot_stds = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        densities = [gaussian_kde(rng.choice(chain, size = len(chain), replace = True))(grid)
                     for chain in chains]
        boot_means[b], boot_stds[b] = product_moments(product_pdf(densities, grid), grid)
    boot_z_scores = (ground_truth - boot_means) / boot_stds
    return (np.std(boot_means, ddof = 1), np.std(boot_stds, ddof = 1),
            np.std(boot_z_scores, ddof = 1))

def significant_digits(value):
    #digits of the shortest decimal that round-trips the float, leading and trailing zeros dropped
    digits = np.format_float_positional(abs(value), trim = "-").replace(".", "")
    digits = digits.lstrip("0").rstrip("0")
    return max(1, len(digits))

def plot_multiplicative_mean(chains, output_dir, ground_truth, param_name):
    """The product distribution's plot. Returns its mean, std and z-score with their bootstrap
    Monte Carlo errors."""
    #the normalized product of the per-map densities (a gaussian_kde for each map)
    grid = density_grid(chains)
    densities = [gaussian_kde(chain)(grid) for chain in chains]
    pdf = product_pdf(densities, grid)
    mode = grid[np.argmax(pdf)]
    mean, std = product_moments(pdf, grid)
    z_score = (ground_truth - mean) / std
    mean_error, std_error, z_error = bootstrap_product_errors(chains, grid, ground_truth)

    plt.figure(figsize = (13, 5))
    plt.plot(grid, pdf, label = "Combined Product Distribution", color = "blue")
    plt.axvline(ground_truth, label = f"Ground Truth = {ground_truth}", color = "green")
    #mode printed to the fewer significant digits of itself and the ground truth
    mode_digits = min(significant_digits(mode), significant_digits(ground_truth))
    plt.axvline(mode, label = f"Mult. Mode = {mode:.{mode_digits}g}", color = "orange")
    plt.axvline(mean, label = f"Mult. Mean = {standard_notation(mean, mean_error)}", color = "pink")
    plt.axvline(mean + std, label = f"+/- 1 Mult. Std = {standard_notation(std, std_error)}",
                color = "grey")
    plt.axvline(mean - std, color = "grey")
    plt.plot([], [], label = f"Product Z-score = {standard_notation(z_score, z_error)}")

    #clip the x range to where the pdf is above a fraction of its peak (ground truth kept in view)
    above = grid[pdf > PDF_XLIM_THRESHOLD * pdf.max()]
    plt.xlim(min(above.min(), ground_truth), max(above.max(), ground_truth))

    plt.xlabel(f"{param_name} Value"); plt.ylabel("Density")
    plt.title(f"{param_name} Combined Product Distribution")
    plt.legend(fontsize = 7.5); plt.grid(alpha = 0.2)
    save_figure(output_dir, f"{param_name.lower()}_mean_distributions.png")
    return mean, std, z_score, mean_error, std_error, z_error

def per_param_analysis(chains_per_map, ground_truth, param_name, output_dir,
                       should_thin = True):
    """Every single-parameter diagnostic and plot, into output_dir. chains_per_map is
    load_chains' chains[param_name]. Returns the numbers of the z_scores.png table."""
    print(f"\nBeginning {param_name} analysis")
    os.makedirs(output_dir, exist_ok = True)

    r_hats = gelman_rubin_per_map(chains_per_map)
    print(f"{param_name} Gelman-Rubin R-hat per map = {r_hats}")
    print(f"{param_name} Gelman-Rubin R-hat mean = {np.nanmean(r_hats)}, "
          f"std = {np.nanstd(r_hats)}")
    acceptance_rate = metropolis_acceptance_rate(chains_per_map)
    print(f"{param_name} Metropolis-Hastings acceptance rate = {acceptance_rate:.4f}")

    acfs_per_map = plot_autocorrelation_chains(chains_per_map, output_dir, param_name)
    iats_per_chain = plot_iat_per_chain(acfs_per_map, output_dir, param_name)
    #one (thinned) array per map from here on
    chains = thin_chains(chains_per_map, iats_per_chain, should_thin)

    plot_traces(chains, output_dir, param_name)
    plot_marginals(chains, output_dir, ground_truth, param_name)
    _, avg_width, widths_std = plot_mean_histogram(chains, output_dir, ground_truth, param_name)
    (product_mean, product_std, product_z_score, product_mean_error, product_std_error,
     product_z_error) = plot_multiplicative_mean(chains, output_dir, ground_truth, param_name)

    return {"product_mean": product_mean, "product_std": product_std,
            "product_z_score": product_z_score,
            "product_mean_error": product_mean_error, "product_std_error": product_std_error,
            "product_z_error": product_z_error,
            "avg_sigma_single": avg_width,
            #error on the average single-map width: the scatter of the widths over sqrt(maps)
            "avg_sigma_error": widths_std / np.sqrt(len(chains)),
            "r_hats": r_hats, "acceptance_rate": acceptance_rate}


# ── Number formatting ─────────────────────────────────────────────────────

def param_symbol(name):
    """A parameter's mathtext symbol, enclosed in $, or its plain name if it has none."""
    if name in PARAM_SYMBOLS:
        return f"${PARAM_SYMBOLS[name]}$"
    return name

def decimal_exponent(value):
    return int(np.floor(np.log10(abs(value))))

def error_decimal(error):
    """The decimal place of the first significant digit of error once it is rounded to one
    significant figure, e.g. 0.0008465 -> 4, 0.00096 (-> 0.001) -> 3, 23 -> -1."""
    decimal = -decimal_exponent(error)
    rounded = round(float(error), decimal)
    return -decimal_exponent(rounded)

def rounded_text(value, decimal):
    """value rounded at the given decimal place (negative: to tens, hundreds, ...)."""
    return f"{round(float(value), decimal):.{max(decimal, 0)}f}"

def standard_notation(value, error):
    """value(e) in the standard concise notation, value +/- e with e in units of the last shown
    digit: the error rounded to one significant figure, the value rounded at that decimal
    place, e.g. (0.109238, 0.00031) -> 0.1092(3) = 0.1092 +/- 0.0003. An error of 10 or more
    is written out, (123, 20) -> 120(20)."""
    decimal = error_decimal(error)
    error_digits = round(float(error) * 10.0**max(decimal, 0), min(decimal, 0))
    return f"{rounded_text(value, decimal)}({error_digits:.0f})"

def format_scaled(value, exponent, decimal, error = None):
    """value (+/- error) as one mathtext mantissa times 10^exponent, rounded at the raw
    decimal place `decimal`, e.g. 0.003268 +/- 0.0008465 at exponent -3 ->
    (3.3 +/- 0.8) x 10^-3, so a measured and a forecast number written at the same exponent
    can be read against each other digit by digit. A forecast number has no error."""
    scale = 10.0**exponent
    mantissa_decimal = decimal + exponent
    if error is None:
        return f"${rounded_text(value / scale, mantissa_decimal)}\\times 10^{{{exponent}}}$"
    return (f"$({rounded_text(value / scale, mantissa_decimal)} \\pm "
            f"{rounded_text(error / scale, mantissa_decimal)})\\times 10^{{{exponent}}}$")

def ratio_text(ratio, error):
    """ratio +/- error, the error to its first significant figure and the ratio rounded at
    the same decimal place, e.g. 1.017 +/- 0.0081 -> 1.017 +/- 0.008."""
    decimal = error_decimal(error)
    return f"{rounded_text(ratio, decimal)} $\\pm$ {rounded_text(error, decimal)}"


# ── Joint statistics ──────────────────────────────────────────────────────

def measured_errors(per_map_cov_mats):
    """(cov_err, sigma_err): the delete-one-MAP jackknife of the map-averaged covariance.

    With C_(m) the average leaving out map m, the widths are recomputed as
    sigma_(m) = sqrt(C_(m),ii) and
        err_sigma^2 = (M - 1) / M * sum_m (sigma_(m) - mean_m sigma_(m))^2 ,
    the jackknife of the widths themselves (no linear propagation from C). err_C is the same
    jackknife applied to each covariance element. Needs at least two maps.
    """
    num_maps = len(per_map_cov_mats)
    total = np.sum(per_map_cov_mats, axis = 0)
    leave_one_out = np.array([(total - cov_mat) / (num_maps - 1)
                              for cov_mat in per_map_cov_mats])
    factor = (num_maps - 1) / num_maps
    cov_err = np.sqrt(factor * np.sum((leave_one_out - np.mean(leave_one_out, axis = 0))**2, axis = 0))
    sigmas_loo = np.sqrt(np.array([np.diag(cov_mat) for cov_mat in leave_one_out]))
    sigma_err = np.sqrt(factor * np.sum((sigmas_loo - np.mean(sigmas_loo, axis = 0))**2, axis = 0))
    return cov_err, sigma_err

def get_avg_covariance_matrix(chains):
    """Map-averaged posterior covariance of the sampled parameters, with its jackknife error.

    `chains` is load_chains' dict, whose sub-chains line up row by row across parameters.
    Each map's sub-chains are concatenated and give one covariance matrix, and those are
    averaged over maps. Samples from different maps are never pooled: each map is a
    different data realization with its own posterior mean, so pooling would add the scatter
    of those means to the diagonal. The average is a single-realization posterior, which is
    what a Fisher forecast predicts.

    Returns:
        (avg_cov_mat, param_names, cov_err, sigma_err), the last two from measured_errors:
        the element-wise error of the averaged covariance and the error of each width
        sqrt(C_ii).
    """
    param_names = list(chains.keys())
    num_maps = len(chains[param_names[0]])
    per_map_cov_mats = []
    for map_idx in range(num_maps):
        columns = [np.concatenate(chains[name][map_idx]) for name in param_names]
        per_map_cov_mats.append(np.atleast_2d(np.cov(np.column_stack(columns), rowvar = False)))
    per_map_cov_mats = np.array(per_map_cov_mats)
    cov_err, sigma_err = measured_errors(per_map_cov_mats)
    return np.mean(per_map_cov_mats, axis = 0), param_names, cov_err, sigma_err

def get_means_per_param(chains):
    """The mean of every post-burn-in sample of each parameter, over all maps and chains."""
    return [np.mean(np.concatenate([chain for sub_chains in chains[name] for chain in sub_chains]))
            for name in chains]

def get_forecasted_covariance(nside, theta_pix, noise, param_names, delensed_covariance,
                              pol = DEFAULT_POL, qe_filter = DEFAULT_QE_FILTER,
                              phi_noise = None, include_bb = True, ground_truth = GROUND_TRUTH,
                              camb_lmax = CAMB_LMAX):
    """Fisher-forecast covariance of the sampled parameters, with white noise and no beam.

    A thin wrapper of fisher_forecast.forecast. Its field blocks are the empirical delensed
    covariance of `delensed_covariance` (plus the analytic noise); its phi block's noise is
    the box QE N0 or a measured matrix. Build the delensed covariance's Wiener filter from
    the same N_phi.

    Args:
        nside, theta_pix, noise: the chains' box (pixels per side, arcmin, uK-arcmin)
        param_names:         the sampled parameters; the others are held fixed
        delensed_covariance: a merged delensed_covariance.npz at this box, noise level and
                             `pol` (required)
        pol:                 "I", "P" or "IP": the data the chains saw
        qe_filter:           "unlensed" (default) or "lensed": the spectra in the QE N0's
                             filter. The response always uses the unlensed spectra
        phi_noise:           a merged map_joint_phi_noise.npz / sampler_phi_noise.npz; its
                             `nphi` then replaces the QE N0
        include_bb:          for "P" / "IP", whether the delensed BB block is counted
        ground_truth:        {param_name: value}, the fiducial point, all five parameters
        camb_lmax:           how far CAMB is run

    Returns:
        The covariance matrix F^-1 over param_names, rows and columns in
        constants.PARAM_ORDER (the order load_chains returns its parameters in).
    """
    is_sampled = {name: name in param_names for name in ground_truth}
    fisher, names = forecast(nside, theta_pix, noise, is_sampled, ground_truth,
                             delensed_covariance, pol = pol, qe_filter = qe_filter,
                             phi_noise = phi_noise, include_bb = include_bb,
                             camb_lmax = camb_lmax)
    return covariance_from_fisher(fisher, names)

def forecast_title(delensed_covariance, qe_filter, phi_noise):
    """The forecast's name in plot titles: where its N_phi and its field blocks come from."""
    if phi_noise is None:
        nphi_text = f"QE ({qe_filter} filter, unlensed response)"
    else:
        nphi_text = f"measured ({os.path.basename(phi_noise)})"
    covariance_folder = os.path.basename(os.path.dirname(os.path.abspath(delensed_covariance)))
    return f"N_phi: {nphi_text}\nField blocks: measured delensed ({covariance_folder})"


# ── Joint plots ───────────────────────────────────────────────────────────

def plot_z_score_table(summaries, ground_truth, joint_dir):
    """z_scores.png in joint_dir: one row per sampled parameter, the ground truth next to its
    symbol, and the columns, a(b) = a +/- b in units of a's last digit (standard_notation)

    product mean        a(b), b its bootstrap Monte Carlo error (bootstrap_product_errors)
    product std         a(b), b its bootstrap Monte Carlo error
    avg single-map std  a(b), b = std of the per-map stds / sqrt(number of maps)
    product z-score     (truth - product mean) / product std, a(b) with b the std of the
                        z-score formed in every bootstrap replicate
    """
    column_labels = ["Product Mean", "Product Std", "Avg. Single-Map Std", "Product Z-score"]
    row_labels = []
    cells = []
    for name, summary in summaries.items():
        row_labels.append(f"{param_symbol(name)}  (truth = {ground_truth[name]})")
        cells.append([standard_notation(summary["product_mean"], summary["product_mean_error"]),
                      standard_notation(summary["product_std"], summary["product_std_error"]),
                      standard_notation(summary["avg_sigma_single"], summary["avg_sigma_error"]),
                      standard_notation(summary["product_z_score"], summary["product_z_error"])])

    figure, axis = plt.subplots(figsize = (11, 0.5 * (len(row_labels) + 1)))
    axis.axis("off")
    table = axis.table(cellText = cells, rowLabels = row_labels, colLabels = column_labels,
                       cellLoc = "center", loc = "center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    table.auto_set_column_width(list(range(-1, len(column_labels))))
    figure.savefig(os.path.join(joint_dir, "z_scores.png"), dpi = FIGURE_DPI, bbox_inches = "tight")
    plt.close(figure)
    return

def plot_covariance_and_correlation(cov_mat, param_names, label, joint_dir, title = None):
    """covariance_matrix_<label>.png and correlation_matrix_<label>.png (every entry printed
    in its box) for one posterior: the chains' (EMPIRICAL_LABEL) or the forecast's. `title`
    names the posterior in the plot titles (`label` when None)."""
    title = label if title is None else title
    figure, axis = plt.subplots(figsize = (7.5, 6.5))
    annotated_heatmap(axis, cov_mat, param_names, f"{title}\ncovariance matrix", "Value")
    figure.tight_layout()
    figure.savefig(os.path.join(joint_dir, f"covariance_matrix_{label}.png"), dpi = FIGURE_DPI)
    plt.close(figure)
    plot_annotated_matrix(correlation_from_covariance(cov_mat), param_names,
                          f"{title}\ncorrelation matrix",
                          os.path.join(joint_dir, f"correlation_matrix_{label}.png"),
                          norm = matplotlib.colors.Normalize(vmin = -1, vmax = 1))
    return

def stacked_text(ax, lines, x, y_top, line_step, ha = "left"):
    """Write (text, colour) lines downward from y_top in axes coordinates. The text is kept
    out of constrained layout: it is placed inside the cells, and counting it as a
    decoration collapses the axes."""
    for k, (text, colour) in enumerate(lines):
        ax.text(x, y_top - k * line_step, text, color = colour, transform = ax.transAxes,
                fontsize = ANNOTATION_FONTSIZE, ha = ha, va = "top", in_layout = False)
    return

def triangle_plot(measured_cov_mat, param_names, means, measured_cov_sigma, measured_sigma_err,
                  joint_dir, forecast_cov_mat = None, sigmas = (1,), nside = None,
                  theta_pix = None, noise = None, pol = None):
    """triangle_plot.png in joint_dir: the sampled posterior (blue) and, when
    `forecast_cov_mat` is given, the Fisher forecast (red).

    The diagonal carries each parameter's marginal Gaussian, the lower triangle each pair's
    covariance ellipse (semi-axes k * sqrt(eigenvalue) of the pair's 2x2 block for every k in
    `sigmas`; the k = 1 ellipse holds 39% of a 2D Gaussian). Each is filled translucently
    under an opaque outline.

    Everything is centred on the MEASURED means: a forecast is a curvature, with a shape but
    no centre, so a common centre makes the plot a direct read of width and orientation.
    Each diagonal panel is annotated with sigma_measured +/- its error and, with a forecast,
    sigma_fisher and their ratio (above 1: the sampler is wider than the forecast says the
    data allows). Each upper-triangle cell (row a, column b) reports the off-diagonal
    covariance C_ab of the lower panel it mirrors, measured and forecast, and their ratio.

    `forecast_cov_mat` must be in param_names order (get_forecasted_covariance's output) and
    carries no error bars. The measured errors are measured_errors' delete-one-map jackknife:
    `measured_cov_sigma` adds dashed blue contours at measured_cov_mat -/+ measured_cov_sigma
    (every element shifted) and is the error of every covariance; `measured_sigma_err` is
    each width's.
    """
    measured = np.asarray(measured_cov_mat)
    measured_cov_sigma = np.asarray(measured_cov_sigma)
    means = np.asarray(means)
    n = len(param_names)
    has_forecast = forecast_cov_mat is not None

    #(covariance, colour, filled, linestyle): the central contours filled, the measured
    #+/- error limits as dashed outlines only
    layers = [(measured, MEASURED_COLOUR, True, "-"),
              (measured - measured_cov_sigma, MEASURED_COLOUR, False, "--"),
              (measured + measured_cov_sigma, MEASURED_COLOUR, False, "--")]
    widest = np.sqrt(np.diag(measured + measured_cov_sigma))
    if has_forecast:
        forecast_cov = np.asarray(forecast_cov_mat)
        layers.append((forecast_cov, FORECAST_COLOUR, True, "-"))
        widest = np.maximum(widest, np.sqrt(np.diag(forecast_cov)))

    #pad the axes out past the widest requested contour of the widest matrix
    pad = max(sigmas) + CONTOUR_PAD
    limits = [(means[i] - pad * widest[i], means[i] + pad * widest[i]) for i in range(n)]

    #annotation line spacing in axes fraction: one text line over the panel's height
    #(constrained layout leaves the axes ~80% of their cell; mathtext sub / superscripts
    #make the lines taller than the font size)
    line_step = 1.5 * ANNOTATION_FONTSIZE / (0.8 * PANEL_HEIGHT * 72)
    #the diagonal's text block sits above the Gaussians; raise the y limit so the
    #unit-height curves stay entirely below it
    num_diagonal_lines = 3 if has_forecast else 1
    text_fraction = 0.04 + num_diagonal_lines * line_step
    diag_ylim = 1.05 / (1 - text_fraction)

    fig, axes = plt.subplots(n, n, figsize = (PANEL_WIDTH * n, PANEL_HEIGHT * n),
                             squeeze = False, layout = "constrained")
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            #the upper triangle carries no panel, only the mirrored pair's covariances
            if j > i:
                ax.axis("off")
                cov_measured = measured[i, j]
                cov_error = measured_cov_sigma[i, j]
                exponent = decimal_exponent(cov_measured)
                #every covariance rounded at the first significant digit of the jackknife error
                decimal = error_decimal(cov_error)
                lines = [(f"Cov({param_symbol(param_names[i])}, {param_symbol(param_names[j])})", "black"),
                         (f"$C_{{\\rm measured}}$ = {format_scaled(cov_measured, exponent, decimal, cov_error)}", MEASURED_COLOUR)]
                if has_forecast:
                    cov_fisher = forecast_cov[i, j]
                    ratio = cov_measured / cov_fisher
                    lines.append((f"$C_{{\\rm fisher}}$ = {format_scaled(cov_fisher, exponent, decimal)}", FORECAST_COLOUR))
                    lines.append((f"$C_{{\\rm measured}}/C_{{\\rm fisher}}$ = {ratio_text(ratio, cov_error / abs(cov_fisher))}", "black"))
                y_top = 0.5 + 0.5 * len(lines) * line_step
                stacked_text(ax, lines, 0.5, y_top, line_step, ha = "center")
                continue
            for cov_mat, colour, filled, linestyle in layers:
                if i == j:
                    sigma = np.sqrt(cov_mat[i, i])
                    grid = np.linspace(*limits[i], GAUSSIAN_GRID_SIZE)
                    curve = np.exp(-0.5 * ((grid - means[i]) / sigma) ** 2)
                    if filled:
                        ax.fill_between(grid, 0, curve, color = colour, alpha = FILL_ALPHA,
                                        lw = 0, zorder = 1)
                    ax.plot(grid, curve, color = colour, ls = linestyle,
                            lw = CONTOUR_LINEWIDTH, zorder = 3)
                else:
                    #column j is the x parameter, row i the y parameter; the eigenvectors of
                    #their 2x2 block are the ellipse axes and the eigenvalues its squared
                    #semi-axes
                    block = cov_mat[np.ix_([j, i], [j, i])]
                    eig_vals, eig_vecs = np.linalg.eigh(block)
                    angle = np.degrees(np.arctan2(eig_vecs[1, -1], eig_vecs[0, -1]))
                    #widest contour first so the tighter ones stay visible on top of it
                    for k in sorted(sigmas, reverse = True):
                        width, height = 2 * k * np.sqrt(eig_vals[::-1])
                        #fill and outline as separate patches so every fill sits beneath
                        #every outline, whatever order the layers come in
                        if filled:
                            ax.add_patch(Ellipse(
                                (means[j], means[i]), width, height, angle = angle,
                                facecolor = colour, edgecolor = "none", alpha = FILL_ALPHA,
                                zorder = 1))
                        ax.add_patch(Ellipse(
                            (means[j], means[i]), width, height, angle = angle,
                            facecolor = "none", edgecolor = colour, ls = linestyle,
                            lw = CONTOUR_LINEWIDTH, zorder = 3))
            if i == j:
                #sigma_measured +/- its error and, with a forecast, its sigma and the ratio
                sigma_measured = np.sqrt(measured[i, i])
                error_measured = measured_sigma_err[i]
                exponent = decimal_exponent(sigma_measured)
                decimal = error_decimal(error_measured)
                lines = [(f"$\\sigma_{{\\rm measured}}$ = {format_scaled(sigma_measured, exponent, decimal, error_measured)}", MEASURED_COLOUR)]
                if has_forecast:
                    sigma_fisher = np.sqrt(forecast_cov[i, i])
                    lines.append((f"$\\sigma_{{\\rm fisher}}$ = {format_scaled(sigma_fisher, exponent, decimal)}", FORECAST_COLOUR))
                    lines.append((f"$\\sigma_{{\\rm measured}}/\\sigma_{{\\rm fisher}}$ = {ratio_text(sigma_measured / sigma_fisher, error_measured / sigma_fisher)}", "black"))
                stacked_text(ax, lines, 0.03, 0.97, line_step)
                ax.set_ylim(0, diag_ylim)
                ax.set_yticks([])
            else:
                ax.set_ylim(*limits[i])
            ax.set_xlim(*limits[j])
            ax.grid(alpha = 0.2)
            ax.tick_params(labelsize = TICK_FONTSIZE)
            ax.locator_params(nbins = 4)
            if i == n - 1:
                ax.set_xlabel(param_symbol(param_names[j]), fontsize = LABEL_FONTSIZE)
                plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right")
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(param_symbol(param_names[i]), fontsize = LABEL_FONTSIZE)
            elif i != j:
                ax.set_yticklabels([])

    #the filled contours get a filled swatch, the dashed measured +/- limits a dashed line
    def filled_handle(colour, label):
        return Patch(facecolor = to_rgba(colour, FILL_ALPHA), edgecolor = colour,
                     lw = CONTOUR_LINEWIDTH, label = label)
    handles = [filled_handle(MEASURED_COLOUR, "Sampler Data"),
               Line2D([0], [0], color = MEASURED_COLOUR, lw = 2, ls = "--",
                      label = "Sampler Data $\\pm$ 1 Jackknife")]
    if has_forecast:
        handles.append(filled_handle(FORECAST_COLOUR, "Fisher"))
    #the upper triangle holds the covariances, so the legend goes below the grid, where
    #constrained layout reserves room for it (above, it collides with the suptitle)
    fig.legend(handles = handles, loc = "outside lower center", fontsize = 12,
               ncol = len(handles))
    title = "Sampler Data v.s. Fisher Forecast" if has_forecast else "Sampler Data"
    if None not in (nside, theta_pix, noise, pol):
        title += (f"\n{POL_DESCRIPTIONS[pol]}, {noise:g}$\\mu$K-arcmin, {theta_pix:g}' pixels, "
                  f"{nside}$^2$ pixels")
    fig.suptitle(title, fontsize = LABEL_FONTSIZE)
    plt.savefig(os.path.join(joint_dir, "triangle_plot.png"), dpi = FIGURE_DPI,
                bbox_inches = "tight")
    plt.close(fig)
    return

def joint_param_analysis(chains, joint_dir, forecast_cov_mat = None, forecast_name = "",
                         nside = None, theta_pix = None, noise = None, pol = None):
    """The chains' covariance and correlation matrices and the triangle plot, into joint_dir;
    with `forecast_cov_mat` (in the chains' parameter order) also the forecast's matrices and
    the fisher-over-empirical ratios. Returns (measured covariance, its element-wise
    jackknife error, the jackknife error of the widths)."""
    print("\nBeginning joint parameter analysis")
    measured_cov_mat, param_names, measured_cov_sigma, measured_sigma_err = get_avg_covariance_matrix(chains)
    plot_covariance_and_correlation(measured_cov_mat, param_names, EMPIRICAL_LABEL, joint_dir,
                                    title = "Chains (empirical)")
    if forecast_cov_mat is not None:
        plot_covariance_and_correlation(forecast_cov_mat, param_names, FISHER_LABEL, joint_dir,
                                        title = f"Forecast\n{forecast_name}")
        #per-entry ratios, coloured about 1 (agreement)
        ratio_label = f"{FISHER_LABEL}_over_{EMPIRICAL_LABEL}"
        plot_annotated_matrix(forecast_cov_mat / measured_cov_mat, param_names,
                              f"Forecast / chains covariance\n{forecast_name}",
                              os.path.join(joint_dir, f"covariance_ratio_{ratio_label}.png"),
                              norm = matplotlib.colors.CenteredNorm(vcenter = 1))
        plot_annotated_matrix(correlation_from_covariance(forecast_cov_mat) /
                              correlation_from_covariance(measured_cov_mat), param_names,
                              f"Forecast / chains correlation\n{forecast_name}",
                              os.path.join(joint_dir, f"correlation_ratio_{ratio_label}.png"),
                              norm = matplotlib.colors.CenteredNorm(vcenter = 1))
    triangle_plot(measured_cov_mat, param_names, get_means_per_param(chains),
                  measured_cov_sigma, measured_sigma_err, joint_dir,
                  forecast_cov_mat = forecast_cov_mat, nside = nside, theta_pix = theta_pix,
                  noise = noise, pol = pol)
    return measured_cov_mat, measured_cov_sigma, measured_sigma_err


# ── Entry points ──────────────────────────────────────────────────────────

def main(chain_dir, nside = DEFAULT_NSIDE, theta_pix = DEFAULT_THETA_PIX, noise = DEFAULT_NOISE,
         pol = DEFAULT_POL, burn_in = DEFAULT_BURN_IN,
         ground_truth = GROUND_TRUTH, should_thin = True, out_dir = None, run_fisher = False,
         delensed_covariance = None, qe_filter = DEFAULT_QE_FILTER, phi_noise = None,
         include_bb = True, camb_lmax = CAMB_LMAX):
    """Analyse a directory of LCDM chains: the MCMC plots and, optionally, the comparison
    with a Fisher forecast.

    Args:
        chain_dir:  directory of sample_joint's history files,
                    `<param>_map_<seed>_chain_<index>_history.txt`; the sampled parameters,
                    maps and sub-chains are read off the file names
        nside, theta_pix, noise: the box the chains ran on (pixels per side, arcmin,
                    uK-arcmin); they label the triangle plot and set the forecast's box
        pol:        "I", "P" or "IP", the data the chains saw; it labels the triangle plot
                    and is the data the forecast models
        burn_in:    sweeps removed from the start of every sub-chain
        ground_truth: {param_name: value} the maps were simulated at (all five parameters)
        should_thin: thin each sub-chain by its integrated autocorrelation time before the
                    per-parameter plots (the covariance always uses the unthinned chains)
        out_dir:    where the plots go (default `<chain_dir>/chain_analysis_output`)
        run_fisher: also run the Fisher forecast for data of this `pol` and compare it with
                    the chains. Needs `delensed_covariance`
        delensed_covariance: a merged delensed_covariance.npz measured at this box, noise
                    level and `pol`: the forecast's field blocks
        qe_filter:  "unlensed" (default) or "lensed": the spectra in the filter of the QE N0
                    the forecast uses as its phi noise (the response is always unlensed)
        phi_noise:  a merged map_joint_phi_noise.npz / sampler_phi_noise.npz whose `nphi`
                    replaces the QE N0
        include_bb: for "P" / "IP" chains, whether the forecast counts the delensed BB block
        camb_lmax:  how far the forecast runs CAMB

    Returns:
        A dict: `param_names`, `map_seeds`, `chains` (load_chains' output), `summaries`
        (per parameter: the product mean / std / z-score with their errors, the average
        single-map std, R-hat per map and the acceptance rate), `measured_covariance` with
        `measured_covariance_error` and `measured_sigma_error` (delete-one-map jackknife),
        `forecast_covariance` (None without run_fisher) and `out_dir`.
    """
    #checked before the chains are analysed, not minutes later inside the forecast
    if run_fisher and delensed_covariance is None:
        raise ValueError("run_fisher needs delensed_covariance, a merged "
                         "delensed_covariance.npz: there is no analytic delensing fallback")

    if out_dir is None:
        out_dir = os.path.join(chain_dir, OUTPUT_DIR_NAME)
    joint_dir = os.path.join(out_dir, JOINT_DIR_NAME)
    os.makedirs(joint_dir, exist_ok = True)

    chains, map_seeds = load_chains(chain_dir, burn_in)
    param_names = list(chains.keys())
    summaries = {}
    for name in param_names:
        summaries[name] = per_param_analysis(chains[name], ground_truth[name], name,
                                             os.path.join(out_dir, name.lower()), should_thin)
    plot_z_score_table(summaries, ground_truth, joint_dir)

    forecast_cov_mat = None
    forecast_name = ""
    if run_fisher:
        forecast_cov_mat = get_forecasted_covariance(nside, theta_pix, noise, param_names,
                                                     delensed_covariance, pol = pol,
                                                     qe_filter = qe_filter,
                                                     phi_noise = phi_noise,
                                                     include_bb = include_bb,
                                                     ground_truth = ground_truth,
                                                     camb_lmax = camb_lmax)
        forecast_name = forecast_title(delensed_covariance, qe_filter, phi_noise)
    measured_cov_mat, measured_cov_sigma, measured_sigma_err = joint_param_analysis(
        chains, joint_dir, forecast_cov_mat = forecast_cov_mat, forecast_name = forecast_name,
        nside = nside, theta_pix = theta_pix, noise = noise, pol = pol)
    print(f"\nPlots written to {out_dir}")

    return {"param_names": param_names, "map_seeds": map_seeds, "chains": chains,
            "summaries": summaries, "measured_covariance": measured_cov_mat,
            "measured_covariance_error": measured_cov_sigma,
            "measured_sigma_error": measured_sigma_err,
            "forecast_covariance": forecast_cov_mat, "out_dir": out_dir}

def parse_args():
    parser = argparse.ArgumentParser(
        description = "MCMC plots for a directory of sample_joint LCDM chains and, with "
                      "--run_fisher, their comparison with a Fisher forecast")
    parser.add_argument("--chain_dir",
                        help = "directory of <param>_map_<seed>_chain_<index>_history.txt files")
    parser.add_argument("--nside", type = int, default = DEFAULT_NSIDE,
                        help = "pixels per side of the chains' box")
    parser.add_argument("--theta_pix", type = float, default = DEFAULT_THETA_PIX,
                        help = "pixel width in arcmin")
    parser.add_argument("--noise", type = float, default = DEFAULT_NOISE,
                        help = "white noise level in uK-arcmin")
    parser.add_argument("--pol", choices = list(POL_DESCRIPTIONS), default = DEFAULT_POL,
                        help = "the data the chains saw; labels the triangle plot")
    parser.add_argument("--burn_in", type = int, default = DEFAULT_BURN_IN,
                        help = "sweeps removed from the start of every sub-chain")
    parser.add_argument("--no_thin", action = "store_true",
                        help = "do not thin the chains by their integrated autocorrelation "
                               "time before the per-parameter plots")
    parser.add_argument("--out_dir", default = None,
                        help = f"where the plots go (default <chain_dir>/{OUTPUT_DIR_NAME})")
    parser.add_argument("--run_fisher", action = "store_true",
                        help = "also run the Fisher forecast and compare it with the chains "
                               "(needs --delensed_covariance)")
    parser.add_argument("--delensed_covariance", default = None,
                        help = "a merged delensed_covariance.npz at this box, noise level and "
                               "pol: the forecast's field blocks")
    parser.add_argument("--qe_filter", choices = QE_FILTERS, default = DEFAULT_QE_FILTER,
                        help = "the spectra in the filter of the forecast's QE N0 (the "
                               "response always uses the unlensed spectra)")
    parser.add_argument("--phi_noise", default = None,
                        help = "a merged map_joint_phi_noise.npz / sampler_phi_noise.npz whose "
                               "`nphi` replaces the QE N0 in the forecast")
    parser.add_argument("--no_bb", action = "store_true",
                        help = "for P / IP chains: leave the delensed BB block out of the "
                               "forecast")
    parser.add_argument("--camb_lmax", type = int, default = CAMB_LMAX,
                        help = "how far the forecast runs CAMB")
    args = parser.parse_args()
    if args.run_fisher and args.delensed_covariance is None:
        parser.error("--run_fisher needs --delensed_covariance")
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.chain_dir, nside = args.nside, theta_pix = args.theta_pix, noise = args.noise,
         pol = args.pol, burn_in = args.burn_in,
         should_thin = not args.no_thin, out_dir = args.out_dir, run_fisher = args.run_fisher,
         delensed_covariance = args.delensed_covariance, qe_filter = args.qe_filter,
         phi_noise = args.phi_noise, include_bb = not args.no_bb, camb_lmax = args.camb_lmax)
