import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from cmb_lensing.constants import GROUND_TRUTH
from cmb_lensing.fisher_forecast import (annotated_heatmap, plot_annotated_matrix,
                                         correlation_from_covariance, covariance_from_fisher,
                                         forecast)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HERE)
CHAIN_DATA_DIR = os.path.join(HERE, "lcdm_sampler_data")
#every plot: one subfolder per sampled parameter and JOINT_DIR_NAME for the joint plots
OUTPUT_DIR = os.path.join(HERE, "chain_analysis_output")
JOINT_DIR_NAME = "joint_information"

#the forecast's temperature block (fisher_forecast.T_BLOCKS): CAMB's "unlensed" or "lensed"
#C_TT, CAMB's analytic "delensed" one, or "measured" (a merged delensed_covariance.npz)
DEFAULT_T_BLOCK = "unlensed"
#the forecast's phi noise: "qe" (the box QE N0) or "measured" (a merged phi noise npz)
DEFAULT_PHI_NOISE_SOURCE = "qe"
#the QE N0's filter (fisher_forecast's "qe_<filter>" nphi_source: "unlensed" or "delensed")
#and response spectrum ("unlensed" or "gradient")
DEFAULT_QE_FILTER = "unlensed"
DEFAULT_QE_RESPONSE = "unlensed"
#the nphi_source a measured phi noise file is forecast under; fisher_forecast reads the
#`nphi` key of a map_joint and a sampler file alike
FILE_NPHI_SOURCE = "map_joint"

#how far CAMB is run for the forecasts (None: the rfft grid's corner mode)
FORECAST_CAMB_LMAX = 4000
BUFFER = 1e-4
GRID_SIZE = 1000
#the multiplicative-mean plot's x range ends where the pdf drops below this fraction of its peak
PDF_XLIM_THRESHOLD = 1e-3
#bootstrap replicates (and their seed) of the product distribution's Monte Carlo errors
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 0

#thin each chain by its integrated autocorrelation time before the per-parameter plots
SHOULD_THIN = True
#a map is discarded when any parameter's Gelman-Rubin R-hat across its sub-chains exceeds this
R_THRESHOLD = 1_000_000

#the labels of the measured (chain) posterior and of the forecast in file names
EMPIRICAL_LABEL = "empirical"
FISHER_LABEL = "fisher"

def gelman_rubin(chains):
    """Gelman-Rubin R-hat convergence diagnostic across multiple chains."""
    m = len(chains)
    n = min(len(c) for c in chains)
    #trim all chains to same length
    chains = [c[:n] for c in chains]

    chain_means = np.array([np.mean(c) for c in chains])
    chain_vars = np.array([np.var(c, ddof = 1) for c in chains])

    overall_mean = np.mean(chain_means)
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = np.mean(chain_vars)

    var_hat = (n - 1) / n * W + (1 / n) * B
    R_hat = np.sqrt(var_hat / W)
    return R_hat

def integrated_autocorrelation_time(acf, c = 5.0):
    """Sokal's automatic windowing — truncate at lag M where M < c * tau(M)."""
    tau = 1.0
    for m in range(1, len(acf)):
        tau += 2.0 * acf[m]
        if m >= c * tau:
            return tau
    return tau

def autocorrelation(chain, max_lag = None):
    """Normalized autocorrelation of a 1D chain of samples."""
    n = len(chain)
    if max_lag is None:
        max_lag = n // 2
    x = chain - np.mean(chain)
    variance = np.var(chain)
    acf = np.correlate(x, x, mode = "full")[n - 1:]  #positive lags only
    acf = acf[:max_lag] / (variance * n)
    return acf

def all_chain_trace(chains, output_path, param_name):
    plt.figure(figsize = (16, 5))
    for chain in chains:
        plt.plot(range(len(chain)), chain, lw = 0.7)

    len_samps_array = np.array([len(chain) for chain in chains])
    mean_num_samps = np.mean(len_samps_array)
    plt.plot([], [], label = f"Mean Num. Samples = {mean_num_samps}", color = "white")
    std_num_samps = np.std(len_samps_array)
    plt.plot([], [], label = f"Std. Num. Samples = {std_num_samps}", color = "white")
    min_num_samps = np.min(len_samps_array)
    plt.plot([], [], label = f"Min. Num. Samples = {min_num_samps}", color = "white")
    max_num_samps = np.max(len_samps_array)
    plt.plot([], [], label = f"Max. Num. Samples = {max_num_samps}", color = "white")

    plt.xlabel("Iteration"); plt.ylabel("Value")
    plt.title(f"{param_name} Markov Chain Traces")
    plt.legend(loc = "upper left", fontsize = 6, ncol = 2); plt.grid(alpha= 0.2)
    plt.savefig(output_path + f"{param_name.lower()}_all_chain_trace.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    return

def plot_marginals(chains, output_path, ground_truth, param_name):

    all_chains = np.concatenate([chain for chain in chains])
    grid = np.linspace(all_chains.min() - BUFFER, all_chains.max() + BUFFER, GRID_SIZE)

    plt.figure()
    for chain in chains:
        pdf = gaussian_kde(chain)(grid)
        plt.plot(grid, pdf)
    plt.axvline(ground_truth, color = "black", label = f"Ground Truth = {ground_truth}")
    plt.legend()
    plt.xlabel("Count"); plt.title(f"{param_name} Marginals"); plt.grid(alpha = 0.2)
    plt.savefig(output_path + f"{param_name.lower()}_all_chain_marginals.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    return

def significant_digits(value):
    #digits of the shortest decimal that round-trips the float, leading and trailing zeros dropped
    digits = np.format_float_positional(abs(value), trim = "-").replace(".", "")
    digits = digits.lstrip("0").rstrip("0")
    return max(1, len(digits))

def product_pdf(densities, dx):
    """The product of the per-map densities on one grid, renormalized (trapezoid rule) after
    every factor so it never underflows."""
    pdf = densities[0].copy()
    norm = np.sum(((pdf[:-1] + pdf[1:]) * dx) / 2)
    pdf /= norm
    for idx in range(1, len(densities)):
        pdf *= densities[idx]
        norm = np.sum(((pdf[:-1] + pdf[1:]) * dx) / 2)
        pdf /= norm
    return pdf

def product_moments(pdf, grid, dx):
    """Mean and standard deviation of a normalized pdf on a uniform grid."""
    mean = np.dot(pdf, grid * dx[0])
    std = np.sqrt(np.sum(dx[0] * pdf * (grid - mean)**2))
    return mean, std

def bootstrap_product_errors(chains, grid, dx, ground_truth):
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
        densities = []
        for chain in chains:
            resampled = rng.choice(chain, size = len(chain), replace = True)
            densities.append(gaussian_kde(resampled)(grid))
        pdf = product_pdf(np.array(densities), dx)
        boot_means[b], boot_stds[b] = product_moments(pdf, grid, dx)
    boot_z_scores = (ground_truth - boot_means) / boot_stds
    return (np.std(boot_means, ddof = 1), np.std(boot_stds, ddof = 1),
            np.std(boot_z_scores, ddof = 1))

def plot_multiplicative_mean(chains, output_path, ground_truth, param_name):
    """The product distribution's plot. Returns its mean, std and z-score with their bootstrap
    Monte Carlo errors."""

    #the normalized product of the per-map densities (a gaussian_kde for each chain)
    all_chains = np.concatenate([chain for chain in chains])
    grid = np.linspace(all_chains.min() - BUFFER, all_chains.max() + BUFFER, GRID_SIZE)
    densities = np.array([gaussian_kde(chain)(grid) for chain in chains])
    dx = np.diff(grid)

    #--------------------------- multiplicative pdf ---------------------------
    multiplicative_pdf = product_pdf(densities, dx)
    multiplicative_mode = grid[np.argmax(multiplicative_pdf)]
    multiplicative_mean, multiplicative_std = product_moments(multiplicative_pdf, grid, dx)
    product_z_score = (ground_truth - multiplicative_mean) / multiplicative_std
    mean_error, std_error, z_error = bootstrap_product_errors(chains, grid, dx, ground_truth)

    mean_text = standard_notation(multiplicative_mean, mean_error)
    std_text = standard_notation(multiplicative_std, std_error)
    z_text = standard_notation(product_z_score, z_error)

    plt.figure(figsize=(13, 5))
    plt.plot(grid, multiplicative_pdf, label="Combined Product Distribution", color = "blue")
    plt.axvline(ground_truth, label = f"Ground Truth = {ground_truth}", color = "green")

    #multiplicative estimators
    #mode printed to the fewer significant digits of itself and the ground truth
    truth_digits = significant_digits(ground_truth)
    mode_digits = min(significant_digits(multiplicative_mode), truth_digits)
    plt.axvline(multiplicative_mode, label = f"Mult. Mode = {multiplicative_mode:.{mode_digits}g}", color = "orange")
    plt.axvline(multiplicative_mean, label = f"Mult. Mean = {mean_text}", color = "pink")
    plt.axvline(multiplicative_mean + multiplicative_std,
                label = f"+/- 1 Mult. Std = {std_text}", color = "grey")
    plt.axvline(multiplicative_mean - multiplicative_std, color = "grey")

    plt.plot([], [], label = f"Product Z-score = {z_text}")

    #clip the x range to where the pdf is above a fraction of its peak (ground truth kept in view)
    above = grid[multiplicative_pdf > PDF_XLIM_THRESHOLD * multiplicative_pdf.max()]
    x_low = min(above.min(), ground_truth)
    x_high = max(above.max(), ground_truth)
    plt.xlim(x_low, x_high)

    plt.xlabel(f"{param_name} Value"); plt.ylabel("Density")
    plt.title(f"{param_name} Combined Product Distribution")
    plt.legend(fontsize=7.5); plt.grid(alpha=0.2)
    plt.savefig(output_path + f"{param_name.lower()}_mean_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    return (multiplicative_mean, multiplicative_std, product_z_score,
            mean_error, std_error, z_error)

def plot_autocorrelation_chains(chains_per_map, output_path, param_name):
    acfs_per_map = []
    plt.figure()
    for chains in chains_per_map:
        acfs = []
        for chain in chains:
            acf = autocorrelation(chain)
            acfs.append(acf)
            plt.plot(acf)
        acfs_per_map.append(acfs)
    plt.xlabel("Lag"); plt.ylabel("Auto-Correlation")
    plt.axhline(0, color = "black")
    plt.title(f"{param_name} Auto-Correlation")
    plt.savefig(output_path + f"{param_name.lower()}_auto_correlation.png")
    plt.close()
    return acfs_per_map

def plot_iat_per_chain(acfs_per_map, output_path, param_name):
    iats_per_chain = [[integrated_autocorrelation_time(acf) for acf in acfs]
                    for acfs in acfs_per_map]
    flat_iats = [iat for iats in iats_per_chain for iat in iats]
    plt.figure()
    plt.barh(range(len(flat_iats)), flat_iats)
    plt.ylabel("Chain"); plt.xlabel("Integrated Auto-Correlation Time")
    mean_iat = np.mean(flat_iats)
    plt.axvline(mean_iat, label = f"Mean IAT = {mean_iat}", color = "black")
    plt.legend()
    plt.title(f"{param_name} Integrated Auto Correlation Time per Chain")
    plt.savefig(output_path + f"{param_name.lower()}_integrated_autocorrelation_times.png")
    plt.close()
    return iats_per_chain

def plot_mean_histogram(chains, output_path, ground_truth, param_name):

    #mean of each per-map marginal, and their frequentist scatter across realizations
    means = np.array([np.mean(chain) for chain in chains])
    means_mean = np.mean(means)
    means_std = np.std(means, ddof = 1)

    #average posterior width of a single marginal, computed directly from the samples
    avg_sigma_single = np.mean([np.std(chain, ddof = 1) for chain in chains])
    std_sigma_single = np.std([np.std(chain, ddof = 1) for chain in chains])

    plt.figure()
    plt.hist(means, bins = 20, alpha = 0.6, edgecolor = "white")
    plt.axvline(ground_truth, color = "green", label = f"Ground Truth = {ground_truth}")
    plt.axvline(means_mean, color = "indianred", label = f"Mean of Means = {round(means_mean, 7)}")
    plt.axvline(means_mean + means_std, color = "black", label = f"+/- 1 Std of Means = {round(means_std, 7)}")
    plt.axvline(means_mean - means_std, color = "black")
    plt.xlabel("Mean Value"); plt.ylabel("Count")
    #calibration check: std of the Means should roughly match the average single-marginal std
    plt.title(f"{param_name} Histogram of Marginal Means\n"
              f"std(Means) = {round(means_std, 7)} vs avg single-marginal std = {round(avg_sigma_single, 7)}\n"
              f"std in single-marginal std = {round(std_sigma_single, 7)}")
    plt.legend(); plt.grid(alpha = 0.2)
    plt.savefig(output_path + f"{param_name.lower()}_mean_histogram.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    return means_std, avg_sigma_single, std_sigma_single

def prune_chains(raw_chains, iats_per_chain, should_thin = SHOULD_THIN):
    """One array per map: its sub-chains, each thinned by its own integrated autocorrelation
    time (when should_thin) so the retained points are not correlated, concatenated."""
    pruned_chains = []
    for i, chains in enumerate(raw_chains):
        processed_chains = []
        for j, chain in enumerate(chains):
            if should_thin:
                #clamp so an IAT below 1 cannot produce a zero slice step
                stride = max(1, int(iats_per_chain[i][j]))
                processed_chains.append(chain[0:-1:stride])
            else:
                processed_chains.append(chain)
        pruned_chains.append(np.concatenate(processed_chains))
    return pruned_chains

def metropolis_acceptance_rate(raw_chains, param_name):
    """Print the Metropolis-Hastings acceptance rate for one parameter, from the UNTHINNED chains.

    raw_chains is the list over maps of lists of post-burn-in chains (one 1D array per
    chain). A rejected proposal leaves the parameter exactly where it was, so the history
    file repeats the previous value bit for bit; an accepted one moves it. Each chain's rate
    is therefore its fraction of transitions that changed the value, and the printed rate is
    the plain mean of those per-chain rates over every chain of every map. This must run
    AFTER burn-in is cut (the fixed-theta burn-in iterations would all count as rejects) and
    BEFORE prune_chains (thinning by the autocorrelation stride destroys the step-by-step
    structure this counts).
    """
    rates = []
    for chains_per_map in raw_chains:
        for chain in chains_per_map:
            chain = np.asarray(chain)
            moved = chain[1:] != chain[:-1]
            rates.append(np.mean(moved) if moved.size > 0 else float("nan"))
    print(f"{param_name} Metropolis-Hastings acceptance rate = {np.mean(rates):.4f}")
    return

def load_chains_per_map(data_path, param_name, map_idx, num_chains, map_pre_factor,
                        default_burn_in, verbose = False):
    """(chains, num_discarded): the post-burn-in sub-chains of one map for one parameter.
    A sub-chain no longer than the burn-in is discarded."""
    chains, num_discarded = [], 0
    for chain_idx in range(1, num_chains + 1):
        file_path = data_path + f"{param_name}_map_{map_pre_factor * map_idx}_chain_{chain_idx}_history.txt"
        if os.path.exists(file_path):
            array = np.loadtxt(file_path)
            if len(array) > default_burn_in:
                chains.append(array[default_burn_in:])
            else:
                if verbose:
                    print(f"Discarded (Map {map_idx}, Chain {chain_idx}) due to burn-in")
                num_discarded += 1
        else:
            if verbose:
                raise ValueError(f"Tried to load {file_path} but it does not exist")
    return chains, num_discarded

def prune_maps_by_r_hat(file_name, num_maps, num_chains, map_pre_factor, default_burn_in,
                        threshold, param_names):
    print("\n")
    print(f"Beginning map pruning procedure. Discarding maps for which any parameter's average R-hat > {threshold}")
    non_converged_maps = []
    data_path = os.path.join(CHAIN_DATA_DIR, file_name) + os.sep
    for param_name in param_names:
        for map_idx in range(1, num_maps + 1):
            chains_per_map, _ = load_chains_per_map(data_path, param_name, map_idx, num_chains,
                                                    map_pre_factor, default_burn_in)
            #gelman-rubin R-statistic only makes sense if more than one chain per map
            if len(chains_per_map) > 1 and gelman_rubin(chains_per_map) > threshold:
                non_converged_maps.append(map_idx)
    return non_converged_maps

def per_param_analysis(file_name, num_maps, num_chains, map_pre_factor, ground_truth,
                       default_burn_in, param_name, non_converged_maps, plot_dir):
    print("\n")
    print(f"Beginning {param_name} analysis")
    #load the data
    raw_chains = []
    r_hats = []
    data_path = os.path.join(CHAIN_DATA_DIR, file_name) + os.sep
    output_path = os.path.join(plot_dir, param_name.lower()) + os.sep
    os.makedirs(output_path, exist_ok = True)
    num_chains_discarded = 0
    for map_idx in range(1, num_maps + 1):
        if map_idx not in non_converged_maps:
            chains_per_map, num_discarded = load_chains_per_map(
                                            data_path, param_name, map_idx, num_chains, 
                                            map_pre_factor, default_burn_in, verbose = True)
            num_chains_discarded += num_discarded
            #gelman-rubin R-statistic only makes sense if more than one chain per map
            if len(chains_per_map) > 1:
                r_hats.append(gelman_rubin(chains_per_map))
                raw_chains.append(chains_per_map)

    print(f"Discarded {num_chains_discarded} total chains")
    r_hats = np.array(r_hats)
    print(f"{param_name} Unpruned Average Gelman-Rubin R Statistic = {np.mean(r_hats)}")
    print(f"{param_name} Unpruned Std in Gelman-Rubin R Statistic = {np.std(r_hats)}")
    print(f"Using chain data from {len(raw_chains)} total maps before pruning")
    print(f"r_hats = {r_hats}")

    #metropolis-hastings acceptance rate per parameter, averaged over maps and chains. this
    #has to see the chains after burn-in but BEFORE thinning
    metropolis_acceptance_rate(raw_chains, param_name)

    #auto-correlation plots
    acfs_per_map = plot_autocorrelation_chains(raw_chains, output_path, param_name)

    #integrated autocorrelation time
    iats_per_chain = plot_iat_per_chain(acfs_per_map, output_path, param_name)

    #thin the chains by their individual IATs
    chains = prune_chains(raw_chains, iats_per_chain)

    #make an all chain trace plot
    all_chain_trace(chains, output_path, param_name)

    #plot each single map's distribution
    plot_marginals(chains, output_path, ground_truth, param_name)

    #histogram of per-map marginal means
    means_std, avg_sigma_single, std_sigma_single = plot_mean_histogram(chains, output_path,
                                                                        ground_truth, param_name)

    #error on the average single-map width: the scatter of the widths over sqrt(number of maps)
    avg_sigma_error = std_sigma_single / np.sqrt(len(chains))

    #combined product distribution, with its bootstrap Monte Carlo errors
    (product_mean, product_std, product_z_score,
     product_mean_error, product_std_error, product_z_error) = plot_multiplicative_mean(
        chains, output_path, ground_truth, param_name)

    #the numbers of the z_scores.png table
    summary = {"product_mean": product_mean, "product_std": product_std,
               "product_z_score": product_z_score,
               "product_mean_error": product_mean_error, "product_std_error": product_std_error,
               "product_z_error": product_z_error,
               "avg_sigma_single": avg_sigma_single, "avg_sigma_error": avg_sigma_error}
    return raw_chains, summary

def standard_notation(value, error):
    """value(e) in the standard concise notation, value +/- e with e in units of the last shown
    digit: the error rounded to one significant figure, the value rounded at that decimal
    place, e.g. (0.109238, 0.00031) -> 0.1092(3) = 0.1092 +/- 0.0003."""
    decimal = error_decimal(error)
    error_digit = round(float(error) * 10.0**decimal)
    return f"{rounded_text(value, decimal)}({error_digit})"

def plot_z_score_table(summaries, ground_truth_values, plot_dir):
    """z_scores.png in plot_dir: one row per sampled parameter, the ground truth next to its
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
        ground_truth = ground_truth_values[name]
        mean_text = standard_notation(summary["product_mean"], summary["product_mean_error"])
        std_text = standard_notation(summary["product_std"], summary["product_std_error"])
        avg_sigma_text = standard_notation(summary["avg_sigma_single"], summary["avg_sigma_error"])
        z_text = standard_notation(summary["product_z_score"], summary["product_z_error"])

        row_labels.append(f"{param_symbol(name)}  (truth = {ground_truth})")
        cells.append([mean_text, std_text, avg_sigma_text, z_text])

    figure, axis = plt.subplots(figsize = (11, 0.5 * (len(row_labels) + 1)))
    axis.axis("off")
    table = axis.table(cellText = cells, rowLabels = row_labels, colLabels = column_labels,
                       cellLoc = "center", loc = "center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    table.auto_set_column_width(list(range(-1, len(column_labels))))
    path = plot_dir + "z_scores.png"
    figure.savefig(path, dpi = 150, bbox_inches = "tight")
    plt.close(figure)
    return

def plot_annotated_heat_map(matrix, names, title, file_name, path):
    figure, axis = plt.subplots(figsize = (7.5, 6.5))
    annotated_heatmap(axis, matrix, names, title, "Value")
    figure.tight_layout()
    figure.savefig(path + file_name, dpi = 150)
    plt.close(figure)
    return

def measured_errors(per_map_cov_mats):
    """(cov_err, sigma_err): the delete-one-MAP jackknife of the map-averaged covariance.

    With C_(m) the average leaving out map m, the widths are recomputed as
    sigma_(m) = sqrt(C_(m),ii) and
        err_sigma^2 = (M - 1) / M * sum_m (sigma_(m) - mean_m sigma_(m))^2 ,
    the jackknife of the widths themselves (no linear propagation from C). err_C is the same
    jackknife applied to each covariance element, for the dashed +/- contours and the error
    on the off-diagonal measured / forecast covariance ratios.
    """
    num_maps = len(per_map_cov_mats)
    total = np.sum(per_map_cov_mats, axis = 0)
    leave_one_out = np.array([(total - cov_mat) / (num_maps - 1)
                              for cov_mat in per_map_cov_mats])
    factor = (num_maps - 1) / num_maps
    cov_err = np.sqrt(factor * np.sum((leave_one_out - np.mean(leave_one_out, axis = 0))**2,
                                      axis = 0))
    sigmas_loo = np.sqrt(np.array([np.diag(cov_mat) for cov_mat in leave_one_out]))
    sigma_err = np.sqrt(factor * np.sum((sigmas_loo - np.mean(sigmas_loo, axis = 0))**2,
                                        axis = 0))
    return cov_err, sigma_err

def plot_covariance_and_correlation(cov_mat, param_names, label, plot_dir, title = None):
    """covariance_matrix_<label>.png and correlation_matrix_<label>.png (every entry printed
    in its box) for one posterior: the chains' (EMPIRICAL_LABEL) or a forecast's. `title`
    names the posterior in the plot titles (`label` when None)."""
    title = label if title is None else title
    plot_annotated_heat_map(cov_mat, param_names, f"{title}\ncovariance matrix",
                            f"covariance_matrix_{label}.png", plot_dir)
    plot_annotated_matrix(correlation_from_covariance(cov_mat), param_names,
                          f"{title}\ncorrelation matrix",
                          plot_dir + f"correlation_matrix_{label}.png",
                          norm = matplotlib.colors.Normalize(vmin = -1, vmax = 1))

def get_avg_covariance_matrix(raw_chains):
    """Map-averaged posterior covariance of the sampled parameters, with its jackknife error.

    raw_chains maps each parameter name to its list over maps of lists of sub-chains (the
    output of per_param_analysis). Every sampled parameter is written to its history file
    once per iteration, and per_param_analysis walks maps and chains in the same order for
    each parameter, so the columns of one map line up row-by-row.

    Each map's sub-chains are concatenated and give one covariance matrix, and those are
    averaged over maps. Samples from different maps are never pooled: each map is a
    different data realization with its own posterior mean, so pooling would add the scatter
    of those means to the diagonal. The average is a single-realization posterior, which is what a
    Fisher forecast predicts.

    Returns (avg_cov_mat, param_names, cov_err, sigma_err), the last two from
    measured_errors: cov_err the element-wise error of the averaged covariance (the dashed
    contours and the off-diagonal ratio errors) and sigma_err the error of each width
    sqrt(C_ii).
    """
    param_names = list(raw_chains.keys())
    num_maps = min(len(raw_chains[name]) for name in param_names)
    per_map_cov_mats = []
    for map_idx in range(num_maps):
        columns = [np.concatenate(raw_chains[name][map_idx]) for name in param_names]
        #clip to the shortest column, in case the chains were copied while still running
        min_length = min(len(column) for column in columns)
        columns = [column[:min_length] for column in columns]
        per_map_cov_mats.append(np.cov(np.column_stack(columns), rowvar = False))

    per_map_cov_mats = np.array(per_map_cov_mats)
    avg_cov_mat = np.mean(per_map_cov_mats, axis = 0)
    cov_err, sigma_err = measured_errors(per_map_cov_mats)
    return avg_cov_mat, param_names, cov_err, sigma_err

#Contour styling for the triangle plots, one per forecast: the chains always blue, the
#forecast always red, each contour filled translucently under an opaque outline so the
#overlap stays legible
#dark shades, so the coloured annotation text reads clearly on white
MEASURED_COLOUR = "blue"
FORECAST_COLOUR = "red"
#face alpha of the filled contours / Gaussians
FILL_ALPHA = 0.18
#size of one triangle-plot panel, in inches (square panels, so the ellipses are not
#stretched horizontally)
PANEL_WIDTH = 3.3
PANEL_HEIGHT = 3.3
ANNOTATION_FONTSIZE = 12
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10
#the parameters' symbols in the triangle plots (mathtext, without the enclosing $)
PARAM_SYMBOLS = {"theta_MC_100": "100*\\theta_{\\rm MC}",
                 "logA": "\\ln(A_s*10^{\\rm 10})",
                 "ns": "n_s",
                 "ombh2": "\\Omega_b h^2",
                 "omch2": "\\Omega_c h^2"}

def forecast_title(t_block, phi_noise_source, phi_noise, qe_filter, qe_response):
    """The forecast's name in plot titles: where its N_phi and its T block come from."""
    if phi_noise_source == "qe":
        nphi_text = f"QE ({qe_filter} filter, {qe_response} response)"
    else:
        nphi_text = f"measured ({os.path.basename(phi_noise)})"
    if t_block == "measured":
        t_text = "measured delensed"
    else:
        t_text = f"CAMB {t_block}"
    return f"N_phi: {nphi_text}\nT block: {t_text}"


#the data each pol setting of the sampler sees, for the triangle plots' subtitle
POL_DESCRIPTIONS = {"I": "Temperature Data",
                    "P": "Polarization Data",
                    "IP": "Temperature + Polarization Data"}

def param_symbol(name):
    """A parameter's mathtext symbol, enclosed in $, or its plain name if it has none."""
    if name in PARAM_SYMBOLS:
        return f"${PARAM_SYMBOLS[name]}$"
    return name

def error_decimal(error):
    """The decimal place of the first significant digit of error once it is rounded to one
    significant figure, e.g. 0.0008465 -> 4, 0.00096 (-> 0.001) -> 3, 23 -> -1."""
    decimal = -decimal_exponent(error)
    rounded = round(float(error), decimal)
    return -decimal_exponent(rounded)

def rounded_text(value, decimal):
    """value rounded at the given decimal place (negative: to tens, hundreds, ...)."""
    return f"{round(float(value), decimal):.{max(decimal, 0)}f}"

def format_scaled(value, exponent, decimal, error = None):
    """value (+/- error) as one mathtext mantissa times 10^exponent, rounded at the raw
    decimal place `decimal` (error_decimal of the jackknife error), e.g. 0.003268 +/- 0.0008465
    at exponent -3 -> (3.3 +/- 0.8) x 10^-3, so a measured and a forecast number written at
    the same exponent can be read against each other digit by digit."""
    scale = 10.0**exponent
    mantissa_decimal = decimal + exponent
    if error is None:
        return f"${rounded_text(value / scale, mantissa_decimal)}\\times 10^{{{exponent}}}$"
    return (f"$({rounded_text(value / scale, mantissa_decimal)} \\pm "
            f"{rounded_text(error / scale, mantissa_decimal)})\\times 10^{{{exponent}}}$")

def ratio_text(ratio, error):
    """ratio +/- error, the error (jackknife error / fisher value) to its first significant
    figure and the ratio rounded at the same decimal place, e.g. 1.017 +/- 0.0081 ->
    1.017 +/- 0.008, 0.72 +/- 0.071 -> 0.72 +/- 0.07."""
    decimal = error_decimal(error)
    return f"{rounded_text(ratio, decimal)} $\\pm$ {rounded_text(error, decimal)}"

def decimal_exponent(value):
    return int(np.floor(np.log10(abs(value))))

def stacked_text(ax, lines, x, y_top, line_step, ha = "left"):
    """Write (text, colour) lines downward from y_top in axes coordinates. The text is kept
    out of constrained layout: it is placed inside the cells, and counting it as a
    decoration collapses the axes."""
    for k, (text, colour) in enumerate(lines):
        ax.text(x, y_top - k * line_step, text, color = colour, transform = ax.transAxes,
                fontsize = ANNOTATION_FONTSIZE, ha = ha, va = "top", in_layout = False)
    return


def triangle_plot(measured_cov_mat, forecast_cov_mat, param_names, means,
                  measured_cov_sigma, measured_sigma_err, plot_dir,
                  sigmas = (1,), nside = None, theta_pix = None, noise = None, pol = None):
    """Triangle plot of the sampled posterior (blue) against the Fisher forecast (red):
    triangle_plot.png in plot_dir.

    Both are filled at FILL_ALPHA under an opaque outline, fills drawn beneath all outlines so
    the overlap stays legible. The diagonal carries each parameter's marginal Gaussian, the
    lower triangle each pair's joint confidence ellipse. sigmas lists which contours to draw
    (1-sigma only by default); each level k gives semi-axes k * sqrt(eigenvalue) of the
    pair's 2x2 covariance block.

    Everything is centred on the MEASURED means. A forecast is a curvature, not a location
    - it has a shape but no centre - so a common centre makes the plot a direct read of
    width and orientation, which is the only thing these matrices can be compared on. Each
    diagonal panel is annotated with sigma_measured +/- its error, sigma_fisher and the
    ratio sigma_measured / sigma_fisher: above 1 means the sampler is doing worse than the forecast says the data
    allows, near 1 means it is saturating it, below 1 means it is extracting information the
    forecast does not model (or the chain has not converged). Each vacant upper-triangle cell
    (row a, column b) reports the off-diagonal covariance C_ab of the lower panel it mirrors
    (row b, column a), measured and forecast, and their ratio C_measured / C_fisher. Unlike
    a degeneracy angle this does not depend on how the axes are scaled.

    `forecast_cov_mat` MUST already be permuted into param_names order - see
    get_forecasted_covariance, which does it. The forecast is exact given its model and
    carries no error bars; only the measured posterior does, all from the delete-one-map
    jackknife of measured_errors: `measured_cov_sigma` (the element-wise error of the
    averaged covariance) adds dashed (unfilled) blue lower / upper limits at
    measured_cov_mat -/+ measured_cov_sigma and is the error of every off-diagonal ratio,
    and `measured_sigma_err` is each width's error and that of the width ratio.
    """
    measured = np.asarray(measured_cov_mat)
    forecast = np.asarray(forecast_cov_mat)
    measured_cov_sigma = np.asarray(measured_cov_sigma)
    means = np.asarray(means)
    n = len(param_names)

    #(covariance, colour, filled, linestyle) - the central contours filled, the measured
    #+/- error limits as dashed outlines only
    layers = [(measured, MEASURED_COLOUR, True, "-"),
              (measured - measured_cov_sigma, MEASURED_COLOUR, False, "--"),
              (measured + measured_cov_sigma, MEASURED_COLOUR, False, "--"),
              (forecast, FORECAST_COLOUR, True, "-")]

    #pad the axes out past the widest requested contour of the WIDER matrix, so whichever
    #one is broader still fits
    pad = max(sigmas) + 0.8
    widest = np.maximum(np.sqrt(np.diag(measured + measured_cov_sigma)),
                        np.sqrt(np.diag(forecast)))
    limits = [(means[i] - pad * widest[i], means[i] + pad * widest[i]) for i in range(n)]

    #annotation line spacing in axes fraction: one text line over the panel's height
    #(constrained layout leaves the axes ~80% of their cell; mathtext sub / superscripts
    #make the lines taller than the font size)
    line_step = 1.5 * ANNOTATION_FONTSIZE / (0.8 * PANEL_HEIGHT * 72)
    #the diagonal's three-line text block sits above the Gaussians; raise the y limit so the
    #unit-height curves stay entirely below it
    text_fraction = 0.04 + 3 * line_step
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
                cov_fisher = forecast[i, j]
                ratio = cov_measured / cov_fisher
                ratio_error = cov_error / abs(cov_fisher)
                exponent = decimal_exponent(cov_measured)
                #both covariances rounded at the first significant digit of the jackknife error
                decimal = error_decimal(cov_error)
                lines = [(f"Cov({param_symbol(param_names[i])}, {param_symbol(param_names[j])})", "black"),
                         (f"$C_{{\\rm measured}}$ = {format_scaled(cov_measured, exponent, decimal, cov_error)}", MEASURED_COLOUR),
                         (f"$C_{{\\rm fisher}}$ = {format_scaled(cov_fisher, exponent, decimal)}", FORECAST_COLOUR),
                         (f"$C_{{\\rm measured}}/C_{{\\rm fisher}}$ = {ratio_text(ratio, ratio_error)}", "black")]
                y_top = 0.5 + 0.5 * len(lines) * line_step
                stacked_text(ax, lines, 0.5, y_top, line_step, ha = "center")
                continue
            for cov_mat, colour, filled, linestyle in layers:
                if i == j:
                    sigma = np.sqrt(cov_mat[i, i])
                    grid = np.linspace(*limits[i], 500)
                    curve = np.exp(-0.5 * ((grid - means[i]) / sigma) ** 2)
                    if filled:
                        ax.fill_between(grid, 0, curve, color = colour, alpha = FILL_ALPHA,
                                        lw = 0, zorder = 1)
                    ax.plot(grid, curve, color = colour, ls = linestyle, lw = 1.6, zorder = 3)
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
                            lw = 1.6, zorder = 3))
            if i == j:
                #sigma_measured +/- its error, the forecast's sigma and the ratio, in this
                #parameter's own units
                sigma_measured = np.sqrt(measured[i, i])
                error_measured = measured_sigma_err[i]
                sigma_fisher = np.sqrt(forecast[i, i])
                exponent = decimal_exponent(sigma_measured)
                #both widths rounded at the first significant digit of the jackknife error
                decimal = error_decimal(error_measured)
                #the colour says which is measured (blue) and which forecast (red)
                lines = [(f"$\\sigma_{{\\rm measured}}$ = {format_scaled(sigma_measured, exponent, decimal, error_measured)}", MEASURED_COLOUR),
                         (f"$\\sigma_{{\\rm fisher}}$ = {format_scaled(sigma_fisher, exponent, decimal)}", FORECAST_COLOUR),
                         (f"$\\sigma_{{\\rm measured}}/\\sigma_{{\\rm fisher}}$ = {ratio_text(sigma_measured / sigma_fisher, error_measured / sigma_fisher)}", "black")]
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
    def filled_handle(colour, linestyle, label):
        return Patch(facecolor = to_rgba(colour, FILL_ALPHA), edgecolor = colour,
                     ls = linestyle, lw = 1.6, label = label)
    handles = [filled_handle(MEASURED_COLOUR, "-", "Sampler Data"),
               Line2D([0], [0], color = MEASURED_COLOUR, lw = 2, ls = "--",
                      label = "Sampler Data $\\pm$ 1 Jackknife"),
               filled_handle(FORECAST_COLOUR, "-", "Fisher")]
    #the upper triangle holds the covariances, so the legend goes below the grid, where
    #constrained layout reserves room for it (above, it collides with the suptitle)
    fig.legend(handles = handles, loc = "outside lower center", fontsize = 12, ncol = 3)
    title = "Sampler Data v.s. Fisher Forecast"
    if None not in (nside, theta_pix, noise, pol):
        title += (f"\n{POL_DESCRIPTIONS[pol]}, {noise:g}$\\mu$K-arcmin, {theta_pix:g}' pixels, "
                  f"{nside}$^2$ pixels")
    fig.suptitle(title, fontsize = LABEL_FONTSIZE)
    plt.savefig(plot_dir + "triangle_plot.png", dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    return

def get_means_per_param(all_chains, param_names):
    """The mean of every post-burn-in sample of each parameter, over all maps and chains."""
    return [np.mean(np.concatenate([chain for chains in all_chains[name] for chain in chains]))
            for name in param_names]

def get_forecasted_covariance(nside, theta_pix, noise, is_sampled, param_names,
                              t_block = DEFAULT_T_BLOCK, delensed_covariance = None,
                              phi_noise_source = DEFAULT_PHI_NOISE_SOURCE, phi_noise = None,
                              qe_filter = DEFAULT_QE_FILTER, qe_response = DEFAULT_QE_RESPONSE):
    """Fisher-forecast covariance for the sampled parameters, reordered to match param_names.
    The arguments are main's.

    forecast() returns its parameters in fisher_forecast's OUTPUT_PARAM_ORDER, while
    param_names follows the chains' (GROUND_TRUTH's) order, so the covariance is permuted here.
    """
    if phi_noise_source == "qe":
        nphi_source = f"qe_{qe_filter}"
    else:
        nphi_source = FILE_NPHI_SOURCE
    fisher, names = forecast(nside, theta_pix, noise, is_sampled, GROUND_TRUTH,
                             nphi_source = nphi_source, qe_response = qe_response,
                             phi_noise = phi_noise, t_block = t_block,
                             delensed_covariance = delensed_covariance,
                             camb_lmax = FORECAST_CAMB_LMAX, l_knee = 0, beam_fwhm = 0)
    covariance = covariance_from_fisher(fisher, names)
    order = [names.index(name) for name in param_names]
    return covariance[np.ix_(order, order)]

def joint_param_analysis(all_chains, nside, theta_pix, noise, is_sampled, pol, joint_dir,
                         t_block, delensed_covariance, phi_noise_source, phi_noise,
                         qe_filter, qe_response):
    """The measured posterior against the Fisher forecast, into joint_dir: the empirical and
    fisher covariance / correlation matrices, their fisher-over-empirical ratios and the
    triangle plot."""

    print("\n")
    print(f"Beginning joint parameter analysis")
    (measured_cov_mat, param_names, measured_cov_sigma,
     measured_sigma_err) = get_avg_covariance_matrix(all_chains)
    means = get_means_per_param(all_chains, param_names)
    forecasted_cov_mat = get_forecasted_covariance(nside, theta_pix, noise, is_sampled,
                                                   param_names, t_block = t_block,
                                                   delensed_covariance = delensed_covariance,
                                                   phi_noise_source = phi_noise_source,
                                                   phi_noise = phi_noise,
                                                   qe_filter = qe_filter,
                                                   qe_response = qe_response)
    title = forecast_title(t_block, phi_noise_source, phi_noise, qe_filter, qe_response)

    plot_covariance_and_correlation(measured_cov_mat, param_names, EMPIRICAL_LABEL, joint_dir,
                                    title = "Chains (empirical)")
    plot_covariance_and_correlation(forecasted_cov_mat, param_names, FISHER_LABEL, joint_dir,
                                    title = f"Forecast\n{title}")
    #per-entry ratios, coloured about 1 (agreement)
    ratio_label = f"{FISHER_LABEL}_over_{EMPIRICAL_LABEL}"
    plot_annotated_matrix(forecasted_cov_mat / measured_cov_mat, param_names,
                          f"Forecast / chains covariance\n{title}",
                          joint_dir + f"covariance_ratio_{ratio_label}.png",
                          norm = matplotlib.colors.CenteredNorm(vcenter = 1))
    plot_annotated_matrix(correlation_from_covariance(forecasted_cov_mat) /
                          correlation_from_covariance(measured_cov_mat), param_names,
                          f"Forecast / chains correlation\n{title}",
                          joint_dir + f"correlation_ratio_{ratio_label}.png",
                          norm = matplotlib.colors.CenteredNorm(vcenter = 1))
    triangle_plot(measured_cov_mat, forecasted_cov_mat, param_names, means, measured_cov_sigma,
                  measured_sigma_err, joint_dir, nside = nside, theta_pix = theta_pix,
                  noise = noise, pol = pol)
    return

def main(file_name, num_maps, num_chains, map_pre_factor, ground_truth_values,
         was_sampled, default_burn_in, nside, theta_pix, noise, pol,
         t_block = DEFAULT_T_BLOCK, delensed_covariance = None,
         phi_noise_source = DEFAULT_PHI_NOISE_SOURCE, phi_noise = None,
         qe_filter = DEFAULT_QE_FILTER, qe_response = DEFAULT_QE_RESPONSE):
    """Every plot goes under OUTPUT_DIR: one subfolder per sampled parameter and
    JOINT_DIR_NAME/ (z_scores.png and the chains-vs-forecast plots).

    The forecast's two blocks:
    t_block:             "unlensed" / "lensed" (CAMB's C_TT), "delensed" (CAMB's analytic
                         delensing at the N_phi below) or "measured" (needs
                         delensed_covariance, a merged delensed_covariance.npz)
    phi_noise_source:    "qe" (the box QE N0 with qe_filter "unlensed" / "delensed" and
                         qe_response "unlensed" / "gradient") or "measured" (needs phi_noise,
                         a merged map_joint_phi_noise.npz / sampler_phi_noise.npz)
    The files must be measured at this run's box and noise level."""
    chain_plot_dir = OUTPUT_DIR + os.sep
    joint_dir = os.path.join(OUTPUT_DIR, JOINT_DIR_NAME) + os.sep
    os.makedirs(joint_dir, exist_ok = True)

    sampled_names = [name for name in ground_truth_values if was_sampled[name]]
    non_converged_maps = prune_maps_by_r_hat(file_name, num_maps, num_chains, map_pre_factor,
                                             default_burn_in, R_THRESHOLD,
                                             list(ground_truth_values))

    #single parameter analyses
    all_chains = {}
    summaries = {}
    for name in sampled_names:
        all_chains[name], summaries[name] = per_param_analysis(file_name, num_maps, num_chains,
                                                               map_pre_factor,
                                                               ground_truth_values[name],
                                                               default_burn_in, name,
                                                               non_converged_maps, chain_plot_dir)

    #the table of product means, widths and z-scores
    plot_z_score_table(summaries, ground_truth_values, joint_dir)

    #the posterior against the Fisher forecast
    # if len(sampled_names) > 1:
    #     joint_param_analysis(all_chains, nside, theta_pix, noise, was_sampled, pol, joint_dir,
    #                          t_block, delensed_covariance, phi_noise_source, phi_noise,
    #                          qe_filter, qe_response)
    return

if __name__ == "__main__":

    #the Fisher forecast the triangle plot compares against is only meaningful at the box and
    #noise level the chains ran on - MATCH sample_lcdm.sh
    nside = 128
    theta_pix = 2.5
    noise = 10
    #"I", "P" or "IP", as pol in sample_lcdm.sh; only labels the triangle plots
    pol = "I"
    num_maps = 10
    num_chains = 5
    default_burn_in = 600
    map_pre_factor = 234567
    # file_name = "joint_inference_2026_10_29_2.5uk"
    # file_name = "joint_inference_2026_10_29_5uk"
    file_name = "joint_inference_2026_10_29_10uk"
    # file_name = "omch2_logA_theta_MC_100_joint_inference_09_02_26_low_noise"
    # file_name = "omch2_logA_theta_MC_100_joint_inference_08_27_26"
    # file_name = "omch2_logA_theta_MC_100_joint_inference_09_02_26_high_noise"

    was_sampled = {}
    was_sampled["omch2"] = True
    was_sampled["ombh2"] = False
    was_sampled["ns"] = False
    was_sampled["theta_MC_100"] = True
    was_sampled["logA"] = True

    #the forecast, at THIS box and noise level. T block: "unlensed", "lensed", "delensed" or
    #"measured" (with delensed_covariance). Phi noise: "qe" (with qe_filter / qe_response) or
    #"measured" (with phi_noise). A measured delensed covariance's W should be built from the
    #same N_phi the forecast uses
    t_block = "measured"
    phi_noise_source = "qe"
    qe_filter = "unlensed"
    qe_response = "unlensed"
    covariance_data = os.path.join(REPO_DIR, "get_delensed_field_covariance",
                                   "delensed_field_covariance_data")
    delensed_covariance = os.path.join(covariance_data,
                          f"delensed_covariance_output_qe_filter_unlensed_response_unlensed_{noise}uk",
                          "delensed_covariance.npz")
    phi_noise = None
    #e.g. map_joint's N_phi with its matching delensed covariance (phi_noise_source = "measured"):
    # phi_noise = os.path.join(REPO_DIR, "measure_phi_noise", "get_map_joint_phi_noise",
    #                          "map_joint_phi_noise", f"map_joint_phi_noise_output_{noise}uk",
    #                          "map_joint_phi_noise.npz")
    # delensed_covariance = os.path.join(covariance_data,
    #                       f"delensed_covariance_output_map_joint_{noise}uk",
    #                       "delensed_covariance.npz")

    main(file_name, num_maps, num_chains, map_pre_factor,
         GROUND_TRUTH, was_sampled, default_burn_in, nside, theta_pix, noise, pol,
         t_block = t_block, delensed_covariance = delensed_covariance,
         phi_noise_source = phi_noise_source, phi_noise = phi_noise,
         qe_filter = qe_filter, qe_response = qe_response)
