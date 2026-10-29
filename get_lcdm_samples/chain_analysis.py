import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import jax.numpy as jnp
from cmb_lensing.util import gen_ell_grid
from cmb_lensing.simulate import covar_matrix_from_cls
from cmb_lensing.constants import GROUND_TRUTH, DEFAULT_ELL_CUTOFF
from cmb_lensing.fisher_forecast import (annotated_heatmap, plot_annotated_matrix,
                                         correlation_from_covariance, covariance_from_fisher,
                                         forecast, FILE_NPHI_SOURCES, fourier_weights,
                                         camb_cls_at_params, iterative_delensing,
                                         qe_noise_matrix, prior_covariances)
from cmb_lensing.delensed_covariance import DEFAULT_DELTA_ELL, band_edges, band_average

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HERE)
CHAIN_DATA_DIR = os.path.join(HERE, "lcdm_sampler_data")
CHAIN_PLOT_ROOT = os.path.join(HERE, "lcdm_chain_plots")

#the Fisher forecasts drawn against the chains, by fisher_forecast nphi_source: the two
#bounds and the two measured N_phi (drawn only when main's forecast_phi_noise has a file for
#them). Each is forecast with CAMB's delensed T block ("camb_delensed") and, when main's
#forecast_delensed_covariance has a file for it, with the empirical one ("empirically_delensed")
FORECAST_NPHI_SOURCES = ("qe_unlensed", "qe_delensed", "map_joint", "sample_joint")

#the QE response spectrum per qe_ source ("unlensed" when absent)
FORECAST_QE_RESPONSE = {"qe_delensed": "gradient"}

#the two temperature blocks every forecast is built with: folder name -> description
T_BLOCKS = {"camb_delensed": "CAMB delensed T",
            "empirically_delensed": "empirically delensed T"}

#how far CAMB is run for the forecasts (None: the rfft grid's corner mode)
FORECAST_CAMB_LMAX = 4000
BUFFER = 1e-4
GRID_SIZE = 1000
#the multiplicative-mean plot's x range ends where the pdf drops below this fraction of its peak
PDF_XLIM_THRESHOLD = 1e-3

#thin each chain by its integrated autocorrelation time before the per-parameter plots
SHOULD_THIN = True
#a map is discarded when any parameter's Gelman-Rubin R-hat across its sub-chains exceeds this
R_THRESHOLD = 1_000_000

#the label of the measured (chain) posterior in file names
EMPIRICAL_LABEL = "empirical"

def plot_dirs(noise):
    """(chain plot dir, spectra plot dir, fisher plot dir) of one noise level, all ending in
    os.sep: lcdm_chain_plots/noise_level_<noise>uk/ (the per-parameter folders) and its
    spectra_plots/ and fisher/ subfolders. fisher/ holds the chains' covariance and
    correlation matrices and one <nphi_source>/<t_block>/ folder per forecast
    (forecast_plot_dir)."""
    chain_dir = os.path.join(CHAIN_PLOT_ROOT, f"noise_level_{noise}uk")
    return (chain_dir + os.sep, os.path.join(chain_dir, "spectra_plots") + os.sep,
            os.path.join(chain_dir, "fisher") + os.sep)

def forecast_plot_dir(fisher_dir, nphi_source, t_block):
    """fisher/<nphi_source>/<t_block>/ (created): one forecast's triangle plot, covariance,
    correlation and covariance-ratio plots."""
    path = os.path.join(fisher_dir, nphi_source, t_block) + os.sep
    os.makedirs(path, exist_ok = True)
    return path

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

def plot_multiplicative_mean(chains, avg_sigma_single, output_path, ground_truth, param_name):

    #the normalized product of the per-map densities (a gaussian_kde for each chain)
    all_chains = np.concatenate([chain for chain in chains])
    grid = np.linspace(all_chains.min() - BUFFER, all_chains.max() + BUFFER, GRID_SIZE)
    densities = np.array([gaussian_kde(chain)(grid) for chain in chains])
    dx = np.diff(grid)

    #--------------------------- multiplicative pdf ---------------------------
    multiplicative_pdf = densities[0]
    norm = np.sum(((multiplicative_pdf[:-1] + multiplicative_pdf[1:]) * dx)/2)
    multiplicative_pdf /= norm
    for idx in range(1, len(densities)):
        multiplicative_pdf *= densities[idx]
        norm = np.sum(((multiplicative_pdf[:-1] + multiplicative_pdf[1:]) * dx)/2)
        multiplicative_pdf /= norm

    multiplicative_mode = grid[np.argmax(multiplicative_pdf)]
    multiplicative_mean = np.dot(multiplicative_pdf, grid * dx[0])
    multiplicative_std = np.sqrt(np.sum(dx[0] * multiplicative_pdf * (grid - multiplicative_mean)**2))

    plt.figure(figsize=(13, 5))
    plt.plot(grid, multiplicative_pdf, label="Combined Product Distribution", color = "blue")
    plt.axvline(ground_truth, label = f"Ground Truth = {ground_truth}", color = "green")

    #multiplicative estimators
    #mode and mean printed to the fewer significant digits of themselves and the ground truth
    truth_digits = significant_digits(ground_truth)
    mode_digits = min(significant_digits(multiplicative_mode), truth_digits)
    mean_digits = min(significant_digits(multiplicative_mean), truth_digits)
    plt.axvline(multiplicative_mode, label = f"Mult. Mode = {multiplicative_mode:.{mode_digits}g}", color = "orange")
    plt.axvline(multiplicative_mean, label = f"Mult. Mean = {multiplicative_mean:.{mean_digits}g}", color = "pink")
    plt.axvline(multiplicative_mean + multiplicative_std,
                label = f"+/- 1 Mult. Std = {multiplicative_std:.2g}", color = "grey")
    plt.axvline(multiplicative_mean - multiplicative_std, color = "grey")

    product_z_score = (ground_truth - multiplicative_mean)/multiplicative_std
    plt.plot([], [], label = f"Product Z-score = {product_z_score:.2f}")

    std_err_z_score = (ground_truth - multiplicative_mean)/avg_sigma_single
    plt.plot([], [], label = f"Std. Err. Z-score = {std_err_z_score:.2f}")

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
    return

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
    return avg_sigma_single

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
    avg_sigma_single = plot_mean_histogram(chains, output_path, ground_truth, param_name)

    #combined product distribution
    plot_multiplicative_mean(chains, avg_sigma_single, output_path, ground_truth, param_name)

    return raw_chains

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
    plot_annotated_heat_map(cov_mat, param_names, f"{title} covariance matrix",
                            f"covariance_matrix_{label}.png", plot_dir)
    plot_annotated_matrix(correlation_from_covariance(cov_mat), param_names,
                          f"{title} correlation matrix",
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
    print(f"Averaged {num_maps} covariance matrices, one per map")
    print(f"Jackknife error (over maps) of the averaged covariance elements, "
          f"err_covar / |covar|:")
    print(cov_err / np.abs(avg_cov_mat))
    print(f"Measured widths (order {param_names}): {np.sqrt(np.diag(avg_cov_mat))} "
          f"+/- {sigma_err} (jackknife)")
    print("Average Correlation matrix (rows / columns ordered as", param_names, ")")
    print(correlation_from_covariance(avg_cov_mat))
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
#per forecast (fisher_forecast's nphi_source): the short symbol in the annotations and the
#legend / title description
FORECAST_LABELS = {
    #the two reported bounds: N_phi from the box QE with the unlensed C_TT in its filter
    #(upper bound on the information) and with the iterated partially delensed C_TT (lower)
    "qe_unlensed": ("QE unlensed", "upper bound (QE, unlensed filter)"),
    "qe_delensed": ("QE delelensed", "lower bound (QE, delensed filter)"),
    "map_joint": ("map_joint", "map_joint N_phi"),
    "sample_joint": ("true_sampler", "sampler N_phi"),
}

def forecast_labels(nphi_source):
    """(short symbol, description) of a forecast's N_phi source."""
    return FORECAST_LABELS.get(nphi_source, ("fisher", nphi_source))

def forecast_file_label(nphi_source, t_block):
    """One forecast's file-name label: <nphi_source>_<t_block>."""
    return f"{nphi_source}_{t_block}"

def forecast_title(nphi_source, t_block):
    """One forecast's name in plot titles: its N_phi source and its T block."""
    return f"N_phi: {nphi_source}, T block: {T_BLOCKS[t_block]}"


#the data each pol setting of the sampler sees, for the triangle plots' subtitle
POL_DESCRIPTIONS = {"I": "Temperature Data",
                    "P": "Polarization Data",
                    "IP": "Temperature + Polarization Data"}

def param_symbol(name):
    """A parameter's mathtext symbol, enclosed in $, or its plain name if it has none."""
    if name in PARAM_SYMBOLS:
        return f"${PARAM_SYMBOLS[name]}$"
    return name

def format_scaled(value, exponent, error = None):
    """value (+/- error) as one mathtext mantissa times 10^exponent, e.g.
    (3.268 +/- 0.043) x 10^-3, so a measured and a forecast number written at the same
    exponent can be read against each other digit by digit."""
    scale = 10.0**exponent
    if error is None:
        return f"${value / scale:.3f}\\times 10^{{{exponent}}}$"
    return f"$({value / scale:.3f} \\pm {error / scale:.3f})\\times 10^{{{exponent}}}$"

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


def triangle_plot(measured_cov_mat, nphi_source, t_block, forecast_cov_mat, param_names, means,
                  measured_cov_sigma, measured_sigma_err, plot_dir,
                  sigmas = (1,), nside = None, theta_pix = None, noise = None, pol = None):
    """Triangle plot of the sampled posterior (blue) against ONE Fisher forecast (red), the
    one with N_phi from `nphi_source` and the `t_block` temperature block (a T_BLOCKS key):
    triangle_plot_<nphi_source>_<t_block>.png in plot_dir.

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
    forecast_label = forecast_file_label(nphi_source, t_block)

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

    print(f"Off-diagonal covariances vs {forecast_label} (measured +/- delete-one-map "
          f"jackknife):")
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
                lines = [(f"Cov({param_symbol(param_names[i])}, "
                          f"{param_symbol(param_names[j])})", "black"),
                         (f"$C_{{\\rm measured}}$ = {format_scaled(cov_measured, exponent, cov_error)}",
                          MEASURED_COLOUR),
                         (f"$C_{{\\rm fisher}}$ = {format_scaled(cov_fisher, exponent)}", FORECAST_COLOUR),
                         (f"$C_{{\\rm measured}}/C_{{\\rm fisher}}$ = {ratio:.3f} $\\pm$ "
                          f"{ratio_error:.3f}", "black")]
                print(f"    {param_names[i]}, {param_names[j]}: measured {cov_measured:+.4e} "
                      f"+/- {cov_error:.2e}  {forecast_label} {cov_fisher:+.4e}  "
                      f"measured / {forecast_label} = {ratio:.3f} +/- {ratio_error:.3f}")
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
                #the colour says which is measured (blue) and which forecast (red)
                lines = [(f"$\\sigma_{{\\rm measured}}$ = {format_scaled(sigma_measured, exponent, error_measured)}",
                          MEASURED_COLOUR),
                         (f"$\\sigma_{{\\rm fisher}}$ = {format_scaled(sigma_fisher, exponent)}", FORECAST_COLOUR),
                         (f"$\\sigma_{{\\rm measured}}/\\sigma_{{\\rm fisher}}$ = "
                          f"{sigma_measured / sigma_fisher:.3f} $\\pm$ "
                          f"{error_measured / sigma_fisher:.3f}", "black")]
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
    plt.savefig(plot_dir + f"triangle_plot_{forecast_label}.png", dpi = 150,
                bbox_inches = "tight")
    plt.close(fig)
    return

def get_means_per_param(all_chains, param_names):
    """The mean of every post-burn-in sample of each parameter, over all maps and chains."""
    return [np.mean(np.concatenate([chain for chains in all_chains[name] for chain in chains]))
            for name in param_names]

def get_forecasted_covariance(nside, theta_pix, noise, is_sampled, param_names,
                              nphi_source = "qe_delensed", phi_noise = None,
                              delensed_covariance = None):
    """Fisher-forecast covariance for the sampled parameters, reordered to match param_names.

    `nphi_source` selects the forecast's N_phi (fisher_forecast.NPHI_SOURCES): "qe_unlensed"
    (the upper bound on the information) or "qe_delensed" (the lower bound), or "map_joint" /
    "sample_joint" with their `phi_noise` file. The response spectrum comes from
    FORECAST_QE_RESPONSE. The T block is CAMB's delensed spectrum, or the empirical one of
    the `delensed_covariance` npz when given. CAMB results are memoized in
    fisher_forecast._CAMB_CACHE and the stencil POINTS are shared, so later calls re-run no
    CAMB, only the reconstruction and the block builds.

    forecast() returns its parameters in fisher_forecast's OUTPUT_PARAM_ORDER (omch2, ombh2,
    ns, theta_MC_100, logA), while param_names follows the chains' (GROUND_TRUTH's) order,
    so the forecast covariance is permuted into param_names order here.
    """
    fisher, names = forecast(nside, theta_pix, noise, is_sampled, GROUND_TRUTH,
                             nphi_source = nphi_source,
                             qe_response = FORECAST_QE_RESPONSE.get(nphi_source, "unlensed"),
                             phi_noise = phi_noise,
                             delensed_covariance = delensed_covariance,
                             camb_lmax = FORECAST_CAMB_LMAX, l_knee = 0, beam_fwhm = 0)
    covariance = covariance_from_fisher(fisher, names)
    order = [names.index(name) for name in param_names]
    return covariance[np.ix_(order, order)]

def joint_param_analysis(all_chains, nside, theta_pix, noise, is_sampled, pol, fisher_dir,
                         forecast_phi_noise, forecast_delensed_covariance):
    """The measured posterior against every forecast, into fisher_dir: the chains'
    covariance_matrix_empirical.png and correlation_matrix_empirical.png at its root, and per
    forecast a <nphi_source>/<t_block>/ folder (forecast_plot_dir) with its triangle plot,
    covariance and correlation matrices and the chains-over-forecast covariance ratio.

    Every source in FORECAST_NPHI_SOURCES is forecast with CAMB's delensed T block
    ("camb_delensed") and, when `forecast_delensed_covariance` has a file for it, with that
    file's empirical one too ("empirically_delensed"). A measured-N_phi source (map_joint /
    sample_joint) with no file in `forecast_phi_noise` is skipped altogether."""

    print("\n")
    print(f"Beginning joint parameter analysis")
    (measured_cov_mat, param_names, measured_cov_sigma,
     measured_sigma_err) = get_avg_covariance_matrix(all_chains)
    means = get_means_per_param(all_chains, param_names)
    plot_covariance_and_correlation(measured_cov_mat, param_names, EMPIRICAL_LABEL, fisher_dir,
                                    title = "Chains (empirical)")
    #both bounds: the measured posterior should sit BETWEEN them, in the widths and in the
    #off-diagonal covariances alike
    forecasts = {}
    for nphi_source in FORECAST_NPHI_SOURCES:
        if nphi_source in FILE_NPHI_SOURCES and nphi_source not in forecast_phi_noise:
            print(f"skipping the {nphi_source} forecasts: no N_phi file given")
            continue
        phi_noise = forecast_phi_noise.get(nphi_source)
        t_block_files = {"camb_delensed": None}
        if nphi_source in forecast_delensed_covariance:
            t_block_files["empirically_delensed"] = forecast_delensed_covariance[nphi_source]
        for t_block, path in t_block_files.items():
            forecasts[(nphi_source, t_block)] = get_forecasted_covariance(
                nside, theta_pix, noise, is_sampled, param_names, nphi_source = nphi_source,
                phi_noise = phi_noise, delensed_covariance = path)

    for (nphi_source, t_block), forecasted_cov_mat in forecasts.items():
        plot_dir = forecast_plot_dir(fisher_dir, nphi_source, t_block)
        label = forecast_file_label(nphi_source, t_block)
        title = forecast_title(nphi_source, t_block)
        triangle_plot(measured_cov_mat, nphi_source, t_block, forecasted_cov_mat, param_names,
                      means, measured_cov_sigma, measured_sigma_err,
                      plot_dir, nside = nside, theta_pix = theta_pix, noise = noise, pol = pol)
        plot_covariance_and_correlation(forecasted_cov_mat, param_names, label, plot_dir,
                                        title = f"Forecast ({title})")
        plot_annotated_heat_map(measured_cov_mat / forecasted_cov_mat, param_names,
                                f"Chains / forecast covariance ({title})",
                                f"covariance_ratio_{EMPIRICAL_LABEL}_over_{label}.png",
                                plot_dir)
    return

# ── Phi-noise and delensed-spectra comparison plots ────────────────────────

#(colour, marker) of each N_phi source's measured series in the spectra plots: its measured
#N_phi in the phi-noise plot, and the empirical delensed spectrum whose W it built in the
#delensed-spectra plot
SOURCE_STYLE = {"qe_unlensed": ("C0", "o"),
                "qe_delensed": ("C1", "s"),
                "map_joint": ("C2", "^"),
                "sample_joint": ("C9", "D")}
#the largest band centre the delensed-spectra plot draws: past CAMB's own range the model
#spectra are a log-log extrapolation and the box's lensed power aliases at the grid's corner
SPECTRA_ELL_MAX = 4000.0

def band_index(ell_grid, edges, use):
    """The |L| band of every rfft-grid mode, -1 for a mode outside the bands or not in `use`."""
    ells = np.asarray(ell_grid)
    index = np.digitize(ells, edges) - 1
    return np.where(use & (index >= 0) & (index < len(edges) - 1), index, -1)

def band_harmonic(matrix, index, n_band):
    """The INFORMATION-weighted band value 1 / <1/N> over the band's finite positive modes
    (the mean that commutes with averaging Fisher information), and the mode count. Each
    rfft entry is weighted by w_k (2 in the bulk, 1 on the self-conjugate columns), the
    number of full-plane modes it stands for, as in fisher_forecast.radial_cl_profile.

    The harmonic mean is dominated by the LOWEST noise modes, the arithmetic mean by the
    LARGEST. E.g. two modes with N = 1 and N = 100 carry information 1/1 + 1/100 = 1.01:
    the harmonic mean 2 / 1.01 = 1.98 gives back 2 / 1.98 = 1.01, while the arithmetic mean
    50.5 implies 2 / 50.5 = 0.04, as if the band held almost no information. The QE's
    diverged modes and an empirical file's poorly measured modes would drag an arithmetic
    band mean up the same way."""
    dof = np.asarray(fourier_weights(matrix.shape[0]))
    inside = (index >= 0) & np.isfinite(matrix) & (matrix > 0)
    weight = dof[inside]
    total = np.bincount(index[inside], weights = weight / matrix[inside], minlength = n_band)
    count = np.bincount(index[inside], weights = weight, minlength = n_band)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.where(count > 0, count / total, np.nan), count

def band_median(matrix, index, n_band):
    """Median over each band's finite modes (NaN for an empty band), each rfft entry counted
    w_k times (2 in the bulk, 1 on the self-conjugate columns) like band_harmonic, i.e. the
    median over the full-plane modes."""
    dof = np.asarray(fourier_weights(matrix.shape[0])).astype(int)
    values = np.full(n_band, np.nan)
    for b in range(n_band):
        inside = (index == b) & np.isfinite(matrix)
        if inside.any():
            values[b] = np.median(np.repeat(matrix[inside], dof[inside]))
    return values

def qe_noise_matrices(params, nside, theta_pix, noise_level, l_knee, beam_fwhm):
    """(C_phi, [(label, color, linestyle, N_phi^QE matrix)]) on the rfft grid from direct CAMB,
    all with the unlensed response and undivided by NPHI_FAC: the box QE with the lensed C_TT
    in its filter (the reference every ratio is taken against, first), the iterated-delensing
    fixed point (fisher_forecast's qe_delensed filter) and the unlensed C_TT in the filter
    (qe_unlensed)."""
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    cls = camb_cls_at_params(params)
    delensed_tt = iterative_delensing(params, cls, nside, pix_width, ell_grid, noise_level,
                                      l_knee, beam_fwhm, DEFAULT_ELL_CUTOFF, verbose = False,
                                      qe_response = "unlensed")[2]
    matrices = []
    for label, color, linestyle, filter_tt in (
            ("QE (box matrix: unlensed response, lensed filter, no NPHI_FAC)", "C3", "--",
             cls["total_TT"]),
            ("QE, iterative delensing (unlensed response)", "C4", "-.", delensed_tt),
            ("QE, unlensed $C^{TT}$ in the filter (unlensed response)", "C5", "-.",
             cls["scalar_TT"])):
        matrices.append((label, color, linestyle, np.asarray(qe_noise_matrix(
            cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, DEFAULT_ELL_CUTOFF,
            filter_tt = filter_tt, qe_response = "unlensed"))))
    cphi = np.asarray(prior_covariances(params, nside, pix_width, ell_grid)[1])
    return cphi, matrices

def plot_phi_noise_vs_qe(nside, theta_pix, noise, forecast_phi_noise, out_dir):
    """Per-mode phi noise spectra against the box QE matrix at this noise level:
    phi_noise_vs_qe_per_mode_<noise>uk.png in out_dir.

    Draws the `nphi` (C_phi (1/r^2 - 1), r^2 = <phi_bar phi*>^2 / (<|phi_bar|^2><|phi|^2>))
    of every merged map_joint_phi_noise.npz / sampler_phi_noise.npz in `forecast_phi_noise`,
    on the chains' box at GROUND_TRUTH (white noise, no beam). Top panel:
    information-weighted band means with CAMB's C_L^phiphi and the QE matrices of
    qe_noise_matrices. Bottom panel: every mode's N_k / N_QE,k as a faint point, with the
    band median on top.
    """
    if not forecast_phi_noise:
        print(f"no phi noise files given: phi_noise_vs_qe_per_mode_{noise}uk.png skipped")
        return
    ell_grid = np.asarray(gen_ell_grid(nside, theta_pix)[0])
    use = ell_grid > 0
    edges = band_edges(ell_grid)
    n_band = len(edges) - 1
    index = band_index(ell_grid, edges, use)
    centres = 0.5 * (edges[1:] + edges[:-1])
    cphi, qe_matrices = qe_noise_matrices(GROUND_TRUTH, nside, theta_pix, noise, 0.0, 0.0)
    nphi_qe = qe_matrices[0][3]
    series = []
    for nphi_source, path in forecast_phi_noise.items():
        merged = np.load(path)
        colour, marker = SOURCE_STYLE[nphi_source]
        name = (f"{nphi_source} $N_{{\\rm eff}}$ = $C_\\phi(1/r^2-1)$ "
                f"({int(merged['n_realizations'])} realizations)")
        series.append((name, colour, marker, np.asarray(merged["nphi"])))

    figure, (top, bottom) = plt.subplots(2, 1, figsize = (10, 10), sharex = True,
                                         gridspec_kw = dict(height_ratios = [3, 2]))
    order = np.argsort(ell_grid[use])
    top.loglog(ell_grid[use][order], cphi[use][order], color = "0.35", linewidth = 1.5,
               label = r"CAMB $C_L^{\phi\phi}$")
    band_qe = band_harmonic(np.where(use, nphi_qe, np.nan), index, n_band)[0]
    keep = np.isfinite(band_qe)
    bottom.axhline(1.0, color = "C3", linestyle = "--", linewidth = 1)
    for position, (name, color, linestyle, matrix) in enumerate(qe_matrices):
        band = band_harmonic(np.where(use, matrix, np.nan), index, n_band)[0]
        good = keep & np.isfinite(band)
        top.loglog(centres[good], band[good], color = color, linestyle = linestyle,
                   linewidth = 1.3 if position else 1.5, label = name)
        if position:
            bottom.plot(centres[good], band[good] / band_qe[good], color = color,
                        linestyle = linestyle, linewidth = 1.3,
                        label = f"{name} / QE (ratio of band means)")

    for name, color, marker, matrix in series:
        measured = use & np.isfinite(matrix) & (matrix > 0)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            ratio = np.where(measured, matrix / nphi_qe, np.nan)
        band = band_harmonic(np.where(measured, matrix, np.nan), index, n_band)[0]
        good = np.isfinite(band)
        top.loglog(centres[good], band[good], color = color, marker = marker, markersize = 3,
                   linewidth = 1, label = name)
        bottom.plot(ell_grid[measured], ratio[measured], linestyle = "none", marker = ".",
                    markersize = 1.5, color = color, alpha = 0.12, rasterized = True)
        median = band_median(ratio, index, n_band)
        good = np.isfinite(median)
        bottom.plot(centres[good], median[good], color = color, marker = marker, markersize = 3,
                    linewidth = 1.2,
                    label = f"{name.split(' (')[0].split(' = ')[0]} / QE (band median)")
        print(f"{noise:g} uK  {name.split(' (')[0][:24]:24s} modes {int(measured.sum()):5d}  "
              f"median N/N_QE {np.nanmedian(ratio):.3f}")

    top.set_ylabel(r"$N_L$ (covar units; information-weighted band mean of the per-mode matrix)")
    top.legend(fontsize = 8)
    bottom.set_xscale("log")
    bottom.set_ylim(0.0, 2.0)
    bottom.set_xlabel(r"$L$")
    bottom.set_ylabel(r"per-mode $N_k / N_k^{\rm QE}$")
    bottom.legend(fontsize = 7, loc = "upper left", ncol = 2)
    figure.suptitle(f"phi noise per mode relative to the QE | nside {nside}, "
                    f"{theta_pix:g}', {noise:g} uK-arcmin", fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"phi_noise_vs_qe_per_mode_{noise}uk.png")
    figure.savefig(path, dpi = 150)
    plt.close(figure)
    print(f"wrote {path}")

#NOTE this method is not currently called anywhere
def plot_delensed_spectra_comparison(nside, theta_pix, noise, forecast_delensed_covariance,
                                     out_dir, delta_ell = DEFAULT_DELTA_ELL):
    """Unlensed / lensed / delensed temperature spectra, CAMB against the box, at this noise
    level: delensed_spectra_comparison_<noise>uk.png in out_dir.

    Series, all band-averaged over the same |l| annuli of the rfft grid: CAMB's unlensed,
    lensed and iteratively delensed TT (get_partially_lensed_cls at the Alens_L
    fisher_forecast.iterative_delensing converges to, unlensed response), and the empirical
    delensed spectrum (`delensed_fid`) of every merged delensed_covariance.npz in
    `forecast_delensed_covariance`, drawn directly (so they carry the realizations' cosmic
    variance). Everything is on the chains' box at GROUND_TRUTH (white noise, no beam).
    Top panel: D_l = l (l + 1) C_l / 2 pi. Bottom panel: every series over CAMB unlensed.
    """
    if not forecast_delensed_covariance:
        print(f"no delensed covariance files given: delensed_spectra_comparison_{noise}uk.png "
              f"skipped")
        return
    ell_grid, pix_width = gen_ell_grid(nside, theta_pix)
    weights = fourier_weights(nside)
    edges = band_edges(ell_grid, delta_ell)

    def band(matrix):
        centres, values, _ = band_average(matrix, ell_grid, weights, edges)
        return centres, values

    def camb_band(cl):
        #CAMB's C_l on the grid exactly as the merge puts it there (C_l / pix_width^2)
        ells = jnp.arange(2, 2 + cl.shape[0]).astype(jnp.float64)
        return band(np.asarray(covar_matrix_from_cls(nside, pix_width, ell_grid, ells, cl,
                                                     origin_value = 0)))

    cls = camb_cls_at_params(GROUND_TRUTH)
    #the converged delensed C_TT of the iteration: box QE matrix (no NPHI_FAC), unlensed
    #response, beam 0 as the data are simulated
    delensed_tt = iterative_delensing(GROUND_TRUTH, cls, nside, pix_width, ell_grid, noise,
                                      0.0, 0.0, 10_000, verbose = False,
                                      qe_response = "unlensed")[2]

    centres, camb_unlensed = camb_band(cls["scalar_TT"])
    empirical = []
    for nphi_source, path in forecast_delensed_covariance.items():
        merged = np.load(path, allow_pickle = True)
        colour, marker = SOURCE_STYLE[nphi_source]
        label = f"empirically delensed, W from N_phi: {forecast_labels(nphi_source)[1]}"
        empirical.append((label, colour, marker, merged))
    #grid units C_l / pix_width^2 -> D_l
    to_dl = centres * (centres + 1) / (2 * np.pi) * float(pix_width)**2
    series = [("CAMB unlensed", "0.2", "-", None, camb_unlensed),
              ("CAMB lensed", "C3", "-", None, camb_band(cls["total_TT"])[1]),
              ("CAMB iteratively delensed", "C4", "--", None, camb_band(delensed_tt)[1])]
    for name, color, marker, merged in empirical:
        delensed = band(np.asarray(merged["delensed_fid"]))[1]
        series.append((name, color, "-", marker, delensed))

    figure, (top, bottom) = plt.subplots(2, 1, figsize = (11, 10), sharex = True,
                                         gridspec_kw = dict(height_ratios = [3, 2]))
    shown = centres <= SPECTRA_ELL_MAX
    centres, camb_unlensed, to_dl = centres[shown], camb_unlensed[shown], to_dl[shown]
    for name, color, linestyle, marker, values in series:
        values = values[shown]
        style = dict(color = color, linestyle = linestyle, marker = marker, markersize = 3,
                     linewidth = 1.3)
        top.plot(centres, values * to_dl, label = name, **style)
        bottom.plot(centres, values / camb_unlensed, label = name, **style)

    top.set_yscale("log")
    top.set_ylabel(r"$D_\ell^{TT} = \ell(\ell+1)\,C_\ell / 2\pi$  [$\mu$K$^2$]")
    top.legend(fontsize = 8)
    bottom.set_xlabel(r"$\ell$")
    bottom.set_ylabel(r"$C_\ell\ /\ C_\ell^{\rm unlensed,\ CAMB}$")
    figure.suptitle(f"Temperature spectra | nside {nside}, {theta_pix:g}', "
                    f"{noise:g} uK-arcmin | bands of {delta_ell:g}", fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"delensed_spectra_comparison_{noise}uk.png")
    figure.savefig(path, dpi = 150)
    plt.close(figure)
    print(f"wrote {path}")

def main(file_name, num_maps, num_chains, map_pre_factor, ground_truth_values,
         was_sampled, default_burn_in, nside, theta_pix, noise, pol,
         forecast_phi_noise, forecast_delensed_covariance):
    """Every plot of one run goes under its noise level's folder (plot_dirs),
    lcdm_chain_plots/noise_level_<noise>uk/: one subfolder per sampled parameter, fisher/ (the
    covariance / correlation / triangle plots, one <nphi_source>/<t_block>/ folder per
    forecast) and spectra_plots/.

    forecast_phi_noise:           {"map_joint" | "sample_joint": path} of merged
                                  map_joint_phi_noise.npz / sampler_phi_noise.npz files
    forecast_delensed_covariance: {nphi_source: path} of merged delensed_covariance.npz
                                  files whose W was built from that source's N_phi
    Both measured at this run's box and noise level; either may be empty."""
    chain_plot_dir, spectra_plot_dir, fisher_plot_dir = plot_dirs(noise)
    for path in (chain_plot_dir, spectra_plot_dir, fisher_plot_dir):
        os.makedirs(path, exist_ok = True)

    sampled_names = [name for name in ground_truth_values if was_sampled[name]]
    non_converged_maps = prune_maps_by_r_hat(file_name, num_maps, num_chains, map_pre_factor,
                                             default_burn_in, R_THRESHOLD,
                                             list(ground_truth_values))

    #single parameter analyses
    all_chains = {name: per_param_analysis(file_name, num_maps, num_chains, map_pre_factor,
                                           ground_truth_values[name], default_burn_in, name,
                                           non_converged_maps, chain_plot_dir)
                  for name in sampled_names}

    #the posterior against the Fisher forecasts
    if len(sampled_names) > 1:
        joint_param_analysis(all_chains, nside, theta_pix, noise, was_sampled, pol,
                             fisher_plot_dir, forecast_phi_noise,
                             forecast_delensed_covariance)

    #the phi noise and the delensed spectra at the chains' noise level
    print("\n")
    print(f"Plotting the phi noise and delensed spectra at {noise:g} uK-arcmin")
    plot_phi_noise_vs_qe(nside, theta_pix, noise, forecast_phi_noise, spectra_plot_dir)
    return

if __name__ == "__main__":

    #the Fisher forecast the triangle plot compares against, and the phi-noise / delensed-
    #spectra plots, are only meaningful at the box and noise level the chains ran on - MATCH
    #sample_lcdm.sh
    nside = 128
    theta_pix = 2.5
    noise = 5
    #"I", "P" or "IP", as pol in sample_lcdm.sh; only labels the triangle plots
    pol = "I"
    num_maps = 10
    num_chains = 5
    default_burn_in = 600
    map_pre_factor = 234567
    file_name = "data_5uk"

    was_sampled = {}
    was_sampled["omch2"] = True
    was_sampled["ombh2"] = False
    was_sampled["ns"] = False
    was_sampled["theta_MC_100"] = True
    was_sampled["logA"] = True

    #the measured inputs to the forecasts and spectra plots, at THIS box and noise level.
    #Edit to wherever the merged files are; drop an entry to skip it

    #File paths to empirically generated phi noise matrices
    forecast_phi_noise = {
        # "map_joint": os.path.join(REPO_DIR, "get_map_joint_phi_noise",
        #                           "map_joint_phi_noise_output", "map_joint_phi_noise.npz"),
        # "sample_joint": os.path.join(REPO_DIR, "get_sampler_phi_noise",
        #                              "sampler_phi_noise_output", "sampler_phi_noise.npz"),
    }

    #File paths to empirically delensed temperature covariance matrices for a given phi noise matrix
    covariance_data = os.path.join(REPO_DIR, "get_delensed_field_covariance",
                                   "delensed_field_covariance_data")
    forecast_delensed_covariance = {
        "qe_unlensed": os.path.join(covariance_data, "lower_qe_bound", 
                                    f"delensed_covariance_output_{noise}uk_lower_qe_bound",
                                    "delensed_covariance.npz"),
        # "qe_delensed": os.path.join(covariance_data, "upper_qe_bound",
        #                             f"delensed_covariance_output_{noise}uk_upper_qe_bound",
        #                             "delensed_covariance.npz"),
        # "map_joint": os.path.join(covariance_data, "empirical_map_joint_noise",
        #                           f"delensed_covariance_output_{noise}uk_map_joint",
        #                           "delensed_covariance.npz"),
        # "sample_joint": os.path.join(covariance_data, "empirical_sample_joint_noise",
        #                              f"delensed_covariance_output_{noise}uk_sample_joint",
        #                              "delensed_covariance.npz"),
    }

    main(file_name, num_maps, num_chains, map_pre_factor,
         GROUND_TRUTH, was_sampled, default_burn_in,
         nside, theta_pix, noise, pol, forecast_phi_noise, forecast_delensed_covariance)
