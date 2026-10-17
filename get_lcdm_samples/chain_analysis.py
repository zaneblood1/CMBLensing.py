import glob
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
from cmb_lensing.constants import PARAM_ORDER
from cmb_lensing.fisher_forecast import (annotated_heatmap, covariance_from_fisher, GROUND_TRUTH,
                                         forecast, fourier_weights, camb_cls_at_params,
                                         iterative_delensing, qe_noise_matrix,
                                         prior_covariances)
from cmb_lensing.delensed_covariance import (MERGED_NAME, DEFAULT_DELTA_ELL, band_edges,
                                             band_average)
from cmb_lensing.sampler_effective_phi_noise import used_modes, band_index, band_harmonic

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HERE)
#the chains (one <file_name>/ directory of *_history.txt per experiment)
CHAIN_DATA_DIR = os.path.join(HERE, "lcdm_sampler_data")
#this script's plots, each run's in a noise_level_<tag>/ subfolder (plot_dirs)
CHAIN_PLOT_ROOT = os.path.join(HERE, "lcdm_chain_plots")
#the merged measurements those plots read, one subfolder per run
SAMPLER_NOISE_DATA = os.path.join(REPO_DIR, "get_sampler_phi_noise", "sampler_phi_noise_data")
COVARIANCE_DATA = os.path.join(REPO_DIR, "get_delensed_field_covariance",
                               "delensed_field_covariance_data")

#the Fisher forecasts drawn against the chains, by fisher_forecast nphi_source: the two
#reported bounds and the two measured N_phi (which need FORECAST_PHI_NOISE). Each is
#forecast with CAMB's delensed T block ("camb_delensed") and, when FORECAST_DELENSED_COVARIANCE
#has a run for it, with the empirical one ("empirically_delensed") as well
FORECAST_NPHI_SOURCES = ("qe_unlensed", "qe_delensed", "map_joint", "sample_joint")

#the QE response spectrum per qe_ source ("unlensed" when absent)
FORECAST_QE_RESPONSE = {"qe_delensed": "gradient"}
#nphi_source -> the npz its N_phi is read from, at the box the chains ran on, as a path with
#the noise tag (e.g. 2dot5uk) left open as {tag} (resolved by forecast_input): map_joint -> a
#merge_delensed_covariance.py npz with phi moments (its phi_noise_fid), sample_joint -> a
#merge_sampler_effective_noise.py npz (its nphi)
FORECAST_PHI_NOISE = {
    "map_joint": os.path.join(COVARIANCE_DATA, "lower_qe_bound",
                              "delensed_covariance_output_{tag}_nphi_unlensedx2",
                              "delensed_covariance.npz"),
    "sample_joint": os.path.join(SAMPLER_NOISE_DATA,
                                 "sampler_effective_noise_output_mean_chains_{tag}",
                                 "sampler_effective_noise.npz"),
}
#nphi_source -> a merge_delensed_covariance.py npz (noise tag left open as {tag}) whose
#EMPIRICAL delensed T block gives that source's second, "empirically_delensed" forecast. Its
#W was built from the same N_phi as the source, so both blocks sit on one reconstruction (the
#forecast warns otherwise). No entry, or no file at this noise level: CAMB's only
FORECAST_DELENSED_COVARIANCE = {
    "qe_unlensed": os.path.join(COVARIANCE_DATA, "lower_qe_bound",
                                "delensed_covariance_output_{tag}_nphi_unlensedx2",
                                "delensed_covariance.npz"),
    "qe_delensed": os.path.join(COVARIANCE_DATA, "upper_qe_bound",
                                "delensed_covariance_output_fiducial_wiener_{tag}_0dot05_sigma_"
                                "iterative_delens_qe_gradient",
                                "delensed_covariance.npz"),
    "map_joint": os.path.join(COVARIANCE_DATA, "empirical_map_joint_noise",
                              "delensed_covariance_output_map_joint_noise_{tag}",
                              "delensed_covariance.npz"),
    "sample_joint": os.path.join(COVARIANCE_DATA, "empirical_sampler_noise",
                                 "delensed_covariance_output_{tag}_sample_joint_neff",
                                 "delensed_covariance.npz")
}

#the two temperature blocks every forecast is built with: folder name -> description
T_BLOCKS = {"camb_delensed": "CAMB delensed T",
            "empirically_delensed": "empirically delensed T"}
#how far CAMB is run for the forecasts (None: the rfft grid's corner mode)
FORECAST_CAMB_LMAX = 4000

BUFFER = 1e-4
GRID_SIZE = 1000

#which stride prune_chains thins each chain by. True -> the lag just before that chain's
#autocorrelation first crosses zero (first_zero_crossing_lag), False -> its integrated
#autocorrelation time (integrated_autocorrelation_time). flip this to compare the two
USE_ZERO_CROSSING_PRUNE = False
SHOULD_THIN = True
#a map is discarded when any parameter's Gelman-Rubin R-hat across its sub-chains exceeds this
R_THRESHOLD = 300 

#which blocks of samples get_avg_covariance_matrix builds one covariance matrix from before
#averaging. "sub_chain" -> one per (map, sub-chain); "map" -> one per map, from all of that
#map's sub-chains concatenated. See covariance_blocks_to_average for what separates them
AVERAGE_OVER_CHOICES = ("sub_chain", "map")
AVERAGE_OVER = "map"

#the label of the measured (chain) posterior in file names
EMPIRICAL_LABEL = "empirical"

def noise_tag(noise):
    """The noise level as the data folders spell it: 2.5 -> "2dot5uk", 5 -> "5uk"."""
    return f"{noise:g}".replace(".", "dot") + "uk"

def plot_dirs(noise):
    """(chain plot dir, spectra plot dir, fisher plot dir) of one noise level, all ending in
    os.sep: lcdm_chain_plots/noise_level_<tag>/ (the per-parameter folders) and its
    spectra_plots/ and fisher/ subfolders. fisher/ holds the chains' covariance and
    correlation matrices and one <nphi_source>/<t_block>/ folder per forecast
    (forecast_plot_dir)."""
    chain_dir = os.path.join(CHAIN_PLOT_ROOT, f"noise_level_{noise_tag(noise)}")
    return (chain_dir + os.sep, os.path.join(chain_dir, "spectra_plots") + os.sep,
            os.path.join(chain_dir, "fisher") + os.sep)

def forecast_plot_dir(fisher_dir, nphi_source, t_block):
    """fisher/<nphi_source>/<t_block>/ (created): one forecast's triangle plot, covariance,
    correlation and covariance-ratio plots."""
    path = os.path.join(fisher_dir, nphi_source, t_block) + os.sep
    os.makedirs(path, exist_ok = True)
    return path

def forecast_input(templates, nphi_source, noise):
    """The file `templates[nphi_source]` names at this noise level, or None when there is no
    template or no file there."""
    template = templates.get(nphi_source)
    if template is None:
        return None
    path = template.format(tag = noise_tag(noise))
    return path if os.path.exists(path) else None

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

def first_zero_crossing_lag(acf):
      """Smallest lag right before the autocorrelation first crosses zero.

      Walks out from lag 0 (where the acf is 1 by construction) to the first lag whose
      acf is non-positive and returns the lag just before it. Used as a thinning stride:
      samples separated by more than this lag are no longer positively correlated. Unlike
      Sokal's IAT this needs no windowing parameter, but it also ignores the size of the
      correlations rather than integrating them, so it is typically the shorter stride.
      Falls back to the largest available lag when the acf never crosses zero (a chain
      that has not decorrelated within max_lag), and never returns less than 1 so the
      result is always usable as a slice step."""
      acf = np.asarray(acf)
      non_positive = np.nonzero(acf <= 0)[0]
      if len(non_positive) == 0:
          return max(1, len(acf) - 1)
      return max(1, int(non_positive[0]) - 1)

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
    plt.plot(grid, multiplicative_pdf, label="Multiplicative Mean PDF", color = "blue")
    plt.axvline(ground_truth, label = f"Ground Truth = {ground_truth}", color = "green")

    #multiplicative estimators
    plt.axvline(multiplicative_mode, label = f"Mult. Mode = {round(multiplicative_mode, 7)}", color = "orange")
    plt.axvline(multiplicative_mean, label = f"Mult. Mean = {round(multiplicative_mean, 7)}", color = "pink")
    plt.axvline(multiplicative_mean + multiplicative_std,
                label = f"+/- 1 Mult. Std = {round(multiplicative_std, 7)}", color = "grey")
    plt.axvline(multiplicative_mean - multiplicative_std, color = "grey")

    product_z_score = (ground_truth - multiplicative_mean)/multiplicative_std
    plt.plot([], [], label=f"Product Z-score = {product_z_score}")

    std_err_z_score = (ground_truth - multiplicative_mean)/avg_sigma_single
    plt.plot([], [], label=f"Std. Err. Z-score = {std_err_z_score}")

    plt.xlabel("Value"); plt.ylabel("Density")
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
    iats_per_map = [[integrated_autocorrelation_time(acf) for acf in acfs]
                    for acfs in acfs_per_map]
    flat_iats = [iat for iats in iats_per_map for iat in iats]
    plt.figure()
    plt.barh(range(len(flat_iats)), flat_iats)
    plt.ylabel("Chain"); plt.xlabel("Integrated Auto-Correlation Time")
    mean_iat = np.mean(flat_iats)
    plt.axvline(mean_iat, label = f"Mean IAT = {mean_iat}", color = "black")
    plt.legend()
    plt.title(f"{param_name} Integrated Auto Correlation Time per Chain")
    plt.savefig(output_path + f"{param_name.lower()}_integrated_autocorrelation_times.png")
    plt.close()
    return iats_per_map

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

def prune_chains(raw_chains, iats_per_map, acfs_per_map = None,
                 use_zero_crossing = USE_ZERO_CROSSING_PRUNE,
                 should_thin = SHOULD_THIN):
    #thin each chain by its own stride so the retained points are not correlated. the
    #stride is either the integrated autocorrelation time (default) or, with
    #use_zero_crossing, the lag just before that chain's acf first crosses zero
    if use_zero_crossing and acfs_per_map is None:
        raise ValueError("use_zero_crossing pruning requires the per-chain acfs")
    pruned_chains = []
    for i, chains in enumerate(raw_chains):
        processed_chains = []
        for j, chain in enumerate(chains):
            if should_thin:
                if use_zero_crossing:
                    stride = first_zero_crossing_lag(acfs_per_map[i][j])
                else:
                    #clamp so an IAT below 1 cannot produce a zero slice step
                    stride = max(1, int(iats_per_map[i][j]))
                processed_chains.append(chain[0:-1:stride])
            else:
                processed_chains.append(chain)
        pruned_chains.append(np.concatenate([chain for chain in processed_chains]))
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

    One caveat: the sampler writes one history entry per Gibbs iteration, and with
    metropolis_num_steps > 1 several Metropolis proposals happen inside one iteration, so a
    "move" here means at least one of them was accepted. With one proposal per iteration
    (the default) this is exactly the per-proposal acceptance rate.
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
                data_path, param_name, map_idx, num_chains, map_pre_factor, default_burn_in,
                verbose = True)
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
    iats_per_map = plot_iat_per_chain(acfs_per_map, output_path, param_name)

    #prune the chains by their individual strides (IAT or first zero crossing, per
    #USE_ZERO_CROSSING_PRUNE):
    chains = prune_chains(raw_chains, iats_per_map, acfs_per_map = acfs_per_map,
                          use_zero_crossing = USE_ZERO_CROSSING_PRUNE,
                          should_thin = SHOULD_THIN)

    #make an all chain trace plot
    all_chain_trace(chains, output_path, param_name)

    #plot each single map's distribution
    plot_marginals(chains, output_path, ground_truth, param_name)

    #histogram of per-map marginal means
    avg_sigma_single = plot_mean_histogram(chains, output_path, ground_truth, param_name)

    #combined product distribution
    plot_multiplicative_mean(chains, avg_sigma_single, output_path, ground_truth, param_name)

    return raw_chains

def plot_annotated_matrix(matrix, param_names, title, output_path, file_name, cmap = "coolwarm",
                          norm = None, fmt = "{:+.3f}"):
    """Draw a square matrix as a colored grid with the value of every entry printed in its box.

    norm is a matplotlib color normalization (defaults to a symmetric linear scale about zero,
    which suits correlation matrices); fmt is the format string applied to each entry. Text is
    drawn black or white depending on the luminance of the cell behind it so it stays legible
    across the whole colormap.
    """
    matrix = np.asarray(matrix)
    n = len(param_names)
    if norm is None:
        limit = np.max(np.abs(matrix))
        norm = matplotlib.colors.Normalize(vmin = -limit, vmax = limit)
    fig, ax = plt.subplots(figsize = (1.7 * n + 2.5, 1.7 * n + 1.5))
    image = ax.imshow(matrix, cmap = cmap, norm = norm)
    ax.set_xticks(range(n)); ax.set_xticklabels(param_names, rotation = 45, ha = "right")
    ax.set_yticks(range(n)); ax.set_yticklabels(param_names)
    #minor ticks give the white grid lines separating the boxes
    ax.set_xticks(np.arange(n + 1) - 0.5, minor = True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor = True)
    ax.grid(which = "minor", color = "white", linewidth = 2)
    ax.tick_params(which = "minor", length = 0)
    for i in range(n):
        for j in range(n):
            rgba = image.cmap(norm(matrix[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt.format(matrix[i, j]), ha = "center", va = "center",
                    color = "white" if luminance < 0.5 else "black", fontsize = 11)
    fig.colorbar(image, ax = ax, fraction = 0.046, pad = 0.04)
    ax.set_title(title)
    plt.savefig(output_path + file_name, dpi = 150, bbox_inches = "tight")
    plt.close(fig)

def plot_annotated_heat_map(matrix, names, title, file_name, path):
    figure, axis = plt.subplots(figsize = (7.5, 6.5))
    annotated_heatmap(axis, matrix, names, title, "Value")
    figure.tight_layout()
    figure.savefig(path + file_name, dpi = 150)
    plt.close(figure)
    return

def get_max_num_maps_and_sub_chains(raw_chains, param_names):
    max_num_maps = np.max([len(raw_chains[name]) for name in param_names])
    max_num_sub_chains = max(len(sub_chains) for name in param_names
                             for sub_chains in raw_chains[name])
    return max_num_maps, max_num_sub_chains

def covariance_blocks_to_average(raw_chains, param_names, average_over):
    """The groups of per-parameter columns that each get one covariance matrix.

    "sub_chain" yields one group per (map, sub-chain), so the average runs over every
    sub-chain of every map. "map" concatenates a map's sub-chains first and yields one group
    per map, so that map's covariance is measured on all of its samples at once.

    The only thing separating them is whether the spread BETWEEN a map's sub-chains counts
    towards the width: it does in "map" and does not in "sub_chain". On well mixed chains
    they agree closely (0.3-0.5% in sigma on the 09_15_26 run), so a large gap between the
    two is a mixing warning rather than a real choice of estimator - compare it against the
    Gelman-Rubin R printed by per_param_analysis.

    NEITHER of them ever mixes samples from different maps. Each map is a different data
    realization with its own posterior mean, so pooling across maps would add the scatter of
    those means to the diagonal - measured at +15% / +38% / +69% in sigma on omch2 /
    theta_MC_100 / logA, and it drags r(omch2, theta_MC_100) from +0.36 down to +0.08
    because the between-map means are themselves anti-correlated in that pair. The
    covariance this returns is meant to be a single-realization posterior, which is what a
    Fisher forecast predicts.

    Returns a list of (map_idx, columns) pairs; the map index lets the caller jackknife over
    MAPS, the independent unit (a map's sub-chains all sample the same posterior).
    """
    if average_over not in AVERAGE_OVER_CHOICES:
        raise ValueError(f"average_over must be one of {AVERAGE_OVER_CHOICES}, "
                         f"got {average_over!r}")
    max_num_maps, max_num_sub_chains = get_max_num_maps_and_sub_chains(raw_chains, param_names)
    blocks = []

    for map_idx in range(max_num_maps):
        if average_over == "sub_chain":
            for chain_idx in range(max_num_sub_chains):
                columns = []
                for name in param_names:
                    if map_idx < len(raw_chains[name]) and chain_idx < len(raw_chains[name][map_idx]):
                        columns.append(raw_chains[name][map_idx][chain_idx])
                blocks.append((map_idx, columns))
        else:
            columns = []
            for name in param_names:
                if map_idx < len(raw_chains[name]):
                    columns.append(np.concatenate(raw_chains[name][map_idx]))
            blocks.append((map_idx, columns))

    return blocks

def measured_errors(block_cov_mats, block_map_idxs):
    """(cov_err, sigma_err, leave_one_out): the delete-one-MAP jackknife of the
    block-averaged covariance.

    The independent unit is the MAP: a map's sub-chains all sample the same posterior, so
    with average_over = "sub_chain" its blocks are correlated and counting each as an
    independent draw would understate the error. With C_(m) the block average leaving out
    map m (`leave_one_out`, the C_(m) stack), the widths are recomputed as
    sigma_(m) = sqrt(C_(m),ii) and
        err_sigma^2 = (M - 1) / M * sum_m (sigma_(m) - mean_m sigma_(m))^2 ,
    the jackknife of the widths themselves (no linear propagation from C). err_C is the same
    jackknife applied to each covariance element, for the dashed +/- contours, and
    triangle_plot jackknifes the degeneracy angles over `leave_one_out`. Needs >= 3 maps.
    """
    unique_maps = np.unique(block_map_idxs)
    num_maps = len(unique_maps)
    if num_maps < 3:
        raise ValueError(f"the jackknife needs at least 3 maps, found {num_maps}")
    leave_one_out = np.array([np.mean(block_cov_mats[block_map_idxs != m], axis = 0)
                              for m in unique_maps])
    factor = (num_maps - 1) / num_maps
    cov_err = np.sqrt(factor * np.sum((leave_one_out - np.mean(leave_one_out, axis = 0))**2,
                                      axis = 0))
    sigmas_loo = np.sqrt(np.array([np.diag(cov) for cov in leave_one_out]))
    sigma_err = np.sqrt(factor * np.sum((sigmas_loo - np.mean(sigmas_loo, axis = 0))**2,
                                        axis = 0))
    return cov_err, sigma_err, leave_one_out

def correlation_from_covariance(cov_mat):
    sigmas = np.sqrt(np.diag(cov_mat))
    return cov_mat / np.outer(sigmas, sigmas)

def plot_covariance_and_correlation(cov_mat, param_names, label, plot_dir, title = None):
    """covariance_matrix_<label>.png and correlation_matrix_<label>.png (every entry printed
    in its box) for one posterior: the chains' (EMPIRICAL_LABEL) or a forecast's. `title`
    names the posterior in the plot titles (`label` when None)."""
    title = label if title is None else title
    plot_annotated_heat_map(cov_mat, param_names, f"{title} covariance matrix",
                            f"covariance_matrix_{label}.png", plot_dir)
    plot_annotated_matrix(correlation_from_covariance(cov_mat), param_names,
                          f"{title} correlation matrix", plot_dir,
                          f"correlation_matrix_{label}.png",
                          norm = matplotlib.colors.Normalize(vmin = -1, vmax = 1),
                          fmt = "{:+.3f}")

def get_avg_covariance_matrix(raw_chains, num_sampled, average_over = AVERAGE_OVER):
    """Averaged posterior covariance of the sampled parameters, with its jackknife error.

    raw_chains maps each parameter name to its list over maps of lists of sub-chains (the
    output of per_param_analysis). Every sampled parameter is written to its history file
    once per iteration, and per_param_analysis walks maps and chains in the same order for
    each parameter, so the columns of any one block line up row-by-row.

    average_over selects which blocks are averaged - "sub_chain" or "map" (AVERAGE_OVER, the
    default for a whole run); see covariance_blocks_to_average, which is where the
    difference between them is documented.

    Returns (avg_cov_mat, param_names, cov_err, leave_one_out, sigma_err), the last three
    from measured_errors: cov_err the element-wise error of the averaged covariance (the
    dashed contours), sigma_err the error of each width sqrt(C_ii), and leave_one_out the
    delete-one-MAP stack of the averaged covariance, which triangle_plot turns into the
    error on the measured degeneracy angles.
    """
    param_names = list(raw_chains.keys())
    num_params = len(param_names)
    avg_cov_mat = np.zeros((num_params, num_params))
    #per-block covariances and the map each came from, for the leave-one-map-out errors
    block_cov_mats = []
    block_map_idxs = []

    for map_idx, columns in covariance_blocks_to_average(raw_chains, param_names, average_over):
        #sanity check
        if len(columns) == num_sampled:
            #CLIP to min length in case we have SCP race conditions... This should
            #not matter once the chains are fully finished running
            min_length = min(len(column) for column in columns)
            columns = [column[:min_length] for column in columns]

            cov_mat = np.cov(np.column_stack(columns), rowvar = False)
            avg_cov_mat += cov_mat
            block_cov_mats.append(cov_mat)
            block_map_idxs.append(map_idx)

    if not block_cov_mats:
        raise ValueError(f"no block averaging over {average_over!r} carried all "
                         f"{num_sampled} sampled parameters, so there is nothing to "
                         f"average; check that every parameter has the same maps and "
                         f"sub-chains on disk")

    avg_cov_mat /= len(block_cov_mats)
    block_cov_mats = np.array(block_cov_mats)
    block_map_idxs = np.array(block_map_idxs)
    cov_err, sigma_err, leave_one_out = measured_errors(block_cov_mats, block_map_idxs)
    print(f"Averaged {len(block_cov_mats)} covariance matrices, one per {average_over}, from "
          f"{len(np.unique(block_map_idxs))} maps")
    print(f"Jackknife error (over maps) of the averaged covariance elements, "
          f"err_covar / |covar|:")
    print(cov_err / np.abs(avg_cov_mat))
    print(f"Measured widths (order {param_names}): {np.sqrt(np.diag(avg_cov_mat))} "
          f"+/- {sigma_err} (jackknife)")
    print("Average Correlation matrix (rows / columns ordered as", param_names, ")")
    print(correlation_from_covariance(avg_cov_mat))
    return avg_cov_mat, param_names, cov_err, leave_one_out, sigma_err

#Contour styling for the triangle plots, one per forecast: the chains always blue, the
#forecast always red, each contour filled translucently under an opaque outline so the
#overlap stays legible
MEASURED_COLOUR = "tab:blue"
FORECAST_COLOUR = "tab:red"
#face alpha of the filled contours / Gaussians
FILL_ALPHA = 0.18
#edge length of one triangle-plot panel, in inches
PANEL_INCHES = 3.6
ANNOTATION_FONTSIZE = 8.5
#per forecast (fisher_forecast's nphi_source): the short symbol in the annotations and the
#legend / title description
FORECAST_LABELS = {
    #the two reported bounds: N_phi from the box QE with the unlensed C_TT in its filter
    #(upper bound on the information) and with the iterated partially delensed C_TT (lower)
    "qe_unlensed": ("QE unl", "upper bound (QE, unlensed filter)"),
    "qe_delensed": ("QE del", "lower bound (QE, delensed filter)"),
    "map_joint": ("mapj", "map_joint N_phi"),
    "sample_joint": ("samp", "sampler N_phi"),
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


def ratio_text(numerator, denominator):
    """numerator / denominator to 3 decimals, or inf when the denominator is not positive."""
    if denominator <= 0:
        return "$\\infty$"
    return f"{numerator / denominator:.3f}"

def format_sigma(symbol, sigma, error = None):
    text = f"$\\sigma_{{{symbol}}}$ = {sigma:.3e}"
    if error is not None:
        text += f" $\\pm$ {error:.1e}"
    return text

def degeneracy_angle(block, scales):
    """Major-axis angle in degrees, in (-90, 90], of a 2x2 (x, y) covariance block.

    The angle is measured counter-clockwise from +x AFTER dividing each axis by `scales`.
    Any such angle depends on the axis scaling; triangle_plot passes each parameter's panel
    half-width (the panels are square), so the number is the angle the ellipse is DRAWN at
    and every matrix in a pair shares one scale. Caveat: the panel width follows the widest
    matrix plotted, so adding a broader forecast changes every angle in that panel; compare
    angles within one plot, and use the correlation coefficient for a scale-free number.

    Scales that were tried and rejected: raw units and the fixed PARAM_SIGMA both put most
    pairs at ~0 or ~90 degrees (PARAM_SIGMA is ~40x the posterior width in theta_MC_100 but
    ~2x in omch2); each matrix's own sigmas give a unit-diagonal block whose major axis is
    always exactly +/-45 degrees, i.e. only the sign of the correlation.
    """
    scaled = np.asarray(block) / np.outer(scales, scales)
    return np.degrees(0.5 * np.arctan2(2 * scaled[0, 1], scaled[0, 0] - scaled[1, 1]))

def wrap_angle_difference(difference):
    """An angle difference in degrees folded into (-90, 90] - an axis has no direction, so a
    major axis at +89 and one at -89 are 2 degrees apart, not 178."""
    return -((-np.asarray(difference) + 90) % 180 - 90)

def degeneracy_angle_error(leave_one_out, idx, scales, reference_angle):
    """Delete-one jackknife error, in degrees, of degeneracy_angle on the (idx[0], idx[1]) pair.

    `leave_one_out` is the stack of leave-one-map-out covariance matrices of the chains,
    None when there are too few maps to jackknife. Each is turned into an angle at the SAME
    `scales` as the central value, so this is the error of the angle as drawn with the panel
    scaling held fixed (the scaling's own uncertainty is not included). The deviations are
    taken relative to `reference_angle` and wrapped, so a pair sitting near +/-90 degrees
    does not pick up a spurious 180-degree spread.
    """
    if leave_one_out is None:
        return None
    angles = np.array([degeneracy_angle(cov_mat[np.ix_(idx, idx)], scales)
                       for cov_mat in leave_one_out])
    deviations = wrap_angle_difference(angles - reference_angle)
    n = len(deviations)
    return np.sqrt((n - 1) / n * np.sum((deviations - np.mean(deviations))**2))

def format_angle(angle, error = None):
    text = f"{angle:+.1f}"
    if error is not None:
        text += f" $\\pm$ {error:.1f}"
    return text + "$^\\circ$"

def stacked_text(ax, lines, x, y_top, line_step, ha = "left"):
    """Write (text, colour) lines downward from y_top in axes coordinates."""
    for k, (text, colour) in enumerate(lines):
        ax.text(x, y_top - k * line_step, text, color = colour, transform = ax.transAxes,
                fontsize = ANNOTATION_FONTSIZE, ha = ha, va = "top")
    return


def triangle_plot(measured_cov_mat, nphi_source, t_block, forecast_cov_mat, param_names, means,
                  measured_cov_sigma, measured_leave_one_out, measured_sigma_err, plot_dir,
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
    diagonal panel is annotated with sigma_measured +/- its error, sigma_forecast and the
    ratio sigma_measured / sigma_forecast with the ratios at sigma_measured +/- its error
    (max / min): above 1 means the sampler is doing worse than the forecast says the data
    allows, near 1 means it is saturating it, below 1 means it is extracting information the
    forecast does not model (or the chain has not converged). Each vacant upper-triangle cell
    (row a, column b) reports the degeneracy angle of the lower panel it mirrors (row b,
    column a) for both, as drawn - see degeneracy_angle for the scaling and its caveat.

    `forecast_cov_mat` MUST already be permuted into param_names order - see
    get_forecasted_covariance, which does it. The forecast is exact given its model and
    carries no error bars; only the measured posterior does, all from the delete-one-map
    jackknife of measured_errors: `measured_cov_sigma` (the element-wise error of the
    averaged covariance) adds dashed (unfilled) blue lower / upper limits at
    measured_cov_mat -/+ measured_cov_sigma, `measured_sigma_err` is each width's error, and
    `measured_leave_one_out` gives every measured degeneracy angle a +/- error
    (degeneracy_angle_error), with the printed per-pair angles quoting the measured -
    forecast difference in units of it.
    """
    measured = np.asarray(measured_cov_mat)
    forecast = np.asarray(forecast_cov_mat)
    measured_cov_sigma = np.asarray(measured_cov_sigma)
    means = np.asarray(means)
    n = len(param_names)
    short, description = forecast_labels(nphi_source)
    description += f", {T_BLOCKS[t_block]}"
    forecast_label = forecast_file_label(nphi_source, t_block)
    if forecast.shape != measured.shape:
        raise ValueError(f"measured covariance is {measured.shape} but the "
                         f"'{forecast_label}' forecast is {forecast.shape}; they must cover "
                         f"the same {n} parameters in the same order ({param_names})")

    #(covariance, colour, filled, linestyle) - the central contours filled, the measured
    #+/- error limits as dashed outlines only
    layers = [(measured, MEASURED_COLOUR, True, "-"),
              (measured - measured_cov_sigma, MEASURED_COLOUR, False, "--"),
              (measured + measured_cov_sigma, MEASURED_COLOUR, False, "--"),
              (forecast, FORECAST_COLOUR, True, "-")]

    #pad the axes out past the widest requested contour of the WIDER matrix, so whichever
    #one is broader still fits
    pad = max(sigmas) + 1
    widest = np.maximum(np.sqrt(np.diag(measured + measured_cov_sigma)),
                        np.sqrt(np.diag(forecast)))
    limits = [(means[i] - pad * widest[i], means[i] + pad * widest[i]) for i in range(n)]

    #annotation line spacing in axes fraction: one text line over the panel's height (the
    #axes are forced square below, and constrained layout leaves them ~80% of their cell)
    line_step = 1.35 * ANNOTATION_FONTSIZE / (0.8 * PANEL_INCHES * 72)
    #the diagonal's five-line text block sits above the Gaussians; raise the y limit so the
    #unit-height curves stay entirely below it
    text_fraction = 0.04 + 5 * line_step
    diag_ylim = 1.05 / (1 - text_fraction)

    print(f"Degeneracy angles vs {forecast_label}, degrees (major axis from +x as drawn, panel "
          f"scales held fixed; measured +/- delete-one-map jackknife):")
    fig, axes = plt.subplots(n, n, figsize = (PANEL_INCHES * n, PANEL_INCHES * n),
                             squeeze = False, layout = "constrained")
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            #the upper triangle carries no panel, only the mirrored pair's degeneracy angles
            if j > i:
                ax.axis("off")
                #cell (i, j) mirrors the lower panel (j, i): x = param i, y = param j
                scales = widest[[i, j]]
                measured_angle = degeneracy_angle(measured[np.ix_([i, j], [i, j])], scales)
                measured_error = degeneracy_angle_error(measured_leave_one_out, [i, j], scales,
                                                        measured_angle)
                lines = [("degeneracy angle", "black"),
                         (f"x: {param_names[i]},  y: {param_names[j]}", "black"),
                         (f"measured  {format_angle(measured_angle, measured_error)}",
                          MEASURED_COLOUR)]
                angle = degeneracy_angle(forecast[np.ix_([i, j], [i, j])], scales)
                #signed ratio: negative when the two tilt opposite ways; it blows up as the
                #forecast ellipse aligns with an axis, so an exactly axis-aligned one gets no
                #ratio at all
                ratio = f"{measured_angle / angle:+.2f}" if angle != 0 else "undefined"
                lines.append((f"{short}  {format_angle(angle)}   meas/{short} = {ratio}",
                              FORECAST_COLOUR))
                difference = wrap_angle_difference(measured_angle - angle)
                print(f"    x {param_names[i]}, y {param_names[j]}: measured "
                      f"{measured_angle:+.2f} +/- {measured_error:.2f}  {forecast_label} "
                      f"{angle:+.2f} (meas - {forecast_label} = {difference:+.2f}, "
                      f"{difference / measured_error:+.2f} sigma)")
                y_top = 0.5 + 0.5 * len(lines) * line_step
                stacked_text(ax, lines, 0.5, y_top, line_step, ha = "center")
                continue
            ax.set_box_aspect(1)
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
                sigma_meas = np.sqrt(measured[i, i])
                error_meas = measured_sigma_err[i]
                sigma_fore = np.sqrt(forecast[i, i])
                lines = [(format_sigma("meas", sigma_meas, error_meas), MEASURED_COLOUR),
                         (format_sigma(short, sigma_fore), FORECAST_COLOUR),
                         (f"$\\sigma_{{meas}}/\\sigma_{{{short}}}$ = "
                          f"{sigma_meas / sigma_fore:.3f}", "black"),
                         (f"max = {ratio_text(sigma_meas + error_meas, sigma_fore)}", "black"),
                         (f"min = {ratio_text(sigma_meas - error_meas, sigma_fore)}", "black")]
                stacked_text(ax, lines, 0.03, 0.97, line_step)
                ax.set_ylim(0, diag_ylim)
                ax.set_yticks([])
            else:
                ax.set_ylim(*limits[i])
            ax.set_xlim(*limits[j])
            ax.grid(alpha = 0.2)
            ax.tick_params(labelsize = 8)
            if i == n - 1:
                ax.set_xlabel(param_names[j])
                plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right")
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(param_names[i])
            elif i != j:
                ax.set_yticklabels([])

    contours = ", ".join(f"{k}$\\sigma$" for k in sorted(sigmas))
    #the filled contours get a filled swatch, the dashed measured +/- limits a dashed line
    def filled_handle(colour, linestyle, label):
        return Patch(facecolor = to_rgba(colour, FILL_ALPHA), edgecolor = colour,
                     ls = linestyle, lw = 1.6, label = label)
    handles = [filled_handle(MEASURED_COLOUR, "-", "measured (chains)"),
               Line2D([0], [0], color = MEASURED_COLOUR, lw = 2, ls = "--",
                      label = "measured $\\pm$ 1 jackknife"),
               filled_handle(FORECAST_COLOUR, "-", f"Fisher forecast: {description}")]
    #the upper triangle holds the degeneracy angles, so the legend goes below the grid, where
    #constrained layout reserves room for it (above, it collides with the suptitle)
    fig.legend(handles = handles, loc = "outside lower center", fontsize = 11, ncol = 3)
    title = (f"Posterior vs Fisher Forecast ({forecast_title(nphi_source, t_block)}; "
             f"Gaussian approximation, {contours})")
    if None not in (nside, theta_pix, noise, pol):
        title += (f"\n{pol}, {noise:g} $\\mu$K-arcmin, {theta_pix:g}' pixels, "
                  f"{nside}$^2$ pixels")
    fig.suptitle(title, fontsize = 13)
    plt.savefig(plot_dir + f"triangle_plot_{forecast_label}.png", dpi = 150,
                bbox_inches = "tight")
    plt.close(fig)
    return

def get_means_per_param(all_chains, param_names):
    """The mean of every post-burn-in sample of each parameter, over all maps and chains."""
    return [np.mean(np.concatenate([chain for chains in all_chains[name] for chain in chains]))
            for name in param_names]

def report_forecast_comparison(measured_cov_mat, forecasts, param_names, measured_sigma_err):
    """Print each parameter's measured sigma beside every forecast, and flag the bracket.

    `forecasts` maps (nphi_source, t_block) to a forecast covariance. Each ratio is followed
    by the ratios at measured sigma +/- 1 jackknife error (`measured_sigma_err`,
    get_avg_covariance_matrix's last return).

    "between" is the expected outcome: each measured sigma should sit between the
    "qe_unlensed" (upper information bound, smaller sigma) and "qe_delensed" (lower bound)
    forecasts, checked separately for each T block both bounds were forecast with. Anything
    outside that bracket is worth investigating - either the chain has not converged, or the
    forecast is not describing this data set.
    """
    print("Measured vs forecast marginal sigmas (order:", param_names, ")")
    for i, name in enumerate(param_names):
        measured = np.sqrt(measured_cov_mat[i, i])
        print(f"    {name}: measured {measured:.4e}")
        for (nphi_source, t_block), cov_mat in forecasts.items():
            forecasted = np.sqrt(cov_mat[i, i])
            upper = measured + measured_sigma_err[i]
            lower = measured - measured_sigma_err[i]
            print(f"        {forecast_file_label(nphi_source, t_block)} {forecasted:.4e} "
                  f"(ratio {measured / forecasted:.3f}, +1 jackknife {upper / forecasted:.3f}, "
                  f"-1 jackknife {lower / forecasted:.3f})")

    #the correlations bracket too, and that is where the bounds can disagree on SIGN
    def correlation(matrix, i, j):
        return matrix[i, j] / np.sqrt(matrix[i, i] * matrix[j, j])
    for t_block in T_BLOCKS:
        upper_key, lower_key = ("qe_unlensed", t_block), ("qe_delensed", t_block)
        if upper_key not in forecasts or lower_key not in forecasts:
            continue
        print(f"The qe_unlensed - qe_delensed bracket, {T_BLOCKS[t_block]}:")
        for i, name in enumerate(param_names):
            measured = np.sqrt(measured_cov_mat[i, i])
            low = np.sqrt(forecasts[upper_key][i, i])
            high = np.sqrt(forecasts[lower_key][i, i])
            inside = min(low, high) <= measured <= max(low, high)
            print(f"    {name}: sigma {'inside' if inside else 'OUTSIDE'} the bracket "
                  f"[{min(low, high):.4e}, {max(low, high):.4e}]")
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                measured = correlation(measured_cov_mat, i, j)
                low = correlation(forecasts[upper_key], i, j)
                high = correlation(forecasts[lower_key], i, j)
                inside = min(low, high) <= measured <= max(low, high)
                print(f"    r({param_names[i]}, {param_names[j]}): measured "
                      f"{measured:+.3f}  qe_unlensed {low:+.3f}  qe_delensed {high:+.3f}  "
                      f"-> {'inside' if inside else 'OUTSIDE'} the bracket")
    return

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
    ns, theta_MC_100, logA), which is the order ground_truth_values is written in here, so
    the permutation below is normally the identity. It is kept because nothing enforces that
    agreement: reorder ground_truth_values and, without it, the triangle plot would silently
    pair each measured parameter with a different forecasted one.

    nside / theta_pix / noise must match the run that produced the chains - see the values
    at the top of sample_lcdm.sh.
    """
    fisher, names = forecast(nside, theta_pix, noise, is_sampled, GROUND_TRUTH,
                             nphi_source = nphi_source,
                             qe_response = FORECAST_QE_RESPONSE.get(nphi_source, "unlensed"),
                             phi_noise = phi_noise,
                             delensed_covariance = delensed_covariance,
                             camb_lmax = FORECAST_CAMB_LMAX, l_knee = 0, beam_fwhm = 0)
    covariance = covariance_from_fisher(fisher, names)
    missing = [name for name in param_names if name not in names]
    if missing:
        raise ValueError(f"the forecast did not cover {missing}; was_sampled and the "
                         f"chains that were actually found disagree on which parameters "
                         f"were sampled")
    order = [names.index(name) for name in param_names]
    return covariance[np.ix_(order, order)]

def joint_param_analysis(all_chains, num_sampled, nside, theta_pix, noise, is_sampled, pol,
                         fisher_dir):
    """The measured posterior against every forecast, into fisher_dir: the chains'
    covariance_matrix_empirical.png and correlation_matrix_empirical.png at its root, and per
    forecast a <nphi_source>/<t_block>/ folder (forecast_plot_dir) with its triangle plot,
    covariance and correlation matrices and the chains-over-forecast covariance ratio.

    Every source in FORECAST_NPHI_SOURCES is forecast with CAMB's delensed T block
    ("camb_delensed") and, when FORECAST_DELENSED_COVARIANCE has a file for it at this noise
    level, with that file's empirical one too ("empirically_delensed"). A measured-N_phi
    source whose FORECAST_PHI_NOISE file is missing is skipped altogether."""
    print("\n")
    print(f"Beginning joint parameter analysis")
    (measured_cov_mat, param_names, measured_cov_sigma, measured_leave_one_out,
     measured_sigma_err) = get_avg_covariance_matrix(all_chains, num_sampled)
    means = get_means_per_param(all_chains, param_names)
    plot_covariance_and_correlation(measured_cov_mat, param_names, EMPIRICAL_LABEL, fisher_dir,
                                    title = "Chains (empirical)")
    #both bounds: the measured posterior should sit BETWEEN them, in the widths and in the
    #degeneracy directions alike
    forecasts = {}
    for nphi_source in FORECAST_NPHI_SOURCES:
        phi_noise = forecast_input(FORECAST_PHI_NOISE, nphi_source, noise)
        if nphi_source in FORECAST_PHI_NOISE and phi_noise is None:
            print(f"skipping the {nphi_source} forecasts: no N_phi file at "
                  f"{FORECAST_PHI_NOISE[nphi_source].format(tag = noise_tag(noise))}")
            continue
        t_block_files = {"camb_delensed": None}
        delensed_covariance = forecast_input(FORECAST_DELENSED_COVARIANCE, nphi_source, noise)
        if delensed_covariance is not None:
            t_block_files["empirically_delensed"] = delensed_covariance
        elif nphi_source in FORECAST_DELENSED_COVARIANCE:
            print(f"no empirically delensed {nphi_source} forecast: no file at "
                  f"{FORECAST_DELENSED_COVARIANCE[nphi_source].format(tag = noise_tag(noise))}")
        for t_block, path in t_block_files.items():
            forecasts[(nphi_source, t_block)] = get_forecasted_covariance(
                nside, theta_pix, noise, is_sampled, param_names, nphi_source = nphi_source,
                phi_noise = phi_noise, delensed_covariance = path)
    report_forecast_comparison(measured_cov_mat, forecasts, param_names, measured_sigma_err)
    for (nphi_source, t_block), forecasted_cov_mat in forecasts.items():
        plot_dir = forecast_plot_dir(fisher_dir, nphi_source, t_block)
        label = forecast_file_label(nphi_source, t_block)
        title = forecast_title(nphi_source, t_block)
        triangle_plot(measured_cov_mat, nphi_source, t_block, forecasted_cov_mat, param_names,
                      means, measured_cov_sigma, measured_leave_one_out, measured_sigma_err,
                      plot_dir, nside = nside, theta_pix = theta_pix, noise = noise, pol = pol)
        plot_covariance_and_correlation(forecasted_cov_mat, param_names, label, plot_dir,
                                        title = f"Forecast ({title})")
        plot_annotated_heat_map(measured_cov_mat / forecasted_cov_mat, param_names,
                                f"Chains / forecast covariance ({title})",
                                f"covariance_ratio_{EMPIRICAL_LABEL}_over_{label}.png",
                                plot_dir)
    return

# ── Phi-noise and delensed-spectra comparison plots ────────────────────────

#(directory glob under COVARIANCE_DATA/<run kind>/ with the noise tag left open, legend label,
#color, marker) of the empirical delensed spectra. The second pattern leaves the stencil step
#open and also matches "terative" spelled without its i
EMPIRICAL_DELENSED_RUNS = (
    ("delensed_covariance_output_{tag}_nphi_unlensedx2",
     "Empirical Delensed N_Phi cfl_tt --> cf_tt", "C0", "o"),
    ("delensed_covariance_output_fiducial_wiener_{tag}_0dot*_sigma_*terative_delens_qe_gradient",
     "Empirical Delensed Iteratively Delensed N_phi Response = Gradient", "C1", "s"),
    ("delensed_covariance_output_map_joint_noise_{tag}",
     "Empirical Delensed N_Phi = MAP_Joint Noise", "C2", "^"),
    ("delensed_covariance_output_{tag}_sample_joint_neff",
     "Empirical Delensed N_Phi = Sample_Joint Effective Noise", "C5", "D"),
)
#the largest band centre the delensed-spectra plot draws: past CAMB's own range the model
#spectra are a log-log extrapolation and the box's lensed power aliases at the grid's corner
SPECTRA_ELL_MAX = 4000.0

def check_spectra_box(path, data, nside, theta_pix, noise):
    """Warn when a measurement used for the spectra plots sits on another box than the chains."""
    measured = (int(data["nside"]), float(data["theta_pix"]), float(data["noise_level"]))
    if measured != (int(nside), float(theta_pix), float(noise)):
        print(f"  WARNING: {path} was measured at (nside, theta_pix, noise) = {measured}, "
              f"not the chains' {(nside, theta_pix, noise)}")

def band_median(matrix, index, n_band):
    """Median over each band's finite modes (NaN for an empty band)."""
    values = np.full(n_band, np.nan)
    for b in range(n_band):
        inside = matrix[(index == b) & np.isfinite(matrix)]
        if inside.size:
            values[b] = np.median(inside)
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
                                      l_knee, beam_fwhm, 10_000, verbose = False,
                                      qe_response = "unlensed")[2]
    matrices = []
    for label, color, linestyle, filter_tt in (
            ("QE (box matrix: unlensed response, lensed filter, no NPHI_FAC)", "C3", "--",
             cls["total_TT"]),
            ("QE, iterative delensing (unlensed response)", "C4", "-.", delensed_tt),
            ("QE, unlensed $C^{TT}$ in the filter (unlensed response)", "C5", "-.",
             cls["scalar_TT"])):
        matrices.append((label, color, linestyle, np.asarray(qe_noise_matrix(
            cls, nside, pix_width, ell_grid, noise_level, l_knee, beam_fwhm, 10_000,
            filter_tt = filter_tt, qe_response = "unlensed"))))
    cphi = np.asarray(prior_covariances(params, nside, pix_width, ell_grid)[1])
    return cphi, matrices

def plot_phi_noise_vs_qe(nside, theta_pix, noise, out_dir):
    """Per-mode phi noise spectra against the box QE matrix at this noise level:
    phi_noise_vs_qe_per_mode_<tag>.png in out_dir.

    Each noise level is anchored on its sampler_effective_noise.npz (the sampler's b_only
    noise C_phi (1/r^2 - 1), r^2 = <phi_bar phi*> / <|phi|^2>), which fixes the box, the bands
    and the cosmology; map_joint's N_eff = C_phi (1/r^2 - 1) is the `phi_noise_fid` of the
    lower_qe_bound merged delensed_covariance.npz. Top panel: information-weighted band means
    with CAMB's C_L^phiphi and the QE matrices of qe_noise_matrices. Bottom panel: every mode's
    N_k / N_QE,k as a faint point, with the band median on top.
    """
    tag = noise_tag(noise)
    chain_path = os.path.join(SAMPLER_NOISE_DATA, f"sampler_effective_noise_output_mean_chains_{tag}",
                              "sampler_effective_noise.npz")
    if not os.path.exists(chain_path):
        print(f"no {chain_path}: phi_noise_vs_qe_per_mode_{tag}.png skipped")
        return
    chain = np.load(chain_path)
    check_spectra_box(chain_path, chain, nside, theta_pix, noise)
    data_nside, data_theta_pix = int(chain["nside"]), float(chain["theta_pix"])
    ell_grid = np.asarray(gen_ell_grid(data_nside, data_theta_pix)[0])
    use = used_modes(ell_grid)
    edges = np.asarray(chain["band_edges"])
    n_band = len(edges) - 1
    index = band_index(ell_grid, edges, use)
    centres = np.asarray(chain["band_ells"])
    params = {name: float(chain["params"][i]) for i, name in enumerate(PARAM_ORDER)}
    cphi, qe_matrices = qe_noise_matrices(params, data_nside, data_theta_pix,
                                          float(chain["noise_level"]), float(chain["l_knee"]),
                                          float(chain["beam_fwhm"]))
    nphi_qe = qe_matrices[0][3]
    series = [("b_only $\\langle\\bar\\phi\\,\\phi^*\\rangle/\\langle|\\phi|^2\\rangle$ "
               f"({int(chain['n_realizations'])} chains, burn-in {int(chain['burn_in'])})",
               "C1", "^", np.asarray(chain["nphi_b_only_raw"]))]
    map_joint_path = os.path.join(COVARIANCE_DATA, "lower_qe_bound",
                                  f"delensed_covariance_output_{tag}_nphi_unlensedx2",
                                  MERGED_NAME)
    if os.path.exists(map_joint_path):
        map_joint = np.load(map_joint_path, allow_pickle = True)
        check_spectra_box(map_joint_path, map_joint, data_nside, data_theta_pix,
                          float(chain["noise_level"]))
        series.append((f"map_joint $N_{{\\rm eff}}$ = $C_\\phi(1/r^2-1)$ "
                       f"({int(map_joint['n_realizations'])} realizations)", "C2", "s",
                       np.asarray(map_joint["phi_noise_fid"])))
    else:
        print(f"no {map_joint_path}: map_joint N_eff left out of the phi noise plot")

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
    figure.suptitle(f"phi noise per mode relative to the QE | nside {data_nside}, "
                    f"{data_theta_pix:g}', {noise:g} uK-arcmin", fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"phi_noise_vs_qe_per_mode_{tag}.png")
    figure.savefig(path, dpi = 150)
    plt.close(figure)
    print(f"wrote {path}")

def plot_delensed_spectra_comparison(nside, theta_pix, noise, out_dir,
                                     delta_ell = DEFAULT_DELTA_ELL):
    """Unlensed / lensed / delensed temperature spectra, CAMB against the box, at this noise
    level: delensed_spectra_comparison_<tag>.png in out_dir.

    Series, all band-averaged over the same |l| annuli of the rfft grid: CAMB's unlensed,
    lensed and iteratively delensed TT (get_partially_lensed_cls at the Alens_L
    fisher_forecast.iterative_delensing converges to, unlensed response), and the empirical
    delensed spectra (`delensed_fid`) of the merged runs in EMPIRICAL_DELENSED_RUNS, drawn
    directly (so they carry the realizations' cosmic variance). Top panel: D_l = l (l + 1) C_l / 2 pi. Bottom panel: every series over CAMB
    unlensed.
    """
    tag = noise_tag(noise)
    empirical = []
    for pattern, label, color, marker in EMPIRICAL_DELENSED_RUNS:
        matches = sorted(glob.glob(os.path.join(COVARIANCE_DATA, "*", pattern.format(tag = tag),
                                                MERGED_NAME)))
        if not matches:
            print(f"no {pattern.format(tag = tag)}/{MERGED_NAME}: {label} left out")
            continue
        if len(matches) > 1:
            print(f"several runs match {pattern.format(tag = tag)}, using {matches[0]}")
        merged = np.load(matches[0], allow_pickle = True)
        check_spectra_box(matches[0], merged, nside, theta_pix, noise)
        empirical.append((label, color, marker, merged))
    if not empirical:
        print(f"no merged delensed covariance at {noise:g} uK: "
              f"delensed_spectra_comparison_{tag}.png skipped")
        return
    first = empirical[0][3]
    for _, _, _, merged in empirical[1:]:
        for key in ("nside", "theta_pix", "noise_level", "l_knee"):
            if float(merged[key]) != float(first[key]):
                raise ValueError(f"{noise:g} uK: the merged delensed covariances disagree on {key}")
        if not np.allclose(merged["params"], first["params"], rtol = 1e-12, atol = 0):
            raise ValueError(f"{noise:g} uK: the merged delensed covariances sit at different "
                             f"cosmologies")

    data_nside, data_theta_pix = int(first["nside"]), float(first["theta_pix"])
    params = {name: float(first["params"][i]) for i, name in enumerate(PARAM_ORDER)}
    ell_grid, pix_width = gen_ell_grid(data_nside, data_theta_pix)
    weights = fourier_weights(data_nside)
    edges = band_edges(ell_grid, delta_ell)

    def band(matrix):
        centres, values, _ = band_average(matrix, ell_grid, weights, edges)
        return centres, values

    def camb_band(cl):
        #CAMB's C_l on the grid exactly as the merge puts it there (C_l / pix_width^2)
        ells = jnp.arange(2, 2 + cl.shape[0]).astype(jnp.float64)
        return band(np.asarray(covar_matrix_from_cls(data_nside, pix_width, ell_grid, ells, cl,
                                                     origin_value = 0)))[1]

    cls = camb_cls_at_params(params)
    #the converged delensed C_TT of the iteration: box QE matrix (no NPHI_FAC), unlensed
    #response, beam 0 as the data are simulated
    delensed_tt = iterative_delensing(params, cls, data_nside, pix_width, ell_grid,
                                      float(first["noise_level"]), float(first["l_knee"]),
                                      0.0, 10_000, verbose = False,
                                      qe_response = "unlensed")[2]

    centres = band(np.asarray(first["delensed_fid"]))[0]
    camb_unlensed = camb_band(cls["scalar_TT"])
    #grid units C_l / pix_width^2 -> D_l
    to_dl = centres * (centres + 1) / (2 * np.pi) * float(pix_width)**2
    series = [("CAMB unlensed", "0.2", "-", None, camb_unlensed),
              ("CAMB lensed", "C3", "-", None, camb_band(cls["total_TT"])),
              ("CAMB iteratively delensed", "C4", "--", None, camb_band(delensed_tt))]
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
    figure.suptitle(f"Temperature spectra | nside {data_nside}, {data_theta_pix:g}', "
                    f"{noise:g} uK-arcmin | bands of {delta_ell:g}", fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"delensed_spectra_comparison_{tag}.png")
    figure.savefig(path, dpi = 150)
    plt.close(figure)
    print(f"wrote {path}")

def main(file_name, num_maps, num_chains, map_pre_factor, ground_truth_values,
         was_sampled, default_burn_in, nside, theta_pix, noise, pol):
    """Every plot of one run goes under its noise level's folder (plot_dirs),
    lcdm_chain_plots/noise_level_<tag>/: one subfolder per sampled parameter, fisher/ (the
    covariance / correlation / triangle plots, one <nphi_source>/<t_block>/ folder per
    forecast) and spectra_plots/."""
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
        joint_param_analysis(all_chains, len(sampled_names), nside, theta_pix, noise,
                             was_sampled, pol, fisher_plot_dir)

    #the phi noise and the delensed spectra at the chains' noise level
    print("\n")
    print(f"Plotting the phi noise and delensed spectra at {noise:g} uK-arcmin")
    plot_phi_noise_vs_qe(nside, theta_pix, noise, spectra_plot_dir)
    plot_delensed_spectra_comparison(nside, theta_pix, noise, spectra_plot_dir)
    return

if __name__ == "__main__":

    num_maps = 10
    num_chains = 5
    default_burn_in = 1000
    map_pre_factor = 234567
    #file_name = "omch2_logA_theta_MC_100_joint_inference_08_27_26"
    #file_name = "omch2_logA_theta_MC_100_joint_inference_08_31_26"
    #file_name = "omch2_logA_theta_MC_100_joint_inference_08_31_26_high_noise"
    #file_name = "omch2_logA_theta_MC_100_joint_inference_09_01_26_high_noise"
    #file_name = "omch2_logA_theta_MC_100_joint_inference_09_02_26_high_noise"
    file_name = "omch2_logA_theta_MC_100_joint_inference_09_02_26_low_noise"
    #file_name = "joint_inference_09_15_26"
    #file_name = "joint_inference_50_maps_2dot5_uk_noise_09_29_2026"
    #file_name = "joint_inference_50_maps_10_uk_noise_09_29_2026"

    #the Fisher forecast the triangle plot compares against, and the phi-noise / delensed-
    #spectra plots, are only meaningful at the box and noise level the chains ran on - MATCH
    #sample_lcdm.sh
    nside = 128
    theta_pix = 2.5
    noise = 2.5
    #"I", "P" or "IP", as pol in sample_lcdm.sh; only labels the triangle plots
    pol = "I"

    ground_truth_values = {}
    ground_truth_values["omch2"] = 0.109381
    ground_truth_values["ombh2"] = 0.022386
    ground_truth_values["ns"] = 0.959814
    ground_truth_values["theta_MC_100"] = 1.031732
    ground_truth_values["logA"] = 3.218387

    was_sampled = {}
    was_sampled["omch2"] = True
    was_sampled["ombh2"] = False
    was_sampled["ns"] = False
    was_sampled["theta_MC_100"] = True
    was_sampled["logA"] = True

    #NOTE we discard chains whose total length is shorter than the requested burn-in AND we discard
    #whole maps if their individual subchains did not converge in time (i.e. the R_hat for that map and
    #parameter combo was greater than some R_hat threshold)... these are removed from
    #both the "raw" chains used for covariance matrix calculations and from the
    #processed chains used for marginal calculations
    main(file_name, num_maps, num_chains, map_pre_factor,
         ground_truth_values, was_sampled, default_burn_in,
         nside, theta_pix, noise, pol)
