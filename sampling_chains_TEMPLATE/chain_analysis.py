import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import gmean
import os
from scipy.stats import gaussian_kde
from scipy.stats import mode
from cmb_lensing.util import precision_load
import jax.numpy.fft as jfft
from cmb_lensing.statistics import *
from cmb_lensing.fisher_forecast import annotated_heatmap, forecast, covariance_from_fisher, plot_annotated_matrix, GROUND_TRUTH

BUFFER = 1e-4
GRID_SIZE = 5_000

#which stride prune_chains thins each chain by. True -> the lag just before that chain's
#autocorrelation first crosses zero (first_zero_crossing_lag), False -> its integrated
#autocorrelation time (integrated_autocorrelation_time). flip this to compare the two
USE_ZERO_CROSSING_PRUNE = False

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
    plt.savefig(output_path + f"/{param_name}_all_chain_trace.png", dpi = 150, bbox_inches = "tight")
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
    plt.savefig(output_path + f"/{param_name}_all_chain_marginals.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    return

def single_chain_plots(chains, output_path, ground_truth, param_name):
    for chain_idx, chain in enumerate(chains):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5),
                        gridspec_kw = {"width_ratios": [3, 1]})
        
        ax1.plot(range(len(chain)), chain, lw = 0.7)
        mean = chain.mean()
        ax1.axhline(mean, ls = "--", lw = 0.8, alpha = 0.4, label = f"Mean {mean}")
        ax1.set_xlabel("Iteration"); ax1.set_ylabel("Value")
        ax1.set_title(f"{param_name} Chain {chain_idx} Trace")
        ax1.set_xlim(0, len(chain) - 1)
        ax1.legend()
        grid = np.linspace(chain.min() - BUFFER, chain.max() + BUFFER, GRID_SIZE)
        pdf = gaussian_kde(chain)(grid)
        mode_val = round(grid[np.argmax(pdf)], 7)
        ax2.plot(pdf, grid, color = "black")
        ax2.axhline(mode_val, color = "indianred", label = f"Mode {mode_val}")
        ax2.axhline(ground_truth, color = "green", label = f"Ground Truth {ground_truth}")
        ax2.set_xlabel("Count"); ax2.set_title("Marginal"); ax2.grid(alpha=0.2)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(output_path + f"{param_name}_chain_{chain_idx}.png", dpi=150, bbox_inches="tight")
        plt.close()
    return

def plot_multiplicative_mean(chains, avg_sigma_single, output_path, ground_truth, param_name):

    #compute the geometric mean and arithmetic mean density across chains
    #using a gaussian_kde for each chain
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
    plt.title(f"{param_name} Combined Geometric Distribution")
    plt.legend(fontsize=7.5); plt.grid(alpha=0.2)
    plt.savefig(output_path + f"{param_name} Mean Distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    return

def plot_geometric_mean_from_pdfs(grid, pdfs, output_path, ground_truth, param_name):

    dx = np.diff(grid)

    #--------------------------- multiplicative pdf ---------------------------
    multiplicative_pdf = np.copy(pdfs[0])
    norm = np.sum(((multiplicative_pdf[:-1] + multiplicative_pdf[1:]) * dx)/2)
    multiplicative_pdf /= norm
    for idx in range(1, len(pdfs)):
        multiplicative_pdf *= pdfs[idx]
        norm = np.sum(((multiplicative_pdf[:-1] + multiplicative_pdf[1:]) * dx)/2)
        multiplicative_pdf /= norm


    multiplicative_mode = grid[np.argmax(multiplicative_pdf)]
    multiplicative_mean = np.dot(multiplicative_pdf, grid * dx[0])
    multiplicative_std = np.sqrt(np.sum(dx[0] * multiplicative_pdf * (grid - multiplicative_mean)**2))

    plt.figure(figsize=(13, 5))
    plt.plot(grid, multiplicative_pdf, label="Multiplicative Mean PDF", color = "blue")
    plt.axvline(ground_truth, label = f"Ground Truth = {ground_truth}", color = "green")
    plt.axvline(multiplicative_mode, label = f"Mult. Mode = {round(multiplicative_mode, 7)}", color = "orange")
    plt.axvline(multiplicative_mean, label = f"Mult. Mean = {round(multiplicative_mean, 7)}", color = "pink")
    plt.axvline(multiplicative_mean + multiplicative_std, 
                label = f"+/- 1 Mult. Std = {round(multiplicative_std, 7)}", color = "grey")
    plt.axvline(multiplicative_mean - multiplicative_std, color = "grey")
    z_score = (ground_truth - multiplicative_mean)/multiplicative_std
    plt.plot([], [], label=f"Z-score = {z_score}")
    plt.xlabel("Value"); plt.ylabel("Density")
    plt.title(f"{param_name} Combined Geometric Distribution")
    plt.legend(fontsize=7.5); plt.grid(alpha=0.2)
    plt.savefig(output_path + f"{param_name} Mean Distributions.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(output_path + f"{param_name}_auto_correlation.png")
    plt.close()
    return acfs_per_map

def plot_iat_per_chain(acfs_per_map, output_path, param_name):
    plt.figure()
    iats_per_map = []
    for acfs in acfs_per_map:
        iats = []
        for acf in acfs:
            iat = integrated_autocorrelation_time(acf)
            iats.append(iat)
        iats_per_map.append(iats)
    plt.barh(range(len(np.array(iats_per_map).flatten())), np.array(iats_per_map).flatten())
    plt.ylabel("Chain"); plt.xlabel("Integrated Auto-Correlation Time")
    plt.axvline(np.mean(np.array(np.array(iats_per_map).flatten())), label = "Mean IAT", color = "black")
    plt.legend()
    plt.title(f"{param_name} Integrated Auto Correlation Time per Chain")
    plt.savefig(output_path + f"{param_name}_integrated_autocorrelation_times.png")
    plt.close()
    return iats_per_map

def scalar_series_map_series_correlation(scalars, arrays):
    #Compute the pixelwise Pearson correlation across the time axis. If you have N scalars
    #and N arrays of shape (H, W). This gives you a 2D map where each pixel's value 
    #is its correlation with the scalar time series. You can then plot it as a heatmap to see
    #which spatial regions track the scalar.

    scalars = np.array(scalars)  # (N,)
    arrays = np.stack(arrays)     # (N, H, W)

    s_mean = scalars.mean()
    a_mean = arrays.mean(axis=0)

    num = ((scalars[:, None, None] - s_mean) * (arrays - a_mean)).mean(axis=0)
    denom = scalars.std() * arrays.std(axis=0)

    correlation_map = num / denom  # (H, W)
    return correlation_map

def map_cross_map_series_correlation(maps_1, maps_2):

    maps_1 = np.array(maps_1)  # (N, H, W)
    maps_2 = np.stack(maps_2)     # (N, H, W)

    map_1_mean = maps_1.mean(axis=0)
    map_2_mean = maps_2.mean(axis=0)

    num = ((maps_1 - map_1_mean) * (maps_2 - map_2_mean)).mean(axis=0)
    denom = maps_1.std(axis=0) * maps_2.std(axis=0)

    correlation_map = num / denom  # (H, W)
    return correlation_map

def get_avg_map_scalar_corr(key, chains, max_iter, max_chains = float("inf")):
    avg_corr = None
    folder_path = os.getcwd() + "/performance_testing/chain_maps/"
    for chain_idx in range(min(len(chains), max_chains)):
        file_path = folder_path + f"chain_{chain_idx}_maps/"
        maps = np.array([jfft.irfft2(np.load(file_path 
                       + f"{key}_iteration_{iter}.npz")["arr_0"]) 
                       for iter in range(len(chains[chain_idx][:max_iter]))])
        a_phi_values = chains[chain_idx][:max_iter]
        corr_map = scalar_series_map_series_correlation(a_phi_values, maps)
        if avg_corr is None:
            avg_corr = corr_map
        else:
            avg_corr += corr_map
    #average map, a_phi correlation across chains
    return avg_corr

def get_avg_map_x_map_corr(key_1, key_2, max_iter, max_chains):
    avg_corr = None
    folder_path = os.getcwd() + "/performance_testing/chain_maps/"
    for chain_idx in range(max_chains):
        file_path = folder_path + f"chain_{chain_idx}_maps/"
        maps_1 = np.array([jfft.irfft2(np.load(file_path 
                       + f"{key_1}_iteration_{iter}.npz")["arr_0"]) 
                       for iter in range(max_iter)])
        maps_2 = np.array([jfft.irfft2(np.load(file_path 
                       + f"{key_2}_iteration_{iter}.npz")["arr_0"]) 
                       for iter in range(max_iter)])
        corr_map = map_cross_map_series_correlation(maps_1, maps_2)
        if avg_corr is None:
            avg_corr = corr_map
        else:
            avg_corr += corr_map
    #average map x map correlation across chains
    return avg_corr

def plot_avg_map_cross_map_corr(max_iter, max_chains, output_path):

    phi_x_phi_corr = get_avg_map_x_map_corr("phi", "phi", max_iter, max_chains)
    phi_x_temp_corr = get_avg_map_x_map_corr("phi", "temperature", max_iter, max_chains)
    temp_x_temp_corr = get_avg_map_x_map_corr("temperature", "temperature", max_iter, max_chains)
    
    plt.figure()
    plt.imshow(phi_x_phi_corr, cmap = "coolwarm")
    plt.colorbar()
    plt.title("Phi x Phi Correlation")
    plt.savefig(output_path + f"Phi x Phi Correlation.png")
    plt.close()

    plt.figure()
    plt.imshow(phi_x_temp_corr, cmap = "coolwarm")
    plt.colorbar()
    plt.title("Phi x Temp Correlation")
    plt.savefig(output_path + f"Phi x Temp Correlation.png")
    plt.close()

    plt.figure()
    plt.imshow(temp_x_temp_corr, cmap = "coolwarm")
    plt.colorbar()
    plt.title("Temp x Temp Correlation")
    plt.savefig(output_path + f"Temp x Temp Correlation.png")
    plt.close()

    return

def plot_avg_map_a_phi_corr(chains, max_iter, output_path, max_chains = float("inf")):

    phi_a_phi_corr = get_avg_map_scalar_corr("phi", chains, max_iter, max_chains)
    plt.figure()
    plt.imshow(phi_a_phi_corr, cmap = "coolwarm")
    plt.colorbar()
    plt.title("Phi & A_phi Correlation")
    plt.savefig(output_path + f"Phi & A_phi Correlation.png")
    plt.close()

    temp_a_phi_corr = get_avg_map_scalar_corr("temperature", chains, max_iter, max_chains)
    plt.figure()
    plt.imshow(temp_a_phi_corr, cmap = "coolwarm")
    plt.colorbar()
    plt.title("Temp & A_phi Correlation")
    plt.savefig(output_path + f"Temp & A_phi Correlation.png")
    plt.close()
    return

def plot_modes_over_time(chains, ground_truth, output_path, param_name):
    final_means = []
    fidelity = 10

    plt.figure()
    for chain in chains:
        means = []
        stds = []
        for iteration in range(100, len(chain), fidelity):
            mean = get_mean(chain[:iteration])
            std = np.std(chain[:iteration])
            means.append(mean)
            stds.append(std)
        means = np.array(means)
        stds = np.array(stds)
        iterations = fidelity * np.arange(len(means))
        line = plt.plot(iterations, means, marker = "o")[0]
        plt.fill_between(iterations, means - stds, means + stds, color = line.get_color(), alpha = 0.2)
        final_means.append(mean)
    mean_final_mean = np.mean(np.array(final_means))
    plt.axhline(mean_final_mean, label = f"Mean = {mean_final_mean}", color = "black")
    plt.axhline(ground_truth, label = f"Ground Truth = {ground_truth}", color = "grey")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Mean")
    plt.title(f"{param_name} Mean over Time")
    plt.savefig(output_path + f"{param_name}_means_over_time.png")
    plt.close()
    return

def get_mode(chain):
    grid = np.linspace(chain.min() - BUFFER, chain.max() + BUFFER, GRID_SIZE)
    density = gaussian_kde(chain)(grid)
    mode = grid[np.argmax(density)]
    return mode

def get_mean(chain):
    grid = np.linspace(chain.min() - BUFFER, chain.max() + BUFFER, GRID_SIZE)
    density = gaussian_kde(chain)(grid)
    dx = np.diff(grid)
    mean = np.dot(density, grid * dx[0])
    return mean

def plot_cov_matrix_data(chains, data_path, output_path, delta_l = 50, theta_pix = 2.5):
    #create a covariance_matrix for each chain and an average as well
    n = len(chains[0])
    avg_cov = None
    num_chains_processed = 0
    #loop through each chain
    for chain_idx, chain in enumerate(chains):

        #initialize the covariance matrix per chain
        cov_mat = []

        #for each chain, loop through the iterations
        for iter in range(n):

            #for each iteration, concatenate the current a_phi, 
            #the Cl^TT from the f map, and the Cl^PP from the phi map. 
            cov_mat_row = []
            cov_mat_row.append(chain[iter])

            temp_map = np.load(data_path + 
                       f"chain_{chain_idx}_maps/temperature_iteration_{iter}.npz")["arr_0"]
            cl_tt = primal_power_spectra(temp_map, theta_pix, delta_l = delta_l)[1]
            cl_tt = np.log(cl_tt)
            for cl_tt_val in cl_tt:
                cov_mat_row.append(cl_tt_val)

            phi_map = np.load(data_path + 
                       f"chain_{chain_idx}_maps/phi_iteration_{iter}.npz")["arr_0"]
            cl_pp = primal_power_spectra(phi_map, theta_pix, delta_l = delta_l)[1]
            cl_pp = np.log(cl_pp)
            for cl_pp_val in cl_pp:
                cov_mat_row.append(cl_pp_val)
            cov_mat.append(cov_mat_row)

        #Then create a covariance matrix with this vector by using np.cov(array_chain_idx)
        #add this to a running average of covariance matrices...
        cov_mat = np.array(cov_mat).T
        cov_plot = np.nan_to_num(np.cov(cov_mat), nan = 0)
        num_chains_processed += 1
        if avg_cov is None:
            avg_cov = cov_plot
        else:
            avg_cov += cov_plot

        plt.figure()
        plt.imshow(cov_plot, cmap = "coolwarm")
        plt.colorbar()
        plt.title(f"Covariance of [A_phi, Cl^TT, Cl^PP] vector: Chain {chain_idx}")
        plt.savefig(output_path + f"Covariance Matrix Chain {chain_idx}.png")
        plt.close()

    #Finally plot the average covariance:
    avg_cov /= num_chains_processed
    plt.figure()
    plt.imshow(avg_cov, cmap = "coolwarm")
    plt.colorbar()
    plt.title("Average Covariance of [A_phi, Cl^TT, Cl^PP] vector")
    plt.savefig(output_path + f"Average Covariance Matrix.png")
    plt.close()
    return

def plot_pooled_distribution(chains, output_path, ground_truth, param_name):

    #loop through the chains and concatenate them into a super chain
    super_chain = np.array([])
    for chain in chains:
        super_chain = np.concatenate((super_chain, chain))

    #perform a single Gaussian KDE on the pooled super chain
    grid = np.linspace(super_chain.min() - BUFFER, super_chain.max() + BUFFER, GRID_SIZE)
    pdf = gaussian_kde(super_chain)(grid)
    dx = np.diff(grid)
    norm = np.sum(0.5*(pdf[1:]+pdf[:-1])*dx)
    pdf /= norm

    #compute mean, mode, and +/- 1 std
    #TODO make this a reusable method
    mode = round(grid[np.argmax(pdf)], 7)
    mean = round(np.dot(pdf, grid * dx[0]), 7)
    std = round(np.sqrt(np.sum(dx[0] * pdf * (grid - mean)**2)), 7)

    plt.figure()
    plt.plot(grid, pdf, label = "Pooled Gaussian KDE", color = "steelblue")
    plt.axvline(mode, label = f"Mode = {mode}", color = "indianred")
    plt.axvline(mean, label = f"Mean = {mean}", color = "purple")
    plt.axvline(mean + std, color = "black", label = f"+/- 1 std = {std}")
    plt.axvline(mean - std, color = "black")
    plt.axvline(ground_truth, color = "green", label = f"Ground Truth = {ground_truth}")
    plt.xlabel("A_phi Value")
    plt.ylabel("PDF")
    plt.legend()
    plt.title(f"{param_name} Pooled PDF via Gaussian KDE")
    plt.savefig(output_path + f"{param_name} Pooled Distribution.png")
    plt.close()

    return

def plot_mode_histogram(chains, output_path, ground_truth, param_name):

    #mode of each per-map marginal via KDE argmax
    modes = np.array([get_mode(chain) for chain in chains])

    #frequentist scatter of the mode estimator across realizations
    mode_mean = np.mean(modes)
    mode_std = np.std(modes, ddof = 1)

    #average posterior width of a single marginal, computed directly from the samples
    avg_sigma_single = np.mean([np.std(chain, ddof = 1) for chain in chains])

    plt.figure()
    plt.hist(modes, bins = 20, alpha = 0.6, edgecolor = "white")
    plt.axvline(ground_truth, color = "green", label = f"Ground Truth = {ground_truth}")
    plt.axvline(mode_mean, color = "indianred", label = f"Mean of Modes = {round(mode_mean, 7)}")
    plt.axvline(mode_mean + mode_std, color = "black", label = f"+/- 1 Std of Modes = {round(mode_std, 7)}")
    plt.axvline(mode_mean - mode_std, color = "black")
    plt.xlabel("Mode Value"); plt.ylabel("Count")
    #calibration check: std of the modes should roughly match the average single-marginal std
    plt.title(f"{param_name} Histogram of Marginal Modes\n"
              f"std(modes) = {round(mode_std, 7)} vs avg single-marginal std = {round(avg_sigma_single, 7)}")
    plt.legend(); plt.grid(alpha = 0.2)
    plt.savefig(output_path + f"{param_name}_mode_histogram.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    return avg_sigma_single

def plot_mode_histogram_from_pdfs(grid, pdfs, output_path, ground_truth, param_name):
    #mode of each per-map marginal via KDE argmax
    modes = np.array([grid[np.argmax(pdf)] for pdf in pdfs])

    #frequentist scatter of the mode estimator across realizations
    mode_mean = np.mean(modes)
    mode_std = np.std(modes, ddof = 1)

    #average posterior width of a single marginal, computed directly from the samples
    dx = np.diff(grid)
    sigmas = []
    for pdf in pdfs:
        multiplicative_mean = np.dot(pdf, grid * dx[0])
        multiplicative_std = np.sqrt(np.sum(dx[0] * pdf * (grid - multiplicative_mean)**2))
        sigmas.append(multiplicative_std)
    avg_sigma_single = np.mean(sigmas)

    plt.figure()
    plt.hist(modes, bins = 20, alpha = 0.6, edgecolor = "white")
    plt.axvline(ground_truth, color = "green", label = f"Ground Truth = {ground_truth}")
    plt.axvline(mode_mean, color = "indianred", label = f"Mean of Modes = {round(mode_mean, 7)}")
    plt.axvline(mode_mean + mode_std, color = "black", label = f"+/- 1 Std of Modes = {round(mode_std, 7)}")
    plt.axvline(mode_mean - mode_std, color = "black")
    plt.xlabel("Mode Value"); plt.ylabel("Count")
    #calibration check: std of the modes should roughly match the average single-marginal std
    plt.title(f"{param_name} Histogram of Marginal Modes\n"
              f"std(modes) = {round(mode_std, 7)} vs avg single-marginal std = {round(avg_sigma_single, 7)}")
    plt.legend(); plt.grid(alpha = 0.2)
    plt.savefig(output_path + f"{param_name}_mode_histogram.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    return

def prune_chains(chains_per_map, iats_per_map, acfs_per_map = None,
                 use_zero_crossing = USE_ZERO_CROSSING_PRUNE):
    #thin each chain by its own stride so the retained points are not correlated. the
    #stride is either the integrated autocorrelation time (default) or, with
    #use_zero_crossing, the lag just before that chain's acf first crosses zero
    if use_zero_crossing and acfs_per_map is None:
        raise ValueError("use_zero_crossing pruning requires the per-chain acfs")
    pruned_chains = []
    for i, chains in enumerate(chains_per_map):
        concatenated_chain = []
        for j, chain in enumerate(chains):
            if use_zero_crossing:
                stride = first_zero_crossing_lag(acfs_per_map[i][j])
            else:
                #clamp so an IAT below 1 cannot produce a zero slice step
                stride = max(1, int(iats_per_map[i][j]))
            concatenated_chain.append(chain[0:-1:stride])
        pruned_chains.append(np.concatenate([chain for chain in concatenated_chain]))
    return pruned_chains

def average_over_phi(chains, output_path, ground_truth, param_name):
    global_min = float("inf")
    global_max = -1*global_min
    for i in range(len(chains)):
        for j in range(len(chains[i])):
            values = chains[i][j]
            min_value = np.min(values)
            max_value = np.max(values)
            if min_value < global_min:
                global_min = min_value
            if max_value > global_max:
                global_max = max_value
    grid = np.linspace(global_min - BUFFER, global_max + BUFFER, GRID_SIZE)
    dx = np.diff(grid)
    pdfs_per_map = []
    plt.figure()
    #loop over rows / data maps
    for phis_per_map in chains:
        #for each phi realization per map, compute the normalized PDF
        pdf_per_map = None
        for phi_chain in phis_per_map:
            pdf = np.array(gaussian_kde(phi_chain)(grid))
            norm = np.sum(((pdf[:-1] + pdf[1:]) * dx)/2)
            pdf /= norm
            if pdf_per_map is None:
                pdf_per_map = pdf
            else:
                pdf_per_map += pdf
        pdf_per_map /= len(phis_per_map)
        pdfs_per_map.append(pdf_per_map)
        plt.plot(grid, pdf_per_map)  
    plt.axvline(ground_truth, color = "black", label = f"Ground Truth = {ground_truth}")
    plt.legend()
    plt.xlabel("Count"); plt.title(f"{param_name} Marginals"); plt.grid(alpha = 0.2)
    plt.savefig(output_path + f"/{param_name}_all_chain_marginals.png", dpi = 150, bbox_inches = "tight")
    plt.close()

    #return the list of the "phi-averaged" per-data-map PDFs
    return grid, pdfs_per_map

def per_param_analysis(file_name, num_maps, num_chains, map_pre_factor, ground_truth, default_burn_in, param_name):

    #load the data
    raw_chains = []
    r_hats = []
    data_path = os.getcwd() + f"/sampling_chains/{file_name}/"
    output_path = os.getcwd() + f"/sampling_chains/lcdm_chain_plots/{param_name}/"
    for map_idx in range(1, num_maps + 1):
        chains_per_map = []
        for chain_idx in range(1, num_chains + 1):
            file_path = data_path + f"{param_name}_map_{map_pre_factor * map_idx}_chain_{chain_idx}_history.txt"
            if os.path.exists(file_path):
                array = np.loadtxt(file_path)
                if len(array) > default_burn_in:
                    chains_per_map.append(np.loadtxt(file_path)[default_burn_in:])
        #gelman-rubin R-statistic
        r_hats.append(gelman_rubin(chains_per_map))
        raw_chains.append(chains_per_map)

    r_hats = np.array(r_hats)
    print(r_hats)
    print(f"{param_name} Average Gelman-Rubin R Statistic = {np.mean(r_hats)}")
    print(f"{param_name} Std in Gelman-Rubin R Statistic = {np.std(r_hats)}")

    #auto-correlation plots
    acfs_per_map = plot_autocorrelation_chains(raw_chains, output_path, param_name)

    #integrated autocorrelation time
    iats_per_map = plot_iat_per_chain(acfs_per_map, output_path, param_name)

    #prune the chains by their individual strides (IAT or first zero crossing, per
    #USE_ZERO_CROSSING_PRUNE):
    chains = prune_chains(raw_chains, iats_per_map, acfs_per_map = acfs_per_map, use_zero_crossing = USE_ZERO_CROSSING_PRUNE)
    
    #make an all chain trace plot
    all_chain_trace(chains, output_path, param_name)

    #plot each single map's distribution
    plot_marginals(chains, output_path, ground_truth, param_name)

    #histogram of per-map marginal modes
    avg_sigma_single = plot_mode_histogram(chains, output_path, ground_truth, param_name)

    #combined geometric mean
    plot_multiplicative_mean(chains, avg_sigma_single, output_path, ground_truth, param_name)

    #concatenate the columns to turn the D x N raw_chains list of list of lists
    #into a D length list of lists
    stacked_chains = stack_chains(raw_chains)

    return stacked_chains

def stack_chains(raw_chains):
    stacked_chains = []
    for chains in raw_chains:
        stacked_chain = np.concatenate([chain for chain in chains])
        stacked_chains.append(stacked_chain)
    return stacked_chains

def get_fisher_matrix(cov_mat, param_names):
    """Gaussian-approximation Fisher matrix F = C^-1 of the sampled parameters, saved to
    fisher_matrix.png. Entries span many orders of magnitude (theta_MC_100 is ~1e3 times
    narrower than logA), so the colors use a symmetric log scale about zero and the boxes are
    annotated in scientific notation."""
    output_path = os.getcwd() + f"/sampling_chains/lcdm_chain_plots/"
    fisher_mat = np.linalg.inv(cov_mat)
    print("Fisher matrix (rows / columns ordered as", param_names, ")")
    print(fisher_mat)
    print("Fisher-implied marginal sigmas (sqrt of diagonal of F^-1):")
    for name, sigma in zip(param_names, np.sqrt(np.diag(cov_mat))):
        print(f"    {name}: {sigma:.4e}")
    limit = np.max(np.abs(fisher_mat))
    #linthresh sets where the log scaling gives way to linear about zero; the smallest
    #nonzero magnitude keeps every entry inside the log regime
    linthresh = np.min(np.abs(fisher_mat[fisher_mat != 0])) if np.any(fisher_mat != 0) else 1.0
    norm = matplotlib.colors.SymLogNorm(linthresh = linthresh, vmin = -limit, vmax = limit)
    plot_annotated_matrix(fisher_mat, param_names, "Fisher Matrix (inverse posterior covariance)",
                          output_path, "fisher_matrix.png", norm = norm, fmt = "{:+.3e}")
    return fisher_mat

def get_correlation_matrix(raw_chains):
    """Pooled posterior covariance and correlation matrices of the sampled parameters.

    raw_chains maps each parameter name to its list of per-map concatenated chains (the output
    of per_param_analysis). Every sampled parameter is written to its history file once per
    iteration, and per_param_analysis walks maps and chains in the same order for each
    parameter, so concatenating over maps gives columns that line up row-by-row. Saves
    correlation_matrix.png with every entry printed in its box.
    """
    param_names = list(raw_chains.keys())
    columns = []
    for name in param_names:
        columns.append(np.concatenate(raw_chains[name]))
    lengths = [len(column) for column in columns]
    #CLIP to min length in case we have SCP race conditions... This should
    #not matter once the chains are fully finished running
    min_length = min(lengths)
    columns = [column[:min_length] for column in columns]
    samples = np.column_stack(columns)
    cov_mat = np.cov(samples, rowvar = False)
    corr_mat = cov_mat / np.sqrt(np.outer(np.diag(cov_mat), np.diag(cov_mat)))
    print("Correlation matrix (rows / columns ordered as", param_names, ")")
    print(corr_mat)
    output_path = os.getcwd() + f"/sampling_chains/lcdm_chain_plots/"
    os.makedirs(output_path, exist_ok = True)
    norm = matplotlib.colors.Normalize(vmin = -1, vmax = 1)
    plot_annotated_matrix(corr_mat, param_names, "Posterior Correlation Matrix", output_path,
                          "correlation_matrix.png", norm = norm, fmt = "{:+.3f}")
    plot_covariance_matrix(cov_mat, param_names, output_path)
    return cov_mat, corr_mat, param_names

def plot_covariance_matrix(cov_mat, names, path):
    figure, axis = plt.subplots(figsize = (7.5, 6.5))
    annotated_heatmap(axis, cov_mat, names,
                      "Covariance Matrix", "Value")
    figure.tight_layout()
    figure.savefig(path + "covariance_matrix.png", dpi = 150)
    plt.close(figure)

#Contour styling for the triangle plot. The measured posterior is the shaded one; every
#forecast is an outline in its own colour so several can share the axes. "marginal" is
#listed ready for merge_marginal_fisher.py's output - nothing is drawn for a label that is
#not actually passed to triangle_plot
MEASURED_COLOUR = "tab:red"
FORECAST_STYLES = {
    "ceiling": {"colour": "tab:blue", "linestyle": "--", "short": "ceil",
                "label": "ceiling (complete-data bound)"},
    "lensed": {"colour": "tab:purple", "linestyle": ":", "short": "lens",
               "label": "lensed (two-point bound)"},
    "marginal": {"colour": "tab:green", "linestyle": "-.", "short": "marg",
                 "label": "marginal (Louis)"},
}
DEFAULT_FORECAST_STYLE = {"colour": "tab:gray", "linestyle": "--", "short": "fisher",
                          "label": "Fisher forecast"}


def triangle_plot(measured_cov_mat, forecasts, param_names, means, sigmas = (1,)):
    """Triangle plot comparing the sampled posterior against one or more Fisher forecasts.

    `forecasts` maps a label from FORECAST_STYLES to a covariance matrix, e.g.
    {"ceiling": ..., "lensed": ...}. The MEASURED posterior (pooled from the Gibbs chains)
    is drawn shaded in red; each forecast is left as a coloured outline so they stay
    legible where they overlap - blue for ceiling, purple for lensed. The diagonal carries
    each parameter's marginal Gaussian, the lower triangle each pair's joint confidence
    ellipse. sigmas lists which contours to draw (1-sigma only by default); each level k
    gives semi-axes k * sqrt(eigenvalue) of the pair's 2x2 covariance block.

    Drawing BOTH bounds at once is the point. Neither is the marginal Fisher of
    p(d | theta) that the sampler targets - "lensed" is a two-point analysis of a surrogate
    Gaussian model, "ceiling" is the complete-data information given (f, phi) - and the
    measured posterior sits BETWEEN them, in the widths and in the degeneracy directions
    alike. A single forecast contour hides that: with only "ceiling" drawn, the
    omch2 - theta_MC_100 ellipse tilts the wrong way, because that pair's bracket straddles
    zero (ceiling -0.13, chains +0.38, lensed +0.55). Seeing the measured ellipse nested
    between the two is the correct reading, not a discrepancy to chase.

    Everything is centred on the MEASURED means. A forecast is a curvature, not a location
    - it has a shape but no centre - so a common centre makes the plot a direct read of
    width and orientation, which is the only thing these matrices can be compared on. Each
    diagonal panel is annotated with sigma_measured / sigma_forecast per forecast: above 1
    means the sampler is doing worse than that forecast says the data allows, near 1 means
    it is saturating it, below 1 means it is extracting information that forecast does not
    model (or the chain has not converged).

    Every covariance in `forecasts` MUST already be permuted into param_names order - see
    get_forecasted_covariance, which does it.
    """
    output_path = os.getcwd() + f"/sampling_chains/lcdm_chain_plots/"
    measured = np.asarray(measured_cov_mat)
    means = np.asarray(means)
    n = len(param_names)

    forecasts = {label: np.asarray(cov_mat) for label, cov_mat in forecasts.items()}
    for label, cov_mat in forecasts.items():
        if cov_mat.shape != measured.shape:
            raise ValueError(f"measured covariance is {measured.shape} but the "
                             f"'{label}' forecast is {cov_mat.shape}; they must cover the "
                             f"same {n} parameters in the same order ({param_names})")

    #(covariance, colour, filled, linestyle) - measured shaded, forecasts as outlines
    layers = [(measured, MEASURED_COLOUR, True, "-")]
    for label, cov_mat in forecasts.items():
        style = FORECAST_STYLES.get(label, DEFAULT_FORECAST_STYLE)
        layers.append((cov_mat, style["colour"], False, style["linestyle"]))

    #pad the axes out past the widest requested contour of the WIDEST matrix, so whichever
    #one is broader still fits
    pad = max(sigmas) + 1
    widest = np.sqrt(np.diag(measured))
    for cov_mat in forecasts.values():
        widest = np.maximum(widest, np.sqrt(np.diag(cov_mat)))
    limits = [(means[i] - pad * widest[i], means[i] + pad * widest[i]) for i in range(n)]

    fig, axes = plt.subplots(n, n, figsize = (2.8 * n, 2.8 * n), squeeze = False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            #only the lower triangle carries a panel
            if j > i:
                ax.axis("off")
                continue
            for cov_mat, colour, filled, linestyle in layers:
                if i == j:
                    sigma = np.sqrt(cov_mat[i, i])
                    grid = np.linspace(*limits[i], 500)
                    ax.plot(grid, np.exp(-0.5 * ((grid - means[i]) / sigma) ** 2),
                            color = colour, ls = linestyle, lw = 1.6)
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
                        ax.add_patch(Ellipse(
                            (means[j], means[i]), width, height, angle = angle,
                            facecolor = colour if filled else "none",
                            edgecolor = "black" if filled else colour,
                            ls = linestyle,
                            lw = 1.0 if filled else 1.6,
                            alpha = 0.55 / k if filled else 1.0))
            if i == j:
                #how far the sampler is from each forecast, in this parameter's own units
                ratios = []
                for label, cov_mat in forecasts.items():
                    short = FORECAST_STYLES.get(label, DEFAULT_FORECAST_STYLE)["short"]
                    ratio = np.sqrt(measured[i, i] / cov_mat[i, i])
                    ratios.append(f"$\\sigma_{{meas}}/\\sigma_{{{short}}}$ = {ratio:.2f}")
                ax.text(0.04, 0.93, "\n".join(ratios), transform = ax.transAxes,
                        fontsize = 8, va = "top")
                ax.set_ylim(0, 1.25)
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
    handles = [Line2D([0], [0], color = MEASURED_COLOUR, lw = 2,
                      label = "measured (chains)")]
    for label in forecasts:
        style = FORECAST_STYLES.get(label, DEFAULT_FORECAST_STYLE)
        handles.append(Line2D([0], [0], color = style["colour"], lw = 2,
                              ls = style["linestyle"], label = style["label"]))
    #the vacant upper triangle is the natural home for the legend; keep it clear of the
    #suptitle, which bbox_inches = "tight" would otherwise let it overlap
    fig.legend(handles = handles, loc = "upper right", fontsize = 11,
               bbox_to_anchor = (0.98, 0.94))
    fig.suptitle(f"Posterior vs Fisher Forecasts (Gaussian approximation, {contours})",
                 y = 0.995)
    plt.savefig(output_path + "triangle_plot.png", dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    return

def report_forecast_comparison(measured_cov_mat, forecasts, param_names):
    """Print each parameter's measured sigma beside every forecast, and flag the bracket.

    "between" is the expected outcome: the sampler works with the full hierarchical
    likelihood, so it should beat the two-point "lensed" bound while staying inside the
    complete-data "ceiling". Anything outside that bracket is worth investigating - either
    the chain has not converged, or the forecast is not describing this data set.
    """
    labels = list(forecasts)
    print("Measured vs forecast marginal sigmas (order:", param_names, ")")
    for i, name in enumerate(param_names):
        measured = np.sqrt(measured_cov_mat[i, i])
        parts = []
        for label in labels:
            forecasted = np.sqrt(forecasts[label][i, i])
            parts.append(f"{label} {forecasted:.4e} (ratio {measured / forecasted:.3f})")
        print(f"    {name}: measured {measured:.4e}  " + "  ".join(parts))

    if "ceiling" in forecasts and "lensed" in forecasts:
        for i, name in enumerate(param_names):
            measured = np.sqrt(measured_cov_mat[i, i])
            low = np.sqrt(forecasts["ceiling"][i, i])
            high = np.sqrt(forecasts["lensed"][i, i])
            inside = min(low, high) <= measured <= max(low, high)
            print(f"    {name}: sigma {'inside' if inside else 'OUTSIDE'} the "
                  f"ceiling-lensed bracket [{min(low, high):.4e}, {max(low, high):.4e}]")
        #the correlations bracket too, and that is where the bounds can disagree on SIGN
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                def correlation(matrix):
                    return matrix[i, j] / np.sqrt(matrix[i, i] * matrix[j, j])
                measured = correlation(measured_cov_mat)
                low = correlation(forecasts["ceiling"])
                high = correlation(forecasts["lensed"])
                inside = min(low, high) <= measured <= max(low, high)
                print(f"    r({param_names[i]}, {param_names[j]}): measured "
                      f"{measured:+.3f}  ceiling {low:+.3f}  lensed {high:+.3f}  "
                      f"-> {'inside' if inside else 'OUTSIDE'} the bracket")
    return


def joint_param_analysis(all_chains, nside, theta_pix, noise, is_sampled):
    measured_cov_mat, _ , param_names= get_correlation_matrix(all_chains)
    get_fisher_matrix(measured_cov_mat, list(all_chains.keys()))
    means = [np.mean(np.concatenate(all_chains[name])) for name in param_names]
    #both bounds: the measured posterior should sit BETWEEN them, in the widths and in the
    #degeneracy directions alike, so plotting only one hides where the sampler really lands
    forecasts = {label: get_forecasted_covariance(nside, theta_pix, noise, is_sampled,
                                                  param_names, spectra = label)
                 for label in ("ceiling", "lensed")}
    report_forecast_comparison(measured_cov_mat, forecasts, param_names)
    triangle_plot(measured_cov_mat, forecasts, param_names, means)
    return

def get_forecasted_covariance(nside, theta_pix, noise, is_sampled, param_names,
                             spectra = "ceiling"):
    """Fisher-forecast covariance for the sampled parameters, reordered to match param_names.

    `spectra` selects which bound: "ceiling" (complete-data information given f and phi)
    or "lensed" (a two-point analysis of the observed map). Call it once per bound and hand
    both to triangle_plot - CAMB results are memoized in fisher_forecast._CAMB_CACHE and
    the two bounds share the same stencil POINTS, so the second call re-runs no CAMB, only
    the covariance-block builds.

    forecast() returns its parameters in fisher_forecast's OUTPUT_PARAM_ORDER (omch2, ombh2,
    ns, theta_MC_100, logA), which is the order ground_truth_values is written in here, so
    the permutation below is normally the identity. It is kept because nothing enforces that
    agreement: reorder ground_truth_values and, without it, the triangle plot would silently
    pair each measured parameter with a different forecasted one.

    nside / theta_pix / noise must match the run that produced the chains - see the values
    at the top of sample_lcdm.sh.
    """
    fisher, names = forecast(nside, theta_pix, noise, is_sampled,
                             GROUND_TRUTH, spectra = spectra,
                             l_knee = 0, beam_fwhm = 0)
    covariance = covariance_from_fisher(fisher, names)
    missing = [name for name in param_names if name not in names]
    if missing:
        raise ValueError(f"the forecast did not cover {missing}; was_sampled and the "
                         f"chains that were actually found disagree on which parameters "
                         f"were sampled")
    order = [names.index(name) for name in param_names]
    return covariance[np.ix_(order, order)]

def main(file_name, num_maps, num_chains, map_pre_factor, ground_truth_values, 
         was_sampled, default_burn_in, nside, theta_pix, noise):

    #single parameter analyses
    all_chains = {}
    for param_name, ground_truth in ground_truth_values.items():
        if was_sampled[param_name]:
            chains_per_param = per_param_analysis(file_name, num_maps, num_chains, map_pre_factor, 
                                                     ground_truth, default_burn_in, param_name)
            all_chains[param_name] = chains_per_param

    #fisher information or other multi-param joint statistics
    if get_num_sampled(was_sampled) > 1:
        joint_param_analysis(all_chains, nside, theta_pix, noise, was_sampled)
    return

def get_num_sampled(was_sampled):
    num_sampled = 0
    for _, sampled in was_sampled.items():
        if sampled:
            num_sampled += 1
    return num_sampled

if __name__ == "__main__":

    num_maps = 50
    num_chains = 5
    default_burn_in = 600
    map_pre_factor = 234567
    file_name = "omch2_logA_theta_MC_100_joint_inference_08_27_26"

    #the Fisher forecast the triangle plot compares against is only meaningful if it is
    #built on the same box and noise level the chains ran on - MATCH sample_lcdm.sh
    nside = 128
    theta_pix = 2.5
    noise = 5

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

    main(file_name, num_maps, num_chains, map_pre_factor,
         ground_truth_values, was_sampled, default_burn_in, nside, theta_pix, noise)
