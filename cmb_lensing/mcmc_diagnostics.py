"""MCMC convergence and mixing diagnostics shared by the chain analyses.

gelman_rubin / gelman_rubin_from_moments: the Gelman-Rubin R-hat across several chains (no
splitting), for scalar chains or elementwise for array-valued samples.
autocorrelation / integrated_autocorrelation_time: one scalar chain's normalized
autocorrelation and Sokal's windowed integrated autocorrelation time.
"""

import numpy as np

#Sokal's window constant: truncate the IAT sum at the first lag M with M >= c * tau(M)
SOKAL_WINDOW = 5.0

def gelman_rubin_from_moments(chain_means, chain_variances, n_samples):
    """R-hat from each chain's sample mean and (ddof = 1) variance, stacked along axis 0
    (shape (n_chains, ...)), every chain `n_samples` long. Elementwise over the trailing axes;
    NaN where the within-chain variance is 0."""
    chain_means = np.asarray(chain_means)
    n_chains = len(chain_means)
    between = n_samples / (n_chains - 1) * np.sum(
        (chain_means - np.mean(chain_means, axis = 0))**2, axis = 0)
    within = np.mean(chain_variances, axis = 0)
    pooled_variance = (n_samples - 1) / n_samples * within + between / n_samples
    with np.errstate(divide = "ignore", invalid = "ignore"):
        rhat = np.where(within > 0, np.sqrt(pooled_variance / within), np.nan)
    #[()] turns a 0-d result (scalar chains) into a numpy scalar
    return rhat[()]

def gelman_rubin(chains):
    """Gelman-Rubin R-hat across chains (each (n_samples, ...)), trimmed to the shortest."""
    n_samples = min(len(chain) for chain in chains)
    chains = [np.asarray(chain[:n_samples]) for chain in chains]
    return gelman_rubin_from_moments([np.mean(chain, axis = 0) for chain in chains],
                                     [np.var(chain, axis = 0, ddof = 1) for chain in chains],
                                     n_samples)

def integrated_autocorrelation_time(acf, c = SOKAL_WINDOW):
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
