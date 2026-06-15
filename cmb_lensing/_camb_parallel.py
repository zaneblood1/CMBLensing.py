"""Lightweight CAMB worker module with no JAX dependency.
Spawned worker processes only import this module (not simulate.py),
avoiding the slow JAX initialization overhead."""
import numpy as np

def camb_worker(params):
    """Run a single CAMB evaluation. Returns numpy arrays.
    OMP_NUM_THREADS should be set via the environment before spawning
    (parallel_camb_batch handles this automatically)."""
    import camb
    lmax_prime = int(params["lmax_prime"])
    try:
        pars = camb.set_params(
            H0 = params["H0"], ombh2 = float(params["ombh2"]),
            omch2 = float(params["omch2"]), cosmomc_theta = float(params["cosmomc_theta"]),
            r = float(params["r"]), mnu = float(params["mnu"]),
            As = float(params["As"]), nt = float(params["nt"]),
            ns = float(params["ns"]), lmax = lmax_prime,
            tau = float(params["tau"]), pivot_scalar = float(params["k_pivot"]),
            pivot_tensor = float(params["k_pivot"]), Alens = float(params["Alens"]))
        pars.max_l_tensor = 2 * lmax_prime
        pars.max_eta_k_tensor = 4 * lmax_prime
        pars.WantScalars = True
        pars.WantTensors = True
        pars.DoLensing = True
        pars.set_nonlinear_lensing(True)
        results = camb.get_results(pars)
        power_spectra = results.get_cmb_power_spectra(pars, lmax = lmax_prime - 1,
                                                       CMB_unit = "muK")
        lens_potential = results.get_lens_potential_cls(lmax = lmax_prime - 1)[:, 0]
        unlensed = np.asarray(power_spectra["unlensed_scalar"], dtype = np.float64)
        tensor = np.asarray(power_spectra["tensor"], dtype = np.float64)
        total = np.asarray(power_spectra["total"], dtype = np.float64)
        lens = np.asarray(lens_potential, dtype = np.float64)
    except Exception:
        unlensed = np.full((lmax_prime, 4), np.nan)
        tensor = np.full((lmax_prime, 4), np.nan)
        total = np.full((lmax_prime, 4), np.nan)
        lens = np.full((lmax_prime,), np.nan)
    return unlensed, tensor, total, lens
