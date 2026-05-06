import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from cmb_lensing.statistics import primal_cross_correlation

POLARITIES = ["I", "P", "IP"]
MAP_SIZES = [128, 256, 512, 1024]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TRIALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
THETA_PIX = 2.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "performance_results")
JULIA_DIR = os.path.join(RESULTS_DIR, "julia_results")
PYTHON_DIR = os.path.join(RESULTS_DIR, "python_results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "performance_analysis")

NUM_SEEDS = len(SEEDS)
NUM_TRIALS = len(TRIALS)

def folder_name(map_size, polarity, seed):
    return f"map_size_{map_size}_polarity_{polarity}_seed_{seed}"


def folder_exists(map_size, polarity, seed):
    julia_path = os.path.join(JULIA_DIR, folder_name(map_size, polarity, seed))
    python_path = os.path.join(PYTHON_DIR, folder_name(map_size, polarity, seed))
    return os.path.isdir(julia_path) and os.path.isdir(python_path)


def get_available_polarities():
    available = []
    for polarity in POLARITIES:
        if any(folder_exists(ms, polarity, s) for ms in MAP_SIZES for s in SEEDS):
            available.append(polarity)
    return available


def get_available_map_sizes(polarity):
    available = []
    for ms in MAP_SIZES:
        if any(folder_exists(ms, polarity, s) for s in SEEDS):
            available.append(ms)
    return available


def get_available_seeds(polarity, map_size):
    return [s for s in SEEDS if folder_exists(map_size, polarity, s)]


def get_available_trials(language_dir, map_size, polarity, seed, cached=True):
    time_type = "cached_times" if cached else "uncached_times"
    time_file = "cached_time" if cached else "uncached_time"
    folder = os.path.join(language_dir, folder_name(map_size, polarity, seed), time_type)
    available = []
    for trial in TRIALS:
        if os.path.isfile(os.path.join(folder, f"{time_file}_{trial}.txt")):
            available.append(trial)
    return available

def load_time(language_dir, map_size, polarity, seed, trial, cached=True):
    time_type = "cached_times" if cached else "uncached_times"
    time_file = "cached_time" if cached else "uncached_time"
    path = os.path.join(
        language_dir,
        folder_name(map_size, polarity, seed),
        time_type,
        f"{time_file}_{trial}.txt"
    )
    return np.loadtxt(path)


JULIA_FIELD_MAP = {"t_field": "fJ_t", "e_field": "fJ_e", "b_field": "fJ_b"}


def load_array(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        return jnp.array(data)
    return jnp.array(data[data.files[0]])


def load_field(language_dir, map_size, polarity, seed, trial, field_name):
    if language_dir == JULIA_DIR:
        filename = f"{JULIA_FIELD_MAP[field_name]}_{trial}.npz"
    else:
        filename = f"{field_name}_{trial}.npz"
    path = os.path.join(
        language_dir,
        folder_name(map_size, polarity, seed),
        "learned_fields", "cmb",
        filename
    )
    return load_array(path)


def load_phi(language_dir, map_size, polarity, seed, trial, phi_name):
    if language_dir == JULIA_DIR:
        filename = f"phiJ_{trial}.npz"
    else:
        filename = f"{phi_name}_{trial}.npz"
    path = os.path.join(
        language_dir,
        folder_name(map_size, polarity, seed),
        "learned_fields", "lensing_potential",
        filename
    )
    return load_array(path)


def get_field_names_for_polarity(polarity):
    if polarity == "I":
        return ["t_field"]
    elif polarity == "P":
        return ["e_field", "b_field"]
    else:
        return ["t_field", "e_field", "b_field"]


def average_times(language_dir, polarity, available_map_sizes, cached=True):
    means = []
    stds = []
    for map_size in available_map_sizes:
        samples = []
        for seed in get_available_seeds(polarity, map_size):
            for trial in get_available_trials(language_dir, map_size, polarity, seed, cached):
                samples.append(load_time(language_dir, map_size, polarity, seed, trial, cached))
        samples = np.array(samples)
        means.append(np.mean(samples) if len(samples) > 0 else 0.0)
        stds.append(np.std(samples) if len(samples) > 0 else 0.0)
    return means, stds


def average_fractional_difference(polarity, available_map_sizes):
    field_names = get_field_names_for_polarity(polarity)
    components = field_names + ["phi"]
    julia_means = {c: [] for c in components}
    julia_stds = {c: [] for c in components}
    python_means = {c: [] for c in components}
    python_stds = {c: [] for c in components}
    for map_size in available_map_sizes:
        julia_samples = {c: [] for c in components}
        python_samples = {c: [] for c in components}
        for seed in get_available_seeds(polarity, map_size):
            for trial in get_available_trials(JULIA_DIR, map_size, polarity, seed, cached=True):
                for fn in field_names:
                    julia_sim = load_field(PYTHON_DIR, map_size, polarity, seed, trial, f"{fn}_julia_simulation")
                    julia_pred = load_field(JULIA_DIR, map_size, polarity, seed, trial, fn)
                    python_pred = load_field(PYTHON_DIR, map_size, polarity, seed, trial, f"{fn}_python_predict")
                    norm_sim = jnp.linalg.norm(julia_sim)
                    julia_samples[fn].append(float(jnp.linalg.norm(julia_sim - julia_pred) / norm_sim))
                    python_samples[fn].append(float(jnp.linalg.norm(julia_sim - python_pred) / norm_sim))
                phi_sim = load_phi(PYTHON_DIR, map_size, polarity, seed, trial, "phi_julia_simulation")
                phi_julia_pred = load_phi(JULIA_DIR, map_size, polarity, seed, trial, "phiJ")
                phi_python_pred = load_phi(PYTHON_DIR, map_size, polarity, seed, trial, "phi_python_predict")
                norm_phi = jnp.linalg.norm(phi_sim)
                julia_samples["phi"].append(float(jnp.linalg.norm(phi_sim - phi_julia_pred) / norm_phi))
                python_samples["phi"].append(float(jnp.linalg.norm(phi_sim - phi_python_pred) / norm_phi))
        for c in components:
            arr_j = np.array(julia_samples[c])
            arr_p = np.array(python_samples[c])
            julia_means[c].append(np.mean(arr_j) if len(arr_j) > 0 else 0.0)
            julia_stds[c].append(np.std(arr_j) if len(arr_j) > 0 else 0.0)
            python_means[c].append(np.mean(arr_p) if len(arr_p) > 0 else 0.0)
            python_stds[c].append(np.std(arr_p) if len(arr_p) > 0 else 0.0)
    return components, julia_means, julia_stds, python_means, python_stds


def average_julia_vs_python_diff(polarity, available_map_sizes):
    field_names = get_field_names_for_polarity(polarity)
    components = field_names + ["phi"]
    means = {c: [] for c in components}
    stds = {c: [] for c in components}
    for map_size in available_map_sizes:
        samples = {c: [] for c in components}
        for seed in get_available_seeds(polarity, map_size):
            for trial in get_available_trials(JULIA_DIR, map_size, polarity, seed, cached=True):
                for fn in field_names:
                    julia_pred = load_field(JULIA_DIR, map_size, polarity, seed, trial, fn)
                    python_pred = load_field(PYTHON_DIR, map_size, polarity, seed, trial, f"{fn}_python_predict")
                    norm_julia = jnp.linalg.norm(julia_pred)
                    samples[fn].append(float(jnp.linalg.norm(julia_pred - python_pred) / norm_julia))
                phi_julia_pred = load_phi(JULIA_DIR, map_size, polarity, seed, trial, "phiJ")
                phi_python_pred = load_phi(PYTHON_DIR, map_size, polarity, seed, trial, "phi_python_predict")
                norm_phi = jnp.linalg.norm(phi_julia_pred)
                samples["phi"].append(float(jnp.linalg.norm(phi_julia_pred - phi_python_pred) / norm_phi))
        for c in components:
            arr = np.array(samples[c])
            means[c].append(np.mean(arr) if len(arr) > 0 else 0.0)
            stds[c].append(np.std(arr) if len(arr) > 0 else 0.0)
    return components, means, stds


def average_cross_correlation(polarity, available_map_sizes):
    field_names = get_field_names_for_polarity(polarity)
    components = field_names + ["phi"]
    js_jp_results = {c: {} for c in components}
    js_pp_results = {c: {} for c in components}
    pp_jp_results = {c: {} for c in components}

    for map_size in available_map_sizes:
        accum_js_jp = {c: None for c in components}
        accum_js_pp = {c: None for c in components}
        accum_pp_jp = {c: None for c in components}
        ell_out = {c: None for c in components}
        count = 0

        for seed in get_available_seeds(polarity, map_size):
            for trial in get_available_trials(JULIA_DIR, map_size, polarity, seed, cached=True):
                for fn in field_names:
                    julia_sim = load_field(PYTHON_DIR, map_size, polarity, seed, trial, f"{fn}_julia_simulation")
                    julia_pred = load_field(JULIA_DIR, map_size, polarity, seed, trial, fn)
                    python_pred = load_field(PYTHON_DIR, map_size, polarity, seed, trial, f"{fn}_python_predict")

                    ell, rho_js_jp = primal_cross_correlation(julia_sim, julia_pred, THETA_PIX)
                    _, rho_js_pp = primal_cross_correlation(julia_sim, python_pred, THETA_PIX)
                    _, rho_pp_jp = primal_cross_correlation(python_pred, julia_pred, THETA_PIX)

                    if accum_js_jp[fn] is None:
                        accum_js_jp[fn] = np.zeros_like(rho_js_jp)
                        accum_js_pp[fn] = np.zeros_like(rho_js_pp)
                        accum_pp_jp[fn] = np.zeros_like(rho_pp_jp)
                        ell_out[fn] = ell

                    accum_js_jp[fn] += np.array(rho_js_jp)
                    accum_js_pp[fn] += np.array(rho_js_pp)
                    accum_pp_jp[fn] += np.array(rho_pp_jp)

                phi_sim = load_phi(PYTHON_DIR, map_size, polarity, seed, trial, "phi_julia_simulation")
                phi_julia_pred = load_phi(JULIA_DIR, map_size, polarity, seed, trial, "phiJ")
                phi_python_pred = load_phi(PYTHON_DIR, map_size, polarity, seed, trial, "phi_python_predict")

                ell_phi, rho_phi_js_jp = primal_cross_correlation(phi_sim, phi_julia_pred, THETA_PIX)
                _, rho_phi_js_pp = primal_cross_correlation(phi_sim, phi_python_pred, THETA_PIX)
                _, rho_phi_pp_jp = primal_cross_correlation(phi_python_pred, phi_julia_pred, THETA_PIX)

                if accum_js_jp["phi"] is None:
                    accum_js_jp["phi"] = np.zeros_like(rho_phi_js_jp)
                    accum_js_pp["phi"] = np.zeros_like(rho_phi_js_pp)
                    accum_pp_jp["phi"] = np.zeros_like(rho_phi_pp_jp)
                    ell_out["phi"] = ell_phi

                accum_js_jp["phi"] += np.array(rho_phi_js_jp)
                accum_js_pp["phi"] += np.array(rho_phi_js_pp)
                accum_pp_jp["phi"] += np.array(rho_phi_pp_jp)
                count += 1

        if count > 0:
            for c in components:
                js_jp_results[c][map_size] = (ell_out[c], accum_js_jp[c] / count)
                js_pp_results[c][map_size] = (ell_out[c], accum_js_pp[c] / count)
                pp_jp_results[c][map_size] = (ell_out[c], accum_pp_jp[c] / count)

    return components, js_jp_results, js_pp_results, pp_jp_results


# --- Goal 1: Cached run time vs map size ---
def plot_cached_run_times():
    out_dir = os.path.join(ANALYSIS_DIR, "run_time_vs_map_size")
    os.makedirs(out_dir, exist_ok=True)

    for polarity in get_available_polarities():
        available_map_sizes = get_available_map_sizes(polarity)
        julia_means, julia_stds = average_times(JULIA_DIR, polarity, available_map_sizes, cached=True)
        python_means, python_stds = average_times(PYTHON_DIR, polarity, available_map_sizes, cached=True)
        x = np.array(available_map_sizes)
        jm, js = np.array(julia_means), np.array(julia_stds)
        pm, ps = np.array(python_means), np.array(python_stds)
        fig, ax = plt.subplots()
        ax.plot(x, jm, "o-", label="Julia (cached)")
        ax.fill_between(x, jm - js, jm + js, alpha=0.3)
        ax.plot(x, pm, "o-", label="Python (cached)")
        ax.fill_between(x, pm - ps, pm + ps, alpha=0.3)
        ax.set_xlabel("Map Size (Nside)")
        ax.set_ylabel("Average Cached Run Time (s)")
        ax.set_title(f"Cached Run Time vs Map Size — Polarity: {polarity}")
        ax.legend()
        ax.set_xticks(available_map_sizes)
        fig.savefig(os.path.join(out_dir, f"cached_run_time_{polarity}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# --- Goal 5: Uncached run time vs map size ---
def plot_uncached_run_times():
    out_dir = os.path.join(ANALYSIS_DIR, "run_time_vs_map_size")
    os.makedirs(out_dir, exist_ok=True)

    for polarity in get_available_polarities():
        available_map_sizes = get_available_map_sizes(polarity)
        julia_means, julia_stds = average_times(JULIA_DIR, polarity, available_map_sizes, cached=False)
        python_means, python_stds = average_times(PYTHON_DIR, polarity, available_map_sizes, cached=False)
        x = np.array(available_map_sizes)
        jm, js = np.array(julia_means), np.array(julia_stds)
        pm, ps = np.array(python_means), np.array(python_stds)
        fig, ax = plt.subplots()
        ax.plot(x, jm, "o-", label="Julia (uncached)")
        ax.fill_between(x, jm - js, jm + js, alpha=0.3)
        ax.plot(x, pm, "o-", label="Python (uncached)")
        ax.fill_between(x, pm - ps, pm + ps, alpha=0.3)
        ax.set_xlabel("Map Size (Nside)")
        ax.set_ylabel("Average Uncached Run Time (s)")
        ax.set_title(f"Uncached Run Time vs Map Size — Polarity: {polarity}")
        ax.legend()
        ax.set_xticks(available_map_sizes)
        fig.savefig(os.path.join(out_dir, f"uncached_run_time_{polarity}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# --- Goal 2: Fractional difference (sim vs prediction) vs map size ---
def plot_fractional_difference():
    out_dir = os.path.join(ANALYSIS_DIR, "perc_diff_vs_map_size")
    os.makedirs(out_dir, exist_ok=True)

    for polarity in get_available_polarities():
        available_map_sizes = get_available_map_sizes(polarity)
        components, julia_means, julia_stds, python_means, python_stds = average_fractional_difference(polarity, available_map_sizes)
        x = np.array(available_map_sizes)
        fig, ax = plt.subplots()
        for c in components:
            jm, js = np.array(julia_means[c]), np.array(julia_stds[c])
            pm, ps = np.array(python_means[c]), np.array(python_stds[c])
            ax.plot(x, jm, "o-", label=f"Julia Pred vs Sim ({c})")
            ax.fill_between(x, jm - js, jm + js, alpha=0.3)
            ax.plot(x, pm, "o-", label=f"Python Pred vs Sim ({c})")
            ax.fill_between(x, pm - ps, pm + ps, alpha=0.3)
        ax.set_xlabel("Map Size (Nside)")
        ax.set_ylabel("Average Fractional Difference")
        ax.set_title(f"Fractional Difference vs Map Size — Polarity: {polarity}")
        ax.legend()
        ax.set_xticks(available_map_sizes)
        fig.savefig(os.path.join(out_dir, f"frac_diff_sim_vs_pred_{polarity}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# --- Goal 3: Julia prediction vs Python prediction fractional difference ---
def plot_julia_vs_python_diff():
    out_dir = os.path.join(ANALYSIS_DIR, "perc_diff_vs_map_size")
    os.makedirs(out_dir, exist_ok=True)

    for polarity in get_available_polarities():
        available_map_sizes = get_available_map_sizes(polarity)
        components, means, stds = average_julia_vs_python_diff(polarity, available_map_sizes)
        x = np.array(available_map_sizes)
        fig, ax = plt.subplots()
        for c in components:
            m, s = np.array(means[c]), np.array(stds[c])
            ax.plot(x, m, "o-", label=f"Julia Pred vs Python Pred ({c})")
            ax.fill_between(x, m - s, m + s, alpha=0.3)
        ax.set_xlabel("Map Size (Nside)")
        ax.set_ylabel("Average Fractional Difference")
        ax.set_title(f"Julia vs Python Prediction Difference — Polarity: {polarity}")
        ax.legend()
        ax.set_xticks(available_map_sizes)
        fig.savefig(os.path.join(out_dir, f"frac_diff_julia_vs_python_{polarity}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# --- Goal 4: Cross correlation vs map size ---
def plot_cross_correlations():
    out_dir = os.path.join(ANALYSIS_DIR, "correlation_vs_map_size")
    os.makedirs(out_dir, exist_ok=True)

    for polarity in get_available_polarities():
        available_map_sizes = get_available_map_sizes(polarity)
        components, js_jp, js_pp, pp_jp = average_cross_correlation(polarity, available_map_sizes)
        for component in components:
            for map_size in available_map_sizes:
                if map_size not in js_jp[component]:
                    continue
                ell_1, rho_1 = js_jp[component][map_size]
                ell_2, rho_2 = js_pp[component][map_size]
                ell_3, rho_3 = pp_jp[component][map_size]
                fig, ax = plt.subplots()
                ax.plot(ell_1, rho_1, label="Julia Sim x Julia Pred")
                ax.plot(ell_2, rho_2, label="Julia Sim x Python Pred")
                ax.plot(ell_3, rho_3, label="Python Pred x Julia Pred")
                ax.set_xlabel("$\\ell$")
                ax.set_ylabel("Cross Correlation $\\rho(\\ell)$")
                ax.set_title(f"Cross Correlation — {component} — N={map_size} — Polarity: {polarity}")
                ax.legend()
                fig.savefig(os.path.join(out_dir, f"cross_correlation_{polarity}_{component}_N{map_size}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)


if __name__ == "__main__":
    plot_cached_run_times()
    plot_uncached_run_times()
    plot_fractional_difference()
    plot_julia_vs_python_diff()
    plot_cross_correlations()
    print("Analysis complete. Plots saved to:", ANALYSIS_DIR)
