"""How the phi chains' integrated autocorrelation time moves with the burn-in: reads every
sampler_effective_noise*.npz of a merge_sampler_effective_noise.py --burn_in_scan directory
(one per burn-in) and plots the per-band IAT against L and against the burn-in.

    python plot_iat_vs_burn_in.py [--chain_dir <dir>] [--out <png>]
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
#the bands followed against the burn-in (lower band edges; the nearest measured band is used)
TRACKED_ELLS = (0, 100, 200, 400, 900, 1900, 2900, 3900, 4900)


def load_scan(chain_dir):
    """[(burn_in, merged)] sorted by burn-in, one entry per merged file in `chain_dir`."""
    scan = {}
    for path in sorted(glob.glob(os.path.join(chain_dir, "sampler_effective_noise*.npz"))):
        merged = np.load(path)
        if "band_iat_median" not in merged:
            continue
        burn_in = int(merged["burn_in"])
        if burn_in in scan:
            raise ValueError(f"{path}: burn-in {burn_in} appears twice in {chain_dir}")
        scan[burn_in] = merged
    if not scan:
        raise FileNotFoundError(f"no merged sampler_effective_noise*.npz in {chain_dir}")
    return sorted(scan.items())


def main():
    parser = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    parser.add_argument("--chain_dir", default = os.path.join(
        HERE, "sampler_effective_noise_output_mean_chains_5uk"))
    parser.add_argument("--out", default = None)
    args = parser.parse_args()
    out = args.out or os.path.join(args.chain_dir, "sampler_effective_noise_iat_vs_burn_in.png")

    scan = load_scan(args.chain_dir)
    burn_ins = np.array([burn_in for burn_in, _ in scan])
    first = scan[0][1]
    ells, edges = np.asarray(first["band_ells"]), np.asarray(first["band_edges"])
    measured = np.asarray(first["band_count"]) > 0

    #per burn-in and band: the median over chains of the band's median IAT, its 16-84%
    #spread over chains, and the worst chain's worst mode
    with np.errstate(all = "ignore"):
        median = np.array([np.nanmedian(m["band_iat_median"], axis = 0) for _, m in scan])
        low = np.array([np.nanpercentile(m["band_iat_median"], 16, axis = 0) for _, m in scan])
        high = np.array([np.nanpercentile(m["band_iat_median"], 84, axis = 0) for _, m in scan])
        worst = np.array([np.nanmax(m["band_iat_max"], axis = 0) for _, m in scan])
    measured &= np.all(np.isfinite(median), axis = 0)
    n_used = [int(np.min(m["n_used"])) for _, m in scan]

    figure, axes = plt.subplots(2, 2, figsize = (13, 10))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(scan)))
    for i, burn_in in enumerate(burn_ins):
        label = f"burn-in {burn_in} ({n_used[i]} sweeps kept)"
        axes[0, 0].loglog(ells[measured], median[i][measured], color = colors[i], marker = ".",
                          linewidth = 1, label = label)
        axes[0, 0].loglog(ells[measured], worst[i][measured], color = colors[i],
                          linestyle = ":", linewidth = 1)
        axes[0, 1].semilogx(ells[measured], median[i][measured] / median[0][measured],
                            color = colors[i], marker = ".", linewidth = 1, label = label)
    axes[0, 0].plot([], [], color = "0.4", linestyle = ":", label = "worst mode of the worst chain")
    axes[0, 0].set_ylabel("IAT (sweeps)")
    axes[0, 0].set_title("solid: median over chains of the band-median IAT")
    axes[0, 1].axhline(1.0, color = "0.5", linestyle = "--", linewidth = 1)
    axes[0, 1].set_ylabel(f"median IAT / median IAT at burn-in {burn_ins[0]}")
    for axis in axes[0]:
        axis.set_xlabel(r"$L$")
        axis.legend(fontsize = 8)

    tracked = sorted({int(np.argmin(np.where(measured, np.abs(edges[:-1] - ell), np.inf)))
                      for ell in TRACKED_ELLS})
    band_colors = plt.cm.plasma(np.linspace(0.0, 0.9, len(tracked)))
    for color, b in zip(band_colors, tracked):
        label = f"L {edges[b]:.0f}-{edges[b + 1]:.0f}"
        axes[1, 0].plot(burn_ins, median[:, b], color = color, marker = "o", label = label)
        axes[1, 0].fill_between(burn_ins, low[:, b], high[:, b], color = color, alpha = 0.15)
        axes[1, 1].plot(burn_ins, worst[:, b], color = color, marker = "o", label = label)
    axes[1, 0].set_ylabel("band-median IAT (sweeps): median and 16-84% over chains")
    axes[1, 1].set_ylabel("worst mode of the worst chain: IAT (sweeps)")
    for axis in axes[1]:
        axis.set_yscale("log")
        axis.set_xlabel("burn-in (sweeps)")
        axis.set_xticks(burn_ins)
        axis.legend(fontsize = 8, ncol = 3)

    figure.suptitle(f"phi-chain IAT vs burn-in | nside {int(first['nside'])}, "
                    f"{float(first['theta_pix']):g}', {float(first['noise_level']):g} uK-arcmin | "
                    f"{int(first['n_realizations'])} chains", fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.96))
    figure.savefig(out, dpi = 150)
    plt.close(figure)

    print(f"{'L band':>12s} " + " ".join(f"{b:>8d}" for b in burn_ins) + "   (band-median IAT, "
          "median over chains, per burn-in)")
    for b in tracked:
        print(f"{edges[b]:5.0f}-{edges[b + 1]:<6.0f} " + " ".join(f"{v:8.1f}" for v in median[:, b]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
