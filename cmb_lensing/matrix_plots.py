"""Annotated heat maps of parameter matrices (Fisher, covariance, correlation), shared by
fisher_forecast.py and chain_analysis.py."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FIGURE_DPI = 150
#the annotation switches to white above this |normalized value| / below this luminance
DARK_CELL_THRESHOLD = 0.6
DARK_LUMINANCE_THRESHOLD = 0.5
#Rec. 601 luma weights of the red, green and blue channels
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def annotated_heatmap(axis, matrix, names, title, colorbar_label):
    """Heatmap coloured by the correlation-normalized matrix (bounded, readable) with the
    raw entries annotated - the raw values span many orders of magnitude across parameters
    with wildly different units, so colouring by them directly is useless."""
    diagonal = np.sqrt(np.abs(np.diag(matrix)))
    normalized = matrix / np.outer(diagonal, diagonal)

    image = axis.imshow(normalized, cmap = "RdBu_r", vmin = -1, vmax = 1)
    axis.set_xticks(range(len(names)))
    axis.set_yticks(range(len(names)))
    axis.set_xticklabels(names, rotation = 45, ha = "right")
    axis.set_yticklabels(names)
    axis.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            shade = "white" if abs(normalized[i, j]) > DARK_CELL_THRESHOLD else "black"
            axis.text(j, i, f"{matrix[i, j]:.2e}", ha = "center", va = "center",
                      color = shade, fontsize = 7)
    colorbar = plt.colorbar(image, ax = axis, fraction = 0.046, pad = 0.04)
    colorbar.set_label(colorbar_label)
    return image


def plot_annotated_matrix(matrix, param_names, title, path, cmap = "coolwarm", norm = None,
                          fmt = "{:+.3f}"):
    """A square matrix as a coloured grid with every entry printed in its box (black or white
    text by the luminance behind it). `norm` defaults to a linear scale symmetric about 0."""
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
            luminance = np.dot(LUMA_WEIGHTS, rgba[:3])
            ax.text(j, i, fmt.format(matrix[i, j]), ha = "center", va = "center",
                    color = "white" if luminance < DARK_LUMINANCE_THRESHOLD else "black",
                    fontsize = 11)
    fig.colorbar(image, ax = ax, fraction = 0.046, pad = 0.04)
    ax.set_title(title)
    plt.savefig(path, dpi = FIGURE_DPI, bbox_inches = "tight")
    plt.close(fig)
