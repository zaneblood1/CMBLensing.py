import os
import re
import matplotlib.pyplot as plt


def figure_filename(name):
    """Lowercase snake_case name that is valid on Windows, Linux and macOS (only a-z, 0-9 and _)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_figure(path, **kwargs):
    """plt.savefig that creates missing folders and makes the file name portable.

    test_generated_figures/ is generated output, so its folders may be absent
    (fresh clone, sparse checkout); never rely on them existing.
    """
    folder, file = os.path.split(path)
    stem, ext = os.path.splitext(file)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, figure_filename(stem) + ext), **kwargs)
