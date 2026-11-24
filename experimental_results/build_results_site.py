"""Builds the static results website's figure tree and manifest from experimental_results/.

The folder layout is the whole contract:

    experimental_results/<section>/<dataset>/     e.g. lcdm_sampling/2.5uk_summer_2026/
        **.png        the figures; a sub-folder is one headed group on the page
        SUMMARY.md    optional, rendered to HTML here and shown above the figures

Every folder inside a section is one option of that section's data set toggle, named after the
folder, so a chain_analysis output folder is a data set as it is: rename it and drop it into
lcdm_sampling/. HIDDEN_FIGURE_PATTERNS lists the figures left out. Nothing but the shown PNGs is
copied to the site (a data/ folder may sit next to them). GitHub
Pages serves no directory listings, so the page (docs/results/) reads the manifest.json written
here.

MkDocs runs on_post_build after every build (mkdocs.yml `hooks`), so `mkdocs serve` and the docs
CI both pick up a newly dropped data set. Standalone:

    python experimental_results/build_results_site.py --out_dir <dir>
"""

import argparse
import json
import os
import re
import shutil

import markdown

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
#where the page lives inside the built site
SITE_SUBDIR = "results"
#the folder of the built site the figures are copied into
SITE_FIGURES_DIR = "figures"
MANIFEST_FILE_NAME = "manifest.json"
SUMMARY_FILE_NAME = "SUMMARY.md"
SUMMARY_MARKDOWN_EXTENSIONS = ["tables"]
FIGURE_EXTENSION = ".png"
#sections in toggle order with their titles; any other folder follows, titled from its name
SECTION_TITLES = {
    "lcdm_sampling": "LCDM Sampler",
    "map_joint_comparison": "Python v.s. Julia Runtime Comparison",
    "phi_noise": "Noise Spectra",
}
#figures that are kept in the repository but not shown on the page: the covariance / correlation
#matrices and their ratios (the triangle plot stands for them), the histograms of per-map
#means / modes, the single-component cross correlations (the "all" panel
#figure has them), the run time tables, the uncached run times and the Julia vs Python
#prediction difference
HIDDEN_FIGURE_PATTERNS = [
    re.compile(r"(covariance|correlation)_(matrix|ratio)"),
    re.compile(r"covar_ratio"),
    re.compile(r"_(mean|mode)_histogram"),
    re.compile(r"cross_correlation_(?!.*_all_N\d+\.png$)"),
    re.compile(r"run_time_table_"),
    re.compile(r"uncached_run_time_"),
    re.compile(r"frac_diff_julia_vs_python_"),
]

def sub_dirs(path):
    """Names of the visible directories inside path, sorted."""
    return sorted(name for name in os.listdir(path)
                  if os.path.isdir(os.path.join(path, name)) and not name.startswith((".", "__")))

def title_from_name(name):
    return name.replace("_", " ").title()

def summary_html(directory):
    """The directory's SUMMARY.md rendered to HTML, None without one."""
    path = os.path.join(directory, SUMMARY_FILE_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as summary_file:
        return markdown.markdown(summary_file.read(), extensions = SUMMARY_MARKDOWN_EXTENSIONS)

def figure_groups(dataset_dir):
    """[{name, files}]: one group per directory holding shown PNGs, files relative to dataset_dir."""
    groups = []
    for directory, dir_names, file_names in os.walk(dataset_dir):
        dir_names.sort()
        figures = sorted(name for name in file_names if name.lower().endswith(FIGURE_EXTENSION)
                         and not any(pattern.search(name) for pattern in HIDDEN_FIGURE_PATTERNS))
        if not figures:
            continue
        group = os.path.relpath(directory, dataset_dir).replace(os.sep, "/")
        group = "" if group == "." else group
        groups.append({"name": group, "files": ["/".join(filter(None, [group, name])) for name in figures]})
    return groups

def scan_section(section, section_dir):
    """The section's data sets, one per folder: [{name, folder, path, groups, summary_html}]."""
    datasets = []
    for name in sub_dirs(section_dir):
        dataset_dir = os.path.join(section_dir, name)
        folder = f"{section}/{name}"
        datasets.append({
            "name": name,
            "folder": folder,
            "path": f"{SITE_FIGURES_DIR}/{folder}",
            "groups": figure_groups(dataset_dir),
            "summary_html": summary_html(dataset_dir),
        })
    return datasets

def scan_results(results_dir = RESULTS_DIR):
    """The manifest the page reads: every section and its data sets."""
    found = sub_dirs(results_dir)
    ordered = [name for name in SECTION_TITLES if name in found]
    ordered += [name for name in found if name not in SECTION_TITLES]
    return {"sections": [{"key": section,
                          "title": SECTION_TITLES.get(section, title_from_name(section)),
                          "datasets": scan_section(section, os.path.join(results_dir, section))}
                         for section in ordered]}

def build(results_dir, out_dir):
    """Copies every shown PNG under out_dir/figures/ and writes out_dir/manifest.json."""
    manifest = scan_results(results_dir)
    os.makedirs(out_dir, exist_ok = True)
    for section in manifest["sections"]:
        for dataset in section["datasets"]:
            source_dir = os.path.join(results_dir, dataset["folder"])
            target_dir = os.path.join(out_dir, dataset["path"])
            for group in dataset["groups"]:
                for file in group["files"]:
                    target = os.path.join(target_dir, file)
                    os.makedirs(os.path.dirname(target), exist_ok = True)
                    shutil.copyfile(os.path.join(source_dir, file), target)
    with open(os.path.join(out_dir, MANIFEST_FILE_NAME), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent = 1)
    return manifest

def on_post_build(config, **kwargs):
    """MkDocs hook: add the figures and the manifest to the built site."""
    build(RESULTS_DIR, os.path.join(config["site_dir"], SITE_SUBDIR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    parser.add_argument("--out_dir", required = True,
                        help = "where figures/ and manifest.json are written")
    args = parser.parse_args()
    build(RESULTS_DIR, args.out_dir)
