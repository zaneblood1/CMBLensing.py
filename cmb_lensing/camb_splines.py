"""Download of the CAMB spline caches from the Hugging Face dataset CAMB_SPLINES_REPO.

The five 1D caches (camb_<param>_grid.npz) and the ~7.9 GB 5D grid (camb_grid_spline.npz)
are not in git. Each one is fetched into camb_splines/ (constants.CAMB_SPLINES_DIR) the first
time it is asked for. To fetch all of them ahead of time (e.g. on a cluster login node, since
compute nodes often have no internet):
    cmb-lensing-download-splines
"""

import os
from huggingface_hub import hf_hub_download

from cmb_lensing.constants import CAMB_SPLINES_DIR, CAMB_SPLINES_REPO, CAMB_SPLINE_FILES

def camb_spline_file(filename):
    #path of filename in camb_splines/, downloading it first if it is not there yet
    path = os.path.join(CAMB_SPLINES_DIR, filename)
    if not os.path.exists(path):
        print(f"downloading {filename} from huggingface.co/datasets/{CAMB_SPLINES_REPO}")
        hf_hub_download(repo_id = CAMB_SPLINES_REPO, filename = filename,
                        repo_type = "dataset", local_dir = CAMB_SPLINES_DIR)
    return path

def main():
    for filename in CAMB_SPLINE_FILES:
        print(camb_spline_file(filename))

if __name__ == "__main__":
    main()
