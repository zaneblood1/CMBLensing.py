# BACKGROUND

CMBLensing.py is the JAX compatible version of the original [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl) Julia package. Just like the original Julia code it was migrated from, it allows the user to generate synthetic temperature and polarization CMB fields, lensing potentials, and covariance matrices. The ```lense_flow``` algorithm can then be used to quickly lense, inverse lense, or adjoint lense these fields. By taking gradients of the logpdf function which computes the log-likelihood that a given ```(f, phi)``` pair were the sources of an observed data field, we can compute the maximum likelihood estimators for ```f``` and ```phi```. For map sizes at or below 256 x 256 square pixels, the Julia code and Python code are on the same order of magnitude in terms of the time it takes for the ```map_joint``` algorithm to finish. For larger maps, the Python code base is slightly slower. Decreasing the time complexity here is an active area of investigation. 

In addition to its ```map_joint``` implementation, CMBLensing.py also re-purposes the sampling algorithm employed in CMBLensing.jl (see e.g. [Millea M, et. al.](https://arxiv.org/abs/2002.00965)) to jointly sample the 5 LCDM parameters in the script ```sample_lcdm.py```. Given a synthetically generated data field as input (where the temperature and / or polarization fields were generated with known ground truth values) and a search range for each LCDM parameter, this algorithm recovers MCMC samples of each parameter which can be used to form probability distributions whose modes represent the inferred value of the cosmological parameter and whose widths indicate the uncertainty in the estimator. A single pilot chain can be run for a single data map by running the  ```__main__``` method in ```sample_lcdm.py``` or a larger experiment can be run by using the slurm scripts provided in ```get_lcdm_samples/``` to average over many data map realizations and minimize cosmic variance.

# GENERAL USE SETUP GUIDE

Once the code is downloaded onto your local computer (e.g. using ```git clone```) run ```pip install -e .``` from the repository root (the folder holding ```pyproject.toml```) to compile the ```pyproject.toml``` file and load all the necessary dependencies. 

An example notebook which shows how to generate synthetic data, lense maps, and run the ```map_joint``` algorithm is located in the ```docs/``` folder (```docs/map_joint_tutorial.ipynb```).

If your main wish is to use the sampling / LCDM parameter inference portion of the codebase over the ```map_joint``` algorithm, more information can be found in the "LCDM SAMPLING" section. 

In order to generate fresh Julia comparison data to run the unit tests, you will need to have [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) installed in your environment and the original [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl) installed as well. Unless you plan to act as a developer for the project and actually edit the source code - it is not necessary nor recommended to run or edit the unit tests.

# DEVELOPER SETUP GUIDE & UNIT TESTS

If you wish to add new features, upgrade existing portions of the code base, or otherwise serve in some similar "developer" role, it is recommended that you run and review the unit tests before pushing any changes. In order to run the unit tests, you must first edit the ```PYTHON_JULIAPKG_PROJECT``` file path in ```_preamble.py``` to match the actual location of ```CMBLensing.jl``` on your local machine. 

The unit tests mainly cover the ```map_joint``` and ```load_sim``` sections of the code base and do not check the sampling algorithm / LCDM parameter inference portion of the code. The unit tests are more of an A/B test against the original Julia code base that this repository was migrated from rather than true unit tests. We mainly check that for the same inputs the two codebases agree up to numerical precision in terms of their corresponding outputs. 

Most of the time the unit tests will only fail if something is horribly wrong. Often there can be silent failures and it is recommended to launch the unit test visualizer from the ```index.html``` file using e.g. VSCode's GoLive extension and visually inspect all of the diff plots. 

The unit tests can either be called in bulk or individually. 

To generate a fresh set of data for a specific set of unit tests, from the VSCode terminal run ```python tests/generate_julia_data/generate_[name of test].py```. 

To generate data for ALL the unit tests, simply run ```python tests/generate_julia_data/generate_all.py```. 

Once the comparison data is generated, in order to run specific unit tests, run ```pytest tests/test_[name of test].py``` or to run all unit tests at once simply call ```pytest```. 

Whenever calling unit tests, you can specify the ```--generate``` flag which tells python to automatically generate a fresh batch of Julia data for that unit test or set of unit tests.

After you have made any development changes and confirmed the unit tests still function, simply submit a pull request to have your changes reviewed and merged.

# SYNTHETIC FIELD GENERATION

CMBLensing.py is able to generate synthetic temperature, polarization, or temperature + polarization CMB fields, lensing potential maps, and data fields (which include lensing effects and white, 1/f, or beam noise and masking added in) with the use of the ```load_sim``` method located in ```simulate.py```. All of these maps are generated in the flat sky approximation using Fourier modes instead of spherical harmonics. Refer to ```docs/map_joint_tutorial.ipynb``` for an example of how to call the field generating method. 

# MAP JOINT

The ```map_joint``` method is based on the algorithm of the same name in CMBLensing.jl and was first introduced in [Millea M, et. al.](https://arxiv.org/pdf/1708.06753). It jointly calculates the Maximum A Posteriori estimates of the unlensed CMB field and lensing potential given a noisy, lensed data map and known covariance matrices (generated using e.g. CAMB). Behind the scenes, the ```map_joint``` algorithm employs an alternating gradient descent algorithm in ```(f, phi)``` parameter space to find the specific pair which minimizes a Gaussian log-likelihood. Refer to the sample Jupyter notebook for examples of how to call and run this method.

# LCDM SAMPLING

The sampling algorithm ```sample_joint``` that CMBLensing.py employs was originally developed in this [paper](https://arxiv.org/pdf/1708.06753) to jointly infer the lensing potential band power and the tensor-to-scalar-ratio from noisy, lensed polarization data. We slightly modify the algorithm here in the Python version to allow the user to jointly sample any or all of the LCDM parameters. 

The code is currently hard-coded to take a known prior value on the optical depth to reionization due to its strong degeneracies with some of the other parameters, but this could be modified if need be.

It is possible to either use temperature only, polarization only, or temperature + polarization data as the input to the inference engine by switching the ```pol``` flag to either ```"I", "P", or "IP"```. Switching to ```IP``` decreases the degeneracies between certain parameters, but it also raises the time complexity of the algorithm considerably. 

Five sequential 1-D Metropolis-Hastings steps are used in the specific step of the MCMC algorithm that samples the LCDM parameters. Before running a larger experiment on an HPC using the code base, it is recommend to run a pilot chain on a smaller map (e.g. 128 x 128 pixels) in order to tune the 5 MH proposal widths for each parameter as well as the number of steps and step size for the HMC step used to sample the lensing potential. One should target an acceptance rate of about 44% for each of the LCDM parameters and 65% for the lensing potential step to effectively explore the parameter space.

Once the tuning has been done for the map size, resolution, polarity, noise levels, masking, and beam configurations you wish to use, you may then follow the example code in the ```get_lcdm_samples/``` folder (```sample_lcdm.sh``` → ```run_single_lcdm_chain.sh/.py```) to run a larger experiment on an HPC by averaging over many data map realizations. Once the chains have converged, ```get_lcdm_samples/chain_analysis.py``` has code to convert these MCMC samples into distributions whose mode / std determine the best estimator / confidence level for the LCDM parameters given the input data maps. 

# PRECOMPUTED CAMB GRID (DOWNLOAD)

The LCDM sampler evaluates CAMB spectra through a 5D cubic spline over ```(H0, logA, ns, ombh2, omch2)``` stored in ```camb_splines/camb_grid_spline.npz``` at the repository root. That file is ~7.9 GB, so it is **not** in the repository (GitHub rejects it). Either rebuild it on an HPC with the scripts in ```camb_grid_generation/``` (```camb_grid.sh``` → ```merge_camb_grid.py``` → ```validate_camb_grid.py```, ~70k CAMB calls) or download the prebuilt copy from Google Drive:

[Download camb_grid_spline.npz (Google Drive, ~7.9 GB)](https://drive.google.com/file/d/1HhVjNPMi4OR3vn7j7DkdK0pDnF_iNVBr/view?usp=sharingg)

Free Drive accounts occasionally hit a daily download quota on large shared files; if ```gdown``` reports "too many users have viewed or downloaded this file", retry the next day or download it manually from the link above and place it at the path shown. The five small 1D caches (```camb_<param>_grid.npz```, ~3 MB each) are tracked in the repository and need no download. ```sample_lcdm.py``` looks for the grid at exactly ```camb_splines/camb_grid_spline.npz``` (```cmb_lensing.constants.CAMB_GRID_PATH```).

# FILE STRUCTURE

```
cmb_lensing/
├── .gitignore
├── CLAUDE.md
├── LICENSE
├── README.md
├── pyproject.toml
├── cmb_lensing/                       (the Python package)
│   ├── __init__.py
│   ├── camb_grid_interp.py
│   ├── constants.py                   (also PARAM_ORDER, GROUND_TRUTH, PARAM_BOUNDS, PARAM_SIGMA, CAMB_GRID_PATH)
│   ├── dataset.py
│   ├── delensed_covariance.py
│   ├── fields.py
│   ├── fisher_forecast.py
│   ├── fixed_theta_chain.py
│   ├── gradients.py
│   ├── lense_flow.py
│   ├── map_joint.py
│   ├── map_joint_phi_noise.py
│   ├── matrix_operators.py
│   ├── mixing.py
│   ├── sample_lcdm.py
│   ├── sampler_effective_phi_noise.py
│   ├── simulate.py
│   ├── statistics.py
│   ├── util.py
│   ├── wiener_filter.py
│   ├── fisher_output/                 (fisher_forecast.py output)
│   └── sample_lcdm_output/            (sample_lcdm.py progress plots)
├── camb_splines/
│   ├── camb_grid_spline.npz           (5D CAMB grid from merge_camb_grid.py; download, not tracked)
│   ├── camb_logA_grid.npz             (1D caches from precompute_camb_1d.py)
│   ├── camb_ns_grid.npz
│   ├── camb_ombh2_grid.npz
│   ├── camb_omch2_grid.npz
│   └── camb_theta_MC_100_grid.npz
├── camb_grid_generation/
│   ├── precompute_camb_1d.py          (python camb_grid_generation/precompute_camb_1d.py <param|all>)
│   ├── camb_grid.sh
│   ├── run_single_camb_grid.py
│   ├── run_single_camb_grid.sh
│   ├── merge_camb_grid.py
│   └── validate_camb_grid.py
├── docs/
│   ├── map_joint_tutorial.ipynb
│   └── sample_joint_tutorial.ipynb
├── get_lcdm_samples/                  (LCDM chains on an HPC + their analysis)
│   ├── sample_lcdm.sh
│   ├── run_single_lcdm_chain.py
│   ├── run_single_lcdm_chain.sh
│   ├── chain_analysis.py
│   ├── spectra_plots/
│   ├── lcdm_sampler_data/             (chain *_history.txt directories)
│   └── lcdm_chain_plots/
├── get_delensed_field_covariance/     (empirical delensed covariance stencil)
│   ├── get_delensed_covariance.sh
│   ├── run_single_delensed_covariance.py
│   ├── run_single_delensed_covariance.sh
│   ├── merge_delensed_covariance.py
│   └── delensed_field_covariance_data/
├── get_sampler_phi_noise/             (the sampler's effective phi noise)
│   ├── get_sampler_effective_noise.sh
│   ├── get_sampler_effective_noise_1_chain.py
│   ├── get_sampler_effective_noise_1_chain.sh
│   ├── merge_sampler_effective_noise.py
│   ├── merge_sampler_effective_noise.sh
│   └── sampler_phi_noise_data/
├── get_map_joint_phi_noise/           (map_joint's band-averaged phi noise)
│   ├── get_effective_phi_noise.sh
│   ├── get_phi_noise_1_realization.py
│   ├── get_phi_noise_1_realization.sh
│   └── merge_phi_noise.py
├── runtime_comparison/                (map_joint Python vs Julia timing)
│   ├── julia_performance_test.jl
│   ├── julia_performance_test.sh
│   ├── performance_analysis.py
│   ├── python_performance_test.py
│   ├── python_performance_test.sh
│   └── run_performance_test.sh
├── experimental_results/
├── paper/
└── tests/
    ├── conftest.py
    ├── index.html
    ├── styles.css
    ├── test_covariance_matrices.py
    ├── test_fisher_forecast.py
    ├── test_native_5d_interp.py
    ├── test_gradients.py
    ├── test_lensing.py
    ├── test_logpdf.py
    ├── test_map_joint.py
    ├── test_simulated_cls.py
    ├── test_wiener_filter.py
    ├── generate_julia_data/
    │   ├── __init__.py
    │   ├── _preamble.py
    │   ├── generate_all.py
    │   ├── generate_covariance_matrices.py
    │   ├── generate_gradients.py
    │   ├── generate_lensing.py
    │   ├── generate_logpdf.py
    │   ├── generate_map_joint.py
    │   ├── generate_simulated_cls.py
    │   └── generate_wiener_filter.py
    ├── ground_truth_data/
    └── test_generated_figures/
```
