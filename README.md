CMBLensing.py is the JAX compatible version of the original CMBLensing.jl Julia package. Just like the original Julia code it was migrated from, it allows the user to generate sample temperature and polarization CMB fields, lensing potentials, and covariance matrices. The lense_flow algorithm can then be used to quickly lense, inverse lense, or adjoint lense these fields. By taking gradients of the logpdf function which computes the loglikelihood that a given f and phi pair were the sources of an observed data field, we can compute the maximum likelihood estimators for f and phi. 

Once the code is downloaded onto your local computer (e.g. using git clone) run "pip install -e ." from the /cmb_lensing folder to compile the pyproject.toml file and load all the necessary dependencies. In order to generate fresh Julia comparison data to run the unit tests, you will need to have juliacall installed in your environment and the original CMBLensing.jl installed as well. 

The unit tests can either be called in bulk or individually. To generate a fresh set of data for a specific set of unit tests, from the VSCode terminal run "python tests/generate_julia_data/generate_[name of test].py". To generate data for ALL the unit tests, simply run "python /tests/generate_julia_data/generate_all.py". In order to run specific unit tests, run "pytest /tests/test_[name of test].py" or to run all unit tests at once simply call "pytest". Whenever calling unit tests, you can specify the "--generate" flag which tells python to automatically generate a fresh batch of Julia data for that unit test or set of unit tests. 

Note that the way these unit tests work is to compare the results from CMBLensing.jl to CMBLensing.py. Therefore, they are more of a set of benchmark tests than actual unit tests. This is just a sanity check to ensure that there is no different behavior between the two different language implementations of CMBLensing. Also note that many of the unit tests do not actually explicitly error if there is a dramatic difference between Julia and Python. Instead, to be safe, the user should launch the unit test visualizer located in /tests/index.html with e.g. the VSCode Go-Live extension and manually inspect the comparison plots. 

The file structure of the code is as follows:

```
cmb_lensing/
├── LICENSE
├── README.md
├── pyproject.toml
├── camb_emulator/
│   ├── emu_derived.npz
│   ├── emu_PP.npz
│   ├── emu_uEE.npz
│   ├── emu_uTE.npz
│   └── emu_uTT.npz
├── cmb_lensing/
│   ├── __init__.py
│   ├── constants.py
│   ├── dataset.py
│   ├── fields.py
│   ├── gradients.py
│   ├── lense_flow.py
│   ├── map_joint.py
│   ├── matrix_operators.py
│   ├── mixing.py
│   ├── sample_lcdm.py
│   ├── sampling_ar.py
│   ├── simulate.py
│   ├── statistics.py
│   ├── util.py
│   └── wiener_filter.py
├── docs/
│   └── tutorial.ipynb
└── tests/
    ├── conftest.py
    ├── index.html
    ├── styles.css
    ├── test_covariance_matrices.py
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
