# CMBLensing.py

CMBLensing.py is the JAX port of the Julia package [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl). With it you can:

- **Simulate** flat-sky CMB temperature and polarization fields, lensing potentials and noisy lensed data maps (`load_sim`).
- **Lens**, inverse-lens and adjoint-lens fields with the LenseFlow ODE (`lense_flow`).
- **Reconstruct** the unlensed field and the lensing potential by joint MAP optimization (`map_joint`).
- **Sample** the LCDM parameters jointly with the field and the lensing potential by Gibbs sampling (`sample_joint`), extending the algorithm developed by [Millea et al. 2020](https://arxiv.org/abs/2002.00965).
- **Forecast** the sampler's posterior widths with a Gaussian Fisher forecast (`fisher_forecast`).

All array operations run in JAX (float64), so the code runs on CPU and GPU and is differentiable where it needs to be.

## Where to start

| I want to... | Go to |
|---|---|
| install the package | [Installation](getting-started/installation.md) |
| simulate a map and run `map_joint` | [MAP Estimation](map_joint_tutorial.ipynb) |
| learn more about sampling the LCDM parameters | [Running a Single LCDM Chain](sample_joint_tutorial.ipynb) |
| browse the results or download the chains and other data behind them | [Data Availability](getting-started/data.md) |
| run a full experiment on a cluster | [HPC setup](getting-started/installation.md#hpc-setup) |
| compare sampler data to the Fisher forecast | [Analyzing Chain Data](chain_analysis_tutorial.ipynb), [Fisher Forecast](api/fisher-forecast.md) |
| change the code | [Developing](getting-started/developing.md) |
