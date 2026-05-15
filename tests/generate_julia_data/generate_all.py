import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _preamble import init_julia

from generate_lensing import run as run_lensing
from generate_logpdf import run as run_logpdf
from generate_gradients import run as run_gradients
from generate_wiener_filter import run as run_wiener_filter
from generate_map_joint import run as run_map_joint
from generate_simulated_cls import run as run_simulated_cls
from generate_covariance_matrices import run as run_covariance_matrices

if __name__ == "__main__":
    print("\n=== Running all data generators ===")
    jl = init_julia()
    run_lensing(jl)
    run_logpdf(jl)
    run_gradients(jl)
    run_wiener_filter(jl)
    run_map_joint(jl)
    run_simulated_cls(jl)
    run_covariance_matrices(jl)
    print("\n=== All generators complete! ===")
