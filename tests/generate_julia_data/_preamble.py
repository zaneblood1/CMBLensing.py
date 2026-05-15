import os
os.environ["PYTHON_JULIAPKG_PROJECT"] = "/home/zane-blood/CMBLensing.jl"
os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"

from cmb_lensing.constants import THETA_PIX_DEFAULT, NSIDE_DEFAULT


def init_julia():
    from juliacall import Main as jl
    jl.seval(f"""
        DIR = pwd()
        FILE_PATH = DIR * "/tests/ground_truth_data/"
             
        using CMBLensing
        using PythonPlot
        using NPZ
             
        seed = 67
        θpix = {THETA_PIX_DEFAULT}
        Nside = {NSIDE_DEFAULT}
        Cℓ = camb(ℓmax = 17000)

        open(FILE_PATH * "theta_pix.txt", "w") do io
            println(io, θpix)
        end
        open(FILE_PATH * "n_side.txt", "w") do io
            println(io, Nside)
        end

        temp_only_sim = load_sim(seed=seed, θpix=θpix, T=Float64, Nside=Nside, pol=:I, Cℓ=Cℓ);
        polar_only_sim = load_sim(seed=seed, θpix=θpix, T=Float64, Nside=Nside, pol=:P, Cℓ=Cℓ);
        polar_and_temp_sim = load_sim(seed=seed, θpix=θpix, T=Float64, Nside=Nside, pol=:IP, Cℓ=Cℓ);

    """)
    return jl
