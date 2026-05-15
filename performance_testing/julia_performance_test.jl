using Pkg
Pkg.activate("/resnick/groups/wugroup/zblood/CMBLensing.jl/Project.toml")
using CMBLensing
using ArgParse
using NPZ

#load in the task id and the f, phi combo id
function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--map_size"
            arg_type = Int
        "--seed"
            arg_type = Int
        "--trial"
            arg_type = Int
        "--pol"
            arg_type = String
        "--theta_pix"
            arg_type = Float64
    end
    return parse_args(settings)
end

args = parse_commandline()
trial = args["trial"]
map_size = args["map_size"]
seed = args["seed"]
theta_pix = args["theta_pix"]

polarity = args["pol"]
if polarity == "I"
    pol = :I
end
if polarity == "P"
    pol = :P
end 
if polarity == "IP"
    pol = :IP
end  

#run load_sim() using the given seed and settings
(;f, f̃, ϕ, ds) = load_sim(
    seed = seed,
    θpix = theta_pix,
    T = Float64,
    Nside = map_size,
    pol = pol
)

#run the algorithm once initially to get the uncached time
start_time = time()
fJ, phiJ = MAP_joint(ds, nsteps = 30, progress = false);
end_time = time()
uncached_time = end_time - start_time

#store the uncached time for this (f, phi) combo in its own proper directory
mkpath("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/uncached_times")
open("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/uncached_times/uncached_time_$(trial).txt", "w") do io
    write(io, string(uncached_time))
end

#now run the algorithm a second time to get the cache time
start_time = time()
_, _ = MAP_joint(ds, nsteps = 30, progress = false);
end_time = time()
cached_time = end_time - start_time

#store the cached time for this (f, phi) combo in its own proper directory
mkpath("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/cached_times")
open("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/cached_times/cached_time_$(trial).txt", "w") do io
    write(io, string(cached_time))
end

#store the learned / estimated f and phi
mkpath("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb")
mkpath("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/lensing_potential")
if polarity == "I"
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/t_field_julia_predict_$(trial).npz", transpose(fJ[:Il]))
end
if polarity == "P"
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/e_field_julia_predict_$(trial).npz", transpose(fJ[:El]))
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/b_field_julia_predict_$(trial).npz", transpose(fJ[:Bl]))
end
if polarity == "IP"
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/t_field_julia_predict_$(trial).npz", transpose(fJ[:Il]))
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/e_field_julia_predict_$(trial).npz", transpose(fJ[:El]))
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/b_field_julia_predict_$(trial).npz", transpose(fJ[:Bl]))
end
npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/lensing_potential/phi_julia_predict_$(trial).npz", transpose(phiJ[:Il]))
