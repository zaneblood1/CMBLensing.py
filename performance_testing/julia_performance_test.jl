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
    end
    return parse_args(settings)
end

args = parse_commandline()
trial = args["trial"]
map_size = args["map_size"]
seed = args["seed"]
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

#We will use the following settings for load_sim()
θpix  = 2.5	 # pixel size in arcmin
Nside = map_size # number of pixels per side in the map
T     = Float64  # data type (Float32 is ~2 as fast as Float64);

#run load_sim() using the given seed and settings
(;f, f̃, ϕ, ds) = load_sim(
    seed = seed,
    θpix = θpix,
    T = T,
    Nside = Nside,
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
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/fJ_t_$(trial).npz", transpose(fJ[:Il]))
end
if polarity == "P"
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/fJ_e_$(trial).npz", transpose(fJ[:El]))
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/fJ_b_$(trial).npz", transpose(fJ[:Bl]))
end
if polarity == "IP"
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/fJ_t_$(trial).npz", transpose(fJ[:Il]))
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/fJ_e_$(trial).npz", transpose(fJ[:El]))
    npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/cmb/fJ_b_$(trial).npz", transpose(fJ[:Bl]))
end
npzwrite("/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/julia_results/map_size_$(map_size)_polarity_$(polarity)_seed_$(seed)/learned_fields/lensing_potential/phiJ_$(trial).npz", transpose(phiJ[:Il]))
