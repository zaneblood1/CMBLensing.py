import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _preamble import init_julia


def run(jl):
    jl.seval("""

        ###############################################
        #Compute the average Cls for temperature only
        ###############################################

        num_trials = 100
        uk_arcmin_t = 10

        #run a trial of batched julia simulations in I-only params
        sim_list = []
        for trial in 1:num_trials
            field_list = []
            local f, ϕ
            (;f, f̃, ϕ, ds) = load_sim(
                    Cℓ = Cℓ,
                    θpix = θpix,
                    T = Float64,
                    pol = :I,
                    Nside = Nside,
                    μKarcminT = uk_arcmin_t
                )
            push!(field_list, f)
            push!(field_list, ϕ)
            push!(field_list, ds.d)
            push!(field_list, f̃)
            push!(field_list, ds.d - ds.M * ds.B * f̃)
            push!(field_list, ds.M * ds.B * f̃)
            push!(sim_list, field_list)
        end

        #loop over the data and find the average Cl of phi, f, and data
        cls_format = get_Cℓ(sim_list[1][1])
        cls_length = length(cls_format.Cℓ)
        ell_I = cls_format.ℓ
        cl_pp_total = zeros(cls_length)
        cl_tt_total = zeros(cls_length)
        cl_dd_total = zeros(cls_length)
        cl_ll_total = zeros(cls_length)
        cl_nn_total = zeros(cls_length)
        cl_ss_total = zeros(cls_length)
        for trial in 1:num_trials
            cl_tt_total .+= get_Cℓ(sim_list[trial][1]).Cℓ
            cl_pp_total .+= get_Cℓ(sim_list[trial][2]).Cℓ
            cl_dd_total .+= get_Cℓ(sim_list[trial][3]).Cℓ
            cl_ll_total .+= get_Cℓ(sim_list[trial][4]).Cℓ
            cl_nn_total .+= get_Cℓ(sim_list[trial][5]).Cℓ
            cl_ss_total .+= get_Cℓ(sim_list[trial][6]).Cℓ
        end
        cl_tt_avg_I = cl_tt_total ./ num_trials
        cl_pp_avg_I = cl_pp_total ./ num_trials
        cl_dd_avg_I = cl_dd_total ./ num_trials
        cl_ll_avg_I = cl_ll_total ./ num_trials
        cl_nn_avg_I = cl_nn_total ./ num_trials
        cl_ss_avg_I = cl_ss_total ./ num_trials

        npzwrite(FILE_PATH * "ell_I.npz", ell_I)
        npzwrite(FILE_PATH * "cl_tt_avg_I.npz", cl_tt_avg_I)
        npzwrite(FILE_PATH * "cl_pp_avg_I.npz", cl_pp_avg_I)
        npzwrite(FILE_PATH * "cl_dd_avg_I.npz", cl_dd_avg_I)
        npzwrite(FILE_PATH * "cl_ll_avg_I.npz", cl_ll_avg_I)
        npzwrite(FILE_PATH * "cl_nn_avg_I.npz", cl_nn_avg_I)
        npzwrite(FILE_PATH * "cl_ss_avg_I.npz", cl_ss_avg_I)

        ###############################################
        #Compute the average Cls for polarization only
        ###############################################

        #run a trial of batched julia simulations in IEB params
        sim_list = []
        for trial in 1:num_trials
            field_list = []
            local f, ϕ
            (;f, f̃, ϕ, ds) = load_sim(
                    Cℓ = Cℓ,
                    θpix = θpix,
                    T = Float64,
                    pol = :P,
                    Nside = Nside,
                    μKarcminT = uk_arcmin_t
                )
            push!(field_list, f[:E])
            push!(field_list, f[:B])
            push!(field_list, ϕ)
            push!(field_list, ds.d[:E])
            push!(field_list, ds.d[:B])
            push!(field_list, f̃[:E])
            push!(field_list, f̃[:B])
            push!(field_list, (ds.d - ds.M * ds.B * f̃)[:E])
            push!(field_list, (ds.d - ds.M * ds.B * f̃)[:B])
            push!(field_list, (ds.M * ds.B * f̃)[:E])
            push!(field_list, (ds.M * ds.B * f̃)[:B])
            push!(sim_list, field_list)
        end

        #loop over the data and find the average Cl of phi, f, and data
        cls_format = get_Cℓ(sim_list[1][1])
        cls_length = length(cls_format.Cℓ)
        ell_P = cls_format.ℓ
        cl_pp_total = zeros(cls_length)
        cl_tt_e_total = zeros(cls_length)
        cl_tt_b_total = zeros(cls_length)
        cl_dd_e_total = zeros(cls_length)
        cl_dd_b_total = zeros(cls_length)
        cl_ll_e_total = zeros(cls_length)
        cl_ll_b_total = zeros(cls_length)
        cl_nn_e_total = zeros(cls_length)
        cl_nn_b_total = zeros(cls_length)
        cl_ss_e_total = zeros(cls_length)
        cl_ss_b_total = zeros(cls_length)
        for trial in 1:num_trials
            cl_tt_e_total .+= get_Cℓ(sim_list[trial][1]).Cℓ
            cl_tt_b_total .+= get_Cℓ(sim_list[trial][2]).Cℓ
            cl_pp_total .+= get_Cℓ(sim_list[trial][3]).Cℓ
            cl_dd_e_total .+= get_Cℓ(sim_list[trial][4]).Cℓ
            cl_dd_b_total .+= get_Cℓ(sim_list[trial][5]).Cℓ
            cl_ll_e_total .+= get_Cℓ(sim_list[trial][6]).Cℓ
            cl_ll_b_total .+= get_Cℓ(sim_list[trial][7]).Cℓ
            cl_nn_e_total .+= get_Cℓ(sim_list[trial][8]).Cℓ
            cl_nn_b_total .+= get_Cℓ(sim_list[trial][9]).Cℓ
            cl_ss_e_total .+= get_Cℓ(sim_list[trial][10]).Cℓ
            cl_ss_b_total .+= get_Cℓ(sim_list[trial][11]).Cℓ
        end

        cl_tt_e_avg_P = cl_tt_e_total ./ num_trials
        cl_tt_b_avg_P = cl_tt_b_total ./ num_trials
        cl_pp_avg_P = cl_pp_total ./ num_trials
        cl_dd_e_avg_P = cl_dd_e_total ./ num_trials
        cl_dd_b_avg_P = cl_dd_b_total ./ num_trials
        cl_ll_e_avg_P = cl_ll_e_total ./ num_trials
        cl_ll_b_avg_P = cl_ll_b_total ./ num_trials
        cl_nn_e_avg_P = cl_nn_e_total ./ num_trials
        cl_nn_b_avg_P = cl_nn_b_total ./ num_trials
        cl_ss_e_avg_P = cl_ss_e_total ./ num_trials
        cl_ss_b_avg_P = cl_ss_b_total ./ num_trials

        npzwrite(FILE_PATH * "ell_P.npz", ell_P)
        npzwrite(FILE_PATH * "cl_tt_e_avg_P.npz", cl_tt_e_avg_P)
        npzwrite(FILE_PATH * "cl_tt_b_avg_P.npz", cl_tt_b_avg_P)
        npzwrite(FILE_PATH * "cl_pp_avg_P.npz", cl_pp_avg_P)
        npzwrite(FILE_PATH * "cl_dd_e_avg_P.npz", cl_dd_e_avg_P)
        npzwrite(FILE_PATH * "cl_dd_b_avg_P.npz", cl_dd_b_avg_P)
        npzwrite(FILE_PATH * "cl_ll_e_avg_P.npz", cl_ll_e_avg_P)
        npzwrite(FILE_PATH * "cl_ll_b_avg_P.npz", cl_ll_b_avg_P)
        npzwrite(FILE_PATH * "cl_nn_e_avg_P.npz", cl_nn_e_avg_P)
        npzwrite(FILE_PATH * "cl_nn_b_avg_P.npz", cl_nn_b_avg_P)
        npzwrite(FILE_PATH * "cl_ss_e_avg_P.npz", cl_ss_e_avg_P)
        npzwrite(FILE_PATH * "cl_ss_b_avg_P.npz", cl_ss_b_avg_P)

        ###########################################################
        #Compute the average Cls for temperature and polarization
        ###########################################################

        #run a trial of batched julia simulations in IEB params
        sim_list = []
        for trial in 1:num_trials
            field_list = []
            local f, ϕ
            (;f, f̃, ϕ, ds) = load_sim(
                    Cℓ = Cℓ,
                    θpix = θpix,
                    T = Float64,
                    pol = :IP,
                    Nside = Nside,
                    μKarcminT = uk_arcmin_t
                )
            push!(field_list, f[:I])
            push!(field_list, f[:E])
            push!(field_list, f[:B])
            push!(field_list, ϕ)
            push!(field_list, ds.d[:I])
            push!(field_list, ds.d[:E])
            push!(field_list, ds.d[:B])
            push!(field_list, f̃[:I])
            push!(field_list, f̃[:E])
            push!(field_list, f̃[:B])
            push!(field_list, (ds.d - ds.M * ds.B * f̃)[:I])
            push!(field_list, (ds.d - ds.M * ds.B * f̃)[:E])
            push!(field_list, (ds.d - ds.M * ds.B * f̃)[:B])
            push!(field_list, (ds.M * ds.B * f̃)[:I])
            push!(field_list, (ds.M * ds.B * f̃)[:E])
            push!(field_list, (ds.M * ds.B * f̃)[:B])
            push!(sim_list, field_list)
        end

        #loop over the data and find the average Cl of phi, f, and data
        cls_format = get_Cℓ(sim_list[1][1])
        cls_length = length(cls_format.Cℓ)
        ell_IP = cls_format.ℓ
        cl_pp_total = zeros(cls_length)
        cl_tt_i_total = zeros(cls_length)
        cl_tt_e_total = zeros(cls_length)
        cl_tt_b_total = zeros(cls_length)
        cl_dd_i_total = zeros(cls_length)
        cl_dd_e_total = zeros(cls_length)
        cl_dd_b_total = zeros(cls_length)
        cl_ll_i_total = zeros(cls_length)
        cl_ll_e_total = zeros(cls_length)
        cl_ll_b_total = zeros(cls_length)
        cl_nn_i_total = zeros(cls_length)
        cl_nn_e_total = zeros(cls_length)
        cl_nn_b_total = zeros(cls_length)
        cl_ss_i_total = zeros(cls_length)
        cl_ss_e_total = zeros(cls_length)
        cl_ss_b_total = zeros(cls_length)
        for trial in 1:num_trials
            cl_tt_i_total .+= get_Cℓ(sim_list[trial][1]).Cℓ
            cl_tt_e_total .+= get_Cℓ(sim_list[trial][2]).Cℓ
            cl_tt_b_total .+= get_Cℓ(sim_list[trial][3]).Cℓ
            cl_pp_total .+= get_Cℓ(sim_list[trial][4]).Cℓ
            cl_dd_i_total .+= get_Cℓ(sim_list[trial][5]).Cℓ
            cl_dd_e_total .+= get_Cℓ(sim_list[trial][6]).Cℓ
            cl_dd_b_total .+= get_Cℓ(sim_list[trial][7]).Cℓ
            cl_ll_i_total .+= get_Cℓ(sim_list[trial][8]).Cℓ
            cl_ll_e_total .+= get_Cℓ(sim_list[trial][9]).Cℓ
            cl_ll_b_total .+= get_Cℓ(sim_list[trial][10]).Cℓ
            cl_nn_i_total .+= get_Cℓ(sim_list[trial][11]).Cℓ
            cl_nn_e_total .+= get_Cℓ(sim_list[trial][12]).Cℓ
            cl_nn_b_total .+= get_Cℓ(sim_list[trial][13]).Cℓ
            cl_ss_i_total .+= get_Cℓ(sim_list[trial][14]).Cℓ
            cl_ss_e_total .+= get_Cℓ(sim_list[trial][15]).Cℓ
            cl_ss_b_total .+= get_Cℓ(sim_list[trial][16]).Cℓ
        end
        cl_tt_i_avg_IP = cl_tt_i_total ./ num_trials
        cl_tt_e_avg_IP = cl_tt_e_total ./ num_trials
        cl_tt_b_avg_IP = cl_tt_b_total ./ num_trials
        cl_pp_avg_IP = cl_pp_total ./ num_trials
        cl_dd_i_avg_IP = cl_dd_i_total ./ num_trials
        cl_dd_e_avg_IP = cl_dd_e_total ./ num_trials
        cl_dd_b_avg_IP = cl_dd_b_total ./ num_trials
        cl_ll_i_avg_IP = cl_ll_i_total ./ num_trials
        cl_ll_e_avg_IP = cl_ll_e_total ./ num_trials
        cl_ll_b_avg_IP = cl_ll_b_total ./ num_trials
        cl_nn_i_avg_IP = cl_nn_i_total ./ num_trials
        cl_nn_e_avg_IP = cl_nn_e_total ./ num_trials
        cl_nn_b_avg_IP = cl_nn_b_total ./ num_trials
        cl_ss_i_avg_IP = cl_ss_i_total ./ num_trials
        cl_ss_e_avg_IP = cl_ss_e_total ./ num_trials
        cl_ss_b_avg_IP = cl_ss_b_total ./ num_trials

        npzwrite(FILE_PATH * "ell_IP.npz", ell_IP)
        npzwrite(FILE_PATH * "cl_tt_i_avg_IP.npz", cl_tt_i_avg_IP)
        npzwrite(FILE_PATH * "cl_tt_e_avg_IP.npz", cl_tt_e_avg_IP)
        npzwrite(FILE_PATH * "cl_tt_b_avg_IP.npz", cl_tt_b_avg_IP)
        npzwrite(FILE_PATH * "cl_pp_avg_IP.npz", cl_pp_avg_IP)
        npzwrite(FILE_PATH * "cl_dd_i_avg_IP.npz", cl_dd_i_avg_IP)
        npzwrite(FILE_PATH * "cl_dd_e_avg_IP.npz", cl_dd_e_avg_IP)
        npzwrite(FILE_PATH * "cl_dd_b_avg_IP.npz", cl_dd_b_avg_IP)
        npzwrite(FILE_PATH * "cl_ll_i_avg_IP.npz", cl_ll_i_avg_IP)
        npzwrite(FILE_PATH * "cl_ll_e_avg_IP.npz", cl_ll_e_avg_IP)
        npzwrite(FILE_PATH * "cl_ll_b_avg_IP.npz", cl_ll_b_avg_IP)
        npzwrite(FILE_PATH * "cl_nn_i_avg_IP.npz", cl_nn_i_avg_IP)
        npzwrite(FILE_PATH * "cl_nn_e_avg_IP.npz", cl_nn_e_avg_IP)
        npzwrite(FILE_PATH * "cl_nn_b_avg_IP.npz", cl_nn_b_avg_IP)
        npzwrite(FILE_PATH * "cl_ss_i_avg_IP.npz", cl_ss_i_avg_IP)
        npzwrite(FILE_PATH * "cl_ss_e_avg_IP.npz", cl_ss_e_avg_IP)
        npzwrite(FILE_PATH * "cl_ss_b_avg_IP.npz", cl_ss_b_avg_IP)
    """)

    print("Done generating simulated Cls data!")


if __name__ == "__main__":
    jl = init_julia()
    run(jl)
