import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _preamble import init_julia


def run(jl):
    jl.seval("""
        #Write the Julia generated temperature only data to disk

        #Fields
        phi_I = transpose((temp_only_sim.ϕ)[:Il])
        t_field_I = transpose((temp_only_sim.f)[:Il])
        t_lensed_field_I = transpose((temp_only_sim.ds.L(temp_only_sim.ϕ) * temp_only_sim.f)[:Il])
        t_adjoint_lensed_field_I = transpose((temp_only_sim.ds.L(temp_only_sim.ϕ)' * temp_only_sim.f)[:Il])
        data_t_field_I = transpose((temp_only_sim.ds.d)[:Il])
        npzwrite(FILE_PATH * "phi_I.npz", phi_I)
        npzwrite(FILE_PATH * "t_field_I.npz", t_field_I)
        npzwrite(FILE_PATH * "t_lensed_field_I.npz", t_lensed_field_I)
        npzwrite(FILE_PATH * "t_adjoint_lensed_field_I.npz", t_adjoint_lensed_field_I)
        npzwrite(FILE_PATH * "data_t_field_I.npz", data_t_field_I)

    """)

    jl.seval("""
        #Write polarization only julia generated data to disk

        #Fields
        phi_P = transpose((polar_only_sim.ϕ)[:Il])
        e_field_P = transpose((polar_only_sim.f)[:El])
        b_field_P = transpose((polar_only_sim.f)[:Bl])
        data_e_field_P = transpose((polar_only_sim.ds.d)[:El])
        data_b_field_P = transpose((polar_only_sim.ds.d)[:Bl])
        lensed_fields = polar_only_sim.ds.L(polar_only_sim.ϕ) * polar_only_sim.f
        e_lensed_field_P = transpose(lensed_fields[:El])
        b_lensed_field_P = transpose(lensed_fields[:Bl])
        adjoint_lensed_fields = polar_only_sim.ds.L(polar_only_sim.ϕ)' * polar_only_sim.f
        e_adjoint_lensed_field_P = transpose(adjoint_lensed_fields[:El])
        b_adjoint_lensed_field_P = transpose(adjoint_lensed_fields[:Bl])
        npzwrite(FILE_PATH * "phi_P.npz", phi_P)
        npzwrite(FILE_PATH * "e_field_P.npz", e_field_P)
        npzwrite(FILE_PATH * "e_lensed_field_P.npz", e_lensed_field_P)
        npzwrite(FILE_PATH * "e_adjoint_lensed_field_P.npz", e_adjoint_lensed_field_P)
        npzwrite(FILE_PATH * "b_field_P.npz", b_field_P)
        npzwrite(FILE_PATH * "b_lensed_field_P.npz", b_lensed_field_P)
        npzwrite(FILE_PATH * "b_adjoint_lensed_field_P.npz", b_adjoint_lensed_field_P)
        npzwrite(FILE_PATH * "data_e_field_P.npz", data_e_field_P)
        npzwrite(FILE_PATH * "data_b_field_P.npz", data_b_field_P)

    """)

    jl.seval("""
        #Write the temperature and polarization combo data to disk

        #Fields
        phi_IP = transpose((polar_and_temp_sim.ϕ)[:Il])
        t_field_IP = transpose((polar_and_temp_sim.f)[:Il])
        e_field_IP = transpose((polar_and_temp_sim.f)[:El])
        b_field_IP = transpose((polar_and_temp_sim.f)[:Bl])
        lensed_fields = polar_and_temp_sim.ds.L(polar_and_temp_sim.ϕ) * polar_and_temp_sim.f
        t_lensed_field_IP = transpose(lensed_fields[:Il])
        e_lensed_field_IP = transpose(lensed_fields[:El])
        b_lensed_field_IP = transpose(lensed_fields[:Bl])
        adjoint_lensed_fields = polar_and_temp_sim.ds.L(polar_and_temp_sim.ϕ)' * polar_and_temp_sim.f
        t_adjoint_lensed_field_IP = transpose(adjoint_lensed_fields[:Il])
        e_adjoint_lensed_field_IP = transpose(adjoint_lensed_fields[:El])
        b_adjoint_lensed_field_IP = transpose(adjoint_lensed_fields[:Bl])
        data_t_field_IP = transpose(polar_and_temp_sim.ds.d[:Il])
        data_e_field_IP = transpose(polar_and_temp_sim.ds.d[:El])
        data_b_field_IP = transpose(polar_and_temp_sim.ds.d[:Bl])
        npzwrite(FILE_PATH * "phi_IP.npz", phi_IP)
        npzwrite(FILE_PATH * "t_field_IP.npz", t_field_IP)
        npzwrite(FILE_PATH * "t_lensed_field_IP.npz", t_lensed_field_IP)
        npzwrite(FILE_PATH * "t_adjoint_lensed_field_IP.npz", t_adjoint_lensed_field_IP)
        npzwrite(FILE_PATH * "e_field_IP.npz", e_field_IP)
        npzwrite(FILE_PATH * "e_lensed_field_IP.npz", e_lensed_field_IP)
        npzwrite(FILE_PATH * "e_adjoint_lensed_field_IP.npz", e_adjoint_lensed_field_IP)
        npzwrite(FILE_PATH * "b_field_IP.npz", b_field_IP)
        npzwrite(FILE_PATH * "b_lensed_field_IP.npz", b_lensed_field_IP)
        npzwrite(FILE_PATH * "b_adjoint_lensed_field_IP.npz", b_adjoint_lensed_field_IP)
        npzwrite(FILE_PATH * "data_t_field_IP.npz", data_t_field_IP)
        npzwrite(FILE_PATH * "data_e_field_IP.npz", data_e_field_IP)
        npzwrite(FILE_PATH * "data_b_field_IP.npz", data_b_field_IP)

    """)

    print("Done generating lensing data!")


if __name__ == "__main__":
    jl = init_julia()
    run(jl)
