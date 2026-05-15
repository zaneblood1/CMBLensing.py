import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _preamble import init_julia


def run(jl):
    jl.seval("""

        #Learned temperature only fields
        fJ, ϕJ, history = MAP_joint(temp_only_sim.ds, nsteps=30, progress=true);

        fJ_I = transpose(fJ[:Il])
        npzwrite(FILE_PATH * "fJ_I.npz", fJ_I)
        ϕJ_I = transpose(ϕJ[:Il])
        npzwrite(FILE_PATH * "phiJ_I.npz", ϕJ_I)

    """)

    jl.seval("""
        #Learned fields
        fJ, ϕJ, history = MAP_joint(polar_only_sim.ds, nsteps=30, progress=true);

        fJ_e_P = transpose(fJ[:El])
        fJ_b_P = transpose(fJ[:Bl])
        npzwrite(FILE_PATH * "fJ_e_P.npz", fJ_e_P)
        npzwrite(FILE_PATH * "fJ_b_P.npz", fJ_b_P)
        ϕJ_P = transpose(ϕJ[:Il])
        npzwrite(FILE_PATH * "phiJ_P.npz", ϕJ_P)
    """)

    jl.seval("""

        #Learned fields
        fJ, ϕJ, history = MAP_joint(polar_and_temp_sim.ds, nsteps=30, progress=true);

        fJ_t_IP = transpose(fJ[:Il])
        fJ_e_IP = transpose(fJ[:El])
        fJ_b_IP = transpose(fJ[:Bl])
        npzwrite(FILE_PATH * "fJ_t_IP.npz", fJ_t_IP)
        npzwrite(FILE_PATH * "fJ_e_IP.npz", fJ_e_IP)
        npzwrite(FILE_PATH * "fJ_b_IP.npz", fJ_b_IP)
        ϕJ_IP = transpose(ϕJ[:Il])
        npzwrite(FILE_PATH * "phiJ_IP.npz", ϕJ_IP)
    """)

    print("Done generating MAP joint data!")


if __name__ == "__main__":
    jl = init_julia()
    run(jl)
