import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _preamble import init_julia


def run(jl):
    jl.seval("""

        #Gradients of the logpdf function
        gradf_I = transpose(CMBLensing.gradientf_logpdf(temp_only_sim.ds; temp_only_sim.f, temp_only_sim.ϕ)[:Il])
        npzwrite(FILE_PATH * "gradf_I.npz", gradf_I)

        grad_phi_I = transpose(gradient(ϕ -> logpdf(temp_only_sim.ds; temp_only_sim.f, ϕ), temp_only_sim.ϕ)[1][:Il]);
        npzwrite(FILE_PATH * "grad_phi_I.npz", grad_phi_I)

        f°, ϕ° = mix(temp_only_sim.ds; temp_only_sim.f, temp_only_sim.ϕ);
        mixed_grad_phi_I = transpose(gradient(ϕ° -> logpdf(Mixed(temp_only_sim.ds); f°, ϕ°), ϕ°)[1][:Il]);
        npzwrite(FILE_PATH * "mixed_grad_phi_I.npz", mixed_grad_phi_I)

    """)

    jl.seval("""

        #Gradients of logpdf
        gradf_e_P = transpose(CMBLensing.gradientf_logpdf(polar_only_sim.ds; polar_only_sim.f, polar_only_sim.ϕ)[:El])
        gradf_b_P = transpose(CMBLensing.gradientf_logpdf(polar_only_sim.ds; polar_only_sim.f, polar_only_sim.ϕ)[:Bl])
        npzwrite(FILE_PATH * "gradf_e_P.npz", gradf_e_P)
        npzwrite(FILE_PATH * "gradf_b_P.npz", gradf_b_P)

        grad_phi_t_P = transpose(gradient(ϕ -> logpdf(polar_only_sim.ds; polar_only_sim.f, ϕ), polar_only_sim.ϕ)[1][:Il]);
        npzwrite(FILE_PATH * "grad_phi_t_P.npz", grad_phi_t_P)

        f°, ϕ° = mix(polar_only_sim.ds; polar_only_sim.f, polar_only_sim.ϕ);
        mixed_grad_phi_P = transpose(gradient(ϕ° -> logpdf(Mixed(polar_only_sim.ds); f°, ϕ°), ϕ°)[1][:Il]);
        npzwrite(FILE_PATH * "mixed_grad_phi_P.npz", mixed_grad_phi_P)

    """)

    jl.seval("""

        #Gradients of logpdf
        gradf_t_IP = transpose(CMBLensing.gradientf_logpdf(polar_and_temp_sim.ds; polar_and_temp_sim.f, polar_and_temp_sim.ϕ)[:Il])
        gradf_e_IP = transpose(CMBLensing.gradientf_logpdf(polar_and_temp_sim.ds; polar_and_temp_sim.f, polar_and_temp_sim.ϕ)[:El])
        gradf_b_IP = transpose(CMBLensing.gradientf_logpdf(polar_and_temp_sim.ds; polar_and_temp_sim.f, polar_and_temp_sim.ϕ)[:Bl])
        npzwrite(FILE_PATH * "gradf_t_IP.npz", gradf_t_IP)
        npzwrite(FILE_PATH * "gradf_e_IP.npz", gradf_e_IP)
        npzwrite(FILE_PATH * "gradf_b_IP.npz", gradf_b_IP)

        grad_phi_t_IP = transpose(gradient(ϕ -> logpdf(polar_and_temp_sim.ds; polar_and_temp_sim.f, ϕ), polar_and_temp_sim.ϕ)[1][:Il]);
        npzwrite(FILE_PATH * "grad_phi_t_IP.npz", grad_phi_t_IP)

        f°, ϕ° = mix(polar_and_temp_sim.ds; polar_and_temp_sim.f, polar_and_temp_sim.ϕ);
        mixed_grad_phi_IP = transpose(gradient(ϕ° -> logpdf(Mixed(polar_and_temp_sim.ds); f°, ϕ°), ϕ°)[1][:Il]);
        npzwrite(FILE_PATH * "mixed_grad_phi_IP.npz", mixed_grad_phi_IP)

    """)

    print("Done generating gradients data!")


if __name__ == "__main__":
    jl = init_julia()
    run(jl)
