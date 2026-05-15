import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _preamble import init_julia


def run(jl):
    jl.seval("""

        #Wiener filtered maps
        zero_f = zero(diag(temp_only_sim.ds.Cf))
        temporary_phi = temp_only_sim.ϕ
        ϕ = temp_only_sim.ϕ
        f = temp_only_sim.f
        f_wf_ffpp_I, = argmaxf_logpdf(temp_only_sim.ds, (;ϕ); fstart = f, conjgrad_kwargs=(tol=1e-1,progress=false));
        f_wf_f0pp_I, = argmaxf_logpdf(temp_only_sim.ds, (;ϕ); fstart = zero_f, conjgrad_kwargs=(tol=1e-1,progress=false));
        ϕ = 0 .* ϕ
        f_wf_f0p0_I, = argmaxf_logpdf(temp_only_sim.ds, (;ϕ); fstart = zero_f, conjgrad_kwargs=(tol=1e-1,progress=false));
        f_wf_ffp0_I, = argmaxf_logpdf(temp_only_sim.ds, (;ϕ); fstart = f, conjgrad_kwargs=(tol=1e-1,progress=false));
        ϕ = temporary_phi

        f_wf_ffpp_I = transpose(f_wf_ffpp_I[:Il])
        npzwrite(FILE_PATH * "f_wf_ffpp_I.npz", f_wf_ffpp_I)
        f_wf_f0pp_I = transpose(f_wf_f0pp_I[:Il])
        npzwrite(FILE_PATH * "f_wf_f0pp_I.npz", f_wf_f0pp_I)
        f_wf_f0p0_I = transpose(f_wf_f0p0_I[:Il])
        npzwrite(FILE_PATH * "f_wf_f0p0_I.npz", f_wf_f0p0_I)
        f_wf_ffp0_I = transpose(f_wf_ffp0_I[:Il])
        npzwrite(FILE_PATH * "f_wf_ffp0_I.npz", f_wf_ffp0_I)

    """)

    jl.seval("""

        #Wiener filtered maps
        ϕ = polar_only_sim.ϕ
        f_wf_P, = argmaxf_logpdf(polar_only_sim.ds, (;ϕ); conjgrad_kwargs=(tol=1e-1,progress=false));
        f_wf_e_P = transpose(f_wf_P[:El])
        f_wf_b_P = transpose(f_wf_P[:Bl])
        npzwrite(FILE_PATH * "f_wf_e_P.npz", f_wf_e_P)
        npzwrite(FILE_PATH * "f_wf_b_P.npz", f_wf_b_P)
        zero_f = zero(diag(polar_only_sim.ds.Cf))
        temporary_phi = polar_only_sim.ϕ
        ϕ = polar_only_sim.ϕ
        f = polar_only_sim.f
        f_wf_ffpp_P, = argmaxf_logpdf(polar_only_sim.ds, (;ϕ); fstart = f, conjgrad_kwargs=(tol=1e-1,progress=false));
        f_wf_f0pp_P, = argmaxf_logpdf(polar_only_sim.ds, (;ϕ); fstart = zero_f, conjgrad_kwargs=(tol=1e-1,progress=false));
        ϕ = 0 .* ϕ
        f_wf_f0p0_P, = argmaxf_logpdf(polar_only_sim.ds, (;ϕ); fstart = zero_f, conjgrad_kwargs=(tol=1e-1,progress=false));
        f_wf_ffp0_P, = argmaxf_logpdf(polar_only_sim.ds, (;ϕ); fstart = f, conjgrad_kwargs=(tol=1e-1,progress=false));
        ϕ = temporary_phi

        f_wf_ffpp_e_P = transpose(f_wf_ffpp_P[:El])
        f_wf_ffpp_b_P = transpose(f_wf_ffpp_P[:Bl])
        npzwrite(FILE_PATH * "f_wf_ffpp_e_P.npz", f_wf_ffpp_e_P)
        npzwrite(FILE_PATH * "f_wf_ffpp_b_P.npz", f_wf_ffpp_b_P)
        f_wf_ffp0_e_P = transpose(f_wf_ffp0_P[:El])
        f_wf_ffp0_b_P = transpose(f_wf_ffp0_P[:Bl])
        npzwrite(FILE_PATH * "f_wf_ffp0_e_P.npz", f_wf_ffp0_e_P)
        npzwrite(FILE_PATH * "f_wf_ffp0_b_P.npz", f_wf_ffp0_b_P)
        f_wf_f0pp_e_P = transpose(f_wf_f0pp_P[:El])
        f_wf_f0pp_b_P = transpose(f_wf_f0pp_P[:Bl])
        npzwrite(FILE_PATH * "f_wf_f0pp_e_P.npz", f_wf_f0pp_e_P)
        npzwrite(FILE_PATH * "f_wf_f0pp_b_P.npz", f_wf_f0pp_b_P)
        f_wf_f0p0_e_P = transpose(f_wf_f0p0_P[:El])
        f_wf_f0p0_b_P = transpose(f_wf_f0p0_P[:Bl])
        npzwrite(FILE_PATH * "f_wf_f0p0_e_P.npz", f_wf_f0p0_e_P)
        npzwrite(FILE_PATH * "f_wf_f0p0_b_P.npz", f_wf_f0p0_b_P)
             
    """)

    jl.seval("""
        #Wiener filtered maps
        ϕ = polar_and_temp_sim.ϕ
        f_wf_IP, = argmaxf_logpdf(polar_and_temp_sim.ds, (;ϕ); conjgrad_kwargs=(tol=1e-1,progress=false));
        f_wf_t_IP = transpose(f_wf_IP[:Il])
        f_wf_e_IP = transpose(f_wf_IP[:El])
        f_wf_b_IP = transpose(f_wf_IP[:Bl])
        npzwrite(FILE_PATH * "f_wf_t_IP.npz", f_wf_t_IP)
        npzwrite(FILE_PATH * "f_wf_e_IP.npz", f_wf_e_IP)
        npzwrite(FILE_PATH * "f_wf_b_IP.npz", f_wf_b_IP)
        zero_f = zero(diag(polar_and_temp_sim.ds.Cf))
        temporary_phi = polar_and_temp_sim.ϕ
        ϕ = polar_and_temp_sim.ϕ
        f = polar_and_temp_sim.f
        f_wf_ffpp_IP, = argmaxf_logpdf(polar_and_temp_sim.ds, (;ϕ); fstart = f, conjgrad_kwargs=(tol=1e-1,progress=false));
        f_wf_f0pp_IP, = argmaxf_logpdf(polar_and_temp_sim.ds, (;ϕ); fstart = zero_f, conjgrad_kwargs=(tol=1e-1,progress=false));
        ϕ = 0 .* ϕ
        f_wf_f0p0_IP, = argmaxf_logpdf(polar_and_temp_sim.ds, (;ϕ); fstart = zero_f, conjgrad_kwargs=(tol=1e-1,progress=false));
        f_wf_ffp0_IP, = argmaxf_logpdf(polar_and_temp_sim.ds, (;ϕ); fstart = f, conjgrad_kwargs=(tol=1e-1,progress=false));
        ϕ = temporary_phi

        f_wf_ffpp_t_IP = transpose(f_wf_ffpp_IP[:Il])
        f_wf_ffpp_e_IP = transpose(f_wf_ffpp_IP[:El])
        f_wf_ffpp_b_IP = transpose(f_wf_ffpp_IP[:Bl])
        npzwrite(FILE_PATH * "f_wf_ffpp_t_IP.npz", f_wf_ffpp_t_IP)
        npzwrite(FILE_PATH * "f_wf_ffpp_e_IP.npz", f_wf_ffpp_e_IP)
        npzwrite(FILE_PATH * "f_wf_ffpp_b_IP.npz", f_wf_ffpp_b_IP)
        f_wf_ffp0_t_IP = transpose(f_wf_ffp0_IP[:Il])
        f_wf_ffp0_e_IP = transpose(f_wf_ffp0_IP[:El])
        f_wf_ffp0_b_IP = transpose(f_wf_ffp0_IP[:Bl])
        npzwrite(FILE_PATH * "f_wf_ffp0_t_IP.npz", f_wf_ffp0_t_IP)
        npzwrite(FILE_PATH * "f_wf_ffp0_e_IP.npz", f_wf_ffp0_e_IP)
        npzwrite(FILE_PATH * "f_wf_ffp0_b_IP.npz", f_wf_ffp0_b_IP)
        f_wf_f0pp_t_IP = transpose(f_wf_f0pp_IP[:Il])
        f_wf_f0pp_e_IP = transpose(f_wf_f0pp_IP[:El])
        f_wf_f0pp_b_IP = transpose(f_wf_f0pp_IP[:Bl])
        npzwrite(FILE_PATH * "f_wf_f0pp_t_IP.npz", f_wf_f0pp_t_IP)
        npzwrite(FILE_PATH * "f_wf_f0pp_e_IP.npz", f_wf_f0pp_e_IP)
        npzwrite(FILE_PATH * "f_wf_f0pp_b_IP.npz", f_wf_f0pp_b_IP)
        f_wf_f0p0_t_IP = transpose(f_wf_f0p0_IP[:Il])
        f_wf_f0p0_e_IP = transpose(f_wf_f0p0_IP[:El])
        f_wf_f0p0_b_IP = transpose(f_wf_f0p0_IP[:Bl])
        npzwrite(FILE_PATH * "f_wf_f0p0_t_IP.npz", f_wf_f0p0_t_IP)
        npzwrite(FILE_PATH * "f_wf_f0p0_e_IP.npz", f_wf_f0p0_e_IP)
        npzwrite(FILE_PATH * "f_wf_f0p0_b_IP.npz", f_wf_f0p0_b_IP)
    """)

    print("Done generating wiener filter data!")


if __name__ == "__main__":
    jl = init_julia()
    run(jl)
