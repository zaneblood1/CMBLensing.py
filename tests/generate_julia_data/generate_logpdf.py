import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _preamble import init_julia


def run(jl):
    jl.seval("""
        #the LogPDF function value
        logpdf_I = logpdf(temp_only_sim.ds; temp_only_sim.f, temp_only_sim.ϕ)
        open(FILE_PATH * "logpdf_I.txt", "w") do io
            println(io, logpdf_I)
        end
    """)

    jl.seval("""
        #The LogPDF value
        logpdf_P = logpdf(polar_only_sim.ds; polar_only_sim.f, polar_only_sim.ϕ)
        open(FILE_PATH * "logpdf_P.txt", "w") do io
            println(io, logpdf_P)
        end
    """)

    jl.seval("""
        #The LogPDF value
        logpdf_IP = logpdf(polar_and_temp_sim.ds; polar_and_temp_sim.f, polar_and_temp_sim.ϕ)
        open(FILE_PATH * "logpdf_IP.txt", "w") do io
            println(io, logpdf_IP)
        end
    """)

    print("Done generating logpdf data!")


if __name__ == "__main__":
    jl = init_julia()
    run(jl)
