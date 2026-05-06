import os
os.environ["PYTHON_JULIAPKG_PROJECT"] = "/home/zane-blood/CMBLensing.jl"
os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"
from juliacall import Main as jl

jl.seval("""
        
#Get the current working directory to write to disk
DIR = pwd()
FILE_PATH = DIR * "/tests/ground_truth_data/"
         
#Load the necessary packages
using CMBLensing
using PythonPlot
using NPZ
         
seed = 67
θpix = 2
Nside = 256
Cℓ = camb(ℓmax = 17000);
         
#Write the theta_pix and Nside values for all of these simulations to disk
open(FILE_PATH * "theta_pix.txt", "w") do io
    println(io, θpix)
end
open(FILE_PATH * "n_side.txt", "w") do io
    println(io, Nside)
end

#################################################################################################################        
#Generate temperature only data set
#################################################################################################################
         
temp_only_sim = load_sim(
    seed = seed,
    θpix = θpix,
    T = Float64,
    Nside = Nside,
    pol = :I,
    Cℓ = Cℓ
);

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

#Covariance matrices    
cn_I = transpose(temp_only_sim.ds.Cn.diag.arr)
cf_I = transpose(temp_only_sim.ds.Cf.op.diag.arr)
cphi_I = transpose(temp_only_sim.ds.Cϕ.op.diag.arr)
nphi_I = transpose(temp_only_sim.ds.Nϕ.diag.arr)
m_I = transpose(temp_only_sim.ds.M.diag.arr)
b_I = transpose(temp_only_sim.ds.B.diag.arr)
d_I = transpose(temp_only_sim.ds.D.op.diag.arr)
g_I = transpose(temp_only_sim.ds.G.op.diag.arr)    
npzwrite(FILE_PATH * "cn_I.npz", cn_I)
npzwrite(FILE_PATH * "cf_I.npz", cf_I)
npzwrite(FILE_PATH * "cphi_I.npz", cphi_I)
npzwrite(FILE_PATH * "nphi_I.npz", nphi_I)
npzwrite(FILE_PATH * "m_I.npz", m_I)
npzwrite(FILE_PATH * "b_I.npz", b_I)
npzwrite(FILE_PATH * "d_I.npz", d_I)
npzwrite(FILE_PATH * "g_I.npz", g_I)

#Learned temperature only fields
fJ, ϕJ, history = MAP_joint(temp_only_sim.ds, nsteps=30, progress=true);
fJ_I = transpose(fJ[:Il])
npzwrite(FILE_PATH * "fJ_I.npz", fJ_I)
ϕJ_I = transpose(ϕJ[:Il])
npzwrite(FILE_PATH * "phiJ_I.npz", ϕJ_I)

#Gradients of the logpdf function
gradf_I = transpose(CMBLensing.gradientf_logpdf(temp_only_sim.ds; temp_only_sim.f, temp_only_sim.ϕ)[:Il])
npzwrite(FILE_PATH * "gradf_I.npz", gradf_I)
grad_phi_I = transpose(gradient(ϕ -> logpdf(temp_only_sim.ds; temp_only_sim.f, ϕ), temp_only_sim.ϕ)[1][:Il]);
npzwrite(FILE_PATH * "grad_phi_I.npz", grad_phi_I)
f°, ϕ° = mix(temp_only_sim.ds; temp_only_sim.f, temp_only_sim.ϕ);
mixed_grad_phi_I = transpose(gradient(ϕ° -> logpdf(Mixed(temp_only_sim.ds); f°, ϕ°), ϕ°)[1][:Il]);
npzwrite(FILE_PATH * "mixed_grad_phi_I.npz", mixed_grad_phi_I)

#the LogPDF function value
logpdf_I = logpdf(temp_only_sim.ds; temp_only_sim.f, temp_only_sim.ϕ)
open(FILE_PATH * "logpdf_I.txt", "w") do io
    println(io, logpdf_I)
end

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

#################################################################################################################
#Generate polarization only data set  
#################################################################################################################
    
polar_only_sim = load_sim(
    seed = seed,
    θpix = θpix,
    T = Float64,
    Nside = Nside,
    pol = :P,
    Cℓ = Cℓ
);
         
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

#Covariance matrices
cn_ee_P = transpose(polar_only_sim.ds.Cn[:E].diag.arr)
cn_bb_P = transpose(polar_only_sim.ds.Cn[:B].diag.arr)
cf_ee_P = transpose(polar_only_sim.ds.Cf[:E].diag.arr)
cf_bb_P = transpose(polar_only_sim.ds.Cf[:B].diag.arr)
cphi_P = transpose(polar_only_sim.ds.Cϕ.op.diag.arr)
nphi_P = transpose(polar_only_sim.ds.Nϕ.diag.arr)
m_ee_P = transpose(polar_only_sim.ds.M[:E].diag.arr)
m_bb_P = transpose(polar_only_sim.ds.M[:B].diag.arr)
b_ee_P = transpose(polar_only_sim.ds.B[:E].diag.arr)
b_bb_P = transpose(polar_only_sim.ds.B[:B].diag.arr)
d_ee_P = transpose(polar_only_sim.ds.D[:E].diag.arr)
d_bb_P = transpose(polar_only_sim.ds.D[:B].diag.arr)
npzwrite(FILE_PATH * "cn_ee_P.npz", cn_ee_P)
npzwrite(FILE_PATH * "cn_bb_P.npz", cn_bb_P)
npzwrite(FILE_PATH * "cf_ee_P.npz", cf_ee_P)
npzwrite(FILE_PATH * "cf_bb_P.npz", cf_bb_P)
npzwrite(FILE_PATH * "cphi_P.npz", cphi_P)
npzwrite(FILE_PATH * "nphi_P.npz", nphi_P)
npzwrite(FILE_PATH * "m_ee_P.npz", m_ee_P)
npzwrite(FILE_PATH * "m_bb_P.npz", m_bb_P)
npzwrite(FILE_PATH * "b_ee_P.npz", b_ee_P)
npzwrite(FILE_PATH * "b_bb_P.npz", b_bb_P)
npzwrite(FILE_PATH * "d_ee_P.npz", d_ee_P)
npzwrite(FILE_PATH * "d_bb_P.npz", d_bb_P)

#The LogPDF value     
logpdf_P = logpdf(polar_only_sim.ds; polar_only_sim.f, polar_only_sim.ϕ)
open(FILE_PATH * "logpdf_P.txt", "w") do io
    println(io, logpdf_P)
end

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

#Learned fields
fJ, ϕJ, history = MAP_joint(polar_only_sim.ds, nsteps=30, progress=true);
fJ_e_P = transpose(fJ[:El])
fJ_b_P = transpose(fJ[:Bl])
npzwrite(FILE_PATH * "fJ_e_P.npz", fJ_e_P)
npzwrite(FILE_PATH * "fJ_b_P.npz", fJ_b_P)
ϕJ_P = transpose(ϕJ[:Il])
npzwrite(FILE_PATH * "phiJ_P.npz", ϕJ_P)
      
#################################################################################################################
#Generate a temperature and polarization combination data set  
#################################################################################################################
      
polar_and_temp_sim = load_sim(
    seed = seed,
    θpix = θpix,
    T = Float64,
    Nside = Nside,
    pol = :IP,
    Cℓ = Cℓ
);
         
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
         
#Covariance matrices
cn_ee_IP = transpose(polar_and_temp_sim.ds.Cn.ΣTE[4].diag.arr)
cn_bb_IP = transpose(polar_and_temp_sim.ds.Cn.ΣB.diag.arr)
cn_te_IP = transpose(polar_and_temp_sim.ds.Cn.ΣTE[2].diag.arr)
cn_tt_IP = transpose(polar_and_temp_sim.ds.Cn.ΣTE[1].diag.arr)
cf_ee_IP = transpose(polar_and_temp_sim.ds.Cf.op.ΣTE[4].diag.arr)
cf_bb_IP = transpose(polar_and_temp_sim.ds.Cf.op.ΣB.diag.arr)
cf_te_IP = transpose(polar_and_temp_sim.ds.Cf.op.ΣTE[2].diag.arr)
cf_tt_IP = transpose(polar_and_temp_sim.ds.Cf.op.ΣTE[1].diag.arr)
cphi_IP = transpose(polar_and_temp_sim.ds.Cϕ.op.diag.arr)
nphi_IP = transpose(polar_and_temp_sim.ds.Nϕ.diag.arr)
b_ee_IP = transpose(polar_and_temp_sim.ds.B.ΣTE[4].diag.arr)
b_bb_IP = transpose(polar_and_temp_sim.ds.B.ΣB.diag.arr)
b_te_IP = transpose(polar_and_temp_sim.ds.B.ΣTE[2].diag.arr)
b_tt_IP = transpose(polar_and_temp_sim.ds.B.ΣTE[1].diag.arr)
m_ee_IP = transpose(polar_and_temp_sim.ds.M.ΣTE[4].diag.arr)
m_bb_IP = transpose(polar_and_temp_sim.ds.M.ΣB.diag.arr)
m_te_IP = transpose(polar_and_temp_sim.ds.M.ΣTE[2].diag.arr)
m_tt_IP = transpose(polar_and_temp_sim.ds.M.ΣTE[1].diag.arr)
d_ee_IP = transpose(polar_and_temp_sim.ds.D.op.ΣTE[4].diag.arr)
d_bb_IP = transpose(polar_and_temp_sim.ds.D.op.ΣB.diag.arr)
d_te_IP = transpose(polar_and_temp_sim.ds.D.op.ΣTE[2].diag.arr)
d_tt_IP = transpose(polar_and_temp_sim.ds.D.op.ΣTE[1].diag.arr)
npzwrite(FILE_PATH * "d_ee_IP.npz", d_ee_IP)
npzwrite(FILE_PATH * "d_bb_IP.npz", d_bb_IP)
npzwrite(FILE_PATH * "d_te_IP.npz", d_te_IP)
npzwrite(FILE_PATH * "d_tt_IP.npz", d_tt_IP)
npzwrite(FILE_PATH * "cn_ee_IP.npz", cn_ee_IP)
npzwrite(FILE_PATH * "cn_bb_IP.npz", cn_bb_IP)
npzwrite(FILE_PATH * "cn_te_IP.npz", cn_te_IP)
npzwrite(FILE_PATH * "cn_tt_IP.npz", cn_tt_IP)
npzwrite(FILE_PATH * "cf_ee_IP.npz", cf_ee_IP)
npzwrite(FILE_PATH * "cf_bb_IP.npz", cf_bb_IP)
npzwrite(FILE_PATH * "cf_te_IP.npz", cf_te_IP)
npzwrite(FILE_PATH * "cf_tt_IP.npz", cf_tt_IP)
npzwrite(FILE_PATH * "cphi_IP.npz", cphi_IP)
npzwrite(FILE_PATH * "nphi_IP.npz", nphi_IP)
npzwrite(FILE_PATH * "m_ee_IP.npz", m_ee_IP)
npzwrite(FILE_PATH * "m_bb_IP.npz", m_bb_IP)
npzwrite(FILE_PATH * "m_te_IP.npz", m_te_IP)
npzwrite(FILE_PATH * "m_tt_IP.npz", m_tt_IP)
npzwrite(FILE_PATH * "b_ee_IP.npz", b_ee_IP)
npzwrite(FILE_PATH * "b_bb_IP.npz", b_bb_IP)
npzwrite(FILE_PATH * "b_te_IP.npz", b_te_IP)
npzwrite(FILE_PATH * "b_tt_IP.npz", b_tt_IP)
         
#The LogPDF value
logpdf_IP = logpdf(polar_and_temp_sim.ds; polar_and_temp_sim.f, polar_and_temp_sim.ϕ)
open(FILE_PATH * "logpdf_IP.txt", "w") do io
    println(io, logpdf_IP)
end

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


