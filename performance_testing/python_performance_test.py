import os
import shutil
os.environ["PYTHON_JULIAPKG_PROJECT"] = "/resnick/groups/wugroup/zblood/CMBLensing.jl"
os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"
#Pre-populate juliapkg's STATE so resolve() returns immediately at its fast-path
#check (STATE["resolved"] == True) without ever acquiring the NFS file lock that
#causes contention across hundreds of concurrent SLURM jobs
from juliapkg.state import STATE
STATE["resolved"] = True
STATE["executable"] = shutil.which("julia")
from cmb_lensing.map_joint import *
from juliacall import Main as jl
import time
import argparse

#Parse the arguments of the slurm job
parser = argparse.ArgumentParser()
parser.add_argument("--map_size", type = int)
parser.add_argument("--seed", type = int)
parser.add_argument("--trial", type = int)
parser.add_argument("--pol", type = str)
parser.add_argument("--theta_pix", type = float)
args = parser.parse_args()

#call julia load_sim via julia call to get the julia generated simulation data
#in order to better compare apples to apples
jl.seval(f"""   
using CMBLensing
data_set = load_sim(
    seed = {args.seed},
    θpix = {args.theta_pix},
    T = Float64,
    Nside = {args.map_size},
    pol = :{args.pol}
);
""")

#These terms are common to all I, P, and IP polarization keys
fourier_weights = get_fourier_weights((args.map_size, args.map_size//2+1))
pix_width = float(jnp.deg2rad(args.theta_pix / ARCMIN_PER_DEGREE))

#branching logic based on polarity key for how to construct python data set object
#given julia matrix data for either I, P, or IP
if args.pol == "I":

    #Load the temperature only covariance matrices
    jl.seval(f"""
    cn_I = transpose(data_set.ds.Cn.diag.arr)
    cf_I = transpose(data_set.ds.Cf.op.diag.arr)
    cphi_I = transpose(data_set.ds.Cϕ.op.diag.arr)
    nphi_I = transpose(data_set.ds.Nϕ.diag.arr)
    m_I = transpose(data_set.ds.M.diag.arr)
    b_I = transpose(data_set.ds.B.diag.arr)
    d_I = transpose(data_set.ds.D.op.diag.arr)
    """)

    #Load the temperature only fields
    jl.seval("""
    phi_I = transpose((data_set.ϕ)[:Il])
    t_field_I = transpose((data_set.f)[:Il])
    data_t_field_I = transpose((data_set.ds.d)[:Il])
    """)

    #Convert these julia matrices to python objects...
    #Example diagonal scalar
    phi_covariance = DiagonalScalar(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        scalar_matrix = jnp.array(jl.cphi_I)
    )

    #Example FlatS0 field
    phi = FlatS0(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        basis = Basis.FOURIER,
        parametrization = Parametrization.T,
        scalar_matrix = jnp.array(jl.phi_I)
    )

    #Put all the data together inside of a temperature only dataset object
    data_set = DataSetT(
        #covariance matrices
        noise_covariance = phi_covariance.replace(scalar_matrix = jnp.array(jl.cn_I)),
        mixing_d = phi_covariance.replace(scalar_matrix = jnp.array(jl.d_I)),
        field_covariance = phi_covariance.replace(scalar_matrix = jnp.array(jl.cf_I)),
        phi_covariance = phi_covariance,
        mask = phi_covariance.replace(scalar_matrix = jnp.array(jl.m_I)),
        beam = phi_covariance.replace(scalar_matrix = jnp.array(jl.b_I)),
        quadratic_estimate = phi_covariance.replace(scalar_matrix = jnp.array(jl.nphi_I)),
        #fields
        #NOTE we don't really care about the lensed field here so just use default
        #FlatS0() since we will never access it...
        data = phi.replace(scalar_matrix = jnp.array(jl.data_t_field_I)),
        unlensed_field = phi.replace(scalar_matrix = jnp.array(jl.t_field_I)),
        phi = phi
    )

elif args.pol == "P":
    #Load the polarization only covariance matrices
    jl.seval(f"""
    cn_ee_P = transpose(data_set.ds.Cn[:E].diag.arr)
    cn_bb_P = transpose(data_set.ds.Cn[:B].diag.arr)
    cf_ee_P = transpose(data_set.ds.Cf[:E].diag.arr)
    cf_bb_P = transpose(data_set.ds.Cf[:B].diag.arr)
    cphi_P = transpose(data_set.ds.Cϕ.op.diag.arr)
    nphi_P = transpose(data_set.ds.Nϕ.diag.arr)
    m_ee_P = transpose(data_set.ds.M[:E].diag.arr)
    m_bb_P = transpose(data_set.ds.M[:B].diag.arr)
    b_ee_P = transpose(data_set.ds.B[:E].diag.arr)
    b_bb_P = transpose(data_set.ds.B[:B].diag.arr)
    d_ee_P = transpose(data_set.ds.D[:E].diag.arr)
    d_bb_P = transpose(data_set.ds.D[:B].diag.arr)
    """)

    #Load the polarization only fields
    jl.seval("""
    phi_P = transpose((data_set.ϕ)[:Il])
    e_field_P = transpose((data_set.f)[:El])
    b_field_P = transpose((data_set.f)[:Bl])
    data_e_field_P = transpose((data_set.ds.d)[:El])
    data_b_field_P = transpose((data_set.ds.d)[:Bl])
    """)

    #Convert these julia matrices to python objects..
    #Example diagonal scalar matrix operator
    phi_covariance = DiagonalScalar(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        scalar_matrix = jnp.array(jl.cphi_P)
    )
    #Example block iagonal EB matrix operator
    noise_covariance = DiagonalEB(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        matrix_EE = jnp.array(jl.cn_ee_P),
        matrix_BB = jnp.array(jl.cn_bb_P)
    )

    #Example FlatS0 field
    phi = FlatS0(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        basis = Basis.FOURIER,
        parametrization = Parametrization.T,
        scalar_matrix = jnp.array(jl.phi_P)
    )
    #Example FlatS2 field
    data = FlatS2(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        basis = Basis.FOURIER,
        parametrization = Parametrization.EB,
        polar_matrix_1 = jnp.array(jl.data_e_field_P),
        polar_matrix_2 = jnp.array(jl.data_b_field_P)
    )

    #Put all the data together inside of a polarization only dataset object
    data_set = DataSetEB(
        #covariance matrices...
        noise_covariance = noise_covariance,
        mixing_d = noise_covariance.replace(polar_matrix_1 = jnp.array(jl.d_ee_P),
                                            polar_matrix_2 = jnp.array(jl.d_bb_P)),
        field_covariance = noise_covariance.replace(polar_matrix_1 = jnp.array(jl.cf_ee_P),
                                                    polar_matrix_2 = jnp.array(jl.cf_bb_P)),
        phi_covariance = phi_covariance,
        mask = noise_covariance.replace(polar_matrix_1 = jnp.array(jl.m_ee_P),
                                        polar_matrix_2 = jnp.array(jl.m_bb_P)),
        beam = noise_covariance.replace(polar_matrix_1 = jnp.array(jl.b_ee_P),
                                        polar_matrix_2 = jnp.array(jl.b_bb_P)),
        quadratic_estimate = phi_covariance.replace(scalar_matrix = jnp.array(jl.nphi_P)),
        #fields...
        #NOTE we don't really care about the lensed field here so just use default
        #FlatS2() since we will never access it...
        data = data,
        unlensed_field = data.replace(polar_matrix_1 = jnp.array(jl.e_field_P),
                                      polar_matrix_2 = jnp.array(jl.b_field_P)),
        phi = phi
    )
else:
    #Load the temperature and polarization combo covariance matrices
    jl.seval(f"""
    cn_ee_IP = transpose(data_set.ds.Cn.ΣTE[4].diag.arr)
    cn_bb_IP = transpose(data_set.ds.Cn.ΣB.diag.arr)
    cn_te_IP = transpose(data_set.ds.Cn.ΣTE[2].diag.arr)
    cn_tt_IP = transpose(data_set.ds.Cn.ΣTE[1].diag.arr)
    cf_ee_IP = transpose(data_set.ds.Cf.op.ΣTE[4].diag.arr)
    cf_bb_IP = transpose(data_set.ds.Cf.op.ΣB.diag.arr)
    cf_te_IP = transpose(data_set.ds.Cf.op.ΣTE[2].diag.arr)
    cf_tt_IP = transpose(data_set.ds.Cf.op.ΣTE[1].diag.arr)
    cphi_IP = transpose(data_set.ds.Cϕ.op.diag.arr)
    nphi_IP = transpose(data_set.ds.Nϕ.diag.arr)
    b_ee_IP = transpose(data_set.ds.B.ΣTE[4].diag.arr)
    b_bb_IP = transpose(data_set.ds.B.ΣB.diag.arr)
    b_te_IP = transpose(data_set.ds.B.ΣTE[2].diag.arr)
    b_tt_IP = transpose(data_set.ds.B.ΣTE[1].diag.arr)
    m_ee_IP = transpose(data_set.ds.M.ΣTE[4].diag.arr)
    m_bb_IP = transpose(data_set.ds.M.ΣB.diag.arr)
    m_te_IP = transpose(data_set.ds.M.ΣTE[2].diag.arr)
    m_tt_IP = transpose(data_set.ds.M.ΣTE[1].diag.arr)
    d_ee_IP = transpose(data_set.ds.D.op.ΣTE[4].diag.arr)
    d_bb_IP = transpose(data_set.ds.D.op.ΣB.diag.arr)
    d_te_IP = transpose(data_set.ds.D.op.ΣTE[2].diag.arr)
    d_tt_IP = transpose(data_set.ds.D.op.ΣTE[1].diag.arr)
    """)

    #Load the temperature and polarization combo fields
    jl.seval("""
    phi_IP = transpose((data_set.ϕ)[:Il])
    t_field_IP = transpose((data_set.f)[:Il])
    e_field_IP = transpose((data_set.f)[:El])
    b_field_IP = transpose((data_set.f)[:Bl])
    data_t_field_IP = transpose(data_set.ds.d[:Il])
    data_e_field_IP = transpose(data_set.ds.d[:El])
    data_b_field_IP = transpose(data_set.ds.d[:Bl])
    """)

    #Convert these julia matrices to python objects..
    #Example diagonal scalar matrix operator
    phi_covariance = DiagonalScalar(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        scalar_matrix = jnp.array(jl.cphi_IP)
    )
    #Example block diagonal TEB matrix operator
    noise_covariance = BlockTEB(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        matrix_TT = jnp.array(jl.cn_tt_IP),
        matrix_TE = jnp.array(jl.cn_te_IP),
        matrix_ET = jnp.array(jl.cn_te_IP),
        matrix_EE = jnp.array(jl.cn_ee_IP),
        matrix_BB = jnp.array(jl.cn_bb_IP)
    )

    #Example FlatS0 field
    phi = FlatS0(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        basis = Basis.FOURIER,
        parametrization = Parametrization.T,
        scalar_matrix = jnp.array(jl.phi_IP)
    )
    #Example FlatS02 field
    data = FlatS02(
        fourier_weights = fourier_weights,
        nside = args.map_size,
        theta_pix = args.theta_pix,
        pix_width = pix_width,
        basis = Basis.FOURIER,
        parametrization = Parametrization.EB,
        scalar_matrix = jnp.array(jl.data_t_field_IP),
        polar_matrix_1 = jnp.array(jl.data_e_field_IP),
        polar_matrix_2 = jnp.array(jl.data_b_field_IP)
    )

    #Put all the data together inside of a polarization and temperature combo dataset object
    data_set = DataSetTEB(
        #covariance matrices...
        noise_covariance = noise_covariance,
        mixing_d = noise_covariance.replace(matrix_TT = jnp.array(jl.d_tt_IP),
                                            matrix_TE = jnp.array(jl.d_te_IP),
                                            matrix_ET = jnp.array(jl.d_te_IP),
                                            matrix_EE = jnp.array(jl.d_ee_IP),
                                            matrix_BB = jnp.array(jl.d_bb_IP)),
        field_covariance = noise_covariance.replace(matrix_TT = jnp.array(jl.cf_tt_IP),
                                                    matrix_TE = jnp.array(jl.cf_te_IP),
                                                    matrix_ET = jnp.array(jl.cf_te_IP),
                                                    matrix_EE = jnp.array(jl.cf_ee_IP),
                                                    matrix_BB = jnp.array(jl.cf_bb_IP)),
        phi_covariance = phi_covariance,
        mask = noise_covariance.replace(matrix_TT = jnp.array(jl.m_tt_IP),
                                        matrix_TE = jnp.array(jl.m_te_IP),
                                        matrix_ET = jnp.array(jl.m_te_IP),
                                        matrix_EE = jnp.array(jl.m_ee_IP),
                                        matrix_BB = jnp.array(jl.m_bb_IP)),
        beam = noise_covariance.replace(matrix_TT = jnp.array(jl.b_tt_IP),
                                        matrix_TE = jnp.array(jl.b_te_IP),
                                        matrix_ET = jnp.array(jl.b_te_IP),
                                        matrix_EE = jnp.array(jl.b_ee_IP),
                                        matrix_BB = jnp.array(jl.b_bb_IP)),
        quadratic_estimate = phi_covariance.replace(scalar_matrix = jnp.array(jl.nphi_IP)),
        #fields...
        #NOTE we don't really care about the lensed field here so just use default
        #FlatS2() since we will never access it...
        data = data,
        unlensed_field = data.replace(scalar_matrix = jnp.array(jl.t_field_IP),
                                      polar_matrix_1 = jnp.array(jl.e_field_IP),
                                      polar_matrix_2 = jnp.array(jl.b_field_IP)),
        phi = phi
    )

#run the algorithm once initially to get the uncached time
start_time = time.time()
field_python_predict, phi_python_predict = map_joint(data_set, num_steps = 30)
end_time = time.time()
total_time = end_time - start_time

#store the uncached time for this (f, phi) combo in its own proper directory
os.makedirs(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/uncached_times", exist_ok = True)
np.savetxt(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/uncached_times/uncached_time_{args.trial}.txt", \
           np.array([total_time]))

#now run the algorithm a second time to get the cached time
start_time = time.time()
field_python_predict, phi_python_predict  = map_joint(data_set, num_steps = 30)
end_time = time.time()
total_time = end_time - start_time

#store the cached time for this (f, phi) combo in its own proper directory
os.makedirs(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/cached_times", exist_ok = True)
np.savetxt(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/cached_times/cached_time_{args.trial}.txt", \
           np.array([total_time]))

#store the learned / estimated f and phi from python to compare with what julia found
os.makedirs(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb", exist_ok = True)
os.makedirs(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/lensing_potential", exist_ok = True)

#cmb fields:
#branching logic on saving the fields based on polarization equals I, P, or IP
if args.pol == "I":
    #Save the python prediction
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/t_field_python_predict_{args.trial}.npz", field_python_predict.scalar_matrix)
    #Also save the simulation ground truth
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/t_field_julia_simulation_{args.trial}.npz", jnp.array(jl.t_field_I))
elif args.pol == "P":
    #Save the python prediction
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/e_field_python_predict_{args.trial}.npz", field_python_predict.polar_matrix_1)
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/b_field_python_predict_{args.trial}.npz", field_python_predict.polar_matrix_2)
    #Also save the simulation ground truth
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/e_field_julia_simulation_{args.trial}.npz", jnp.array(jl.e_field_P))
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/b_field_julia_simulation_{args.trial}.npz", jnp.array(jl.b_field_P))
else:
    #Save the python prediction
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/t_field_python_predict_{args.trial}.npz", field_python_predict.scalar_matrix)
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/e_field_python_predict_{args.trial}.npz", field_python_predict.polar_matrix_1)
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/b_field_python_predict_{args.trial}.npz", field_python_predict.polar_matrix_2)
    #Also save the simulation ground truth
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/t_field_julia_simulation_{args.trial}.npz", jnp.array(jl.t_field_IP))
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/e_field_julia_simulation_{args.trial}.npz", jnp.array(jl.e_field_IP))
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/cmb/b_field_julia_simulation_{args.trial}.npz", jnp.array(jl.b_field_IP))

#Lensing potential fields:
#Save the python prediction
np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/lensing_potential/phi_python_predict_{args.trial}.npz", phi_python_predict.scalar_matrix)
#Also save the simulation ground truth
if args.pol == "I":
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/lensing_potential/phi_julia_simulation_{args.trial}.npz", jnp.array(jl.phi_I))
elif args.pol == "P":
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/lensing_potential/phi_julia_simulation_{args.trial}.npz", jnp.array(jl.phi_P))
else:
    np.savez(f"/resnick/groups/wugroup/zblood/cmb_lensing/performance_testing/performance_results/python_results/map_size_{args.map_size}_polarity_{args.pol}_seed_{args.seed}/learned_fields/lensing_potential/phi_julia_simulation_{args.trial}.npz", jnp.array(jl.phi_IP))