from cmb_lensing.simulate import *
from cmb_lensing.wiener_filter import *

#sample the field
def gibbs_sample_f(field_start, data_set, cosmo_params, args):
    #NOTE really the only thing we use the "original" data set here for
    #besides meta data is the original "data" field that we are using as 
    #our input to our data --> LCDM parameters black box...

    #Run a new simulation
    pol = data_set_polarity(data_set)
    #TODO we need to JIT load_sim and switch to using CosmoPower over CAMB
    new_sim = load_sim(data_set.nside, data_set.theta_pix, pol, **cosmo_params)

    #Extract the unlensed field and data objects from the new simulation run
    new_field = new_sim.unlensed_field
    new_data = new_sim.data

    #Call the wiener filter with the field_start initial guess and data difference
    #between new and old simulations as the data term
    data_delta = data_set.data - new_data
    #When calling wiener filter we use the current phi and covariance matrices from 
    #our sampling algorithms...
    delta_field = wiener_filter(field_start, args["phi"], data_delta, 
                                args["field_covariance"], args["noise_covariance"], 
                                args["mask"], args["beam"])

    #Return the new simulated unlensed field plus the wiener filter contribution
    return new_field + delta_field


#---------------------------------------------------

#sample phi

#mass matrix

#hmc step

#symplectic integration

#---------------------------------------------------

#sample cosmo parameters "thetas"

#grid and sample single parameter "theta" via inverse transform

#---------------------------------------------------

#define main entry point which defines starting values for "thetas"

#1. we start by definining the number of steps "N" that each chain will loop over
#2. we then spawn off separate parallel chains for different vectors theta_i where each
#   parameter is sampled from a prior range
#3. Question... Is the result we want the mean final theta? Or do we essentially build N distributions
#   over theta and then take the geometric mean to find the final distribution at the end?

def sample_joint():
    return
