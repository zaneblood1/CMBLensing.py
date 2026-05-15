#TODO actually implement...

#---------------------------------------------------

#sample f

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
