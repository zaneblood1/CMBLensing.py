#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on... Since no MPI just set to 1
#SBATCH --ntasks=1 #number of processor cores / tasks... Since no MPI just set to 1
#SBATCH --mail-user=<USER>@<INSTITUTE>.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on failure only

#NOTE: This submission script is only a template. You must at the very least replace the email address
#and the absolute path on your HPC for the out_dir variable...

#shared CAMB settings for every grid point
lmax=4000 #must equal constants.DEFAULT_MAX_ELL so the grid shares load_sim's ell support
tau=0.05 #pinned by prior, not gridded
mnu=0.06 #pinned by prior, not gridded
k_pivot=0.05
alens=1

#CAMB accuracy. Defaults (all boosts = 1) match the existing 1D caches and load_sim's
#data-map path, and cost ~3.6 s/call. Raising accuracy_boost to 2 costs ~46 s/call (13x)
#and buys ~9e-4 -> better in lnCl, but then simulate.py:_run_camb must be raised to match
#or the data map and the model disagree. l_sample_boost is nearly free but buys almost
#nothing on its own
accuracy_boost=1
l_sample_boost=1
l_accuracy_boost=1

#output folder shared by every job
out_dir="ABSOLUTE_PATH_TO/sampling_chains/multi_param_CAMB_grid"

#Grid ranges. logA / ns / ombh2 / omch2 use the same -5 sigma to +5 sigma endpoints as
#sample_lcdm.sh so the existing 1D caches stay a valid check along each axis. H0 replaces
#theta_MC_100 because the grid has to be rectangular: theta is derived from H0, ombh2 and
#omch2 together, so a rectangular theta box has corners CAMB cannot solve (the 1D cache
#already loses 8 of its 50 nodes to exactly that). H0 in [25, 130] covers
#theta_MC_100 in [0.8804, 1.1899] at the fiducial ombh2 / omch2
h0_min=25
h0_max=130
log_a_min=2.661635
log_a_max=3.782861
ns_min=0.867143
ns_max=1.042186
ombh2_min=0.020413
ombh2_max=0.024389
omch2_min=0.079704
omch2_max=0.155541

#Nodes per axis, set by measuring the cubic-spline error on the existing 1D caches with
#CAMB's own numerical noise projected out. lnCl^TT is very nearly analytic in logA and ns
#(residual ~1e-13), so those axes converge at 4-5 nodes. H0 is the only expensive axis
#because moving it shifts the acoustic peaks in ell: 81 linearly spaced nodes hold the
#theta spacing at 0.0053 across the whole box, which the 1D theta cache says is what
#~1e-3 lnCl costs. Total = 81 x 5 x 5 x 5 x 7 = 70875 CAMB calls in 875 jobs
h0_nodes=81
log_a_nodes=5
ns_nodes=5
ombh2_nodes=5
omch2_nodes=7

#store all these common parameters in an
#array to pass to our slurm jobs
args=(
      "$h0_min"
      "$h0_max"
      "$h0_nodes"
      "$lmax"
      "$tau"
      "$mnu"
      "$k_pivot"
      "$alens"
      "$accuracy_boost"
      "$l_sample_boost"
      "$l_accuracy_boost"
      "$out_dir"
)

mkdir -p "$out_dir"

#value of node i on a linearly spaced axis of n nodes running from min to max
node_value() {
    local min=$1
    local max=$2
    local n=$3
    local i=$4
    echo "$min + ($max - $min) * $i / ($n - 1)" | bc -l
}

#spawn off a separate job for each (logA, ns, ombh2, omch2) combination. The H0 sweep is
#done inside the job rather than spawned out: one job per single grid point would be
#70875 sbatch calls, which takes hours just to submit and blows past MaxSubmitJobs on
#most clusters. To change the granularity, move the h0 loop out of
#run_single_camb_grid.py and into this script
for ((a=0; a<log_a_nodes; a++)); do
    log_a=$(node_value "$log_a_min" "$log_a_max" "$log_a_nodes" "$a")
    for ((b=0; b<ns_nodes; b++)); do
        ns=$(node_value "$ns_min" "$ns_max" "$ns_nodes" "$b")
        for ((c=0; c<ombh2_nodes; c++)); do
            ombh2=$(node_value "$ombh2_min" "$ombh2_max" "$ombh2_nodes" "$c")
            for ((d=0; d<omch2_nodes; d++)); do
                omch2=$(node_value "$omch2_min" "$omch2_max" "$omch2_nodes" "$d")
                sbatch run_single_camb_grid.sh "$a" "$log_a" "$b" "$ns" "$c" "$ombh2" "$d" "$omch2" "${args[@]}"
            done
        done
    done
done
