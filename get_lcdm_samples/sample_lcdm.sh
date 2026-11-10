#!/bin/bash

#Run with "bash sample_lcdm.sh" (not sbatch): it only submits the jobs. It moves into its own
#folder and sources the untracked repo-root hpc_local.sh (conda, SBATCH_MAIL_USER; copy
#hpc_local.example.sh to create it), which the jobs inherit. Submission is throttled
#(../hpc_submit.sh), so the driver runs until the last job is submitted: start it with
#"nohup bash sample_lcdm.sh > submit.log 2>&1 &" or inside tmux
#
#Every setting below is a default; override any of them as --<name>=<value>, e.g.
#    bash sample_lcdm.sh --noise_level=10 --omch2_proposal_sigma=2e-3 --out_dir_name=run_10uk
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1

#every --name=value argument becomes the variable name=value
for arg in "$@"; do declare "${arg#--}"; done

source ../hpc_local.sh || exit 1
source ../hpc_submit.sh || exit 1

#systematics
nside=${nside:-128}
theta_pix=${theta_pix:-2.5}
noise_level=${noise_level:-5}
map_prefix=${map_prefix:-234567}
pol=${pol:-I}

#parameter values for the ground truth simulated data
ombh2_ground=${ombh2_ground:-0.022386}
omch2_ground=${omch2_ground:-0.109381}
theta_MC_100_ground=${theta_MC_100_ground:-1.031732}
log_a_ground=${log_a_ground:-3.218387}
ns_ground=${ns_ground:-0.959814}

#MCMC parameters for the experiment
num_maps=${num_maps:-10} #data map realizations of the ground truth
num_chains=${num_chains:-5} #MCMC chains per data map

#We sample linearly from -5 sigma to +5 sigma with the ground truth equal to the mean (0 sigma)
ombh2_min=${ombh2_min:-0.020413}
ombh2_max=${ombh2_max:-0.024389}

omch2_min=${omch2_min:-0.085}
omch2_max=${omch2_max:-0.155541}

theta_MC_100_min=${theta_MC_100_min:-0.9328}
theta_MC_100_max=${theta_MC_100_max:-1.1452}

log_a_min=${log_a_min:-2.661635}
log_a_max=${log_a_max:-3.782861}

ns_min=${ns_min:-0.867143}
ns_max=${ns_max:-1.042186}

#which parameters are sampled (true / false). A sampled parameter starts every chain at a
#uniform random point in its [min, max] box; an unsampled one is held at its ground truth
sample_ombh2=${sample_ombh2:-false}
sample_omch2=${sample_omch2:-true}
sample_theta_MC_100=${sample_theta_MC_100:-true}
sample_log_a=${sample_log_a:-true}
sample_ns=${sample_ns:-false}

#widths of the Gaussian random-walk Metropolis proposals, per parameter. Tune them with a
#single pilot chain toward ~44% acceptance per sampled parameter
ombh2_proposal_sigma=${ombh2_proposal_sigma:-1e-4}
omch2_proposal_sigma=${omch2_proposal_sigma:-1e-3}
theta_MC_100_proposal_sigma=${theta_MC_100_proposal_sigma:-1e-3}
log_a_proposal_sigma=${log_a_proposal_sigma:-1e-2}
ns_proposal_sigma=${ns_proposal_sigma:-1e-2}

#other sampler hyper-parameters
num_burn_in_fix_theta=${num_burn_in_fix_theta:-100}
iters_per_chain=${iters_per_chain:-3000}
num_burn_in_always_accept=${num_burn_in_always_accept:-0}
hmc_num_steps=${hmc_num_steps:-10}
hmc_step_size=${hmc_step_size:-0.05}

#name of the folder every chain of this run writes to, inside lcdm_sampler_data/ (default: the
#submission date), where chain_analysis.py reads it
out_dir_name=${out_dir_name:-joint_inference_$(date +%Y_%m_%d)}
out_dir="$here/lcdm_sampler_data/$out_dir_name"
mkdir -p "$out_dir"

#store all these common parameters in an
#array to pass to our slurm jobs
args=(
      "$nside"
      "$theta_pix"
      "$noise_level"
      "$ombh2_min"
      "$ombh2_ground"
      "$ombh2_max"
      "$omch2_min"
      "$omch2_ground"
      "$omch2_max"
      "$theta_MC_100_min"
      "$theta_MC_100_ground"
      "$theta_MC_100_max"
      "$log_a_min"
      "$log_a_ground"
      "$log_a_max"
      "$ns_min"
      "$ns_ground"
      "$ns_max"
      "$pol"
      "$num_burn_in_fix_theta"
      "$iters_per_chain"
      "$num_burn_in_always_accept"
      "$hmc_num_steps"
      "$hmc_step_size"
      "$ombh2_proposal_sigma"
      "$omch2_proposal_sigma"
      "$theta_MC_100_proposal_sigma"
      "$log_a_proposal_sigma"
      "$ns_proposal_sigma"
      "$out_dir"
      "$sample_ombh2"
      "$sample_omch2"
      "$sample_theta_MC_100"
      "$sample_log_a"
      "$sample_ns"
)

#audit trail: the submission date and every setting, rewritten on every run; each chain's
#random starting point is added below as it is submitted
audit="$out_dir/submission_settings.txt"
cat > "$audit" <<EOF
submitted $(date "+%Y-%m-%d %H:%M:%S %Z") by $USER on $(hostname)
nside=$nside
theta_pix=$theta_pix
noise_level=$noise_level
map_prefix=$map_prefix
pol=$pol
ombh2_ground=$ombh2_ground
omch2_ground=$omch2_ground
theta_MC_100_ground=$theta_MC_100_ground
log_a_ground=$log_a_ground
ns_ground=$ns_ground
num_maps=$num_maps
num_chains=$num_chains
ombh2_min=$ombh2_min
ombh2_max=$ombh2_max
omch2_min=$omch2_min
omch2_max=$omch2_max
theta_MC_100_min=$theta_MC_100_min
theta_MC_100_max=$theta_MC_100_max
log_a_min=$log_a_min
log_a_max=$log_a_max
ns_min=$ns_min
ns_max=$ns_max
sample_ombh2=$sample_ombh2
sample_omch2=$sample_omch2
sample_theta_MC_100=$sample_theta_MC_100
sample_log_a=$sample_log_a
sample_ns=$sample_ns
ombh2_proposal_sigma=$ombh2_proposal_sigma
omch2_proposal_sigma=$omch2_proposal_sigma
theta_MC_100_proposal_sigma=$theta_MC_100_proposal_sigma
log_a_proposal_sigma=$log_a_proposal_sigma
ns_proposal_sigma=$ns_proposal_sigma
num_burn_in_fix_theta=$num_burn_in_fix_theta
iters_per_chain=$iters_per_chain
num_burn_in_always_accept=$num_burn_in_always_accept
hmc_num_steps=$hmc_num_steps
hmc_step_size=$hmc_step_size
out_dir_name=$out_dir_name

chain starting points (map_seed chain ombh2 omch2 theta_MC_100 logA ns):
EOF

#draw a uniform random value in [min, max] using bash's $RANDOM (0..32767)
rand_uniform() {
    local min=$1
    local max=$2
    echo "$min + ($RANDOM / 32767) * ($max - $min)" | bc -l
}

#a chain's starting value: uniform in [min, max] if the parameter is sampled, else the truth
start_value() {
    local sampled=$1
    local min=$2
    local max=$3
    local ground=$4
    if [ "$sampled" = true ]; then rand_uniform "$min" "$max"; else echo "$ground"; fi
}

for ((map=1; map<=num_maps; map++)); do
    map_seed=$(echo "$map_prefix * $map" | bc -l)
    for ((chain=1; chain<=num_chains; chain++)); do
        ombh2_init=$(start_value "$sample_ombh2" "$ombh2_min" "$ombh2_max" "$ombh2_ground")
        omch2_init=$(start_value "$sample_omch2" "$omch2_min" "$omch2_max" "$omch2_ground")
        theta_MC_100_init=$(start_value "$sample_theta_MC_100" "$theta_MC_100_min" "$theta_MC_100_max" "$theta_MC_100_ground")
        log_a_init=$(start_value "$sample_log_a" "$log_a_min" "$log_a_max" "$log_a_ground")
        ns_init=$(start_value "$sample_ns" "$ns_min" "$ns_max" "$ns_ground")
        echo "$map_seed $chain $ombh2_init $omch2_init $theta_MC_100_init $log_a_init $ns_init" >> "$audit"
        #random initializations for each parameter for each (map, phi) combination
        submit_throttled run_single_lcdm_chain.sh "$map_seed" \
        "$chain" "$ombh2_init" "$omch2_init" "$theta_MC_100_init" \
        "$log_a_init" "$ns_init" "${args[@]}"
    done
done
