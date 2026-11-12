#!/bin/bash

#Run with "bash get_sampler_phi_noise.sh" (not sbatch): it only submits the jobs. It
#moves into its own folder and sources the untracked repo-root hpc_local.sh (conda,
#SBATCH_MAIL_USER; copy hpc_local.example.sh to create it), which the jobs inherit.
#Submission is throttled (../../hpc_submit.sh), so the driver runs until the last job is
#submitted: start it with "nohup bash get_sampler_phi_noise.sh > submit.log 2>&1 &" or in tmux
#
#Every setting below is a default; override any of them as --<name>=<value>, e.g.
#    bash get_sampler_phi_noise.sh --noise_level=10 --num_chains=4 --out_dir_name=sampler_phi_noise_output_10uk
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || exit 1

#every --name=value argument becomes the variable name=value
for arg in "$@"; do declare "${arg#--}"; done

source ../../hpc_local.sh || exit 1
source ../../hpc_submit.sh || exit 1

#Driver for the SAMPLER PHI NOISE: the phi reconstruction noise of the sampler's own
#posterior mean. Spawns one job per (data map, chain). Each job builds the deterministic
#(f, phi, d) realization load_sim gives at its map seed, then runs ONE chain of sample_joint's
#Gibbs sweep with theta HELD at the ground truth - (f, phi) are sampled, theta is not - and
#stores EVERY phi sample (no burn-in cut, no thinning). Chain 0 of a map starts at phi = 0,
#every other chain at its own draw from the C_phi prior, so the chains' agreement after the
#burn-in (Gelman-Rubin R-hat per mode, plot_sampler_convergence.py) tests convergence.
#
#The analysis burns every chain in, pools a map's chains into its posterior mean phi_mean,
#and compares that with the realization's true phi. Per mode, with the averages taken over
#the realizations before the ratio:
#
#    r_L^2 = <Re(phi_mean phi_true*)> / <|phi_true|^2>
#    N_L   = C_L^phiphi (1 / r_L^2 - 1)
#
#with a delete-one-realization jackknife error. See cmb_lensing/sampler_phi_noise.py.
#
#The jobs can be analysed WHILE THEY RUN (each flushes its samples every 10 sweeps and the
#merge reads only the sweeps a job reports as complete), on the HPC, from this folder (set
#burn_in in it first):
# sbatch merge_sampler_phi_noise.sh

#systematics: MATCH sample_lcdm.sh so the noise describes the box the chains run on
nside=${nside:-128}
theta_pix=${theta_pix:-2.5}
noise_level=${noise_level:-5}
l_knee=${l_knee:-0}
#the data's polarization: I (temperature), P (polarization) or IP (both)
pol=${pol:-I}
hmc_step_size=${hmc_step_size:-0.05}
hmc_num_steps=${hmc_num_steps:-10}

#Gibbs sweeps per chain
num_sweeps=${num_sweeps:-6000}

#data maps (one seed each). The averages are taken over the maps per mode, so the per-mode
#noise of N falls as 1/sqrt(num_maps)
num_maps=${num_maps:-100}
seed_prefix=${seed_prefix:-234567}
#chains per data map, one slurm job each (at least 2 for the R-hat convergence check)
num_chains=${num_chains:-2}
#seeds of the chains' random numbers, one per chain index and reused for every map (space
#separated, e.g. --chain_seeds="11 22"). Empty (the default): each chain draws its own at
#random and stores it in its sidecar sampler_phi_NNNN_cK.npz (`chain_seed`)
chain_seeds=${chain_seeds:-}

#wall time and memory per job. A job killed at the wall clock keeps every flushed sweep and
#is analysed as a shorter chain.
chain_time=${chain_time:-18:00:00}
chain_mem=${chain_mem:-16G}

#name of the output folder shared by every job, created inside this folder; the merge reads it
out_dir_name=${out_dir_name:-sampler_phi_noise_output}
out_dir="$here/$out_dir_name"
mkdir -p "$out_dir"

#audit trail: the submission date and every setting, rewritten on every run
cat > "$out_dir/submission_settings.txt" <<EOF
submitted $(date "+%Y-%m-%d %H:%M:%S %Z") by $USER on $(hostname)
nside=$nside
theta_pix=$theta_pix
noise_level=$noise_level
l_knee=$l_knee
pol=$pol
hmc_step_size=$hmc_step_size
hmc_num_steps=$hmc_num_steps
num_sweeps=$num_sweeps
num_maps=$num_maps
seed_prefix=$seed_prefix
num_chains=$num_chains
chain_seeds=$chain_seeds
chain_time=$chain_time
chain_mem=$chain_mem
out_dir_name=$out_dir_name
EOF

#chain_seeds as an array; ${seeds[chain]} is empty past its end, i.e. a random seed
read -r -a seeds <<< "$chain_seeds"

#one job per (map seed, chain); the merge groups a map's chains by its map seed
for ((m=0; m<num_maps; m++)); do
    map_seed=$((seed_prefix + m))
    for ((chain=0; chain<num_chains; chain++)); do
        submit_throttled --time="$chain_time" --mem-per-cpu="$chain_mem" \
            get_sampler_phi_noise_1_chain.sh "$m" "$map_seed" "$nside" "$theta_pix" \
            "$noise_level" "$l_knee" "$num_sweeps" "$out_dir" "$hmc_step_size" \
            "$hmc_num_steps" "$chain" "$pol" "${seeds[chain]}"
    done
done
