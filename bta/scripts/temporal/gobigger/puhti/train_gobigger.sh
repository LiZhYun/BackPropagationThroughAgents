#!/bin/bash
#SBATCH --account=project_2007776
#SBATCH --job-name=gobigger-temporal
#SBATCH --output=./out/gobigger-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/gobigger-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-4

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

env="GoBigger"

# t2p2, t3p2, t4p3
team_num=$1
player_num_per_team=$2
scenario="t"${team_num}"p"${player_num_per_team}
num_agents=$2
algo="temporal"
exp="check"

# train param
num_env_steps=25000000
episode_length=200

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}"

apptainer_wrapper exec python ../../../train/train_gobigger.py \
--env_name ${env} --team_num ${team_num} --player_num_per_team ${player_num_per_team} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 10 --eval_episodes 100 \
--n_rollout_threads 20 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--use_action_attention --kl_coef 0. --token_factor $SLURM_ARRAY_TASK_ID