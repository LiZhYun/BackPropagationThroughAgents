#!/bin/bash

#SBATCH --job-name=gobigger-temporal
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-3

module restore bpta
source activate pytorch-env

env="GoBigger"

team_num=2
player_num_per_team=2
scenario="t2p2"
num_agents=2
algo="temporal"
exp="check"
deno=100
threshold=1.0

# train param
num_env_steps=2500000
episode_length=1000

python ../../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --team_num ${team_num} --player_num_per_team ${player_num_per_team} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 50000 --eval_interval 200000 --n_eval_rollout_threads 10 --eval_episodes 100 --n_rollout_threads 50 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--threshold ${threshold}
