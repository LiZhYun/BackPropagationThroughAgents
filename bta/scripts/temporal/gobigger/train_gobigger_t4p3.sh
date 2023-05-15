#!/bin/bash

#SBATCH --job-name=gobigger-temporal
#SBATCH --partition=hugemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=340G
#SBATCH --time=72:00:00
#SBATCH --array=0-2

module restore bpta
source activate pytorch-env

env="GoBigger"

team_num=4
player_num_per_team=3
scenario="t4p3"
num_agents=3
algo="temporal"
exp="check"

# train param
num_env_steps=10000000000
episode_length=256

python ../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --team_num ${team_num} --player_num_per_team ${player_num_per_team} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 100000 --use_eval --eval_interval 200000 --n_eval_rollout_threads 5 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 32 --ppo_epoch 15 --clip_param 0.2 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--max_edges 10  --time_channels 100 --time_gap 10 --time_gap 10 --lr 1e-4 --critic_lr 1e-4 --entropy_coef 0.05 --max_grad_norm 0.5 --value_loss_coef 0.5
