#!/bin/bash

#SBATCH --job-name=gobigger-temporal
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-3

module restore pytorch2002-di
conda activate pytorch-env

env="GoBigger"

team_num=3
player_num_per_team=2
senario="t3p2"
num_agents=2
algo="temporal"
exp="check"
threshold=1.0

# train param
num_env_steps=2500000
episode_length=256

echo "env is ${env}, senario is ${senario}, algo is ${algo}, exp is ${exp}"

python ../../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --team_num ${team_num} --player_num_per_team ${player_num_per_team} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 100000 --use_eval --eval_interval 200000 --n_eval_rollout_threads 5 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 32 --ppo_epoch 15 --clip_param 0.2 --num_mini_batch 32 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--max_edges 10  --time_channels 100 --time_gap 10 --lr 1e-4 --critic_lr 1e-4 --threshold ${threshold} --max_grad_norm 0.5 --value_loss_coef 0.5