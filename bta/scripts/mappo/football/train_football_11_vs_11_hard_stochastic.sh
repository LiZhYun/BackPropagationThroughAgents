#!/bin/bash

#SBATCH --job-name=football-mappo
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-10

module restore pytorch2002-di
conda activate pytorch-env

# exp param
env="Football"
scenario="11_vs_11_hard_stochastic"
algo="mappo" # "mappo" "ippo"
exp="check"

# football param
num_agents=10

# train param
num_env_steps=25000000
episode_length=200

echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python ../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
--save_interval 200000 --log_interval 200000 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" 