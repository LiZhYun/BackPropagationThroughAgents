#!/bin/bash

#SBATCH --job-name=overcooked-temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10G
#SBATCH --time=72:00:00
#SBATCH --array=0-2

module restore bpta
source activate pytorch-env

env="Overcooked"

# unident_s, random1, random3, distant_tomato, many_orders
layout="random1"

version="old"
if [[ "${layout}" == "distant_tomato" || "${layout}" == "many_orders" ]]; then
    version="new"
fi

num_agents=2
algo="temporal"
exp="check"
deno=100
threshold=1.0

python ../../train/train_overcooked.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed 1 --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
    --ppo_epoch 15 --clip_param 0.2 --threshold ${threshold} --entropy_coef 0.05 \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_eval --n_eval_rollout_threads 100 --save_interval 25 --log_inerval 10 \
    --overcooked_version ${version} \
    --wandb_name "zhiyuanli" --user_name "zhiyuanli" 
