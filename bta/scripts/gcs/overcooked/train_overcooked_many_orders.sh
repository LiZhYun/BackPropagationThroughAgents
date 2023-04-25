#!/bin/bash

#SBATCH --job-name=overcooked-gcs
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-10

module restore pytorch2002-di
source activate pytorch-env

env="Overcooked"

# unident_s, random1, random3, distant_tomato, many_orders
layout="many_orders"

version="old"
if [[ "${layout}" == "distant_tomato" || "${layout}" == "many_orders" ]]; then
    version="new"
fi

num_agents=2
algo="gcs"
exp="check"

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 python train/train_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
    --ppo_epoch 15 \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --save_interval 25 --log_inerval 10 --use_recurrent_policy\
    --overcooked_version ${version} \
    --wandb_name "zhiyuanli" --user_name "zhiyuanli" 

# Deactivate the environment
source deactivate
