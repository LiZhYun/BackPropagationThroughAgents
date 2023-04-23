#!/bin/bash

#SBATCH --job-name=overcooked-temporal
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-9

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

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/BackpropagationThroughAgents/bpta.sif python train/train_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed 1 --n_training_threads 1 --n_rollout_threads 50 --num_mini_batch 1 --episode_length 200 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
    --overcooked_version ${version} \
    --wandb_name "zhiyuanli" --user_name "zhiyuanli" \
    --max_edges 10 --time_channels 100 --time_gap 4
