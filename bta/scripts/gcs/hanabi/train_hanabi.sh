#!/bin/bash

#SBATCH --job-name=hanabi-gcs
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10G
#SBATCH --time=72:00:00
#SBATCH --array=0-1

module restore bpta
source activate pytorch-env

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="gcs"
exp="check"

python ../../train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--hanabi_name ${hanabi} --num_agents ${num_agents} --seed 1 --n_training_threads 1 --n_rollout_threads 1000 \
--num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 \
--gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 --user_name "zhiyuanli" --wandb_name "zhiyuanli"

# Deactivate the environment
source deactivate

