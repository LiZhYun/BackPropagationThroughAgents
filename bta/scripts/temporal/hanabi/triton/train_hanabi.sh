#!/bin/bash

#SBATCH --job-name=hanabi-temporal
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10G
#SBATCH --time=120:00:00
#SBATCH --array=0-100:5

module restore bpta
source activate pytorch-env

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=5
algo="temporal"
exp="Hanabi_5"
deno=100
threshold=`echo "scale=2; $SLURM_ARRAY_TASK_ID / $deno" | bc`

python ../../train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--hanabi_name ${hanabi} --num_agents ${num_agents} --seed 1 --n_training_threads 1 --n_rollout_threads 1000 \
--num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 --clip_param 0.2 \
--threshold ${threshold} --lr 7e-4 --critic_lr 1e-3 --gain 0.01 --hidden_size 512 --layer_N 2 --entropy_coef 0.025 --user_name "zhiyuanli" --wandb_name "zhiyuanli"

# Deactivate the environment
source deactivate