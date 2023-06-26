#!/bin/bash

#SBATCH --account=project_2007776
#SBATCH --job-name=hanabi-temporal
#SBATCH --output=./out/hanabi-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/hanabi-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=$1
algo="temporal"
exp="Hanabi_"${num_agents}

echo "env is ${env}, algo is ${algo}, exp is ${exp}"

apptainer_wrapper exec python ../../../train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--hanabi_name ${hanabi} --num_agents ${num_agents} --seed $SLURM_ARRAY_TASK_ID --n_training_threads 1 --n_rollout_threads 1000 --n_eval_rollout_threads 1000 --use_eval \
--num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 \
--gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 --max_grad_norm 10.0 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"