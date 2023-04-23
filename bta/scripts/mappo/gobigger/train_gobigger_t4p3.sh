#!/bin/bash

#SBATCH --job-name=gobigger-temporal
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-1

env="GoBigger"

team_num=4
player_num_per_team=3
scenario="t4p3"
num_agents=3
algo="mappo"
exp="check"

# train param
num_env_steps=25000000
episode_length=200

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/bpta.sif python ../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
