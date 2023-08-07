#!/bin/bash

#SBATCH --job-name=football-3v1-temporal_ar
#SBATCH --output=./out/football-3v1-temporal_ar_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/football-3v1-temporal_ar_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --array=0-4

# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="temporal" # "mappo" "ippo"
exp="check"

# football param
num_agents=3

# train param
num_env_steps=25000000
episode_length=200

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/bpta.sif python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed $SLURM_ARRAY_TASK_ID \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"