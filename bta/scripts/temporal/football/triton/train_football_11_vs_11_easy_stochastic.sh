#!/bin/bash

#SBATCH --job-name=football-11_vs_11-temporal
#SBATCH --output=./out/football-11_vs_11-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/football-11_vs_11-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --array=0-4

# exp param
env="Football"
scenario="11_vs_11_easy_stochastic"
algo="temporal"
exp="check"

# football param
num_agents=10

# train param
num_env_steps=500000000
episode_length=1000

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/bpta.sif python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed $SLURM_ARRAY_TASK_ID \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--use_action_attention --kl_coef 0.