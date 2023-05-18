#!/bin/bash

#SBATCH --job-name=football-temporal
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=500M
#SBATCH --gres=gpu:1
#SBATCH --array=0-10

# exp param
env="Football"
scenario="11_vs_11_hard_stochastic"
algo="temporal" # "mappo" "ippo"
exp="check"

# football param
num_agents=10

# train param
num_env_steps=10000000
episode_length=200

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/BackpropagationThroughAgents/bpta.sif python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 12 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--max_edges 10  --time_channels 100 --time_gap 10