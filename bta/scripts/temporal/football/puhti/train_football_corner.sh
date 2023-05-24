#!/bin/bash

#SBATCH --job-name=football-corner-temporal
#SBATCH --output=./out/football-corner-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/football-corner-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --account=project_2007776
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-4

#--gres=gpu:v100:1

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

# exp param
env="Football"
scenario="academy_corner"
algo="temporal" # "mappo" "ippo"
exp="check"

# football param
num_agents=10

# train param
num_env_steps=50000000
episode_length=1000

apptainer_wrapper exec python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --num_mini_batch 2 --max_grad_norm 10.0 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"