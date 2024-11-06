#!/bin/bash

#SBATCH --job-name=football-3v1-temporal
#SBATCH --output=./out/football-3v1-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/football-3v1-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4

# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="temporal" # "mappo" "ippo"
exp="check"

# football param
num_agents=3

# train param
num_env_steps=1000000
episode_length=200

sigmoid_gain=0.3
mix_std_y_coef=0.75

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/zhaow7/gfootball_triton.sif /bin/sh -c \
"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/bin/../lib/libstdc++.so.6; \
python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed $SLURM_ARRAY_TASK_ID \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 250 --log_interval 250 --use_eval --eval_interval 500 --eval_episodes 5 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 40 --num_mini_batch 1 \
--layer_N 2 --ppo_epoch 5 --lr 3e-4 --critic_lr 3e-4 --attention_lr 3e-5 --std_x_coef 1 --std_y_coef 5e-1 --entropy_coef 0 --wandb_name "zhiyuanli" --user_name "zhiyuanli" \
--use_action_attention --sigmoid_gain ${sigmoid_gain} --mix_std_y_coef ${mix_std_y_coef}"