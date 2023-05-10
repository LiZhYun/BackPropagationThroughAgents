#!/bin/bash

#SBATCH --job-name=football-temporal
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-1

#--partition=gpu
#--array=0-1
# --gres=gpu:1
#--gres=gpu:1
# exp param

module restore bpta
source activate pytorch-env

env="matrix"
scenario="climbing"
algo="temporal" # "mappo" "ippo"
exp="check"

# football param
num_agents=2

# train param
num_env_steps=25000000
episode_length=200
threshold=1.0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, threshold is ${threshold}"

python ../../train/train_matrix.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --clip_param 0.2 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--max_edges 10  --time_channels 100 --time_gap 10 --threshold ${threshold}
# --max_grad_norm 3.5 --lr 5e-4 --critic_lr 5e-4 --entropy_coef 0.05 \