#!/bin/bash

#SBATCH --job-name=gobigger-temporal
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-10

module restore pytorch2002-di
conda activate pytorch-env

env="GoBigger"

team_num=3
player_num_per_team=2
senario="t3p2"
num_agents=2
algo="temporal"
exp="check"

echo "env is ${env}, senario is ${senario}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 python train/train_gobigger.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
    --ppo_epoch 15 \
    --save_interval 25 --log_inerval 10 --use_recurrent_policy\
    --wandb_name "zhiyuanli" --user_name "zhiyuanli" \
    --max_edges 5  --time_channels 10 --time_gap 5