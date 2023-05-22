#!/bin/bash
#SBATCH --job-name=overcooked-cramped-corridor-temporal
#SBATCH --output=./out/overcooked-cramped-corridor_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/overcooked-cramped-corridor_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-5

module restore bpta
source activate pytorch-env

env="Overcooked"

# cramped_corridor diff_orders h_far_tomato multiplayer_schelling(old) soup_coordination
layout="cramped_corridor"

version="new"
if [[ "${layout}" == "multiplayer_schelling" ]]; then
    version="old"
fi

num_agents=2
algo="temporal"
exp="check"

case $SLURM_ARRAY_TASK_ID in
   0)
      threshold=0.0
      ;;
   1)
      threshold=0.2
      ;;
   2)
      threshold=0.4
      ;;
   3)
      threshold=0.6
      ;;
   4)
      threshold=0.8
      ;;
   5)
      threshold=1.0
      ;;
   *)
     threshold=1.0
     ;;
esac

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp} version is ${version}"

ython ../../../train/train_overcooked.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed 1 --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
    --ppo_epoch 15 --threshold ${threshold} \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_eval --n_eval_rollout_threads 100 --save_interval 25 --log_inerval 10 \
    --overcooked_version ${version} \
    --wandb_name "zhiyuanli" --user_name "zhiyuanli" 
