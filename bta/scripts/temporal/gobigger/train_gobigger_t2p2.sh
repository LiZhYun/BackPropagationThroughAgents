#!/bin/bash

#SBATCH --job-name=football-temporal
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=500M
#SBATCH --gres=gpu:1
#SBATCH --array=0-10

env="GoBigger"

team_num=2
player_num_per_team=2
senario="t2p2"
num_agents=2
algo="temporal"
exp="check"

echo "env is ${env}, senario is ${senario}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/BackpropagationThroughAgents/bpta.sif python train/train_gobigger.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 12 --num_mini_batch 1 --episode_length 200 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
    --wandb_name "zhiyuanli" --user_name "zhiyuanli" \
    --max_edges 10  --time_channels 100 --time_gap 4
