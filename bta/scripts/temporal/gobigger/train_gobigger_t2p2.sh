#!/bin/bash

#SBATCH --job-name=gobigger-temporal
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100

module restore bpta
source activate pytorch-env

env="GoBigger"

team_num=2
player_num_per_team=2
scenario="t2p2"
num_agents=2
algo="temporal"
exp="check"
deno=100
threshold=`echo "scale=2; $SLURM_ARRAY_TASK_ID / $deno" | bc`

# train param
num_env_steps=10000000
episode_length=200

python ../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --team_num ${team_num} --player_num_per_team ${player_num_per_team} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 100000 --use_eval --eval_interval 200000 --n_eval_rollout_threads 5 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 32 --ppo_epoch 15 --clip_param 0.2 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--max_edges 10  --time_channels 100 --time_gap 10 --threshold ${threshold} --entropy_coef 0.05
