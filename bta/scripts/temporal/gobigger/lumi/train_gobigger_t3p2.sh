#!/bin/bash
#SBATCH --job-name=gobigger-t3p2-temporal
#SBATCH --output=gobigger-t3p2-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=gobigger-t3p2-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=small-g
#SBATCH --account=project_462000277
#SBATCH --array=0-3

env="GoBigger"

team_num=3
player_num_per_team=2
scenario="t3p2"
num_agents=2
algo="temporal"
exp="check"
deno=100
threshold=1.0

# train param
num_env_steps=2500000
episode_length=200

srun singularity exec -B $SCRATCH $SCRATCH/bpta_lumi.sif python ../../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --team_num ${team_num} --player_num_per_team ${player_num_per_team} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 10000 --eval_interval 200000 --n_eval_rollout_threads 10 --eval_episodes 100 --n_rollout_threads 50 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--threshold ${threshold}
