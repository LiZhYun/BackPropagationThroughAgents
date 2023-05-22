#!/bin/bash
#SBATCH --job-name=gobigger-t4p3-temporal
#SBATCH --output=gobigger-t4p3-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=gobigger-t4p3-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=small
#SBATCH --account=project_462000277
#SBATCH --array=0-3

#--gpus-per-node=1
env="GoBigger"

team_num=4
player_num_per_team=3
scenario="t4p3"
num_agents=3
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
