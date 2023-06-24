#!/bin/bash
#SBATCH --job-name=football-ca-temporal
#SBATCH --output=./out/football-ca-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/football-ca-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000277
#SBATCH --array=0-4

# exp param
env="Football"
scenario="academy_counterattack_hard"
algo="temporal"
exp="check"

# football param
num_agents=4

# train param
num_env_steps=50000000
episode_length=1000

srun singularity exec -B"$SCRATCH:$SCRATCH" $SCRATCH/bpta_lumi.sif python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--use_action_attention --kl_coef 0.