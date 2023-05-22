#!/bin/bash

#SBATCH --job-name=football-lazy-temporal
#SBATCH --output=./out/football-lazy-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/football-lazy-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-5

# exp param
env="Football"
scenario="academy_single_goal_versus_lazy"
algo="temporal"
exp="check"

# football param
num_agents=10

# train param
num_env_steps=50000000
episode_length=1000

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

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/bpta.sif python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --num_mini_batch 2 --threshold ${threshold} \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"