#!/bin/bash
#SBATCH --job-name=gobigger-t3p2-temporal
#SBATCH --output=gobigger-t3p2-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=gobigger-t3p2-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=small-g
#SBATCH --account=project_462000277
#SBATCH --array=0-5

env="GoBigger"

team_num=3
player_num_per_team=2
scenario="t3p2"
num_agents=2
algo="temporal"
exp="check"

# train param
num_env_steps=25000000
episode_length=200

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

srun singularity exec -B $SCRATCH $SCRATCH/bpta_lumi.sif python ../../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 5 --eval_episodes 100 \
--n_rollout_threads 50 --num_mini_batch 2  --threshold ${threshold} \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"
