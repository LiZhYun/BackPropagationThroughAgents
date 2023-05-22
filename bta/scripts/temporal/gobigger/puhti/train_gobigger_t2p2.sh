#!/bin/bash
#SBATCH --account=project_2007776
#SBATCH --job-name=gobigger-temporal
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-5

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

env="GoBigger"

team_num=2
player_num_per_team=2
scenario="t2p2"
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

apptainer_wrapper exec python ../../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 5 --eval_episodes 100 \
--n_rollout_threads 50 --num_mini_batch 2  --threshold ${threshold} \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"
