#!/bin/bash
#SBATCH --job-name=overcooked-soup-coordination-temporal
#SBATCH --output=./out/overcooked-soup-coordination_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/overcooked-soup-coordination_err_%A_%a.txt  # Name of stderr error file
#SBATCH --account=project_2007776
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-5

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

env="Overcooked"

# cramped_corridor diff_orders h_far_tomato multiplayer_schelling(old) soup_coordination
layout="soup_coordination"

version="new"
if [[ "${layout}" == "multiplayer_schelling"]]; then
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

apptainer_wrapper exec python ../../../train/train_overcooked.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed 1 --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
    --ppo_epoch 15 --threshold ${threshold} \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_eval --n_eval_rollout_threads 100 --save_interval 25 --log_inerval 10 \
    --overcooked_version ${version} \
    --wandb_name "zhiyuanli" --user_name "zhiyuanli" 
