#!/bin/bash

#SBATCH --account=project_2007776
#SBATCH --job-name=hanabi-temporal
#SBATCH --partition=longrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --array=0-100:5

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=5
algo="temporal"
exp="Hanabi_5"
deno=100
threshold=`echo "scale=2; $SLURM_ARRAY_TASK_ID / $deno" | bc`

apptainer_wrapper exec python ../../train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--hanabi_name ${hanabi} --num_agents ${num_agents} --seed 1 --n_training_threads 1 --n_rollout_threads 1000 \
--num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 --clip_param 0.2 \
--threshold ${threshold} --lr 7e-4 --critic_lr 1e-3 --gain 0.01 --hidden_size 512 --layer_N 2 --entropy_coef 0.025 --user_name "zhiyuanli" --wandb_name "zhiyuanli"

# Deactivate the environment
source deactivate