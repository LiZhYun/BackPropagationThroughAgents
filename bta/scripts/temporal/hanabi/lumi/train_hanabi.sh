#!/bin/bash
#SBATCH --job-name=hanabi-temporal
#SBATCH --output=hanabi-temporal.o%j # Name of stdout output file
#SBATCH --error=hanabi-temporal.e%j  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=small-g
#SBATCH --account=project_462000277
#SBATCH --array=0-100

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="temporal"
exp="Hanabi_2"
deno=100
threshold=`echo "scale=2; $SLURM_ARRAY_TASK_ID / $deno" | bc`

srun singularity exec -B"$SCRATCH:$SCRATCH" $SCRATCH/bpta_lumi.sif python ../../../train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--hanabi_name ${hanabi} --num_agents ${num_agents} --seed 1 --n_training_threads 1 --n_rollout_threads 1000 \
--num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 \
--threshold ${threshold} --lr 7e-4 --critic_lr 1e-3 --gain 0.01 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 \ 
--user_name "zhiyuanli" --wandb_name "zhiyuanli"