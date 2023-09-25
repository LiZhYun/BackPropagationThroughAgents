#!/bin/bash
#SBATCH --job-name=hanabi-mappo
#SBATCH --output=./out/hanabi-mappo_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/hanabi-mappo_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=small
#SBATCH --account=project_462000277
#SBATCH --array=0-4

env="Hanabi"
hanabi="Hanabi-Full"
num_agents=$1
algo="mappo"
exp="Hanabi_"${num_agents}

echo "env is ${env}, algo is ${algo}, exp is ${exp}"

srun singularity exec -B"$SCRATCH:$SCRATCH" $SCRATCH/bpta_lumi.sif python ../../../train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--hanabi_name ${hanabi} --num_agents ${num_agents} --seed $SLURM_ARRAY_TASK_ID --n_training_threads 1 --n_rollout_threads 1000 --n_eval_rollout_threads 1000 --use_eval \
--num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 \
--gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 --max_grad_norm 10.0 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"