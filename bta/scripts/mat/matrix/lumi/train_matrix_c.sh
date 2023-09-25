#!/bin/bash
#SBATCH --job-name=matrix-mat
#SBATCH --output=./out/matrix-mat_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/matrix-mat_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000277
#SBATCH --array=0-4

#--gpus-per-node=1
# exp param
env="matrix"
scenario="climbing"
algo="mat"
exp="check"

# train param
num_env_steps=1000000
episode_length=200

srun singularity exec -B"$SCRATCH:$SCRATCH" $SCRATCH/bpta_lumi.sif python ../../../train/train_matrix.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed $SLURM_ARRAY_TASK_ID \
--num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 10000 --use_eval --eval_interval 10000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--n_rollout_threads 50 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"
