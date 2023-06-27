#!/bin/bash
#SBATCH --job-name=overcooked-happo
#SBATCH --output=./out/overcooked-happo_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/overcooked-happo_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=small
#SBATCH --account=project_462000277
#SBATCH --array=0-4

env="Overcooked"

layout=$1

version="new"
num_agents=2
if [[ "${layout}" == "multiplayer_schelling" ]]; then
    version="old"
    num_agents=4
fi

algo="happo"
exp="check"

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp} version is ${version}"

srun singularity exec -B $SCRATCH $SCRATCH/bpta_lumi.sif python ../../../train/train_overcooked.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
--seed $SLURM_ARRAY_TASK_ID --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
--ppo_epoch 15 --max_grad_norm 10.0 \
--cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_eval --n_eval_rollout_threads 100 --save_interval 25 --log_inerval 10 \
--overcooked_version ${version} \
--wandb_name "zhiyuanli" --user_name "zhiyuanli" 
