#!/bin/bash
#SBATCH --job-name=matrix-temporal
#SBATCH --output=./out/matrix-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/matrix-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000277
#SBATCH --array=0-1

#--gpus-per-node=1
# exp param
env="matrix"
scenario="permutation"
algo="temporal"
exp="check"

# train param
num_env_steps=25000000
episode_length=200
num_agents=50
layer=$1

srun singularity exec -B"$SCRATCH:$SCRATCH" $SCRATCH/bpta_lumi.sif python ../../../train/train_matrix.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--use_action_attention --kl_coef 0. --attn_N ${layer}