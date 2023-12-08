#!/bin/bash
#SBATCH --job-name=matrix-maven
#SBATCH --output=./out/matrix-maven_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/matrix-maven_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-4

#--gpus-per-node=1
# exp param
env="matrix"
scenario="climbing"
algo="maven"
exp="check"

# train param
num_env_steps=200000
episode_length=200
# num_agents=50
# mix_std_y_coef=$1
# layer=$1
# token_factor=$2
# channel_factor=$3

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/bpta.sif python ../../../train/train_matrix.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed $SLURM_ARRAY_TASK_ID \
--num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 10000 --use_eval --eval_interval 10000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--n_rollout_threads 50 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"
# --mix_std_y_coef ${mix_std_y_coef}
# --use_action_attention --attn_N ${layer} --token_factor ${token_factor} --channel_factor ${channel_factor}
