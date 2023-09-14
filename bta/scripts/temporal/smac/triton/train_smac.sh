#!/bin/bash

#SBATCH --job-name=smac-temporal
#SBATCH --output=./out/smac-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/smac-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4

echo "export SC2PATH=/scratch/work/liz23/3rdparty/StarCraftII/" >> ~/.bashrc

env="StarCraft2v2"
map=$1 # zerg_20_vs_20 zerg_10_vs_11 zerg_20_vs_23 protoss_20_vs_20 protoss_10_vs_11 protoss_20_vs_23 terran_20_vs_20 terran_10_vs_11 terran_20_vs_23
units=$2
IGM_coef=$3
# train_sim_seq=$3

algo="temporal"
exp="check"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/bpta.sif python ../../../train/train_smac.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed $SLURM_ARRAY_TASK_ID --units ${units} --n_rollout_threads 50 --num_mini_batch 1 --episode_length 200 \
--num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 \
--use_action_attention --IGM_coef ${IGM_coef}
# --train_sim_seq ${train_sim_seq}

