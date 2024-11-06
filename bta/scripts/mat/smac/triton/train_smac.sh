#!/bin/bash

#SBATCH --job-name=smac-mat
#SBATCH --output=./out/smac-mat_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/smac-mat_err_%A_%a.txt  # Name of stderr error file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4

echo "export SC2PATH=/scratch/work/zhaow7/3rdparty/StarCraftII/" >> ~/.bashrc

env="StarCraft2v2"
map=$1 # zerg_20_vs_20 zerg_10_vs_11 zerg_20_vs_23 protoss_20_vs_20 protoss_10_vs_11 protoss_20_vs_23 terran_20_vs_20 terran_10_vs_11 terran_20_vs_23
units=$2

algo="mat"
exp="check"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/zhaow7/gfootball_mujoco.sif /bin/sh -c \
"export SC2PATH=/scratch/work/zhaow7/3rdparty/StarCraftII/; \
python ../../../train/train_smac.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed $SLURM_ARRAY_TASK_ID --units ${units} --n_rollout_threads 50 --num_mini_batch 2 --episode_length 200 \
--num_env_steps 10000000 --use_value_active_masks --use_eval --eval_episodes 32 \
--layer_N 1 --clip_param 0.05 --ppo_epoch 5 --lr 5e-4 --critic_lr 5e-4 --std_x_coef 1 --std_y_coef 5e-1 --entropy_coef 0.01 --wandb_name "zhiyuanli" --user_name "zhiyuanli" \
"
# --train_sim_seq ${train_sim_seq}

#0 --sigmoid_gain 0.3 --mix_std_y_coef 0.75 --layer_N 2 --clip_param 0.2 --ppo_epoch 5 --lr 3e-4 --critic_lr 3e-4 --attention_lr 3e-5 --std_x_coef 1 --std_y_coef 5e-1 --entropy_coef 0 --wandb_name "zhiyuanli" --user_name "zhiyuanli" \
#1 --sigmoid_gain 1 --mix_std_y_coef 0.5 --layer_N 2 --clip_param 0.2 --ppo_epoch 5 --lr 3e-4 --critic_lr 3e-4 --attention_lr 3e-5 --std_x_coef 1 --std_y_coef 5e-1 --entropy_coef 0.01 --wandb_name "zhiyuanli" --user_name "zhiyuanli" \
#2 --sigmoid_gain 0.3 --mix_std_y_coef 0.75 --layer_N 2 --clip_param 0.2 --ppo_epoch 5 --lr 3e-4 --critic_lr 3e-4 --attention_lr 3e-5 --std_x_coef 1 --std_y_coef 5e-1 --entropy_coef 0.01 --wandb_name "zhiyuanli" --user_name "zhiyuanli" \

