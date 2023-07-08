#!/bin/bash
#SBATCH --job-name=mujoco-temporal_ar
#SBATCH --output=./out/mujoco-temporal_ar_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/mujoco-temporal_ar_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --time=1-00:00:00
#SBATCH --partition=small-g
#SBATCH --account=project_462000277
#SBATCH --array=0-4

env="mujoco"

scenario=$1
agent_conf=$2
agent_obsk=0

algo="temporal"
exp="check"

echo "env is ${env}, scenario is ${scenario}, agent_conf is ${agent_conf}, algo is ${algo}, exp is ${exp}"

srun singularity exec -B $SCRATCH $SCRATCH/mujo_gfoot_v2.sif /bin/sh -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/lizhiyua/.mujoco/mujoco210/bin; python ../../../train/train_mujoco.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} \
--seed $SLURM_ARRAY_TASK_ID --n_rollout_threads 40 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 \
--use_eval --eval_interval 25 --eval_episodes 5 --add_center_xy --use_state_agent --use_value_active_masks --use_policy_active_masks \
--layer_N 2 --ppo_epoch 5 --lr 3e-4 --critic_lr 3e-4 --attention_lr 3e-4 --std_x_coef 1 --std_y_coef 5e-1 --entropy_coef 0 --wandb_name "zhiyuanli" --user_name "zhiyuanli""