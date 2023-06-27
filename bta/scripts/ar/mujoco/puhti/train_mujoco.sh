#!/bin/bash
#SBATCH --job-name=mujoco-ar
#SBATCH --output=./out/mujoco-ar_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/mujoco-ar_err_%A_%a.txt  # Name of stderr error file
#SBATCH --account=project_2007776
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

env="mujoco"

scenario=$1
agent_conf=$2
agent_obsk=2

algo="ar"
exp="check"

echo "env is ${env}, scenario is ${scenario}, agent_conf is ${agent_conf}, algo is ${algo}, exp is ${exp}"

apptainer_wrapper exec /bin/sh -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/lizhiyua/.mujoco/mujoco210/bin; python ../../../train/train_mujoco.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} \
--seed $SLURM_ARRAY_TASK_ID --n_training_threads 8 --n_rollout_threads 32 --num_mini_batch 32 --episode_length 1000 --num_env_steps 10000000 \
--use_eval --eval_interval 25 --eval_episodes 5 --add_center_xy --use_state_agent --use_value_active_masks --use_policy_active_masks \
--ppo_epoch 5 --lr 5e-5 --critic_lr 5e-3 --attention_lr 5e-6 --std_x_coef 1 --std_y_coef 5e-1 --wandb_name "zhiyuanli" --user_name "zhiyuanli" "
