#!/bin/bash
#SBATCH --job-name=mujoco-temporal
#SBATCH --output=./out/mujoco-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/mujoco-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4

#--gres=gpu:1
env="mujoco"

scenario=$1
agent_conf=$2
# agent_obsk=None

algo="temporal"
exp="check"

# layer=$3
# mix_id=$3
# token_factor=$3
# channel_factor=$4
# threshold=$3
# decay_factor=$4
# decay_id=$5sigmoid_gain
sigmoid_gain=0.3
mix_std_y_coef=0.75
# IGM_coef=$3

echo "env is ${env}, scenario is ${scenario}, agent_conf is ${agent_conf}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/zhaow7/mujoco_football_triton.sif /bin/sh -c \
"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhaow7/.mujoco/mujoco210/bin; \
python ../../../train/train_mujoco.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} \
--seed $SLURM_ARRAY_TASK_ID --n_rollout_threads 40 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 \
--use_eval --eval_interval 25 --eval_episodes 5 --add_center_xy --use_state_agent --use_value_active_masks --use_policy_active_masks \
--layer_N 2 --ppo_epoch 5 --lr 3e-4 --critic_lr 3e-4 --attention_lr 3e-5 --std_x_coef 1 --std_y_coef 5e-1 --entropy_coef 0 --wandb_name "zhiyuanli" --user_name "zhiyuanli" \
--use_action_attention --sigmoid_gain ${sigmoid_gain} --mix_std_y_coef ${mix_std_y_coef}"
#--threshold ${threshold} --decay_factor ${decay_factor} --decay_id ${decay_id}
        