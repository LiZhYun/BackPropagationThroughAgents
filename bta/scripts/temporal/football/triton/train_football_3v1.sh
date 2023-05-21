#!/bin/bash

#SBATCH --job-name=football-temporal
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-3

#--partition=gpu
#--array=0-1
# --gres=gpu:1
#--gres=gpu:1
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="temporal" # "mappo" "ippo"
exp="check"

# football param
num_agents=3

# train param
num_env_steps=2500000
episode_length=200
deno=100
threshold=1.0
# threshold=`echo "scale=2; $SLURM_ARRAY_TASK_ID / $deno" | bc`

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, threshold is ${threshold}"

CUDA_VISIBLE_DEVICES=0 singularity exec --bind /scratch --nv /scratch/work/liz23/bpta.sif python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 50000 --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --clip_param 0.2 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--max_edges 10  --time_channels 100 --time_gap 10 --threshold ${threshold}
# --max_grad_norm 3.5 --lr 5e-4 --critic_lr 5e-4 --entropy_coef 0.05 \