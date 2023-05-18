#!/bin/bash
#SBATCH --account=project_2007776
#SBATCH --job-name=gobigger-temporal
#SBATCH --partition=hugemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --time=72:00:00
#SBATCH --array=0-100:5

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

env="GoBigger"

team_num=2
player_num_per_team=2
scenario="t2p2"
num_agents=2
algo="temporal"
exp="check"
deno=100
threshold=`echo "scale=2; $SLURM_ARRAY_TASK_ID / $deno" | bc`

# train param
num_env_steps=10000000000
episode_length=256

apptainer_wrapper exec python ../../train/train_gobigger.py \
--env_name ${env} --scenario_name ${scenario} --team_num ${team_num} --player_num_per_team ${player_num_per_team} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 100000 --use_eval --eval_interval 200000 --n_eval_rollout_threads 5 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 32 --ppo_epoch 15 --clip_param 0.2 --num_mini_batch 32 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--max_edges 10  --time_channels 100 --time_gap 10 --lr 1e-4 --critic_lr 1e-4 --threshold ${threshold} --entropy_coef 0.05 --max_grad_norm 0.5 --value_loss_coef 0.5
