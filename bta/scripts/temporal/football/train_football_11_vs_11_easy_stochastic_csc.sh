#!/bin/bash

#SBATCH --account=project_2007776
#SBATCH --job-name=football-temporal
#SBATCH --partition=hugemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --time=48:00:00
#SBATCH --array=0-100:5

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

# exp param
env="Football"
scenario="11_vs_11_easy_stochastic"
algo="temporal" # "mappo" "ippo"
exp="check"

# football param
num_agents=10

# train param
num_env_steps=25000000
episode_length=200
deno=100
threshold=`echo "scale=2; $SLURM_ARRAY_TASK_ID / $deno" | bc`

apptainer_wrapper exec python ../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --clip_param 0.2 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--max_edges 10  --time_channels 100 --time_gap 10 --threshold ${threshold} --entropy_coef 0.05