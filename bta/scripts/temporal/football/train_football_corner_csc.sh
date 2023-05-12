#!/bin/bash
#SBATCH --account=project_2007776
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100

#--gres=gpu:v100:1

export SING_IMAGE=/projappl/project_2007776/bpta.sif
export SING_FLAGS=--nv
export SING_FLAGS="-B /scratch/project_2007776 $SING_FLAGS"

# exp param
env="Football"
scenario="academy_corner"
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