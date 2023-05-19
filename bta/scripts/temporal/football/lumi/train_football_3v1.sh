#!/bin/bash
#SBATCH --job-name=football-3v1-temporal
#SBATCH --output=football-3v1-temporal.o%j # Name of stdout output file
#SBATCH --error=football-3v1-temporal.e%j  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --partition=small-g
#SBATCH --account=project_462000277
#SBATCH --array=0-100

# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="temporal"
exp="check"

# football param
num_agents=3

# train param
num_env_steps=25000000
episode_length=200
deno=100
# threshold=0.2
threshold=`echo "scale=2; $SLURM_ARRAY_TASK_ID / $deno" | bc`

srun singularity exec -B"$SCRATCH:$SCRATCH" $SCRATCH/bpta_lumi.sif python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --num_mini_batch 2  --threshold ${threshold} \
--user_name "zhiyuanli" --wandb_name "zhiyuanli"