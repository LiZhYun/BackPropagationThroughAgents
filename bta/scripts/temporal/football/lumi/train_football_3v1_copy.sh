#!/bin/bash
#SBATCH --job-name=football-3v1-temporal
#SBATCH --output=./out/football-3v1-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/football-3v1-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=small
#SBATCH --account=project_462000277
#SBATCH --array=0-5

#--gpus-per-node=1
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

case $SLURM_ARRAY_TASK_ID in
    0)
        inner_clip=0.05
        ;;
    1)
        inner_clip=0.1
        ;;
    2)
        inner_clip=0.15
        ;;
    3)
        inner_clip=0.2
        ;;
    4)
        inner_clip=0.25
        ;;
    5)
        inner_clip=0.
        ;;
esac

srun singularity exec -B"$SCRATCH:$SCRATCH" $SCRATCH/bpta_lumi.sif python ../../../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --num_mini_batch 2 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" --inner_clip_param ${inner_clip} --automatic_entropy_tuning