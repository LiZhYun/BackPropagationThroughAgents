#!/bin/bash
#SBATCH --job-name=predatorprey-mappo
#SBATCH --output=./out/predatorprey-mappo_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/predatorprey-mappo_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --partition=small
#SBATCH --account=project_462000277
#SBATCH --array=0-4

#--gpus-per-node=1
# exp param
env="predator_prey"
penalty=$1
algo="mappo"
exp="check"

# train param
num_env_steps=1000000
episode_length=200
# num_agents=50
# mix_std_y_coef=$1
# layer=$1
# token_factor=$2
# channel_factor=$3

srun singularity exec -B"$SCRATCH:$SCRATCH" $SCRATCH/bpta_lumi.sif python ../../../train/train_predatorprey.py \
--env_name ${env} --penalty ${penalty} --algorithm_name ${algo} --experiment_name ${exp} --seed $SLURM_ARRAY_TASK_ID \
--num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--save_interval 100000 --log_interval 10000 --use_eval --eval_interval 10000 --n_eval_rollout_threads 100 --eval_episodes 100 \
--n_rollout_threads 50 --num_mini_batch 1 \
--user_name "zhiyuanli" --wandb_name "zhiyuanli" \
--use_action_attention --mix_std_y_coef 0.95 --sigmoid_gain 1.0 --decay_id 3 --decay_factor 1.0 --temperature 1.0 --attention_lr 5e-5 --threshold 1.
# --mix_std_y_coef ${mix_std_y_coef}
# --use_action_attention --attn_N ${layer} --token_factor ${token_factor} --channel_factor ${channel_factor}
