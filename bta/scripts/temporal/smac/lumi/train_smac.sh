#!/bin/bash
#SBATCH --job-name=smac-temporal
#SBATCH --output=./out/smac-temporal_%A_%a.out # Name of stdout output file
#SBATCH --error=./out/smac-temporal_err_%A_%a.txt  # Name of stderr error file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000277
#SBATCH --array=0-4

env="StarCraft2v2"
map=$1 # zerg_20_vs_20 zerg_10_vs_11 zerg_20_vs_23 protoss_20_vs_20 protoss_10_vs_11 protoss_20_vs_23 terran_20_vs_20 terran_10_vs_11 terran_20_vs_23
units=$2
# IGM_coef=$3
# sigmoid_gain=$3
# mix_std_y_coef=$4
# temperature=$3
# decay_id=$6
# decay_factor=$7

algo="temporal"
exp="check"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

srun singularity exec -B $SCRATCH $SCRATCH/smac_lumi.sif python ../../../train/train_smac.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed $SLURM_ARRAY_TASK_ID --units ${units} --n_training_threads 1 --n_rollout_threads 25 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 --use_action_attention \
--mix_std_y_coef 0.95 --sigmoid_gain 1.0 --decay_id 3 --decay_factor 1.0 --temperature 1.0 --attention_lr 5e-5 --threshold 1.
# --attention_lr 5e-5 --mix_std_y_coef ${mix_std_y_coef} \
# --temperature ${temperature} --decay_id ${decay_id} --decay_factor ${decay_factor} --mix_std_y_coef 0.95
# --IGM_coef ${IGM_coef} 

