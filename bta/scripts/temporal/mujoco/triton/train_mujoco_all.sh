#!/bin/bash

# cramped_corridor diff_orders h_far_tomato multiplayer_schelling(old) soup_coordination
# sbatch train_mujoco.sh HalfCheetah-v2 2x3
# sbatch train_mujoco.sh HalfCheetah-v2 3x2
# sbatch train_mujoco.sh HalfCheetah-v2 6x1
sbatch train_mujoco.sh Ant-v2 2x4
sbatch train_mujoco.sh Ant-v2 4x2
sbatch train_mujoco.sh Ant-v2 8x1
sbatch train_mujoco.sh Ant-v2 2x4d
# sbatch train_mujoco.sh Walker2d-v2 2x3
# sbatch train_mujoco.sh Walker2d-v2 3x2
# sbatch train_mujoco.sh Walker2d-v2 6x1
# sbatch train_mujoco.sh Humanoid-v2 17x1
# sbatch train_mujoco.sh HumanoidStandup-v2 17x1
# sbatch train_mujoco.sh Hopper-v2 3x1
# sbatch train_mujoco.sh Reacher-v2 2x1
# sbatch train_mujoco.sh coupled_half_cheetah 1p1
# sbatch train_mujoco.sh manyagent_swimmer 10x2
# sbatch train_mujoco.sh manyagent_swimmer 20x1
# sbatch train_mujoco.sh manyagent_swimmer 5x4
# sbatch train_mujoco.sh manyagent_ant 2x3
# sbatch train_mujoco.sh manyagent_ant 3x2
# sbatch train_mujoco.sh manyagent_ant 6x1