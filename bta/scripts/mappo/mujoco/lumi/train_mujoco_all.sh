#!/bin/bash

# cramped_corridor diff_orders h_far_tomato multiplayer_schelling(old) soup_coordination
sbatch train_mujoco.sh HalfCheetah-v2 2x3
sbatch train_mujoco.sh HalfCheetah-v2 3x2
sbatch train_mujoco.sh HalfCheetah-v2 6x1
sbatch train_mujoco.sh Ant-v2 2x4
sbatch train_mujoco.sh Ant-v2 4x2
sbatch train_mujoco.sh Ant-v2 8x1
sbatch train_mujoco.sh Walker2d-v2 2x3
sbatch train_mujoco.sh Walker2d-v2 3x2
sbatch train_mujoco.sh Walker2d-v2 6x1
