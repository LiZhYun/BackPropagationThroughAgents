#!/bin/bash

# cramped_corridor diff_orders h_far_tomato multiplayer_schelling(old) soup_coordination
sbatch train_mujoco_dec.sh HalfCheetah-v2 2x3
sbatch train_mujoco_dec.sh HalfCheetah-v2 3x2
sbatch train_mujoco_dec.sh HalfCheetah-v2 6x1
sbatch train_mujoco_dec.sh Ant-v2 2x4
sbatch train_mujoco_dec.sh Ant-v2 4x2
sbatch train_mujoco_dec.sh Ant-v2 8x1
sbatch train_mujoco_dec.sh Walker2d-v2 2x3
sbatch train_mujoco_dec.sh Walker2d-v2 3x2
sbatch train_mujoco_dec.sh Walker2d-v2 6x1
sbatch train_mujoco_dec.sh Humanoid-v2 17x1
sbatch train_mujoco_dec.sh HumanoidStandup-v2 17x1
sbatch train_mujoco_dec.sh Hopper-v2 3x1
sbatch train_mujoco_dec.sh Reacher-v2 2x1
sbatch train_mujoco_dec.sh Ant-v2 2x4d