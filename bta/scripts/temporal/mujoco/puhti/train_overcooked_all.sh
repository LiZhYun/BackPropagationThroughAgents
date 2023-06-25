#!/bin/bash

# cramped_corridor diff_orders h_far_tomato multiplayer_schelling(old) soup_coordination
sbatch train_overcooked.sh cramped_corridor
sbatch train_overcooked.sh diff_orders
sbatch train_overcooked.sh h_far_tomato
sbatch train_overcooked.sh multiplayer_schelling
sbatch train_overcooked.sh soup_coordination
