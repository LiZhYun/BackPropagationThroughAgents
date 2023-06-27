#!/bin/bash

# cramped_corridor diff_orders h_far_tomato multiplayer_schelling(old) soup_coordination
sbatch train_overcooked_ar.sh cramped_corridor
sbatch train_overcooked_ar.sh diff_orders
sbatch train_overcooked_ar.sh h_far_tomato
sbatch train_overcooked_ar.sh multiplayer_schelling
sbatch train_overcooked_ar.sh soup_coordination
