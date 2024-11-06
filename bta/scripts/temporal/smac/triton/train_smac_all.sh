#!/bin/bash

#SBATCH --gres=gpu:1
# zerg_20_vs_20 zerg_10_vs_11 zerg_20_vs_23 protoss_20_vs_20 protoss_10_vs_11 protoss_20_vs_23 terran_20_vs_20 terran_10_vs_11 terran_20_vs_23
sbatch train_smac.sh 10gen_zerg 5v5
sbatch train_smac.sh 10gen_zerg 10v10
sbatch train_smac.sh 10gen_zerg 10v11
sbatch train_smac.sh 10gen_zerg 20v20
sbatch train_smac.sh 10gen_zerg 20v23
sbatch train_smac.sh 10gen_protoss 5v5
sbatch train_smac.sh 10gen_protoss 10v10
sbatch train_smac.sh 10gen_protoss 10v11
sbatch train_smac.sh 10gen_protoss 20v20
sbatch train_smac.sh 10gen_protoss 20v23
sbatch train_smac.sh 10gen_terran 5v5
sbatch train_smac.sh 10gen_terran 10v10
sbatch train_smac.sh 10gen_terran 10v11
sbatch train_smac.sh 10gen_terran 20v20 
sbatch train_smac.sh 10gen_terran 20v23 #todo