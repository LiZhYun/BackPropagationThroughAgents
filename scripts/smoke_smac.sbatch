#!/bin/bash
#SBATCH --job-name=bpta-agentmixer-smac-smoke
#SBATCH --partition=batch-milan,batch-skl,batch-csl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/work/liz23/reproduce/bpta-agentmixer/results/smoke/smac_%j.out
#SBATCH --error=/scratch/work/liz23/reproduce/bpta-agentmixer/results/smoke/smac_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SC2PATH=/scratch/work/liz23/3rdparty/StarCraftII
cd /scratch/work/liz23/reproduce/bpta-agentmixer
exec .venv-smac/bin/python scripts/smoke_smac.py --output "results/smoke/smac_${SLURM_JOB_ID}.json"
