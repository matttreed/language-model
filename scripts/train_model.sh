#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/train_model_%j.out
#SBATCH --error=scripts/logs/train_model_%j.err
#SBATCH --mem=100G

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

python src/main.py --version 3.0 --train