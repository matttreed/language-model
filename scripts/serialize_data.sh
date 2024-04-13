#!/bin/bash
#SBATCH --job-name=serialize_data
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/serialize_data_%j.out
#SBATCH --error=scripts/logs/serialize_data_%j.err
#SBATCH --mem=100G

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

python data/data.py
