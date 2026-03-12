#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -N 1

#SBATCH --job-name=novel_sampling_xil
#SBATCH --output=./cluster/outputs/output.out
#SBATCH --error=./cluster/outputs/error.err
#SBATCH --partition edu-long
#SBATCH -t 0:30:0

# Activate conda environment
source ~/.bashrc
conda activate "thesis"

python try.py