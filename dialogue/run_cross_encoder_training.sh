#!/bin/bash
#SBATCH --job-name=cross_encoder_train
#SBATCH --output=logs/cross_encoder_training_%j.out
#SBATCH --error=logs/cross_encoder_training_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

echo "================================================================================"
echo "CROSS-ENCODER TRAINING"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start:  $(date)"
echo "================================================================================"

eval "$(conda shell.bash hook)"
conda activate /dist_home/suryansh/dialogue/mtdrenv

cd /dist_home/suryansh/dialogue

python code/train_cross_encoder.py

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE"
echo "End: $(date)"
echo "================================================================================"
