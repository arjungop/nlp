#!/bin/bash
#SBATCH --job-name=prep_cross_encoder_data
#SBATCH --output=logs/prep_cross_encoder_%j.out
#SBATCH --error=logs/prep_cross_encoder_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

echo "================================================================================"
echo "PREPARE CROSS-ENCODER DATA"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start:  $(date)"
echo "================================================================================"

eval "$(conda shell.bash hook)"
conda activate /dist_home/suryansh/dialogue/mtdrenv

cd /dist_home/suryansh/dialogue

# Phase 1: Build response bank
echo ""
echo "PHASE 1: Building response bank..."
python code/build_response_bank.py

# Phase 2: Mine hard negatives
echo ""
echo "PHASE 2: Mining hard negatives..."
python code/mine_hard_negatives.py

echo ""
echo "================================================================================"
echo "DATA PREPARATION COMPLETE"
echo "End: $(date)"
echo "================================================================================"
