#!/bin/bash
#SBATCH --job-name=dual_encoder_train
#SBATCH --output=logs/dual_encoder_training_%j.out
#SBATCH --error=logs/dual_encoder_training_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

echo "================================================================================"
echo "DUAL ENCODER TRAINING JOB"
echo "================================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPU:          $SLURM_GPUS"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Memory:       64GB"
echo "================================================================================"

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate conda environment
echo ""
echo "--------------------------------------------------------------------------------"
echo "Activating conda environment..."
echo "--------------------------------------------------------------------------------"
conda activate /dist_home/suryansh/dialogue/mtdrenv

if [ $? -ne 0 ]; then
    echo ""
    echo "✗✗✗ ERROR: Failed to activate conda environment!"
    echo "    Environment path: /dist_home/suryansh/dialogue/mtdrenv"
    exit 1
fi

echo "✓ Environment activated successfully"
echo ""
echo "Python executable: $(which python)"
echo "Python version:    $(python --version)"
echo ""

# Verify working directory
echo "--------------------------------------------------------------------------------"
echo "Verifying working directory..."
echo "--------------------------------------------------------------------------------"
echo "Current directory: $(pwd)"
echo "Expected: /dist_home/suryansh/dialogue"
echo ""

if [ "$(pwd)" != "/dist_home/suryansh/dialogue" ]; then
    echo "✗✗✗ ERROR: Not in correct directory!"
    echo "    Please cd to /dist_home/suryansh/dialogue before submitting"
    exit 1
fi

echo "✓ Working directory correct"
echo ""

# Check required files and directories
echo "--------------------------------------------------------------------------------"
echo "Checking required files and directories..."
echo "--------------------------------------------------------------------------------"

REQUIRED_FILES=(
    "code/model.py"
    "code/dataset.py"
    "code/utils.py"
    "code/train_dual_encoder.py"
    "triplets_output/train_triplets.jsonl"
    "triplets_output/val_triplets.jsonl"
)

MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ MISSING: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "✓ Found: $file"
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "✗✗✗ ERROR: $MISSING_FILES required file(s) missing!"
    exit 1
fi

# Check required directories
if [ ! -d "logs" ]; then
    echo ""
    echo "Creating logs directory..."
    mkdir -p logs
fi

if [ ! -d "dual_encoder_checkpoints" ]; then
    echo ""
    echo "Creating checkpoints directory..."
    mkdir -p dual_encoder_checkpoints
fi

if [ ! -d "training_logs" ]; then
    echo ""
    echo "Creating training logs directory..."
    mkdir -p training_logs
fi

echo ""
echo "✓ All required files present"
echo ""

# Display Python package versions
echo "--------------------------------------------------------------------------------"
echo "Python package versions:"
echo "--------------------------------------------------------------------------------"
python -c "import torch; print(f'torch:              {torch.__version__}')" 2>/dev/null || echo "torch: NOT FOUND"
python -c "import transformers; print(f'transformers:       {transformers.__version__}')" 2>/dev/null || echo "transformers: NOT FOUND"
python -c "import numpy; print(f'numpy:              {numpy.__version__}')" 2>/dev/null || echo "numpy: NOT FOUND"
python -c "from indicnlp import __version__; print(f'indic-nlp-library:  {__version__}')" 2>/dev/null || echo "indic-nlp-library: version check failed (OK)"
python -c "import bm25s; print(f'bm25s:              installed')" 2>/dev/null || echo "bm25s: NOT FOUND"
echo ""

# Check CUDA availability
echo "--------------------------------------------------------------------------------"
echo "GPU and CUDA information:"
echo "--------------------------------------------------------------------------------"
python << 'EOF'
import torch
print(f"CUDA available:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:       {torch.version.cuda}")
    print(f"GPU count:          {torch.cuda.device_count()}")
    print(f"GPU name:           {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ WARNING: No GPU detected!")
EOF
echo ""

# Check data files
echo "--------------------------------------------------------------------------------"
echo "Data file information:"
echo "--------------------------------------------------------------------------------"

TRAIN_FILE="triplets_output/train_triplets.jsonl"
VAL_FILE="triplets_output/val_triplets.jsonl"

if [ -f "$TRAIN_FILE" ]; then
    TRAIN_LINES=$(wc -l < "$TRAIN_FILE")
    TRAIN_SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
    echo "Training data:"
    echo "  File:     $TRAIN_FILE"
    echo "  Samples:  $TRAIN_LINES"
    echo "  Size:     $TRAIN_SIZE"
else
    echo "✗ Training file not found: $TRAIN_FILE"
fi

echo ""

if [ -f "$VAL_FILE" ]; then
    VAL_LINES=$(wc -l < "$VAL_FILE")
    VAL_SIZE=$(du -h "$VAL_FILE" | cut -f1)
    echo "Validation data:"
    echo "  File:     $VAL_FILE"
    echo "  Samples:  $VAL_LINES"
    echo "  Size:     $VAL_SIZE"
else
    echo "✗ Validation file not found: $VAL_FILE"
fi

echo ""

# Record start time
JOB_START_TIME=$SECONDS

# ============================================================================
# TRAINING
# ============================================================================
echo "================================================================================"
echo "STARTING DUAL ENCODER TRAINING"
echo "================================================================================"
echo ""
echo "Script: code/train_dual_encoder.py"
echo "Model:  Dual Encoder with frozen MuRIL"
echo ""
echo "Training configuration:"
echo "  Epochs:        10"
echo "  Batch size:    32"
echo "  Learning rate: 2e-5"
echo "  Output dim:    256"
echo "  Num layers:    2"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

TRAIN_START=$SECONDS

python code/train_dual_encoder.py

TRAIN_EXIT_CODE=$?
TRAIN_END=$SECONDS
TRAIN_RUNTIME=$((TRAIN_END - TRAIN_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✗✗✗ ERROR: Training failed!"
    echo "================================================================================"
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Runtime:   ${TRAIN_RUNTIME}s"
    echo "Check logs above for error details."
    exit 1
fi

echo "✓ Training completed successfully"

TRAIN_MINUTES=$((TRAIN_RUNTIME / 60))
TRAIN_SECONDS=$((TRAIN_RUNTIME % 60))
echo "Runtime: ${TRAIN_MINUTES}m ${TRAIN_SECONDS}s"

# ============================================================================
# Verify Outputs
# ============================================================================
echo ""
echo "--------------------------------------------------------------------------------"
echo "Verifying training outputs..."
echo "--------------------------------------------------------------------------------"

if [ -d "dual_encoder_checkpoints" ]; then
    echo "✓ Checkpoint directory created"
    
    if [ -f "dual_encoder_checkpoints/best_model.pt" ]; then
        BEST_SIZE=$(du -h "dual_encoder_checkpoints/best_model.pt" | cut -f1)
        echo "  ✓ best_model.pt (Size: $BEST_SIZE)"
    else
        echo "  ✗ MISSING: best_model.pt"
    fi
    
    CHECKPOINT_COUNT=$(ls -1 dual_encoder_checkpoints/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "  ✓ Epoch checkpoints: $CHECKPOINT_COUNT files"
    fi
else
    echo "✗ Checkpoint directory not found"
fi

echo ""

if [ -d "training_logs" ]; then
    LOG_COUNT=$(ls -1 training_logs/epoch_*_metrics.json 2>/dev/null | wc -l)
    if [ $LOG_COUNT -gt 0 ]; then
        echo "✓ Training logs: $LOG_COUNT epoch metric files"
    fi
else
    echo "✗ Training logs directory not found"
fi

# ============================================================================
# Job Summary
# ============================================================================
JOB_END_TIME=$SECONDS
TOTAL_RUNTIME=$((JOB_END_TIME - JOB_START_TIME))
TOTAL_HOURS=$((TOTAL_RUNTIME / 3600))
REMAINING_SECONDS=$((TOTAL_RUNTIME % 3600))
TOTAL_MINUTES=$((REMAINING_SECONDS / 60))
TOTAL_SECONDS=$((REMAINING_SECONDS % 60))

echo ""
echo ""
echo "================================================================================"
echo "JOB COMPLETED SUCCESSFULLY ✓✓✓"
echo "================================================================================"
echo ""
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "End time:      $(date)"
echo ""
echo "Timing Summary:"
echo "  Training runtime: ${TRAIN_MINUTES}m ${TRAIN_SECONDS}s"
if [ $TOTAL_HOURS -gt 0 ]; then
    echo "  Total runtime:    ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
else
    echo "  Total runtime:    ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
fi
echo ""
echo "Output Files:"
echo "  ✓ dual_encoder_checkpoints/best_model.pt"
echo "  ✓ dual_encoder_checkpoints/checkpoint_epoch_*.pt"
echo "  ✓ training_logs/epoch_*_metrics.json"
echo ""
echo "Next Steps:"
echo "  1. Check training_logs/ for metrics"
echo "  2. Verify best_model.pt exists"
echo "  3. Proceed to evaluation or cross-encoder training"
echo ""
echo "================================================================================"
echo "Dual encoder training complete!"
echo "================================================================================"
