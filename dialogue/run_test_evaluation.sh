#!/bin/bash
#SBATCH --job-name=test_dual_encoder
#SBATCH --output=logs/test_evaluation_%j.out
#SBATCH --error=logs/test_evaluation_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

echo "================================================================================"
echo "DUAL ENCODER TEST EVALUATION JOB"
echo "================================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPU:          $SLURM_GPUS"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Memory:       32GB"
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
    "code/test_dual_encoder.py"
    "triplets_output/test_triplets.jsonl"
    "dual_encoder_checkpoints/best_model.pt"
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
    echo ""
    if [ ! -f "dual_encoder_checkpoints/best_model.pt" ]; then
        echo "    Training checkpoint not found!"
        echo "    Please complete training first (run_training.sh)"
    fi
    if [ ! -f "code/test_dual_encoder.py" ]; then
        echo "    Test script not found!"
        echo "    Please create code/test_dual_encoder.py"
    fi
    exit 1
fi

# Check and create output directory
if [ ! -d "test_results" ]; then
    echo ""
    echo "Creating test_results directory..."
    mkdir -p test_results
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
    print(" WARNING: No GPU detected!")
EOF
echo ""

# Check test data file
echo "--------------------------------------------------------------------------------"
echo "Test data information:"
echo "--------------------------------------------------------------------------------"

TEST_FILE="triplets_output/test_triplets.jsonl"

if [ -f "$TEST_FILE" ]; then
    TEST_LINES=$(wc -l < "$TEST_FILE")
    TEST_SIZE=$(du -h "$TEST_FILE" | cut -f1)
    echo "Test data:"
    echo "  File:     $TEST_FILE"
    echo "  Samples:  $TEST_LINES"
    echo "  Size:     $TEST_SIZE"
else
    echo "✗ Test file not found: $TEST_FILE"
    exit 1
fi

echo ""

# Check checkpoint
echo "--------------------------------------------------------------------------------"
echo "Model checkpoint information:"
echo "--------------------------------------------------------------------------------"

CHECKPOINT_FILE="dual_encoder_checkpoints/best_model.pt"

if [ -f "$CHECKPOINT_FILE" ]; then
    CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_FILE" | cut -f1)
    echo "Checkpoint:"
    echo "  File:  $CHECKPOINT_FILE"
    echo "  Size:  $CHECKPOINT_SIZE"
else
    echo "✗ Checkpoint not found: $CHECKPOINT_FILE"
    exit 1
fi

echo ""

# Record start time
JOB_START_TIME=$SECONDS

# ============================================================================
# TEST EVALUATION
# ============================================================================
echo "================================================================================"
echo "STARTING TEST EVALUATION"
echo "================================================================================"
echo ""
echo "Script: code/test_dual_encoder.py"
echo "Model:  Dual Encoder (best_model.pt)"
echo ""
echo "Evaluation configuration:"
echo "  Test samples:  $TEST_LINES"
echo "  Batch size:    32"
echo "  Max seq len:   256"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

EVAL_START=$SECONDS

python code/test_dual_encoder.py

EVAL_EXIT_CODE=$?
EVAL_END=$SECONDS
EVAL_RUNTIME=$((EVAL_END - EVAL_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✗✗✗ ERROR: Test evaluation failed!"
    echo "================================================================================"
    echo "Exit code: $EVAL_EXIT_CODE"
    echo "Runtime:   ${EVAL_RUNTIME}s"
    echo "Check logs above for error details."
    exit 1
fi

echo "✓ Test evaluation completed successfully"

EVAL_MINUTES=$((EVAL_RUNTIME / 60))
EVAL_SECONDS=$((EVAL_RUNTIME % 60))
echo "Runtime: ${EVAL_MINUTES}m ${EVAL_SECONDS}s"

# ============================================================================
# Verify Outputs
# ============================================================================
echo ""
echo "--------------------------------------------------------------------------------"
echo "Verifying evaluation outputs..."
echo "--------------------------------------------------------------------------------"

if [ -d "test_results" ]; then
    echo "✓ Results directory created: test_results/"
    echo ""
    
    # Check for expected output files
    if [ -f "test_results/test_metrics.json" ]; then
        echo "  ✓ test_metrics.json"
        echo ""
        echo "    Key Metrics:"
        python << 'EOF'
import json
with open('test_results/test_metrics.json', 'r') as f:
    metrics = json.load(f)
print(f"      Recall@1:  {metrics['recall@1']:.4f} ({metrics['recall@1']*100:.2f}%)")
print(f"      Recall@5:  {metrics['recall@5']:.4f} ({metrics['recall@5']*100:.2f}%)")
print(f"      Recall@10: {metrics['recall@10']:.4f} ({metrics['recall@10']*100:.2f}%)")
print(f"      MRR:       {metrics['mrr']:.4f}")
EOF
    else
        echo "  ✗ MISSING: test_metrics.json"
    fi
    
    echo ""
    
    if [ -f "test_results/detailed_predictions.jsonl" ]; then
        PRED_LINES=$(wc -l < "test_results/detailed_predictions.jsonl")
        PRED_SIZE=$(du -h "test_results/detailed_predictions.jsonl" | cut -f1)
        echo "  ✓ detailed_predictions.jsonl"
        echo "      Predictions: $PRED_LINES"
        echo "      Size:        $PRED_SIZE"
    else
        echo "  ✗ MISSING: detailed_predictions.jsonl"
    fi
    
    if [ -f "test_results/successful_predictions.jsonl" ]; then
        SUCCESS_LINES=$(wc -l < "test_results/successful_predictions.jsonl")
        echo "  ✓ successful_predictions.jsonl ($SUCCESS_LINES samples)"
    else
        echo "  ✗ MISSING: successful_predictions.jsonl"
    fi
    
    if [ -f "test_results/failed_predictions.jsonl" ]; then
        FAIL_LINES=$(wc -l < "test_results/failed_predictions.jsonl")
        echo "  ✓ failed_predictions.jsonl ($FAIL_LINES samples)"
    else
        echo "  ✗ MISSING: failed_predictions.jsonl"
    fi
    
else
    echo "✗ Results directory not found: test_results/"
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
echo "  Evaluation runtime: ${EVAL_MINUTES}m ${EVAL_SECONDS}s"
if [ $TOTAL_HOURS -gt 0 ]; then
    echo "  Total runtime:      ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
else
    echo "  Total runtime:      ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
fi
echo ""
echo "Output Files:"
echo "  ✓ test_results/test_metrics.json"
echo "  ✓ test_results/detailed_predictions.jsonl"
echo "  ✓ test_results/successful_predictions.jsonl"
echo "  ✓ test_results/failed_predictions.jsonl"
echo ""
echo "Next Steps:"
echo "  1. Review test_results/test_metrics.json for performance"
echo "  2. Analyze failed_predictions.jsonl for error patterns"
echo "  3. Decide: Build cross-encoder OR iterate OR deploy"
echo ""
echo "================================================================================"
echo "Test evaluation complete!"
echo "================================================================================"
