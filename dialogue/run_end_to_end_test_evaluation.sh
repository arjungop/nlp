#!/bin/bash
#SBATCH --job-name=test_e2e_system
#SBATCH --output=logs/end_to_end_test_evaluation_%j.out
#SBATCH --error=logs/end_to_end_test_evaluation_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

echo "================================================================================"
echo "END-TO-END TEST EVALUATION JOB (DUAL ENCODER + CROSS ENCODER)"
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
echo ""

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate conda environment
echo "--------------------------------------------------------------------------------"
echo "Activating conda environment..."
echo "--------------------------------------------------------------------------------"
conda activate /dist_home/suryansh/dialogue/mtdrenv

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to activate conda environment!"
    echo "    Environment path: /dist_home/suryansh/dialogue/mtdrenv"
    exit 1
fi

echo "Environment activated successfully"
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
    echo "ERROR: Not in correct directory!"
    echo "    Please cd to /dist_home/suryansh/dialogue before submitting"
    exit 1
fi

echo "Working directory correct"
echo ""

# Check required code files
echo "--------------------------------------------------------------------------------"
echo "Checking required code files..."
echo "--------------------------------------------------------------------------------"

REQUIRED_CODE_FILES=(
    "code/model.py"
    "code/cross_encoder_model.py"
    "code/inference.py"
    "code/evaluate_test_set.py"
)

MISSING_CODE=0

for file in "${REQUIRED_CODE_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "MISSING: $file"
        MISSING_CODE=$((MISSING_CODE + 1))
    else
        echo "Found: $file"
    fi
done

if [ $MISSING_CODE -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_CODE required code file(s) missing!"
    echo ""
    echo "Please ensure all code files are present:"
    echo "  - code/model.py (dual encoder definition)"
    echo "  - code/cross_encoder_model.py (cross encoder definition)"
    echo "  - code/inference.py (two-stage inference pipeline)"
    echo "  - code/evaluate_test_set.py (evaluation script)"
    exit 1
fi

echo ""
echo "All required code files present"
echo ""

# Check model checkpoints
echo "--------------------------------------------------------------------------------"
echo "Checking model checkpoints..."
echo "--------------------------------------------------------------------------------"

DUAL_CHECKPOINT="dual_encoder_checkpoints/best_model.pt"
CROSS_CHECKPOINT="cross_encoder_checkpoints/best_model.pt"

MISSING_CHECKPOINTS=0

if [ -f "$DUAL_CHECKPOINT" ]; then
    DUAL_SIZE=$(du -h "$DUAL_CHECKPOINT" | cut -f1)
    echo "Found dual encoder checkpoint:"
    echo "  File: $DUAL_CHECKPOINT"
    echo "  Size: $DUAL_SIZE"
else
    echo "MISSING: $DUAL_CHECKPOINT"
    MISSING_CHECKPOINTS=$((MISSING_CHECKPOINTS + 1))
fi

echo ""

if [ -f "$CROSS_CHECKPOINT" ]; then
    CROSS_SIZE=$(du -h "$CROSS_CHECKPOINT" | cut -f1)
    echo "Found cross encoder checkpoint:"
    echo "  File: $CROSS_CHECKPOINT"
    echo "  Size: $CROSS_SIZE"
else
    echo "MISSING: $CROSS_CHECKPOINT"
    MISSING_CHECKPOINTS=$((MISSING_CHECKPOINTS + 1))
fi

if [ $MISSING_CHECKPOINTS -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_CHECKPOINTS model checkpoint(s) missing!"
    echo "Please complete training for both models first."
    exit 1
fi

echo ""
echo "All model checkpoints present"
echo ""

# Check response bank
echo "--------------------------------------------------------------------------------"
echo "Checking response bank..."
echo "--------------------------------------------------------------------------------"

RESPONSES_FILE="response_bank/responses.txt"
EMBEDDINGS_FILE="response_bank/embeddings.pt"

MISSING_BANK=0

if [ -f "$RESPONSES_FILE" ]; then
    RESPONSE_LINES=$(wc -l < "$RESPONSES_FILE")
    RESPONSE_SIZE=$(du -h "$RESPONSES_FILE" | cut -f1)
    echo "Found response bank:"
    echo "  File:      $RESPONSES_FILE"
    echo "  Responses: $RESPONSE_LINES"
    echo "  Size:      $RESPONSE_SIZE"
else
    echo "MISSING: $RESPONSES_FILE"
    MISSING_BANK=$((MISSING_BANK + 1))
fi

echo ""

if [ -f "$EMBEDDINGS_FILE" ]; then
    EMB_SIZE=$(du -h "$EMBEDDINGS_FILE" | cut -f1)
    echo "Found response embeddings:"
    echo "  File: $EMBEDDINGS_FILE"
    echo "  Size: $EMB_SIZE"
else
    echo "MISSING: $EMBEDDINGS_FILE"
    MISSING_BANK=$((MISSING_BANK + 1))
fi

if [ $MISSING_BANK -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_BANK response bank file(s) missing!"
    echo "Please run response bank creation first (build_response_bank.py)"
    exit 1
fi

echo ""
echo "Response bank complete"
echo ""

# Check test data
echo "--------------------------------------------------------------------------------"
echo "Checking test data..."
echo "--------------------------------------------------------------------------------"

TEST_FILE="triplets_output/test_triplets.jsonl"

if [ ! -f "$TEST_FILE" ]; then
    TEST_FILE="triplets/test_triplets.jsonl"
    if [ ! -f "$TEST_FILE" ]; then
        echo "ERROR: Test triplets file not found!"
        echo "Tried:"
        echo "  - triplets_output/test_triplets.jsonl (expected location)"
        echo "  - triplets/test_triplets.jsonl (alternative location)"
        exit 1
    fi
fi

TEST_LINES=$(wc -l < "$TEST_FILE")
TEST_SIZE=$(du -h "$TEST_FILE" | cut -f1)
echo "Found test data:"
echo "  File:     $TEST_FILE"
echo "  Contexts: $TEST_LINES"
echo "  Size:     $TEST_SIZE"
echo ""

# Display Python package versions
echo "--------------------------------------------------------------------------------"
echo "Python package versions:"
echo "--------------------------------------------------------------------------------"
python -c "import torch; print(f'torch:         {torch.__version__}')" 2>/dev/null || echo "torch: NOT FOUND"
python -c "import transformers; print(f'transformers:  {transformers.__version__}')" 2>/dev/null || echo "transformers: NOT FOUND"
python -c "import numpy; print(f'numpy:         {numpy.__version__}')" 2>/dev/null || echo "numpy: NOT FOUND"
python -c "import sklearn; print(f'scikit-learn:  {sklearn.__version__}')" 2>/dev/null || echo "scikit-learn: NOT FOUND"
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
    print("WARNING: No GPU detected!")
EOF
echo ""

# Display checkpoint information
echo "--------------------------------------------------------------------------------"
echo "Model checkpoint details:"
echo "--------------------------------------------------------------------------------"
python << 'EOF'
import torch

# Dual encoder
try:
    checkpoint = torch.load('dual_encoder_checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    print("Dual Encoder:")
    print(f"  Epoch:          {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val loss:       {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print("")
except Exception as e:
    print(f"ERROR loading dual encoder checkpoint: {e}")

# Cross encoder
try:
    checkpoint = torch.load('cross_encoder_checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    print("Cross Encoder:")
    print(f"  Epoch:          {checkpoint.get('epoch', 'N/A')}")
    print(f"  MRR:            {checkpoint.get('best_mrr', 'N/A'):.4f}")
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        print(f"  Val Recall@1:   {metrics.get('recall@1', 'N/A'):.4f}")
        print(f"  Val Recall@3:   {metrics.get('recall@3', 'N/A'):.4f}")
    print("")
except Exception as e:
    print(f"ERROR loading cross encoder checkpoint: {e}")
EOF

# Record start time
JOB_START_TIME=$SECONDS

# Run test evaluation
echo "================================================================================"
echo "STARTING END-TO-END TEST EVALUATION"
echo "================================================================================"
echo ""
echo "System Configuration:"
echo "  Stage 1: Dual Encoder (fast retrieval from 153K responses)"
echo "  Stage 2: Cross Encoder (precise re-ranking of top-12)"
echo ""
echo "Evaluation Configuration:"
echo "  Test contexts:  $TEST_LINES"
echo "  Top-K for rerank: 12"
echo "  Metrics:        MRR, Recall@K, Precision@K"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

EVAL_START=$SECONDS

python code/evaluate_test_set.py

EVAL_EXIT_CODE=$?
EVAL_END=$SECONDS
EVAL_RUNTIME=$((EVAL_END - EVAL_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Test evaluation failed!"
    echo "================================================================================"
    echo "Exit code: $EVAL_EXIT_CODE"
    echo "Runtime:   ${EVAL_RUNTIME}s"
    echo "Check logs above for error details."
    exit 1
fi

echo "Test evaluation completed successfully"

EVAL_MINUTES=$((EVAL_RUNTIME / 60))
EVAL_SECONDS=$((EVAL_RUNTIME % 60))
echo "Runtime: ${EVAL_MINUTES}m ${EVAL_SECONDS}s"

# Verify outputs
echo ""
echo "--------------------------------------------------------------------------------"
echo "Verifying evaluation outputs..."
echo "--------------------------------------------------------------------------------"

if [ -f "test_set_results.json" ]; then
    echo "Found: test_set_results.json"
    echo ""
    echo "Key Metrics:"
    python << 'EOF'
import json
try:
    with open('test_set_results.json', 'r') as f:
        metrics = json.load(f)
    print(f"  Contexts evaluated:        {metrics.get('num_contexts', 'N/A'):,}")
    print(f"  Dual encoder failures:     {metrics.get('dual_encoder_failures', 'N/A'):,} ({metrics.get('dual_encoder_failure_rate', 0)*100:.2f}%)")
    print(f"")
    print(f"  MRR (Mean Reciprocal Rank): {metrics.get('mrr', 'N/A'):.4f}")
    print(f"")
    print(f"  Recall@1:                  {metrics.get('recall@1', 'N/A'):.4f} ({metrics.get('recall@1', 0)*100:.2f}%)")
    print(f"  Recall@3:                  {metrics.get('recall@3', 'N/A'):.4f} ({metrics.get('recall@3', 0)*100:.2f}%)")
    print(f"  Recall@5:                  {metrics.get('recall@5', 'N/A'):.4f} ({metrics.get('recall@5', 0)*100:.2f}%)")
    print(f"  Recall@10:                 {metrics.get('recall@10', 'N/A'):.4f} ({metrics.get('recall@10', 0)*100:.2f}%)")
except Exception as e:
    print(f"ERROR reading results: {e}")
EOF
else
    echo "WARNING: test_set_results.json not found"
fi

# Job summary
JOB_END_TIME=$SECONDS
TOTAL_RUNTIME=$((JOB_END_TIME - JOB_START_TIME))
TOTAL_HOURS=$((TOTAL_RUNTIME / 3600))
REMAINING_SECONDS=$((TOTAL_RUNTIME % 3600))
TOTAL_MINUTES=$((REMAINING_SECONDS / 60))
TOTAL_SECONDS=$((REMAINING_SECONDS % 60))

echo ""
echo ""
echo "================================================================================"
echo "JOB COMPLETED SUCCESSFULLY"
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
echo "  - test_set_results.json (comprehensive metrics)"
echo ""
echo "Review Results:"
echo "  cat test_set_results.json | python -m json.tool"
echo ""
echo "Next Steps:"
echo "  1. Review test_set_results.json for performance metrics"
echo "  2. Compare test MRR with validation MRR (0.980)"
echo "  3. If MRR > 0.97: System is production-ready"
echo "  4. If MRR < 0.90: Investigate failures and iterate"
echo "  5. Consider: Deploy system OR add seq2seq refinement"
echo ""
echo "================================================================================"
echo "End-to-end test evaluation complete"
echo "================================================================================"
