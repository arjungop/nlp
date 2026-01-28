#!/bin/bash
#SBATCH --job-name=improved_dual_encoder
#SBATCH --output=logs_improved/improved_training_%j.out
#SBATCH --error=logs_improved/improved_training_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

echo "================================================================================"
echo "IMPROVED DUAL ENCODER TRAINING JOB"
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

eval "$(conda shell.bash hook)"

echo ""
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

echo "--------------------------------------------------------------------------------"
echo "Creating required directories..."
echo "--------------------------------------------------------------------------------"

mkdir -p logs_improved
mkdir -p checkpoints_improved
mkdir -p mined_negatives
mkdir -p response_bank
mkdir -p indices

echo "Directories created successfully"
echo ""

echo "--------------------------------------------------------------------------------"
echo "Checking required files..."
echo "--------------------------------------------------------------------------------"

REQUIRED_FILES=(
    "code/model.py"
    "code/build_response_bank_v2.py"
    "code/prepare_val_test_ids.py"
    "code/build_retrieval_indices.py"
    "code/hybrid_retriever.py"
    "code/mine_hard_negatives_v2.py"
    "code/dataset_improved.py"
    "code/train_dual_encoder_improved.py"
    "tamil_dialogues_clean.jsonl"
    "triplets_output/train_triplets.jsonl"
    "triplets_output/val_triplets.jsonl"
    "triplets_output/test_triplets.jsonl"
)

MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "MISSING: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "Found: $file"
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_FILES required file(s) missing!"
    exit 1
fi

echo ""
echo "All required files present"
echo ""

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

JOB_START_TIME=$SECONDS

echo "================================================================================"
echo "STEP 1: BUILDING RESPONSE BANK"
echo "================================================================================"
echo ""
echo "Script: code/build_response_bank_v2.py"
echo "Input:  tamil_dialogues_clean.jsonl"
echo "Output: response_bank/response_bank.jsonl"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

STEP1_START=$SECONDS
python code/build_response_bank_v2.py
STEP1_EXIT=$?
STEP1_END=$SECONDS
STEP1_RUNTIME=$((STEP1_END - STEP1_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $STEP1_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Step 1 failed!"
    echo "Exit code: $STEP1_EXIT"
    exit 1
fi

echo "Step 1 completed successfully"
echo "Runtime: ${STEP1_RUNTIME}s"
echo ""

if [ -f "response_bank/response_bank.jsonl" ]; then
    BANK_LINES=$(wc -l < "response_bank/response_bank.jsonl")
    BANK_SIZE=$(du -h "response_bank/response_bank.jsonl" | cut -f1)
    echo "Response bank created:"
    echo "  File:      response_bank/response_bank.jsonl"
    echo "  Responses: $BANK_LINES"
    echo "  Size:      $BANK_SIZE"
else
    echo "ERROR: Response bank file not created!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 2: PREPARING VAL/TEST IDS"
echo "================================================================================"
echo ""
echo "Script: code/prepare_val_test_ids.py"
echo "Input:  triplets_output/val_triplets.jsonl, triplets_output/test_triplets.jsonl"
echo "Output: val_test_ids.json"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

STEP2_START=$SECONDS
python code/prepare_val_test_ids.py
STEP2_EXIT=$?
STEP2_END=$SECONDS
STEP2_RUNTIME=$((STEP2_END - STEP2_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $STEP2_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Step 2 failed!"
    echo "Exit code: $STEP2_EXIT"
    exit 1
fi

echo "Step 2 completed successfully"
echo "Runtime: ${STEP2_RUNTIME}s"
echo ""

if [ -f "val_test_ids.json" ]; then
    echo "Exclusion list created: val_test_ids.json"
else
    echo "ERROR: val_test_ids.json not created!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 3: BUILDING RETRIEVAL INDICES"
echo "================================================================================"
echo ""
echo "Script: code/build_retrieval_indices.py"
echo "Input:  response_bank/response_bank.jsonl"
echo "Output: indices/bm25_index.pkl, indices/faiss_index.bin"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

STEP3_START=$SECONDS
python code/build_retrieval_indices.py
STEP3_EXIT=$?
STEP3_END=$SECONDS
STEP3_RUNTIME=$((STEP3_END - STEP3_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $STEP3_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Step 3 failed!"
    echo "Exit code: $STEP3_EXIT"
    exit 1
fi

echo "Step 3 completed successfully"
echo "Runtime: ${STEP3_RUNTIME}s"
echo ""

if [ -f "indices/bm25_index.pkl" ] && [ -f "indices/faiss_index.bin" ]; then
    echo "Indices created successfully:"
    echo "  BM25:  indices/bm25_index.pkl"
    echo "  FAISS: indices/faiss_index.bin"
else
    echo "ERROR: Index files not created!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 4: TRAINING WITH HARD NEGATIVE MINING"
echo "================================================================================"
echo ""
echo "Script: code/train_dual_encoder_improved.py"
echo "Model:  Dual Encoder with 3-stage curriculum + hard negative mining"
echo ""
echo "Training configuration:"
echo "  Epochs:         20"
echo "  Batch size:     32"
echo "  Stage 1 (1-3):  Warmup - freeze encoders, 5 BM25 negatives"
echo "  Stage 2 (4-12): Mining - mine at 4,6,8,10,12, 30 negatives"
echo "  Stage 3 (13-20): Intensive - mine every epoch, 30 negatives"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

STEP4_START=$SECONDS
python code/train_dual_encoder_improved.py
STEP4_EXIT=$?
STEP4_END=$SECONDS
STEP4_RUNTIME=$((STEP4_END - STEP4_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $STEP4_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed!"
    echo "Exit code: $STEP4_EXIT"
    exit 1
fi

echo "Training completed successfully"

STEP4_HOURS=$((STEP4_RUNTIME / 3600))
STEP4_MINS=$(( (STEP4_RUNTIME % 3600) / 60 ))
STEP4_SECS=$(( (STEP4_RUNTIME % 3600) % 60 ))
echo "Runtime: ${STEP4_HOURS}h ${STEP4_MINS}m ${STEP4_SECS}s"

echo ""
echo "--------------------------------------------------------------------------------"
echo "Verifying training outputs..."
echo "--------------------------------------------------------------------------------"

if [ -d "checkpoints_improved" ]; then
    echo "Checkpoint directory created"
    
    CHECKPOINT_COUNT=$(ls -1 checkpoints_improved/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "  Epoch checkpoints: $CHECKPOINT_COUNT files"
    fi
else
    echo "WARNING: Checkpoint directory not found"
fi

echo ""

if [ -d "mined_negatives" ]; then
    MINED_COUNT=$(ls -1 mined_negatives/mined_negatives_epoch_*.json 2>/dev/null | wc -l)
    if [ $MINED_COUNT -gt 0 ]; then
        echo "Mined negatives: $MINED_COUNT files"
    fi
fi

echo ""

if [ -d "logs_improved" ]; then
    LOG_COUNT=$(ls -1 logs_improved/epoch_*_metrics.json 2>/dev/null | wc -l)
    if [ $LOG_COUNT -gt 0 ]; then
        echo "Training logs: $LOG_COUNT epoch metric files"
    fi
fi

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
echo "  Step 1 (Response bank):  ${STEP1_RUNTIME}s"
echo "  Step 2 (Val/test IDs):   ${STEP2_RUNTIME}s"
echo "  Step 3 (Build indices):  ${STEP3_RUNTIME}s"
echo "  Step 4 (Training):       ${STEP4_HOURS}h ${STEP4_MINS}m ${STEP4_SECS}s"
if [ $TOTAL_HOURS -gt 0 ]; then
    echo "  Total runtime:           ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
else
    echo "  Total runtime:           ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
fi
echo ""
echo "Output Files:"
echo "  response_bank/response_bank.jsonl"
echo "  val_test_ids.json"
echo "  indices/bm25_index.pkl"
echo "  indices/faiss_index.bin"
echo "  checkpoints_improved/checkpoint_epoch_*.pt"
echo "  mined_negatives/mined_negatives_epoch_*.json"
echo "  logs_improved/epoch_*_metrics.json"
echo ""
echo "Next Steps:"
echo "  1. Check logs_improved/ for training metrics"
echo "  2. Verify checkpoints exist"
echo "  3. Proceed to evaluation"
echo ""
echo "================================================================================"
echo "Improved dual encoder training complete!"
echo "================================================================================"
                                                                                                                                                                                                                                           
