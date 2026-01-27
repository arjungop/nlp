#!/bin/bash
#SBATCH --job-name=indic_data_prep
#SBATCH --output=logs/preprocessing_%j.out
#SBATCH --error=logs/preprocessing_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64GB
#SBATCH --time=42:00:00

echo "================================================================================"
echo "TAMIL DIALOGUE PREPROCESSING JOB"
echo "================================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Partition:    $SLURM_JOB_PARTITION"
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
    "code/clean_tamil_dialogues.py"
    "code/create_triplets.py"
    "data/IndicDialogue/Tamil/Tamil.jsonl"
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

echo ""
echo "✓ All required files present"
echo ""

# Display Python package versions
echo "--------------------------------------------------------------------------------"
echo "Python package versions:"
echo "--------------------------------------------------------------------------------"
python -c "import numpy; print(f'numpy:              {numpy.__version__}')" 2>/dev/null || echo "numpy: NOT FOUND"
python -c "import pandas; print(f'pandas:             {pandas.__version__}')" 2>/dev/null || echo "pandas: NOT FOUND"
python -c "import rank_bm25; print(f'rank-bm25:          {rank_bm25.__version__}')" 2>/dev/null || echo "rank-bm25: NOT FOUND"
python -c "from indicnlp import __version__; print(f'indic-nlp-library:  {__version__}')" 2>/dev/null || echo "indic-nlp-library: version check failed (OK)"
echo ""

# Record start time
JOB_START_TIME=$SECONDS

# ============================================================================
# STEP 1: Data Cleaning
# ============================================================================
echo "================================================================================"
echo "STEP 1: CLEANING TAMIL DIALOGUES"
echo "================================================================================"
echo ""
echo "Script: code/clean_tamil_dialogues.py"
echo "Input:  data/IndicDialogue/Tamil/Tamil.jsonl"
echo "Output: tamil_dialogues_clean.jsonl"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

STEP1_START=$SECONDS

python code/clean_tamil_dialogues.py

STEP1_EXIT_CODE=$?
STEP1_END=$SECONDS
STEP1_RUNTIME=$((STEP1_END - STEP1_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $STEP1_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✗✗✗ ERROR: Data cleaning failed!"
    echo "================================================================================"
    echo "Exit code: $STEP1_EXIT_CODE"
    echo "Runtime:   ${STEP1_RUNTIME}s"
    echo "Check logs above for error details."
    exit 1
fi

echo "✓ Data cleaning completed successfully"
echo "Runtime: ${STEP1_RUNTIME}s"

# Verify output file
if [ -f "tamil_dialogues_clean.jsonl" ]; then
    NUM_DIALOGUES=$(wc -l < tamil_dialogues_clean.jsonl)
    FILE_SIZE=$(du -h tamil_dialogues_clean.jsonl | cut -f1)
    echo ""
    echo "Output file created:"
    echo "  File:      tamil_dialogues_clean.jsonl"
    echo "  Dialogues: $NUM_DIALOGUES"
    echo "  Size:      $FILE_SIZE"
else
    echo ""
    echo "✗✗✗ WARNING: Expected output file not found: tamil_dialogues_clean.jsonl"
fi

# ============================================================================
# STEP 2: Triplet Creation
# ============================================================================
echo ""
echo ""
echo "================================================================================"
echo "STEP 2: CREATING TRIPLETS WITH BM25 HARD NEGATIVE SAMPLING"
echo "================================================================================"
echo ""
echo "Script: code/create_triplets.py"
echo "Input:  tamil_dialogues_clean.jsonl"
echo "Output: triplets_output/"
echo ""
echo "Starting at: $(date)"
echo "--------------------------------------------------------------------------------"
echo ""

STEP2_START=$SECONDS

python code/create_triplets.py

STEP2_EXIT_CODE=$?
STEP2_END=$SECONDS
STEP2_RUNTIME=$((STEP2_END - STEP2_START))

echo ""
echo "--------------------------------------------------------------------------------"

if [ $STEP2_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✗✗✗ ERROR: Triplet creation failed!"
    echo "================================================================================"
    echo "Exit code: $STEP2_EXIT_CODE"
    echo "Runtime:   ${STEP2_RUNTIME}s"
    echo "Check logs above for error details."
    exit 1
fi

echo "✓ Triplet creation completed successfully"
echo "Runtime: ${STEP2_RUNTIME}s"

# Verify output files
echo ""
echo "Verifying output files..."

if [ -d "triplets_output" ]; then
    echo "✓ Output directory created: triplets_output/"
    echo ""
    
    SPLITS=("train" "val" "test")
    TOTAL_TRIPLETS=0
    
    for split in "${SPLITS[@]}"; do
        FILE="triplets_output/${split}_triplets.jsonl"
        if [ -f "$FILE" ]; then
            COUNT=$(wc -l < "$FILE")
            SIZE=$(du -h "$FILE" | cut -f1)
            TOTAL_TRIPLETS=$((TOTAL_TRIPLETS + COUNT))
            echo "  ✓ ${split}_triplets.jsonl"
            echo "      Triplets: $COUNT"
            echo "      Size:     $SIZE"
        else
            echo "  ✗ MISSING: ${split}_triplets.jsonl"
        fi
    done
    
    echo ""
    echo "  Total triplets: $TOTAL_TRIPLETS"
    
    # Check other output files
    echo ""
    if [ -f "triplets_output/bm25_index.pkl" ]; then
        SIZE=$(du -h "triplets_output/bm25_index.pkl" | cut -f1)
        echo "  ✓ bm25_index.pkl (Size: $SIZE)"
    else
        echo "  ✗ MISSING: bm25_index.pkl"
    fi
    
    if [ -f "triplets_output/triplet_statistics.json" ]; then
        echo "  ✓ triplet_statistics.json"
    else
        echo "  ✗ MISSING: triplet_statistics.json"
    fi
    
else
    echo "✗✗✗ WARNING: Output directory not found: triplets_output/"
fi

# ============================================================================
# Job Summary
# ============================================================================
JOB_END_TIME=$SECONDS
TOTAL_RUNTIME=$((JOB_END_TIME - JOB_START_TIME))
TOTAL_MINUTES=$((TOTAL_RUNTIME / 60))
TOTAL_SECONDS=$((TOTAL_RUNTIME % 60))

STEP1_MINUTES=$((STEP1_RUNTIME / 60))
STEP1_SECONDS=$((STEP1_RUNTIME % 60))

STEP2_MINUTES=$((STEP2_RUNTIME / 60))
STEP2_SECONDS=$((STEP2_RUNTIME % 60))

echo ""
echo ""
echo "================================================================================"
echo "JOB COMPLETED SUCCESSFULLY ✓✓✓"
echo "================================================================================"
echo ""
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "End time:      $(date)"
echo ""
echo "Timing Summary:"
echo "  Step 1 (Cleaning):         ${STEP1_MINUTES}m ${STEP1_SECONDS}s"
echo "  Step 2 (Triplet creation): ${STEP2_MINUTES}m ${STEP2_SECONDS}s"
echo "  Total runtime:             ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "Output Files:"
echo "  ✓ tamil_dialogues_clean.jsonl"
echo "  ✓ triplets_output/train_triplets.jsonl"
echo "  ✓ triplets_output/val_triplets.jsonl"
echo "  ✓ triplets_output/test_triplets.jsonl"
echo "  ✓ triplets_output/bm25_index.pkl"
echo "  ✓ triplets_output/triplet_statistics.json"
echo ""
echo "Next Steps:"
echo "  1. Verify output files"
echo "  2. Check triplet_statistics.json for dataset stats"
echo "  3. Proceed to dual encoder training"
echo ""
echo "================================================================================"
echo "All preprocessing complete! Ready for model training."
echo "================================================================================"
