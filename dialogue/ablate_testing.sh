#!/bin/bash
#SBATCH --job-name=ablation_comprehensive
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00

echo "================================================================================"
echo "COMPREHENSIVE ABLATION TESTING"
echo "================================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start time:   $(date)"
echo "================================================================================"
echo ""

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
echo "Activating conda environment..."
conda activate /dist_home/suryansh/dialogue/mtdrenv

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment!"
    exit 1
fi

echo "✓ Environment activated"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'ERROR')"
echo ""

# Verify working directory
echo "Verifying working directory..."
cd /dist_home/suryansh/dialogue

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change to /dist_home/suryansh/dialogue"
    exit 1
fi

echo "✓ Working directory: $(pwd)"
echo ""

# Create output directory
echo "Creating output directory..."
mkdir -p ablate_testing

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create ablate_testing directory"
    exit 1
fi

echo "✓ Output directory created: ablate_testing/"
echo ""

# Verify required files exist
echo "Verifying required files..."

REQUIRED_FILES=(
    "dual_encoder_checkpoints/best_model.pt"
    "cross_encoder_checkpoints/best_model.pt"
    "response_bank/responses.txt"
    "response_bank/embeddings.pt"
    "triplets_output/train_triplets.jsonl"
    "triplets_output/val_triplets.jsonl"
    "triplets_output/test_triplets.jsonl"
    "code/model.py"
    "code/cross_encoder_model.py"
)

MISSING_COUNT=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ✗ MISSING: $file"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    else
        echo "  ✓ Found: $file"
    fi
done

if [ $MISSING_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_COUNT required file(s) missing!"
    exit 1
fi

echo ""
echo "✓ All required files present"
echo ""

# Run comprehensive ablation testing
python << 'ABLATION_EOF'

import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from collections import defaultdict, Counter
from datetime import datetime
import traceback

# Add code directory to path
sys.path.insert(0, 'code')

try:
    from model import DualEncoder
    from cross_encoder_model import CrossEncoder
except ImportError as e:
    print(f"ERROR: Failed to import models: {e}")
    traceback.print_exc()
    sys.exit(1)

print("="*80)
print("COMPREHENSIVE ABLATION TESTING FRAMEWORK")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path('ablate_testing')
OUTPUT_DIR.mkdir(exist_ok=True)

# Checkpoint paths
DUAL_CHECKPOINT = 'dual_encoder_checkpoints/best_model.pt'
CROSS_CHECKPOINT = 'cross_encoder_checkpoints/best_model.pt'

# Data paths
RESPONSE_BANK_PATH = 'response_bank/responses.txt'
RESPONSE_EMBEDDINGS_PATH = 'response_bank/embeddings.pt'
TRAIN_TRIPLETS_PATH = 'triplets_output/train_triplets.jsonl'
VAL_TRIPLETS_PATH = 'triplets_output/val_triplets.jsonl'
TEST_TRIPLETS_PATH = 'triplets_output/test_triplets.jsonl'
CROSS_VAL_PAIRS_PATH = 'cross_encoder_data/val_pairs.jsonl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ============================================================================
# STEP 0: LOAD ALL DATA
# ============================================================================

print("="*80)
print("STEP 0: LOADING DATA")
print("="*80)
print()

try:
    # Load response bank
    print("Loading response bank...")
    with open(RESPONSE_BANK_PATH, 'r', encoding='utf-8') as f:
        responses = [line.strip() for line in f if line.strip()]
    response_to_idx = {resp: idx for idx, resp in enumerate(responses)}
    print(f"✓ {len(responses):,} responses")
    
    response_embeddings = torch.load(RESPONSE_EMBEDDINGS_PATH, map_location=device, weights_only=False)
    print(f"✓ Embeddings: {response_embeddings.shape}")
    print()
    
except Exception as e:
    print(f"ERROR loading response bank: {e}")
    traceback.print_exc()
    sys.exit(1)

# Load triplets
def load_triplets(path):
    triplets = []
    print(f"Loading {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        triplets.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"  Warning: Invalid JSON at line {line_num}")
        return triplets
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        traceback.print_exc()
        return []

train_triplets = load_triplets(TRAIN_TRIPLETS_PATH)
val_triplets = load_triplets(VAL_TRIPLETS_PATH)
test_triplets = load_triplets(TEST_TRIPLETS_PATH)

print(f"✓ Train: {len(train_triplets):,} triplets")
print(f"✓ Val:   {len(val_triplets):,} triplets")
print(f"✓ Test:  {len(test_triplets):,} triplets")
print()

if len(test_triplets) == 0:
    print("ERROR: No test triplets loaded!")
    sys.exit(1)

# ============================================================================
# STEP 1: CONTAMINATION ANALYSIS & CLEAN TEST SET CREATION
# ============================================================================

print("="*80)
print("STEP 1: CONTAMINATION ANALYSIS")
print("="*80)
print()

print("Collecting test utterances...")
test_utterances = set()
test_contexts = set()
test_responses = set()

for triplet in test_triplets:
    test_contexts.add(tuple(triplet['context']))
    test_responses.add(triplet['positive'])
    test_utterances.update(triplet['context'])
    test_utterances.add(triplet['positive'])

print(f"Unique test utterances: {len(test_utterances):,}")
print(f"Unique test contexts: {len(test_contexts):,}")
print(f"Unique test responses: {len(test_responses):,}")
print()

print("Analyzing training negatives...")
training_negatives = set()
test_utterance_as_negative = Counter()
test_response_as_negative = set()

for triplet in tqdm(train_triplets, desc="Scanning train negatives"):
    for neg in triplet['negatives']:
        training_negatives.add(neg)
        if neg in test_utterances:
            test_utterance_as_negative[neg] += 1
        if neg in test_responses:
            test_response_as_negative.add(neg)

contaminated_utterances = len(test_utterance_as_negative)
contaminated_responses = len(test_response_as_negative)

print()
print(f"Training negatives (unique): {len(training_negatives):,}")
print(f"Test utterances in train negatives: {contaminated_utterances:,} / {len(test_utterances):,} ({contaminated_utterances/len(test_utterances)*100:.2f}%)")
print(f"Test RESPONSES in train negatives: {contaminated_responses:,} / {len(test_responses):,} ({contaminated_responses/len(test_responses)*100:.2f}%)")
print()

# Create clean test set
print("Creating clean test set...")
clean_test_triplets = []
contaminated_test_triplets = []

for triplet in test_triplets:
    if triplet['positive'] in test_response_as_negative:
        contaminated_test_triplets.append(triplet)
    else:
        clean_test_triplets.append(triplet)

print(f"Clean test triplets: {len(clean_test_triplets):,} ({len(clean_test_triplets)/len(test_triplets)*100:.2f}%)")
print(f"Contaminated test triplets: {len(contaminated_test_triplets):,} ({len(contaminated_test_triplets)/len(test_triplets)*100:.2f}%)")
print()

# Save contamination analysis
contamination_results = {
    'total_test_utterances': len(test_utterances),
    'total_test_responses': len(test_responses),
    'contaminated_utterances': contaminated_utterances,
    'contaminated_responses': contaminated_responses,
    'contamination_rate_utterances': float(contaminated_utterances / len(test_utterances)) if len(test_utterances) > 0 else 0.0,
    'contamination_rate_responses': float(contaminated_responses / len(test_responses)) if len(test_responses) > 0 else 0.0,
    'total_test_triplets': len(test_triplets),
    'clean_test_triplets': len(clean_test_triplets),
    'contaminated_test_triplets': len(contaminated_test_triplets),
    'top_contaminated': [
        {'utterance': utt, 'count': int(count)}
        for utt, count in test_utterance_as_negative.most_common(20)
    ]
}

with open(OUTPUT_DIR / 'contamination_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(contamination_results, f, indent=2, ensure_ascii=False)

print(f"✓ Saved: {OUTPUT_DIR / 'contamination_analysis.json'}")
print()

# ============================================================================
# STEP 2: LOAD MODELS
# ============================================================================

print("="*80)
print("STEP 2: LOADING MODELS")
print("="*80)
print()

try:
    print("Loading dual encoder...")
    dual_encoder = DualEncoder(
        model_name='google/muril-base-cased',
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        temperature=0.07
    ).to(device)
    
    dual_checkpoint = torch.load(DUAL_CHECKPOINT, map_location=device, weights_only=False)
    dual_encoder.load_state_dict(dual_checkpoint['model_state_dict'])
    dual_encoder.eval()
    
    print(f"✓ Epoch: {dual_checkpoint['epoch']+1}")
    print(f"✓ Val loss: {dual_checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print()
    
except Exception as e:
    print(f"ERROR loading dual encoder: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Loading cross encoder...")
    cross_encoder = CrossEncoder(dropout=0.1).to(device)
    
    cross_checkpoint = torch.load(CROSS_CHECKPOINT, map_location=device, weights_only=False)
    cross_encoder.load_state_dict(cross_checkpoint['model_state_dict'])
    cross_encoder.eval()
    
    print(f"✓ Epoch: {cross_checkpoint['epoch']+1}")
    print(f"✓ Best MRR: {cross_checkpoint.get('best_mrr', 'N/A'):.4f}")
    print()
    
except Exception as e:
    print(f"ERROR loading cross encoder: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: ABLATION TEST 1 - DUAL ENCODER ON 13-WAY TASK
# ============================================================================

print("="*80)
print("ABLATION TEST 1: DUAL ENCODER - 13-WAY CLASSIFICATION")
print("(Testing on the task it was TRAINED for)")
print("="*80)
print()

def evaluate_dual_encoder_13way(triplets, split_name):
    """Evaluate dual encoder on original 13-way task (1 pos + 12 neg)"""
    
    if len(triplets) == 0:
        print(f"WARNING: {split_name} has 0 triplets, skipping...")
        return {
            'total': 0,
            'accuracy': 0.0,
            'recall@1': 0.0,
            'recall@3': 0.0,
            'recall@5': 0.0,
            'mean_mrr': 0.0
        }
    
    print(f"Evaluating on {split_name} ({len(triplets):,} samples)...")
    
    results = {
        'total': len(triplets),
        'correct': 0,
        'recall@1': 0,
        'recall@3': 0,
        'recall@5': 0,
        'mrr': []
    }
    
    with torch.no_grad():
        for idx, triplet in enumerate(tqdm(triplets, desc=f"{split_name} 13-way")):
            context = triplet['context']
            positive = triplet['positive']
            negatives = triplet['negatives']
            
            # Create 13-way classification
            all_candidates = [positive] + negatives
            
            # Encode
            context_emb = dual_encoder.encode_context([context], max_length=256, device=device)[0]
            candidate_embs = dual_encoder.encode_response(all_candidates, max_length=256, device=device)
            
            # Similarities
            similarities = torch.matmul(candidate_embs, context_emb)
            
            # Rank
            sorted_indices = torch.argsort(similarities, descending=True)
            rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
            
            # Metrics
            if rank == 1:
                results['correct'] += 1
                results['recall@1'] += 1
            if rank <= 3:
                results['recall@3'] += 1
            if rank <= 5:
                results['recall@5'] += 1
            
            results['mrr'].append(1.0 / rank)
            
            # Clear cache periodically
            if (idx + 1) % 1000 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Compute final metrics
    results['accuracy'] = float(results['correct'] / results['total'])
    results['recall@1'] = float(results['recall@1'] / results['total'])
    results['recall@3'] = float(results['recall@3'] / results['total'])
    results['recall@5'] = float(results['recall@5'] / results['total'])
    results['mean_mrr'] = float(np.mean(results['mrr']))
    
    return results

# Test on all splits
test_13way_full = evaluate_dual_encoder_13way(test_triplets, "Full Test")
test_13way_clean = evaluate_dual_encoder_13way(clean_test_triplets, "Clean Test")
val_13way = evaluate_dual_encoder_13way(val_triplets, "Validation")

print()
print("RESULTS - 13-WAY CLASSIFICATION:")
print("-"*80)
print(f"{'Split':<15} {'Samples':>8} {'Acc':>7} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7}")
print("-"*80)
print(f"{'Full Test':<15} {test_13way_full['total']:>8,} {test_13way_full['accuracy']:>7.4f} {test_13way_full['recall@1']:>7.4f} {test_13way_full['recall@3']:>7.4f} {test_13way_full['recall@5']:>7.4f} {test_13way_full['mean_mrr']:>7.4f}")
print(f"{'Clean Test':<15} {test_13way_clean['total']:>8,} {test_13way_clean['accuracy']:>7.4f} {test_13way_clean['recall@1']:>7.4f} {test_13way_clean['recall@3']:>7.4f} {test_13way_clean['recall@5']:>7.4f} {test_13way_clean['mean_mrr']:>7.4f}")
print(f"{'Validation':<15} {val_13way['total']:>8,} {val_13way['accuracy']:>7.4f} {val_13way['recall@1']:>7.4f} {val_13way['recall@3']:>7.4f} {val_13way['recall@5']:>7.4f} {val_13way['mean_mrr']:>7.4f}")
print()

# Save results
with open(OUTPUT_DIR / 'ablation1_dual_encoder_13way.json', 'w') as f:
    json.dump({
        'full_test': test_13way_full,
        'clean_test': test_13way_clean,
        'validation': val_13way
    }, f, indent=2)

print(f"✓ Saved: {OUTPUT_DIR / 'ablation1_dual_encoder_13way.json'}")
print()

# ============================================================================
# STEP 4: ABLATION TEST 2 - DUAL ENCODER ON 153K-WAY RETRIEVAL
# ============================================================================

print("="*80)
print("ABLATION TEST 2: DUAL ENCODER - 153K-WAY RETRIEVAL")
print("(Testing on the task we NEED it for)")
print("="*80)
print()

def evaluate_dual_encoder_retrieval(triplets, split_name, max_samples=1000):
    """Evaluate dual encoder on full 153K-way retrieval"""
    
    if len(triplets) == 0:
        print(f"WARNING: {split_name} has 0 triplets, skipping...")
        return {
            'total': 0,
            'top12_rate': 0.0,
            'top100_rate': 0.0,
            'negative_rate': 0.0,
            'mrr': 0.0
        }
    
    if max_samples and len(triplets) > max_samples:
        print(f"Evaluating on {split_name} (first {max_samples:,} of {len(triplets):,} samples)...")
        triplets = triplets[:max_samples]
    else:
        print(f"Evaluating on {split_name} ({len(triplets):,} samples)...")
    
    results = {
        'total': len(triplets),
        'negative_similarity': 0,
        'top12_success': 0,
        'top100_success': 0,
        'top1000_success': 0,
        'similarities': [],
        'ranks': [],
        'not_found': 0
    }
    
    with torch.no_grad():
        for idx, triplet in enumerate(tqdm(triplets, desc=f"{split_name} retrieval")):
            context = triplet['context']
            true_response = triplet['positive']
            
            # Encode context
            context_emb = dual_encoder.encode_context([context], max_length=256, device=device)[0]
            
            # Compute similarities with ALL responses
            similarities = torch.matmul(response_embeddings, context_emb)
            
            # Find true response
            if true_response in response_to_idx:
                true_idx = response_to_idx[true_response]
                true_sim = similarities[true_idx].item()
                
                results['similarities'].append(true_sim)
                
                if true_sim < 0:
                    results['negative_similarity'] += 1
                
                # Find rank
                all_ranks = torch.argsort(similarities, descending=True)
                rank = (all_ranks == true_idx).nonzero(as_tuple=True)[0].item() + 1
                results['ranks'].append(rank)
                
                if rank <= 12:
                    results['top12_success'] += 1
                if rank <= 100:
                    results['top100_success'] += 1
                if rank <= 1000:
                    results['top1000_success'] += 1
            else:
                results['not_found'] += 1
            
            # Clear cache periodically
            if (idx + 1) % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Compute metrics
    if results['similarities']:
        results['mean_similarity'] = float(np.mean(results['similarities']))
        results['median_similarity'] = float(np.median(results['similarities']))
        results['min_similarity'] = float(np.min(results['similarities']))
        results['max_similarity'] = float(np.max(results['similarities']))
    
    if results['ranks']:
        results['mean_rank'] = float(np.mean(results['ranks']))
        results['median_rank'] = float(np.median(results['ranks']))
        results['mrr'] = float(np.mean([1.0/r for r in results['ranks']]))
    
    results['top12_rate'] = float(results['top12_success'] / results['total'])
    results['top100_rate'] = float(results['top100_success'] / results['total'])
    results['top1000_rate'] = float(results['top1000_success'] / results['total'])
    results['negative_rate'] = float(results['negative_similarity'] / results['total'])
    
    return results

# Test on all splits
test_retrieval_full = evaluate_dual_encoder_retrieval(test_triplets, "Full Test", max_samples=1000)
test_retrieval_clean = evaluate_dual_encoder_retrieval(clean_test_triplets, "Clean Test", max_samples=min(1000, len(clean_test_triplets)))
val_retrieval = evaluate_dual_encoder_retrieval(val_triplets, "Validation", max_samples=1000)

print()
print("RESULTS - 153K-WAY RETRIEVAL:")
print("-"*80)
print(f"{'Split':<15} {'Samples':>8} {'Neg%':>7} {'Top12%':>7} {'Top100%':>8} {'MeanSim':>8} {'MRR':>9}")
print("-"*80)
print(f"{'Full Test':<15} {test_retrieval_full['total']:>8,} {test_retrieval_full['negative_rate']*100:>6.2f}% {test_retrieval_full['top12_rate']*100:>6.2f}% {test_retrieval_full['top100_rate']*100:>7.2f}% {test_retrieval_full.get('mean_similarity', 0):>8.4f} {test_retrieval_full.get('mrr', 0):>9.6f}")
print(f"{'Clean Test':<15} {test_retrieval_clean['total']:>8,} {test_retrieval_clean['negative_rate']*100:>6.2f}% {test_retrieval_clean['top12_rate']*100:>6.2f}% {test_retrieval_clean['top100_rate']*100:>7.2f}% {test_retrieval_clean.get('mean_similarity', 0):>8.4f} {test_retrieval_clean.get('mrr', 0):>9.6f}")
print(f"{'Validation':<15} {val_retrieval['total']:>8,} {val_retrieval['negative_rate']*100:>6.2f}% {val_retrieval['top12_rate']*100:>6.2f}% {val_retrieval['top100_rate']*100:>7.2f}% {val_retrieval.get('mean_similarity', 0):>8.4f} {val_retrieval.get('mrr', 0):>9.6f}")
print()

# Save results
with open(OUTPUT_DIR / 'ablation2_dual_encoder_retrieval.json', 'w') as f:
    json.dump({
        'full_test': test_retrieval_full,
        'clean_test': test_retrieval_clean,
        'validation': val_retrieval
    }, f, indent=2)

print(f"✓ Saved: {OUTPUT_DIR / 'ablation2_dual_encoder_retrieval.json'}")
print()

# ============================================================================
# STEP 5: FINAL SUMMARY
# ============================================================================

print("="*80)
print("COMPREHENSIVE ABLATION TESTING COMPLETE")
print("="*80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

print("CRITICAL FINDINGS:")
print("="*80)
print()

print("1. DATA CONTAMINATION:")
print(f"   - Test responses in training negatives: {contaminated_responses:,} / {len(test_responses):,} ({contaminated_responses/len(test_responses)*100:.1f}%)")
print(f"   - Clean test set size: {len(clean_test_triplets):,} ({len(clean_test_triplets)/len(test_triplets)*100:.1f}%)")
print()

print("2. DUAL ENCODER - 13-WAY TASK (What it was trained for):")
print(f"   Full Test:  Acc={test_13way_full['accuracy']:.4f}, MRR={test_13way_full['mean_mrr']:.4f}")
print(f"   Clean Test: Acc={test_13way_clean['accuracy']:.4f}, MRR={test_13way_clean['mean_mrr']:.4f}")
print(f"   Validation: Acc={val_13way['accuracy']:.4f}, MRR={val_13way['mean_mrr']:.4f}")
print()

print("3. DUAL ENCODER - 153K-WAY RETRIEVAL (What we need):")
print(f"   Full Test:  Top12={test_retrieval_full['top12_rate']*100:.1f}%, Neg={test_retrieval_full['negative_rate']*100:.1f}%, MeanSim={test_retrieval_full.get('mean_similarity', 0):.4f}")
print(f"   Clean Test: Top12={test_retrieval_clean['top12_rate']*100:.1f}%, Neg={test_retrieval_clean['negative_rate']*100:.1f}%, MeanSim={test_retrieval_clean.get('mean_similarity', 0):.4f}")
print(f"   Validation: Top12={val_retrieval['top12_rate']*100:.1f}%, Neg={val_retrieval['negative_rate']*100:.1f}%, MeanSim={val_retrieval.get('mean_similarity', 0):.4f}")
print()

print(f"All results saved to: {OUTPUT_DIR}/")
print()
print("="*80)

ABLATION_EOF

echo ""
echo "================================================================================"
echo "JOB COMPLETE"
echo "End time: $(date)"
echo "================================================================================"
