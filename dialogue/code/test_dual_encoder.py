import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch
from pathlib import Path
from tqdm import tqdm

from model import DualEncoder
from dataset import TripletDataset, collate_fn
from utils import compute_recall_at_k, compute_mrr
from torch.utils.data import DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_PATH = 'triplets_output/test_triplets.jsonl'
CHECKPOINT_PATH = 'dual_encoder_checkpoints/best_model.pt'
RESULTS_DIR = 'test_results'
BATCH_SIZE = 32
MAX_SEQ_LEN = 256


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_model(checkpoint_path, device):
    """Load trained dual encoder from checkpoint"""
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()
    
    # Initialize model with same architecture as training
    model = DualEncoder(
        model_name='google/muril-base-cased',
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        temperature=0.07
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f" Model loaded successfully")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def evaluate_model(model, test_loader, device):
    """
    Run model on test set and collect all predictions
    Returns: metrics dict, detailed samples, logits tensor
    """
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Batches: {len(test_loader):,}")
    print()
    
    all_logits = []
    all_samples = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Forward pass (identical to training)
            loss, logits = model(
                batch['context'],
                batch['positive'],
                batch['negatives'],
                max_length=MAX_SEQ_LEN,
                device=device
            )
            
            all_logits.append(logits.cpu())
            
            # Store samples with predictions
            batch_size = len(batch['context'])
            for i in range(batch_size):
                all_samples.append({
                    'context': batch['context'][i],
                    'positive': batch['positive'][i],
                    'negatives': batch['negatives'][i],
                    'logits': logits[i].cpu().tolist()
                })
    
    # Concatenate all logits
    all_logits = torch.cat(all_logits, dim=0)
    
    # Compute metrics
    print("\nComputing metrics...")
    recall_1 = compute_recall_at_k(all_logits, k=1)
    recall_5 = compute_recall_at_k(all_logits, k=5)
    recall_10 = compute_recall_at_k(all_logits, k=10)
    mrr = compute_mrr(all_logits)
    
    metrics = {
        'num_samples': len(all_samples),
        'recall@1': float(recall_1),
        'recall@5': float(recall_5),
        'recall@10': float(recall_10),
        'mrr': float(mrr)
    }
    
    return metrics, all_samples, all_logits


def analyze_predictions(samples, logits):
    """
    Categorize predictions into successes and failures
    Returns: lists of successful and failed predictions with details
    """
    print("\nAnalyzing predictions...")
    
    successes = []
    failures = []
    
    for i, sample in enumerate(samples):
        sample_logits = logits[i]
        
        # Get predicted index (highest score)
        predicted_idx = torch.argmax(sample_logits).item()
        
        # Get ranking of positive (index 0)
        sorted_indices = torch.argsort(sample_logits, descending=True)
        positive_rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
        
        result = {
            'context': sample['context'],
            'positive': sample['positive'],
            'negatives': sample['negatives'],
            'predicted_idx': predicted_idx,
            'positive_rank': positive_rank,
            'logits': sample['logits']
        }
        
        if predicted_idx == 0:
            successes.append(result)
        else:
            # For failures, include what was predicted
            result['predicted_response'] = sample['negatives'][predicted_idx - 1]  # -1 because positive is at 0
            failures.append(result)
    
    return successes, failures


def print_metrics(metrics):
    """Print formatted metrics"""
    print("\n" + "=" * 80)
    print("TEST SET METRICS")
    print("=" * 80)
    print(f"\nTotal test samples: {metrics['num_samples']:,}")
    print()
    print(f"Recall@1:   {metrics['recall@1']:.4f}  ({metrics['recall@1']*100:.2f}%)")
    print(f"Recall@5:   {metrics['recall@5']:.4f}  ({metrics['recall@5']*100:.2f}%)")
    print(f"Recall@10:  {metrics['recall@10']:.4f}  ({metrics['recall@10']*100:.2f}%)")
    print(f"MRR:        {metrics['mrr']:.4f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    print(f"  - Model ranks correct response in top-1:  {metrics['recall@1']*100:.1f}% of the time")
    print(f"  - Model ranks correct response in top-5:  {metrics['recall@5']*100:.1f}% of the time")
    print(f"  - Model ranks correct response in top-10: {metrics['recall@10']*100:.1f}% of the time")


def print_examples(successes, failures, num_each=3):
    """Print example predictions for inspection"""
    
    # Successful predictions
    print("\n" + "=" * 80)
    print("SUCCESSFUL PREDICTIONS (Top-1 Correct)")
    print("=" * 80)
    
    for i, result in enumerate(successes[:num_each]):
        print(f"\n{'─' * 80}")
        print(f"Success Example {i+1}")
        print(f"{'─' * 80}")
        print(f"\nContext:")
        for j, utt in enumerate(result['context']):
            print(f"  {j+1}. {utt}")
        print(f"\nCorrect Response (Model's Top Pick):")
        print(f"  {result['positive']}")
        print(f"\nScores:")
        print(f"  Positive: {result['logits'][0]:.3f}")
        print(f"  Top negative: {max(result['logits'][1:]):.3f}")
    
    # Failed predictions
    print("\n" + "=" * 80)
    print("FAILED PREDICTIONS (Top-1 Incorrect)")
    print("=" * 80)
    
    for i, result in enumerate(failures[:num_each]):
        print(f"\n{'─' * 80}")
        print(f"Failure Example {i+1} (Correct answer ranked #{result['positive_rank']})")
        print(f"{'─' * 80}")
        print(f"\nContext:")
        for j, utt in enumerate(result['context']):
            print(f"  {j+1}. {utt}")
        print(f"\nCorrect Response:")
        print(f"  {result['positive']}")
        print(f"\nModel's Top Pick (WRONG):")
        print(f"  {result['predicted_response']}")
        print(f"\nScores:")
        print(f"  Predicted (wrong): {result['logits'][result['predicted_idx']]:.3f}")
        print(f"  Correct:           {result['logits'][0]:.3f}")
        print(f"  Difference:        {result['logits'][result['predicted_idx']] - result['logits'][0]:.3f}")


def save_results(metrics, samples, successes, failures, results_dir):
    """Save all results to disk"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save summary metrics
    metrics_path = results_dir / 'test_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f" Metrics: {metrics_path}")
    
    # Save detailed predictions
    details_path = results_dir / 'detailed_predictions.jsonl'
    with open(details_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f" All predictions: {details_path} ({len(samples):,} samples)")
    
    # Save successes
    success_path = results_dir / 'successful_predictions.jsonl'
    with open(success_path, 'w', encoding='utf-8') as f:
        for item in successes:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f" Successes: {success_path} ({len(successes):,} samples)")
    
    # Save failures
    failure_path = results_dir / 'failed_predictions.jsonl'
    with open(failure_path, 'w', encoding='utf-8') as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f" Failures: {failure_path} ({len(failures):,} samples)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("DUAL ENCODER TEST EVALUATION")
    print("=" * 80)
    print()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Verify files exist
    print("Checking required files...")
    if not os.path.exists(TEST_PATH):
        print(f" ERROR: Test file not found: {TEST_PATH}")
        sys.exit(1)
    print(f" Test data: {TEST_PATH}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f" ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        print(f"  Make sure training has completed and best_model.pt exists")
        sys.exit(1)
    print(f" Checkpoint: {CHECKPOINT_PATH}")
    print()
    
    # Load model
    model = load_model(CHECKPOINT_PATH, device)
    
    # Load test dataset
    print("\n" + "=" * 80)
    print("LOADING TEST DATASET")
    print("=" * 80)
    test_dataset = TripletDataset(TEST_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Run evaluation
    metrics, samples, logits = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print_metrics(metrics)
    
    # Analyze predictions
    successes, failures = analyze_predictions(samples, logits)
    
    print("\n" + "=" * 80)
    print("PREDICTION BREAKDOWN")
    print("=" * 80)
    print(f"Successful (top-1 correct): {len(successes):,} ({len(successes)/len(samples)*100:.2f}%)")
    print(f"Failed (top-1 wrong):       {len(failures):,} ({len(failures)/len(samples)*100:.2f}%)")
    
    # Show examples
    print_examples(successes, failures, num_each=3)
    
    # Save all results
    save_results(metrics, samples, successes, failures, RESULTS_DIR)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {RESULTS_DIR}/")
    print(f"\nKey metric: Recall@1 = {metrics['recall@1']*100:.2f}%")
    print()


if __name__ == '__main__':
    main()
                                                                                                                                                                                                                                           