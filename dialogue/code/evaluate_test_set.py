import sys
import json
import torch
from pathlib import Path
from datetime import datetime
import numpy as np

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import DialogueResponseRanker, log


def load_test_triplets(triplets_path):
    """Load test triplets with logging"""
    log("="*80)
    log("LOADING TEST TRIPLETS")
    log("="*80)
    log(f"Path: {triplets_path}")
    
    try:
        triplets = []
        with open(triplets_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    triplet = json.loads(line)
                    triplets.append(triplet)
                except json.JSONDecodeError as e:
                    log(f"  WARNING: Skipping line {line_num} due to JSON error: {e}")
        
        log(f"SUCCESS: Loaded {len(triplets):,} test triplets")
        
        # Show sample
        if len(triplets) > 0:
            log("")
            log("Sample triplet:")
            sample = triplets[0]
            log(f"  Context ({len(sample['context'])} utterances):")
            for i, utt in enumerate(sample['context'][:3], 1):
                preview = utt[:60] + '...' if len(utt) > 60 else utt
                log(f"    [{i}] {preview}")
            if len(sample['context']) > 3:
                log(f"    ... ({len(sample['context'])-3} more)")
            
            pos_preview = sample['positive'][:60] + '...' if len(sample['positive']) > 60 else sample['positive']
            log(f"  Positive: {pos_preview}")
            
            neg_preview = sample['negatives'][0][:60] + '...' if len(sample['negatives'][0]) > 60 else sample['negatives'][0]
            log(f"  Negatives: {len(sample['negatives'])} (first: {neg_preview})")
        
        log("")
        return triplets
        
    except FileNotFoundError:
        log(f"ERROR: File not found: {triplets_path}")
        raise
    except Exception as e:
        log(f"ERROR loading triplets: {e}")
        raise


def evaluate_end_to_end(ranker, triplets, top_k_values=[1, 3, 5, 10]):
    """
    Evaluate complete retrieval system on test set with detailed logging
    
    Args:
        ranker: DialogueResponseRanker instance
        triplets: List of test triplets
        top_k_values: List of K values for Recall@K and Precision@K
    
    Returns:
        dict with comprehensive metrics
    """
    
    log("="*80)
    log("STARTING END-TO-END EVALUATION")
    log("="*80)
    log(f"Number of test contexts: {len(triplets):,}")
    log(f"Evaluation metrics: MRR, Recall@K, Precision@K for K in {top_k_values}")
    log("")
    
    # Initialize tracking
    mrr_scores = []
    recall_at_k = {k: [] for k in top_k_values}
    precision_at_k = {k: [] for k in top_k_values}
    
    dual_encoder_failures = 0
    rank_distribution = {1: 0, 2: 0, 3: 0, '4-5': 0, '6-10': 0, '11+': 0}
    
    log("Processing contexts...")
    log("-"*80)
    
    # Process each triplet
    for idx, triplet in enumerate(triplets):
        if (idx + 1) % 1000 == 0:
            log(f"  Processed {idx+1:,} / {len(triplets):,} contexts ({(idx+1)/len(triplets)*100:.1f}%)")
        
        context = triplet['context']
        true_response = triplet['positive']
        
        try:
            # Get predictions (verbose=False for speed)
            results = ranker.rank_responses(
                context, 
                return_top_n=12, 
                return_scores=False,
                verbose=False
            )
            
            # Find rank of true response
            rank = None
            for i, result in enumerate(results, 1):
                if result['response'].strip() == true_response.strip():
                    rank = i
                    break
            
            # Track results
            if rank is None:
                dual_encoder_failures += 1
                rank = 999
                rank_distribution['11+'] += 1
            else:
                if rank == 1:
                    rank_distribution[1] += 1
                elif rank == 2:
                    rank_distribution[2] += 1
                elif rank == 3:
                    rank_distribution[3] += 1
                elif rank <= 5:
                    rank_distribution['4-5'] += 1
                elif rank <= 10:
                    rank_distribution['6-10'] += 1
                else:
                    rank_distribution['11+'] += 1
            
            # MRR
            reciprocal_rank = 1.0 / rank if rank <= max(top_k_values) else 0.0
            mrr_scores.append(reciprocal_rank)
            
            # Recall@K and Precision@K
            for k in top_k_values:
                is_in_top_k = (rank is not None and rank <= k)
                recall_at_k[k].append(1.0 if is_in_top_k else 0.0)
                precision_at_k[k].append(1.0/k if is_in_top_k else 0.0)
        
        except Exception as e:
            log(f"  ERROR processing context {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Count as failure
            dual_encoder_failures += 1
            mrr_scores.append(0.0)
            for k in top_k_values:
                recall_at_k[k].append(0.0)
                precision_at_k[k].append(0.0)
    
    log(f"  Processed all {len(triplets):,} contexts")
    log("")
    
    # Compute final metrics
    log("Computing final metrics...")
    
    metrics = {
        'num_contexts': len(triplets),
        'dual_encoder_failures': dual_encoder_failures,
        'dual_encoder_failure_rate': dual_encoder_failures / len(triplets),
        'mrr': float(np.mean(mrr_scores)),
        'rank_distribution': rank_distribution
    }
    
    for k in top_k_values:
        metrics[f'recall@{k}'] = float(np.mean(recall_at_k[k]))
        metrics[f'precision@{k}'] = float(np.mean(precision_at_k[k]))
    
    log("Metrics computed successfully")
    log("")
    
    return metrics


def print_results(metrics):
    """Print results in formatted manner"""
    log("="*80)
    log("EVALUATION RESULTS")
    log("="*80)
    log("")
    
    log("DATASET STATISTICS:")
    log(f"  Total contexts evaluated: {metrics['num_contexts']:,}")
    log(f"  Dual encoder failures: {metrics['dual_encoder_failures']:,} ({metrics['dual_encoder_failure_rate']*100:.2f}%)")
    log("")
    
    log("RANK DISTRIBUTION:")
    log(f"  Rank 1: {metrics['rank_distribution'][1]:,} ({metrics['rank_distribution'][1]/metrics['num_contexts']*100:.2f}%)")
    log(f"  Rank 2: {metrics['rank_distribution'][2]:,} ({metrics['rank_distribution'][2]/metrics['num_contexts']*100:.2f}%)")
    log(f"  Rank 3: {metrics['rank_distribution'][3]:,} ({metrics['rank_distribution'][3]/metrics['num_contexts']*100:.2f}%)")
    log(f"  Rank 4-5: {metrics['rank_distribution']['4-5']:,} ({metrics['rank_distribution']['4-5']/metrics['num_contexts']*100:.2f}%)")
    log(f"  Rank 6-10: {metrics['rank_distribution']['6-10']:,} ({metrics['rank_distribution']['6-10']/metrics['num_contexts']*100:.2f}%)")
    log(f"  Rank 11+: {metrics['rank_distribution']['11+']:,} ({metrics['rank_distribution']['11+']/metrics['num_contexts']*100:.2f}%)")
    log("")
    
    log("PRIMARY RANKING METRICS:")
    log(f"  MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
    log("")
    
    log("RECALL METRICS:")
    for k in [1, 3, 5, 10]:
        if f'recall@{k}' in metrics:
            log(f"  Recall@{k:2d}: {metrics[f'recall@{k}']:.4f} ({metrics[f'recall@{k}']*100:.2f}%)")
    log("")
    
    log("PRECISION METRICS:")
    for k in [1, 3, 5, 10]:
        if f'precision@{k}' in metrics:
            log(f"  Precision@{k:2d}: {metrics[f'precision@{k}']:.4f}")
    log("")


def main():
    log("="*80)
    log("END-TO-END TEST SET EVALUATION")
    log("="*80)
    log("")
    
    # Configuration
    DUAL_ENCODER_CHECKPOINT = 'dual_encoder_checkpoints/best_model.pt'
    CROSS_ENCODER_CHECKPOINT = 'cross_encoder_checkpoints/best_model.pt'
    RESPONSE_BANK_PATH = 'response_bank/responses.txt'
    RESPONSE_EMBEDDINGS_PATH = 'response_bank/embeddings.pt'
    TEST_TRIPLETS_PATH = 'triplets_output/test_triplets.jsonl'
    OUTPUT_PATH = 'test_set_results.json'
    
    log("CONFIGURATION:")
    log(f"  Dual encoder checkpoint: {DUAL_ENCODER_CHECKPOINT}")
    log(f"  Cross encoder checkpoint: {CROSS_ENCODER_CHECKPOINT}")
    log(f"  Response bank: {RESPONSE_BANK_PATH}")
    log(f"  Response embeddings: {RESPONSE_EMBEDDINGS_PATH}")
    log(f"  Test triplets: {TEST_TRIPLETS_PATH}")
    log(f"  Output file: {OUTPUT_PATH}")
    log("")
    
    # Load ranker
    try:
        ranker = DialogueResponseRanker(
            dual_encoder_checkpoint=DUAL_ENCODER_CHECKPOINT,
            cross_encoder_checkpoint=CROSS_ENCODER_CHECKPOINT,
            response_bank_path=RESPONSE_BANK_PATH,
            response_embeddings_path=RESPONSE_EMBEDDINGS_PATH,
            top_k=12,
            device='cuda',
            verbose=False
        )
    except Exception as e:
        log(f"FATAL ERROR: Failed to initialize ranker: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load test data
    try:
        test_triplets = load_test_triplets(TEST_TRIPLETS_PATH)
    except Exception as e:
        log(f"FATAL ERROR: Failed to load test triplets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate
    try:
        metrics = evaluate_end_to_end(ranker, test_triplets, top_k_values=[1, 3, 5, 10])
    except Exception as e:
        log(f"FATAL ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print results
    print_results(metrics)
    
    # Save results
    log("="*80)
    log("SAVING RESULTS")
    log("="*80)
    
    try:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        log(f"SUCCESS: Results saved to {OUTPUT_PATH}")
    except Exception as e:
        log(f"ERROR: Failed to save results: {e}")
    
    log("")
    log("="*80)
    log("EVALUATION COMPLETE")
    log("="*80)


if __name__ == '__main__':
    main()
                                                                                                                                                                                                                                           