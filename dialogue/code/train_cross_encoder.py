import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from cross_encoder_model import CrossEncoder
from cross_encoder_dataset import CrossEncoderDataset, collate_fn


# Configuration
TRAIN_PATH = 'cross_encoder_data/train_pairs.jsonl'
VAL_PATH = 'cross_encoder_data/val_pairs.jsonl'
CHECKPOINT_DIR = 'cross_encoder_checkpoints'
LOG_DIR = 'cross_encoder_logs'

# Hyperparameters
DROPOUT = 0.1
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 384
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
POS_WEIGHT = 12.0  # 12 negatives per positive

LOG_INTERVAL = 500
PAIRS_PER_CONTEXT = 13  # 1 positive + 12 negatives


def setup_directories():
    """Create directories"""
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(LOG_DIR).mkdir(exist_ok=True)


def compute_ranking_metrics(scores, labels):
    """
    Compute ranking metrics for grouped pairs (1 positive + 12 negatives per context)
    
    Args:
        scores: numpy array of predicted scores
        labels: numpy array of true labels (0 or 1)
    
    Returns:
        dict with MRR, Recall@K, Precision@K metrics
    """
    num_pairs = len(scores)
    num_contexts = num_pairs // PAIRS_PER_CONTEXT
    
    mrr_scores = []
    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    precision_at_1 = []
    precision_at_3 = []
    precision_at_5 = []
    
    for i in range(num_contexts):
        start_idx = i * PAIRS_PER_CONTEXT
        end_idx = start_idx + PAIRS_PER_CONTEXT
        
        # Get scores and labels for this context
        context_scores = scores[start_idx:end_idx]
        context_labels = labels[start_idx:end_idx]
        
        # Find the index of the positive response
        pos_idx = np.where(context_labels == 1)[0]
        
        if len(pos_idx) != 1:
            # Skip if not exactly 1 positive (shouldn't happen)
            continue
        
        pos_idx = pos_idx[0]
        
        # Rank responses by score (descending)
        ranked_indices = np.argsort(-context_scores)
        
        # Find rank of positive response (1-indexed)
        rank = np.where(ranked_indices == pos_idx)[0][0] + 1
        
        # MRR
        mrr_scores.append(1.0 / rank)
        
        # Recall@K
        recall_at_1.append(1.0 if rank <= 1 else 0.0)
        recall_at_3.append(1.0 if rank <= 3 else 0.0)
        recall_at_5.append(1.0 if rank <= 5 else 0.0)
        
        # Precision@K (for binary relevance, this is 1/K if positive is in top-K)
        precision_at_1.append(1.0 if rank <= 1 else 0.0)
        precision_at_3.append(1.0/3.0 if rank <= 3 else 0.0)
        precision_at_5.append(1.0/5.0 if rank <= 5 else 0.0)
    
    return {
        'mrr': np.mean(mrr_scores),
        'recall@1': np.mean(recall_at_1),
        'recall@3': np.mean(recall_at_3),
        'recall@5': np.mean(recall_at_5),
        'precision@1': np.mean(precision_at_1),
        'precision@3': np.mean(precision_at_3),
        'precision@5': np.mean(precision_at_5)
    }


def validate(model, val_loader, device):
    """Run validation with comprehensive metrics"""
    print("\nValidating...")
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            logits = model(batch['context'], batch['response'], max_length=MAX_SEQ_LEN)
            labels = torch.tensor(batch['label'], dtype=torch.float32).to(device)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get probabilities
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
    
    avg_loss = total_loss / len(val_loader)
    
    # Convert to numpy
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Binary classification metrics
    auc_roc = roc_auc_score(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    
    # Binary accuracy
    binary_preds = (all_probs > 0.5).astype(int)
    accuracy = (binary_preds == all_labels).mean()
    
    # Positive recall (how many positives we catch)
    pos_indices = all_labels == 1
    if pos_indices.sum() > 0:
        pos_recall = (binary_preds[pos_indices] == 1).mean()
    else:
        pos_recall = 0.0
    
    # Ranking metrics (MRR, Recall@K, Precision@K)
    ranking_metrics = compute_ranking_metrics(all_probs, all_labels)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'avg_precision': avg_precision,
        'positive_recall': pos_recall,
        **ranking_metrics  # Add all ranking metrics
    }
    
    return metrics


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        # Forward
        logits = model(batch['context'], batch['response'], max_length=MAX_SEQ_LEN)
        labels = torch.tensor(batch['label'], dtype=torch.float32).to(device)
        
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Logging
        if (step + 1) % LOG_INTERVAL == 0:
            avg_loss = total_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
    
    return total_loss / len(train_loader)


def main():
    print("=" * 80)
    print("CROSS-ENCODER TRAINING")
    print("=" * 80)
    print()
    
    setup_directories()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load datasets
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    train_dataset = CrossEncoderDataset(TRAIN_PATH)
    val_dataset = CrossEncoderDataset(VAL_PATH)
    print()
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Initialize model
    print("=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    model = CrossEncoder(dropout=DROPOUT).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")
    print(f"Warmup steps: {warmup_steps:,} / {total_steps:,}")
    print(f"Pos weight: {POS_WEIGHT}")
    print()
    
    # Training loop
    best_mrr = 0.0
    
    for epoch in range(EPOCHS):
        print("=" * 80)
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print("=" * 80)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_metrics = validate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss:       {train_loss:.4f}")
        print(f"  Val Loss:         {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy:     {val_metrics['accuracy']:.4f}")
        print(f"  Val AUC-ROC:      {val_metrics['auc_roc']:.4f}")
        print(f"  Val Avg Precision:{val_metrics['avg_precision']:.4f}")
        print(f"  Val Pos Recall:   {val_metrics['positive_recall']:.4f}")
        print(f"\n  Ranking Metrics:")
        print(f"    MRR:            {val_metrics['mrr']:.4f}")
        print(f"    Recall@1:       {val_metrics['recall@1']:.4f}")
        print(f"    Recall@3:       {val_metrics['recall@3']:.4f}")
        print(f"    Recall@5:       {val_metrics['recall@5']:.4f}")
        print(f"    Precision@1:    {val_metrics['precision@1']:.4f}")
        print(f"    Precision@3:    {val_metrics['precision@3']:.4f}")
        print(f"    Precision@5:    {val_metrics['precision@5']:.4f}")
        
        # Save checkpoint
        checkpoint_path = Path(CHECKPOINT_DIR) / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mrr': best_mrr,
            'val_metrics': val_metrics
        }, checkpoint_path)
        
        # Save best based on MRR
        if val_metrics['mrr'] > best_mrr:
            best_mrr = val_metrics['mrr']
            best_path = Path(CHECKPOINT_DIR) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mrr': best_mrr,
                'val_metrics': val_metrics
            }, best_path)
            print(f"\n  ★ New best model! MRR: {best_mrr:.4f}")
        
        # Save metrics
        log_path = Path(LOG_DIR) / f'epoch_{epoch+1}_metrics.json'
        with open(log_path, 'w') as f:
            json.dump({'epoch': epoch+1, 'train_loss': train_loss, **val_metrics}, f, indent=2)
        
        print()
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best MRR: {best_mrr:.4f}")
    print()


if __name__ == '__main__':
    main()
                                                                                                                                                                                                                                           