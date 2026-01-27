import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

print("Checking imports...")
try:
    from transformers import get_linear_schedule_with_warmup
    print("transformers imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import transformers: {e}")
    print("Please install: pip install transformers")
    sys.exit(1)

try:
    from model import DualEncoder
    from dataset import TripletDataset, collate_fn
    from utils import compute_recall_at_k, compute_mrr, format_time
    print("Local modules imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import local modules: {e}")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_PATH = 'triplets_output/train_triplets.jsonl'
VAL_PATH = 'triplets_output/val_triplets.jsonl'
CHECKPOINT_DIR = 'dual_encoder_checkpoints'
LOG_DIR = 'training_logs'

# Model hyperparameters
OUTPUT_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.1
TEMPERATURE = 0.07

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
MAX_SEQ_LEN = 256
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Logging
LOG_INTERVAL = 100


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories():
    """Create checkpoint and log directories"""
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(LOG_DIR).mkdir(exist_ok=True)
    print(f" Checkpoint directory: {CHECKPOINT_DIR}")
    print(f" Log directory: {LOG_DIR}")


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    """Save model checkpoint"""
    print(f"  Saving checkpoint to {path}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)
    print(f" Checkpoint saved")


def validate(model, val_loader, device):
    """
    Run validation and compute metrics
    Returns dict with loss, recall@k, and MRR
    """
    print("\n" + "─" * 80)
    print("VALIDATION")
    print("─" * 80)
    
    model.eval()
    total_loss = 0
    all_logits = []
    
    val_start = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                loss, logits = model(
                    batch['context'],
                    batch['positive'],
                    batch['negatives'],
                    max_length=MAX_SEQ_LEN,
                    device=device
                )
                total_loss += loss.item()
                all_logits.append(logits)
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Validated {batch_idx + 1}/{len(val_loader)} batches", end='\r')
            except Exception as e:
                print(f"\n ERROR in validation batch {batch_idx}: {e}")
                raise
    
    print(f"\n  Validated all {len(val_loader)} batches")
    
    avg_loss = total_loss / len(val_loader)
    
    # Compute metrics
    print("  Computing metrics...")
    all_logits = torch.cat(all_logits, dim=0)
    recall_1 = compute_recall_at_k(all_logits, k=1)
    recall_5 = compute_recall_at_k(all_logits, k=5)
    recall_10 = compute_recall_at_k(all_logits, k=10)
    mrr = compute_mrr(all_logits)
    
    val_time = time.time() - val_start
    print(f"  Validation completed in {format_time(val_time)}")
    
    return {
        'loss': avg_loss,
        'recall@1': recall_1,
        'recall@5': recall_5,
        'recall@10': recall_10,
        'mrr': mrr
    }


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    print("\n" + "─" * 80)
    print(f"TRAINING EPOCH {epoch + 1}")
    print("─" * 80)
    
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for step, batch in enumerate(train_loader):
        try:
            # Debug first batch
            if step == 0:
                print(f"\nFirst batch structure:")
                print(f"  Context: {len(batch['context'])} samples")
                print(f"  Positive: {len(batch['positive'])} samples")
                print(f"  Negatives: {len(batch['negatives'])} samples x {len(batch['negatives'][0])} negatives")
                print(f"  Example context: {batch['context'][0][:2]}...")
                print(f"  Example positive: {batch['positive'][0][:50]}...")
                print()
            
            # Forward pass
            loss, logits = model(
                batch['context'],
                batch['positive'],
                batch['negatives'],
                max_length=MAX_SEQ_LEN,
                device=device
            )
            
            # Debug first batch output
            if step == 0:
                print(f"First batch forward pass:")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits sample: {logits[0][:5]}")
                if torch.cuda.is_available():
                    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Logging
            if (step + 1) % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                avg_loss = total_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                
                # Calculate ETA
                remaining_steps = len(train_loader) - (step + 1)
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                # GPU memory
                gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB" if torch.cuda.is_available() else ""
                
                print(f"  Step {step+1:5d}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"Speed: {steps_per_sec:.2f} steps/s | "
                      f"ETA: {format_time(eta_seconds)}{gpu_mem}")
        
        except Exception as e:
            print(f"\n ERROR in training step {step}: {e}")
            print(f"  Batch context type: {type(batch['context'])}")
            print(f"  Batch context length: {len(batch['context'])}")
            raise
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    print(f"\n  Epoch completed in {format_time(epoch_time)}")
    print(f"  Average loss: {avg_loss:.4f}")
    
    return avg_loss


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("=" * 80)
    print("DUAL ENCODER TRAINING")
    print("=" * 80)
    print()
    
    # Check working directory
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Setup
    setup_directories()
    
    # Device
    print("\n" + "=" * 80)
    print("DEVICE SETUP")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print(" WARNING: No GPU available, training will be slow!")
    print()
    
    # Verify data files exist
    print("=" * 80)
    print("CHECKING DATA FILES")
    print("=" * 80)
    if not os.path.exists(TRAIN_PATH):
        print(f" ERROR: Training file not found: {TRAIN_PATH}")
        sys.exit(1)
    if not os.path.exists(VAL_PATH):
        print(f" ERROR: Validation file not found: {VAL_PATH}")
        sys.exit(1)
    print(f" Training file: {TRAIN_PATH}")
    print(f" Validation file: {VAL_PATH}")
    print()
    
    # Load datasets
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    try:
        train_dataset = TripletDataset(TRAIN_PATH)
        val_dataset = TripletDataset(VAL_PATH)
    except Exception as e:
        print(f" ERROR loading datasets: {e}")
        sys.exit(1)
    
    # Show sample
    print("\nSample triplet:")
    sample = train_dataset[0]
    print(f"  Context: {sample['context']}")
    print(f"  Positive: {sample['positive'][:100]}...")
    print(f"  Negatives: {len(sample['negatives'])} samples")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f" Train batches: {len(train_loader):,}")
        print(f" Val batches: {len(val_loader):,}")
    except Exception as e:
        print(f" ERROR creating data loaders: {e}")
        sys.exit(1)
    print()
    
    # Initialize model
    print("=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    try:
        model = DualEncoder(
            output_dim=OUTPUT_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            temperature=TEMPERATURE
        )
        print(" Model created")
        
        model = model.to(device)
        print(f" Model moved to {device}")
        
    except Exception as e:
        print(f" ERROR initializing model: {e}")
        sys.exit(1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter counts:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen:    {frozen_params:,}")
    print()
    
    # Optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    try:
        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(" Optimizer and scheduler created")
    except Exception as e:
        print(f" ERROR setting up optimizer: {e}")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Epochs:          {NUM_EPOCHS}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print(f"Warmup steps:    {warmup_steps:,} / {total_steps:,} ({WARMUP_RATIO*100:.0f}%)")
    print(f"Weight decay:    {WEIGHT_DECAY}")
    print(f"Max grad norm:   {MAX_GRAD_NORM}")
    print(f"Temperature:     {TEMPERATURE}")
    print(f"Output dim:      {OUTPUT_DIM}")
    print(f"Num layers:      {NUM_LAYERS}")
    print(f"Dropout:         {DROPOUT}")
    print(f"Max seq length:  {MAX_SEQ_LEN}")
    print()
    
    # Training loop
    best_val_loss = float('inf')
    
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    try:
        for epoch in range(NUM_EPOCHS):
            print(f"\n{'=' * 80}")
            print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
            print(f"{'=' * 80}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            
            # Validate
            val_metrics = validate(model, val_loader, device)
            
            # Print epoch summary
            print("\n" + "=" * 80)
            print(f"EPOCH {epoch + 1} SUMMARY")
            print("=" * 80)
            print(f"Train Loss:    {train_loss:.4f}")
            print(f"Val Loss:      {val_metrics['loss']:.4f}")
            print(f"Recall@1:      {val_metrics['recall@1']:.4f}")
            print(f"Recall@5:      {val_metrics['recall@5']:.4f}")
            print(f"Recall@10:     {val_metrics['recall@10']:.4f}")
            print(f"MRR:           {val_metrics['mrr']:.4f}")
            print("=" * 80)
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_path)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_path)
                print(f" NEW BEST MODEL! Val loss: {best_val_loss:.4f}")
            
            # Save metrics log
            log_path = os.path.join(LOG_DIR, f'epoch_{epoch+1}_metrics.json')
            with open(log_path, 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    **val_metrics
                }, f, indent=2)
            print(f" Metrics saved to {log_path}")
            print()
    
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n FATAL ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")
    print("=" * 80)

if __name__ == '__main__':
    main()