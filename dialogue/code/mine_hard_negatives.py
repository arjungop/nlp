import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model import DualEncoder


def load_response_bank(response_bank_dir):
    """Load response bank"""
    print("Loading response bank...")
    
    # Load responses
    responses_path = response_bank_dir / 'responses.txt'
    with open(responses_path, 'r', encoding='utf-8') as f:
        responses = [line.strip() for line in f]
    
    # Load embeddings
    embeddings_path = response_bank_dir / 'embeddings.pt'
    embeddings = torch.load(embeddings_path)
    
    print(f"✓ Loaded {len(responses):,} responses")
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    return responses, embeddings


def load_triplets(triplets_path):
    """Load triplets from JSONL"""
    triplets = []
    with open(triplets_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    return triplets


def retrieve_hard_negatives(context, positive, response_bank, response_embeddings, 
                            dual_encoder, device, k=12):
    """
    Retrieve top-k hard negatives using dual encoder
    Always includes ground-truth positive
    """
    # Encode context
    context_emb = dual_encoder.encode_context([context], max_length=256, device=device)
    
    # Compute similarities with all responses
    similarities = torch.matmul(context_emb, response_embeddings.T).squeeze(0)
    
    # Get top-(k+10) to ensure we have k negatives after filtering
    top_indices = torch.topk(similarities, k=min(k+10, len(response_bank))).indices.tolist()
    
    # Get candidates
    candidates = [response_bank[i] for i in top_indices]
    
    # Separate positive and negatives
    negatives = [c for c in candidates if c != positive][:k]  # Keep exactly k negatives
    
    # If positive wasn't retrieved, we still return k negatives
    # The positive will be added separately as label=1 pair
    
    return negatives


def mine_for_split(triplets, response_bank, response_embeddings, dual_encoder, 
                   device, output_path, k=12):
    """Mine hard negatives for a data split"""
    print(f"\nMining hard negatives for {len(triplets):,} triplets...")
    
    pairs = []
    contexts_without_positive = 0
    
    for triplet in tqdm(triplets, desc="Mining"):
        context = triplet['context']
        positive = triplet['positive']
        
        # Retrieve hard negatives
        hard_negatives = retrieve_hard_negatives(
            context, positive, response_bank, response_embeddings,
            dual_encoder, device, k=k
        )
        
        # Create positive pair
        pairs.append({
            'context': context,
            'response': positive,
            'label': 1
        })
        
        # Create negative pairs
        for negative in hard_negatives:
            pairs.append({
                'context': context,
                'response': negative,
                'label': 0
            })
        
        # Check if positive was in retrieved candidates
        if positive not in hard_negatives and len(hard_negatives) == k:
            # Positive was naturally retrieved
            pass
        else:
            contexts_without_positive += 1
    
    # Save pairs
    print(f"\nSaving {len(pairs):,} pairs to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(pairs):,} pairs")
    print(f"  Contexts where positive wasn't in top-{k}: {contexts_without_positive:,} ({contexts_without_positive/len(triplets)*100:.1f}%)")
    
    # Compute label distribution
    num_pos = sum(1 for p in pairs if p['label'] == 1)
    num_neg = len(pairs) - num_pos
    print(f"  Positive pairs: {num_pos:,} ({num_pos/len(pairs)*100:.1f}%)")
    print(f"  Negative pairs: {num_neg:,} ({num_neg/len(pairs)*100:.1f}%)")


def main():
    print("=" * 80)
    print("MINING HARD NEGATIVES")
    print("=" * 80)
    print()
    
    # Paths
    RESPONSE_BANK_DIR = Path('response_bank')
    DUAL_ENCODER_CHECKPOINT = 'dual_encoder_checkpoints/best_model.pt'
    TRIPLETS_DIR = Path('triplets_output')
    OUTPUT_DIR = Path('cross_encoder_data')
    
    K = 12  # Number of hard negatives
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load response bank
    response_bank, response_embeddings = load_response_bank(RESPONSE_BANK_DIR)
    response_embeddings = response_embeddings.to(device)
    
    # Load dual encoder
    print("\nLoading dual encoder...")
    dual_encoder = DualEncoder(
        model_name='google/muril-base-cased',
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        temperature=0.07
    ).to(device)
    
    checkpoint = torch.load(DUAL_ENCODER_CHECKPOINT, map_location=device)
    dual_encoder.load_state_dict(checkpoint['model_state_dict'])
    dual_encoder.eval()
    print(f"✓ Model loaded (epoch {checkpoint['epoch'] + 1})")
    
    # Mine for each split
    for split in ['train', 'val']:
        print("\n" + "=" * 80)
        print(f"PROCESSING {split.upper()} SET")
        print("=" * 80)
        
        triplets_path = TRIPLETS_DIR / f'{split}_triplets.jsonl'
        output_path = OUTPUT_DIR / f'{split}_pairs.jsonl'
        
        triplets = load_triplets(triplets_path)
        print(f"Loaded {len(triplets):,} triplets")
        
        mine_for_split(triplets, response_bank, response_embeddings, 
                      dual_encoder, device, output_path, k=K)
    
    print("\n" + "=" * 80)
    print("HARD NEGATIVE MINING COMPLETE")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
                                                                                                                                                                                                                                           