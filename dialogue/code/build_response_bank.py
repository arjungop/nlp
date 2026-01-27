import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import DualEncoder


def load_all_utterances(clean_dialogues_path):
    """Load all unique utterances from cleaned dialogues"""
    print("Loading all utterances from cleaned dialogues...")
    
    utterances_set = set()
    
    with open(clean_dialogues_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading dialogues"):
            line = line.strip()
            if not line:
                continue
            
            dialogue = json.loads(line)
            
            # Extract all utterances
            for utterance in dialogue['utterances']:
                text = utterance.strip()
                if text:
                    utterances_set.add(text)
    
    utterances = sorted(list(utterances_set))
    print(f"\nTotal unique utterances: {len(utterances):,}")
    
    return utterances


def encode_responses(utterances, dual_encoder_checkpoint, device, batch_size=128):
    """Encode all utterances using trained dual encoder"""
    print("\nLoading trained dual encoder...")
    
    # Load model
    model = DualEncoder(
        model_name='google/muril-base-cased',
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        temperature=0.07
    ).to(device)
    
    checkpoint = torch.load(dual_encoder_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Encode all utterances
    print(f"\nEncoding {len(utterances):,} utterances...")
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(utterances), batch_size), desc="Encoding"):
            batch = utterances[i:i+batch_size]
            embeddings = model.encode_response(batch, max_length=256, device=device)
            all_embeddings.append(embeddings.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    return all_embeddings


def main():
    print("=" * 80)
    print("BUILDING RESPONSE BANK")
    print("=" * 80)
    print()
    
    # Paths
    CLEAN_DIALOGUES = 'tamil_dialogues_clean.jsonl'
    DUAL_ENCODER_CHECKPOINT = 'dual_encoder_checkpoints/best_model.pt'
    OUTPUT_DIR = Path('response_bank')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load all utterances
    utterances = load_all_utterances(CLEAN_DIALOGUES)
    
    # Encode with dual encoder
    embeddings = encode_responses(utterances, DUAL_ENCODER_CHECKPOINT, device)
    
    # Save responses
    print("\nSaving response bank...")
    responses_path = OUTPUT_DIR / 'responses.txt'
    with open(responses_path, 'w', encoding='utf-8') as f:
        for utterance in utterances:
            f.write(utterance + '\n')
    print(f"Saved {len(utterances):,} responses to {responses_path}")
    
    # Save embeddings
    embeddings_path = OUTPUT_DIR / 'embeddings.pt'
    torch.save(embeddings, embeddings_path)
    print(f"Saved embeddings ({embeddings.shape}) to {embeddings_path}")
    
    print("\n" + "=" * 80)
    print("RESPONSE BANK COMPLETE")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  {responses_path} ({len(utterances):,} responses)")
    print(f"  {embeddings_path} ({embeddings.shape[0]:,} embeddings)")
    print()


if __name__ == '__main__':
    main()
                                                                                                                                                                                                                                           