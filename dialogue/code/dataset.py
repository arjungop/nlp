import json
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """Dataset for loading dialogue triplets from JSONL"""
    
    def __init__(self, jsonl_path):
        self.samples = []
        print(f"Loading dataset from {jsonl_path}...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'context': sample['context'],  # List of 3 strings
            'positive': sample['positive'],  # Single string
            'negatives': sample['negatives']  # List of 12 strings
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    Returns batched lists (no tokenization yet)
    """
    contexts = [item['context'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negatives'] for item in batch]
    
    return {
        'context': contexts,
        'positive': positives,
        'negatives': negatives
    }
                                                                                                                                                                                                                                           