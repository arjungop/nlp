import json
from torch.utils.data import Dataset


class CrossEncoderDataset(Dataset):
    """Dataset for cross-encoder binary classification pairs"""
    
    def __init__(self, pairs_path):
        self.pairs = []
        print(f"Loading pairs from {pairs_path}...")
        
        with open(pairs_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pairs.append(json.loads(line))
        
        print(f"Loaded {len(self.pairs):,} pairs")
        
        # Count labels
        num_pos = sum(1 for p in self.pairs if p['label'] == 1)
        num_neg = len(self.pairs) - num_pos
        print(f"  Positive: {num_pos:,} ({num_pos/len(self.pairs)*100:.1f}%)")
        print(f"  Negative: {num_neg:,} ({num_neg/len(self.pairs)*100:.1f}%)")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            'context': pair['context'],
            'response': pair['response'],
            'label': pair['label']
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    contexts = [item['context'] for item in batch]
    responses = [item['response'] for item in batch]
    labels = [item['label'] for item in batch]
    
    return {
        'context': contexts,
        'response': responses,
        'label': labels
    }
                                                                                                                                                                                                                                           