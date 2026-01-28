
import json
import torch
from torch.utils.data import Dataset

class ImprovedTripletDataset(Dataset):
    def __init__(self, triplet_path, response_bank_path, mined_negatives_path=None, stage='warmup'):
        self.stage = stage
        
        print(f"Loading triplets from: {triplet_path}")
        with open(triplet_path, 'r', encoding='utf-8') as f:
            self.triplets = [json.loads(line.strip()) for line in f if line.strip()]
        
        print(f"Loading response bank from: {response_bank_path}")
        self.response_bank = {}
        with open(response_bank_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                self.response_bank[entry['index']] = entry['text']
        
        self.mined_negatives = None
        if mined_negatives_path:
            print(f"Loading mined negatives from: {mined_negatives_path}")
            with open(mined_negatives_path, 'r', encoding='utf-8') as f:
                self.mined_negatives = json.load(f)
        
        print(f"Dataset loaded: {len(self.triplets)} samples, stage={stage}")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        context = triplet['context']
        positive = triplet['positive']
        sample_id = triplet['metadata']['sample_id']
        
        if self.stage == 'warmup':
            original_negatives = triplet['negatives'][:5]
            negatives = original_negatives
        
        elif self.mined_negatives and sample_id in self.mined_negatives:
            mined = self.mined_negatives[sample_id]
            
            negative_indices = (
                mined['faiss_negatives'][:20] +
                mined['bm25_negatives'][:7] +
                mined['random_negatives'][:3]
            )
            
            negative_texts = []
            for neg_idx in negative_indices:
                if neg_idx in self.response_bank:
                    negative_texts.append(self.response_bank[neg_idx])
                else:
                    print(f"WARNING: Negative index {neg_idx} not found in response bank for sample {sample_id}")
            
            if len(negative_texts) < 10:
                print(f"WARNING: Sample {sample_id} only has {len(negative_texts)} negatives, filling with originals")
                original_negatives = triplet['negatives']
                while len(negative_texts) < 10 and len(original_negatives) > 0:
                    negative_texts.append(original_negatives.pop(0))
            
            negatives = negative_texts
        
        else:
            original_negatives = triplet['negatives'][:10]
            negatives = original_negatives
        
        return {
            'context': context,
            'positive': positive,
            'negatives': negatives
        }
                                                                                                                                                                                                                                           
