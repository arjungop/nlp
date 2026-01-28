import json
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('/dist_home/suryansh/dialogue/code')
from model import DualEncoderModel
from hybrid_retriever import HybridRetriever

class HardNegativeMiner:
    def __init__(self, checkpoint_path, indices_dir, response_bank_path, excluded_ids_path, device='cuda'):
        self.device = device
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.model = DualEncoderModel().to(device)
        self.model.context_encoder.load_state_dict(checkpoint['context_encoder_state_dict'])
        self.model.response_encoder.load_state_dict(checkpoint['response_encoder_state_dict'])
        self.model.eval()
        
        print("Loading retriever...")
        self.retriever = HybridRetriever(indices_dir)
        
        print("Loading response bank...")
        self.responses = []
        with open(response_bank_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.responses.append(entry['text'])
        
        print("Loading excluded IDs...")
        with open(excluded_ids_path, 'r') as f:
            self.excluded_indices = set(json.load(f))
        
        print(f"Miner initialized with {len(self.responses)} responses")
        print(f"Excluding {len(self.excluded_indices)} validation/test indices")
    
    def encode_contexts(self, contexts, batch_size=128):
        """
        Encode contexts using the context encoder in inference mode.
        
        Args:
            contexts: List of context (dialogue history) lists
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of context embeddings
        """
        embeddings = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(contexts), batch_size), desc="Encoding contexts"):
                batch = contexts[i:i+batch_size]
                batch_embeddings = self.model.encode_context(batch, device=self.device)
                embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)
    
    def encode_all_responses(self, batch_size=128):
        """
        Encode all responses using the response encoder in inference mode.
        
        Args:
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of response embeddings
        """
        embeddings = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(self.responses), batch_size), desc="Encoding responses"):
                batch = self.responses[i:i+batch_size]
                batch_embeddings = self.model.encode_response(batch, device=self.device)
                embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)
    
    def mine_negatives(self, triplets_path, num_faiss=20, num_bm25=7, num_random=3):
        """
        Mine hard negatives using hybrid retrieval (FAISS + BM25).
        
        Args:
            triplets_path: Path to training triplets file
            num_faiss: Number of FAISS negatives to mine
            num_bm25: Number of BM25 negatives to mine
            num_random: Number of random negatives to mine
            
        Returns:
            Dictionary mapping sample_id to mined negative indices
        """
        print("Loading training triplets...")
        contexts = []
        positive_responses = []
        positive_indices = []
        sample_ids = []
        
        with open(triplets_path, 'r', encoding='utf-8') as f:
            for line in f:
                triplet = json.loads(line.strip())
                contexts.append(triplet['context'])
                positive_responses.append(triplet['positive'])
                positive_indices.append(triplet['metadata']['positive_index'])
                sample_ids.append(triplet['metadata']['sample_id'])
        
        print(f"Loaded {len(contexts)} training samples")
        
        print("Encoding all responses with current model...")
        response_embeddings = self.encode_all_responses()
        
        print("Updating FAISS index...")
        self.retriever.update_faiss_embeddings(response_embeddings)
        
        print("Encoding contexts...")
        context_embeddings = self.encode_contexts(contexts)
        
        print("Retrieving FAISS candidates...")
        faiss_indices = self.retriever.faiss_retrieve(context_embeddings, k=100)
        
        print("Retrieving BM25 candidates...")
        context_texts = [' '.join(ctx) for ctx in contexts]
        bm25_results = self.retriever.bm25_retrieve(context_texts, k=100)
        
        mined_negatives = {}
        
        print("Mining hard negatives...")
        for i in tqdm(range(len(contexts)), desc="Sampling negatives"):
            sample_id = sample_ids[i]
            positive_idx = positive_indices[i]
            
            faiss_negs = []
            min_rank, max_rank = 20, 60
            for faiss_idx in faiss_indices[i][min_rank:max_rank]:
                if (faiss_idx != positive_idx and 
                    faiss_idx not in self.excluded_indices):
                    faiss_negs.append(int(faiss_idx))
                    if len(faiss_negs) >= num_faiss:
                        break
            
            if len(faiss_negs) < num_faiss:
                print(f"WARNING: Sample {sample_id} only got {len(faiss_negs)}/{num_faiss} FAISS negatives")
            
            bm25_negs = []
            for bm25_idx in bm25_results[i][min_rank:max_rank]:
                if (bm25_idx != positive_idx and 
                    bm25_idx not in self.excluded_indices and
                    bm25_idx not in faiss_negs):
                    bm25_negs.append(int(bm25_idx))
                    if len(bm25_negs) >= num_bm25:
                        break
            
            if len(bm25_negs) < num_bm25:
                print(f"WARNING: Sample {sample_id} only got {len(bm25_negs)}/{num_bm25} BM25 negatives")
            
            random_negs = []
            max_attempts = num_random * 10
            attempts = 0
            while len(random_negs) < num_random and attempts < max_attempts:
                rand_idx = np.random.randint(0, len(self.responses))
                if (rand_idx != positive_idx and 
                    rand_idx not in self.excluded_indices and
                    rand_idx not in faiss_negs and
                    rand_idx not in bm25_negs):
                    random_negs.append(int(rand_idx))
                attempts += 1
            
            mined_negatives[sample_id] = {
                'faiss_negatives': faiss_negs,
                'bm25_negatives': bm25_negs,
                'random_negatives': random_negs
            }
        
        return mined_negatives


def mine_for_epoch(epoch, checkpoint_dir, output_dir):
    """
    Mine hard negatives for a specific epoch using the previous epoch's checkpoint.
    
    Args:
        epoch: Current epoch number
        checkpoint_dir: Directory containing checkpoints
        output_dir: Directory to save mined negatives
        
    Returns:
        Path to saved mined negatives file, or None if mining failed
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch-1}.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None
    
    indices_dir = '/dist_home/suryansh/dialogue/indices'
    response_bank_path = '/dist_home/suryansh/dialogue/response_bank/response_bank.jsonl'
    excluded_ids_path = '/dist_home/suryansh/dialogue/val_test_ids.json'
    triplets_path = '/dist_home/suryansh/dialogue/triplets_output/train_triplets.jsonl'
    
    miner = HardNegativeMiner(
        checkpoint_path=checkpoint_path,
        indices_dir=indices_dir,
        response_bank_path=response_bank_path,
        excluded_ids_path=excluded_ids_path
    )
    
    mined_negatives = miner.mine_negatives(triplets_path)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'mined_negatives_epoch_{epoch}.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mined_negatives, f, indent=2)
    
    print(f"Mined negatives saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python mine_hard_negatives_v2.py <epoch> <checkpoint_dir> <output_dir>")
        sys.exit(1)
    
    epoch = int(sys.argv[1])
    checkpoint_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    mine_for_epoch(epoch, checkpoint_dir, output_dir)
                                                                                                                                                                                                                                           
