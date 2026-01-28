import json
import os
import numpy as np
import faiss
import bm25s
import pickle
from tqdm import tqdm

RESPONSE_BANK = '/dist_home/suryansh/dialogue/response_bank/response_bank.jsonl'
OUTPUT_DIR = '/dist_home/suryansh/dialogue/indices'
EMBEDDING_DIM = 512

def build_indices():
    print("Building BM25 and FAISS indices...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    responses = []
    indices = []
    
    print("Loading response bank...")
    with open(RESPONSE_BANK, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading responses"):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                responses.append(entry['text'])
                indices.append(entry['index'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"ERROR: Failed to parse response: {e}")
                continue
    
    num_responses = len(responses)
    print(f"Loaded {num_responses} responses")
    
    print("Building BM25 index using bm25s...")
    print("Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(responses, stopwords='en')
    
    print("Creating BM25 retriever...")
    retriever = bm25s.BM25()
    
    print("Indexing corpus...")
    retriever.index(corpus_tokens)
    
    bm25_path = os.path.join(OUTPUT_DIR, 'bm25_index.pkl')
    with open(bm25_path, 'wb') as f:
        pickle.dump(retriever, f)
    print(f"BM25 index saved to: {bm25_path}")
    
    print("Building FAISS index...")
    embeddings = np.random.randn(num_responses, EMBEDDING_DIM).astype('float32')
    faiss.normalize_L2(embeddings)
    
    nlist = min(4096, int(np.sqrt(num_responses)))
    print(f"Using {nlist} clusters for IVF")
    
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print("Training FAISS index...")
    index.train(embeddings)
    
    print("Adding vectors to FAISS index...")
    index.add(embeddings)
    
    faiss_path = os.path.join(OUTPUT_DIR, 'faiss_index.bin')
    faiss.write_index(index, faiss_path)
    print(f"FAISS index saved to: {faiss_path}")
    
    metadata = {
        'num_responses': num_responses,
        'embedding_dim': EMBEDDING_DIM,
        'nlist': nlist,
        'response_indices': indices
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    print("Index building complete!")

if __name__ == '__main__':
    build_indices()
                                                                                                                                                  
