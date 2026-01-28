import json
import os
import numpy as np
import faiss
import bm25s
import pickle

class HybridRetriever:
    def __init__(self, indices_dir):
        self.indices_dir = indices_dir
        
        print("Loading BM25 index...")
        bm25_path = os.path.join(indices_dir, 'bm25_index.pkl')
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        print("Loading FAISS index...")
        faiss_path = os.path.join(indices_dir, 'faiss_index.bin')
        self.faiss_index = faiss.read_index(faiss_path)
        
        print("Loading metadata...")
        metadata_path = os.path.join(indices_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.response_indices = metadata['response_indices']
        self.num_responses = metadata['num_responses']
        
        print(f"Loaded indices with {self.num_responses} responses")
    
    def bm25_retrieve(self, queries, k=100):
        """
        Retrieve top-k documents using BM25S.
        
        Args:
            queries: List of query strings
            k: Number of results to return per query
            
        Returns:
            2D numpy array of shape (num_queries, k) containing document indices
        """
        query_tokens = bm25s.tokenize(queries, stopwords='en')
        
        results, scores = self.bm25.retrieve(query_tokens, k=k)
        
        return results
    
    def faiss_retrieve(self, embeddings, k=100):
        """
        Retrieve top-k documents using FAISS.
        
        Args:
            embeddings: numpy array of query embeddings
            k: Number of results to return per query
            
        Returns:
            2D numpy array of shape (num_queries, k) containing document indices
        """
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.faiss_index.nprobe = min(10, self.faiss_index.nlist)
        
        distances, indices = self.faiss_index.search(embeddings, k)
        return indices
    
    def update_faiss_embeddings(self, new_embeddings):
        """
        Update FAISS index with new embeddings (re-train and re-add).
        
        Args:
            new_embeddings: numpy array of new embeddings to index
        """
        print("Updating FAISS index with new embeddings...")
        
        new_embeddings = new_embeddings.astype('float32')
        faiss.normalize_L2(new_embeddings)
        
        self.faiss_index.reset()
        self.faiss_index.train(new_embeddings)
        self.faiss_index.add(new_embeddings)
        
        print("FAISS index updated successfully")
                                                                                                                                                                                                                                           
