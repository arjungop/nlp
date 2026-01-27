import torch
import json
import sys
from pathlib import Path
from datetime import datetime

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import DualEncoder
from cross_encoder_model import CrossEncoder


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


class DialogueResponseRanker:
    """Complete two-stage retrieval system: Dual Encoder + Cross Encoder"""
    
    def __init__(
        self,
        dual_encoder_checkpoint,
        cross_encoder_checkpoint,
        response_bank_path,
        response_embeddings_path,
        top_k=12,
        device='cuda',
        verbose=True
    ):
        self.device = device
        self.top_k = top_k
        self.verbose = verbose
        
        log("="*80)
        log("INITIALIZING DIALOGUE RESPONSE RANKER")
        log("="*80)
        log(f"Device: {device}")
        log(f"Top-K candidates for re-ranking: {top_k}")
        log("")
        
        # Load dual encoder (Stage 1: Fast retrieval)
        log("STAGE 1: Loading Dual Encoder...")
        log(f"  Checkpoint path: {dual_encoder_checkpoint}")
        
        try:
            self.dual_encoder = DualEncoder(
                model_name='google/muril-base-cased',
                output_dim=256,
                num_layers=2,
                dropout=0.1,
                temperature=0.07
            ).to(device)
            
            checkpoint = torch.load(dual_encoder_checkpoint, map_location=device, weights_only=False)
            self.dual_encoder.load_state_dict(checkpoint['model_state_dict'])
            self.dual_encoder.eval()
            
            log(f"  SUCCESS: Dual encoder loaded")
            log(f"  Training epoch: {checkpoint['epoch']+1}")
            log(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
            log(f"  Parameters: {sum(p.numel() for p in self.dual_encoder.parameters()):,}")
            log("")
            
        except Exception as e:
            log(f"  ERROR loading dual encoder: {e}")
            raise
        
        # Load cross encoder (Stage 2: Precise re-ranking)
        log("STAGE 2: Loading Cross Encoder...")
        log(f"  Checkpoint path: {cross_encoder_checkpoint}")
        
        try:
            self.cross_encoder = CrossEncoder(dropout=0.1).to(device)
            
            checkpoint = torch.load(cross_encoder_checkpoint, map_location=device, weights_only=False)
            self.cross_encoder.load_state_dict(checkpoint['model_state_dict'])
            self.cross_encoder.eval()
            
            log(f"  SUCCESS: Cross encoder loaded")
            log(f"  Training epoch: {checkpoint['epoch']+1}")
            log(f"  Best MRR: {checkpoint.get('best_mrr', 'N/A'):.4f}")
            log(f"  Parameters: {sum(p.numel() for p in self.cross_encoder.parameters()):,}")
            log("")
            
        except Exception as e:
            log(f"  ERROR loading cross encoder: {e}")
            raise
        
        # Load response bank
        log("Loading Response Bank...")
        log(f"  Responses path: {response_bank_path}")
        
        try:
            with open(response_bank_path, 'r', encoding='utf-8') as f:
                self.responses = [line.strip() for line in f if line.strip()]
            
            log(f"  SUCCESS: Loaded {len(self.responses):,} responses")
            log(f"  Sample responses:")
            for i in range(min(3, len(self.responses))):
                preview = self.responses[i][:80] + '...' if len(self.responses[i]) > 80 else self.responses[i]
                log(f"    [{i}] {preview}")
            log("")
            
        except Exception as e:
            log(f"  ERROR loading response bank: {e}")
            raise
        
        # Load response embeddings
        log("Loading Response Embeddings...")
        log(f"  Embeddings path: {response_embeddings_path}")
        
        try:
            self.response_embeddings = torch.load(response_embeddings_path, map_location=device, weights_only=False)
            
            log(f"  SUCCESS: Loaded embeddings")
            log(f"  Shape: {self.response_embeddings.shape}")
            log(f"  Dimension: {self.response_embeddings.shape[1]}")
            log(f"  Device: {self.response_embeddings.device}")
            
            # Verify count matches
            if len(self.responses) != self.response_embeddings.shape[0]:
                log(f"  WARNING: Mismatch between responses ({len(self.responses)}) and embeddings ({self.response_embeddings.shape[0]})")
            
            log("")
            
        except Exception as e:
            log(f"  ERROR loading embeddings: {e}")
            raise
        
        log("="*80)
        log("INITIALIZATION COMPLETE")
        log("="*80)
        log("")
        
    def rank_responses(self, context, return_top_n=1, return_scores=True, verbose=None):
        """
        Given a context, return the best response(s)
        
        Args:
            context: List of utterances ["utt1", "utt2", "utt3"]
            return_top_n: Number of top responses to return
            return_scores: Whether to return scores
            verbose: Override instance verbose setting
        
        Returns:
            List of dicts with response, scores, and rank
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            log("-"*80)
            log("RANKING RESPONSES FOR CONTEXT")
            log("-"*80)
            log(f"Context ({len(context)} utterances):")
            for i, utt in enumerate(context, 1):
                preview = utt[:80] + '...' if len(utt) > 80 else utt
                log(f"  [{i}] {preview}")
            log(f"Requesting top-{return_top_n} responses")
            log("")
        
        with torch.no_grad():
            # STAGE 1: Dual Encoder Retrieval
            if verbose:
                log("STAGE 1: DUAL ENCODER RETRIEVAL")
                log(f"  Encoding context...")
            
            try:
                context_emb = self.dual_encoder.encode_context(
                    [context], 
                    max_length=256, 
                    device=self.device
                )[0]
                
                if verbose:
                    log(f"  Context embedding shape: {context_emb.shape}")
                    log(f"  Context embedding norm: {torch.norm(context_emb).item():.4f}")
                
            except Exception as e:
                log(f"  ERROR encoding context: {e}")
                raise
            
            # Compute similarity with all responses
            if verbose:
                log(f"  Computing similarity with {len(self.responses):,} responses...")
            
            try:
                similarities = torch.matmul(
                    self.response_embeddings.to(self.device),
                    context_emb
                )
                
                if verbose:
                    log(f"  Similarities shape: {similarities.shape}")
                    log(f"  Similarity range: [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")
                    log(f"  Mean similarity: {similarities.mean().item():.4f}")
                
            except Exception as e:
                log(f"  ERROR computing similarities: {e}")
                raise
            
            # Get top-K candidates
            if verbose:
                log(f"  Retrieving top-{self.top_k} candidates...")
            
            try:
                top_k_scores, top_k_indices = torch.topk(similarities, self.top_k)
                candidate_responses = [self.responses[idx] for idx in top_k_indices.cpu()]
                
                if verbose:
                    log(f"  Retrieved {len(candidate_responses)} candidates")
                    log(f"  Score range: [{top_k_scores.min().item():.4f}, {top_k_scores.max().item():.4f}]")
                    log(f"  Top-3 dual encoder candidates:")
                    for i in range(min(3, len(candidate_responses))):
                        preview = candidate_responses[i][:60] + '...' if len(candidate_responses[i]) > 60 else candidate_responses[i]
                        log(f"    [{i+1}] Score={top_k_scores[i].item():.4f}: {preview}")
                    log("")
                
            except Exception as e:
                log(f"  ERROR retrieving top-K: {e}")
                raise
            
            # STAGE 2: Cross Encoder Re-ranking
            if verbose:
                log("STAGE 2: CROSS ENCODER RE-RANKING")
                log(f"  Preparing batch for cross-encoder...")
            
            try:
                contexts_batch = [context] * len(candidate_responses)
                
                if verbose:
                    log(f"  Batch size: {len(contexts_batch)}")
                    log(f"  Running cross-encoder...")
                
                logits = self.cross_encoder(
                    contexts_batch, 
                    candidate_responses, 
                    max_length=384
                )
                cross_scores = torch.sigmoid(logits)
                
                if verbose:
                    log(f"  Cross-encoder scores shape: {cross_scores.shape}")
                    log(f"  Score range: [{cross_scores.min().item():.4f}, {cross_scores.max().item():.4f}]")
                    log(f"  Mean score: {cross_scores.mean().item():.4f}")
                
            except Exception as e:
                log(f"  ERROR in cross-encoder: {e}")
                raise
            
            # Sort by cross-encoder score
            if verbose:
                log(f"  Sorting by cross-encoder scores...")
            
            try:
                sorted_indices = torch.argsort(cross_scores, descending=True)
                
                if verbose:
                    log(f"  Top-3 after re-ranking:")
                    for i in range(min(3, len(candidate_responses))):
                        idx = sorted_indices[i].item()
                        preview = candidate_responses[idx][:60] + '...' if len(candidate_responses[idx]) > 60 else candidate_responses[idx]
                        log(f"    [{i+1}] Cross={cross_scores[idx].item():.4f}, Dual={top_k_scores[idx].item():.4f}: {preview}")
                    log("")
                
            except Exception as e:
                log(f"  ERROR sorting results: {e}")
                raise
            
            # Build final results
            results = []
            for idx in sorted_indices[:return_top_n]:
                idx_val = idx.item()
                result = {
                    'response': candidate_responses[idx_val],
                    'rank': len(results) + 1
                }
                
                if return_scores:
                    result['cross_encoder_score'] = cross_scores[idx_val].item()
                    result['dual_encoder_score'] = top_k_scores[idx_val].item()
                
                results.append(result)
            
            if verbose:
                log("="*80)
                log(f"FINAL TOP-{return_top_n} RESPONSE(S)")
                log("="*80)
                for result in results:
                    log(f"Rank {result['rank']}:")
                    log(f"  Response: {result['response']}")
                    if return_scores:
                        log(f"  Cross-encoder score: {result['cross_encoder_score']:.4f}")
                        log(f"  Dual-encoder score: {result['dual_encoder_score']:.4f}")
                    log("")
            
            return results
                                                                                                                                                                                                                                           