import torch


def compute_recall_at_k(logits, k):
    """
    Compute recall@k where positive response is at index 0
    Args:
        logits: [batch, num_candidates] - positive at index 0
        k: int - top-k to check
    Returns:
        recall: float - proportion of samples where positive is in top-k
    """
    batch_size = logits.size(0)
    
    # Get top-k indices
    _, top_k_indices = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    
    # Check if 0 is in top-k for each sample
    correct = (top_k_indices == 0).any(dim=1).float()
    
    return correct.mean().item()


def compute_mrr(logits):
    """
    Compute Mean Reciprocal Rank where positive is at index 0
    Args:
        logits: [batch, num_candidates]
    Returns:
        mrr: float - mean reciprocal rank
    """
    # Sort indices in descending order
    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    
    # Find position of index 0 (positive) in sorted list
    ranks = (sorted_indices == 0).nonzero(as_tuple=True)[1].float() + 1  # +1 for 1-indexed
    
    # Compute reciprocal ranks
    reciprocal_ranks = 1.0 / ranks
    
    return reciprocal_ranks.mean().item()


def format_time(seconds):
    """Format seconds into readable string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"