cat model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TransformerEncoder(nn.Module):
    """Transformer encoder that processes MuRIL embeddings"""
    
    def __init__(self, output_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection layer
        self.projection = nn.Linear(768, output_dim)
    
    def forward(self, embeddings, attention_mask):
        """
        Args:
            embeddings: [batch, seq_len, 768] - MuRIL token embeddings
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding
        Returns:
            [batch, output_dim] - L2 normalized embeddings
        """
        # Create padding mask (True = positions to ignore)
        padding_mask = ~attention_mask.bool()
        
        # Apply transformer
        x = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        
        # Mean pooling (weighted by attention mask)
        mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_embeddings = x * mask_expanded  # Zero out padding
        sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, 768]
        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask  # [batch, 768]
        
        # Project and normalize
        projected = self.projection(pooled)  # [batch, output_dim]
        return F.normalize(projected, p=2, dim=-1)


class DualEncoder(nn.Module):
    """Dual encoder with frozen MuRIL and trainable transformer projections"""
    
    def __init__(self, model_name='google/muril-base-cased', 
                 output_dim=256, num_layers=2, dropout=0.1, temperature=0.07):
        super().__init__()
        
        # Load and freeze MuRIL
        print(f"Loading MuRIL: {model_name}")
        self.muril = AutoModel.from_pretrained(model_name)
        for param in self.muril.parameters():
            param.requires_grad = False
        self.muril.eval()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Trainable encoders (separate weights)
        self.context_encoder = TransformerEncoder(output_dim, num_layers, dropout)
        self.response_encoder = TransformerEncoder(output_dim, num_layers, dropout)
        
        self.temperature = temperature
    
    def get_muril_embeddings(self, input_ids, attention_mask):
        """Get frozen MuRIL embeddings for all tokens"""
        with torch.no_grad():
            outputs = self.muril(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state.detach()
    
    def encode_context(self, context_texts, max_length=256, device='cuda'):
        """
        Encode context (list of 3 utterances joined with [SEP])
        Args:
            context_texts: List of lists [[utt1, utt2, utt3], ...]
        """
        # Join utterances with [SEP]
        joined_contexts = [' [SEP] '.join(ctx) for ctx in context_texts]
        
        # Tokenize
        encoded = self.tokenizer(
            joined_contexts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get MuRIL embeddings
        muril_emb = self.get_muril_embeddings(input_ids, attention_mask)
        
        # Encode with trainable transformer
        return self.context_encoder(muril_emb, attention_mask)
    
    def encode_response(self, response_texts, max_length=256, device='cuda'):
        """
        Encode responses
        Args:
            response_texts: List of strings
        """
        # Tokenize
        encoded = self.tokenizer(
            response_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get MuRIL embeddings
        muril_emb = self.get_muril_embeddings(input_ids, attention_mask)
        
        # Encode with trainable transformer
        return self.response_encoder(muril_emb, attention_mask)
    
    def forward(self, context_texts, positive_texts, negative_texts, 
                max_length=256, device='cuda'):
        """
        Forward pass with InfoNCE loss
        Args:
            context_texts: List of lists [[utt1, utt2, utt3], ...] - batch_size items
            positive_texts: List of strings - batch_size items
            negative_texts: List of lists of strings - batch_size x 12
        Returns:
            loss: scalar
            logits: [batch_size, 13] - positive at index 0
        """
        batch_size = len(context_texts)
        
        # Encode context
        context_emb = self.encode_context(context_texts, max_length, device)  # [batch, output_dim]
        
        # Encode positive
        positive_emb = self.encode_response(positive_texts, max_length, device)  # [batch, output_dim]
        
        # Encode negatives (flatten, encode, reshape)
        negative_texts_flat = [neg for negs in negative_texts for neg in negs]  # batch*12 items
        negative_emb = self.encode_response(negative_texts_flat, max_length, device)  # [batch*12, output_dim]
        negative_emb = negative_emb.view(batch_size, -1, negative_emb.size(-1))  # [batch, 12, output_dim]
        
        # Compute similarities (dot product, already L2 normalized so = cosine sim)
        sim_positive = (context_emb * positive_emb).sum(dim=-1, keepdim=True)  # [batch, 1]
        sim_negatives = torch.bmm(negative_emb, context_emb.unsqueeze(-1)).squeeze(-1)  # [batch, 12]
        
        # Concatenate: positive at index 0
        similarities = torch.cat([sim_positive, sim_negatives], dim=1)  # [batch, 13]
        
        # Apply temperature scaling
        logits = similarities / self.temperature  # [batch, 13]
        
        # InfoNCE loss (positive is always at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        
        return loss, logits