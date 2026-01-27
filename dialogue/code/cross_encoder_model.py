import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CrossEncoder(nn.Module):
    """Cross-encoder with fine-tuned MuRIL for binary classification"""
    
    def __init__(self, model_name='google/muril-base-cased', dropout=0.1):
        super().__init__()
        
        # Load MuRIL (will be fine-tuned)
        print(f"Loading MuRIL: {model_name}")
        self.muril = AutoModel.from_pretrained(model_name)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)
        
        print(f"✓ Cross-encoder initialized")
        print(f"  MuRIL parameters: {sum(p.numel() for p in self.muril.parameters()):,}")
        print(f"  Classifier parameters: {sum(p.numel() for p in self.classifier.parameters()):,}")
    
    def forward(self, context_texts, response_texts, max_length=384):
        """
        Args:
            context_texts: List of lists [[utt1, utt2, utt3], ...]
            response_texts: List of strings
        Returns:
            logits: [batch_size] - raw scores (before sigmoid)
        """
        # Join context utterances
        joined_contexts = [' [SEP] '.join(ctx) for ctx in context_texts]
        
        # Tokenize (context + response concatenated)
        encoded = self.tokenizer(
            text=joined_contexts,
            text_pair=response_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.muril.device)
        attention_mask = encoded['attention_mask'].to(self.muril.device)
        
        # Forward through MuRIL (fine-tuned)
        outputs = self.muril(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Classification head
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding).squeeze(-1)  # [batch]
        
        return logits
                                                                                                                                                                                                                                           