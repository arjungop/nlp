import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('/dist_home/suryansh/dialogue/code')
from model import DualEncoderModel
from dataset_improved import ImprovedTripletDataset
from mine_hard_negatives_v2 import mine_for_epoch

class ImprovedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = DualEncoderModel().to(self.device)
        
        print("Loading training dataset...")
        self.train_dataset = ImprovedTripletDataset(
            triplet_path=config['train_data_path'],
            response_bank_path=config['response_bank_path'],
            stage='warmup'
        )
        
        self.train_loader = self.create_dataloader(self.train_dataset)
        
        self.criterion = nn.CrossEntropyLoss()
        
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
    
    def create_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def get_stage(self, epoch):
        if epoch <= 3:
            return 'warmup'
        elif epoch <= 12:
            return 'mining'
        else:
            return 'intensive'
    
    def configure_optimizer(self, stage):
        lr_config = self.config['learning_rates'][stage]
        
        encoder_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'muril' in name:
                param.requires_grad = False
            elif 'context_encoder' in name or 'response_encoder' in name:
                param.requires_grad = (lr_config['encoder'] > 0)
                if param.requires_grad:
                    encoder_params.append(param)
            else:
                param.requires_grad = True
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr_config['encoder']},
            {'params': other_params, 'lr': lr_config['other']}
        ])
        
        return optimizer
    
    def mine_negatives_if_needed(self, epoch):
        stage = self.get_stage(epoch)
        mining_schedule = self.config['mining_schedule'].get(stage, [])
        
        if epoch not in mining_schedule:
            return False
        
        checkpoint_path = Path(self.config['output_dir']) / f"checkpoint_epoch_{epoch-1}.pt"
        if not checkpoint_path.exists():
            print(f"WARNING: Cannot mine - checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"\n{'='*60}")
        print(f"MINING HARD NEGATIVES FOR EPOCH {epoch}")
        print(f"{'='*60}\n")
        
        output_path = mine_for_epoch(
            epoch=epoch,
            checkpoint_dir=self.config['output_dir'],
            output_dir=self.config['mined_negatives_dir']
        )
        
        if not output_path:
            print("ERROR: Mining failed")
            return False
        
        print("\nReloading dataset with mined negatives...")
        self.train_dataset = ImprovedTripletDataset(
            triplet_path=self.config['train_data_path'],
            response_bank_path=self.config['response_bank_path'],
            mined_negatives_path=str(output_path),
            stage=stage
        )
        
        print("Dataset reloaded successfully")
        return True
    
    def train_epoch(self, epoch, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            contexts = batch['context']
            positives = batch['positive']
            negatives = batch['negatives']
            
            batch_size = len(contexts)
            num_negatives = len(negatives[0])
            
            context_embeddings = self.model.encode_context(contexts, device=self.device)
            
            all_responses = []
            for i in range(batch_size):
                all_responses.append(positives[i])
                all_responses.extend(negatives[i])
            
            response_embeddings = self.model.encode_response(all_responses, device=self.device)
            
            logits_list = []
            for i in range(batch_size):
                ctx_emb = context_embeddings[i:i+1]
                start_idx = i * (num_negatives + 1)
                end_idx = start_idx + num_negatives + 1
                resp_embs = response_embeddings[start_idx:end_idx]
                
                logits = torch.matmul(ctx_emb, resp_embs.T) / self.model.temperature
                logits_list.append(logits)
            
            logits = torch.cat(logits_list, dim=0)
            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            loss = self.criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch):
        checkpoint_path = Path(self.config['output_dir']) / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'context_encoder_state_dict': self.model.context_encoder.state_dict(),
            'response_encoder_state_dict': self.model.response_encoder.state_dict(),
            'temperature': self.model.temperature
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        print(f"\n{'='*60}")
        print("STARTING IMPROVED TRAINING")
        print(f"{'='*60}\n")
        
        current_stage = None
        optimizer = None
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            stage = self.get_stage(epoch)
            
            if stage != current_stage:
                print(f"\n{'='*60}")
                print(f"ENTERING STAGE: {stage.upper()} (Epoch {epoch})")
                print(f"{'='*60}\n")
                current_stage = stage
                optimizer = self.configure_optimizer(stage)
                self.train_dataset.stage = stage
            
            needs_new_loader = self.mine_negatives_if_needed(epoch)
            if needs_new_loader:
                self.train_loader = self.create_dataloader(self.train_dataset)
            
            avg_loss = self.train_epoch(epoch, optimizer)
            
            print(f"Epoch {epoch}/{self.config['num_epochs']} - Loss: {avg_loss:.4f}")
            
            self.save_checkpoint(epoch)
            
            log_path = Path(self.config['log_dir']) / f"epoch_{epoch}_metrics.json"
            with open(log_path, 'w') as f:
                json.dump({'epoch': epoch, 'loss': avg_loss, 'stage': stage}, f, indent=2)
        
        print("\n Training complete!")


CONFIG = {
    'train_data_path': '/dist_home/suryansh/dialogue/triplets_output/train_triplets.jsonl',
    'response_bank_path': '/dist_home/suryansh/dialogue/response_bank/response_bank.jsonl',
    'output_dir': '/dist_home/suryansh/dialogue/checkpoints_improved',
    'log_dir': '/dist_home/suryansh/dialogue/logs_improved',
    'mined_negatives_dir': '/dist_home/suryansh/dialogue/mined_negatives',
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rates': {
        'warmup': {'encoder': 0.0, 'other': 2e-3},
        'mining': {'encoder': 5e-6, 'other': 1e-4},
        'intensive': {'encoder': 2e-6, 'other': 5e-5}
    },
    'mining_schedule': {
        'warmup': [],
        'mining': [4, 6, 8, 10, 12],
        'intensive': [13, 14, 15, 16, 17, 18, 19, 20]
    }
}


if __name__ == '__main__':
    trainer = ImprovedTrainer(CONFIG)
    trainer.train()
                                                                                                                                                                                                                                           
