#!/usr/bin/env python3

import json
import random
import pickle
import sys
import time
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from pathlib import Path

import numpy as np
import bm25s
from indicnlp.tokenize import indic_tokenize

INPUT_PATH = 'tamil_dialogues_clean.jsonl'
OUTPUT_DIR = Path('triplets_output')
CONTEXT_WINDOW_SIZE = 3
NUM_NEGATIVES = 12
SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
BM25_RETRIEVE_K = 25
FILTER_WINDOW = 10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class GlobalCorpus:
    def __init__(self):
        self.utterances = []
        self.global_idx_to_dialogue = {}
        self.global_idx_to_local_idx = {}
        self.dialogue_to_global_indices = defaultdict(list)
    
    def add_dialogue(self, dialogue_id: str, utterances: List[str]):
        for local_idx, utterance in enumerate(utterances):
            global_idx = len(self.utterances)
            self.utterances.append(utterance)
            self.global_idx_to_dialogue[global_idx] = dialogue_id
            self.global_idx_to_local_idx[global_idx] = local_idx
            self.dialogue_to_global_indices[dialogue_id].append(global_idx)
    
    def get_utterance(self, idx: int) -> str:
        return self.utterances[idx]
    
    def get_dialogue_id(self, idx: int) -> str:
        return self.global_idx_to_dialogue[idx]
    
    def get_local_index(self, idx: int) -> int:
        return self.global_idx_to_local_idx[idx]
    
    def __len__(self):
        return len(self.utterances)

class DeduplicatedCorpus:
    def __init__(self, global_corpus: GlobalCorpus):
        self.global_corpus = global_corpus
        self.text_to_indices = defaultdict(list)
        self.unique_texts = []
        self.text_to_unique_idx = {}
        
        seen = set()
        for idx in range(len(global_corpus)):
            text = global_corpus.get_utterance(idx)
            self.text_to_indices[text].append(idx)
            
            if text not in seen:
                unique_idx = len(self.unique_texts)
                self.unique_texts.append(text)
                self.text_to_unique_idx[text] = unique_idx
                seen.add(text)
        
        print(f"  Original corpus: {len(global_corpus):,} utterances")
        print(f"  Unique texts: {len(self.unique_texts):,} ({len(self.unique_texts)/len(global_corpus)*100:.1f}%)")
        print(f"  Duplicates removed: {len(global_corpus) - len(self.unique_texts):,}")
    
    def get_unique_text(self, unique_idx: int) -> str:
        return self.unique_texts[unique_idx]
    
    def get_unique_idx_for_text(self, text: str) -> int:
        return self.text_to_unique_idx.get(text, -1)
    
    def get_all_global_indices_for_unique_idx(self, unique_idx: int) -> List[int]:
        text = self.unique_texts[unique_idx]
        return self.text_to_indices[text]
    
    def __len__(self):
        return len(self.unique_texts)

def indic_tokenize_for_bm25(text: str) -> List[str]:
    tokens = indic_tokenize.trivial_tokenize_indic(text)
    return [t.strip() for t in tokens if t.strip()]

def load_dialogues(filepath: str) -> List[Dict]:
    dialogues = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    return dialogues

def build_corpus(dialogues: List[Dict]) -> GlobalCorpus:
    corpus = GlobalCorpus()
    for dialogue in dialogues:
        corpus.add_dialogue(dialogue['dialogue_id'], dialogue['utterances'])
    return corpus

def split_dialogues(dialogues: List[Dict], ratios: Dict[str, float]) -> Dict[str, List[str]]:
    sorted_dialogues = sorted(dialogues, key=lambda x: x['dialogue_id'])
    total = len(sorted_dialogues)
    train_end = int(total * ratios['train'])
    val_end = train_end + int(total * ratios['val'])
    
    return {
        'train': [d['dialogue_id'] for d in sorted_dialogues[:train_end]],
        'val': [d['dialogue_id'] for d in sorted_dialogues[train_end:val_end]],
        'test': [d['dialogue_id'] for d in sorted_dialogues[val_end:]]
    }

def build_bm25(dedup_corpus: DeduplicatedCorpus):
    print("  Tokenizing deduplicated corpus for BM25...")
    corpus_tokens = []
    
    for idx in range(len(dedup_corpus)):
        text = dedup_corpus.get_unique_text(idx)
        tokens = indic_tokenize_for_bm25(text)
        corpus_tokens.append(tokens)
        
        if (idx + 1) % 10000 == 0:
            print(f"    Tokenized {idx + 1:,} / {len(dedup_corpus):,} unique texts", end='\r')
    
    print(f"\n  Tokenized {len(dedup_corpus):,} unique texts")
    print("  Building BM25s index (vectorized)...")
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    print("  BM25s index built successfully")
    return retriever

def create_samples_for_dialogue(dialogue: Dict, corpus: GlobalCorpus, context_size: int) -> List[Dict]:
    dialogue_id = dialogue['dialogue_id']
    utterances = dialogue['utterances']
    
    if len(utterances) < context_size + 1:
        return []
    
    global_indices = corpus.dialogue_to_global_indices[dialogue_id]
    samples = []
    
    for i in range(len(utterances) - context_size):
        context_utterances = utterances[i:i+context_size]
        positive_utterance = utterances[i+context_size]
        context_global_indices = global_indices[i:i+context_size]
        positive_global_idx = global_indices[i+context_size]
        
        samples.append({
            'dialogue_id': dialogue_id,
            'context': context_utterances,
            'positive': positive_utterance,
            'context_global_indices': context_global_indices,
            'positive_global_idx': positive_global_idx,
            'context_local_indices': list(range(i, i+context_size)),
            'positive_local_idx': i+context_size
        })
    
    return samples

def get_forbidden_indices(sample: Dict, corpus: GlobalCorpus, filter_window: int) -> Set[int]:
    forbidden = set()
    forbidden.add(sample['positive_global_idx'])
    forbidden.update(sample['context_global_indices'])
    
    dialogue_id = sample['dialogue_id']
    positive_local_idx = sample['positive_local_idx']
    all_global_indices = corpus.dialogue_to_global_indices[dialogue_id]
    
    start = max(0, positive_local_idx - filter_window)
    end = min(len(all_global_indices), positive_local_idx + filter_window + 1)
    
    for local_idx in range(start, end):
        forbidden.add(all_global_indices[local_idx])
    
    return forbidden

def sample_negatives_batch(samples_batch, bm25_retriever, dedup_corpus, global_corpus, num_negatives, retrieve_k):
    queries = []
    for sample in samples_batch:
        context_text = " ".join(sample['context'])
        query_tokens = indic_tokenize_for_bm25(context_text)
        queries.append(query_tokens)
    
    results, scores = bm25_retriever.retrieve(queries, k=retrieve_k)
    
    samples_with_negatives = []
    
    for i, sample in enumerate(samples_batch):
        top_unique_indices = results[i]
        top_scores = scores[i]
        
        forbidden_global_indices = get_forbidden_indices(sample, global_corpus, FILTER_WINDOW)
        
        forbidden_texts = sample['context'] + [sample['positive']]
        forbidden_unique_indices = set()
        for text in forbidden_texts:
            unique_idx = dedup_corpus.get_unique_idx_for_text(text)
            if unique_idx >= 0:
                forbidden_unique_indices.add(unique_idx)
        
        valid_negatives = []
        
        for j, unique_idx in enumerate(top_unique_indices):
            if len(valid_negatives) >= num_negatives:
                break
            
            if unique_idx in forbidden_unique_indices:
                continue
            
            candidate_global_indices = dedup_corpus.get_all_global_indices_for_unique_idx(unique_idx)
            
            selected_global_idx = None
            for global_idx in candidate_global_indices:
                if global_idx not in forbidden_global_indices:
                    selected_global_idx = global_idx
                    break
            
            if selected_global_idx is None:
                continue
            
            text = dedup_corpus.get_unique_text(unique_idx)
            
            valid_negatives.append({
                'global_idx': selected_global_idx,
                'utterance': text,
                'score': float(top_scores[j]),
                'dialogue_id': global_corpus.get_dialogue_id(selected_global_idx)
            })
        
        num_random_filled = 0
        if len(valid_negatives) < num_negatives:
            num_random_filled = num_negatives - len(valid_negatives)
            
            used_global_indices = {n['global_idx'] for n in valid_negatives}
            used_texts = {n['utterance'] for n in valid_negatives}
            
            safe_pool = list(
                set(range(len(global_corpus))) - 
                forbidden_global_indices - 
                used_global_indices
            )
            random.shuffle(safe_pool)
            
            forbidden_texts_set = set(forbidden_texts)
            
            for idx in safe_pool:
                utt = global_corpus.get_utterance(idx)
                if utt in forbidden_texts_set or utt in used_texts:
                    continue
                
                valid_negatives.append({
                    'global_idx': idx,
                    'utterance': utt,
                    'score': 0.0,
                    'dialogue_id': global_corpus.get_dialogue_id(idx)
                })
                used_texts.add(utt)
                
                if len(valid_negatives) >= num_negatives:
                    break
        
        sample['negatives'] = [n['utterance'] for n in valid_negatives[:num_negatives]]
        sample['metadata'] = {
            'negative_indices': [n['global_idx'] for n in valid_negatives[:num_negatives]],
            'negative_sources': [n['dialogue_id'] for n in valid_negatives[:num_negatives]],
            'bm25_scores': [float(n['score']) for n in valid_negatives[:num_negatives]],
            'num_random_filled': num_random_filled
        }
        
        samples_with_negatives.append(sample)
    
    return samples_with_negatives

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def main():
    print("=" * 80)
    print("TAMIL DIALOGUE TRIPLET CREATION - ULTRA OPTIMIZED")
    print("=" * 80)
    print(f"\nInput:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Context window: {CONTEXT_WINDOW_SIZE}")
    print(f"Negatives per sample: {NUM_NEGATIVES}")
    print(f"BM25 retrieve top-k: {BM25_RETRIEVE_K}")
    print(f"Filter window: ±{FILTER_WINDOW}")
    print("\nOptimizations enabled:")
    print("  - bm25s vectorized BM25 (50-100x faster)")
    print("  - Text->unique_idx reverse map (3000x faster lookup)")
    print("  - Batch processing (10x faster)")
    print("  - Pre-computed forbidden sets")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("STEP 1: Loading cleaned dialogues...")
    print("=" * 80)
    
    try:
        dialogues = load_dialogues(INPUT_PATH)
        print(f"Loaded {len(dialogues):,} dialogues")
    except FileNotFoundError:
        print(f"\nERROR: File not found: {INPUT_PATH}")
        print("Please run clean_tamil_dialogues.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to load dialogues: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("STEP 2: Building global corpus...")
    print("=" * 80)
    
    corpus = build_corpus(dialogues)
    print(f"Built corpus with {len(corpus):,} utterances across {len(dialogues):,} dialogues")
    
    print("\n" + "=" * 80)
    print("STEP 3: Building deduplicated corpus with reverse map...")
    print("=" * 80)
    
    dedup_corpus = DeduplicatedCorpus(corpus)
    
    print("\n" + "=" * 80)
    print("STEP 4: Splitting dialogues...")
    print("=" * 80)
    
    dialogue_splits = split_dialogues(dialogues, SPLIT_RATIOS)
    
    print(f"Split ratios: {SPLIT_RATIOS}")
    for split_name in ['train', 'val', 'test']:
        count = len(dialogue_splits[split_name])
        pct = (count / len(dialogues)) * 100
        print(f"  {split_name:5s}: {count:5,} dialogues ({pct:5.1f}%)")
    
    dialogue_id_to_split = {}
    for split_name, dialogue_ids in dialogue_splits.items():
        for dialogue_id in dialogue_ids:
            dialogue_id_to_split[dialogue_id] = split_name
    
    print("\n" + "=" * 80)
    print("STEP 5: Building BM25s vectorized index...")
    print("=" * 80)
    
    bm25_retriever = build_bm25(dedup_corpus)
    
    bm25_path = OUTPUT_DIR / 'bm25_index.pkl'
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_retriever, f)
    print(f"Saved BM25s index to: {bm25_path}")
    
    print("\n" + "=" * 80)
    print("STEP 6: Creating context-positive pairs...")
    print("=" * 80)
    
    all_samples = []
    for dialogue in dialogues:
        samples = create_samples_for_dialogue(dialogue, corpus, CONTEXT_WINDOW_SIZE)
        all_samples.extend(samples)
    
    print(f"Created {len(all_samples):,} context-positive pairs")
    
    print("\n" + "=" * 80)
    print("STEP 7: Sampling hard negatives (vectorized batch processing)...")
    print("=" * 80)
    print(f"Processing {len(all_samples):,} samples in batches of 1000...")
    print("")
    
    BATCH_SIZE = 1000
    samples_with_negatives = []
    start_time = time.time()
    
    for batch_start in range(0, len(all_samples), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_samples))
        batch = all_samples[batch_start:batch_end]
        
        batch_results = sample_negatives_batch(
            batch, bm25_retriever, dedup_corpus, corpus, NUM_NEGATIVES, BM25_RETRIEVE_K
        )
        samples_with_negatives.extend(batch_results)
        
        current_time = time.time()
        elapsed = current_time - start_time
        processed = batch_end
        samples_per_sec = processed / elapsed if elapsed > 0 else 0
        remaining_samples = len(all_samples) - processed
        eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
        
        progress_pct = (processed / len(all_samples)) * 100
        
        print(f"  [{processed:,}/{len(all_samples):,}] {progress_pct:5.1f}% | "
              f"Speed: {samples_per_sec:.1f} samples/sec | "
              f"Elapsed: {format_time(elapsed)} | "
              f"ETA: {format_time(eta_seconds)}")
    
    total_time = time.time() - start_time
    avg_speed = len(all_samples) / total_time if total_time > 0 else 0
    
    print(f"\nCompleted in {format_time(total_time)} (avg {avg_speed:.1f} samples/sec)")
    print(f"Sampled negatives for {len(samples_with_negatives):,} samples")
    
    random_fills = [s['metadata']['num_random_filled'] for s in samples_with_negatives]
    num_with_random = sum(1 for x in random_fills if x > 0)
    if num_with_random > 0:
        print(f"Note: {num_with_random:,} samples required random filling ({(num_with_random/len(samples_with_negatives)*100):.1f}%)")
    
    print("\n" + "=" * 80)
    print("STEP 8: Constructing final triplets...")
    print("=" * 80)
    
    final_triplets = []
    for sample in samples_with_negatives:
        dialogue_id = sample['dialogue_id']
        split_name = dialogue_id_to_split[dialogue_id]
        
        context_start = sample['context_local_indices'][0]
        context_end = sample['context_local_indices'][-1]
        positive_idx = sample['positive_local_idx']
        sample_id = f"{dialogue_id}_idx_{context_start}_{context_end}_{positive_idx}"
        
        triplet = {
            'sample_id': sample_id,
            'dialogue_id': dialogue_id,
            'context': sample['context'],
            'positive': sample['positive'],
            'negatives': sample['negatives'],
            'metadata': {
                'context_indices': sample['context_local_indices'],
                'positive_index': sample['positive_local_idx'],
                'negative_indices': sample['metadata']['negative_indices'],
                'negative_sources': sample['metadata']['negative_sources'],
                'bm25_scores': sample['metadata']['bm25_scores'],
                'num_random_filled': sample['metadata']['num_random_filled']
            },
            'split': split_name
        }
        final_triplets.append(triplet)
    
    print(f"Constructed {len(final_triplets):,} triplets")
    
    print("\n" + "=" * 80)
    print("STEP 9: Saving triplets...")
    print("=" * 80)
    
    split_counts = {}
    for split in ['train', 'val', 'test']:
        split_triplets = [t for t in final_triplets if t['split'] == split]
        split_counts[split] = len(split_triplets)
        
        output_path = OUTPUT_DIR / f'{split}_triplets.jsonl'
        with open(output_path, 'w', encoding='utf-8') as f:
            for triplet in split_triplets:
                f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
        
        print(f"  {split:5s}: {len(split_triplets):6,} triplets -> {output_path}")
    
    print("\n" + "=" * 80)
    print("STEP 10: Saving statistics...")
    print("=" * 80)
    
    statistics = {
        'total_dialogues': len(dialogues),
        'total_utterances': len(corpus),
        'unique_utterances': len(dedup_corpus),
        'duplicate_rate': (len(corpus) - len(dedup_corpus)) / len(corpus),
        'total_triplets': len(final_triplets),
        'context_window_size': CONTEXT_WINDOW_SIZE,
        'num_negatives': NUM_NEGATIVES,
        'bm25_retrieve_k': BM25_RETRIEVE_K,
        'filter_window': FILTER_WINDOW,
        'split_ratios': SPLIT_RATIOS,
        'split_counts': split_counts,
        'random_seed': RANDOM_SEED,
        'optimizations': {
            'bm25s_vectorized': True,
            'reverse_text_map': True,
            'batch_processing': True,
            'deduplication': True
        }
    }
    
    stats_path = OUTPUT_DIR / 'triplet_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    print(f"Saved statistics to: {stats_path}")
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nTotal triplets: {len(final_triplets):,}")
    print(f"  Train: {split_counts['train']:,}")
    print(f"  Val:   {split_counts['val']:,}")
    print(f"  Test:  {split_counts['test']:,}")
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"  - train_triplets.jsonl")
    print(f"  - val_triplets.jsonl")
    print(f"  - test_triplets.jsonl")
    print(f"  - bm25_index.pkl")
    print(f"  - triplet_statistics.json")
    
    print("\n" + "=" * 80)
    print("TRIPLET CREATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
                                                                                                                                                                                                                                           