import json
import os

VAL_TRIPLETS = '/dist_home/suryansh/dialogue/triplets_output/val_triplets.jsonl'
TEST_TRIPLETS = '/dist_home/suryansh/dialogue/triplets_output/test_triplets.jsonl'
OUTPUT_FILE = '/dist_home/suryansh/dialogue/val_test_ids.json'

def extract_positive_indices():
    print("Extracting positive indices from validation and test sets...")
    
    indices = set()
    
    for triplets_path, name in [(VAL_TRIPLETS, 'validation'), (TEST_TRIPLETS, 'test')]:
        if not os.path.exists(triplets_path):
            print(f"WARNING: {name} triplets file not found: {triplets_path}")
            continue
        
        with open(triplets_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    triplet = json.loads(line)
                    positive_idx = triplet['metadata']['positive_index']
                    indices.add(positive_idx)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"ERROR: Failed to parse triplet: {e}")
                    continue
        
        print(f"Loaded {len(indices)} unique indices so far from {name}")
    
    indices_list = sorted(list(indices))
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(indices_list, f)
    
    print(f"Saved {len(indices_list)} excluded indices to: {OUTPUT_FILE}")

if __name__ == '__main__':
    extract_positive_indices()
                                                                                                                                                                                                                                           
