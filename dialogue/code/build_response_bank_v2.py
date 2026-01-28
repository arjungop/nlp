import json
import os
from pathlib import Path
from tqdm import tqdm

CLEANED_DIALOGUES = '/dist_home/suryansh/dialogue/tamil_dialogues_clean.jsonl'
OUTPUT_DIR = '/dist_home/suryansh/dialogue/response_bank'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'response_bank.jsonl')

def build_response_bank():
    print("Building response bank from cleaned dialogues...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_utterances = []
    
    with open(CLEANED_DIALOGUES, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing dialogues"):
            line = line.strip()
            if not line:
                continue
            
            try:
                dialogue = json.loads(line)
                if 'utterances' in dialogue:
                    all_utterances.extend(dialogue['utterances'])
                else:
                    print(f"WARNING: No utterances found in dialogue: {dialogue.get('dialogue_id', 'unknown')}")
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse line: {e}")
                continue
    
    print(f"Total utterances collected: {len(all_utterances)}")
    
    unique_utterances = list(set(all_utterances))
    print(f"Unique utterances: {len(unique_utterances)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for idx, utterance in enumerate(tqdm(unique_utterances, desc="Writing response bank")):
            entry = {
                'index': idx,
                'text': utterance
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Response bank saved to: {OUTPUT_FILE}")
    print(f"Total responses: {len(unique_utterances)}")

if __name__ == '__main__':
    build_response_bank()
                                                                                                                                                                                                                                           
