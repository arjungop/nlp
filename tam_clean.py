
#!/usr/bin/env python3
"""
Tamil Dialogue Data Cleaning Script
Filters Tamil/code-mixed utterances, removes English and noise
"""

import json
import re
from collections import Counter
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_PATH = 'data/IndicDialogue/Tamil/Tamil.jsonl'
OUTPUT_PATH = 'tamil_dialogues_clean.jsonl'
MIN_DIALOGUE_LENGTH = 4
RANDOM_SEED = 42

# =============================================================================
# ENGLISH WORDS FOR FILTERING
# =============================================================================
ENGLISH_WORDS = {
    # Question words
    'what', 'how', 'why', 'where', 'when', 'who', 'which', 'whose', 'whom',
    
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
    'this', 'that', 'these', 'those',
    
    # Articles and determiners
    'a', 'an', 'the', 'some', 'any', 'all', 'each', 'every', 'both', 'few', 'many', 'much',
    'more', 'most', 'several', 'enough', 'such', 'other', 'another',
    
    # Be verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    
    # Auxiliary verbs
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'done',
    'can', 'could', 'will', 'would', 'should', 'shall', 'may', 'might', 'must',
    
    # Common verbs
    'go', 'come', 'see', 'know', 'think', 'want', 'get', 'make', 'give', 'take',
    'say', 'tell', 'let', 'help', 'look', 'need', 'feel', 'try', 'leave', 'call',
    'ask', 'work', 'seem', 'find', 'use', 'keep', 'talk', 'turn', 'start', 'show',
    'hear', 'play', 'run', 'move', 'live', 'believe', 'bring', 'happen', 'write',
    'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn',
    'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak',
    'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember',
    'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect',
    'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest', 'raise',
    'pass', 'sell', 'require', 'report', 'decide', 'pull',
    
    # Common nouns
    'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand',
    'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government',
    'company', 'number', 'group', 'problem', 'fact', 'people', 'water', 'room', 'money',
    'story', 'month', 'lot', 'right', 'study', 'book', 'word', 'business', 'issue', 'side',
    'kind', 'head', 'house', 'service', 'friend', 'father', 'power', 'hour', 'game', 'line',
    'end', 'member', 'law', 'car', 'city', 'name', 'president', 'team', 'minute', 'idea',
    'body', 'information', 'back', 'parent', 'face', 'others', 'level', 'office', 'door',
    'health', 'art', 'war', 'history', 'party', 'result', 'change', 'morning', 'reason',
    'research', 'girl', 'guy', 'moment', 'air', 'teacher', 'force', 'education',
    
    # Adjectives
    'good', 'bad', 'great', 'big', 'small', 'long', 'short', 'new', 'old', 'high',
    'different', 'large', 'next', 'early', 'young', 'important', 'public', 'able', 'bad',
    'free', 'human', 'local', 'late', 'hard', 'major', 'better', 'economic', 'strong',
    'possible', 'whole', 'free', 'military', 'true', 'federal', 'international', 'full',
    'special', 'easy', 'clear', 'recent', 'certain', 'personal', 'open', 'red', 'difficult',
    'available', 'likely', 'national', 'political', 'sorry', 'real', 'black', 'white',
    'least', 'poor', 'natural', 'nice', 'beautiful', 'happy', 'sad', 'fine', 'pretty',
    
    # Prepositions
    'in', 'on', 'at', 'to', 'from', 'with', 'by', 'for', 'of', 'about', 'like',
    'through', 'over', 'before', 'between', 'after', 'since', 'without', 'under',
    'within', 'along', 'following', 'across', 'behind', 'beyond', 'plus', 'except',
    'up', 'out', 'around', 'down', 'off', 'above', 'near',
    
    # Conjunctions
    'and', 'or', 'but', 'so', 'because', 'if', 'when', 'while', 'than', 'although',
    'though', 'whether', 'unless', 'until', 'as', 'nor',
    
    # Adverbs
    'not', 'now', 'just', 'very', 'also', 'here', 'well', 'only', 'then', 'there',
    'really', 'never', 'always', 'too', 'again', 'still', 'often', 'however', 'already',
    'quite', 'almost', 'away', 'today', 'tonight', 'yesterday', 'tomorrow', 'soon',
    'probably', 'perhaps', 'maybe', 'finally', 'actually', 'exactly', 'certainly',
    
    # Common expressions
    'yes', 'no', 'okay', 'ok', 'oh', 'yeah', 'yep', 'nope', 'right', 'wrong',
    'please', 'sorry', 'thanks', 'thank', 'hello', 'hi', 'hey', 'bye', 'goodbye',
    'welcome', 'excuse', 'pardon',
    
    # Numbers and time
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'last', 'next', 'previous',
    
    # Common movie/subtitle words
    'movie', 'film', 'scene', 'character', 'actor', 'music', 'song', 'dialogue',
    'subtitle', 'caption', 'credits', 'episode', 'season', 'series', 'show',
}

# =============================================================================
# NOISE PATTERNS
# =============================================================================
NOISE_PATTERNS = [
    r'^[\♪♫🎵🎶\s]+$',           # Music symbols
    r'^\[.*\]$',                   # Bracketed text
    r'^\(.*\)$',                   # Parentheses
    r'^[^\w\u0B80-\u0BFF]+$',     # No alphanumeric or Tamil
    r'^\s*$',                      # Empty/whitespace
    r'^-+$',                       # Only dashes
    r'^\.+$',                      # Only dots
    r'^\d+$',                      # Only numbers
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def has_tamil_script(text):
    """Check if text contains Tamil Unicode characters (U+0B80-U+0BFF)"""
    return any('\u0B80' <= char <= '\u0BFF' for char in text)

def has_latin_script(text):
    """Check if text contains Latin alphabet characters"""
    return any('a' <= char.lower() <= 'z' for char in text)

def is_noise(text):
    """Check if text matches noise patterns"""
    text_stripped = text.strip()
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, text_stripped, re.UNICODE):
            return True
    return False

def categorize_string(text):
    """
    Categorize string into one of four categories:
    - tamil_only: Contains Tamil script, no Latin
    - mixed: Contains both Tamil and Latin
    - english: Contains Latin only
    - noise: Matches noise patterns or neither Tamil nor Latin
    """
    if is_noise(text):
        return 'noise'
    
    has_tamil = has_tamil_script(text)
    has_latin = has_latin_script(text)
    
    if has_tamil:
        return 'mixed' if has_latin else 'tamil_only'
    else:
        return 'english' if has_latin else 'noise'

def count_tamil_chars(text):
    """Count Tamil Unicode characters in text"""
    return sum(1 for char in text if '\u0B80' <= char <= '\u0BFF')

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    print("=" * 80)
    print("TAMIL DIALOGUE CLEANING SCRIPT")
    print("=" * 80)
    print(f"\nInput:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Min dialogue length: {MIN_DIALOGUE_LENGTH}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Load raw data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: Loading raw data...")
    print("=" * 80)
    
    try:
        raw_data = []
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
        
        print(f"✓ Loaded {len(raw_data):,} JSON objects")
    
    except FileNotFoundError:
        print(f"\n✗ ERROR: File not found: {INPUT_PATH}")
        print("Please check the file path and try again.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ ERROR: Failed to load data: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 2: Categorize all strings
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: Categorizing strings...")
    print("=" * 80)
    
    all_raw_strings = []
    all_categories = []
    
    for obj_idx, obj in enumerate(raw_data):
        if 'dialogs' in obj and 'tam' in obj['dialogs']:
            for string in obj['dialogs']['tam']:
                all_raw_strings.append(string)
                all_categories.append(categorize_string(string))
        
        # Progress indicator
        if (obj_idx + 1) % 50 == 0:
            print(f"  Processed {obj_idx + 1}/{len(raw_data)} objects", end='\r')
    
    print(f"\n✓ Categorized {len(all_raw_strings):,} strings")
    
    # Category statistics
    category_counts = Counter(all_categories)
    total = len(all_raw_strings)
    
    print("\nCategory Distribution:")
    print("-" * 80)
    for cat in ['tamil_only', 'mixed', 'english', 'noise']:
        count = category_counts[cat]
        pct = (count / total) * 100
        action = "KEEP" if cat in ['tamil_only', 'mixed'] else "REMOVE"
        print(f"  {cat:<15} {count:>10,}  ({pct:>6.2f}%)  [{action}]")
    
    # -------------------------------------------------------------------------
    # STEP 3: Clean dialogues
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: Cleaning dialogues...")
    print("=" * 80)
    
    cleaned_dialogues = []
    total_utterances_before = 0
    total_utterances_after = 0
    
    for obj_idx, obj in enumerate(raw_data):
        if 'dialogs' in obj and 'tam' in obj['dialogs']:
            original_utterances = obj['dialogs']['tam']
            total_utterances_before += len(original_utterances)
            
            # Keep only tamil_only and mixed utterances
            clean_utterances = [
                u for u in original_utterances 
                if categorize_string(u) in ['tamil_only', 'mixed']
            ]
            
            total_utterances_after += len(clean_utterances)
            
            # Only keep dialogues with minimum length
            if len(clean_utterances) >= MIN_DIALOGUE_LENGTH:
                cleaned_dialogues.append({
                    'dialogue_id': f"dialogue_{obj_idx + 1:05d}",
                    'utterances': clean_utterances
                })
    
    retention_rate = (total_utterances_after / total_utterances_before) * 100
    
    print(f"✓ Utterances: {total_utterances_before:,} → {total_utterances_after:,} ({retention_rate:.2f}% retained)")
    print(f"✓ Dialogues with ≥{MIN_DIALOGUE_LENGTH} utterances: {len(cleaned_dialogues):,}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Save cleaned data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: Saving cleaned data...")
    print("=" * 80)
    
    try:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for dialogue in cleaned_dialogues:
                f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved to: {OUTPUT_PATH}")
    
    except Exception as e:
        print(f"\n✗ ERROR: Failed to save data: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 5: Final statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    
    all_clean_utterances = [u for d in cleaned_dialogues for u in d['utterances']]
    total_chars = sum(len(s) for s in all_clean_utterances)
    tamil_chars = sum(count_tamil_chars(s) for s in all_clean_utterances)
    unique_utterances = len(set(all_clean_utterances))
    
    dialogue_lengths = [len(d['utterances']) for d in cleaned_dialogues]
    mean_length = sum(dialogue_lengths) / len(dialogue_lengths) if dialogue_lengths else 0
    
    print(f"\nDialogues:        {len(cleaned_dialogues):,}")
    print(f"Total utterances: {len(all_clean_utterances):,}")
    print(f"Unique utterances: {unique_utterances:,} ({(unique_utterances/len(all_clean_utterances)*100):.1f}% unique)")
    print(f"Tamil characters: {(tamil_chars/total_chars*100):.2f}%")
    print(f"Mean dialogue length: {mean_length:.1f} utterances")
    print(f"Min dialogue length:  {min(dialogue_lengths)}")
    print(f"Max dialogue length:  {max(dialogue_lengths)}")
    
    print("\n" + "=" * 80)
    print("CLEANING COMPLETE ✓")
    print("=" * 80)