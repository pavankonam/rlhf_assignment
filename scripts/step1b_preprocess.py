"""
Step 1b: Data Preprocessing Pipeline
Part 1.1 Task B: Tokenization, balanced splits, edge case handling
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# Configuration
MAX_LENGTH = 512
MODEL_NAME = "gpt2"
TRAIN_VAL_SPLIT = 0.1
SEED = 42

def extract_prompt_and_response(text):
    """
    Extract the prompt and final assistant response from conversation.
    Format: Human: ... Assistant: ... Human: ... Assistant: ...
    """
    last_assistant_idx = text.rfind("\n\nAssistant:")
    
    if last_assistant_idx == -1:
        last_assistant_idx = text.rfind("Assistant:")
        if last_assistant_idx == -1:
            return text, ""
    
    prompt = text[:last_assistant_idx + len("\n\nAssistant:")]
    response = text[last_assistant_idx + len("\n\nAssistant:"):]
    
    return prompt.strip(), response.strip()

def preprocess_example(example, tokenizer, max_length):
    """
    Preprocess a single example:
    - Extract prompt and responses
    - Tokenize
    - Handle length constraints
    """
    chosen_prompt, chosen_response = extract_prompt_and_response(example['chosen'])
    rejected_prompt, rejected_response = extract_prompt_and_response(example['rejected'])
    
    prompt = chosen_prompt
    
    # Tokenize prompt
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length // 2,
        add_special_tokens=True
    )
    
    # Tokenize responses
    chosen_tokens = tokenizer(
        chosen_response,
        truncation=True,
        max_length=max_length // 2,
        add_special_tokens=False
    )
    
    rejected_tokens = tokenizer(
        rejected_response,
        truncation=True,
        max_length=max_length // 2,
        add_special_tokens=False
    )
    
    # Combine prompt + response
    chosen_input_ids = prompt_tokens['input_ids'] + chosen_tokens['input_ids']
    rejected_input_ids = prompt_tokens['input_ids'] + rejected_tokens['input_ids']
    
    # Truncate to max_length
    chosen_input_ids = chosen_input_ids[:max_length]
    rejected_input_ids = rejected_input_ids[:max_length]
    
    # Create attention masks
    chosen_attention_mask = [1] * len(chosen_input_ids)
    rejected_attention_mask = [1] * len(rejected_input_ids)
    
    return {
        'prompt': prompt,
        'chosen_response': chosen_response,
        'rejected_response': rejected_response,
        'prompt_input_ids': prompt_tokens['input_ids'],
        'chosen_input_ids': chosen_input_ids,
        'rejected_input_ids': rejected_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'rejected_attention_mask': rejected_attention_mask,
        'chosen_length': len(chosen_input_ids),
        'rejected_length': len(rejected_input_ids),
    }

def is_valid_example(processed):
    """
    Check if example is valid (handles edge cases):
    - Not a tie (responses are different)
    - Both responses have content
    - Reasonable length
    """
    MIN_RESPONSE_LENGTH = 5
    
    # Check for ties
    if processed['chosen_response'] == processed['rejected_response']:
        return False, "tie"
    
    # Check minimum length
    prompt_len = len(processed['prompt_input_ids'])
    chosen_resp_len = processed['chosen_length'] - prompt_len
    rejected_resp_len = processed['rejected_length'] - prompt_len
    
    if chosen_resp_len < MIN_RESPONSE_LENGTH:
        return False, "chosen_too_short"
    if rejected_resp_len < MIN_RESPONSE_LENGTH:
        return False, "rejected_too_short"
    
    # Check for empty responses
    if not processed['chosen_response'].strip():
        return False, "empty_chosen"
    if not processed['rejected_response'].strip():
        return False, "empty_rejected"
    
    return True, "valid"

def create_balanced_splits(dataset, val_ratio=0.1, seed=42):
    """
    Create balanced training/validation splits.
    Stratify by response length buckets.
    """
    lengths = [ex['chosen_length'] for ex in dataset]
    length_buckets = np.digitize(lengths, bins=[100, 200, 300, 400, 500])
    
    indices = list(range(len(dataset)))
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=length_buckets
    )
    
    return train_idx, val_idx

def main():
    print("="*70)
    print("DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load dataset
    print("\nLoading Anthropic HH-RLHF dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf")
    
    # Process training data
    print(f"\nProcessing {len(dataset['train'])} training examples...")
    
    processed_train = []
    edge_case_stats = {"tie": 0, "chosen_too_short": 0, "rejected_too_short": 0, 
                       "empty_chosen": 0, "empty_rejected": 0, "valid": 0}
    
    for example in tqdm(dataset['train'], desc="Processing train"):
        processed = preprocess_example(example, tokenizer, MAX_LENGTH)
        is_valid, reason = is_valid_example(processed)
        edge_case_stats[reason] += 1
        
        if is_valid:
            processed_train.append(processed)
    
    print(f"\n--- Edge Case Statistics (Train) ---")
    for reason, count in edge_case_stats.items():
        pct = 100 * count / len(dataset['train'])
        print(f"  {reason}: {count} ({pct:.2f}%)")
    
    print(f"\nValid training examples: {len(processed_train)}")
    
    # Process test data
    print(f"\nProcessing {len(dataset['test'])} test examples...")
    
    processed_test = []
    edge_case_stats_test = {"tie": 0, "chosen_too_short": 0, "rejected_too_short": 0, 
                            "empty_chosen": 0, "empty_rejected": 0, "valid": 0}
    
    for example in tqdm(dataset['test'], desc="Processing test"):
        processed = preprocess_example(example, tokenizer, MAX_LENGTH)
        is_valid, reason = is_valid_example(processed)
        edge_case_stats_test[reason] += 1
        
        if is_valid:
            processed_test.append(processed)
    
    print(f"\n--- Edge Case Statistics (Test) ---")
    for reason, count in edge_case_stats_test.items():
        pct = 100 * count / len(dataset['test'])
        print(f"  {reason}: {count} ({pct:.2f}%)")
    
    print(f"\nValid test examples: {len(processed_test)}")
    
    # Create balanced train/validation split
    print(f"\nCreating balanced train/validation split ({100*(1-TRAIN_VAL_SPLIT):.0f}%/{100*TRAIN_VAL_SPLIT:.0f}%)...")
    
    train_idx, val_idx = create_balanced_splits(processed_train, val_ratio=TRAIN_VAL_SPLIT, seed=SEED)
    
    train_data = [processed_train[i] for i in train_idx]
    val_data = [processed_train[i] for i in val_idx]
    
    print(f"  Training set: {len(train_data)} examples")
    print(f"  Validation set: {len(val_data)} examples")
    print(f"  Test set: {len(processed_test)} examples")
    
    # Analyze length distributions
    print("\n--- Length Distribution Analysis ---")
    
    def analyze_lengths(data, name):
        chosen_lens = [ex['chosen_length'] for ex in data]
        rejected_lens = [ex['rejected_length'] for ex in data]
        print(f"\n{name}:")
        print(f"  Chosen - Mean: {np.mean(chosen_lens):.1f}, Std: {np.std(chosen_lens):.1f}")
        print(f"  Rejected - Mean: {np.mean(rejected_lens):.1f}, Std: {np.std(rejected_lens):.1f}")
    
    analyze_lengths(train_data, "Train")
    analyze_lengths(val_data, "Validation")
    analyze_lengths(processed_test, "Test")
    
    # Save processed data
    output_dir = "outputs/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving processed data to {output_dir}/...")
    
    def save_split(data, split_name):
        metadata = {
            'prompts': [ex['prompt'] for ex in data],
            'chosen_responses': [ex['chosen_response'] for ex in data],
            'rejected_responses': [ex['rejected_response'] for ex in data],
        }
        with open(f"{output_dir}/{split_name}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        torch.save({
            'prompt_input_ids': [ex['prompt_input_ids'] for ex in data],
            'chosen_input_ids': [ex['chosen_input_ids'] for ex in data],
            'rejected_input_ids': [ex['rejected_input_ids'] for ex in data],
            'chosen_attention_mask': [ex['chosen_attention_mask'] for ex in data],
            'rejected_attention_mask': [ex['rejected_attention_mask'] for ex in data],
        }, f"{output_dir}/{split_name}_tokenized.pt")
        
        print(f"  Saved {split_name}: {len(data)} examples")
    
    save_split(train_data, "train")
    save_split(val_data, "val")
    save_split(processed_test, "test")
    
    # Save config
    config = {
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'train_val_split': TRAIN_VAL_SPLIT,
        'seed': SEED,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(processed_test),
        'vocab_size': tokenizer.vocab_size,
    }
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir}/")
    
    print("\n✓ Preprocessing complete!")
    print("Next step: python scripts/step2_train_reward_model.py")

if __name__ == "__main__":
    main()
