"""
Step 2: Reward Model Training
Part 1.2 Task A: Train reward model with pairwise ranking loss
Part 1.2 Task B: Evaluate and analyze errors on 20+ examples
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.reward_model import RewardModel, PairwiseRankingLoss, create_reward_model

# Configuration for GPU
CONFIG = {
    'model_name': 'gpt2',
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'num_epochs': 3,
    'warmup_ratio': 0.1,
    'gradient_accumulation_steps': 2,
    'max_grad_norm': 1.0,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_subset': None,  # None = full dataset
    'val_subset': None,
    'test_subset': None,
}


class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""
    
    def __init__(self, tokenized_path, max_length=512, subset_size=None):
        self.data = torch.load(tokenized_path)
        self.max_length = max_length
        
        if subset_size is not None and subset_size < len(self.data['chosen_input_ids']):
            self.indices = list(range(subset_size))
        else:
            self.indices = list(range(len(self.data['chosen_input_ids'])))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {
            'chosen_input_ids': self.data['chosen_input_ids'][real_idx][:self.max_length],
            'rejected_input_ids': self.data['rejected_input_ids'][real_idx][:self.max_length],
            'chosen_attention_mask': self.data['chosen_attention_mask'][real_idx][:self.max_length],
            'rejected_attention_mask': self.data['rejected_attention_mask'][real_idx][:self.max_length],
        }


def collate_fn(batch, pad_token_id, max_length):
    """Collate function with padding."""
    
    def pad_sequence(sequences, max_len, pad_value):
        padded = []
        for seq in sequences:
            seq = list(seq)[:max_len]
            padding = [pad_value] * (max_len - len(seq))
            padded.append(seq + padding)
        return torch.tensor(padded, dtype=torch.long)
    
    max_chosen = min(max(len(item['chosen_input_ids']) for item in batch), max_length)
    max_rejected = min(max(len(item['rejected_input_ids']) for item in batch), max_length)
    max_len = max(max_chosen, max_rejected)
    
    return {
        'chosen_input_ids': pad_sequence([item['chosen_input_ids'] for item in batch], max_len, pad_token_id),
        'rejected_input_ids': pad_sequence([item['rejected_input_ids'] for item in batch], max_len, pad_token_id),
        'chosen_attention_mask': pad_sequence([item['chosen_attention_mask'] for item in batch], max_len, 0),
        'rejected_attention_mask': pad_sequence([item['rejected_attention_mask'] for item in batch], max_len, 0),
    }


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, config, epoch, global_step):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_acc = 0
    num_batches = 0
    grad_norms = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        chosen_input_ids = batch['chosen_input_ids'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(device)
        
        chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = model(rejected_input_ids, rejected_attention_mask)
        
        loss, acc = loss_fn(chosen_rewards, rejected_rewards)
        loss = loss / config['gradient_accumulation_steps']
        
        loss.backward()
        
        total_loss += loss.item() * config['gradient_accumulation_steps']
        total_acc += acc.item()
        num_batches += 1
        
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            grad_norms.append(grad_norm.item())
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'acc': total_acc / num_batches,
            'grad_norm': grad_norms[-1] if grad_norms else 0,
        })
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'grad_norms': grad_norms,
        'global_step': global_step
    }


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0
    total_acc = 0
    num_batches = 0
    all_chosen_rewards = []
    all_rejected_rewards = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            chosen_input_ids = batch['chosen_input_ids'].to(device)
            rejected_input_ids = batch['rejected_input_ids'].to(device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(device)
            
            chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = model(rejected_input_ids, rejected_attention_mask)
            
            loss, acc = loss_fn(chosen_rewards, rejected_rewards)
            
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1
            
            all_chosen_rewards.extend(chosen_rewards.cpu().tolist())
            all_rejected_rewards.extend(rejected_rewards.cpu().tolist())
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'chosen_rewards': all_chosen_rewards,
        'rejected_rewards': all_rejected_rewards,
    }


def error_analysis(model, tokenizer, metadata_path, tokenized_path, device, num_samples=25):
    """
    Part 1.2 Task B: Error analysis on 20+ examples
    """
    print("\n" + "="*70)
    print("ERROR ANALYSIS (25 examples)")
    print("="*70)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    data = torch.load(tokenized_path)
    
    model.eval()
    errors = []
    correct = []
    
    num_analyze = min(len(data['chosen_input_ids']), 1000)
    
    with torch.no_grad():
        for i in tqdm(range(num_analyze), desc="Analyzing"):
            chosen_ids = torch.tensor([data['chosen_input_ids'][i][:512]]).to(device)
            rejected_ids = torch.tensor([data['rejected_input_ids'][i][:512]]).to(device)
            chosen_mask = torch.tensor([data['chosen_attention_mask'][i][:512]]).to(device)
            rejected_mask = torch.tensor([data['rejected_attention_mask'][i][:512]]).to(device)
            
            chosen_reward = model(chosen_ids, chosen_mask).item()
            rejected_reward = model(rejected_ids, rejected_mask).item()
            
            example = {
                'index': i,
                'prompt': metadata['prompts'][i][:300],
                'chosen': metadata['chosen_responses'][i][:200],
                'rejected': metadata['rejected_responses'][i][:200],
                'chosen_reward': chosen_reward,
                'rejected_reward': rejected_reward,
                'reward_diff': chosen_reward - rejected_reward,
                'chosen_len': len(data['chosen_input_ids'][i]),
                'rejected_len': len(data['rejected_input_ids'][i]),
            }
            
            if chosen_reward <= rejected_reward:
                errors.append(example)
            else:
                correct.append(example)
    
    print(f"\nTotal analyzed: {len(errors) + len(correct)}")
    print(f"Errors: {len(errors)} ({100*len(errors)/(len(errors)+len(correct)):.1f}%)")
    print(f"Correct: {len(correct)} ({100*len(correct)/(len(errors)+len(correct)):.1f}%)")
    
    # Error pattern analysis
    print("\n--- Error Pattern Analysis ---")
    
    error_len_diffs = [e['chosen_len'] - e['rejected_len'] for e in errors]
    correct_len_diffs = [e['chosen_len'] - e['rejected_len'] for e in correct]
    
    print(f"\nLength difference (chosen - rejected):")
    print(f"  Errors - Mean: {np.mean(error_len_diffs):.1f}, Std: {np.std(error_len_diffs):.1f}")
    print(f"  Correct - Mean: {np.mean(correct_len_diffs):.1f}, Std: {np.std(correct_len_diffs):.1f}")
    
    error_reward_mags = [abs(e['reward_diff']) for e in errors]
    correct_reward_mags = [abs(e['reward_diff']) for e in correct]
    
    print(f"\nReward difference magnitude:")
    print(f"  Errors - Mean: {np.mean(error_reward_mags):.4f}")
    print(f"  Correct - Mean: {np.mean(correct_reward_mags):.4f}")
    
    # Print sample errors
    print("\n" + "-"*70)
    print(f"DETAILED ERROR EXAMPLES ({min(num_samples, len(errors))} examples)")
    print("-"*70)
    
    for i, error in enumerate(errors[:num_samples]):
        print(f"\n--- Error {i+1} (Index: {error['index']}) ---")
        print(f"Prompt: {error['prompt']}...")
        print(f"Chosen (reward={error['chosen_reward']:.4f}): {error['chosen']}...")
        print(f"Rejected (reward={error['rejected_reward']:.4f}): {error['rejected']}...")
        print(f"Length diff (chosen-rejected): {error['chosen_len'] - error['rejected_len']} tokens")
    
    return errors, correct


def plot_training_curves(history, output_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Validation', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_acc'], label='Train', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Validation', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    all_grad_norms = []
    for epoch_norms in history['grad_norms']:
        all_grad_norms.extend(epoch_norms)
    axes[1, 0].plot(all_grad_norms)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norms During Training')
    axes[1, 0].grid(True)
    
    if 'chosen_rewards' in history and 'rejected_rewards' in history:
        axes[1, 1].hist(history['chosen_rewards'], bins=50, alpha=0.5, label='Chosen', density=True)
        axes[1, 1].hist(history['rejected_rewards'], bins=50, alpha=0.5, label='Rejected', density=True)
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Reward Distribution (Test Set)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_model_training_curves.png", dpi=150)
    plt.close()
    print(f"Saved training curves to {output_dir}/reward_model_training_curves.png")


def main():
    print("="*70)
    print("REWARD MODEL TRAINING - Part 1.2")
    print("="*70)
    
    config = CONFIG.copy()
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = config['device']
    print(f"\nUsing device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("\nLoading datasets...")
    data_dir = "outputs/processed_data"
    
    train_dataset = PreferenceDataset(
        f"{data_dir}/train_tokenized.pt", 
        config['max_length'],
        subset_size=config['train_subset']
    )
    val_dataset = PreferenceDataset(
        f"{data_dir}/val_tokenized.pt", 
        config['max_length'],
        subset_size=config['val_subset']
    )
    test_dataset = PreferenceDataset(
        f"{data_dir}/test_tokenized.pt", 
        config['max_length'],
        subset_size=config['test_subset']
    )
    
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    
    collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id, config['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate, num_workers=4)
    
    # Create model
    print("\nCreating reward model...")
    model = create_reward_model(config['model_name'], device)
    
    # Setup training
    loss_fn = PairwiseRankingLoss()
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    num_training_steps = len(train_loader) * config['num_epochs'] // config['gradient_accumulation_steps']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    print(f"\nTraining steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'grad_norms': [],
    }
    
    output_dir = "outputs/models/reward_model"
    os.makedirs(output_dir, exist_ok=True)
    
    global_step = 0
    best_val_acc = 0
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(config['num_epochs']):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, config, epoch, global_step)
        global_step = train_metrics['global_step']
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['grad_norms'].append(train_metrics['grad_norms'])
        
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Avg Grad Norm: {np.mean(train_metrics['grad_norms']):.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_accuracy': val_metrics['accuracy'],
            }, f"{output_dir}/best_model.pt")
            print(f"  ✓ Saved best model (val_acc: {best_val_acc:.4f})")
    
    # Final test evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    checkpoint = torch.load(f"{output_dir}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    history['chosen_rewards'] = test_metrics['chosen_rewards']
    history['rejected_rewards'] = test_metrics['rejected_rewards']
    history['test_accuracy'] = test_metrics['accuracy']
    history['test_loss'] = test_metrics['loss']
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    # Error analysis (Part 1.2 Task B)
    errors, correct = error_analysis(
        model, tokenizer,
        f"{data_dir}/test_metadata.json",
        f"{data_dir}/test_tokenized.pt",
        device,
        num_samples=25
    )
    
    # Save results
    error_results = {
        'num_errors': len(errors),
        'num_correct': len(correct),
        'accuracy': len(correct) / (len(errors) + len(correct)),
        'sample_errors': errors[:25],
    }
    with open(f"{output_dir}/error_analysis.json", 'w') as f:
        json.dump(error_results, f, indent=2)
    
    history_save = {k: v for k, v in history.items() if k not in ['chosen_rewards', 'rejected_rewards']}
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history_save, f, indent=2)
    
    print("\n" + "="*70)
    print("REWARD MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"\nOutputs saved to: {output_dir}/")
    
    print("\n✓ Reward model training complete!")
    print("Next step: python scripts/step3_train_ppo.py")

if __name__ == "__main__":
    main()
