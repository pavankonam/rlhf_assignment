"""
Step 5: DPO (Direct Preference Optimization) Training
Part 3: Implement DPO that bypasses explicit reward modeling

DPO Loss: L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
where y_w = chosen, y_l = rejected
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, AutoConfig
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.reward_model import RewardModel

# Configuration
CONFIG = {
    'policy_model': 'gpt2',
    'reward_model_path': 'outputs/models/reward_model/best_model.pt',  # For evaluation only
    'max_length': 512,
    'batch_size': 4,
    'learning_rate': 5e-7,  # Lower LR for DPO stability
    'beta': 0.1,            # DPO temperature parameter
    'num_epochs': 1,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_subset': None,
    'val_subset': None,
}


class DPODataset(Dataset):
    """Dataset for DPO training with preference pairs."""
    
    def __init__(self, tokenized_path, metadata_path, max_length=512, subset_size=None):
        self.data = torch.load(tokenized_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
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
            'prompt': self.metadata['prompts'][real_idx],
            'prompt_input_ids': self.data['prompt_input_ids'][real_idx],
            'chosen_input_ids': self.data['chosen_input_ids'][real_idx][:self.max_length],
            'rejected_input_ids': self.data['rejected_input_ids'][real_idx][:self.max_length],
            'chosen_attention_mask': self.data['chosen_attention_mask'][real_idx][:self.max_length],
            'rejected_attention_mask': self.data['rejected_attention_mask'][real_idx][:self.max_length],
        }


def dpo_collate_fn(batch, pad_token_id, max_length):
    """Collate function for DPO."""
    
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
        'prompts': [item['prompt'] for item in batch],
        'prompt_lens': [len(item['prompt_input_ids']) for item in batch],
        'chosen_input_ids': pad_sequence([item['chosen_input_ids'] for item in batch], max_len, pad_token_id),
        'rejected_input_ids': pad_sequence([item['rejected_input_ids'] for item in batch], max_len, pad_token_id),
        'chosen_attention_mask': pad_sequence([item['chosen_attention_mask'] for item in batch], max_len, 0),
        'rejected_attention_mask': pad_sequence([item['rejected_attention_mask'] for item in batch], max_len, 0),
    }


class DPOTrainer:
    """
    DPO (Direct Preference Optimization) Trainer.
    
    Directly optimizes policy using preference data without explicit reward model.
    Loss: L = -E[log σ(β * (log_ratio_chosen - log_ratio_rejected))]
    where log_ratio = log π(y|x) - log π_ref(y|x)
    """
    
    def __init__(self, config, policy_model, ref_model, reward_model, tokenizer):
        self.config = config
        self.device = config['device']
        self.beta = config['beta']
        
        self.policy = policy_model.to(self.device)
        
        self.ref_model = ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Reward model only for evaluation (not training)
        self.reward_model = reward_model.to(self.device) if reward_model else None
        if self.reward_model:
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad = False
        
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.policy.parameters(), lr=config['learning_rate'])
        
        self.stats = {'training_time': 0, 'samples_processed': 0}
    
    def get_log_probs(self, model, input_ids, attention_mask, prompt_lens):
        """
        Compute log probabilities of the response tokens.
        Only compute log probs for response part (after prompt).
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Compute per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask out prompt tokens (only count response log probs)
        batch_size = input_ids.shape[0]
        response_mask = shift_mask.clone()
        for i in range(batch_size):
            # Zero out prompt tokens
            response_mask[i, :prompt_lens[i]-1] = 0
        
        # Sum log probs over response tokens
        masked_log_probs = token_log_probs * response_mask
        sum_log_probs = masked_log_probs.sum(dim=-1)
        
        # Normalize by number of response tokens
        num_response_tokens = response_mask.sum(dim=-1).clamp(min=1)
        avg_log_probs = sum_log_probs / num_response_tokens
        
        return avg_log_probs
    
    def compute_dpo_loss(self, batch):
        """
        Compute DPO loss:
        L = -E[log σ(β * (log_ratio_w - log_ratio_l))]
        
        where:
        log_ratio_w = log π(y_w|x) - log π_ref(y_w|x)  (chosen)
        log_ratio_l = log π(y_l|x) - log π_ref(y_l|x)  (rejected)
        """
        chosen_ids = batch['chosen_input_ids'].to(self.device)
        rejected_ids = batch['rejected_input_ids'].to(self.device)
        chosen_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_mask = batch['rejected_attention_mask'].to(self.device)
        prompt_lens = batch['prompt_lens']
        
        # Policy log probs
        policy_chosen_logprobs = self.get_log_probs(
            self.policy, chosen_ids, chosen_mask, prompt_lens
        )
        policy_rejected_logprobs = self.get_log_probs(
            self.policy, rejected_ids, rejected_mask, prompt_lens
        )
        
        # Reference log probs (no grad)
        with torch.no_grad():
            ref_chosen_logprobs = self.get_log_probs(
                self.ref_model, chosen_ids, chosen_mask, prompt_lens
            )
            ref_rejected_logprobs = self.get_log_probs(
                self.ref_model, rejected_ids, rejected_mask, prompt_lens
            )
        
        # Log ratios
        chosen_log_ratio = policy_chosen_logprobs - ref_chosen_logprobs
        rejected_log_ratio = policy_rejected_logprobs - ref_rejected_logprobs
        
        # DPO loss
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()
        
        # Accuracy (implicit reward margin)
        accuracy = (logits > 0).float().mean()
        
        # Implicit reward (for monitoring)
        chosen_reward = self.beta * chosen_log_ratio
        rejected_reward = self.beta * rejected_log_ratio
        reward_margin = (chosen_reward - rejected_reward).mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'reward_margin': reward_margin,
            'chosen_reward': chosen_reward.mean(),
            'rejected_reward': rejected_reward.mean(),
        }
    
    def train_step(self, batch):
        self.policy.train()
        start_time = time.time()
        
        results = self.compute_dpo_loss(batch)
        loss = results['loss']
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        
        self.stats['training_time'] += time.time() - start_time
        self.stats['samples_processed'] += batch['chosen_input_ids'].shape[0]
        
        return {
            'loss': loss.item(),
            'accuracy': results['accuracy'].item(),
            'reward_margin': results['reward_margin'].item(),
            'chosen_reward': results['chosen_reward'].item(),
            'rejected_reward': results['rejected_reward'].item(),
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.policy.eval()
        
        total_loss = 0
        total_acc = 0
        total_margin = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            results = self.compute_dpo_loss(batch)
            
            total_loss += results['loss'].item()
            total_acc += results['accuracy'].item()
            total_margin += results['reward_margin'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches,
            'reward_margin': total_margin / num_batches,
        }
    
    @torch.no_grad()
    def evaluate_with_reward_model(self, prompts, max_new_tokens=64):
        """Evaluate using external reward model (for comparison with PPO/GRPO)."""
        if self.reward_model is None:
            return None
        
        self.policy.eval()
        all_rewards = []
        
        for i in range(0, len(prompts), self.config['batch_size']):
            batch = prompts[i:i + self.config['batch_size']]
            
            encodings = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config['max_length'] // 2,
                return_tensors='pt'
            ).to(self.device)
            
            generated_ids = self.policy.generate(
                encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            gen_mask = (generated_ids != self.tokenizer.pad_token_id).long()
            rewards = self.reward_model(generated_ids, gen_mask)
            all_rewards.extend(rewards.cpu().tolist())
        
        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
        }
    
    def generate_samples(self, prompts, num_samples=20):
        self.policy.eval()
        samples = []
        
        for prompt in prompts[:num_samples]:
            encodings = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.config['max_length'] // 2
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.policy.generate(
                    encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                    max_new_tokens=64,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            samples.append({'prompt': prompt, 'response': response})
        
        return samples


def main():
    print("="*70)
    print("DPO (Direct Preference Optimization) TRAINING - Part 3")
    print("="*70)
    
    config = CONFIG.copy()
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    device = config['device']
    print(f"\nUsing device: {device}")
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    tokenizer = AutoTokenizer.from_pretrained(config['policy_model'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    # Load datasets
    print("\nLoading datasets...")
    data_dir = "outputs/processed_data"
    
    train_dataset = DPODataset(
        f"{data_dir}/train_tokenized.pt",
        f"{data_dir}/train_metadata.json",
        config['max_length'],
        config['train_subset']
    )
    val_dataset = DPODataset(
        f"{data_dir}/val_tokenized.pt",
        f"{data_dir}/val_metadata.json",
        config['max_length'],
        config['val_subset']
    )
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    collate = lambda batch: dpo_collate_fn(batch, tokenizer.pad_token_id, config['max_length'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate, num_workers=0)
    
    # Create models
    print("\nCreating policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(config['policy_model'])
    print(f"  Parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    
    print("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(config['policy_model'])
    
    # Load reward model for evaluation comparison
    print("Loading reward model (for evaluation only)...")
    reward_model = None
    if os.path.exists(config['reward_model_path']):
        reward_checkpoint = torch.load(config['reward_model_path'], map_location=device)
        reward_config = AutoConfig.from_pretrained(config['policy_model'])
        reward_model = RewardModel(reward_config)
        reward_model.load_state_dict(reward_checkpoint['model_state_dict'])
        print("  Reward model loaded for evaluation")
    
    # Setup trainer
    trainer = DPOTrainer(config, policy_model, ref_model, reward_model, tokenizer)
    
    # Scheduler
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(trainer.optimizer, num_warmup_steps, num_training_steps)
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_margin': [],
        'val_loss': [], 'val_acc': [], 'val_margin': [],
        'eval_reward': [],
    }
    
    output_dir = "outputs/models/dpo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initial evaluation
    print("\nInitial evaluation...")
    val_metrics = trainer.evaluate(val_loader)
    print(f"  Initial val loss: {val_metrics['loss']:.4f}")
    print(f"  Initial val accuracy: {val_metrics['accuracy']:.4f}")
    
    if reward_model:
        with open(f"{data_dir}/val_metadata.json", 'r') as f:
            val_prompts = json.load(f)['prompts'][:200]
        reward_metrics = trainer.evaluate_with_reward_model(val_prompts)
        print(f"  Initial reward: {reward_metrics['mean_reward']:.4f}")
        history['eval_reward'].append(reward_metrics['mean_reward'])
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        epoch_metrics = {'loss': [], 'accuracy': [], 'reward_margin': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            metrics = trainer.train_step(batch)
            scheduler.step()
            
            epoch_metrics['loss'].append(metrics['loss'])
            epoch_metrics['accuracy'].append(metrics['accuracy'])
            epoch_metrics['reward_margin'].append(metrics['reward_margin'])
            
            pbar.set_postfix({
                'loss': np.mean(epoch_metrics['loss'][-10:]),
                'acc': np.mean(epoch_metrics['accuracy'][-10:]),
            })
        
        history['train_loss'].append(np.mean(epoch_metrics['loss']))
        history['train_acc'].append(np.mean(epoch_metrics['accuracy']))
        history['train_margin'].append(np.mean(epoch_metrics['reward_margin']))
        
        # Validation
        val_metrics = trainer.evaluate(val_loader)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_margin'].append(val_metrics['reward_margin'])
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}, Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Val Loss: {history['val_loss'][-1]:.4f}, Acc: {history['val_acc'][-1]:.4f}")
        
        if reward_model:
            reward_metrics = trainer.evaluate_with_reward_model(val_prompts)
            history['eval_reward'].append(reward_metrics['mean_reward'])
            print(f"  Reward Model Score: {reward_metrics['mean_reward']:.4f}")
        
        torch.save({
            'policy_state_dict': trainer.policy.state_dict(),
            'config': config,
            'epoch': epoch,
            'history': history,
        }, f"{output_dir}/checkpoint_epoch{epoch+1}.pt")
    
    # Save final model
    torch.save({
        'policy_state_dict': trainer.policy.state_dict(),
        'config': config,
        'history': history,
        'stats': trainer.stats,
    }, f"{output_dir}/final_model.pt")
    
    # Generate samples
    print("\nGenerating samples...")
    with open(f"{data_dir}/val_metadata.json", 'r') as f:
        val_prompts = json.load(f)['prompts']
    samples = trainer.generate_samples(val_prompts, num_samples=20)
    with open(f"{output_dir}/generated_samples.json", 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Plot curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('DPO Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_acc'], label='Train', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Preference Prediction Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['train_margin'], label='Train', marker='o')
    axes[1, 0].plot(history['val_margin'], label='Val', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Reward Margin')
    axes[1, 0].set_title('Implicit Reward Margin (chosen - rejected)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    if history['eval_reward']:
        axes[1, 1].plot(history['eval_reward'], marker='o', color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Reward Model Score')
        axes[1, 1].set_title('External Reward Model Evaluation')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close()
    
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("DPO TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"  Val Accuracy: {history['val_acc'][-1]:.4f}")
    if history['eval_reward']:
        print(f"  Final Reward Score: {history['eval_reward'][-1]:.4f}")
    print(f"  Training Time: {trainer.stats['training_time']:.1f}s")
    print(f"\nOutputs saved to: {output_dir}/")
    
    print("\n✓ DPO training complete!")
    print("Next step: python scripts/step6_evaluate.py")

if __name__ == "__main__":
    main()
