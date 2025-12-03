"""
Step 4: GRPO (Group Relative Policy Optimization) Training
Part 2.2 Task A: Group-based advantage estimation (group size 4-8)
Part 2.2 Task B: Train and compare with PPO (stability, efficiency, quality)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.reward_model import RewardModel

# Configuration
CONFIG = {
    'policy_model': 'gpt2',
    'reward_model_path': 'outputs/models/reward_model/best_model.pt',
    'max_length': 256,
    'max_new_tokens': 64,
    'batch_size': 4,      # Number of prompts per batch
    'group_size': 4,      # G: Number of responses per prompt
    'learning_rate': 1e-5,
    'num_epochs': 2,
    'kl_coef': 0.1,
    'max_grad_norm': 1.0,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_train_prompts': 10000,
    'num_eval_prompts': 500,
}


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer.
    
    Key differences from PPO:
    1. Samples G responses per prompt (group)
    2. Computes advantages relative to group mean: A_i = (r_i - mean(r)) / std(r)
    3. Uses simplified policy gradient without clipping
    """
    
    def __init__(self, config, policy_model, ref_model, reward_model, tokenizer):
        self.config = config
        self.device = config['device']
        
        self.policy = policy_model.to(self.device)
        
        self.ref_model = ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.reward_model = reward_model.to(self.device)
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.policy.parameters(), lr=config['learning_rate'])
        
        # Efficiency tracking for Part 2.2 Task B comparison
        self.stats = {
            'training_time': 0,
            'samples_generated': 0,
            'memory_peak': 0,
            'time_per_iteration': [],
        }
    
    def compute_rewards(self, input_ids, attention_mask):
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        return rewards
    
    def compute_group_advantages(self, rewards, group_size):
        """
        Compute advantages relative to group mean.
        A_i = (r_i - μ_group) / σ_group
        
        Args:
            rewards: [batch_size * group_size] rewards
            group_size: G, number of responses per prompt
            
        Returns:
            advantages: Normalized group-relative advantages
        """
        batch_size = rewards.shape[0] // group_size
        rewards_grouped = rewards.view(batch_size, group_size)
        
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        
        advantages = (rewards_grouped - group_mean) / group_std
        
        return advantages.view(-1)
    
    def get_log_probs(self, logits, actions, mask=None):
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        if mask is not None:
            action_log_probs = action_log_probs * mask
            return action_log_probs.sum(-1) / mask.sum(-1).clamp(min=1)
        return action_log_probs.mean(-1)
    
    def compute_kl_penalty(self, policy_logits, ref_logits, mask):
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        kl = (torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)).sum(-1)
        
        if mask is not None:
            kl = (kl * mask).sum() / mask.sum()
        else:
            kl = kl.mean()
        return kl
    
    def train_step(self, prompts_batch):
        """
        Single GRPO training step.
        For each prompt, generate G responses and compute group-relative advantages.
        """
        self.policy.train()
        start_time = time.time()
        
        group_size = self.config['group_size']
        batch_size = len(prompts_batch)
        
        # Expand prompts: each prompt repeated G times
        expanded_prompts = []
        for prompt in prompts_batch:
            expanded_prompts.extend([prompt] * group_size)
        
        encodings = self.tokenizer(
            expanded_prompts,
            padding=True,
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='pt'
        ).to(self.device)
        
        prompt_len = encodings['input_ids'].shape[1]
        
        # Generate G responses per prompt (with sampling for diversity)
        with torch.no_grad():
            generated_ids = self.policy.generate(
                encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                max_new_tokens=self.config['max_new_tokens'],
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        self.stats['samples_generated'] += generated_ids.shape[0]
        
        gen_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        
        # Get rewards for all G*batch_size responses
        rewards = self.compute_rewards(generated_ids, gen_mask)
        
        # Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards, group_size)
        
        # Get reference log probs
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=generated_ids, attention_mask=gen_mask)
            ref_logits = ref_outputs.logits
        
        # Forward pass through policy
        policy_outputs = self.policy(input_ids=generated_ids, attention_mask=gen_mask)
        policy_logits = policy_outputs.logits
        
        # Compute log probs for generated tokens (response part only)
        response_mask = gen_mask[:, prompt_len:].contiguous()
        log_probs = self.get_log_probs(
            policy_logits[:, prompt_len-1:-1, :],
            generated_ids[:, prompt_len:],
            response_mask
        )
        
        # GRPO loss: -E[A * log π(a|s)]
        # Simplified policy gradient weighted by group-relative advantages
        policy_loss = -(advantages * log_probs).mean()
        
        # KL penalty from reference
        kl = self.compute_kl_penalty(policy_logits, ref_logits, gen_mask)
        
        # Total loss
        loss = policy_loss + self.config['kl_coef'] * kl
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        
        iter_time = time.time() - start_time
        self.stats['training_time'] += iter_time
        self.stats['time_per_iteration'].append(iter_time)
        
        if torch.cuda.is_available():
            self.stats['memory_peak'] = max(
                self.stats['memory_peak'],
                torch.cuda.max_memory_allocated() / 1e9
            )
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'kl': kl.item(),
            'mean_reward': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'advantage_std': advantages.std().item(),
        }
    
    @torch.no_grad()
    def evaluate(self, prompts):
        self.policy.eval()
        
        all_rewards = []
        all_kls = []
        
        for i in range(0, len(prompts), self.config['batch_size']):
            batch = prompts[i:i + self.config['batch_size']]
            
            encodings = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config['max_length'],
                return_tensors='pt'
            ).to(self.device)
            
            generated_ids = self.policy.generate(
                encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                max_new_tokens=self.config['max_new_tokens'],
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            gen_mask = (generated_ids != self.tokenizer.pad_token_id).long()
            
            rewards = self.compute_rewards(generated_ids, gen_mask)
            all_rewards.extend(rewards.cpu().tolist())
            
            policy_outputs = self.policy(input_ids=generated_ids, attention_mask=gen_mask)
            ref_outputs = self.ref_model(input_ids=generated_ids, attention_mask=gen_mask)
            kl = self.compute_kl_penalty(policy_outputs.logits, ref_outputs.logits, gen_mask)
            all_kls.append(kl.item())
        
        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_kl': np.mean(all_kls),
        }
    
    def generate_samples(self, prompts, num_samples=20):
        self.policy.eval()
        samples = []
        
        for prompt in prompts[:num_samples]:
            encodings = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.config['max_length']
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.policy.generate(
                    encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                    max_new_tokens=self.config['max_new_tokens'],
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            samples.append({'prompt': prompt, 'response': response})
        
        return samples
    
    def get_efficiency_stats(self):
        """Get computational efficiency statistics for Part 2.2 Task B comparison."""
        return {
            'total_training_time': self.stats['training_time'],
            'total_samples_generated': self.stats['samples_generated'],
            'time_per_sample': self.stats['training_time'] / max(self.stats['samples_generated'], 1),
            'avg_time_per_iteration': np.mean(self.stats['time_per_iteration']) if self.stats['time_per_iteration'] else 0,
            'memory_peak_gb': self.stats['memory_peak'],
        }


def load_prompts(data_dir, split='train'):
    with open(f"{data_dir}/{split}_metadata.json", 'r') as f:
        metadata = json.load(f)
    return metadata['prompts']


def main():
    print("="*70)
    print("GRPO (Group Relative Policy Optimization) TRAINING - Part 2.2")
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
    
    print("\nLoading prompts...")
    data_dir = "outputs/processed_data"
    train_prompts = load_prompts(data_dir, 'train')[:config['num_train_prompts']]
    eval_prompts = load_prompts(data_dir, 'val')[:config['num_eval_prompts']]
    print(f"  Train: {len(train_prompts)}, Eval: {len(eval_prompts)}")
    
    print("\nCreating policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(config['policy_model'])
    print(f"  Parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    
    print("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(config['policy_model'])
    
    print("Loading reward model...")
    reward_checkpoint = torch.load(config['reward_model_path'], map_location=device)
    reward_config = AutoConfig.from_pretrained(config['policy_model'])
    reward_model = RewardModel(reward_config)
    reward_model.load_state_dict(reward_checkpoint['model_state_dict'])
    
    trainer = GRPOTrainer(config, policy_model, ref_model, reward_model, tokenizer)
    
    history = {
        'loss': [], 'policy_loss': [], 'kl': [],
        'reward': [], 'reward_std': [],
        'eval_reward': [], 'eval_kl': [],
    }
    
    # Initial evaluation
    print("\nInitial evaluation...")
    eval_metrics = trainer.evaluate(eval_prompts[:100])
    print(f"  Initial reward: {eval_metrics['mean_reward']:.4f}")
    print(f"  Initial KL: {eval_metrics['mean_kl']:.4f}")
    history['eval_reward'].append(eval_metrics['mean_reward'])
    history['eval_kl'].append(eval_metrics['mean_kl'])
    
    output_dir = "outputs/models/grpo"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        np.random.shuffle(train_prompts)
        
        epoch_metrics = {k: [] for k in ['loss', 'policy_loss', 'kl', 'mean_reward', 'reward_std']}
        
        pbar = tqdm(range(0, len(train_prompts), config['batch_size']), desc=f"Epoch {epoch+1}")
        for i in pbar:
            batch = train_prompts[i:i + config['batch_size']]
            if len(batch) < config['batch_size']:
                continue
            
            metrics = trainer.train_step(batch)
            
            for k in epoch_metrics.keys():
                if k in metrics:
                    epoch_metrics[k].append(metrics[k])
            
            pbar.set_postfix({
                'reward': np.mean(epoch_metrics['mean_reward'][-10:]),
                'kl': np.mean(epoch_metrics['kl'][-10:]),
            })
        
        for k in ['loss', 'policy_loss', 'kl']:
            history[k].append(np.mean(epoch_metrics[k]))
        history['reward'].append(np.mean(epoch_metrics['mean_reward']))
        history['reward_std'].append(np.mean(epoch_metrics['reward_std']))
        
        eval_metrics = trainer.evaluate(eval_prompts[:100])
        history['eval_reward'].append(eval_metrics['mean_reward'])
        history['eval_kl'].append(eval_metrics['mean_kl'])
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Reward: {history['reward'][-1]:.4f} (std: {history['reward_std'][-1]:.4f})")
        print(f"  Train KL: {history['kl'][-1]:.4f}")
        print(f"  Eval Reward: {eval_metrics['mean_reward']:.4f}")
        print(f"  Eval KL: {eval_metrics['mean_kl']:.4f}")
        
        torch.save({
            'policy_state_dict': trainer.policy.state_dict(),
            'config': config,
            'epoch': epoch,
            'history': history,
        }, f"{output_dir}/checkpoint_epoch{epoch+1}.pt")
    
    # Get efficiency stats for comparison
    efficiency = trainer.get_efficiency_stats()
    
    # Save final model
    torch.save({
        'policy_state_dict': trainer.policy.state_dict(),
        'config': config,
        'history': history,
        'efficiency': efficiency,
    }, f"{output_dir}/final_model.pt")
    
    # Generate samples
    print("\nGenerating samples...")
    samples = trainer.generate_samples(eval_prompts, num_samples=20)
    with open(f"{output_dir}/generated_samples.json", 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Plot curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['reward'], label='Train', marker='o')
    axes[0, 0].plot(history['eval_reward'][1:], label='Eval', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['kl'], label='Train', marker='o')
    axes[0, 1].plot(history['eval_kl'][1:], label='Eval', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence from Reference')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['policy_loss'], marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Policy Loss')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['reward_std'], marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reward Std')
    axes[1, 1].set_title('Within-Group Reward Variance')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close()
    
    history['efficiency'] = efficiency
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("GRPO TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Train Reward: {history['reward'][-1]:.4f}")
    print(f"  Eval Reward: {history['eval_reward'][-1]:.4f}")
    print(f"  Final KL: {history['eval_kl'][-1]:.4f}")
    print(f"\nEfficiency Statistics (for Part 2.2 Task B comparison):")
    print(f"  Total training time: {efficiency['total_training_time']:.1f}s")
    print(f"  Samples generated: {efficiency['total_samples_generated']}")
    print(f"  Time per sample: {efficiency['time_per_sample']*1000:.2f}ms")
    print(f"  Memory peak: {efficiency['memory_peak_gb']:.2f}GB")
    print(f"\nOutputs saved to: {output_dir}/")
    
    print("\n✓ GRPO training complete!")
    print("Next step: python scripts/step5_train_dpo.py")

if __name__ == "__main__":
    main()
