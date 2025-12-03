"""
Step 3: PPO-based RLHF Training
Part 2.1 Task A: Implement PPO loss function (clipped surrogate, KL penalty, entropy bonus)
Part 2.1 Task B: Train policy model with reward model, hyperparameter tuning
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    'batch_size': 8,
    'learning_rate': 1e-5,
    'num_epochs': 2,
    'ppo_epochs': 4,
    'clip_ratio': 0.2,        # PPO clipping parameter ε
    'kl_coef': 0.1,           # KL divergence coefficient β
    'entropy_coef': 0.01,     # Entropy bonus coefficient
    'value_coef': 0.5,        # Value loss coefficient
    'gamma': 1.0,
    'lam': 0.95,
    'max_grad_norm': 1.0,
    'target_kl': 0.02,        # Early stopping KL threshold
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_train_prompts': 10000,
    'num_eval_prompts': 500,
}


class PolicyValueModel(nn.Module):
    """Policy model with value head for PPO."""
    
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.model.config.n_embd
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_idx, seq_lens]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        values = self.value_head(last_hidden).squeeze(-1)
        
        return logits, values
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=64, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.model.config.eos_token_id,
            **kwargs
        )


class PPOTrainer:
    """
    PPO Trainer implementing:
    - Clipped surrogate objective: L_CLIP = min(r*A, clip(r,1-ε,1+ε)*A)
    - KL divergence penalty from reference policy
    - Entropy bonus for exploration
    - Value function baseline
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
        
        self.stats = {'training_time': 0, 'samples_generated': 0}
    
    def compute_rewards(self, input_ids, attention_mask):
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        return rewards
    
    def get_log_probs(self, logits, actions, mask=None):
        """Get log probabilities of actions under the policy."""
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        if mask is not None:
            action_log_probs = action_log_probs * mask
            return action_log_probs.sum(-1) / mask.sum(-1).clamp(min=1)
        return action_log_probs.mean(-1)
    
    def compute_kl_divergence(self, policy_logits, ref_logits, mask):
        """Compute KL(policy || reference)."""
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        kl = (torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)).sum(-1)
        
        if mask is not None:
            kl = (kl * mask).sum() / mask.sum()
        else:
            kl = kl.mean()
        return kl
    
    def compute_entropy(self, logits, mask=None):
        """Compute entropy of policy distribution."""
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(torch.exp(log_probs) * log_probs).sum(-1)
        
        if mask is not None:
            entropy = (entropy * mask).sum() / mask.sum()
        else:
            entropy = entropy.mean()
        return entropy
    
    def compute_ppo_loss(self, old_log_probs, new_log_probs, advantages):
        """
        Clipped surrogate objective:
        L_CLIP = E[min(r*A, clip(r, 1-ε, 1+ε)*A)]
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss, ratio.mean()
    
    def train_step(self, prompts_batch):
        """Single PPO training step."""
        self.policy.train()
        start_time = time.time()
        
        # Tokenize prompts
        encodings = self.tokenizer(
            prompts_batch,
            padding=True,
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='pt'
        ).to(self.device)
        
        prompt_len = encodings['input_ids'].shape[1]
        
        # Generate responses
        with torch.no_grad():
            generated_ids = self.policy.generate(
                encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                max_new_tokens=self.config['max_new_tokens']
            )
        
        self.stats['samples_generated'] += generated_ids.shape[0]
        
        gen_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        
        # Get rewards
        rewards = self.compute_rewards(generated_ids, gen_mask)
        
        # Get reference log probs
        with torch.no_grad():
            ref_outputs = self.ref_model(generated_ids, attention_mask=gen_mask)
            ref_logits = ref_outputs.logits
        
        # Get old policy outputs
        with torch.no_grad():
            old_logits, old_values = self.policy(generated_ids, gen_mask)
            response_mask = gen_mask[:, prompt_len:].contiguous()
            old_log_probs = self.get_log_probs(
                old_logits[:, prompt_len-1:-1, :],
                generated_ids[:, prompt_len:],
                response_mask
            )
        
        # Compute advantages
        advantages = rewards - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards
        
        # PPO update epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_kl = 0
        total_entropy = 0
        
        for ppo_epoch in range(self.config['ppo_epochs']):
            new_logits, new_values = self.policy(generated_ids, gen_mask)
            
            new_log_probs = self.get_log_probs(
                new_logits[:, prompt_len-1:-1, :],
                generated_ids[:, prompt_len:],
                response_mask
            )
            
            # PPO clipped loss
            policy_loss, ratio = self.compute_ppo_loss(old_log_probs, new_log_probs, advantages)
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # KL divergence from reference
            kl = self.compute_kl_divergence(new_logits, ref_logits, gen_mask)
            
            # Entropy bonus
            entropy = self.compute_entropy(new_logits, gen_mask)
            
            # Overall loss
            loss = (
                policy_loss
                + self.config['value_coef'] * value_loss
                + self.config['kl_coef'] * kl
                - self.config['entropy_coef'] * entropy
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += kl.item()
            total_entropy += entropy.item()
            
            # Early stopping on KL
            if kl.item() > self.config['target_kl'] * 1.5:
                break
        
        n = ppo_epoch + 1
        self.stats['training_time'] += time.time() - start_time
        
        return {
            'loss': total_loss / n,
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'kl': total_kl / n,
            'entropy': total_entropy / n,
            'reward': rewards.mean().item(),
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
                max_new_tokens=self.config['max_new_tokens']
            )
            
            gen_mask = (generated_ids != self.tokenizer.pad_token_id).long()
            
            rewards = self.compute_rewards(generated_ids, gen_mask)
            all_rewards.extend(rewards.cpu().tolist())
            
            policy_logits, _ = self.policy(generated_ids, attention_mask=gen_mask)
            ref_outputs = self.ref_model(input_ids=generated_ids, attention_mask=gen_mask)
            kl = self.compute_kl_divergence(policy_logits, ref_outputs.logits, gen_mask)
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
                    max_new_tokens=self.config['max_new_tokens']
                )
            
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            samples.append({'prompt': prompt, 'response': response})
        
        return samples


def load_prompts(data_dir, split='train'):
    with open(f"{data_dir}/{split}_metadata.json", 'r') as f:
        metadata = json.load(f)
    return metadata['prompts']


def main():
    print("="*70)
    print("PPO-BASED RLHF TRAINING - Part 2.1")
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
    
    # Load prompts
    print("\nLoading prompts...")
    data_dir = "outputs/processed_data"
    train_prompts = load_prompts(data_dir, 'train')[:config['num_train_prompts']]
    eval_prompts = load_prompts(data_dir, 'val')[:config['num_eval_prompts']]
    print(f"  Train: {len(train_prompts)}, Eval: {len(eval_prompts)}")
    
    # Create models
    print("\nCreating policy model...")
    policy_model = PolicyValueModel(config['policy_model'])
    print(f"  Parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    
    print("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(config['policy_model'])
    
    print("Loading reward model...")
    reward_checkpoint = torch.load(config['reward_model_path'], map_location=device)
    reward_config = AutoConfig.from_pretrained(config['policy_model'])
    reward_model = RewardModel(reward_config)
    reward_model.load_state_dict(reward_checkpoint['model_state_dict'])
    print(f"  Reward model val_acc: {reward_checkpoint.get('val_accuracy', 'N/A')}")
    
    trainer = PPOTrainer(config, policy_model, ref_model, reward_model, tokenizer)
    
    history = {
        'loss': [], 'policy_loss': [], 'value_loss': [],
        'kl': [], 'entropy': [], 'reward': [],
        'eval_reward': [], 'eval_kl': [],
    }
    
    # Initial evaluation
    print("\nInitial evaluation...")
    eval_metrics = trainer.evaluate(eval_prompts[:100])
    print(f"  Initial reward: {eval_metrics['mean_reward']:.4f}")
    print(f"  Initial KL: {eval_metrics['mean_kl']:.4f}")
    history['eval_reward'].append(eval_metrics['mean_reward'])
    history['eval_kl'].append(eval_metrics['mean_kl'])
    
    output_dir = "outputs/models/ppo"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        np.random.shuffle(train_prompts)
        
        epoch_metrics = {k: [] for k in ['loss', 'policy_loss', 'value_loss', 'kl', 'entropy', 'reward']}
        
        pbar = tqdm(range(0, len(train_prompts), config['batch_size']), desc=f"Epoch {epoch+1}")
        for i in pbar:
            batch = train_prompts[i:i + config['batch_size']]
            if len(batch) < config['batch_size']:
                continue
            
            metrics = trainer.train_step(batch)
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix({
                'reward': np.mean(epoch_metrics['reward'][-10:]),
                'kl': np.mean(epoch_metrics['kl'][-10:]),
            })
        
        for k in ['loss', 'policy_loss', 'value_loss', 'kl', 'entropy', 'reward']:
            history[k].append(np.mean(epoch_metrics[k]))
        
        eval_metrics = trainer.evaluate(eval_prompts[:100])
        history['eval_reward'].append(eval_metrics['mean_reward'])
        history['eval_kl'].append(eval_metrics['mean_kl'])
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Reward: {history['reward'][-1]:.4f}, KL: {history['kl'][-1]:.4f}")
        print(f"  Eval Reward: {eval_metrics['mean_reward']:.4f}, KL: {eval_metrics['mean_kl']:.4f}")
        
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
    axes[1, 0].set_title('Policy Loss (Clipped Surrogate)')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['entropy'], marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close()
    
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("PPO TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Train Reward: {history['reward'][-1]:.4f}")
    print(f"  Eval Reward: {history['eval_reward'][-1]:.4f}")
    print(f"  Final KL: {history['eval_kl'][-1]:.4f}")
    print(f"  Training Time: {trainer.stats['training_time']:.1f}s")
    print(f"\nOutputs saved to: {output_dir}/")
    
    print("\n✓ PPO training complete!")
    print("Next step: python scripts/step4_train_grpo.py")

if __name__ == "__main__":
    main()
