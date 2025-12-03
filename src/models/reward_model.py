"""
Reward Model Implementation
Part 1.2 Task A: GPT-2 based reward model with scalar head
Uses pairwise ranking loss: L = -log(σ(r(x, y_chosen) - r(x, y_rejected)))
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel, AutoConfig

class RewardModel(GPT2PreTrainedModel):
    """
    Reward Model based on GPT-2.
    Adds a scalar reward head on top of the transformer.
    Uses mean pooling for more stable reward estimation.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        
        # Reward head with hidden layer
        self.reward_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd, 1)
        )
        
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, return_hidden_states=False):
        """
        Forward pass to compute reward for a sequence.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return hidden states
            
        Returns:
            rewards: Scalar reward for each sequence [batch_size]
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)
        
        rewards = self.reward_head(pooled).squeeze(-1)
        
        if return_hidden_states:
            return rewards, outputs.hidden_states
        
        return rewards


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for reward model training.
    L = -log(σ(r(x, y_chosen) - r(x, y_rejected)))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, chosen_rewards, rejected_rewards):
        """
        Compute pairwise ranking loss.
        
        Args:
            chosen_rewards: Rewards for chosen responses [batch_size]
            rejected_rewards: Rewards for rejected responses [batch_size]
            
        Returns:
            loss: Scalar loss value
            accuracy: Prediction accuracy (chosen > rejected)
        """
        reward_diff = chosen_rewards - rejected_rewards
        loss = -torch.nn.functional.logsigmoid(reward_diff).mean()
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return loss, accuracy


def create_reward_model(model_name="gpt2", device="cuda"):
    """Create a reward model from pretrained GPT-2."""
    config = AutoConfig.from_pretrained(model_name)
    model = RewardModel.from_pretrained(model_name, config=config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model
