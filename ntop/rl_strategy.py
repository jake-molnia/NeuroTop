import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import math


class RLPruningStrategy:
    """
    RL-based pruning strategy that learns optimal neuron selection.
    Uses Q-learning to update a probability distribution over layers.
    """
    
    def __init__(
        self,
        rf_values: Dict[str, Dict[str, np.ndarray]],
        total_neurons: int,
        layer_names: List[str],
        target_sparsity: float = 0.5,
        prune_steps: int = 50,
        sample_num: int = 10,
        sample_step: int = 1,
        discount_factor: float = 0.9,
        step_length: float = 0.1,
        noise_var: float = 0.04,
        greedy_epsilon: float = 0.4,
        explore_strategy: str = 'cosine',
        Q_FLOP_coef: float = 0.0,
        Q_Para_coef: float = 0.0,
        rf_dim: str = 'rf_0'
    ):
        """
        Args:
            rf_values: Dict mapping layer names to RF importance values
            total_neurons: Total number of neurons across all layers
            layer_names: List of layer names (must match rf_values keys)
            target_sparsity: Final target sparsity (0-1)
            prune_steps: Number of pruning iterations
            sample_num: How many architectures to sample per step
            sample_step: Depth of lookahead sampling
            discount_factor: Q-learning discount factor
            step_length: How much to update distribution per step
            noise_var: Variance for exploration noise
            greedy_epsilon: Initial epsilon for epsilon-greedy
            explore_strategy: 'constant', 'linear', or 'cosine'
            Q_FLOP_coef: Coefficient for FLOPs in Q-value
            Q_Para_coef: Coefficient for parameters in Q-value
            rf_dim: Which RF dimension to use ('rf_0', 'rf_1', etc.)
        """
        self.rf_values = rf_values
        self.layer_names = layer_names
        self.total_neurons = total_neurons
        self.target_sparsity = target_sparsity
        self.prune_steps = prune_steps
        self.sample_num = sample_num
        self.sample_step = sample_step
        self.discount_factor = discount_factor
        self.step_length = step_length
        self.noise_var = noise_var
        self.initial_greedy_epsilon = greedy_epsilon
        self.greedy_epsilon = greedy_epsilon
        self.explore_strategy = explore_strategy
        self.Q_FLOP_coef = Q_FLOP_coef
        self.Q_Para_coef = Q_Para_coef
        self.rf_dim = rf_dim
        
        # Initialize uniform probability distribution over layers
        self.prune_distribution = self._initialize_distribution()
        
        # Calculate neurons to prune per step
        self.prune_filter_ratio = target_sparsity / prune_steps
        self.neurons_per_step = int(total_neurons * self.prune_filter_ratio)
        
        # Replay buffer: [Q_value, prune_distribution]
        n_layers = len(layer_names)
        self.replay_buffer = torch.zeros([sample_num, 1 + n_layers])
        
        # Track current step
        self.current_step = 0
        
        print(f"RL Strategy initialized: {prune_steps} steps, "
              f"{self.neurons_per_step} neurons/step, "
              f"{len(layer_names)} layers")
    
    def _initialize_distribution(self) -> torch.Tensor:
        """Initialize uniform probability distribution based on layer sizes."""
        layer_sizes = []
        for layer_name in self.layer_names:
            if layer_name in self.rf_values and self.rf_dim in self.rf_values[layer_name]:
                layer_sizes.append(len(self.rf_values[layer_name][self.rf_dim]))
            else:
                layer_sizes.append(1)  # Fallback
        
        distribution = torch.tensor(layer_sizes, dtype=torch.float32)
        distribution /= distribution.sum()
        return distribution
    
    def get_neuron_importance(self, use_rf: bool = True) -> Dict[str, np.ndarray]:
        """
        Get neuron importance scores for all layers.
        
        Args:
            use_rf: If True, use RF values; if False, use random
            
        Returns:
            Dict mapping layer names to importance arrays
        """
        importance_scores = {}
        
        for layer_name in self.layer_names:
            if use_rf and layer_name in self.rf_values:
                if self.rf_dim in self.rf_values[layer_name]:
                    # Use RF values (lower RF = less important)
                    rf_vals = self.rf_values[layer_name][self.rf_dim]
                    importance_scores[layer_name] = np.array(rf_vals)
                else:
                    # Fallback to random
                    size = len(next(iter(self.rf_values[layer_name].values())))
                    importance_scores[layer_name] = np.random.rand(size)
            else:
                # Random importance
                if layer_name in self.rf_values:
                    size = len(next(iter(self.rf_values[layer_name].values())))
                else:
                    size = 10  # Fallback size
                importance_scores[layer_name] = np.random.rand(size)
        
        return importance_scores
    
    def sample_pruning_decision(self) -> Tuple[Dict[str, List[int]], torch.Tensor]:
        """
        Sample a pruning decision based on current distribution with noise.
        
        Returns:
            (neurons_to_prune, noised_distribution)
            neurons_to_prune: Dict mapping layer_name -> list of neuron indices
            noised_distribution: The actual distribution used for this sample
        """
        # Add exploration noise
        noise = torch.randn(len(self.prune_distribution)) * self.noise_var * torch.rand(1).item()
        noised_pd = self.prune_distribution + noise
        noised_pd = torch.clamp(noised_pd, min=1e-5)
        noised_pd /= noised_pd.sum()
        
        # Decide how many neurons to prune from each layer
        prune_counts = torch.round(noised_pd * self.neurons_per_step).int()
        
        # Adjust to match exact target
        diff = self.neurons_per_step - prune_counts.sum().item()
        if diff != 0:
            # Add/subtract from layer with highest probability
            max_idx = torch.argmax(noised_pd)
            prune_counts[max_idx] += diff
        
        # Get importance scores
        importance_scores = self.get_neuron_importance(use_rf=True)
        
        # Select specific neurons to prune
        neurons_to_prune = {}
        for i, layer_name in enumerate(self.layer_names):
            count = prune_counts[i].item()
            if count > 0 and layer_name in importance_scores:
                scores = importance_scores[layer_name]
                if len(scores) > count:
                    # Prune neurons with lowest RF values
                    indices = np.argsort(scores)[:count]
                    neurons_to_prune[layer_name] = indices.tolist()
                else:
                    # Can't prune more than exist
                    neurons_to_prune[layer_name] = []
        
        return neurons_to_prune, noised_pd
    
    def compute_Q_value(
        self, 
        accuracy: float, 
        neurons_pruned: int,
        params_pruned: int = 0
    ) -> float:
        """
        Compute Q-value for a pruning decision.
        
        Args:
            accuracy: Model accuracy after pruning (0-1)
            neurons_pruned: Number of neurons pruned
            params_pruned: Number of parameters pruned
            
        Returns:
            Q-value (higher is better)
        """
        compression_ratio = neurons_pruned / self.total_neurons
        param_ratio = params_pruned / self.total_neurons if params_pruned > 0 else compression_ratio
        
        q_value = (accuracy + 
                   self.Q_FLOP_coef * compression_ratio + 
                   self.Q_Para_coef * param_ratio)
        
        return q_value
    
    def update_replay_buffer(
        self, 
        q_value: float, 
        distribution: torch.Tensor, 
        buffer_idx: int
    ):
        """Update replay buffer with new sample."""
        self.replay_buffer[buffer_idx, 0] = q_value
        self.replay_buffer[buffer_idx, 1:] = distribution
    
    def update_distribution(self, ppo_clip: float = 0.25) -> torch.Tensor:
        """
        Update pruning distribution based on best Q-value in replay buffer.
        
        Args:
            ppo_clip: PPO clipping value for stable updates
            
        Returns:
            Change in distribution (for logging)
        """
        original_pd = self.prune_distribution.clone()
        
        # Find best sample in replay buffer
        best_idx = torch.argmax(self.replay_buffer[:, 0])
        best_pd = self.replay_buffer[best_idx, 1:]
        
        # Update towards best
        updated_pd = original_pd + self.step_length * (best_pd - original_pd)
        updated_pd = torch.clamp(updated_pd, min=1e-5)
        updated_pd /= updated_pd.sum()
        
        # Apply PPO clipping for stability
        original_pd[original_pd == 0] = 1e-6
        ratio = updated_pd / original_pd
        updated_pd = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * original_pd
        updated_pd = torch.clamp(updated_pd, min=1e-5)
        updated_pd /= updated_pd.sum()
        
        self.prune_distribution = updated_pd
        
        return updated_pd - original_pd
    
    def step(self):
        """Advance to next pruning step and update exploration rate."""
        self.current_step += 1
        
        # Update epsilon based on strategy
        if self.current_step <= self.prune_steps * 0.1:
            progress = self.current_step / (self.prune_steps * 0.1)
            
            if self.explore_strategy == "linear":
                self.greedy_epsilon = max(
                    (1 - progress) * self.initial_greedy_epsilon, 0
                )
            elif self.explore_strategy == "cosine":
                self.greedy_epsilon = max(
                    0.5 * (1 + math.cos(math.pi * progress)) * self.initial_greedy_epsilon, 0
                )
        else:
            self.greedy_epsilon = 0
    
    def clear_replay_buffer(self):
        """Clear replay buffer for next round of sampling."""
        self.replay_buffer.zero_()
    
    def get_best_decision(self) -> Tuple[Dict[str, List[int]], torch.Tensor]:
        """
        Get best pruning decision using epsilon-greedy.
        
        Returns:
            (neurons_to_prune, distribution_used)
        """
        # Epsilon-greedy: explore or exploit
        if torch.rand(1).item() < self.greedy_epsilon:
            # Random sample
            idx = torch.randint(0, self.sample_num, (1,)).item()
            print(f"  Exploration: using random sample {idx}")
        else:
            # Best sample
            idx = torch.argmax(self.replay_buffer[:, 0])
            print(f"  Exploitation: using best sample {idx}")
        
        best_distribution = self.replay_buffer[idx, 1:]
        
        # Generate decision from this distribution
        neurons_to_prune, _ = self.sample_pruning_decision()
        
        return neurons_to_prune, best_distribution