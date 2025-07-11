import torch
import torch.nn as nn
from typing import List, Dict, Optional

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = {}
        self.hooks = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        
        self.activation_fn = getattr(torch, activation) if hasattr(torch, activation) else torch.relu
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
            if self.hooks:
                self.activations[f'layer_{i}'] = x.detach().clone()
        
        x = self.layers[-1](x)
        return x
    
    def enable_hooks(self):
        self.hooks = ['enabled']
        self.activations = {}
    
    def disable_hooks(self):
        self.hooks = []
        self.activations = {}
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations.copy()

def simple_mlp(input_dim: int = 784, hidden_dims: List[int] = [256, 128], 
               output_dim: int = 10, **kwargs) -> MLP:
    return MLP(input_dim, hidden_dims, output_dim, **kwargs)