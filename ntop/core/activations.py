import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from torch.utils.data import DataLoader

class ActivationHook:
    def __init__(self):
        self.activations = {}
        self.handles = []
    
    def hook_fn(self, name: str):
        def fn(module, input, output):
            self.activations[name] = output.detach().clone()
        return fn
    
    def register(self, model: nn.Module, layer_names: List[str]):
        self.clear()
        for name in layer_names:
            if hasattr(model, name):
                layer = getattr(model, name)
                handle = layer.register_forward_hook(self.hook_fn(name))
                self.handles.append(handle)
    
    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}

def extract_activations(model: nn.Module, dataloader: DataLoader, 
                       layer_names: Optional[List[str]] = None,
                       max_batches: Optional[int] = None) -> Dict[str, torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    
    if hasattr(model, 'enable_hooks'):
        model.enable_hooks()
        use_model_hooks = True
    else:
        hook = ActivationHook()
        hook.register(model, layer_names or [])
        use_model_hooks = False
    
    all_activations = {}
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            x = x.to(device)
            _ = model(x)
            
            activations = model.get_activations() if use_model_hooks else hook.activations
            
            for name, acts in activations.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(acts.cpu())
    
    if not use_model_hooks:
        hook.clear()
    else:
        model.disable_hooks()
    
    return {name: torch.cat(acts_list, dim=0) for name, acts_list in all_activations.items()}

def get_layer_activations(activations: Dict[str, torch.Tensor], 
                         layer_name: str) -> torch.Tensor:
    return activations.get(layer_name, torch.empty(0))

def flatten_activations(activations: torch.Tensor) -> torch.Tensor:
    return activations.view(activations.size(0), -1)