import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader


class ActivationMonitor:
    def __init__(self, model: nn.Module):
        assert isinstance(model, nn.Module), "Model must be a PyTorch nn.Module"
        self.model = model
        self.activations = {}
        self.hooks = []
        self.snapshots = []
        self._register_hooks()
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(handle)
            elif isinstance(module, (nn.Conv2d, nn.Conv1d, nn.LSTM, nn.GRU, nn.RNN)):
                raise NotImplementedError(f"Layer type {type(module).__name__} not supported. MLP-only.")
    
    def _hook_fn(self, name: str): return lambda module, input, output: self.activations.__setitem__(name, output.detach().clone())
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        assert self.activations, "No activations captured. Run a forward pass first."
        return self.activations.copy()
    
    def capture_snapshot(self, dataloader: DataLoader, epoch: int, max_samples: int = 500):
        self.model.eval()
        all_activations = {}
        with torch.no_grad():
            samples_collected = 0
            for batch_idx, (x, _) in enumerate(dataloader):
                if samples_collected >= max_samples:  break
                x = x.to(next(self.model.parameters()).device)
                _ = self.model(x)
                batch_size = x.size(0)
                for name, acts in self.activations.items():
                    if name not in all_activations: all_activations[name] = []
                    all_activations[name].append(acts.cpu())
                samples_collected += batch_size
        unified_activations = {
            name: torch.cat(acts_list, dim=0)[:max_samples] 
            for name, acts_list in all_activations.items()
        }
        snapshot = {
            'epoch': epoch,
            'activations': unified_activations,
            'samples_used': min(samples_collected, max_samples)
        }
        self.snapshots.append(snapshot)
    
    def get_evolution(self) -> Dict[str, Any]:
        assert self.snapshots, "No snapshots captured. Use capture_snapshot() during training."
        return {
            'epochs': [s['epoch'] for s in self.snapshots],
            'snapshots': self.snapshots
        }
    
    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.activations = {}