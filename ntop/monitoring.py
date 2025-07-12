import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader
import numpy as np


class ActivationMonitor:
    def __init__(self, model: nn.Module):
        assert isinstance(model, nn.Module), "Model must be a PyTorch nn.Module"
        self.model = model
        self.activations = {}
        self.hooks = []
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
    
    def set_config(self, output_file: str, **analysis_kwargs):
        self.analysis_config = {
            'output_file': output_file,
            'analysis_kwargs': analysis_kwargs
        }
        self.analysis_results = []
    
    def analyze(self, test_loader: DataLoader, epoch: int, description: str = "", 
                save: bool = False) -> Dict[str, Any]:
        assert hasattr(self, 'analysis_config'), "Analysis configuration not set. Use set_config() to configure analysis."        
            
        # Run the topology analysis
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            data_batch = next(iter(test_loader))
            data = data_batch[0].to(device)
            _ = self.model(data)
        activations = self.get_activations()
        
        from . import analysis
        params = {**self.analysis_config['analysis_kwargs']}
        state = analysis.analyze(activations, **params)
        if description: print(f"{description} (Epoch {epoch}): Neurons: {state['total_neurons']}, Betti: {state['betti_numbers']}")
        should_save = save
        
        if should_save:
            analysis_data = {
                'epoch': epoch,
                'betti_numbers': state['betti_numbers'],
                'total_neurons': state['total_neurons'],
                'n_samples': state['n_samples'],
                'distance_metric': state['distance_metric']
            }
            self.analysis_results.append(analysis_data)
            self._save_analysis()
        
        return state
        
    def _save_analysis(self):
        assert self.analysis_results, "No analysis results to save"
            
        data = {
            'epochs': np.array([r['epoch'] for r in self.analysis_results]),
            'betti_numbers': [r['betti_numbers'] for r in self.analysis_results],
            'total_neurons': np.array([r['total_neurons'] for r in self.analysis_results]),
            'n_samples': np.array([r['n_samples'] for r in self.analysis_results]),
            'config': self.analysis_config
        }
        np.savez(self.analysis_config['output_file'], **data)
    
    def load_analysis(self, filepath: str) -> Dict[str, Any]: return dict(np.load(filepath, allow_pickle=True))
    
    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.activations = {}