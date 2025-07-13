import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader
import numpy as np


class ModelAdapter:
    def should_hook(self, name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)
    
    def process_batch(self, data_batch, device):
        if isinstance(data_batch, (list, tuple)):
            return data_batch[0].to(device), {}
        else:
            return data_batch.to(device), {}
    
    def process_output(self, output, strategy):
        if strategy == 'auto':
            if output.dim() == 3 and output.size(1) > 1:
                return output[:, 0, :].detach().clone()
            else:
                return output.detach().clone()
        elif strategy == 'cls':
            if output.dim() == 3:
                return output[:, 0, :].detach().clone()
            else:
                return output.detach().clone()
        elif strategy == 'full':
            return output.detach().clone()
        elif strategy == 'mean':
            if output.dim() == 3:
                return output.mean(dim=1).detach().clone()
            else:
                return output.detach().clone()
        else:
            return output.detach().clone()


class TransformerAdapter(ModelAdapter):
    def should_hook(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, nn.Linear):
            return True
        if hasattr(module, 'weight') and hasattr(module, 'bias'):
            module_type = str(type(module)).lower()
            return any(x in module_type for x in ['linear', 'dense', 'projection'])
        return False
    
    def process_batch(self, data_batch, device):
        if isinstance(data_batch, dict):
            bert_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids'}
            data = {k: v.to(device) for k, v in data_batch.items() 
                   if k in bert_keys and hasattr(v, 'to')}
            return None, data
        elif isinstance(data_batch, (list, tuple)):
            return data_batch[0].to(device), {}
        else:
            return data_batch.to(device), {}


def _detect_model_type(model: nn.Module) -> str:
    model_type = str(type(model)).lower()
    if any(x in model_type for x in ['bert', 'roberta', 'distilbert', 'transformer']):
        return 'transformer'
    
    module_names = [name.lower() for name, _ in model.named_modules()]
    if any('attention' in name for name in module_names):
        return 'transformer'
    
    raise ValueError("Model type could not be detected. Please specify model_type explicitly.")


class ActivationMonitor:
    def __init__(self, model: nn.Module, model_type: Optional[str] = None):
        assert isinstance(model, nn.Module), "Model must be a PyTorch nn.Module"
        self.model = model
        self.activations = {}
        self.hooks = []
        
        model_type = model_type or _detect_model_type(model)
        self.adapter = TransformerAdapter() if model_type == 'transformer' else ModelAdapter()
        self.sequence_strategy = 'auto'
        
        self._register_hooks()
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if self.adapter.should_hook(name, module):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(handle)
    
    def _hook_fn(self, name: str):
        def hook(module, input, output):
            processed_output = self.adapter.process_output(output, self.sequence_strategy)
            self.activations[name] = processed_output
        return hook
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        assert self.activations, "No activations captured. Run a forward pass first."
        return self.activations.copy()
    
    def set_config(self, output_file: str, sequence_strategy: str = 'auto', **analysis_kwargs):
        self.sequence_strategy = sequence_strategy
        self.analysis_config = {
            'output_file': output_file,
            'analysis_kwargs': analysis_kwargs
        }
        self.topology_states = []
    
    def analyze(self, test_loader: DataLoader, epoch: int, description: str = "", 
                save: bool = False) -> Dict[str, Any]:
        assert hasattr(self, 'analysis_config'), "Analysis configuration not set. Use set_config() to configure analysis."        
            
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            data_batch = next(iter(test_loader))
            args, kwargs = self.adapter.process_batch(data_batch, device)
            
            if args is not None:
                _ = self.model(args)
            else:
                _ = self.model(**kwargs)
        
        activations = self.get_activations()
        
        from . import analysis
        params = {**self.analysis_config['analysis_kwargs']}
        state = analysis.analyze(activations, **params)
        rf_info = self._format_rf_output(state)
        if description: 
            print(f"{description} (Epoch {epoch}): Neurons: {state['total_neurons']}, Betti: {state['betti_numbers']}{rf_info}")
        if save:
            topology_data = self._prepare_save_data(state, epoch)
            self.topology_states.append(topology_data)
            self._save_analysis()
        return state
        
    def _save_analysis(self):
        assert self.topology_states, "No topology states to save"
        
        data = {
            'epochs': np.array([s['epoch'] for s in self.topology_states]),
            'topology_states': self.topology_states,
            'config': self.analysis_config
        }
        np.savez_compressed(self.analysis_config['output_file'], **data)
    
    def load_analysis(self, filepath: str) -> Dict[str, Any]: 
        return dict(np.load(filepath, allow_pickle=True))
    
    def _format_rf_output(self, state: Dict[str, Any]) -> str:
        assert 'rf_values' in state, "State must contain 'rf_values' key"
        rf_values = state['rf_values']
        rf_dims = set()
        for layer_rf in rf_values.values():
            if isinstance(layer_rf, dict):
                rf_dims.update(layer_rf.keys())
        assert rf_dims, "No RF dimensions found in rf_values"
        
        rf_summary = {}
        for dim in sorted(rf_dims):
            all_rf_values = []
            for layer_rf in rf_values.values():
                if isinstance(layer_rf, dict) and dim in layer_rf:
                    all_rf_values.extend(layer_rf[dim])
            if all_rf_values:
                rf_summary[dim] = np.median(all_rf_values)
        
        if rf_summary:
            rf_str = ", ".join([f"{dim}_median: {val:.4f}" for dim, val in rf_summary.items()])
            return f", {rf_str}"
        return ""

    def get_rf_evolution(self) -> Dict[str, Any]:
        assert self.topology_states, "No topology states available. Run analysis with save=True first."
        epochs = [s['epoch'] for s in self.topology_states]
        rf_evolution = {'epochs': epochs}
        
        if not self.topology_states: 
            return rf_evolution
        
        first_rf_values = self.topology_states[0].get('rf_values', {})
        layer_names = list(first_rf_values.keys())
        
        rf_dims = set()
        for layer_rf in first_rf_values.values():
            if isinstance(layer_rf, dict):
                rf_dims.update(layer_rf.keys())
        
        rf_stats_evolution = {}
        for layer in layer_names:
            rf_stats_evolution[layer] = {}
            for dim in sorted(rf_dims):
                rf_stats_evolution[layer][dim] = {
                    'mean': [],
                    'median': [],
                    'std': []
                }
                for state in self.topology_states:
                    layer_rf = state.get('rf_values', {}).get(layer, {})
                    if isinstance(layer_rf, dict) and dim in layer_rf:
                        dim_rf = layer_rf[dim]
                        if len(dim_rf) > 0:
                            rf_stats_evolution[layer][dim]['mean'].append(np.mean(dim_rf))
                            rf_stats_evolution[layer][dim]['median'].append(np.median(dim_rf))
                            rf_stats_evolution[layer][dim]['std'].append(np.std(dim_rf))
                        else:
                            rf_stats_evolution[layer][dim]['mean'].append(0.0)
                            rf_stats_evolution[layer][dim]['median'].append(0.0)
                            rf_stats_evolution[layer][dim]['std'].append(0.0)
                    else:
                        rf_stats_evolution[layer][dim]['mean'].append(0.0)
                        rf_stats_evolution[layer][dim]['median'].append(0.0)
                        rf_stats_evolution[layer][dim]['std'].append(0.0)
        
        rf_evolution['rf_stats_evolution'] = rf_stats_evolution
        return rf_evolution
    
    def _prepare_save_data(self, state: Dict[str, Any], epoch: int) -> Dict[str, Any]:
        topology_data = {
            'epoch': epoch,
            'neuron_matrix': state['neuron_matrix'],
            'neuron_info': state['neuron_info'],
            'distance_matrix': state['distance_matrix'],
            'persistence': state['persistence'],
            'betti_numbers': state['betti_numbers'],
            'total_neurons': state['total_neurons'],
            'n_samples': state['n_samples'],
            'distance_metric': state['distance_metric']
        }
        
        if 'rf_values' in state:
            topology_data['rf_values'] = state['rf_values']
        
        return topology_data

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.activations = {}