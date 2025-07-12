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
        self.topology_states = []
    
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
        rf_info = self._format_rf_output(state)
        if description: print(f"{description} (Epoch {epoch}): Neurons: {state['total_neurons']}, Betti: {state['betti_numbers']}{rf_info}")
        if save:
            topology_data = self._prepare_save_data(state, epoch)
            self.topology_states.append(topology_data)
            self._save_analysis()
        return state
    
    def _format_rf_output(self, state: Dict[str, Any]) -> str:
        """Format rf information for console output"""
        if 'rf_values' not in state: return ""
        rf_values = state['rf_values']
        # Compute overall median rf across all layers
        all_rf_values = []
        for layer_rf in rf_values.values():
            all_rf_values.extend(layer_rf)
        if all_rf_values:
            overall_median = np.median(all_rf_values)
            return f", rf_median: {overall_median:.4f}"
        return ""
        
    def _save_analysis(self):
        assert self.topology_states, "No topology states to save"
        
        data = {
            'epochs': np.array([s['epoch'] for s in self.topology_states]),
            'topology_states': self.topology_states,
            'config': self.analysis_config
        }
        np.savez_compressed(self.analysis_config['output_file'], **data)
    
    def load_analysis(self, filepath: str) -> Dict[str, Any]: return dict(np.load(filepath, allow_pickle=True))
    
    def get_rf_evolution(self) -> Dict[str, Any]:
        """Get evolution of rf values over epochs"""
        assert self.topology_states, "No topology states available. Run analysis with save=True first."
        epochs = [s['epoch'] for s in self.topology_states]
        rf_evolution = {'epochs': epochs}
        
        if not self.topology_states: return rf_evolution
        
        # Get layer names from first state
        first_rf_values = self.topology_states[0].get('rf_values', {})
        layer_names = list(first_rf_values.keys())
        
        # Collect rf statistics over time
        rf_stats_evolution = {}
        for layer in layer_names:
            rf_stats_evolution[layer] = {
                'mean': [],
                'median': [],
                'std': []
            }
            for state in self.topology_states:
                layer_rf = state.get('rf_values', {}).get(layer, np.array([]))
                if len(layer_rf) > 0:
                    rf_stats_evolution[layer]['mean'].append(np.mean(layer_rf))
                    rf_stats_evolution[layer]['median'].append(np.median(layer_rf))
                    rf_stats_evolution[layer]['std'].append(np.std(layer_rf))
                else:
                    rf_stats_evolution[layer]['mean'].append(0.0)
                    rf_stats_evolution[layer]['median'].append(0.0)
                    rf_stats_evolution[layer]['std'].append(0.0)
        
        rf_evolution['rf_stats_evolution'] = rf_stats_evolution
        return rf_evolution

    def _prepare_save_data(self, state: Dict[str, Any], epoch: int) -> Dict[str, Any]:
        """Prepare topology data for saving"""
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