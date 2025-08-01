from tinygrad.tensor import Tensor
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import copy


class ModelAdapter:
    def should_hook(self, name: str, module: Any) -> bool:
        return hasattr(module, 'weight') and hasattr(module, '__call__')
    
    def process_batch(self, data_batch, device=None):
        if isinstance(data_batch, (list, tuple)):
            return data_batch[0], {}
        else:
            return data_batch, {}
    
    def process_output(self, output, strategy):
        if strategy == 'auto':
            if len(output.shape) == 3 and output.shape[1] > 1:
                return output[:, 0, :].numpy()
            else:
                return output.numpy()
        elif strategy == 'cls':
            if len(output.shape) == 3:
                return output[:, 0, :].numpy()
            else:
                return output.numpy()
        elif strategy == 'full':
            return output.numpy()
        elif strategy == 'mean':
            if len(output.shape) == 3:
                return output.mean(axis=1).numpy()
            else:
                return output.numpy()
        else:
            return output.numpy()


class TransformerAdapter(ModelAdapter):
    def should_hook(self, name: str, module: Any) -> bool:
        if hasattr(module, 'weight') and hasattr(module, '__call__'):
            return True
        if hasattr(module, 'weight') and hasattr(module, 'bias'):
            return True
        return False
    
    def process_batch(self, data_batch, device=None):
        if isinstance(data_batch, dict):
            bert_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids'}
            data = {k: v for k, v in data_batch.items() if k in bert_keys}
            return None, data
        elif isinstance(data_batch, (list, tuple)):
            return data_batch[0], {}
        else:
            return data_batch, {}


def _detect_model_type(model: Any) -> str:
    model_type = str(type(model)).lower()
    if any(x in model_type for x in ['bert', 'roberta', 'distilbert', 'transformer']):
        return 'transformer'
    
    # Check for attention in attribute names
    attr_names = []
    def collect_attrs(obj, prefix=""):
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                attr_names.append(f"{prefix}.{attr_name}" if prefix else attr_name)
                try:
                    attr = getattr(obj, attr_name)
                    if hasattr(attr, '__dict__') and not isinstance(attr, type):
                        collect_attrs(attr, f"{prefix}.{attr_name}" if prefix else attr_name)
                except:
                    continue
    
    collect_attrs(model)
    if any('attention' in name.lower() for name in attr_names):
        return 'transformer'
    
    # Check for linear layers
    if any('linear' in name.lower() for name in attr_names):
        return 'mlp'
    
    # Check for actual weight tensors
    for attr_name in dir(model):
        try:
            attr = getattr(model, attr_name)
            if isinstance(attr, Tensor):
                return 'mlp'
        except:
            continue
    
    raise ValueError("Model type could not be detected. Please specify model_type explicitly.")


class ActivationMonitor:
    def __init__(self, model: Any, model_type: Optional[str] = None):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.original_methods = {}
        
        model_type = model_type or _detect_model_type(model)
        self.adapter = TransformerAdapter() if model_type == 'transformer' else ModelAdapter()
        self.sequence_strategy = 'auto'
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks by monkey patching __call__ methods"""
        def find_hookable_modules(obj, prefix=""):
            modules = []
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                    full_name = f"{prefix}.{attr_name}" if prefix else attr_name
                    
                    if self.adapter.should_hook(full_name, attr):
                        print(f"[DEBUG] Will hook: {full_name} ({type(attr)})")
                        modules.append((full_name, attr))
                    elif hasattr(attr, '__dict__') and not isinstance(attr, (type, Tensor)):
                        modules.extend(find_hookable_modules(attr, full_name))
                except:
                    continue
            return modules
        
        hookable_modules = find_hookable_modules(self.model)
        
        for name, module in hookable_modules:
            if hasattr(module, '__call__'):
                # Store original method
                self.original_methods[name] = module.__call__
                
                # Create hook wrapper
                def make_hook(orig_method, hook_name):
                    def hooked_call(*args, **kwargs):
                        print(f"[DEBUG] Hooked __call__ for {hook_name} with args={args}, kwargs={kwargs}")
                        result = orig_method(*args, **kwargs)
                        processed_output = self.adapter.process_output(result, self.sequence_strategy)
                        print(f"[DEBUG] Captured activation for {hook_name}: shape={getattr(processed_output, 'shape', None)}")
                        self.activations[hook_name] = processed_output
                        return result
                    return hooked_call
                
                # Replace method with hooked version
                module.__call__ = make_hook(self.original_methods[name], name)
                self.hooks.append((name, module))
    
    def get_activations(self) -> Dict[str, np.ndarray]:
        assert self.activations, "No activations captured. Run a forward pass first."
        return self.activations.copy()
    
    def set_config(self, output_file: str, sequence_strategy: str = 'auto', 
                use_quantization: bool = False, quantization_resolution: float = 0.1,
                **analysis_kwargs):
        self.sequence_strategy = sequence_strategy
        self.analysis_config = {
            'output_file': output_file,
            'analysis_kwargs': {
                'use_quantization': use_quantization,
                'quantization_resolution': quantization_resolution,
                **analysis_kwargs
            }
        }
        self.topology_states = []
    
    def analyze(self, test_loader, epoch: int, description: str = "", 
                save: bool = False, **analysis_flags) -> Dict[str, Any]:
        assert hasattr(self, 'analysis_config'), "Analysis configuration not set. Use set_config() to configure analysis."        
        
        # Clear previous activations
        self.activations = {}
        
        # Run forward pass to capture activations
        data_batch = next(iter(test_loader))
        args, kwargs = self.adapter.process_batch(data_batch)
        
        if args is not None:
            _ = self.model(args)
        else:
            _ = self.model(**kwargs)
        
        # Get captured activations and convert to torch tensors for analysis
        tinygrad_activations = self.get_activations()
        
        # Convert numpy arrays back to torch tensors for existing analysis
        import torch
        torch_activations = {}
        for name, activation in tinygrad_activations.items():
            torch_activations[name] = torch.from_numpy(activation).float()
        
        # Use existing analysis
        from . import analysis
        params = {**self.analysis_config['analysis_kwargs'], **analysis_flags}
        state = analysis.analyze(torch_activations, **params)        
        self.last_analysis = state
        
        rf_info = self._format_rf_output(state)
        if description: 
            print(f"{description} (Epoch {epoch}): {rf_info}")
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
        """Handle nested result structure"""
        if 'full_network' in state and state['full_network']:
            return self._format_single_rf_output(state['full_network'])
        elif 'by_layers' in state and state['by_layers']:
            layer_count = len(state['by_layers'])
            return f"Analyzed {layer_count} layers"
        elif 'by_components' in state and state['by_components']:
            comp_count = len(state['by_components'])
            return f"Analyzed {comp_count} components"
        else:
            return "No analysis results"
        
    def _format_single_rf_output(self, single_state: Dict[str, Any]) -> str:
        assert 'rf_values' in single_state, "State must contain 'rf_values' key"
        rf_values = single_state['rf_values']
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
        
        rf_str = ""
        if rf_summary:
            rf_str = ", ".join([f"{dim}_median: {val:.4f}" for dim, val in rf_summary.items()])
            rf_str = f", {rf_str}"
        
        if 'quantization_info' in single_state:
            qinfo = single_state['quantization_info']
            quant_str = f", Compressed: {qinfo['n_original_neurons']}→{qinfo['n_unique_clusters']} ({qinfo['compression_ratio']:.1f}x)"
            rf_str += quant_str
        
        total_neurons = single_state['total_neurons']
        betti_numbers = single_state['betti_numbers']
        return f"Neurons: {total_neurons}, Betti: {betti_numbers}{rf_str}"

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
        """Restore original methods"""
        for name, module in self.hooks:
            if name in self.original_methods:
                module.__call__ = self.original_methods[name]
        self.hooks = []
        self.original_methods = {}
        self.activations = {}

    def ablate(self, model: Any, strategy: str, value: float, rf_dim: str = 'rf_0', 
            state: Optional[Dict[str, Any]] = None) -> Any:
        """
        Ablate neurons based on RF importance scores
        
        Args:
            model: tinygrad model to ablate
            strategy: 'percent' or 'random'
            value: percentage (0-100) of neurons to ablate
            rf_dim: which RF dimension to use for importance ('rf_0', 'rf_1', etc.)
            state: analysis state to use (if None, uses last_analysis)
        
        Returns:
            Copy of model with selected neurons masked (set to 0)
        """
        # Validation
        assert hasattr(self, 'last_analysis') or state is not None, \
            "Must run analyze() first or provide analysis state"
        assert strategy in ['percent', 'random'], f"Invalid strategy: {strategy}. Use 'percent' or 'random'"
        assert 0 <= value <= 100, f"Value must be 0-100, got {value}"
        
        # Use provided state or last analysis
        analysis_state = state if state is not None else self.last_analysis
        
        # Handle nested analysis results (by_components, by_layers, full_network)
        if 'full_network' in analysis_state:
            rf_values = analysis_state['full_network']['rf_values']
            neuron_selection = self._select_neurons(rf_values, strategy, value, rf_dim)
        elif 'by_components' in analysis_state:
            # Combine all components for global selection
            combined_rf = {}
            for comp_name, comp_data in analysis_state['by_components'].items():
                combined_rf.update(comp_data['rf_values'])
            neuron_selection = self._select_neurons(combined_rf, strategy, value, rf_dim)
        elif 'by_layers' in analysis_state:
            # Combine all layers for global selection
            combined_rf = {}
            for layer_num, layer_data in analysis_state['by_layers'].items():
                combined_rf.update(layer_data['rf_values'])
            neuron_selection = self._select_neurons(combined_rf, strategy, value, rf_dim)
        else:
            raise ValueError("Analysis state must contain 'full_network', 'by_components', or 'by_layers'")
        return self._mask_neurons(model, neuron_selection)

    def _select_neurons(self, rf_values: Dict[str, Dict[str, np.ndarray]], 
                    strategy: str, percentage: float, rf_dim: str) -> List[Tuple[str, int]]:
        """
        Select neurons to ablate based on strategy
        
        Returns:
            List of (layer_name, neuron_index) tuples to mask
        """
        all_neurons = []
        
        # Collect all neurons with their RF values
        for layer_name, layer_rf in rf_values.items():
            assert rf_dim in layer_rf, f"RF dimension {rf_dim} not found in layer {layer_name}"
            
            rf_scores = layer_rf[rf_dim]
            for neuron_idx, rf_score in enumerate(rf_scores):
                all_neurons.append((layer_name, neuron_idx, rf_score))
        
        # Select neurons based on strategy
        num_to_select = int(len(all_neurons) * percentage / 100)
        
        if strategy == 'percent':
            # Sort by RF score (ascending) and take lowest percentage
            all_neurons.sort(key=lambda x: x[2])  # Sort by RF score
            selected = all_neurons[:num_to_select]
        elif strategy == 'random':
            # Randomly select percentage of neurons
            import random
            selected = random.sample(all_neurons, num_to_select)
        
        # Return (layer_name, neuron_index) pairs
        return [(layer_name, neuron_idx) for layer_name, neuron_idx, _ in selected]

    def _mask_neurons(self, model: Any, neuron_selection: List[Tuple[str, int]]) -> Any:
        """
        Create a copy of the model with selected neurons masked (set to 0)
        """
        # Create deep copy of the model
        ablated_model = copy.deepcopy(model)
        
        # Group selections by layer for efficient masking
        layer_masks = {}
        for layer_name, neuron_idx in neuron_selection:
            if layer_name not in layer_masks:
                layer_masks[layer_name] = []
            layer_masks[layer_name].append(neuron_idx)
        
        # Apply masks to each layer
        for layer_name, neuron_indices in layer_masks.items():
            # Find the corresponding module in the model
            module = self._get_module_by_name(ablated_model, layer_name)
            
            if hasattr(module, 'weight'):
                # Mask weights and biases for linear layers
                for neuron_idx in neuron_indices:
                    # Zero out outgoing weights (this neuron's effect on next layer)
                    if hasattr(module.weight, 'data'):
                        module.weight.data[neuron_idx, :] = 0
                    else:
                        # For tinygrad tensors, direct assignment
                        weight_data = module.weight.numpy()
                        weight_data[neuron_idx, :] = 0
                        module.weight = Tensor(weight_data)
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        if hasattr(module.bias, 'data'):
                            module.bias.data[neuron_idx] = 0
                        else:
                            bias_data = module.bias.numpy()
                            bias_data[neuron_idx] = 0
                            module.bias = Tensor(bias_data)
            else:
                print(f"Warning: Cannot mask module type {type(module)} for layer {layer_name}")
        
        return ablated_model

    def _get_module_by_name(self, model: Any, layer_name: str) -> Any:
        """
        Get a module from the model by its registered name from hooks
        """
        def find_module(obj, target_name, current_name=""):
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                try:
                    full_name = f"{current_name}.{attr_name}" if current_name else attr_name
                    if full_name == target_name:
                        return getattr(obj, attr_name)
                    
                    attr = getattr(obj, attr_name)
                    if hasattr(attr, '__dict__') and not isinstance(attr, (type, Tensor)):
                        result = find_module(attr, target_name, full_name)
                        if result is not None:
                            return result
                except:
                    continue
            return None
        
        module = find_module(model, layer_name)
        if module is None:
            raise ValueError(f"Module {layer_name} not found in model")
        return module