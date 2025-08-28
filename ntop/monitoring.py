import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import copy


class ModelAdapter:
    def should_hook(self, name: str, module: Any) -> bool:
        return isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))
    
    def process_batch(self, data_batch, device=None):
        if isinstance(data_batch, (list, tuple)):
            return data_batch[0], {}
        else:
            return data_batch, {}
    
    def process_output(self, output, strategy):
        if isinstance(output, torch.Tensor):
            output_np = output.detach().cpu().numpy()
        else:
            output_np = output
            
        if strategy == 'auto':
            if len(output_np.shape) == 3 and output_np.shape[1] > 1:
                return output_np[:, 0, :]  # CLS token for transformers
            else:
                return output_np
        elif strategy == 'cls':
            if len(output_np.shape) == 3:
                return output_np[:, 0, :]  # CLS token
            else:
                return output_np
        elif strategy == 'full':
            return output_np
        elif strategy == 'mean':
            if len(output_np.shape) == 3:
                return output_np.mean(axis=1)
            else:
                return output_np
        else:
            return output_np


class TransformerAdapter(ModelAdapter):
    def should_hook(self, name: str, module: Any) -> bool:
        # Hook key transformer components
        if isinstance(module, nn.Linear):
            # Look for attention and feedforward layers
            name_lower = name.lower()
            if any(x in name_lower for x in ['query', 'key', 'value', 'attention', 'dense', 'intermediate', 'output']):
                return True
        return False
    
    def process_batch(self, data_batch, device=None):
        # Accept both dict and BatchEncoding as HuggingFace style batch
        try:
            from transformers.tokenization_utils_base import BatchEncoding
            is_batch_encoding = isinstance(data_batch, BatchEncoding)
        except ImportError:
            is_batch_encoding = False
        if isinstance(data_batch, dict) or is_batch_encoding:
            return None, dict(data_batch)
        elif isinstance(data_batch, (list, tuple)):
            return data_batch[0], {}
        else:
            return data_batch, {}


def _detect_model_type(model: Any) -> str:
    """Detect model type for appropriate adapter selection"""
    model_type = str(type(model)).lower()
    model_name = model.__class__.__name__.lower()
    
    # Check for HuggingFace transformers
    if any(x in model_type for x in ['bert', 'roberta', 'distilbert', 'transformer']):
        return 'transformer'
    
    if any(x in model_name for x in ['bert', 'roberta', 'distilbert']):
        return 'transformer'
    
    # Check for attention layers in the model
    for name, module in model.named_modules():
        name_lower = name.lower()
        if 'attention' in name_lower or 'transformer' in name_lower:
            return 'transformer'
    
    # Default to MLP for simple models
    return 'mlp'


class ActivationMonitor:
    def __init__(self, model: Any, model_type: Optional[str] = None):
        self.model = model
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        self.activations = {}
        self.hooks = []
        
        model_type = model_type or _detect_model_type(model)
        self.adapter = TransformerAdapter() if model_type == 'transformer' else ModelAdapter()
        self.sequence_strategy = 'auto'
        
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on appropriate modules"""
        print("Registering forward hooks...")
        
        hook_count = 0
        for name, module in self.model.named_modules():
            if self.adapter.should_hook(name, module):
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append((name, hook))
                hook_count += 1
        
        print(f"Registered {hook_count} forward hooks")
        
        if hook_count == 0:
            print("Warning: No modules found to hook. Trying fallback approach...")
            # Fallback: hook all Linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    hook = module.register_forward_hook(self._create_hook(name))
                    self.hooks.append((name, hook))
                    hook_count += 1
            
            print(f"Fallback: Registered {hook_count} hooks on Linear layers")

    def _create_hook(self, layer_name: str):
        """Create a forward hook for a specific layer"""
        def hook_fn(module, input, output):
            # Store activation
            if isinstance(output, torch.Tensor):
                # Detach to avoid keeping gradients
                self.activations[layer_name] = output.detach()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                # Take first output if multiple outputs
                self.activations[layer_name] = output[0].detach()
        
        return hook_fn
    
    def get_activations(self) -> Dict[str, np.ndarray]:
        """Get stored activations as numpy arrays"""
        result = {}
        for name, activation in self.activations.items():
            if isinstance(activation, torch.Tensor):
                result[name] = activation.cpu().numpy()
            else:
                result[name] = activation
        return result
    
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
        """Analyze model topology using activation patterns"""
        assert hasattr(self, 'analysis_config'), "Analysis configuration not set. Use set_config() to configure analysis."
        
        # Clear previous activations
        self.activations = {}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward passes to collect activations
        print(f"Collecting activations from {len(test_loader)} batches...")
        
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(test_loader):
                if batch_idx >= self.analysis_config['analysis_kwargs'].get('max_samples', 500) // test_loader.batch_size:
                    break

                # Skip empty or invalid batches
                if not data_batch or (isinstance(data_batch, dict) and not data_batch):
                    print(f"Skipping empty batch at index {batch_idx}")
                    continue

                # Process batch for model input
                args, kwargs = self.adapter.process_batch(data_batch, self.device)

                # Move all tensors in kwargs to device
                if isinstance(self.adapter, TransformerAdapter):
                    # Always use kwargs for transformers
                    if not kwargs:
                        print(f"Skipping batch at index {batch_idx} due to empty kwargs for TransformerAdapter.")
                        continue
                    kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in kwargs.items()}
                    output = self.model(**kwargs)
                else:
                    if kwargs:
                        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in kwargs.items()}
                        output = self.model(**kwargs)
                    elif args is not None:
                        if isinstance(args, torch.Tensor):
                            args = args.to(self.device)
                        output = self.model(args)
                    else:
                        print(f"Skipping batch at index {batch_idx} due to no valid input for model forward pass.")
                        continue
        
        # Process collected activations
        print(f"Processing {len(self.activations)} collected activations...")
        
        if not self.activations:
            raise RuntimeError("No activations collected. Check if hooks are properly registered.")
        
        # Convert to the format expected by analysis module
        processed_activations = {}
        for name, activation in self.activations.items():
            processed = self.adapter.process_output(activation, self.sequence_strategy)
            processed_activations[name] = torch.from_numpy(processed).float()
        
        # Use existing analysis framework
        from . import analysis
        params = {**self.analysis_config['analysis_kwargs'], **analysis_flags}
        state = analysis.analyze(processed_activations, **params)
        
        self.last_analysis = state
        
        # Format output and save if requested
        rf_info = self._format_rf_output(state)
        if description: 
            print(f"{description} (Epoch {epoch}): {rf_info}")
        
        if save:
            topology_data = self._prepare_save_data(state, epoch)
            self.topology_states.append(topology_data)
            self._save_analysis()
        
        return state

    def _save_analysis(self):
        """Save topology analysis results"""
        assert self.topology_states, "No topology states to save"
        
        data = {
            'epochs': np.array([s['epoch'] for s in self.topology_states]),
            'topology_states': self.topology_states,
            'config': self.analysis_config
        }
        np.savez_compressed(self.analysis_config['output_file'], **data)
    
    def load_analysis(self, filepath: str) -> Dict[str, Any]: 
        """Load previously saved analysis"""
        return dict(np.load(filepath, allow_pickle=True))
    
    def _format_rf_output(self, state: Dict[str, Any]) -> str:
        """Format RF output for display"""
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
        """Format single analysis state output"""
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
        """Get RF evolution data from saved topology states"""
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
        """Prepare data for saving"""
        topology_data = {'epoch': epoch}
        
        # Handle nested analysis structure (by_components, by_layers, full_network)
        if 'by_components' in state:
            topology_data['by_components'] = state['by_components']
            # Extract summary statistics
            total_neurons = sum(comp_data['total_neurons'] for comp_data in state['by_components'].values())
            topology_data['total_neurons'] = total_neurons
            
            # Extract RF values from all components
            all_rf_values = {}
            for comp_name, comp_data in state['by_components'].items():
                if 'rf_values' in comp_data:
                    all_rf_values.update(comp_data['rf_values'])
            if all_rf_values:
                topology_data['rf_values'] = all_rf_values
        
        elif 'by_layers' in state:
            topology_data['by_layers'] = state['by_layers']
            # Extract summary statistics
            total_neurons = sum(layer_data['total_neurons'] for layer_data in state['by_layers'].values())
            topology_data['total_neurons'] = total_neurons
            
            # Extract RF values from all layers
            all_rf_values = {}
            for layer_num, layer_data in state['by_layers'].items():
                if 'rf_values' in layer_data:
                    all_rf_values.update(layer_data['rf_values'])
            if all_rf_values:
                topology_data['rf_values'] = all_rf_values
        
        elif 'full_network' in state:
            # Single network analysis
            network_data = state['full_network']
            topology_data.update({
                'neuron_matrix': network_data['neuron_matrix'],
                'neuron_info': network_data['neuron_info'],
                'distance_matrix': network_data['distance_matrix'],
                'persistence': network_data['persistence'],
                'betti_numbers': network_data['betti_numbers'],
                'total_neurons': network_data['total_neurons'],
                'n_samples': network_data['n_samples'],
                'distance_metric': network_data['distance_metric']
            })
            
            if 'rf_values' in network_data:
                topology_data['rf_values'] = network_data['rf_values']
        
        return topology_data

    def remove_hooks(self):
        """Remove all registered hooks"""
        for name, hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def ablate(self, model: Any, strategy: str, value: float, rf_dim: str = 'rf_0', 
            state: Optional[Dict[str, Any]] = None) -> Any:
        """
        Ablate neurons based on RF importance scores
        
        Args:
            model: PyTorch model to ablate
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
            
            if isinstance(module, nn.Linear):
                # Mask weights and biases for linear layers
                with torch.no_grad():
                    for neuron_idx in neuron_indices:
                        # Zero out outgoing weights (this neuron's effect on next layer)
                        if neuron_idx < module.weight.size(0):
                            module.weight[neuron_idx, :] = 0
                        
                        if module.bias is not None and neuron_idx < module.bias.size(0):
                            module.bias[neuron_idx] = 0
            else:
                print(f"Warning: Cannot mask module type {type(module)} for layer {layer_name}")
        
        return ablated_model

    def _get_module_by_name(self, model: Any, layer_name: str) -> Any:
        """
        Get a module from the model by its name
        """
        # Use PyTorch's built-in method to get module by name
        try:
            # Convert dots to attribute access
            attrs = layer_name.split('.')
            module = model
            for attr in attrs:
                module = getattr(module, attr)
            return module
        except AttributeError:
            raise ValueError(f"Module {layer_name} not found in model")