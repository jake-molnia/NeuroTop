import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import copy

def _detect_model_type(model: Any) -> str:
    """Detects model type to select the appropriate adapter."""
    model_type_str = str(type(model)).lower()
    
    # Specific checks for HuggingFace Transformers.
    if any(lib in model_type_str for lib in ['transformers', 'bert', 'roberta', 'gpt']):
        return 'transformer'
        
    # A more general check for transformer-like architectures.
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'transformer' in name.lower():
            return 'transformer'
            
    return 'mlp' # Default

class ModelAdapter:
    """Default adapter for standard MLP-like models."""
    def should_hook(self, name: str, module: Any) -> bool:
        # Hook standard layers that produce meaningful activations.
        return isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))
    
    def process_batch(self, data_batch, device=None):
        # Assumes the first element of a tuple is the input tensor.
        if isinstance(data_batch, (list, tuple)):
            return data_batch[0], {}
        return data_batch, {}
    
    def process_output(self, output, strategy):
        # Handles different output shapes, especially for sequence models.
        if not isinstance(output, torch.Tensor):
            return output
            
        output_np = output.detach().cpu().numpy()
        
        if strategy == 'auto':
            # Default to CLS token for transformers, otherwise use the full output.
            if len(output_np.shape) == 3 and output_np.shape[1] > 1:
                return output_np[:, 0, :]
            return output_np.reshape(output_np.shape[0], -1)
            
        if strategy == 'cls' and len(output_np.shape) == 3:
            return output_np[:, 0, :]
            
        if strategy == 'mean' and len(output_np.shape) == 3:
            return output_np.mean(axis=1)
            
        return output_np.reshape(output_np.shape[0], -1)


class TransformerAdapter(ModelAdapter):
    """Adapter specifically for HuggingFace Transformer models."""
    def should_hook(self, name: str, module: Any) -> bool:
        # Target key components in transformer blocks.
        if isinstance(module, nn.Linear):
            name_lower = name.lower()
            if any(part in name_lower for part in ['query', 'key', 'value', 'attention.output.dense', 'intermediate.dense', 'output.dense']):
                return True
        return False
    
    def process_batch(self, data_batch, device=None):
        # Handle dictionary-style inputs common with HuggingFace.
        try:
            from transformers.tokenization_utils_base import BatchEncoding
            is_hf_batch = isinstance(data_batch, (dict, BatchEncoding))
        except ImportError:
            is_hf_batch = isinstance(data_batch, dict)

        if is_hf_batch:
            return None, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}
            
        return super().process_batch(data_batch, device)

class ActivationMonitor:
    def __init__(self, model: Any, model_type: Optional[str] = None):
        self.model = model
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        self.activations = {}
        self.hooks = []
        self.topology_states = []
        
        model_type = model_type or _detect_model_type(model)
        self.adapter = TransformerAdapter() if model_type == 'transformer' else ModelAdapter()
        
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on model layers identified by the adapter."""
        print("Registering forward hooks...")
        hook_count = 0
        for name, module in self.model.named_modules():
            if self.adapter.should_hook(name, module):
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
                hook_count += 1
                
        print(f"Registered {hook_count} hooks.")
        if hook_count == 0:
            print("Warning: No layers were hooked.")

    def _create_hook(self, layer_name: str):
        """
        Creates a hook function to store activations.
        **FIX:** Averages over the sequence dimension for transformer layers.
        """
        def hook_fn(module, input, output):
            # The output of some layers might be a tuple
            output_tensor = output[0] if isinstance(output, tuple) else output

            if isinstance(output_tensor, torch.Tensor):
                # If the tensor has 3 dimensions (batch, seq_len, features),
                # average over the sequence length (dim 1) to get (batch, features).
                # This ensures the number of "neurons" matches the layer's output features.
                if output_tensor.dim() == 3:
                    processed_output = output_tensor.mean(dim=1)
                else:
                    processed_output = output_tensor
                
                self.activations[layer_name] = processed_output.detach().cpu()
        return hook_fn
    
    def analyze(self, test_loader, epoch: int, save: bool = False, **analysis_kwargs) -> Dict[str, Any]:
        """Analyzes model topology. If save=True, stores the state for evolution tracking."""
        from . import analysis
        
        self.activations.clear()
        self.model.eval()
        
        print(f"Collecting activations for epoch {epoch}...")
        with torch.no_grad():
            for data_batch in test_loader:
                args, kwargs = self.adapter.process_batch(data_batch, self.device)
                if kwargs: _ = self.model(**kwargs)
                elif args: _ = self.model(args)
                else: continue
                break

        if not self.activations:
            raise RuntimeError("No activations collected.")

        results = {}
        analyze_by_components = analysis_kwargs.pop('analyze_by_components', False)
        
        if analyze_by_components:
            components = self._detect_components(self.activations)
            results['by_components'] = {}
            for name, acts in components.items():
                if acts:
                    results['by_components'][name] = analysis.analyze(acts, **analysis_kwargs)
        else:
            results['full_network'] = analysis.analyze(self.activations, **analysis_kwargs)
        
        if save:
            state_to_save = results.copy()
            state_to_save['epoch'] = epoch
            self.topology_states.append(state_to_save)
            
        return results

    def save_states(self, output_path: str):
        """Saves the collected list of topology states to a single .npz file."""
        if not self.topology_states:
            print("Warning: No topology states to save.")
            return
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, topology_states=self.topology_states)
        print(f"Topology evolution data saved to {output_path}")

    def ablate(self, model: Any, strategy: str, value: float, rf_dim: str = 'rf_0', 
               state: Optional[Dict[str, Any]] = None, component_name: Optional[str] = None) -> Any:
        """Ablates neurons based on RF importance scores."""
        analysis_state = state
        if not analysis_state:
            raise ValueError("Must provide an analysis state.")
            
        if component_name:
            if 'by_components' not in analysis_state or component_name not in analysis_state['by_components']:
                raise ValueError(f"Component '{component_name}' not found in the analysis state.")
            rf_values = analysis_state['by_components'][component_name]['rf_values']
        elif 'full_network' in analysis_state:
            rf_values = analysis_state['full_network']['rf_values']
        else:
            rf_values = {}
            for comp_data in analysis_state.get('by_components', {}).values():
                rf_values.update(comp_data.get('rf_values', {}))
        
        if not rf_values:
            raise ValueError("No RF values found in the provided state for ablation.")
            
        neuron_selection = self._select_neurons_to_ablate(rf_values, strategy, value, rf_dim)
        return self._mask_neurons(model, neuron_selection)

    def _select_neurons_to_ablate(self, rf_values: Dict[str, Dict[str, np.ndarray]], 
                                strategy: str, percentage: float, rf_dim: str,
                                balance_threshold: float = 1.5) -> List[Tuple[str, int]]:
        """
        Select neurons with dynamic load balancing.
        
        balance_threshold: How much a layer can deviate from average.
                        1.5 means a layer can be pruned 50% more than average before skipping.
        """
        
        if strategy == 'random':
            return self._random_selection_balanced(rf_values, percentage, rf_dim)
        
        # Collect all neurons with their RF scores
        all_neurons = []
        layer_sizes = {}
        
        for layer_name, layer_rf in rf_values.items():
            if rf_dim not in layer_rf:
                continue
            
            layer_sizes[layer_name] = len(layer_rf[rf_dim])
            
            for idx, score in enumerate(layer_rf[rf_dim]):
                all_neurons.append((layer_name, idx, float(score)))
        
        # Sort globally by RF (ascending - lowest first)
        all_neurons.sort(key=lambda x: x[2])
        
        # Calculate target
        total_neurons = len(all_neurons)
        target_pruned = int(total_neurons * percentage / 100)
        
        # Track pruning per layer
        layer_pruned = {layer: 0 for layer in layer_sizes.keys()}
        selected = []
        
        print(f"\nDynamic load balancing: selecting {target_pruned} neurons from {total_neurons}")
        
        # Iterate through neurons in RF order
        for layer_name, idx, score in all_neurons:
            if len(selected) >= target_pruned:
                break
            
            # Calculate current pruning ratios
            current_pruned = len(selected)
            if current_pruned == 0:
                # First neuron, just take it
                selected.append((layer_name, idx))
                layer_pruned[layer_name] += 1
                continue
            
            # Calculate average pruning ratio across all layers
            avg_ratio = current_pruned / total_neurons
            
            # Calculate this layer's current ratio
            layer_ratio = layer_pruned[layer_name] / layer_sizes[layer_name]
            
            # Check if this layer is over-represented
            if layer_ratio > avg_ratio * balance_threshold:
                # Skip this neuron - layer is already pruned too much
                continue
            
            # Prune this neuron
            selected.append((layer_name, idx))
            layer_pruned[layer_name] += 1
        
        # Print summary
        print(f"\nPruning distribution:")
        for layer_name in sorted(layer_sizes.keys()):
            pruned = layer_pruned[layer_name]
            total = layer_sizes[layer_name]
            pct = (pruned / total * 100) if total > 0 else 0
            print(f"  {layer_name}: {pruned}/{total} ({pct:.1f}%)")
        
        return selected

    def _random_selection_balanced(self, rf_values, percentage, rf_dim):
        """Random selection. NOTE: I am uncertain we need this as random already is uniform"""
        import random
        
        # Collect all neurons
        all_neurons = []
        layer_sizes = {}
        
        for layer_name, layer_rf in rf_values.items():
            if rf_dim not in layer_rf:
                continue
            
            layer_sizes[layer_name] = len(layer_rf[rf_dim])
            for idx in range(len(layer_rf[rf_dim])):
                all_neurons.append((layer_name, idx))
        
        # Shuffle randomly
        random.shuffle(all_neurons)
        
        # Apply same balancing logic
        total_neurons = len(all_neurons)
        target_pruned = int(total_neurons * percentage / 100)
        
        layer_pruned = {layer: 0 for layer in layer_sizes.keys()}
        selected = []
        
        for layer_name, idx in all_neurons:
            if len(selected) >= target_pruned:
                break
            
            current_pruned = len(selected)
            if current_pruned == 0:
                selected.append((layer_name, idx))
                layer_pruned[layer_name] += 1
                continue
            
            avg_ratio = current_pruned / total_neurons
            layer_ratio = layer_pruned[layer_name] / layer_sizes[layer_name]
            
            if layer_ratio > avg_ratio * 1.5:
                continue
            
            selected.append((layer_name, idx))
            layer_pruned[layer_name] += 1
        
        return selected

    def _mask_neurons(self, model: Any, neuron_selection: List[Tuple[str, int]]) -> Any:
        """Creates a copy of the model with selected neurons masked via hooks."""
        ablated_model = copy.deepcopy(model)
        
        # Group by layer
        layer_masks = {}
        for layer_name, neuron_idx in neuron_selection:
            layer_masks.setdefault(layer_name, []).append(neuron_idx)
        
        # Create mask hooks instead of zeroing weights
        self.mask_hooks = []
        
        for layer_name, indices in layer_masks.items():
            try:
                module = dict(ablated_model.named_modules())[layer_name]
                
                if isinstance(module, nn.Linear):
                    # Zero the weights in addition to masking
                    with torch.no_grad():
                        module.weight.data[indices, :] = 0
                        if module.bias is not None:
                            module.bias.data[indices] = 0
                    
                    # Then also apply hook (for gradient blocking during fine-tuning)
                    mask = torch.ones(module.out_features, device=module.weight.device)
                    mask[indices] = 0.0

                    
                    # Register hook that applies mask to output
                    def create_masking_hook(mask_tensor):
                        def hook(module, input, output):
                            # Apply mask: zero out pruned neurons
                            if output.dim() == 3:  # [batch, seq, features]
                                return output * mask_tensor.view(1, 1, -1)
                            elif output.dim() == 2:  # [batch, features]
                                return output * mask_tensor.view(1, -1)
                            return output
                        return hook
                    
                    hook_handle = module.register_forward_hook(create_masking_hook(mask))
                    self.mask_hooks.append(hook_handle)
                                        
            except KeyError:
                print(f"Warning: Layer '{layer_name}' not found for masking")
        
        return ablated_model

    def remove_mask_hooks(self):
        """Remove masking hooks."""
        if hasattr(self, 'mask_hooks'):
            for hook in self.mask_hooks:
                hook.remove()
            self.mask_hooks.clear()

    def _detect_components(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Detects components in a hierarchical, network-agnostic way."""
        components = {}
        
        for name, acts in activations.items():
            # Parse the component hierarchy from layer names
            component_path = self._parse_component_path(name)
            
            # Create nested structure: e.g., 'attention.query', 'mlp.intermediate'
            if component_path not in components:
                components[component_path] = {}
            components[component_path][name] = acts
        
        return components

    def _parse_component_path(self, layer_name: str) -> str:
        """Extract semantic component name from layer."""
        name_lower = layer_name.lower()
        
        # Attention subcomponents
        if 'query' in name_lower or '.q.' in name_lower:
            return 'attention.query'
        elif 'key' in name_lower or '.k.' in name_lower:
            return 'attention.key'
        elif 'value' in name_lower or '.v.' in name_lower:
            return 'attention.value'
        elif 'attention.output.dense' in name_lower:
            return 'attention.output'
        
        # MLP subcomponents
        elif 'intermediate' in name_lower:
            return 'mlp.intermediate'
        elif 'output.dense' in name_lower and 'attention' not in name_lower:
            return 'mlp.output'
        
        # Fallback for unknown layers
        else:
            return 'other'

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
