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
        def find_tensors(obj, prefix=""):
            tensors = []
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                    full_name = f"{prefix}.{attr_name}" if prefix else attr_name
                    
                    if isinstance(attr, Tensor) and attr_name == 'weight':
                        tensors.append((full_name, attr))
                    elif hasattr(attr, '__dict__') and not isinstance(attr, (type, Tensor)):
                        tensors.extend(find_tensors(attr, full_name))
                except:
                    continue
            return tensors
        
        # Hook weight tensors instead of __call__ methods
        weight_tensors = find_tensors(self.model)
        print(f"Found {len(weight_tensors)} weight tensors to monitor")
    
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
        
        # Run forward pass to build UOp graph (no hooks needed!)
        data_batch = next(iter(test_loader))
        args, kwargs = self.adapter.process_batch(data_batch)
        
        print("DEBUG: Running forward pass to build UOp graph...")
        if args is not None:
            output = self.model(args)
        else:
            output = self.model(**kwargs)
        
        print("DEBUG: Extracting activations from UOp graph...")
        # Extract activations from UOp graph
        self.activations = self._extract_from_uop_graph(output, data_batch)
        
        print(f"DEBUG: Extracted {len(self.activations)} activations")
        for name in self.activations:
            print(f"  {name}: shape {self.activations[name].shape}")
        
        if not self.activations:
            print("WARNING: No activations extracted! Falling back to direct module execution...")
            self.activations = self._extract_activations_directly(data_batch)
        
        # Convert numpy arrays to torch tensors for existing analysis
        import torch
        torch_activations = {}
        for name, activation in self.activations.items():
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

    def _extract_from_uop_graph(self, output, data_batch):
        """Extract intermediate activations from tinygrad's UOp graph"""
        activations = {}
        visited = set()
        
        # Store weight tensors for identification
        self._build_weight_tensor_map()
        
        def walk_uop(uop, depth=0):
            if id(uop) in visited or depth > 20:  # Prevent infinite recursion
                return
            visited.add(id(uop))
            
            # Check if this UOp corresponds to a module we care about
            module_name = self._identify_module_from_uop(uop)
            if module_name:
                try:
                    print(f"DEBUG: Found UOp for {module_name} at depth {depth}")
                    # Extract activation from this UOp
                    activation = self._extract_activation_from_uop(uop)
                    if activation is not None:
                        processed = self.adapter.process_output(activation, self.sequence_strategy)
                        activations[module_name] = processed
                        print(f"DEBUG: Extracted activation for {module_name}: shape {processed.shape}")
                except Exception as e:
                    print(f"DEBUG: Failed to extract activation for {module_name}: {e}")
            
            # Recursively walk source UOps
            if hasattr(uop, 'src') and uop.src:
                for src_uop in uop.src:
                    walk_uop(src_uop, depth + 1)
        
        # Start walking from output tensor(s)
        if isinstance(output, dict):
            for key, tensor in output.items():
                if hasattr(tensor, 'uop'):
                    print(f"DEBUG: Walking UOp graph from output['{key}']")
                    walk_uop(tensor.uop)
        elif hasattr(output, 'uop'):
            print("DEBUG: Walking UOp graph from single output")
            walk_uop(output.uop)
        else:
            print(f"DEBUG: Output type {type(output)} has no uop attribute")
        
        return activations

    def _build_weight_tensor_map(self):
        """Build mapping from weight tensors to module names"""
        self.weight_tensor_map = {}
        
        for name, module in self.hooks:  # Use the modules we identified for hooking
            if hasattr(module, 'weight') and isinstance(module.weight, Tensor):
                # Store the weight tensor ID and corresponding module name
                self.weight_tensor_map[id(module.weight.uop)] = name
                print(f"DEBUG: Mapped weight tensor for {name}")

    def _identify_module_from_uop(self, uop):
        """Try to identify which module this UOp corresponds to"""
        # Check if this UOp uses any of our tracked weight tensors
        if hasattr(uop, 'src') and uop.src:
            for src_uop in uop.src:
                if id(src_uop) in self.weight_tensor_map:
                    return self.weight_tensor_map[id(src_uop)]
        
        # Pattern matching for common operations
        if hasattr(uop, 'op'):
            # Look for linear layer patterns (matrix multiplication)
            if str(uop.op) in ['DOT', 'MATMUL'] and hasattr(uop, 'src') and len(uop.src) >= 2:
                # This might be a linear layer - check if we can identify it
                for src_uop in uop.src:
                    if id(src_uop) in self.weight_tensor_map:
                        return self.weight_tensor_map[id(src_uop)]
            
            # Look for layer norm patterns
            if str(uop.op) in ['MUL', 'ADD'] and self._looks_like_layer_norm(uop):
                return self._find_layer_norm_module(uop)
        
        return None

    def _looks_like_layer_norm(self, uop):
        """Check if UOp pattern looks like layer normalization"""
        # Layer norm typically involves mean, variance computation, then normalize
        # This is a heuristic - might need refinement
        if not hasattr(uop, 'src') or len(uop.src) < 2:
            return False
        
        # Look for patterns involving statistics computation
        for src_uop in uop.src:
            if hasattr(src_uop, 'op') and str(src_uop.op) in ['REDUCE', 'SUM']:
                return True
        return False

    def _find_layer_norm_module(self, uop):
        """Try to identify which layer norm this UOp belongs to"""
        # This is heuristic - match against known layer norm modules
        for name, module in self.hooks:
            if 'layer_norm' in name.lower() or 'layernorm' in name.lower():
                return name
        return None

    def _extract_activation_from_uop(self, uop):
        """Extract activation tensor from UOp"""
        try:
            # Try to create a tensor from the UOp and realize it
            if hasattr(uop, 'dtype') and hasattr(uop, 'shape'):
                # Create tensor from UOp - this might need adjustment based on tinygrad version
                from tinygrad.tensor import Tensor
                
                # Different approaches to try
                try:
                    # Approach 1: Direct tensor creation from UOp
                    tensor = Tensor(uop=uop)
                    return tensor
                except:
                    # Approach 2: If UOp is already realized
                    if hasattr(uop, 'realized') and uop.realized:
                        return Tensor(uop.realized.toCPU())
                    else:
                        # Approach 3: Create tensor and realize
                        temp_tensor = Tensor(uop=uop)
                        temp_tensor.realize()
                        return temp_tensor
        except Exception as e:
            print(f"DEBUG: Failed to extract tensor from UOp: {e}")
            return None

    def _extract_activations_directly(self, data_batch):
        """Fallback: Extract activations by running modules directly"""
        print("DEBUG: Using direct module execution fallback...")
        activations = {}
        
        try:
            args, kwargs = self.adapter.process_batch(data_batch)
            
            # For BERT models, manually trace through execution
            if hasattr(self.model, 'bert'):
                print("DEBUG: Detected BERT model, tracing execution...")
                
                # Get inputs
                if args is not None:
                    input_ids = args
                    attention_mask = None
                    token_type_ids = None
                else:
                    input_ids = kwargs['input_ids']
                    attention_mask = kwargs.get('attention_mask', None)
                    token_type_ids = kwargs.get('token_type_ids', None)
                
                # Run embeddings
                embedding_output = self.model.bert.embeddings(input_ids, token_type_ids)
                
                # Capture embeddings layer norm if it exists
                if hasattr(self.model.bert.embeddings, 'layer_norm'):
                    try:
                        ln_output = self.model.bert.embeddings.layer_norm(embedding_output)
                        ln_output.realize()  # Force computation
                        processed = self.adapter.process_output(ln_output, self.sequence_strategy)
                        activations['bert.embeddings.layer_norm'] = processed
                        print(f"DEBUG: Captured embeddings layer_norm: {processed.shape}")
                    except Exception as e:
                        print(f"DEBUG: Failed to capture embeddings layer_norm: {e}")
                
                # Run encoder
                encoder_output = self.model.bert.encoder(embedding_output, attention_mask)
                
                # Run pooler
                pooled_output = self.model.bert.pooler(encoder_output)
                
                # Capture pooler dense
                if hasattr(self.model.bert.pooler, 'dense'):
                    try:
                        # The pooler.dense is already called inside pooler, so we get the result
                        pooled_output.realize()  # Force computation
                        processed = self.adapter.process_output(pooled_output, self.sequence_strategy)
                        activations['bert.pooler.dense'] = processed
                        print(f"DEBUG: Captured pooler.dense: {processed.shape}")
                    except Exception as e:
                        print(f"DEBUG: Failed to capture pooler.dense: {e}")
                
                # Run classifier
                if hasattr(self.model, 'classifier'):
                    try:
                        classifier_output = self.model.classifier(pooled_output)
                        classifier_output.realize()  # Force computation
                        processed = self.adapter.process_output(classifier_output, self.sequence_strategy)
                        activations['classifier'] = processed
                        print(f"DEBUG: Captured classifier: {processed.shape}")
                    except Exception as e:
                        print(f"DEBUG: Failed to capture classifier: {e}")
                
                # Capture other modules we care about
                for name, module in self.hooks:
                    if name not in activations:
                        try:
                            # Try to run the module on appropriate input
                            if 'embedding' in name:
                                test_output = module(input_ids)
                            elif 'pooler' in name:
                                test_output = module(encoder_output)
                            elif 'classifier' in name:
                                test_output = module(pooled_output)
                            else:
                                # Skip modules we can't easily trace
                                continue
                            
                            test_output.realize()
                            processed = self.adapter.process_output(test_output, self.sequence_strategy)
                            activations[name] = processed
                            print(f"DEBUG: Captured {name}: {processed.shape}")
                        except Exception as e:
                            print(f"DEBUG: Failed to capture {name}: {e}")
            
            else:
                print("DEBUG: Non-BERT model - using generic approach")
                # For non-BERT models, try running each hooked module individually
                # This is more generic but less reliable
                pass
                
        except Exception as e:
            print(f"DEBUG: Direct extraction failed: {e}")
        
        return activations
        
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
        topology_data = {'epoch': epoch}
        
        # Handle nested analysis structure (by_components, by_layers, full_network)
        if 'by_components' in state:
            topology_data['by_components'] = state['by_components']
            # Also extract summary statistics
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
            # Single network analysis - use original structure
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
        
        else:
            # Fallback - assume flat structure (old format)
            topology_data.update({
                'neuron_matrix': state.get('neuron_matrix'),
                'neuron_info': state.get('neuron_info'),
                'distance_matrix': state.get('distance_matrix'),
                'persistence': state.get('persistence'),
                'betti_numbers': state.get('betti_numbers'),
                'total_neurons': state.get('total_neurons'),
                'n_samples': state.get('n_samples'),
                'distance_metric': state.get('distance_metric')
            })
            
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