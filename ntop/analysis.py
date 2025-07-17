import torch
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import pairwise_distances
from ripser import ripser

def _compute_distances(activations: torch.Tensor, metric: str = 'euclidean') -> np.ndarray:
    x_np = activations.cpu().numpy()
    return pairwise_distances(x_np, metric=metric)

# This is per activation layer normalization which could be useful or harful i really dont know if you get no results try running with it set to 'none'
def _normalize_activations(activations: Dict[str, torch.Tensor], method: str) -> Dict[str, torch.Tensor]:
    if method == 'none': return activations
    normalized = {}
    for layer_name, layer_acts in activations.items():
        if method == 'l2':
            normalized[layer_name] = torch.nn.functional.normalize(layer_acts, p=2, dim=0)
        elif method == 'zscore':
            normalized[layer_name] = (layer_acts - layer_acts.mean(dim=0, keepdim=True)) / (layer_acts.std(dim=0, keepdim=True) + 1e-8)
        elif method == 'minmax':
            min_vals = layer_acts.min(dim=0, keepdim=True)[0]
            max_vals = layer_acts.max(dim=0, keepdim=True)[0]
            normalized[layer_name] = (layer_acts - min_vals) / (max_vals - min_vals + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    return normalized

def _filter_inactive_neurons(activations: Dict[str, torch.Tensor], threshold: float = 1e-6) -> Dict[str, torch.Tensor]:
    filtered = {}
    for layer_name, layer_acts in activations.items():
        if layer_acts.dim() > 2: layer_acts = layer_acts.flatten(1)
        neuron_variance = layer_acts.var(dim=0)
        active_mask = neuron_variance > threshold
        if active_mask.sum() > 0: filtered[layer_name] = layer_acts[:, active_mask]
        else: raise ValueError(f"No active neurons found in layer {layer_name} with threshold {threshold}")
    return filtered

def _unify_neuron_space(activations: Dict[str, torch.Tensor], max_samples: Optional[int] = None, 
                       random_seed: Optional[int] = None, 
                       use_quantization: bool = False,
                       quantization_resolution: float = 0.1) -> Dict[str, Any]:
    if random_seed is not None: torch.manual_seed(random_seed)
    all_neurons = []
    neuron_info = [] 
    for layer_name, layer_acts in activations.items():
        if layer_acts.dim() > 2: layer_acts = layer_acts.flatten(1)
        layer_neurons = layer_acts.transpose(0, 1)        # Transpose: [n_samples, n_neurons] -> [n_neurons, n_samples]
        n_neurons = layer_neurons.shape[0]
        for i in range(n_neurons): neuron_info.append({'layer': layer_name, 'local_idx': i})        
        all_neurons.append(layer_neurons)
    unified_matrix = torch.cat(all_neurons, dim=0)    
    if max_samples is not None and unified_matrix.shape[1] > max_samples:
        indices = torch.randperm(unified_matrix.shape[1])[:max_samples]
        unified_matrix = unified_matrix[:, indices]
    
    quantization_info = None
    if use_quantization and quantization_resolution > 0:
        quantized_matrix = _quantize_neuron_activations(unified_matrix, quantization_resolution)
        quantization_result = _deduplicate_quantized_neurons(quantized_matrix, neuron_info)
        
        unified_matrix = quantization_result['representative_matrix']
        neuron_info = quantization_result['representative_info']
        quantization_info = quantization_result
    
    return {
        'neuron_matrix': unified_matrix,
        'neuron_info': neuron_info,
        'total_neurons': unified_matrix.shape[0],
        'n_samples': unified_matrix.shape[1],
        'quantization_info': quantization_info
    }

def _compute_persistence(distance_matrix: np.ndarray, max_dim: int = 2) -> Dict: return ripser(distance_matrix, maxdim=max_dim, distance_matrix=True)

def _filter_persistence(persistence_result: Dict, threshold: float) -> Dict:
    if threshold <= 0: return persistence_result
    filtered_diagrams = []
    for diagram in persistence_result['dgms']:
        if len(diagram) == 0:
            filtered_diagrams.append(diagram)
            continue
        # Keep infinite points and points with persistence > threshold
        finite_mask = ~np.isinf(diagram[:, 1])
        infinite_points = diagram[~finite_mask]
        finite_points = diagram[finite_mask]
        if len(finite_points) > 0:
            persistence_lengths = finite_points[:, 1] - finite_points[:, 0]
            significant_mask = persistence_lengths > threshold
            significant_finite = finite_points[significant_mask]
            filtered_diagram = np.vstack([infinite_points, significant_finite]) if len(infinite_points) > 0 and len(significant_finite) > 0 else significant_finite if len(significant_finite) > 0 else infinite_points
        # FIXME: This should probably raise and error if no points are left except infinite points
        else: filtered_diagram = infinite_points
        filtered_diagrams.append(filtered_diagram)
    return {'dgms': filtered_diagrams}

def _extract_betti_numbers(persistence_result: Dict) -> Dict[int, int]:
    betti = {}
    diagrams = persistence_result['dgms']    
    for dim, diagram in enumerate(diagrams):
        if len(diagram) == 0:
            betti[dim] = 0
        else:
            infinite_features = np.isinf(diagram[:, 1])
            betti[dim] = np.sum(infinite_features)
    return betti

def _compute_per_neuron_rf(activations: Dict[str, torch.Tensor], distance_metric: str = 'euclidean', max_dim: int = 2) -> Dict[str, Dict[str, np.ndarray]]:
    rf_values = {}
    for layer_name, layer_acts in activations.items():
        if layer_acts.dim() > 2: layer_acts = layer_acts.flatten(1)
        n_samples, n_neurons = layer_acts.shape
        layer_rf_values = {} # Initialize RF arrays for each dimension
        for dim in range(max_dim + 1):
            layer_rf_values[f'rf_{dim}'] = np.zeros(n_neurons)
        for neuron_idx in range(n_neurons):
            neuron_activations = layer_acts[:, neuron_idx].cpu().numpy().reshape(-1, 1)
            if np.std(neuron_activations) < 1e-10:
                # Set all dimensions to 0 for inactive neurons
                for dim in range(max_dim + 1):
                    layer_rf_values[f'rf_{dim}'][neuron_idx] = 0.0
                continue            
            distance_matrix = _compute_distances(torch.tensor(neuron_activations), distance_metric)
            persistence = _compute_persistence(distance_matrix, max_dim=max_dim)
            for dim in range(max_dim + 1): # Extract rf values for each dimension
                if dim < len(persistence['dgms']):
                    diagram = persistence['dgms'][dim]
                    if len(diagram) == 0:
                        layer_rf_values[f'rf_{dim}'][neuron_idx] = 0.0
                    else:
                        finite_mask = ~np.isinf(diagram[:, 1])
                        finite_deaths = diagram[finite_mask, 1]
                        layer_rf_values[f'rf_{dim}'][neuron_idx] = float(finite_deaths.max()) if len(finite_deaths) > 0 else 0.0
                else:
                    layer_rf_values[f'rf_{dim}'][neuron_idx] = 0.0
        rf_values[layer_name] = layer_rf_values
    return rf_values

def _quantize_neuron_activations(neuron_matrix: torch.Tensor, quantization_resolution: float) -> torch.Tensor:
    if quantization_resolution <= 0: return neuron_matrix
    return torch.round(neuron_matrix / quantization_resolution) * quantization_resolution

def _deduplicate_quantized_neurons(quantized_matrix: torch.Tensor, neuron_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert quantized_matrix.shape[0] > 0, "Quantized matrix cannot be empty"
    assert len(neuron_info) == quantized_matrix.shape[0], "Neuron info must match matrix dimensions"
    
    n_neurons = quantized_matrix.shape[0]
    quantized_np = quantized_matrix.cpu().numpy()
    
    signatures = {}
    original_to_cluster = np.zeros(n_neurons, dtype=np.int32)
    cluster_to_original = {}
    cluster_representatives = []
    
    cluster_id = 0
    for neuron_idx in range(n_neurons):
        signature = tuple(quantized_np[neuron_idx].round(6))
        
        if signature not in signatures:
            signatures[signature] = cluster_id
            cluster_to_original[cluster_id] = [neuron_idx]
            cluster_representatives.append(neuron_idx)
            original_to_cluster[neuron_idx] = cluster_id
            cluster_id += 1
        else:
            existing_cluster = signatures[signature]
            cluster_to_original[existing_cluster].append(neuron_idx)
            original_to_cluster[neuron_idx] = existing_cluster
    
    representative_matrix = quantized_matrix[cluster_representatives]
    representative_info = [neuron_info[idx] for idx in cluster_representatives]
    
    return {
        'representative_matrix': representative_matrix,
        'representative_info': representative_info,
        'original_to_cluster': original_to_cluster,
        'cluster_to_original': cluster_to_original,
        'cluster_representatives': cluster_representatives,
        'n_original_neurons': n_neurons,
        'n_unique_clusters': len(cluster_representatives),
        'compression_ratio': n_neurons / len(cluster_representatives)
    }

def analyze(activations: Dict[str, torch.Tensor], 
           distance_metric: str = 'euclidean', 
           max_dim: int = 2,
           max_samples: Optional[int] = None,
           normalize_activations: str = 'none',
           persistence_threshold: float = 0.0,
           random_seed: Optional[int] = None,
           filter_inactive_neurons: bool = False,
           use_quantization: bool = False,
           quantization_resolution: float = 0.1,
           ) -> Dict[str, Any]:
    assert activations, "Activations dictionary cannot be empty"    
    if filter_inactive_neurons: activations = _filter_inactive_neurons(activations)
    activations = _normalize_activations(activations, normalize_activations)    
    unified = _unify_neuron_space(activations, max_samples, random_seed, 
                                 use_quantization, quantization_resolution)
    distance_matrix = _compute_distances(unified['neuron_matrix'], distance_metric)
    persistence = _compute_persistence(distance_matrix, max_dim)
    
    if use_quantization and unified['quantization_info'] is not None:
        quantization_info = unified['quantization_info']
        
        # Work exclusively with quantized neuron space
        # The unified['neuron_matrix'] already contains only representative neurons
        # and unified['neuron_info'] contains their layer information
        representative_activations = {}
        current_idx = 0
        
        for layer_name in activations.keys():
            # Find how many representative neurons belong to this layer
            layer_neuron_count = sum(1 for info in unified['neuron_info'] if info['layer'] == layer_name)
            if layer_neuron_count > 0:
                # Extract representative neurons for this layer
                layer_matrix = unified['neuron_matrix'][current_idx:current_idx + layer_neuron_count]
                representative_activations[layer_name] = layer_matrix.T  # Convert to [n_samples, n_neurons]
                current_idx += layer_neuron_count
        
        # Compute RF only on representative neurons
        rf_values = _compute_per_neuron_rf(representative_activations, distance_metric, max_dim)
        expanded_neuron_info = unified['neuron_info']  # Already contains only representatives
        total_neurons = len(unified['neuron_info'])  # Use compressed count
    else:
        rf_values = _compute_per_neuron_rf(activations, distance_metric, max_dim)
        expanded_neuron_info = unified['neuron_info']
        total_neurons = unified['total_neurons']
                    
    if persistence_threshold > 0: persistence = _filter_persistence(persistence, persistence_threshold)
    betti_numbers = _extract_betti_numbers(persistence)
    
    result = {
        'neuron_matrix': unified['neuron_matrix'].cpu().numpy(),
        'neuron_info': expanded_neuron_info,
        'distance_matrix': distance_matrix,
        'persistence': persistence,
        'betti_numbers': betti_numbers,
        'total_neurons': total_neurons,
        'n_samples': unified['n_samples'],
        'distance_metric': distance_metric,
        'rf_values' : rf_values
    }

    if use_quantization and unified['quantization_info'] is not None:
        result['quantization_info'] = unified['quantization_info']

    return result