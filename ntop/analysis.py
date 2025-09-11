import torch
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import pairwise_distances
from ripser import ripser
from joblib import Parallel, delayed

def _compute_distances(activations: torch.Tensor, metric: str = 'euclidean') -> np.ndarray:
    """Computes pairwise distances between the rows of the activation matrix."""
    # Ensure tensor is on the CPU and converted to a NumPy array for scikit-learn.
    x_np = activations.cpu().numpy()
    return pairwise_distances(x_np, metric=metric)

def _normalize_activations(activations: Dict[str, torch.Tensor], method: str) -> Dict[str, torch.Tensor]:
    """Normalizes activations for each layer."""
    if method == 'none':
        return activations
    
    normalized = {}
    for layer_name, layer_acts in activations.items():
        if method == 'l2':
            # L2 normalization across neuron dimension.
            normalized[layer_name] = torch.nn.functional.normalize(layer_acts, p=2, dim=1)
        elif method == 'zscore':
            # Z-score normalization.
            mean = layer_acts.mean(dim=0, keepdim=True)
            std = layer_acts.std(dim=0, keepdim=True)
            normalized[layer_name] = (layer_acts - mean) / (std + 1e-8)
        elif method == 'minmax':
            # Min-max normalization.
            min_vals = layer_acts.min(dim=0, keepdim=True)[0]
            max_vals = layer_acts.max(dim=0, keepdim=True)[0]
            normalized[layer_name] = (layer_acts - min_vals) / (max_vals - min_vals + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
    return normalized

def _filter_inactive_neurons(activations: Dict[str, torch.Tensor], threshold: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Filters out neurons with low variance."""
    filtered = {}
    for layer_name, layer_acts in activations.items():
        if layer_acts.dim() > 2:
            # Flatten to [samples, features]
            layer_acts = layer_acts.flatten(1)
        
        # Calculate variance across the batch dimension.
        neuron_variance = layer_acts.var(dim=0)
        active_mask = neuron_variance > threshold
        
        if active_mask.sum() > 0:
            filtered[layer_name] = layer_acts[:, active_mask]
        else:
            # It's better to warn than to raise an error, to allow analysis to continue.
            print(f"Warning: No active neurons found in layer {layer_name} with threshold {threshold}. Skipping layer.")
            
    return filtered

def _compute_persistence(distance_matrix: np.ndarray, max_dim: int = 2) -> Dict:
    """Computes persistence diagrams from a distance matrix."""
    return ripser(distance_matrix, maxdim=max_dim, distance_matrix=True)

def _filter_persistence(persistence_result: Dict, threshold: float) -> Dict:
    """Filters persistence pairs with a lifetime below a threshold."""
    if threshold <= 0:
        return persistence_result
        
    filtered_diagrams = []
    for diagram in persistence_result['dgms']:
        if len(diagram) == 0:
            filtered_diagrams.append(diagram)
            continue
        
        # Keep infinite points and points with persistence > threshold.
        persistence = diagram[:, 1] - diagram[:, 0]
        significant_mask = (persistence > threshold) | np.isinf(persistence)
        filtered_diagrams.append(diagram[significant_mask])
        
    return {'dgms': filtered_diagrams}

def _extract_betti_numbers(persistence_result: Dict) -> Dict[int, int]:
    """Extracts Betti numbers (number of infinite persistence pairs)."""
    betti = {}
    for dim, diagram in enumerate(persistence_result['dgms']):
        if len(diagram) == 0:
            betti[dim] = 0
        else:
            # Betti number is the count of features that never "die".
            infinite_features = np.isinf(diagram[:, 1])
            betti[dim] = int(np.sum(infinite_features))
            
    return betti

def _compute_rf_for_neuron(neuron_activations: np.ndarray, distance_metric: str, max_dim: int) -> List[float]:
    """Helper function to compute RF values for a single neuron."""
    rf_values = [0.0] * (max_dim + 1)
    
    # Skip inactive neurons.
    if np.std(neuron_activations) < 1e-10:
        return rf_values
        
    # Reshape for pairwise_distances.
    neuron_activations = neuron_activations.reshape(-1, 1)
    distance_matrix = pairwise_distances(neuron_activations, metric=distance_metric)
    persistence = _compute_persistence(distance_matrix, max_dim=max_dim)
    
    for dim in range(max_dim + 1):
        if dim < len(persistence['dgms']):
            diagram = persistence['dgms'][dim]
            if len(diagram) > 0:
                # RF value is the maximum finite death time.
                finite_deaths = diagram[~np.isinf(diagram[:, 1]), 1]
                if len(finite_deaths) > 0:
                    rf_values[dim] = float(finite_deaths.max())
                    
    return rf_values

def _compute_per_neuron_rf(activations: Dict[str, torch.Tensor], distance_metric: str = 'euclidean', max_dim: int = 2) -> Dict[str, Dict[str, np.ndarray]]:
    """Computes RF values for each neuron in parallel."""
    rf_values = {}
    
    for layer_name, layer_acts in activations.items():
        if layer_acts.dim() > 2:
            layer_acts = layer_acts.flatten(1)
            
        n_neurons = layer_acts.shape[1]
        
        # Parallelize the computation for each neuron in the layer.
        results = Parallel(n_jobs=-1)(
            delayed(_compute_rf_for_neuron)(
                layer_acts[:, i].cpu().numpy(), distance_metric, max_dim
            ) for i in range(n_neurons)
        )
        
        # Process results.
        layer_rf_values = {}
        for dim in range(max_dim + 1):
            layer_rf_values[f'rf_{dim}'] = np.array([res[dim] for res in results])
            
        rf_values[layer_name] = layer_rf_values
        
    return rf_values

def analyze(activations: Dict[str, torch.Tensor], 
           distance_metric: str = 'euclidean', 
           max_dim: int = 2,
           max_samples: Optional[int] = None,
           normalize_activations: str = 'none',
           persistence_threshold: float = 0.0,
           random_seed: Optional[int] = None,
           filter_inactive_neurons: bool = False,
           ) -> Dict[str, Any]:
    """
    Analyzes the topology of a set of activations, intended for a single component or the full network.
    This version avoids creating a unified neuron space for better memory efficiency.
    """
    if not activations:
        print("Warning: Empty activation dictionary passed to analyze.")
        return {}

    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Sub-sample activations if requested.
    if max_samples is not None:
        for name, acts in activations.items():
            if acts.shape[0] > max_samples:
                indices = torch.randperm(acts.shape[0])[:max_samples]
                activations[name] = acts[indices]
    
    # Pre-processing steps.
    if filter_inactive_neurons:
        activations = _filter_inactive_neurons(activations)
        
    activations = _normalize_activations(activations, normalize_activations)
    
    # We no longer create a giant distance matrix for the whole component.
    # Instead, we compute per-neuron RF values directly.
    print(f"Computing RF values for {sum(act.shape[1] for act in activations.values())} neurons...")
    rf_values = _compute_per_neuron_rf(activations, distance_metric, max_dim)
    
    # Note: Betti numbers and persistence diagrams for the *entire component* are no longer computed
    # by default as it doesn't scale. These metrics are more meaningful at the per-neuron level (RF values).
    # If a global view is needed, it should be done with caution on smaller components.
    
    total_neurons = sum(act.shape[1] for act in activations.values())
    n_samples = next(iter(activations.values())).shape[0] if activations else 0

    return {
        'rf_values': rf_values,
        'total_neurons': total_neurons,
        'n_samples': n_samples,
        'distance_metric': distance_metric,
        # 'persistence' and 'betti_numbers' are omitted as they are not scalable for large components.
    }