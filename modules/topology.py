import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
import pandas as pd
import logging
import ot  

logger = logging.getLogger(__name__)

# =============================================================================
# CORE DISTANCE COMPUTATION
# =============================================================================

def compute_distance_matrix(
    activations: torch.Tensor,
    metric: str = 'euclidean',
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """Compute pairwise distance matrix between neurons."""
    neuron_data = activations.T.cpu().numpy()
    n_neurons = neuron_data.shape[0]
    
    if chunk_size is None or n_neurons <= chunk_size:
        distances = pdist(neuron_data, metric=metric)
        dist_matrix = squareform(distances)
    else:
        dist_matrix = _compute_chunked_distances(neuron_data, metric, chunk_size)
    
    logger.info(f"Distance matrix: {dist_matrix.shape}, range: [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")
    return dist_matrix

def compute_wasserstein_distance_matrix(activations: torch.Tensor) -> np.ndarray:
    """Compute Wasserstein distance matrix between neuron distributions."""
    n_neurons = activations.shape[1]
    n_samples = activations.shape[0]
    dist_matrix = np.zeros((n_neurons, n_neurons))
    
    # Ensure bin locations are float64.
    bin_locations = np.arange(n_samples, dtype=np.float64)

    distributions = []
    for i in range(n_neurons):
        # Ensure activations are float64.
        acts = activations[:, i].cpu().numpy().astype(np.float64)
        acts = np.maximum(acts, 0)
        acts_sum = acts.sum()
        
        if acts_sum == 0:
            distributions.append(np.ones(n_samples, dtype=np.float64) / n_samples)
        else:
            distributions.append(acts / acts_sum)
    
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            dist = ot.emd2_1d(
                bin_locations,
                bin_locations,
                distributions[i], 
                distributions[j]
            )
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    logger.info(f"Wasserstein distance matrix: {dist_matrix.shape}, range: [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")
    return dist_matrix

# =============================================================================
# PERSISTENCE AND TOPOLOGY
# =============================================================================

def compute_persistence_diagrams(
    distance_matrix: np.ndarray,
    max_dimension: int = 2
) -> Dict:
    """Compute persistence diagrams using Ripser."""
    logger.info(f"Computing persistence up to H{max_dimension}")
    
    diagrams = ripser(distance_matrix, maxdim=max_dimension, distance_matrix=True)
    
    for dim in range(max_dimension + 1):
        if dim < len(diagrams['dgms']):
            n_features = len(diagrams['dgms'][dim])
            logger.info(f"H{dim}: {n_features} topological features")
    
    return diagrams

def get_filtration_scales(
    distance_matrix: np.ndarray,
    num_scales: int = 10,
    percentile_range: Tuple[float, float] = (5.0, 95.0)
) -> np.ndarray:
    """Generate filtration scales based on distance distribution."""
    upper_triangle = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
    percentiles = np.linspace(percentile_range[0], percentile_range[1], num_scales)
    scales = np.percentile(upper_triangle, percentiles)
    return scales

def compute_degree_centrality_at_scale(distance_matrix: np.ndarray, scale: float) -> np.ndarray:
    """Compute degree centrality in graph at given filtration scale."""
    adjacency = (distance_matrix <= scale).astype(int)
    np.fill_diagonal(adjacency, 0)
    degrees = np.sum(adjacency, axis=1)
    return degrees

def classify_neuron_criticality(
    distance_matrix: np.ndarray,
    scales: np.ndarray
) -> pd.DataFrame:
    """Classify neurons based on degree evolution across scales."""
    n_neurons = distance_matrix.shape[0]
    degree_matrix = np.zeros((n_neurons, len(scales)))
    
    for i, scale in enumerate(scales):
        degree_matrix[:, i] = compute_degree_centrality_at_scale(distance_matrix, scale)
    
    classifications = []
    max_possible_degree = np.max(degree_matrix)
    
    for neuron_idx in range(n_neurons):
        degrees = degree_matrix[neuron_idx, :]
        max_degree = np.max(degrees)
        
        early_scale_cutoff = int(len(scales) * 0.3)
        is_core = np.any(degrees[:early_scale_cutoff] > max_degree * 0.5)
        is_redundant = max_degree < max_possible_degree * 0.2
        
        if is_core:
            classifications.append('Core')
        elif is_redundant:
            classifications.append('Redundant')
        else:
            classifications.append('Conditional')
    
    result_df = pd.DataFrame({
        'neuron_id': range(n_neurons),
        'classification': classifications,
        'max_degree': np.max(degree_matrix, axis=1),
        'mean_degree': np.mean(degree_matrix, axis=1)
    })
    
    for i, scale in enumerate(scales):
        result_df[f'degree_scale_{i}'] = degree_matrix[:, i]
    
    logger.info(f"Neuron classification: {result_df['classification'].value_counts().to_dict()}")
    return result_df

# =============================================================================
# OPTIMAL TRANSPORT AND FLOW ANALYSIS
# =============================================================================

def compute_optimal_transport_plan(
    source_activations: torch.Tensor,
    target_activations: torch.Tensor,
    method: str = 'sinkhorn',
    reg: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal transport plan and the cost matrix used.
    Returns a tuple of (transport_plan, cost_matrix).
    """
    # 1. Convert to NumPy arrays with high precision (float64).
    source = source_activations.cpu().numpy().astype(np.float64)
    target = target_activations.cpu().numpy().astype(np.float64)

    def to_dist(acts):
        acts = np.maximum(acts, 0)
        sums = acts.sum(axis=0, keepdims=True)
        sums[sums == 0] = 1.0
        return acts / sums

    source_dist = to_dist(source)
    target_dist = to_dist(target)

    # 2. Compute the cost matrix. It will inherit the float64 type.
    cost_matrix = ot.dist(source.T, target.T)
    
    normalized_cost_matrix = cost_matrix.copy()
    if normalized_cost_matrix.max() > 0:
        normalized_cost_matrix /= normalized_cost_matrix.max()

    # 3. Ensure the marginals are also float64.
    a = np.ones(source_dist.shape[1], dtype=np.float64) / source_dist.shape[1]
    b = np.ones(target_dist.shape[1], dtype=np.float64) / target_dist.shape[1]

    if method == 'sinkhorn':
        transport_plan = ot.sinkhorn(a, b, normalized_cost_matrix, reg)
    else:
        transport_plan = ot.emd(a, b, normalized_cost_matrix)

    return transport_plan, cost_matrix

def compute_information_flow_between_layers(
    layer1_acts: torch.Tensor,
    layer2_acts: torch.Tensor
) -> Dict:
    """Analyze information flow between consecutive layers."""
    try:
        transport_plan, cost_matrix = compute_optimal_transport_plan(layer1_acts, layer2_acts)
        
        transport_cost = np.sum(transport_plan * cost_matrix)
        # Add a slightly larger epsilon for log stability
        flow_entropy = -np.sum(transport_plan * np.log(transport_plan + 1e-9))

        outgoing_flow = np.sum(transport_plan, axis=1)
        incoming_flow = np.sum(transport_plan, axis=0)
        
        return {
            'transport_plan': transport_plan,
            'transport_cost': transport_cost,
            'flow_entropy': flow_entropy,
            'outgoing_flow': outgoing_flow,
            'incoming_flow': incoming_flow,
            'bottleneck_neurons': {
                'layer1': np.argsort(outgoing_flow)[-5:],
                'layer2': np.argsort(incoming_flow)[-5:]
            }
        }
    except Exception as e:
        logger.error(f"Failed to compute information flow due to: {e}")
        # Return a dictionary with default "error" values to prevent crashing downstream.
        return {
            'transport_plan': np.array([]),
            'transport_cost': -1.0,
            'flow_entropy': -1.0,
            'outgoing_flow': np.array([]),
            'incoming_flow': np.array([]),
            'bottleneck_neurons': {}
        }
        
def wasserstein_geodesic_interpolation(
    acts1: torch.Tensor,
    acts2: torch.Tensor,
    num_steps: int = 10
) -> List[torch.Tensor]:
    """Interpolate between activation states using Wasserstein geodesics."""
    # Convert to distributions
    dist1 = acts1.cpu().numpy()
    dist2 = acts2.cpu().numpy()
    
    # Normalize
    dist1 = dist1 / dist1.sum(axis=0, keepdims=True)
    dist2 = dist2 / dist2.sum(axis=0, keepdims=True)
    
    interpolated = []
    for t in np.linspace(0, 1, num_steps):
        # Linear interpolation in Wasserstein space (approximation)
        interp_dist = (1-t) * dist1 + t * dist2
        interpolated.append(torch.from_numpy(interp_dist).float())
    
    return interpolated

# =============================================================================
# ENERGY FUNCTIONALS
# =============================================================================

def compute_activation_diversity_energy(activations: torch.Tensor) -> float:
    """Compute diversity energy functional."""
    # Encourage diverse activations
    corr_matrix = torch.corrcoef(activations.T)
    off_diagonal = corr_matrix[torch.triu(torch.ones_like(corr_matrix), diagonal=1) == 1]
    diversity_energy = torch.mean(torch.abs(off_diagonal))  # Penalize high correlations
    return diversity_energy.item()

def compute_transport_cost_energy(
    activations: torch.Tensor,
    reference_distribution: Optional[torch.Tensor] = None
) -> float:
    """Compute transport cost energy functional."""
    if reference_distribution is None:
        # Use uniform distribution as reference
        reference_distribution = torch.ones_like(activations) / activations.numel()
    
    # Approximate transport cost using Wasserstein distance
    dist_matrix = compute_wasserstein_distance_matrix(torch.stack([
        activations.flatten(),
        reference_distribution.flatten()
    ]).T)
    
    return dist_matrix[0, 1]

def compute_total_energy_functional(
    activations: torch.Tensor,
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """Compute combined energy functional."""
    if weights is None:
        weights = {'diversity': 1.0, 'transport_cost': 0.5}
    
    energies = {}
    
    if 'diversity' in weights:
        energies['diversity'] = compute_activation_diversity_energy(activations)
    
    if 'transport_cost' in weights:
        energies['transport_cost'] = compute_transport_cost_energy(activations)
    
    # Compute weighted total
    total = sum(weights.get(name, 0) * value for name, value in energies.items())
    energies['total'] = total
    
    return energies

# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

class TemporalTopologyTracker:
    """Track topological changes over training time."""
    
    def __init__(self):
        self.snapshots = []
        self.timestamps = []
    
    def add_snapshot(
        self,
        activations: torch.Tensor,
        epoch: int,
        use_wasserstein: bool = False
    ):
        """Add a temporal snapshot."""
        if use_wasserstein:
            dist_matrix = compute_wasserstein_distance_matrix(activations)
        else:
            dist_matrix = compute_distance_matrix(activations)
        
        persistence = compute_persistence_diagrams(dist_matrix)
        scales = get_filtration_scales(dist_matrix)
        criticality = classify_neuron_criticality(dist_matrix, scales)
        energies = compute_total_energy_functional(activations)
        
        snapshot = {
            'epoch': epoch,
            'distance_matrix': dist_matrix,
            'persistence': persistence,
            'criticality': criticality,
            'energies': energies,
            'betti_numbers': self._compute_betti_numbers(persistence)
        }
        
        self.snapshots.append(snapshot)
        self.timestamps.append(epoch)
    
    def _compute_betti_numbers(self, persistence: Dict) -> Dict[int, int]:
        """Compute Betti numbers from persistence diagrams."""
        betti = {}
        for dim, diagram in enumerate(persistence['dgms']):
            # Count features that are still alive at some reasonable threshold
            if len(diagram) > 0:
                alive_features = diagram[diagram[:, 1] == np.inf] if len(diagram) > 0 else []
                betti[dim] = len(alive_features)
            else:
                betti[dim] = 0
        return betti
    
    def compute_flow_evolution(self) -> Dict:
        """Compute evolution statistics across snapshots."""
        if len(self.snapshots) < 2:
            return {}
        
        evolution = {
            'epochs': self.timestamps,
            'energy_evolution': [s['energies']['total'] for s in self.snapshots],
            'betti_evolution': {},
            'criticality_changes': []
        }
        
        # Track Betti number evolution
        max_dim = max(max(s['betti_numbers'].keys()) for s in self.snapshots)
        for dim in range(max_dim + 1):
            evolution['betti_evolution'][f'H{dim}'] = [
                s['betti_numbers'].get(dim, 0) for s in self.snapshots
            ]
        
        # Track criticality changes
        for i in range(1, len(self.snapshots)):
            prev_crit = self.snapshots[i-1]['criticality']
            curr_crit = self.snapshots[i]['criticality']
            
            changes = {
                'epoch': self.timestamps[i],
                'classification_changes': 0
            }
            
            # Count classification changes
            for neuron_id in prev_crit['neuron_id']:
                prev_class = prev_crit[prev_crit['neuron_id'] == neuron_id]['classification'].iloc[0]
                curr_class = curr_crit[curr_crit['neuron_id'] == neuron_id]['classification'].iloc[0]
                if prev_class != curr_class:
                    changes['classification_changes'] += 1
            
            evolution['criticality_changes'].append(changes)
        
        return evolution

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_chunked_distances(neuron_data: np.ndarray, metric: str, chunk_size: int) -> np.ndarray:
    """Compute distance matrix in chunks to manage memory usage."""
    n_neurons = neuron_data.shape[0]
    dist_matrix = np.zeros((n_neurons, n_neurons))
    
    for i in range(0, n_neurons, chunk_size):
        end_i = min(i + chunk_size, n_neurons)
        for j in range(i, n_neurons, chunk_size):
            end_j = min(j + chunk_size, n_neurons)
            
            chunk_i = neuron_data[i:end_i]
            chunk_j = neuron_data[j:end_j]
            
            if i == j:
                chunk_distances = pdist(chunk_i, metric=metric)
                chunk_matrix = squareform(chunk_distances)
                dist_matrix[i:end_i, j:end_j] = chunk_matrix
            else:
                from scipy.spatial.distance import cdist
                chunk_matrix = cdist(chunk_i, chunk_j, metric=metric)
                dist_matrix[i:end_i, j:end_j] = chunk_matrix
                dist_matrix[j:end_j, i:end_i] = chunk_matrix.T
    
    return dist_matrix

def get_characteristic_radius(
    persistence_diagrams: Dict,
    homology_dimension: int
) -> Optional[float]:
    """Get characteristic radius from most persistent feature."""
    
    if homology_dimension >= len(persistence_diagrams['dgms']):
        return None
    
    diagram = persistence_diagrams['dgms'][homology_dimension]
    
    if len(diagram) == 0:
        return None
    
    # Calculate persistence for all features
    persistence_values = diagram[:, 1] - diagram[:, 0]
    most_persistent_idx = np.argmax(persistence_values)
    
    birth, death = diagram[most_persistent_idx]
    return (birth + death) / 2.0