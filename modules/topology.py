"""
Pure topological computations on neural activations.
All functions are stateless and side-effect free.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_distance_matrix(
    activations: torch.Tensor,
    metric: str = 'euclidean',
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """
    Compute pairwise distance matrix between neurons.
    Uses chunking for memory efficiency with large neuron sets.
    """
    
    # Transpose: neurons as rows, samples as columns
    neuron_data = activations.T.cpu().numpy()
    n_neurons = neuron_data.shape[0]
    
    if chunk_size is None or n_neurons <= chunk_size:
        # Standard computation for small matrices
        distances = pdist(neuron_data, metric=metric)
        dist_matrix = squareform(distances)
    else:
        # Chunked computation for large matrices
        logger.info(f"Using chunked computation for {n_neurons} neurons")
        dist_matrix = _compute_chunked_distances(neuron_data, metric, chunk_size)
    
    logger.info(f"Distance matrix: {dist_matrix.shape}, range: [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")
    return dist_matrix

def compute_persistence_diagrams(
    distance_matrix: np.ndarray,
    max_dimension: int = 2
) -> Dict:
    """Compute persistence diagrams using Ripser."""
    
    logger.info(f"Computing persistence up to H{max_dimension}")
    
    try:
        diagrams = ripser(distance_matrix, maxdim=max_dimension, distance_matrix=True)
        
        # Log feature counts
        for dim in range(max_dimension + 1):
            if dim < len(diagrams['dgms']):
                n_features = len(diagrams['dgms'][dim])
                logger.info(f"H{dim}: {n_features} topological features")
        
        return diagrams
    
    except Exception as e:
        logger.error(f"Persistence computation failed: {e}")
        raise

def get_filtration_scales(
    distance_matrix: np.ndarray,
    num_scales: int = 10,
    percentile_range: Tuple[float, float] = (5.0, 95.0)
) -> np.ndarray:
    """Generate filtration scales based on distance distribution."""
    
    # Use upper triangle to avoid diagonal zeros and duplicates
    upper_triangle = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
    
    percentiles = np.linspace(percentile_range[0], percentile_range[1], num_scales)
    scales = np.percentile(upper_triangle, percentiles)
    
    logger.info(f"Generated {num_scales} filtration scales from {percentile_range[0]}% to {percentile_range[1]}%")
    return scales

def compute_degree_centrality_at_scale(
    distance_matrix: np.ndarray,
    scale: float
) -> np.ndarray:
    """Compute degree centrality in graph at given filtration scale."""
    
    # Create adjacency matrix: edge if distance <= scale
    adjacency = (distance_matrix <= scale).astype(int)
    np.fill_diagonal(adjacency, 0)  # Remove self-loops
    
    degrees = np.sum(adjacency, axis=1)
    return degrees

def classify_neuron_criticality(
    distance_matrix: np.ndarray,
    scales: np.ndarray
) -> pd.DataFrame:
    """
    Classify neurons based on degree evolution across scales.
    
    Core: High connectivity at small scales
    Conditional: Becomes important at larger scales  
    Redundant: Low connectivity throughout
    """
    
    n_neurons = distance_matrix.shape[0]
    degree_matrix = np.zeros((n_neurons, len(scales)))
    
    # Compute degree at each scale
    for i, scale in enumerate(scales):
        degree_matrix[:, i] = compute_degree_centrality_at_scale(distance_matrix, scale)
    
    # Classify based on degree evolution patterns
    classifications = []
    max_possible_degree = np.max(degree_matrix)
    
    for neuron_idx in range(n_neurons):
        degrees = degree_matrix[neuron_idx, :]
        max_degree = np.max(degrees)
        
        # Heuristic classification rules
        early_scale_cutoff = int(len(scales) * 0.3)
        is_core = np.any(degrees[:early_scale_cutoff] > max_degree * 0.5)
        is_redundant = max_degree < max_possible_degree * 0.2
        
        if is_core:
            classifications.append('Core')
        elif is_redundant:
            classifications.append('Redundant')
        else:
            classifications.append('Conditional')
    
    # Build result dataframe
    result_df = pd.DataFrame({
        'neuron_id': range(n_neurons),
        'classification': classifications,
        'max_degree': np.max(degree_matrix, axis=1),
        'mean_degree': np.mean(degree_matrix, axis=1)
    })
    
    # Add degree at each scale
    for i, scale in enumerate(scales):
        result_df[f'degree_scale_{i}'] = degree_matrix[:, i]
    
    logger.info(f"Neuron classification: {result_df['classification'].value_counts().to_dict()}")
    return result_df

def find_most_persistent_feature(
    persistence_diagrams: Dict,
    homology_dimension: int
) -> Optional[Tuple[float, float, float]]:
    """
    Find the most persistent feature in given homology dimension.
    Returns (birth, death, persistence) or None if no features exist.
    """
    
    if homology_dimension >= len(persistence_diagrams['dgms']):
        return None
    
    diagram = persistence_diagrams['dgms'][homology_dimension]
    
    if len(diagram) == 0:
        return None
    
    # Calculate persistence for all features
    persistence_values = diagram[:, 1] - diagram[:, 0]
    most_persistent_idx = np.argmax(persistence_values)
    
    birth, death = diagram[most_persistent_idx]
    persistence = persistence_values[most_persistent_idx]
    
    return birth, death, persistence

def get_characteristic_radius(
    persistence_diagrams: Dict,
    homology_dimension: int
) -> Optional[float]:
    """Get characteristic radius from most persistent feature."""
    
    feature = find_most_persistent_feature(persistence_diagrams, homology_dimension)
    if feature is None:
        return None
    
    birth, death, _ = feature
    return (birth + death) / 2.0

def _compute_chunked_distances(
    neuron_data: np.ndarray,
    metric: str,
    chunk_size: int
) -> np.ndarray:
    """Compute distance matrix in chunks to manage memory usage."""
    
    n_neurons = neuron_data.shape[0]
    dist_matrix = np.zeros((n_neurons, n_neurons))
    
    for i in range(0, n_neurons, chunk_size):
        end_i = min(i + chunk_size, n_neurons)
        for j in range(i, n_neurons, chunk_size):
            end_j = min(j + chunk_size, n_neurons)
            
            # Compute pairwise distances for this chunk
            chunk_i = neuron_data[i:end_i]
            chunk_j = neuron_data[j:end_j]
            
            if i == j:
                # Diagonal chunk - use pdist for efficiency
                chunk_distances = pdist(chunk_i, metric=metric)
                chunk_matrix = squareform(chunk_distances)
                dist_matrix[i:end_i, j:end_j] = chunk_matrix
            else:
                # Off-diagonal chunk
                from scipy.spatial.distance import cdist
                chunk_matrix = cdist(chunk_i, chunk_j, metric=metric)
                dist_matrix[i:end_i, j:end_j] = chunk_matrix
                dist_matrix[j:end_j, i:end_i] = chunk_matrix.T
    
    return dist_matrix