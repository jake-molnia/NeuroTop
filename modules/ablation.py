"""
Neuron ablation strategies and testing framework.
Separates ranking (analysis) from removal (testing).
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Protocol
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging

from .topology import (
    compute_persistence_diagrams, 
    get_characteristic_radius,
    compute_degree_centrality_at_scale
)
from .training import evaluate

logger = logging.getLogger(__name__)

class AblationStrategy(Protocol):
    """Protocol for neuron ranking strategies."""
    
    def rank_neurons(
        self, 
        activations: torch.Tensor, 
        distance_matrix: np.ndarray,
        **kwargs
    ) -> List[int]:
        """Return list of neuron indices ordered by importance (most important first)."""
        ...

class RandomAblation:
    """Random neuron ordering baseline."""
    
    def rank_neurons(
        self, 
        activations: torch.Tensor, 
        distance_matrix: np.ndarray,
        **kwargs
    ) -> List[int]:
        num_neurons = activations.shape[1]
        ranking = np.random.permutation(num_neurons).tolist()
        logger.info(f"Generated random ranking for {num_neurons} neurons")
        return ranking

class KNNDistanceAblation:
    """Rank neurons by average k-nearest neighbor distance."""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def rank_neurons(
        self, 
        activations: torch.Tensor, 
        distance_matrix: np.ndarray,
        **kwargs
    ) -> List[int]:
        
        # Sort distances to find k-nearest neighbors
        sorted_distances = np.sort(distance_matrix, axis=1)
        knn_distances = sorted_distances[:, 1:self.k+1]  # Exclude self-distance
        mean_knn_distances = np.mean(knn_distances, axis=1)
        
        # Higher average distance = more important
        ranking = np.argsort(mean_knn_distances)[::-1].tolist()
        
        logger.info(f"KNN ablation (k={self.k}): distance range [{mean_knn_distances.min():.3f}, {mean_knn_distances.max():.3f}]")
        return ranking

class HomologyDegreeAblation:
    """Rank neurons by degree centrality at characteristic topological scale."""
    
    def __init__(self, homology_dim: int = 1):
        self.homology_dim = homology_dim
    
    def rank_neurons(
        self, 
        activations: torch.Tensor, 
        distance_matrix: np.ndarray,
        **kwargs
    ) -> List[int]:
        
        # Compute persistence diagrams
        diagrams = compute_persistence_diagrams(distance_matrix, max_dimension=self.homology_dim)
        
        # Get characteristic radius from most persistent feature
        char_radius = get_characteristic_radius(diagrams, self.homology_dim)
        
        if char_radius is None:
            logger.warning(f"No H{self.homology_dim} features found, falling back to random")
            return RandomAblation().rank_neurons(activations, distance_matrix)
        
        # Compute degree centrality at characteristic scale
        degrees = compute_degree_centrality_at_scale(distance_matrix, char_radius)
        
        # Higher degree = more important (for removal, we want least important first)
        ranking = np.argsort(degrees).tolist()
        
        logger.info(f"Homology degree (H{self.homology_dim}): radius={char_radius:.3f}, degrees=[{degrees.min()}, {degrees.max()}]")
        return ranking

class HomologyPersistenceAblation:
    """Rank neurons by persistence-weighted centrality across all topological features."""
    
    def __init__(self, homology_dim: int = 1):
        self.homology_dim = homology_dim
    
    def rank_neurons(
        self, 
        activations: torch.Tensor, 
        distance_matrix: np.ndarray,
        **kwargs
    ) -> List[int]:
        
        n_neurons = distance_matrix.shape[0]
        diagrams = compute_persistence_diagrams(distance_matrix, max_dimension=self.homology_dim)
        
        if self.homology_dim >= len(diagrams['dgms']):
            logger.warning(f"H{self.homology_dim} not computed, falling back to random")
            return RandomAblation().rank_neurons(activations, distance_matrix)
        
        diagram = diagrams['dgms'][self.homology_dim]
        
        if len(diagram) == 0:  
            logger.warning(f"No H{self.homology_dim} features found, falling back to random")
            return RandomAblation().rank_neurons(activations, distance_matrix)
        
        # Weighted importance across all features
        total_importance = np.zeros(n_neurons)
        
        for birth, death in diagram:
            persistence = death - birth
            if persistence <= 0:
                continue
                
            radius = (birth + death) / 2.0
            degrees = compute_degree_centrality_at_scale(distance_matrix, radius)
            total_importance += degrees * persistence
        
        ranking = np.argsort(total_importance)[::-1].tolist()
        
        logger.info(f"Persistence-weighted H{self.homology_dim}: {len(diagram)} features, importance range=[{total_importance.min():.3f}, {total_importance.max():.3f}]")
        return ranking

def get_ablation_strategy(strategy_name: str, **params) -> AblationStrategy:
    """Factory function for ablation strategies."""
    
    strategies = {
        'random': RandomAblation,
        'knn_distance': KNNDistanceAblation,
        'homology_degree': HomologyDegreeAblation,
        'homology_persistence': HomologyPersistenceAblation
    }
    
    # Handle suffixed strategies
    base_name = strategy_name.split('_')[0] + '_' + strategy_name.split('_')[1] if '_' in strategy_name else strategy_name
    if base_name not in strategies and strategy_name.startswith('homology_degree'):
        base_name = 'homology_degree'
    
    if base_name not in strategies:
        raise ValueError(f"Unknown ablation strategy: {strategy_name}")
    
    strategy_class = strategies[base_name]
    return strategy_class(**params)

def test_ablation_performance(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    neuron_ranking: List[int],
    global_to_local_map: List[Tuple[str, int]],
    removal_percentages: List[float],
    device: torch.device
) -> pd.DataFrame:
    """
    Test model performance with progressive neuron removal.
    
    Args:
        neuron_ranking: Global neuron indices ordered by removal priority
        global_to_local_map: Maps global index to (layer_name, local_index)
        removal_percentages: Percentages of neurons to remove for testing
    """
    
    results = []
    layer_shapes = model.layer_shapes
    total_neurons = len(neuron_ranking)
    
    logger.info(f"Testing ablation on {total_neurons} neurons across {len(layer_shapes)} layers")
    
    # Baseline performance (no ablation)
    _, baseline_acc = evaluate(model, test_loader, device, masks=None)
    results.append({'percent_removed': 0.0, 'accuracy': baseline_acc})
    logger.info(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    for percent in tqdm(removal_percentages, desc="Testing ablation"):
        if percent == 0:
            continue
            
        # Create masks for this removal percentage
        masks = {name: torch.ones(shape, device=device) for name, shape in layer_shapes.items()}
        
        num_to_remove = int(total_neurons * (percent / 100.0))
        neurons_to_remove = neuron_ranking[:num_to_remove]
        
        # Apply masks based on global-to-local mapping
        for global_idx in neurons_to_remove:
            layer_name, local_idx = global_to_local_map[global_idx]
            if layer_name in masks:
                # Flatten mask and set neuron to zero
                mask_flat = masks[layer_name].view(-1)
                if local_idx < len(mask_flat):
                    mask_flat[local_idx] = 0.0
        
        # Test performance with masks
        _, accuracy = evaluate(model, test_loader, device, masks=masks)
        results.append({'percent_removed': percent, 'accuracy': accuracy})
        
        logger.debug(f"{percent}% removed: {accuracy:.2f}% accuracy")
    
    return pd.DataFrame(results)

def run_ablation_analysis(
    activations: torch.Tensor,
    distance_matrix: np.ndarray,
    strategy_name: str,
    strategy_params: Dict
) -> List[int]:
    """
    Run ablation analysis to get neuron ranking.
    Separated from testing for modularity.
    """
    
    logger.info(f"Running ablation analysis: {strategy_name}")
    
    try:
        strategy = get_ablation_strategy(strategy_name, **strategy_params)
        ranking = strategy.rank_neurons(activations, distance_matrix)
        
        logger.info(f"Generated ranking for {len(ranking)} neurons")
        return ranking
        
    except Exception as e:
        logger.error(f"Ablation analysis failed: {e}")
        logger.warning("Falling back to random ranking")
        return RandomAblation().rank_neurons(activations, distance_matrix)