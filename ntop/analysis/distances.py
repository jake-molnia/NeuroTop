import torch
import numpy as np
from typing import Union, Callable, Dict
from sklearn.metrics.pairwise import pairwise_distances

def euclidean_distance(x: torch.Tensor) -> np.ndarray:
    x_np = x.cpu().numpy()
    return pairwise_distances(x_np, metric='euclidean')

def cosine_distance(x: torch.Tensor) -> np.ndarray:
    x_np = x.cpu().numpy()
    return pairwise_distances(x_np, metric='cosine')

def correlation_distance(x: torch.Tensor) -> np.ndarray:
    x_np = x.cpu().numpy()
    return pairwise_distances(x_np, metric='correlation')

def manhattan_distance(x: torch.Tensor) -> np.ndarray:
    x_np = x.cpu().numpy()
    return pairwise_distances(x_np, metric='manhattan')

def wasserstein_distance(x: torch.Tensor) -> np.ndarray:
    try:
        import ot
    except ImportError:
        raise ImportError("POT library required for Wasserstein distance: pip install pot")
    
    x_np = x.cpu().numpy()
    n_samples = x_np.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            a = np.abs(x_np[i]) + 1e-8
            b = np.abs(x_np[j]) + 1e-8
            a = a / a.sum()
            b = b / b.sum()
            
            M = ot.dist(a.reshape(-1, 1), b.reshape(-1, 1))
            dist = ot.emd2(a, b, M)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    return dist_matrix

DISTANCE_FUNCTIONS: Dict[str, Callable] = {
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
    'correlation': correlation_distance,
    'manhattan': manhattan_distance,
    'wasserstein': wasserstein_distance,
}

def compute_distances(activations: torch.Tensor, metric: str = 'euclidean') -> np.ndarray:
    if metric not in DISTANCE_FUNCTIONS:
        available = list(DISTANCE_FUNCTIONS.keys())
        raise ValueError(f"Unknown metric '{metric}'. Available: {available}")
    
    if activations.dim() > 2:
        activations = activations.flatten(1)
    
    return DISTANCE_FUNCTIONS[metric](activations)

def available_metrics() -> list:
    return list(DISTANCE_FUNCTIONS.keys())