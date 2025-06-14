# modules/__init__.py

import torch

# --- Utility Functions ---
def label_smooth(y, n_class=10):
    """Applies label smoothing."""
    # Ensure y is on the correct device before creating the one_hot tensor
    device = y.device
    y_one_hot = torch.ones(len(y), n_class, device=device) * (0.1 / (n_class - 1))
    # Scatter expects the index tensor to be on the same device
    y_one_hot.scatter_(1, y.unsqueeze(1), 0.9)
    return y_one_hot

# --- Import all model classes ---
from .models import MLPnet, ResNetForCifar, VGGForCifar

from .analysis import (
    collect_activations,
    compute_neuron_distances, 
    run_persistent_homology,
    plot_distance_matrix,
    plot_persistence_diagram,
    plot_betti_curves,
    plot_neuron_embedding_2d,
    plot_neuron_embedding_3d,
    identify_neuron_clusters,
    evaluate_model_performance,
    plot_performance_degradation,
    _get_tsne_embedding,
    identify_by_distance,
    identify_by_knn_distance,
    identify_by_homology_degree
)

# --- Make sure all names are exported ---
__all__ = [
    # Utility Functions
    "label_smooth",
    # Models
    "MLPnet",
    "ResNetForCifar",
    "VGGForCifar",
    # Analysis Functions
    "collect_activations",
    "compute_neuron_distances", 
    "run_persistent_homology",
    "plot_distance_matrix",
    "plot_persistence_diagram",
    "plot_betti_curves",
    "plot_neuron_embedding_2d",
    "plot_neuron_embedding_3d",
    "identify_neuron_clusters",
    "evaluate_model_performance",
    "plot_performance_degradation",
    "_get_tsne_embedding",
    "identify_by_distance",
    "identify_by_knn_distance",
    "identify_by_homology_degree"
]
