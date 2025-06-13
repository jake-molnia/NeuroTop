# modules/__init__.py

from .models import MLPnet
from .analysis import (
    collect_activations,
    compute_neuron_distances, 
    run_persistent_homology,
    plot_distance_matrix,
    plot_persistence_diagram,
    plot_betti_curves,
    plot_neuron_embedding_2d,
    plot_neuron_embedding_3d
)

__all__ = [
    "MLPnet",
    "collect_activations",
    "compute_neuron_distances", 
    "run_persistent_homology",
    "plot_distance_matrix",
    "plot_persistence_diagram",
    "plot_betti_curves",
    "plot_neuron_embedding_2d",
    "plot_neuron_embedding_3d"
]
