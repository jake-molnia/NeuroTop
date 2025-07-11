from .core.models import MLP, simple_mlp
from .core.activations import extract_activations, ActivationHook
from .core.training import Trainer
from .core.monitoring import TopologyMonitor
from .analysis.distances import compute_distances
from .analysis.persistence import compute_persistence, PersistenceAnalyzer
from .analysis.evolution import EvolutionTracker
from .viz.plots import plot_persistence_diagram
from .viz.temporal import plot_betti_evolution, create_training_summary
from .data.mnist import get_mnist_loaders, create_simple_mnist_model
from .data.fashion_mnist import get_fashion_mnist_loaders, create_simple_fashion_mnist_model

__version__ = "0.1.0"