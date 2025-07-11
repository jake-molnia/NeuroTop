from .core.models import MLP
from .core.activations import extract_activations, ActivationHook
from .core.training import Trainer
from .analysis.distances import compute_distances
from .analysis.persistence import compute_persistence
from .viz.plots import plot_persistence_diagram

__version__ = "0.1.0"