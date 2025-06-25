"""
Neural topology analysis framework.
Clean, modular implementation for studying topological properties of neural networks.
"""

import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Version info
__version__ = "0.0.2a"
__author__ = "Jacob Molnia"

# Core imports for easy access
from .config import load_config, create_from_template, validate_config, get_experiment_name
from .data import get_dataloaders, get_dataset_info
from .models import get_model, get_model_for_dataset
from .logging import ExperimentLogger, init_experiment
from .utils import set_seed, get_device, Timer

# Analysis imports
from .activations import extract_activations, normalize_activations, concatenate_layer_activations
from .topology import compute_distance_matrix, compute_persistence_diagrams, classify_neuron_criticality
from .ablation import get_ablation_strategy, run_ablation_analysis, test_ablation_performance
from .visualization import create_experiment_summary_plots

__all__ = [
    # Core utilities
    'load_config', 'create_from_template', 'validate_config', 'get_experiment_name',
    'get_dataloaders', 'get_dataset_info',
    'get_model', 'get_model_for_dataset',
    'ExperimentLogger', 'init_experiment',
    'set_seed', 'get_device', 'Timer',
    
    # Analysis components
    'extract_activations', 'normalize_activations', 'concatenate_layer_activations',
    'compute_distance_matrix', 'compute_persistence_diagrams', 'classify_neuron_criticality',
    'get_ablation_strategy', 'run_ablation_analysis', 'test_ablation_performance',
    'create_experiment_summary_plots'
]