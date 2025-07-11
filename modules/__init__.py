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
from .activations import (
    extract_activations, normalize_activations, concatenate_layer_activations,
    TemporalActivationExtractor, extract_activations_at_intervals,
    track_activation_evolution
)

from .topology import (
    # Distance computations
    compute_distance_matrix, compute_wasserstein_distance_matrix,
    
    # Persistence and topology
    compute_persistence_diagrams, classify_neuron_criticality,
    get_filtration_scales, compute_degree_centrality_at_scale,
    
    # Optimal transport and flow
    compute_optimal_transport_plan, compute_information_flow_between_layers,
    wasserstein_geodesic_interpolation,
    
    # Energy functionals
    compute_total_energy_functional, compute_activation_diversity_energy,
    compute_transport_cost_energy,
    
    # Temporal analysis
    TemporalTopologyTracker
)

from .ablation import (
    get_ablation_strategy, run_ablation_analysis, test_ablation_performance
)

from .visualization import (
    # Standard plots
    create_experiment_summary_plots, plot_persistence_diagram,
    plot_distance_matrix, plot_tsne_embedding, plot_ablation_curves,
    plot_criticality_distribution, plot_degree_evolution,
    
    # Flow visualizations
    plot_wasserstein_flow_evolution, plot_transport_matrix_heatmap,
    plot_energy_landscape_3d, plot_geodesic_interpolation,
    plot_information_bottlenecks, create_flow_animation,
    create_comprehensive_flow_plots
)

from .training import (
    train_and_evaluate, FlowAwareTrainer
)

# Storage and temporal management
from .storage import (
    TemporalActivationStore, FlowArtifactStore, TemporalExperimentManager
)

__all__ = [
    # Core utilities
    'load_config', 'create_from_template', 'validate_config', 'get_experiment_name',
    'get_dataloaders', 'get_dataset_info',
    'get_model', 'get_model_for_dataset',
    'ExperimentLogger', 'init_experiment',
    'set_seed', 'get_device', 'Timer',
    
    # Activation analysis
    'extract_activations', 'normalize_activations', 'concatenate_layer_activations',
    'TemporalActivationExtractor', 'extract_activations_at_intervals',
    'track_activation_evolution',
    
    # Topology and flow analysis
    'compute_distance_matrix', 'compute_wasserstein_distance_matrix',
    'compute_persistence_diagrams', 'classify_neuron_criticality',
    'get_filtration_scales', 'compute_degree_centrality_at_scale',
    'compute_optimal_transport_plan', 'compute_information_flow_between_layers',
    'wasserstein_geodesic_interpolation',
    'compute_total_energy_functional', 'compute_activation_diversity_energy',
    'compute_transport_cost_energy',
    'TemporalTopologyTracker',
    
    # Ablation analysis
    'get_ablation_strategy', 'run_ablation_analysis', 'test_ablation_performance',
    
    # Visualizations
    'create_experiment_summary_plots', 'plot_persistence_diagram',
    'plot_distance_matrix', 'plot_tsne_embedding', 'plot_ablation_curves',
    'plot_criticality_distribution', 'plot_degree_evolution',
    'plot_wasserstein_flow_evolution', 'plot_transport_matrix_heatmap',
    'plot_energy_landscape_3d', 'plot_geodesic_interpolation',
    'plot_information_bottlenecks', 'create_flow_animation',
    'create_comprehensive_flow_plots',
    
    # Training
    'train_and_evaluate', 'FlowAwareTrainer',
    
    # Storage and temporal management
    'TemporalActivationStore', 'FlowArtifactStore', 'TemporalExperimentManager'
]