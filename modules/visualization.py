"""
Visualization utilities with wandb integration.
Clean, consistent plotting interface across all experiment types.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.manifold import TSNE
from persim import plot_diagrams
import logging

logger = logging.getLogger(__name__)

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PlotConfig:
    """Configuration for consistent plotting."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300,
        save_format: str = 'png',
        style: str = 'seaborn-v0_8-whitegrid'
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.save_format = save_format
        self.style = style

DEFAULT_CONFIG = PlotConfig()

def plot_persistence_diagram(
    persistence_diagrams: Dict,
    title: str = "Persistence Diagram",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot persistence diagram from ripser output."""
    
    fig, ax = plt.subplots(figsize=config.figsize)
    plot_diagrams(persistence_diagrams['dgms'], ax=ax)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    
    logger.debug(f"Generated persistence diagram: {title}")
    return fig

def plot_distance_matrix(
    distance_matrix: np.ndarray,
    title: str = "Distance Matrix",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot neuron distance matrix heatmap."""
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Subsample for large matrices to keep visualization readable
    if distance_matrix.shape[0] > 500:
        step = distance_matrix.shape[0] // 500
        subsample = distance_matrix[::step, ::step]
        logger.info(f"Subsampled distance matrix from {distance_matrix.shape} to {subsample.shape}")
    else:
        subsample = distance_matrix
    
    sns.heatmap(subsample, cmap='viridis', ax=ax, cbar=True)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Neuron Index")
    plt.tight_layout()
    
    return fig

def plot_tsne_embedding(
    activations: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "t-SNE Embedding",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot t-SNE embedding of neuron activations."""
    
    # Transpose so neurons are rows
    if activations.shape[0] > activations.shape[1]:
        neuron_data = activations.T
    else:
        neuron_data = activations
    
    logger.info(f"Computing t-SNE for {neuron_data.shape[0]} neurons")
    
    # Subsample for large datasets
    if neuron_data.shape[0] > 1000:
        indices = np.random.choice(neuron_data.shape[0], 1000, replace=False)
        neuron_data = neuron_data[indices]
        if labels is not None:
            labels = labels[indices]
        logger.info("Subsampled to 1000 neurons for t-SNE")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, neuron_data.shape[0]-1))
    embedding = tsne.fit_transform(neuron_data)
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    if labels is not None:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    
    return fig

def plot_ablation_curves(
    results_df: pd.DataFrame,
    group_by: Optional[str] = None,
    title: str = "Ablation Performance",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot ablation performance curves."""
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    if group_by and group_by in results_df.columns:
        # Multiple curves grouped by specified column
        for group_name, group_data in results_df.groupby(group_by):
            ax.plot(group_data['percent_removed'], group_data['accuracy'], 
                   marker='o', label=str(group_name), linewidth=2, markersize=4)
        ax.legend(title=group_by.replace('_', ' ').title())
    else:
        # Single curve
        ax.plot('percent_removed', 'accuracy', data=results_df, 
               marker='o', linewidth=2, markersize=6)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Percentage of Neurons Removed (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_criticality_distribution(
    criticality_df: pd.DataFrame,
    title: str = "Neuron Criticality Distribution",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot distribution of neuron classifications."""
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Order categories logically
    order = ['Core', 'Conditional', 'Redundant']
    available_categories = [cat for cat in order if cat in criticality_df['classification'].values]
    
    bars = sns.countplot(data=criticality_df, x='classification', order=available_categories, ax=ax)
    
    # Add count labels on bars
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Neuron Classification")
    ax.set_ylabel("Number of Neurons")
    plt.tight_layout()
    
    return fig

def plot_degree_evolution(
    criticality_df: pd.DataFrame,
    scales: np.ndarray,
    max_neurons_per_class: int = 10,
    title: str = "Degree Evolution Across Scales",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Plot degree evolution for sample neurons from each class."""
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    degree_cols = [col for col in criticality_df.columns if col.startswith('degree_scale_')]
    
    colors = {'Core': 'red', 'Conditional': 'orange', 'Redundant': 'blue'}
    
    for classification in ['Core', 'Conditional', 'Redundant']:
        subset = criticality_df[criticality_df['classification'] == classification]
        if len(subset) == 0:
            continue
        
        # Sample neurons to avoid overcrowding
        sample_size = min(max_neurons_per_class, len(subset))
        sampled = subset.sample(n=sample_size, random_state=42)
        
        for idx, row in sampled.iterrows():
            degrees = row[degree_cols].values
            ax.plot(scales, degrees, color=colors.get(classification, 'gray'), 
                   alpha=0.6, linewidth=1)
        
        # Plot mean for each class
        mean_degrees = subset[degree_cols].mean().values
        ax.plot(scales, mean_degrees, color=colors.get(classification, 'gray'),
               linewidth=3, label=f'{classification} (mean)')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Filtration Scale")
    ax.set_ylabel("Degree Centrality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_ranking_comparison(
    ranking1: List[int],
    ranking2: List[int],
    name1: str,
    name2: str,
    title: str = "Ranking Comparison",
    config: PlotConfig = DEFAULT_CONFIG
) -> plt.Figure:
    """Compare two neuron rankings with scatter plot."""
    
    # Create rank mappings
    rank1_map = {neuron_id: rank for rank, neuron_id in enumerate(ranking1)}
    rank2_map = {neuron_id: rank for rank, neuron_id in enumerate(ranking2)}
    
    all_neurons = set(ranking1) | set(ranking2)
    max_rank = max(len(ranking1), len(ranking2))
    
    ranks1, ranks2 = [], []
    for neuron_id in all_neurons:
        ranks1.append(rank1_map.get(neuron_id, max_rank))
        ranks2.append(rank2_map.get(neuron_id, max_rank))
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    ax.scatter(ranks1, ranks2, alpha=0.6)
    
    # Add diagonal line for perfect agreement
    max_val = max(max(ranks1), max(ranks2))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Perfect Agreement')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f"Rank by {name1}")
    ax.set_ylabel(f"Rank by {name2}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def save_figure(
    fig: plt.Figure,
    filename: str,
    config: PlotConfig = DEFAULT_CONFIG,
    close: bool = True
) -> None:
    """Save figure with consistent settings."""
    
    fig.savefig(filename, dpi=config.dpi, format=config.save_format, bbox_inches='tight')
    logger.debug(f"Saved figure: {filename}")
    
    if close:
        plt.close(fig)

def create_experiment_summary_plots(
    results_df: pd.DataFrame,
    criticality_df: Optional[pd.DataFrame] = None,
    persistence_diagrams: Optional[Dict] = None,
    config: PlotConfig = DEFAULT_CONFIG
) -> Dict[str, plt.Figure]:
    """Create a standard set of summary plots for an experiment."""
    
    plots = {}
    
    # Ablation performance curve
    plots['ablation_curve'] = plot_ablation_curves(results_df, config=config)
    
    # Criticality distribution if available
    if criticality_df is not None:
        plots['criticality_dist'] = plot_criticality_distribution(criticality_df, config=config)
    
    # Persistence diagram if available
    if persistence_diagrams is not None:
        plots['persistence'] = plot_persistence_diagram(persistence_diagrams, config=config)
    
    logger.info(f"Created {len(plots)} summary plots")
    return plots

def plot_flow_evolution(temporal_data, title="Flow Evolution"):
    """Plot evolution of flow metrics over training."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    epochs = sorted(temporal_data.keys())
    
    # Energy evolution
    energies = [temporal_data[epoch]['energies']['total'] for epoch in epochs]
    axes[0, 0].plot(epochs, energies, 'o-', linewidth=2)
    axes[0, 0].set_title('Energy Evolution')
    
    # Betti numbers
    for dim in [0, 1, 2]:
        betti_values = [temporal_data[epoch]['betti_numbers'].get(dim, 0) for epoch in epochs]
        axes[0, 1].plot(epochs, betti_values, 'o-', label=f'H{dim}')
    axes[0, 1].set_title('Topological Evolution')
    axes[0, 1].legend()
    
    plt.tight_layout()
    return fig

def plot_transport_heatmap(transport_plan, title="Transport Matrix"):
    """Plot transport matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(transport_plan, cmap='viridis')
    plt.colorbar(im, label='Transport Probability')
    ax.set_title(title)
    return fig

def plot_energy_landscape(activations, energy_values, title="Energy Landscape"):
    """Plot 3D energy landscape."""
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(activations)
    
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], energy_values, c=energy_values, cmap='viridis')
    plt.colorbar(scatter)
    ax.set_title(title)
    return fig

def plot_wasserstein_flow_evolution(temporal_data, title="Wasserstein Flow Evolution", config=DEFAULT_CONFIG):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    epochs = sorted(temporal_data.keys())
    
    energies = [temporal_data[epoch]['energies']['total'] for epoch in epochs]
    axes[0, 0].plot(epochs, energies, 'o-', linewidth=2)
    axes[0, 0].set_title('Energy Evolution')
    
    plt.tight_layout()
    return fig

def plot_transport_matrix_heatmap(transport_plan, title="Transport Matrix", config=DEFAULT_CONFIG):
    fig, ax = plt.subplots(figsize=config.figsize)
    im = ax.imshow(transport_plan, cmap='viridis')
    plt.colorbar(im)
    ax.set_title(title)
    return fig

def plot_energy_landscape_3d(activations, energy_values, title="Energy Landscape", config=DEFAULT_CONFIG):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(activations[:, 0], activations[:, 1], energy_values, c=energy_values, cmap='viridis')
    ax.set_title(title)
    return fig

def plot_geodesic_interpolation(interpolation_points, title="Geodesic Interpolation", config=DEFAULT_CONFIG):
    fig, ax = plt.subplots(figsize=config.figsize)
    ax.plot([p.mean().item() for p in interpolation_points], 'o-')
    ax.set_title(title)
    return fig

def plot_information_bottlenecks(flow_analysis, title="Information Bottlenecks", config=DEFAULT_CONFIG):
    fig, ax = plt.subplots(figsize=config.figsize)
    costs = [analysis['transport_cost'] for analysis in flow_analysis.values()]
    ax.bar(range(len(costs)), costs)
    ax.set_title(title)
    return fig

def create_flow_animation(temporal_data, output_path, metric='energy', fps=2):
    return output_path

def create_comprehensive_flow_plots(temporal_data, flow_analyses=None, config=DEFAULT_CONFIG):
    plots = {}
    plots['flow_evolution'] = plot_wasserstein_flow_evolution(temporal_data, config=config)
    return plots