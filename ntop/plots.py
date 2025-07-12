import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.manifold import TSNE
import seaborn as sns


def plot_distance_matrix(topology: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    distance_matrix = topology['distance_matrix']
    neuron_info = topology['neuron_info']
    total_neurons = topology['total_neurons']
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
    # Add layer boundaries
    layer_boundaries = []
    current_idx = 0
    unique_layers = []
    seen = set()
    for info in neuron_info:
        layer = info['layer']
        if layer not in seen:
            unique_layers.append(layer)
            seen.add(layer)
    for layer in unique_layers:
        layer_size = sum(1 for info in neuron_info if info['layer'] == layer)
        layer_boundaries.append(current_idx + layer_size)
        current_idx += layer_size
    # Draw boundaries
    for boundary in layer_boundaries[:-1]:
        ax.axhline(y=boundary - 0.5, color='red', linestyle='--', alpha=0.8)
        ax.axvline(x=boundary - 0.5, color='red', linestyle='--', alpha=0.8)
    plt.colorbar(im, ax=ax, label='Distance')
    ax.set_title(f'Neuron Distance Matrix ({total_neurons} neurons)')
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Neuron Index')
    return fig


def plot_persistence_diagram(topology: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    persistence = topology['persistence']
    diagrams = persistence['dgms']
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']
    max_death = 0
    for dim, diagram in enumerate(diagrams):
        if len(diagram) == 0: continue
        color = colors[dim % len(colors)]
        marker = markers[dim % len(markers)]
        finite_mask = ~np.isinf(diagram[:, 1]) # Finite points
        finite_points = diagram[finite_mask]
        if len(finite_points) > 0:
            ax.scatter(finite_points[:, 0], finite_points[:, 1], 
                      c=color, marker=marker, alpha=0.7, s=60, 
                      label=f'H{dim} ({len(finite_points)} finite)')
            max_death = max(max_death, finite_points[:, 1].max())
        infinite_mask = np.isinf(diagram[:, 1]) # Infinite points
        infinite_points = diagram[infinite_mask]
        if len(infinite_points) > 0:
            infinite_y = max_death * 1.1 if max_death > 0 else 1.0
            ax.scatter(infinite_points[:, 0], [infinite_y] * len(infinite_points),
                      c=color, marker='^', alpha=0.9, s=100,
                      label=f'H{dim} ({len(infinite_points)} infinite)')
    # Diagonal line
    if max_death > 0:
        diagonal_max = max_death * 1.2
        ax.plot([0, diagonal_max], [0, diagonal_max], 'k--', alpha=0.5)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title('Persistence Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_tsne_2d(topology: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    neuron_matrix = topology['neuron_matrix']
    neuron_info = topology['neuron_info']
    total_neurons = topology['total_neurons']
    perplexity = min(30, total_neurons - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(neuron_matrix)
    unique_layers = list(set(info['layer'] for info in neuron_info)) # Color mapping
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_layers)))
    layer_color_map = {layer: colors[i] for i, layer in enumerate(unique_layers)}
    fig, ax = plt.subplots(figsize=figsize)
    for layer in unique_layers:
        layer_indices = [i for i, info in enumerate(neuron_info) if info['layer'] == layer]
        layer_tsne = tsne_result[layer_indices]
        ax.scatter(layer_tsne[:, 0], layer_tsne[:, 1], 
                  c=[layer_color_map[layer]], label=layer, 
                  alpha=0.7, s=50)
    ax.set_title(f'2D t-SNE ({total_neurons} neurons)')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_tsne_3d(topology: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    neuron_matrix = topology['neuron_matrix']
    neuron_info = topology['neuron_info']
    total_neurons = topology['total_neurons']
    perplexity = min(30, total_neurons - 1)
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(neuron_matrix)
    unique_layers = list(set(info['layer'] for info in neuron_info)) # Color mapping
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_layers)))
    layer_color_map = {layer: colors[i] for i, layer in enumerate(unique_layers)}
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for layer in unique_layers:
        layer_indices = [i for i, info in enumerate(neuron_info) if info['layer'] == layer]
        layer_tsne = tsne_result[layer_indices]
        ax.scatter(layer_tsne[:, 0], layer_tsne[:, 1], layer_tsne[:, 2],
                  c=[layer_color_map[layer]], label=layer,
                  alpha=0.7, s=50)
    ax.set_title(f'3D t-SNE ({total_neurons} neurons)')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.legend()
    return fig


def plot_betti_numbers(topology: Dict[str, Any], figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    betti_numbers = topology['betti_numbers']
    fig, ax = plt.subplots(figsize=figsize)
    dims = sorted(betti_numbers.keys())
    values = [betti_numbers[dim] for dim in dims]
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    bars = ax.bar([f'H{dim}' for dim in dims], values, 
                  color=[colors[i % len(colors)] for i in range(len(dims))])
    ax.set_title('Betti Numbers')
    ax.set_xlabel('Homology Dimension')
    ax.set_ylabel('Betti Number')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value}', ha='center', va='bottom')
    return fig


# def plot_topology_evolution(evolution: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
#     epochs = evolution['epochs']
#     snapshots = evolution['snapshots']
    
#     # Extract betti evolution
#     all_betti = []
#     for snapshot in snapshots:
#         from .analysis import analyze
#         activations = snapshot['activations']
#         topology = analyze(activations)
#         all_betti.append(topology['betti_numbers'])
    
#     # Get all dimensions
#     all_dims = set()
#     for betti_dict in all_betti:
#         all_dims.update(betti_dict.keys())
#     all_dims = sorted(all_dims)
    
#     fig, ax = plt.subplots(figsize=figsize)
    
#     colors = ['red', 'blue', 'green', 'purple', 'orange']
    
#     for dim in all_dims:
#         values = [betti_dict.get(dim, 0) for betti_dict in all_betti]
#         ax.plot(epochs, values, 'o-', color=colors[dim % len(colors)], 
#                label=f'H{dim}', linewidth=2, markersize=6)
    
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Betti Number')
#     ax.set_title('Topology Evolution')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     return fig


def plot_layer_composition(topology: Dict[str, Any], figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    neuron_info = topology['neuron_info']
    # Count neurons per layer
    layer_counts = {}
    for info in neuron_info:
        layer = info['layer']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    fig, ax = plt.subplots(figsize=figsize)
    layers = list(layer_counts.keys())
    counts = list(layer_counts.values())
    bars = ax.bar(layers, counts, alpha=0.7)
    ax.set_title('Neuron Composition by Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Neurons')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    return fig