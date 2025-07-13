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

def plot_rf_heatmap_by_layer(topology_states: List[Dict[str, Any]], figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Plot RF heatmaps for each layer separately"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get all unique layers
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    # Create subplots for each layer
    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers, 1, figsize=(figsize[0], figsize[1] * n_layers // 3))
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx, layer_name in enumerate(layer_names):
        # Collect RF values for this layer across epochs
        layer_rf_matrix = []
        for state in topology_states:
            rf_values = state.get('rf_values', {}).get(layer_name, np.array([]))
            layer_rf_matrix.append(rf_values)
        
        # Convert to matrix (neurons x epochs)
        if layer_rf_matrix and len(layer_rf_matrix[0]) > 0:
            rf_matrix = np.array(layer_rf_matrix).T  # Transpose to get neurons x epochs
            
            # Create heatmap
            im = axes[layer_idx].imshow(rf_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
            axes[layer_idx].set_title(f'RF Evolution - {layer_name}')
            axes[layer_idx].set_xlabel('Epoch')
            axes[layer_idx].set_ylabel('Neuron Index')
            
            # Set x-ticks to epoch numbers
            axes[layer_idx].set_xticks(range(len(epochs)))
            axes[layer_idx].set_xticklabels(epochs)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[layer_idx], label='RF Value')
    
    plt.tight_layout()
    return fig


def plot_rf_heatmap_network(topology_states: List[Dict[str, Any]], figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """Plot network-wide RF heatmap with all neurons stacked"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Collect all RF values across layers and epochs
    all_rf_matrix = []
    layer_boundaries = []
    layer_labels = []
    current_neuron_idx = 0
    
    # Get layer names from first state
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    for layer_name in layer_names:
        layer_rf_across_epochs = []
        for state in topology_states:
            rf_values = state.get('rf_values', {}).get(layer_name, np.array([]))
            layer_rf_across_epochs.append(rf_values)
        
        if layer_rf_across_epochs and len(layer_rf_across_epochs[0]) > 0:
            layer_matrix = np.array(layer_rf_across_epochs).T  # neurons x epochs
            all_rf_matrix.append(layer_matrix)
            
            n_neurons_in_layer = layer_matrix.shape[0]
            layer_boundaries.append(current_neuron_idx + n_neurons_in_layer)
            layer_labels.append((current_neuron_idx + n_neurons_in_layer // 2, layer_name))
            current_neuron_idx += n_neurons_in_layer
    
    # Concatenate all layers
    if all_rf_matrix:
        full_matrix = np.vstack(all_rf_matrix)
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(full_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Add layer boundaries
        for boundary in layer_boundaries[:-1]:
            ax.axhline(y=boundary - 0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add layer labels
        for pos, label in layer_labels:
            ax.text(-0.5, pos, label, rotation=90, ha='right', va='center', fontweight='bold')
        
        ax.set_title('Network-wide RF Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Neuron Index (by Layer)')
        
        # Set x-ticks to epoch numbers
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels(epochs)
        
        plt.colorbar(im, ax=ax, label='RF Value')
        plt.tight_layout()
        
        return fig
    else:
        # Return empty figure if no data
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No RF data available', ha='center', va='center', transform=ax.transAxes)
        return fig


def plot_rf_distribution_evolution(topology_states: List[Dict[str, Any]], figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Plot RF distribution evolution as ridge plots (mountain range effect)"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get layer names
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    fig, axes = plt.subplots(len(layer_names), 1, figsize=(figsize[0], figsize[1] * len(layer_names) // 3))
    if len(layer_names) == 1:
        axes = [axes]
    
    for layer_idx, layer_name in enumerate(layer_names):
        ax = axes[layer_idx]
        
        # Collect RF values for this layer across epochs
        for epoch_idx, state in enumerate(topology_states):
            rf_values = state.get('rf_values', {}).get(layer_name, np.array([]))
            
            if len(rf_values) > 0:
                # Create histogram/density
                counts, bins = np.histogram(rf_values, bins=30, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Offset by epoch for ridge effect
                y_offset = epoch_idx * 0.5
                ax.fill_between(bin_centers, y_offset, y_offset + counts * 0.4, 
                               alpha=0.7, label=f'Epoch {epochs[epoch_idx]}')
                ax.plot(bin_centers, y_offset + counts * 0.4, color='black', alpha=0.8, linewidth=1)
        
        ax.set_title(f'RF Distribution Evolution - {layer_name}')
        ax.set_xlabel('RF Value')
        ax.set_ylabel('Epoch (offset)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_rf_violin_evolution(topology_states: List[Dict[str, Any]], figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """Plot RF distribution evolution as violin plots"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get layer names
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    fig, axes = plt.subplots(1, len(layer_names), figsize=(figsize[0], figsize[1]))
    if len(layer_names) == 1:
        axes = [axes]
    
    for layer_idx, layer_name in enumerate(layer_names):
        ax = axes[layer_idx]
        
        # Collect RF values for violin plots
        rf_data = []
        epoch_labels = []
        
        for state in topology_states:
            rf_values = state.get('rf_values', {}).get(layer_name, np.array([]))
            if len(rf_values) > 0:
                rf_data.append(rf_values)
                epoch_labels.append(f"E{state['epoch']}")
        
        if rf_data:
            # Create violin plot
            parts = ax.violinplot(rf_data, positions=range(len(rf_data)), showmeans=True, showmedians=True)
            
            # Customize colors
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)
            
            ax.set_title(f'RF Distribution Evolution\n{layer_name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RF Value')
            ax.set_xticks(range(len(epoch_labels)))
            ax.set_xticklabels(epoch_labels, rotation=45)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_rf_box_evolution(topology_states: List[Dict[str, Any]], figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """Plot RF quartile evolution as box plots with trajectories"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get layer names
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    fig, axes = plt.subplots(1, len(layer_names), figsize=(figsize[0], figsize[1]))
    if len(layer_names) == 1:
        axes = [axes]
    
    for layer_idx, layer_name in enumerate(layer_names):
        ax = axes[layer_idx]
        
        # Collect statistics
        medians, q25s, q75s, means = [], [], [], []
        outliers_x, outliers_y = [], []
        
        for epoch_idx, state in enumerate(topology_states):
            rf_values = state.get('rf_values', {}).get(layer_name, np.array([]))
            
            if len(rf_values) > 0:
                medians.append(np.median(rf_values))
                q25s.append(np.percentile(rf_values, 25))
                q75s.append(np.percentile(rf_values, 75))
                means.append(np.mean(rf_values))
                
                # Find outliers (beyond 1.5 * IQR)
                iqr = q75s[-1] - q25s[-1]
                lower_bound = q25s[-1] - 1.5 * iqr
                upper_bound = q75s[-1] + 1.5 * iqr
                outliers = rf_values[(rf_values < lower_bound) | (rf_values > upper_bound)]
                outliers_x.extend([epoch_idx] * len(outliers))
                outliers_y.extend(outliers)
        
        if medians:
            # Plot trajectories
            epoch_range = range(len(epochs))
            ax.plot(epoch_range, medians, 'o-', label='Median', linewidth=2, markersize=6)
            ax.plot(epoch_range, means, 's-', label='Mean', linewidth=2, markersize=6)
            ax.fill_between(epoch_range, q25s, q75s, alpha=0.3, label='IQR')
            
            # Plot outliers
            if outliers_x:
                ax.scatter(outliers_x, outliers_y, alpha=0.5, s=10, color='red', label='Outliers')
            
            ax.set_title(f'RF Statistics Evolution\n{layer_name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RF Value')
            ax.set_xticks(epoch_range)
            ax.set_xticklabels(epochs)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig