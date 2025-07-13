import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.manifold import TSNE
import seaborn as sns


def plot_distance_matrix(topology: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)) -> Figure:
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

def plot_persistence_diagram(topology: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)) -> Figure:
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

def plot_tsne_2d(topology: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)) -> Figure:
    neuron_matrix = topology['neuron_matrix']
    neuron_info = topology['neuron_info']
    total_neurons = topology['total_neurons']
    perplexity = min(30, total_neurons - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(neuron_matrix)
    unique_layers = list(set(info['layer'] for info in neuron_info)) # Color mapping
    colors = [f'C{i}' for i in range(len(unique_layers))]  # Use default matplotlib color cycle
    layer_color_map = {layer: colors[i % len(colors)] for i, layer in enumerate(unique_layers)}
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

def plot_tsne_3d(topology: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)) -> Figure:
    neuron_matrix = topology['neuron_matrix']
    neuron_info = topology['neuron_info']
    total_neurons = topology['total_neurons']
    perplexity = min(30, total_neurons - 1)
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(neuron_matrix)
    unique_layers = list(set(info['layer'] for info in neuron_info)) # Color mapping
    colors = [f'C{i}' for i in range(len(unique_layers))]  # Use default matplotlib color cycle
    layer_color_map = {layer: colors[i % len(colors)] for i, layer in enumerate(unique_layers)}
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

def plot_betti_numbers(topology: Dict[str, Any], figsize: Tuple[int, int] = (8, 6)) -> Figure:
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

def plot_layer_composition(topology: Dict[str, Any], figsize: Tuple[int, int] = (10, 6)) -> Figure:
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

def plot_rf_heatmap_by_layer(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (15, 10)) -> Figure:
    """Plot RF heatmaps for each layer separately for a specific RF dimension"""
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
            rf_values = state.get('rf_values', {}).get(layer_name, {})
            if isinstance(rf_values, dict) and rf_dim in rf_values:
                layer_rf_matrix.append(rf_values[rf_dim])
            else:
                # Fallback for old format or missing dimension
                layer_rf_matrix.append(np.array([]))
        
        # Convert to matrix (neurons x epochs)
        if layer_rf_matrix and len(layer_rf_matrix[0]) > 0:
            rf_matrix = np.array(layer_rf_matrix).T  # Transpose to get neurons x epochs
            
            # Create heatmap
            im = axes[layer_idx].imshow(rf_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
            axes[layer_idx].set_title(f'{rf_dim.upper()} Evolution - {layer_name}')
            axes[layer_idx].set_xlabel('Epoch')
            axes[layer_idx].set_ylabel('Neuron Index')
            
            # Set x-ticks to epoch numbers
            axes[layer_idx].set_xticks(range(len(epochs)))
            axes[layer_idx].set_xticklabels(epochs)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[layer_idx], label=f'{rf_dim.upper()} Value')
    
    plt.tight_layout()
    return fig

def plot_rf_distribution_evolution(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (15, 10)) -> Figure:
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
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            
            # Handle new RF data structure with dimensions
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
            elif isinstance(layer_rf, np.ndarray) and len(layer_rf) > 0:
                # Fallback for old format
                rf_values = layer_rf
            else:
                rf_values = np.array([])
            
            if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                # Create histogram/density
                counts, bins = np.histogram(rf_values, bins=30, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Offset by epoch for ridge effect
                y_offset = epoch_idx * 0.5
                ax.fill_between(bin_centers, y_offset, y_offset + counts * 0.4, 
                               alpha=0.7, label=f'Epoch {epochs[epoch_idx]}')
                ax.plot(bin_centers, y_offset + counts * 0.4, color='black', alpha=0.8, linewidth=1)
        
        ax.set_title(f'{rf_dim.upper()} Distribution Evolution - {layer_name}')
        ax.set_xlabel(f'{rf_dim.upper()} Value')
        ax.set_ylabel('Epoch (offset)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_rf_violin_evolution(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (15, 8)) -> Figure:
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
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            
            # Handle new RF data structure with dimensions
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
            elif isinstance(layer_rf, np.ndarray) and len(layer_rf) > 0:
                # Fallback for old format
                rf_values = layer_rf
            else:
                rf_values = np.array([])
            
            if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                rf_data.append(rf_values)
                epoch_labels.append(f"E{state['epoch']}")
        
        if rf_data:
            # Create violin plot
            parts = ax.violinplot(rf_data, positions=range(len(rf_data)), showmeans=True, showmedians=True)
            
            # Customize colors
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)
            
            ax.set_title(f'{rf_dim.upper()} Distribution Evolution\n{layer_name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{rf_dim.upper()} Value')
            ax.set_xticks(range(len(epoch_labels)))
            ax.set_xticklabels(epoch_labels, rotation=45)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_rf_box_evolution(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (15, 8)) -> Figure:
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
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            
            # Handle new RF data structure with dimensions
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
            elif isinstance(layer_rf, np.ndarray) and len(layer_rf) > 0:
                # Fallback for old format
                rf_values = layer_rf
            else:
                rf_values = np.array([])
            
            if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
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
            
            ax.set_title(f'{rf_dim.upper()} Statistics Evolution\n{layer_name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{rf_dim.upper()} Value')
            ax.set_xticks(epoch_range)
            ax.set_xticklabels(epochs)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_rf_distribution_evolution_network(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (15, 12)) -> Figure:
    """Plot network-wide RF distribution evolution as ridge plots combining all layers"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get layer names
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all RF values across all layers for each epoch
    for epoch_idx, state in enumerate(topology_states):
        all_rf_values = []
        
        for layer_name in layer_names:
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            
            # Handle new RF data structure with dimensions
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
            elif isinstance(layer_rf, np.ndarray) and len(layer_rf) > 0:
                # Fallback for old format
                rf_values = layer_rf
            else:
                rf_values = np.array([])
            
            if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                all_rf_values.extend(rf_values)
        
        if all_rf_values:
            all_rf_values = np.array(all_rf_values)
            
            # Create histogram/density
            counts, bins = np.histogram(all_rf_values, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Offset by epoch for ridge effect
            y_offset = epoch_idx * 0.5
            ax.fill_between(bin_centers, y_offset, y_offset + counts * 0.4, 
                           alpha=0.7, label=f'Epoch {epochs[epoch_idx]}')
            ax.plot(bin_centers, y_offset + counts * 0.4, color='black', alpha=0.8, linewidth=1)
    
    ax.set_title(f'Network-wide {rf_dim.upper()} Distribution Evolution')
    ax.set_xlabel(f'{rf_dim.upper()} Value')
    ax.set_ylabel('Epoch (offset)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_rf_violin_evolution_network(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (15, 8)) -> Figure:
    """Plot network-wide RF distribution evolution as violin plots combining all layers"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get layer names
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect RF values for violin plots
    rf_data = []
    epoch_labels = []
    
    for state in topology_states:
        all_rf_values = []
        
        for layer_name in layer_names:
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            
            # Handle new RF data structure with dimensions
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
            elif isinstance(layer_rf, np.ndarray) and len(layer_rf) > 0:
                # Fallback for old format
                rf_values = layer_rf
            else:
                rf_values = np.array([])
            
            if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                all_rf_values.extend(rf_values)
        
        if all_rf_values:
            rf_data.append(np.array(all_rf_values))
            epoch_labels.append(f"E{state['epoch']}")
    
    if rf_data:
        # Create violin plot
        parts = ax.violinplot(rf_data, positions=range(len(rf_data)), showmeans=True, showmedians=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        ax.set_title(f'Network-wide {rf_dim.upper()} Distribution Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{rf_dim.upper()} Value')
        ax.set_xticks(range(len(epoch_labels)))
        ax.set_xticklabels(epoch_labels, rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_rf_box_evolution_network(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (15, 8)) -> Figure:
    """Plot network-wide RF quartile evolution as box plots with trajectories combining all layers"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get layer names
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect statistics
    medians, q25s, q75s, means = [], [], [], []
    outliers_x, outliers_y = [], []
    
    for epoch_idx, state in enumerate(topology_states):
        all_rf_values = []
        
        for layer_name in layer_names:
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            
            # Handle new RF data structure with dimensions
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
            elif isinstance(layer_rf, np.ndarray) and len(layer_rf) > 0:
                # Fallback for old format
                rf_values = layer_rf
            else:
                rf_values = np.array([])
            
            if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                all_rf_values.extend(rf_values)
        
        if all_rf_values:
            all_rf_values = np.array(all_rf_values)
            
            medians.append(np.median(all_rf_values))
            q25s.append(np.percentile(all_rf_values, 25))
            q75s.append(np.percentile(all_rf_values, 75))
            means.append(np.mean(all_rf_values))
            
            # Find outliers (beyond 1.5 * IQR)
            iqr = q75s[-1] - q25s[-1]
            lower_bound = q25s[-1] - 1.5 * iqr
            upper_bound = q75s[-1] + 1.5 * iqr
            outliers = all_rf_values[(all_rf_values < lower_bound) | (all_rf_values > upper_bound)]
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
        
        ax.set_title(f'Network-wide {rf_dim.upper()} Statistics Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{rf_dim.upper()} Value')
        ax.set_xticks(epoch_range)
        ax.set_xticklabels(epochs)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_rf_evolution_comparison(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (20, 12)) -> Figure:
    """Create a comprehensive comparison of RF evolution across all visualization types"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get data for all plots
    epochs = [state['epoch'] for state in topology_states]
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    # 1. Network-wide box plot (top-left)
    ax = axes[0, 0]
    medians, q25s, q75s, means = [], [], [], []
    
    for state in topology_states:
        all_rf_values = []
        for layer_name in layer_names:
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
                if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                    all_rf_values.extend(rf_values)
        
        if all_rf_values:
            all_rf_values = np.array(all_rf_values)
            medians.append(np.median(all_rf_values))
            q25s.append(np.percentile(all_rf_values, 25))
            q75s.append(np.percentile(all_rf_values, 75))
            means.append(np.mean(all_rf_values))
    
    if medians:
        epoch_range = range(len(epochs))
        ax.plot(epoch_range, medians, 'o-', label='Median', linewidth=2)
        ax.plot(epoch_range, means, 's-', label='Mean', linewidth=2)
        ax.fill_between(epoch_range, q25s, q75s, alpha=0.3, label='IQR')
        ax.set_title(f'Network-wide {rf_dim.upper()} Statistics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{rf_dim.upper()} Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Per-layer statistics (top-right)
    ax = axes[0, 1]
    colors = [f'C{i}' for i in range(len(layer_names))]  # Use default matplotlib color cycle
    
    for layer_idx, layer_name in enumerate(layer_names):
        layer_medians = []
        for state in topology_states:
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
                if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                    layer_medians.append(np.median(rf_values))
                else:
                    layer_medians.append(np.nan)
            else:
                layer_medians.append(np.nan)
        
        if layer_medians:
            ax.plot(range(len(epochs)), layer_medians, 'o-', 
                   label=layer_name, color=colors[layer_idx], linewidth=2)
    
    ax.set_title(f'Per-Layer {rf_dim.upper()} Median Evolution')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{rf_dim.upper()} Median')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Network-wide distribution evolution (bottom-left)
    ax = axes[1, 0]
    for epoch_idx, state in enumerate(topology_states):
        all_rf_values = []
        for layer_name in layer_names:
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
                if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                    all_rf_values.extend(rf_values)
        
        if all_rf_values:
            all_rf_values = np.array(all_rf_values)
            counts, bins = np.histogram(all_rf_values, bins=30, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            y_offset = epoch_idx * 0.3
            ax.fill_between(bin_centers, y_offset, y_offset + counts * 0.25, 
                           alpha=0.7, label=f'E{epochs[epoch_idx]}')
    
    ax.set_title(f'Network-wide {rf_dim.upper()} Distribution Evolution')
    ax.set_xlabel(f'{rf_dim.upper()} Value')
    ax.set_ylabel('Epoch (offset)')
    
    # 4. Range evolution (bottom-right)
    ax = axes[1, 1]
    ranges, stds = [], []
    
    for state in topology_states:
        all_rf_values = []
        for layer_name in layer_names:
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                rf_values = np.array(layer_rf[rf_dim])
                if len(rf_values) > 0 and np.all(np.isfinite(rf_values)):
                    all_rf_values.extend(rf_values)
        
        if all_rf_values:
            all_rf_values = np.array(all_rf_values)
            ranges.append(np.ptp(all_rf_values))  # peak-to-peak range
            stds.append(np.std(all_rf_values))
    
    if ranges:
        ax.plot(range(len(epochs)), ranges, 'o-', label='Range', linewidth=2)
        ax.plot(range(len(epochs)), stds, 's-', label='Std Dev', linewidth=2)
        ax.set_title(f'Network-wide {rf_dim.upper()} Variability')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{rf_dim.upper()} Variability')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_rf_heatmap_network(topology_states: List[Dict[str, Any]], rf_dim: str = 'rf_0', figsize: Tuple[int, int] = (15, 12)) -> Figure:
    """Plot network-wide RF heatmap with all neurons stacked for a specific RF dimension"""
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
            layer_rf = state.get('rf_values', {}).get(layer_name, {})
            if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                layer_rf_across_epochs.append(layer_rf[rf_dim])
            else:
                layer_rf_across_epochs.append(np.array([]))
        
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
        
        ax.set_title(f'Network-wide {rf_dim.upper()} Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Neuron Index (by Layer)')
        
        # Set x-ticks to epoch numbers
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels(epochs)
        
        plt.colorbar(im, ax=ax, label=f'{rf_dim.upper()} Value')
        plt.tight_layout()
        
        return fig
    else:
        # Return empty figure if no data
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'No {rf_dim} data available', ha='center', va='center', transform=ax.transAxes)
        return fig

def plot_rf_multidim_heatmap_by_layer(topology_states: List[Dict[str, Any]], figsize: Tuple[int, int] = (20, 12)) -> Figure:
    """Plot RF heatmaps for all dimensions side by side for each layer"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get all unique layers and RF dimensions
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    # Find all RF dimensions
    rf_dims = set()
    for layer_rf in first_rf.values():
        if isinstance(layer_rf, dict):
            rf_dims.update(layer_rf.keys())
    rf_dims = sorted(rf_dims)
    
    if not rf_dims:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No RF data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create subplots: layers x dimensions
    n_layers = len(layer_names)
    n_dims = len(rf_dims)
    fig, axes = plt.subplots(n_layers, n_dims, figsize=(figsize[0], figsize[1] * n_layers // 3))
    
    if n_layers == 1 and n_dims == 1:
        axes = [[axes]]
    elif n_layers == 1:
        axes = [axes]
    elif n_dims == 1:
        axes = [[ax] for ax in axes]
    
    for layer_idx, layer_name in enumerate(layer_names):
        for dim_idx, rf_dim in enumerate(rf_dims):
            ax = axes[layer_idx][dim_idx]
            
            # Collect RF values for this layer and dimension across epochs
            layer_rf_matrix = []
            for state in topology_states:
                layer_rf = state.get('rf_values', {}).get(layer_name, {})
                if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                    layer_rf_matrix.append(layer_rf[rf_dim])
                else:
                    layer_rf_matrix.append(np.array([]))
            
            # Convert to matrix (neurons x epochs)
            if layer_rf_matrix and len(layer_rf_matrix[0]) > 0:
                rf_matrix = np.array(layer_rf_matrix).T  # Transpose to get neurons x epochs
                
                # Create heatmap
                im = ax.imshow(rf_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
                ax.set_title(f'{rf_dim.upper()} - {layer_name}')
                ax.set_xlabel('Epoch')
                if dim_idx == 0:  # Only leftmost column gets y-label
                    ax.set_ylabel('Neuron Index')
                
                # Set x-ticks to epoch numbers
                ax.set_xticks(range(len(epochs)))
                ax.set_xticklabels(epochs)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label=f'{rf_dim.upper()} Value', shrink=0.8)
            else:
                ax.text(0.5, 0.5, f'No {rf_dim} data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{rf_dim.upper()} - {layer_name}')
    
    plt.tight_layout()
    return fig

def plot_rf_box_evolution_multidim(topology_states: List[Dict[str, Any]], figsize: Tuple[int, int] = (18, 10)) -> Figure:
    """Plot RF quartile evolution for all dimensions as box plots"""
    epochs = [state['epoch'] for state in topology_states]
    
    # Get layer names and RF dimensions
    first_rf = topology_states[0].get('rf_values', {})
    layer_names = list(first_rf.keys())
    
    # Find all RF dimensions
    rf_dims = set()
    for layer_rf in first_rf.values():
        if isinstance(layer_rf, dict):
            rf_dims.update(layer_rf.keys())
    rf_dims = sorted(rf_dims)
    
    if not rf_dims:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No RF data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create subplots: layers x dimensions
    fig, axes = plt.subplots(len(layer_names), len(rf_dims), figsize=figsize)
    if len(layer_names) == 1 and len(rf_dims) == 1:
        axes = [[axes]]
    elif len(layer_names) == 1:
        axes = [axes]
    elif len(rf_dims) == 1:
        axes = [[ax] for ax in axes]
    
    for layer_idx, layer_name in enumerate(layer_names):
        for dim_idx, rf_dim in enumerate(rf_dims):
            ax = axes[layer_idx][dim_idx]
            
            # Collect statistics for this dimension
            medians, q25s, q75s, means = [], [], [], []
            
            for state in topology_states:
                layer_rf = state.get('rf_values', {}).get(layer_name, {})
                if isinstance(layer_rf, dict) and rf_dim in layer_rf:
                    dim_values = layer_rf[rf_dim]
                    if len(dim_values) > 0:
                        medians.append(np.median(dim_values))
                        q25s.append(np.percentile(dim_values, 25))
                        q75s.append(np.percentile(dim_values, 75))
                        means.append(np.mean(dim_values))
                    else:
                        medians.append(0.0)
                        q25s.append(0.0)
                        q75s.append(0.0)
                        means.append(0.0)
                else:
                    medians.append(0.0)
                    q25s.append(0.0)
                    q75s.append(0.0)
                    means.append(0.0)
            
            if medians:
                # Plot trajectories
                epoch_range = range(len(epochs))
                ax.plot(epoch_range, medians, 'o-', label='Median', linewidth=2, markersize=6)
                ax.plot(epoch_range, means, 's-', label='Mean', linewidth=2, markersize=6)
                ax.fill_between(epoch_range, q25s, q75s, alpha=0.3, label='IQR')
                
                ax.set_title(f'{rf_dim.upper()} - {layer_name}')
                ax.set_xlabel('Epoch')
                if dim_idx == 0:  # Only leftmost column gets y-label
                    ax.set_ylabel(f'{rf_dim.upper()} Value')
                ax.set_xticks(epoch_range)
                ax.set_xticklabels(epochs)
                if layer_idx == 0 and dim_idx == 0:  # Only show legend once
                    ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig