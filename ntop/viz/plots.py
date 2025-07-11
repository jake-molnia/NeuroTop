import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

def plot_persistence_diagram(persistence_result: Dict, max_dim: int = 2, 
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    diagrams = persistence_result['dgms']
    
    fig, axes = plt.subplots(1, max_dim + 1, figsize=figsize)
    if max_dim == 0:
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for dim in range(max_dim + 1):
        ax = axes[dim]
        
        if dim >= len(diagrams) or len(diagrams[dim]) == 0:
            ax.set_title(f'H{dim} (empty)')
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            continue
        
        diagram = diagrams[dim]
        finite_points = diagram[~np.isinf(diagram[:, 1])]
        infinite_points = diagram[np.isinf(diagram[:, 1])]
        
        if len(finite_points) > 0:
            ax.scatter(finite_points[:, 0], finite_points[:, 1], 
                      c=colors[dim % len(colors)], alpha=0.7, s=30)
        
        if len(infinite_points) > 0:
            max_death = finite_points[:, 1].max() if len(finite_points) > 0 else 1.0
            ax.scatter(infinite_points[:, 0], [max_death * 1.1] * len(infinite_points),
                      c=colors[dim % len(colors)], marker='^', s=50, alpha=0.7)
        
        max_val = diagram[:, 0].max() if len(diagram) > 0 else 1.0
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        ax.set_title(f'H{dim} ({len(diagram)} features)')
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_distance_matrix(distance_matrix: np.ndarray, 
                        title: str = 'Distance Matrix',
                        figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Distance')
    
    ax.set_title(title)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Neuron Index')
    
    return fig

def plot_betti_numbers(betti_numbers: Dict[int, int], 
                      title: str = 'Betti Numbers',
                      figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    dims = sorted(betti_numbers.keys())
    values = [betti_numbers[dim] for dim in dims]
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    bars = ax.bar([f'H{dim}' for dim in dims], values, 
                  color=[colors[i % len(colors)] for i in range(len(dims))])
    
    ax.set_title(title)
    ax.set_xlabel('Homology Dimension')
    ax.set_ylabel('Betti Number')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value}', ha='center', va='bottom')
    
    return fig

def plot_betti_evolution(betti_evolution: Dict[int, List[int]], 
                        epochs: Optional[List[int]] = None,
                        title: str = 'Betti Number Evolution',
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    if epochs is None:
        epochs = list(range(len(list(betti_evolution.values())[0])))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for dim, values in betti_evolution.items():
        ax.plot(epochs, values, 'o-', color=colors[dim % len(colors)], 
               label=f'H{dim}', linewidth=2, markersize=4)
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Betti Number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig