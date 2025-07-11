import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from matplotlib.animation import FuncAnimation
from ..analysis.evolution import EvolutionTracker, detect_phase_transitions

def plot_betti_evolution(evolution_data: Dict, layer_name: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    if layer_name is None:
        layer_name = list(evolution_data['layers'].keys())[0]
    
    layer_data = evolution_data['layers'][layer_name]
    epochs = evolution_data['epochs']
    betti_numbers = layer_data['betti_numbers']
    
    if not betti_numbers:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    max_dim = max(max(b.keys()) for b in betti_numbers)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for dim in range(max_dim + 1):
        values = [b.get(dim, 0) for b in betti_numbers]
        ax.plot(epochs, values, 'o-', color=colors[dim % len(colors)], 
               label=f'H{dim}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Betti Number')
    ax.set_title(f'Topology Evolution - {layer_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_topology_heatmap(evolution_data: Dict, layer_name: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    if layer_name is None:
        layer_name = list(evolution_data['layers'].keys())[0]
    
    layer_data = evolution_data['layers'][layer_name]
    epochs = evolution_data['epochs']
    betti_numbers = layer_data['betti_numbers']
    
    if not betti_numbers:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    max_dim = max(max(b.keys()) for b in betti_numbers)
    
    heatmap_data = []
    for dim in range(max_dim + 1):
        values = [b.get(dim, 0) for b in betti_numbers]
        heatmap_data.append(values)
    
    heatmap_data = np.array(heatmap_data)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Homology Dimension')
    ax.set_title(f'Topology Heatmap - {layer_name}')
    
    ax.set_xticks(range(0, len(epochs), max(1, len(epochs) // 10)))
    ax.set_xticklabels([epochs[i] for i in range(0, len(epochs), max(1, len(epochs) // 10))])
    ax.set_yticks(range(max_dim + 1))
    ax.set_yticklabels([f'H{dim}' for dim in range(max_dim + 1)])
    
    plt.colorbar(im, ax=ax, label='Betti Number')
    
    return fig

def plot_critical_transitions(evolution_data: Dict, transitions: List[Dict],
                             layer_name: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    if layer_name is None:
        layer_name = list(evolution_data['layers'].keys())[0]
    
    layer_data = evolution_data['layers'][layer_name]
    epochs = evolution_data['epochs']
    betti_numbers = layer_data['betti_numbers']
    
    if not betti_numbers:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    max_dim = max(max(b.keys()) for b in betti_numbers)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for dim in range(max_dim + 1):
        values = [b.get(dim, 0) for b in betti_numbers]
        ax.plot(epochs, values, 'o-', color=colors[dim % len(colors)], 
               label=f'H{dim}', linewidth=2, markersize=4, alpha=0.7)
    
    for transition in transitions:
        if transition['epoch'] in epochs:
            ax.axvline(x=transition['epoch'], color='red', linestyle='--', alpha=0.8)
            ax.text(transition['epoch'], ax.get_ylim()[1] * 0.9, 
                   f"H{transition['dimension']}", rotation=90, ha='right')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Betti Number')
    ax.set_title(f'Critical Transitions - {layer_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_learning_phases(evolution_data: Dict, phases: List[Dict],
                        layer_name: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    if layer_name is None:
        layer_name = list(evolution_data['layers'].keys())[0]
    
    layer_data = evolution_data['layers'][layer_name]
    epochs = evolution_data['epochs']
    betti_numbers = layer_data['betti_numbers']
    
    if not betti_numbers:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    max_dim = max(max(b.keys()) for b in betti_numbers)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    phase_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightgray']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, phase in enumerate(phases):
        ax.axvspan(phase['start_epoch'], phase['end_epoch'], 
                  alpha=0.3, color=phase_colors[i % len(phase_colors)],
                  label=f'Phase {i+1}')
    
    for dim in range(max_dim + 1):
        values = [b.get(dim, 0) for b in betti_numbers]
        ax.plot(epochs, values, 'o-', color=colors[dim % len(colors)], 
               label=f'H{dim}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Betti Number')
    ax.set_title(f'Learning Phases - {layer_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_entropy_evolution(evolution_data: Dict, layer_name: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    if layer_name is None:
        layer_name = list(evolution_data['layers'].keys())[0]
    
    layer_data = evolution_data['layers'][layer_name]
    epochs = evolution_data['epochs']
    entropy_data = layer_data['entropy']
    
    if not entropy_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    max_dim = max(max(e.keys()) for e in entropy_data)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for dim in range(max_dim + 1):
        values = [e.get(dim, 0) for e in entropy_data]
        ax.plot(epochs, values, 'o-', color=colors[dim % len(colors)], 
               label=f'H{dim} Entropy', linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Persistence Entropy')
    ax.set_title(f'Entropy Evolution - {layer_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_training_summary(evolution_data: Dict, transitions: List[Dict],
                           phases: List[Dict], figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    layer_names = list(evolution_data['layers'].keys())
    n_layers = len(layer_names)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Betti evolution for first layer
    ax1 = fig.add_subplot(gs[0, :])
    layer_name = layer_names[0]
    layer_data = evolution_data['layers'][layer_name]
    epochs = evolution_data['epochs']
    betti_numbers = layer_data['betti_numbers']
    
    if betti_numbers:
        max_dim = max(max(b.keys()) for b in betti_numbers)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for dim in range(max_dim + 1):
            values = [b.get(dim, 0) for b in betti_numbers]
            ax1.plot(epochs, values, 'o-', color=colors[dim % len(colors)], 
                    label=f'H{dim}', linewidth=2, markersize=4)
        
        for transition in transitions:
            if transition['epoch'] in epochs:
                ax1.axvline(x=transition['epoch'], color='red', linestyle='--', alpha=0.6)
    
    ax1.set_title(f'Topology Evolution - {layer_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Betti Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    if betti_numbers:
        heatmap_data = []
        for dim in range(max_dim + 1):
            values = [b.get(dim, 0) for b in betti_numbers]
            heatmap_data.append(values)
        
        heatmap_data = np.array(heatmap_data)
        im = ax2.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax2.set_title('Topology Heatmap')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dimension')
        plt.colorbar(im, ax=ax2)
    
    # Learning phases
    ax3 = fig.add_subplot(gs[1, 1])
    if phases and betti_numbers:
        phase_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        
        for i, phase in enumerate(phases):
            ax3.axvspan(phase['start_epoch'], phase['end_epoch'], 
                       alpha=0.5, color=phase_colors[i % len(phase_colors)],
                       label=f'Phase {i+1}')
        
        values = [b.get(0, 0) for b in betti_numbers]
        ax3.plot(epochs, values, 'k-', linewidth=2)
        ax3.set_title('Learning Phases')
        ax3.set_xlabel('Epoch')
        ax3.legend()
    
    # Summary statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_text = f"Training Summary\n"
    summary_text += f"Epochs monitored: {len(epochs)}\n"
    summary_text += f"Layers analyzed: {', '.join(layer_names)}\n"
    summary_text += f"Critical transitions detected: {len(transitions)}\n"
    summary_text += f"Learning phases identified: {len(phases)}\n"
    
    if transitions:
        transition_epochs = [t['epoch'] for t in transitions]
        summary_text += f"Transition epochs: {transition_epochs}\n"
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    return fig

def plot_multi_layer_comparison(evolution_data: Dict, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    layer_names = list(evolution_data['layers'].keys())
    n_layers = len(layer_names)
    epochs = evolution_data['epochs']
    
    if n_layers == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No layers to compare', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, sharex=True)
    if n_layers == 1:
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        layer_data = evolution_data['layers'][layer_name]
        betti_numbers = layer_data['betti_numbers']
        
        if betti_numbers:
            max_dim = max(max(b.keys()) for b in betti_numbers)
            
            for dim in range(max_dim + 1):
                values = [b.get(dim, 0) for b in betti_numbers]
                ax.plot(epochs, values, 'o-', color=colors[dim % len(colors)], 
                       label=f'H{dim}', linewidth=2, markersize=3)
        
        ax.set_ylabel('Betti Number')
        ax.set_title(f'{layer_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    
    return fig