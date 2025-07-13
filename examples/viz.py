#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Tuple, Dict, Any, Optional

class TopologyAnimator:
    def __init__(self, data_file: str):
        """Load topology evolution data from ntop monitoring"""
        self.data = dict(np.load(data_file, allow_pickle=True))
        self.topology_states = self.data['topology_states'].tolist()  # Convert to list
        self.epochs = self.data['epochs']
        self.rf_approach = self._detect_rf_approach()
        print(f"Loaded {len(self.topology_states)} topology states from epochs {self.epochs[0]} to {self.epochs[-1]}")
        if self.rf_approach: print(f"Detected rf approach: {self.rf_approach}")
    
    def _detect_rf_approach(self) -> Optional[str]:
        """Detect which rf approach was used"""
        if len(self.topology_states) == 0: return None  # Fixed: use len() instead of direct boolean
        first_state = self.topology_states[0]
        if 'rf_analysis' in first_state:
            return first_state['rf_analysis'].get('approach')
        elif 'rf_global' in first_state:
            return 'unified'
        elif 'rf_per_layer' in first_state:
            return 'per_layer'
        elif 'rf_per_neuron' in first_state:
            return 'per_neuron'
        return None
    
    def animate_persistence_evolution(self, figsize: Tuple[int, int] = (10, 8)) -> animation.FuncAnimation:
        """Create animation of persistence diagram evolution"""
        fig, ax = plt.subplots(figsize=figsize)
        global_max = 0
        for state in self.topology_states:
            diagrams = state['persistence']['dgms']
            for diagram in diagrams:
                if len(diagram) > 0:
                    finite_mask = ~np.isinf(diagram[:, 1])
                    if np.any(finite_mask):
                        global_max = max(global_max, diagram[finite_mask][:, 1].max())
        
        def animate(frame):
            ax.clear()
            state = self.topology_states[frame]
            epoch = state['epoch']
            diagrams = state['persistence']['dgms']
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            markers = ['o', 's', '^', 'D', 'v']
            
            for dim, diagram in enumerate(diagrams):
                if len(diagram) == 0: continue
                color, marker = colors[dim % len(colors)], markers[dim % len(markers)]
                finite_mask = ~np.isinf(diagram[:, 1])
                finite_points = diagram[finite_mask]
                if len(finite_points) > 0:
                    ax.scatter(finite_points[:, 0], finite_points[:, 1], 
                              c=color, marker=marker, alpha=0.7, s=60, 
                              label=f'H{dim} ({len(finite_points)} finite)')
                infinite_mask = np.isinf(diagram[:, 1])
                infinite_points = diagram[infinite_mask]
                if len(infinite_points) > 0:
                    infinite_y = global_max * 1.1 if global_max > 0 else 1.0
                    ax.scatter(infinite_points[:, 0], [infinite_y] * len(infinite_points),
                              c=color, marker='^', alpha=0.9, s=100,
                              label=f'H{dim} ({len(infinite_points)} infinite)')
            
            if global_max > 0:
                diagonal_max = global_max * 1.2
                ax.plot([0, diagonal_max], [0, diagonal_max], 'k--', alpha=0.5)
                ax.set_xlim(0, diagonal_max)
                ax.set_ylim(0, diagonal_max)
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(f'Persistence Diagram - Epoch {epoch}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        return animation.FuncAnimation(fig, animate, frames=len(self.topology_states), 
                                     interval=1000, repeat=True)
    
    def animate_tsne_evolution(self, figsize: Tuple[int, int] = (12, 8)) -> animation.FuncAnimation:
        """Create animation of t-SNE evolution"""
        fig, ax = plt.subplots(figsize=figsize)
        
        def animate(frame):
            ax.clear()
            state = self.topology_states[frame]
            epoch = state['epoch']
            neuron_matrix = state['neuron_matrix']
            neuron_info = state['neuron_info']
            total_neurons = state['total_neurons']
            perplexity = min(30, total_neurons - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_result = tsne.fit_transform(neuron_matrix)
            unique_layers = list(set(info['layer'] for info in neuron_info))
            colors_map = plt.cm.Set3(np.linspace(0, 1, len(unique_layers)))
            layer_color_map = {layer: colors_map[i] for i, layer in enumerate(unique_layers)}
            
            for layer in unique_layers:
                layer_indices = [i for i, info in enumerate(neuron_info) if info['layer'] == layer]
                layer_tsne = tsne_result[layer_indices]
                ax.scatter(layer_tsne[:, 0], layer_tsne[:, 1], 
                          c=[layer_color_map[layer]], label=layer, alpha=0.7, s=50)
            ax.set_title(f'2D t-SNE - Epoch {epoch} ({total_neurons} neurons)')
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        return animation.FuncAnimation(fig, animate, frames=len(self.topology_states), 
                                     interval=2000, repeat=True)
    
    def animate_rf_evolution(self, figsize: Tuple[int, int] = (12, 8)) -> Optional[animation.FuncAnimation]:
        """Create animation of rf values evolution based on detected approach"""
        if not self.rf_approach:
            print("No rf data found in topology states")
            return None
        
        if self.rf_approach == 'unified':
            return self._animate_rf_unified(figsize)
        elif self.rf_approach == 'per_layer':
            return self._animate_rf_per_layer(figsize)
        elif self.rf_approach == 'per_neuron':
            return self._animate_rf_per_neuron(figsize)
        else:
            print(f"Unknown rf approach: {self.rf_approach}")
            return None
    
    def _animate_rf_unified(self, figsize: Tuple[int, int]) -> animation.FuncAnimation:
        """Animate unified rf values (single global value per epoch)"""
        fig, ax = plt.subplots(figsize=figsize)
        rf_values = []
        for state in self.topology_states:
            if 'rf_analysis' in state:
                rf_values.append(state['rf_analysis'].get('rf_global', 0.0))
            else:
                rf_values.append(state.get('rf_global', 0.0))
        
        def animate(frame):
            ax.clear()
            current_epochs = self.epochs[:frame+1]
            current_rf = rf_values[:frame+1]
            ax.plot(current_epochs, current_rf, 'b-o', linewidth=2, markersize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Global rf Value')
            ax.set_title(f'Global rf Evolution - Epoch {self.epochs[frame]}')
            ax.grid(True, alpha=0.3)
            if len(current_rf) > 0:
                ax.set_ylim(0, max(rf_values) * 1.1)
            ax.set_xlim(self.epochs[0], self.epochs[-1])
        
        return animation.FuncAnimation(fig, animate, frames=len(self.topology_states), 
                                     interval=1000, repeat=True)
    
    def _animate_rf_per_layer(self, figsize: Tuple[int, int]) -> animation.FuncAnimation:
        """Animate per-layer rf values"""
        fig, ax = plt.subplots(figsize=figsize)
        # Check how rf_per_layer is stored - could be in rf_analysis or direct
        first_state = self.topology_states[0]
        if 'rf_analysis' in first_state and 'rf_per_layer' in first_state['rf_analysis']:
            layer_data_key = lambda state: state['rf_analysis']['rf_per_layer']
        else:
            layer_data_key = lambda state: state['rf_per_layer']
        
        layer_names = list(layer_data_key(first_state).keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))
        layer_colors = {layer: colors[i] for i, layer in enumerate(layer_names)}
        all_rf_values = {}
        for layer in layer_names:
            all_rf_values[layer] = []
            for state in self.topology_states:
                rf_data = layer_data_key(state)[layer]
                all_rf_values[layer].append(rf_data.get('rf_value', 0.0))
        max_rf = max(max(values) for values in all_rf_values.values()) if all_rf_values else 1.0
        
        def animate(frame):
            ax.clear()
            current_epochs = self.epochs[:frame+1]
            for layer in layer_names:
                current_rf = all_rf_values[layer][:frame+1]
                ax.plot(current_epochs, current_rf, 'o-', color=layer_colors[layer], 
                       label=layer, linewidth=2, markersize=5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('rf Value')
            ax.set_title(f'Per-Layer rf Evolution - Epoch {self.epochs[frame]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max_rf * 1.1)
            ax.set_xlim(self.epochs[0], self.epochs[-1])
        
        return animation.FuncAnimation(fig, animate, frames=len(self.topology_states), 
                                     interval=1000, repeat=True)
    
    def _animate_rf_per_neuron(self, figsize: Tuple[int, int]) -> animation.FuncAnimation:
        """Animate per-neuron rf statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Check how rf_statistics is stored
        first_state = self.topology_states[0]
        if 'rf_statistics' in first_state:
            stats_key = lambda state: state['rf_statistics']
        else:
            print("No rf_statistics found, cannot create per-neuron animation")
            return None
            
        layer_names = list(stats_key(first_state).keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))
        layer_colors = {layer: colors[i] for i, layer in enumerate(layer_names)}
        stats_evolution = {layer: {'mean': [], 'median': [], 'q70': [], 'max': []} for layer in layer_names}
        
        for state in self.topology_states:
            rf_stats = stats_key(state)
            for layer in layer_names:
                layer_stats = rf_stats.get(layer, {})
                stats_evolution[layer]['mean'].append(layer_stats.get('mean', 0.0))
                stats_evolution[layer]['median'].append(layer_stats.get('median', 0.0))
                stats_evolution[layer]['q70'].append(layer_stats.get('q70', 0.0))
                stats_evolution[layer]['max'].append(layer_stats.get('max', 0.0))
        
        def animate(frame):
            for ax in [ax1, ax2, ax3, ax4]: ax.clear()
            current_epochs = self.epochs[:frame+1]
            
            for layer in layer_names:
                color = layer_colors[layer]
                ax1.plot(current_epochs, stats_evolution[layer]['mean'][:frame+1], 
                        'o-', color=color, label=layer, linewidth=2, markersize=4)
                ax2.plot(current_epochs, stats_evolution[layer]['median'][:frame+1], 
                        'o-', color=color, label=layer, linewidth=2, markersize=4)
                ax3.plot(current_epochs, stats_evolution[layer]['q70'][:frame+1], 
                        'o-', color=color, label=layer, linewidth=2, markersize=4)
                ax4.plot(current_epochs, stats_evolution[layer]['max'][:frame+1], 
                        'o-', color=color, label=layer, linewidth=2, markersize=4)
            
            for ax, title, ylabel in [(ax1, 'Mean rf', 'Mean rf'), (ax2, 'Median rf', 'Median rf'), 
                                     (ax3, '70th Percentile rf', 'Q70 rf'), (ax4, 'Max rf', 'Max rf')]:
                ax.set_title(f'{title} - Epoch {self.epochs[frame]}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(self.epochs[0], self.epochs[-1])
            
            plt.tight_layout()
        
        return animation.FuncAnimation(fig, animate, frames=len(self.topology_states), 
                                     interval=1500, repeat=True)
    
    def animate_rf_heatmap_evolution(self, figsize: Tuple[int, int] = (12, 8)) -> Optional[animation.FuncAnimation]:
        """Create heatmap animation of per-neuron rf values evolution"""
        if self.rf_approach != 'per_neuron':
            print("rf heatmap animation only available for per_neuron approach")
            return None
        
        # Check how per_neuron data is stored
        first_state = self.topology_states[0]
        if 'rf_per_neuron' in first_state:
            neuron_data_key = lambda state: state['rf_per_neuron']
        elif 'rf_analysis' in first_state and 'rf_per_neuron' in first_state['rf_analysis']:
            neuron_data_key = lambda state: state['rf_analysis']['rf_per_neuron']
        else:
            print("No per_neuron rf data found")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        layer_names = list(neuron_data_key(first_state).keys())
        max_neurons = max(len(neuron_data_key(first_state)[layer]) for layer in layer_names)
        all_rf_values = []
        for state in self.topology_states:
            rf_matrix = np.full((len(layer_names), max_neurons), np.nan)
            rf_data = neuron_data_key(state)
            for i, layer in enumerate(layer_names):
                layer_rf = rf_data[layer]
                rf_matrix[i, :len(layer_rf)] = layer_rf
            all_rf_values.append(rf_matrix)
        vmax = max(np.nanmax(rf_matrix) for rf_matrix in all_rf_values)
        
        def animate(frame):
            ax.clear()
            rf_matrix = all_rf_values[frame]
            epoch = self.epochs[frame]
            masked_array = np.ma.masked_invalid(rf_matrix)
            im = ax.imshow(masked_array, cmap='viridis', aspect='auto', 
                          interpolation='nearest', vmin=0, vmax=vmax)
            ax.set_title(f'rf Values Heatmap - Epoch {epoch}')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Layer')
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels(layer_names)
            if frame == 0:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('rf Value')
        
        return animation.FuncAnimation(fig, animate, frames=len(self.topology_states), 
                                     interval=1000, repeat=True)
    
    def save_animation(self, anim: animation.FuncAnimation, filename: str, fps: int = 2):
        """Save animation to file"""
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved to {filename}")

def main():
    """Create all topology evolution animations"""
    data_file = './outputs/fashion_mnist_MLP/topology_evolution.npz'
    output_dir = './outputs/fashion_mnist_MLP'
    
    animator = TopologyAnimator(data_file)
    
    print("Creating persistence evolution animation...")
    persist_anim = animator.animate_persistence_evolution()
    animator.save_animation(persist_anim, f'{output_dir}/persistence_evolution.gif')
    plt.close()
    
    print("Creating t-SNE evolution animation...")
    tsne_anim = animator.animate_tsne_evolution()
    animator.save_animation(tsne_anim, f'{output_dir}/tsne_evolution.gif')
    plt.close()
    
    print("Creating rf evolution animation...")
    rf_anim = animator.animate_rf_evolution()
    if rf_anim:
        animator.save_animation(rf_anim, f'{output_dir}/rf_evolution.gif')
        plt.close()
    
    if animator.rf_approach == 'per_neuron':
        print("Creating rf heatmap evolution animation...")
        rf_heatmap_anim = animator.animate_rf_heatmap_evolution()
        if rf_heatmap_anim:
            animator.save_animation(rf_heatmap_anim, f'{output_dir}/rf_heatmap_evolution.gif')
            plt.close()
    
    print("All animations complete!")

if __name__ == "__main__":
    main()