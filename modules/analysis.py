# modules/analysis.py

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns

# --- Activation Collection ---

class ActivationCollector:
    """A helper class to collect activations using forward hooks."""
    def __init__(self):
        self.activations = {}
    
    def hook(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach().cpu().numpy()
        return fn
    
    def clear(self):
        self.activations = {}

def collect_activations(model, dataloader, device):
    """
    Runs the dataloader through the model and collects activations from
    the layers specified in model.get_feature_layers().
    """
    ic("Collecting activations...")
    
    collector = ActivationCollector()
    feature_layers = model.get_feature_layers()
    
    hooks = [layer.register_forward_hook(collector.hook(name)) for name, layer in feature_layers.items()]
    
    all_activations = {name: [] for name in feature_layers.keys()}
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            ic(f"Processing batch {i+1}/{len(dataloader)}")
            inputs = inputs.to(device)
            collector.clear()
            model(inputs)
            
            for name in feature_layers.keys():
                all_activations[name].append(collector.activations[name])
    
    for hook in hooks:
        hook.remove()
    
    for name, act_list in all_activations.items():
        all_activations[name] = np.vstack(act_list)
    
    ic("Activation collection complete.")
    for name, acts in all_activations.items():
        ic(f"  {name}: {acts.shape}")
        
    return all_activations

# --- Topological and Distance Analysis ---

def compute_neuron_distances(activations_dict, device, metric='euclidean', max_samples=5000, reduce_dims=None):
    ic(f"Computing neuron distances with metric: '{metric}' on device: '{device}'...")
    
    layer_names = list(activations_dict.keys())
    layer_activations = list(activations_dict.values())
    
    layer_labels = []
    layer_indices = [0]
    cumulative_neurons = 0
    for i, name in enumerate(layer_names):
        num_neurons = layer_activations[i].shape[1]
        layer_labels.extend([name] * num_neurons)
        cumulative_neurons += num_neurons
        layer_indices.append(cumulative_neurons)

    all_neurons_activation = np.hstack(layer_activations)
    neurons_np = all_neurons_activation.T
    
    ic(f"Neuron data shape before processing: {neurons_np.shape}")
    
    # NEW: Reduce feature dimensions (not number of neurons)
    if reduce_dims is not None and neurons_np.shape[1] > reduce_dims:
        ic(f"Reducing feature dimensions from {neurons_np.shape[1]} to {reduce_dims} using Random Projection.")
        from sklearn.random_projection import GaussianRandomProjection
        rp = GaussianRandomProjection(n_components=reduce_dims, random_state=42)
        neurons_np = rp.fit_transform(neurons_np)
        ic(f"Neuron data shape after Random Projection: {neurons_np.shape}")
    
    if neurons_np.shape[1] > max_samples:
        ic(f"Subsampling from {neurons_np.shape[1]} to {max_samples} samples.")
        idx = np.random.choice(neurons_np.shape[1], max_samples, replace=False)
        neurons_np = neurons_np[:, idx]
    
    # --- GPU Accelerated Distance Calculation ---
    neurons_torch = torch.from_numpy(neurons_np).to(device)
    
    if metric == 'euclidean':
        distance_matrix_torch = torch.cdist(neurons_torch, neurons_torch, p=2)
    elif metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        norm_neurons = F.normalize(neurons_torch, p=2, dim=1)
        cos_sim = torch.mm(norm_neurons, norm_neurons.t())
        distance_matrix_torch = 1 - cos_sim
        # Clamp for numerical stability
        distance_matrix_torch = torch.clamp(distance_matrix_torch, min=0.0)
    else:
        raise ValueError(f"Metric '{metric}' not supported for GPU computation.")
        
    distance_matrix = distance_matrix_torch.cpu().numpy()
    
    ic(f"Distance matrix shape: {distance_matrix.shape}")
    return distance_matrix, neurons_np, layer_labels, layer_indices


def run_persistent_homology(distance_matrix, maxdim=2, thresh=25.0):
    """Runs Ripser to compute persistence diagrams."""
    ic(f"Running persistent homology (maxdim={maxdim}, thresh={thresh})...")
    result = ripser(distance_matrix, metric='precomputed', maxdim=maxdim, thresh=thresh, distance_matrix=True)

    for dim in range(maxdim + 1):
        ic(f"Found {len(result['dgms'][dim])} features in dimension {dim}.")
    
    return result

# --- Visualization ---

def plot_distance_matrix(distance_matrix, layer_indices, layer_names, title, save_path=None):
    """
    Plots a heatmap of the neuron distance matrix using imshow for efficiency.
    """
    ic(f"Plotting distance matrix: {title}")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=ax)
    
    for idx in layer_indices[1:-1]:
        ax.axhline(y=idx - 0.5, color='w', linestyle='--', linewidth=1.5)
        ax.axvline(x=idx - 0.5, color='w', linestyle='--', linewidth=1.5)
        
    ax.set_title(title)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Neuron Index")
    
    tick_positions = [idx + (next_idx - idx) / 2 for idx, next_idx in zip(layer_indices, layer_indices[1:])]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(layer_names)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        ic(f"Saved plot to: {save_path}")
    plt.close(fig)


def plot_persistence_diagram(result, title, save_path=None):
    """Plots persistence diagrams for dimensions 0, 1, and 2."""
    ic(f"Plotting persistence diagram: {title}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dims = min(3, len(result['dgms']))

    for dim in range(dims):
        dgm = result['dgms'][dim]
        ax = axes[dim]
        if len(dgm) > 0:
            ax.scatter(dgm[:, 0], dgm[:, 1], alpha=0.75, s=25)
            max_val = np.max(dgm[dgm != np.inf]) if np.any(dgm != np.inf) else 1
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(f'Dimension {dim} Features')
            ax.set_aspect('equal', 'box')
        else:
            ax.text(0.5, 0.5, 'No Features', ha='center', va='center')
            ax.set_title(f'Dimension {dim} Features')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        ic(f"Saved plot to: {save_path}")
    plt.close(fig)
    
def plot_betti_curves(result, thresh, title, save_path=None):
    """Calculates and plots Betti curves from persistence diagrams."""
    ic(f"Plotting Betti curves: {title}")

    filtration_steps = np.linspace(0, thresh, 100)
    betti_numbers = {}

    for dim in range(len(result['dgms'])):
        dgm = result['dgms'][dim]
        dgm[dgm == np.inf] = thresh
        betti_dim = [np.sum((dgm[:, 0] <= t) & (dgm[:, 1] > t)) for t in filtration_steps]
        betti_numbers[f'H{dim}'] = betti_dim
    
    plt.figure(figsize=(10, 6))
    for dim_name, numbers in betti_numbers.items():
        plt.plot(filtration_steps, numbers, label=dim_name)
    
    plt.title(title)
    plt.xlabel('Filtration Value (Radius)')
    plt.ylabel('Betti Number (Number of Features)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        ic(f"Saved plot to: {save_path}")
    plt.close()


def _get_tsne_embedding(neurons, perplexity, n_components=2):
    """Helper function to compute t-SNE embedding."""
    if neurons.shape[1] > 50:
      ic(f"Running PCA to reduce features for t-SNE (dim={n_components}).")
      pca = PCA(n_components=50, random_state=42)
      neurons_reduced = pca.fit_transform(neurons)
    else:
      neurons_reduced = neurons
    
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    embedding = tsne.fit_transform(neurons_reduced)
    return embedding

def plot_neuron_embedding_2d(neurons, layer_labels, model_name, perplexity, save_path=None):
    """Computes and plots a 2D t-SNE embedding of the neurons."""
    ic("Computing and plotting 2D neuron t-SNE embedding...")
    embedding = _get_tsne_embedding(neurons, perplexity, n_components=2)
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=layer_labels,
                    palette='viridis', s=20, alpha=0.8, linewidth=0)
    
    plt.title(f'2D t-SNE Embedding of Neurons for {model_name} (Perplexity: {perplexity})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Layer', markerscale=2)
    plt.gca().set_aspect('equal', 'box')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        ic(f"Saved plot to: {save_path}")
    plt.close()

def plot_neuron_embedding_3d(neurons, layer_labels, model_name, perplexity, save_path=None):
    """Computes and plots a 3D t-SNE embedding of the neurons."""
    ic("Computing and plotting 3D neuron t-SNE embedding...")
    embedding = _get_tsne_embedding(neurons, perplexity, n_components=3)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_layers = sorted(list(set(layer_labels)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
    layer_to_color = dict(zip(unique_layers, colors))
    
    scatter_colors = [layer_to_color[label] for label in layer_labels]

    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
               c=scatter_colors, s=15, alpha=0.7)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                              label=layer, markerfacecolor=color, markersize=10)
                       for layer, color in layer_to_color.items()]
    ax.legend(handles=legend_elements, title='Layer')

    ax.set_title(f'3D t-SNE Embedding of Neurons for {model_name} (Perplexity: {perplexity})')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        ic(f"Saved plot to: {save_path}")
    plt.close()
