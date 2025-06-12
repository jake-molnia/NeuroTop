# modules/analysis.py

import torch
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
            # The hook function stores the detached output tensor
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
    
    # Register hooks to the specified layers
    hooks = [layer.register_forward_hook(collector.hook(name)) for name, layer in feature_layers.items()]
    
    all_activations = {name: [] for name in feature_layers.keys()}
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            ic(f"Processing batch {i+1}/{len(dataloader)}")
            inputs = inputs.to(device)
            collector.clear()
            model(inputs) # Forward pass triggers the hooks
            
            # Append the collected activations for this batch
            for name in feature_layers.keys():
                all_activations[name].append(collector.activations[name])
    
    # Clean up hooks to prevent memory leaks
    for hook in hooks:
        hook.remove()
    
    # Concatenate activations from all batches into single numpy arrays
    for name, act_list in all_activations.items():
        all_activations[name] = np.vstack(act_list)
    
    ic("Activation collection complete.")
    for name, acts in all_activations.items():
        ic(f"  {name}: {acts.shape}")
        
    return all_activations

# --- Topological Analysis ---

def compute_neuron_distances(activations_dict, max_samples=5000):
    """
    Computes the pairwise Euclidean distance between all neurons,
    treating each neuron's activation profile across samples as a vector.
    """
    ic("Computing neuron distances...")
    
    # Stack all layer activations horizontally: [samples, all_neurons]
    # Note: This assumes the activations in the dict are from the same dataset run.
    all_neurons_activation = np.hstack(list(activations_dict.values()))
    
    # Transpose to get [all_neurons, samples]
    neurons = all_neurons_activation.T
    
    # Subsample samples for computational efficiency if necessary
    if neurons.shape[1] > max_samples:
        ic(f"Subsampling from {neurons.shape[1]} to {max_samples} samples.")
        idx = np.random.choice(neurons.shape[1], max_samples, replace=False)
        neurons = neurons[:, idx]
    
    # Compute pairwise distance matrix
    distance_matrix = squareform(pdist(neurons, metric='euclidean'))
    
    ic(f"Distance matrix shape: {distance_matrix.shape}")
    return distance_matrix, neurons

def run_persistent_homology(distance_matrix, maxdim=2, thresh=25.0):
    """Runs Ripser to compute persistence diagrams from the distance matrix."""
    ic(f"Running persistent homology (maxdim={maxdim}, thresh={thresh})...")
    result = ripser(distance_matrix, metric='precomputed', maxdim=maxdim, thresh=thresh)
    
    for dim in range(maxdim + 1):
        ic(f"Found {len(result['dgms'][dim])} features in dimension {dim}.")
    
    return result

# --- Visualization ---

def plot_persistence_diagram(result, title, save_path=None):
    """Plots the persistence diagrams for dimensions 0, 1, and 2."""
    ic(f"Plotting persistence diagram: {title}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dims = min(3, len(result['dgms']))

    for dim in range(dims):
        dgm = result['dgms'][dim]
        ax = axes[dim]
        if len(dgm) > 0:
            ax.scatter(dgm[:, 0], dgm[:, 1], alpha=0.75, s=25)
            max_val = np.max(dgm) if dgm.size > 0 else 1
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

def plot_neuron_embedding(neurons, model_name, save_path=None):
    """
    Computes and plots a 2D t-SNE embedding of the neurons,
    colored by their layer of origin.
    """
    ic("Computing and plotting neuron t-SNE embedding...")
    
    # NOTE: These layer sizes are hardcoded for MLPnet.
    # For a different model, this part would need to be adapted.
    layer_sizes = [512, 512, 256, 256, 128]
    layer_labels = []
    for i, size in enumerate(layer_sizes):
        layer_labels.extend([f'Layer {i+1}'] * size)

    # PCA for dimensionality reduction before t-SNE
    if neurons.shape[1] > 50:
      ic("Running PCA to reduce features for t-SNE.")
      pca = PCA(n_components=50, random_state=42)
      neurons_reduced = pca.fit_transform(neurons)
    else:
      neurons_reduced = neurons

    # t-SNE for 2D embedding
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding = tsne.fit_transform(neurons_reduced)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=layer_labels,
                    palette='viridis', s=20, alpha=0.8, linewidth=0)
    
    plt.title(f't-SNE Embedding of Neurons for {model_name}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Layer', markerscale=2)
    plt.gca().set_aspect('equal', 'box')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        ic(f"Saved plot to: {save_path}")
    plt.close()
