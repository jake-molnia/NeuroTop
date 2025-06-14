# modules/analysis.py
import torch
import numpy as np
from tqdm import tqdm
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import pandas as pd
from .training import evaluate

# --- Activation Extraction ---
def extract_activations(model, data_loader, layers_to_hook, max_samples, device):
    model.to(device)
    model.eval()
    activations = {layer: [] for layer in layers_to_hook}
    feature_layers = model.get_feature_layers()
    hooks = []

    def get_hook(name):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())
        return hook

    for layer_name in layers_to_hook:
        if layer_name not in feature_layers:
            raise ValueError(f"Layer '{layer_name}' not in model's feature layers.")
        layer_to_hook = feature_layers[layer_name]
        hooks.append(layer_to_hook.register_forward_hook(get_hook(layer_name)))

    num_samples = 0
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Extracting Activations"):
            model(data.to(device))
            num_samples += data.size(0)
            if max_samples and num_samples >= max_samples:
                break
    for hook in hooks:
        hook.remove()

    for name, acts in activations.items():
        activations[name] = torch.cat(acts, dim=0)
        if max_samples:
            activations[name] = activations[name][:max_samples]
    
    print("Activation shapes:")
    for name, acts in activations.items():
        print(f"  - {name}: {acts.shape}")
    return activations

# --- Topological & Geometric Analysis ---
def get_distance_matrix(activations, metric='euclidean'):
    """Calculates the pairwise distance matrix between neurons."""
    activations_T = activations.T
    return torch.cdist(activations_T, activations_T, p=2 if metric == 'euclidean' else 1)

def calculate_betti_curves(dist_matrix, maxdim=1):
    diagrams = ripser(dist_matrix.numpy(), maxdim=maxdim, do_cocycles=True, distance_matrix=True)
    return diagrams

# --- Ablation Strategies (These now operate on global data) ---
def identify_by_homology_degree(activations, dist_matrix, **strategy_params):
    print("Calculating neuron order via homology degree...")
    homology_dim = strategy_params.get('homology_dim', 1)
    # FIXME: THIS SOULD ALLOW FOR ALL
    if homology_dim not in [0, 1, 2, 3]:
        raise ValueError(f"Homology dimension must be 0 or 1, got {homology_dim}")

    diagrams = calculate_betti_curves(dist_matrix, maxdim=homology_dim)
    cocycles = diagrams['cocycles'][homology_dim]
    num_neurons = activations.shape[1]
    neuron_importance = np.zeros(num_neurons)

    for cocycle_rep in cocycles:
        for simplex_data in cocycle_rep:
            value = simplex_data[-1]
            indices = simplex_data[:-1]
            for idx in indices:
                idx = int(idx)
                if idx < num_neurons:
                    neuron_importance[idx] += abs(value)
    
    return np.argsort(neuron_importance)[::-1].tolist()

def identify_by_knn_distance(activations, dist_matrix, **strategy_params):
    print("Calculating neuron order via KNN distance...")
    k = strategy_params.get('k', 5)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='precomputed')
    nn.fit(dist_matrix.numpy())
    distances, indices = nn.kneighbors(dist_matrix.numpy())
    
    mean_distances = distances[:, 1:].mean(axis=1)
    
    return np.argsort(mean_distances)[::-1].tolist()

def identify_by_random(activations, dist_matrix, **strategy_params):
    print("Calculating neuron order via random permutation...")
    num_neurons = activations.shape[1]
    return torch.randperm(num_neurons).tolist()

def get_ablation_strategy(name):
    strategies = {
        'homology_degree': identify_by_homology_degree,
        'knn_distance': identify_by_knn_distance,
        'random': identify_by_random,
    }
    if name not in strategies:
        raise ValueError(f"Ablation strategy '{name}' not found!")
    return strategies[name]

# --- Ablation Testing ---
def run_ablation_test(model, test_loader, global_neuron_order, global_to_local_map, percentages, device):
    """
    Performs ablation using a global neuron order and a map to translate
    global indices to specific neurons in specific layers.
    """
    results = []
    layer_shapes = model.layer_shapes
    total_neurons_in_scope = len(global_neuron_order)

    for p in tqdm(percentages, desc=f"Ablating model globally via masking"):
        # Create fresh masks of all ones for each percentage step
        masks = {name: torch.ones(shape, device=device) for name, shape in layer_shapes.items()}
        
        num_to_remove = int(total_neurons_in_scope * (p / 100.0))
        global_indices_to_remove = global_neuron_order[:num_to_remove]
        
        # Use the map to turn off the correct neurons in the correct layers
        for global_idx in global_indices_to_remove:
            layer_name, local_neuron_idx = global_to_local_map[global_idx]
            
            # Ensure the layer is one we intend to mask and has a mask prepared
            if layer_name in masks:
                masks[layer_name][local_neuron_idx] = 0.0

        # Evaluate the model with the generated masks for this percentage
        _, acc = evaluate(model, test_loader, device, masks=masks)
        results.append({'percent_removed': p, 'accuracy': acc})

    return pd.DataFrame(results)

# --- Visualization ---
def plot_persistence_diagram(diagrams, output_path):
    plot_diagrams(diagrams['dgms'], show=False)
    plt.title("Persistence Diagram")
    plt.savefig(output_path)
    plt.close()

def plot_tsne(activations, output_path):
    print("Generating t-SNE plot...")
    activations_T = activations.T.numpy()
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=300, metric='euclidean')
    tsne_results = tsne.fit_transform(activations_T)
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1])
    plt.title("t-SNE of Neurons (Global)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(output_path)
    plt.close()

def plot_distance_matrix(dist_matrix, output_path):
    print("Generating distance matrix plot...")
    plt.figure(figsize=(10, 10))
    sns.heatmap(dist_matrix, cmap='viridis')
    plt.title("Neuron Distance Matrix (Global)")
    plt.savefig(output_path)
    plt.close()
