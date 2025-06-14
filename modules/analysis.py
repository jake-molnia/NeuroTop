# modules/analysis.py

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import random

# --- Activation Collection ---
class ActivationCollector:
    def __init__(self):
        self.activations = {}
    def hook(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach().cpu().numpy()
        return fn
    def clear(self):
        self.activations = {}

def collect_activations(model, dataloader, device):
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
            # Pass masks=None to use the default (unmasked) forward pass
            model(inputs, masks=None) 
            for name in feature_layers.keys():
                all_activations[name].append(collector.activations[name])
            collector.clear()
    for hook in hooks:
        hook.remove()
    for name, act_list in all_activations.items():
        all_activations[name] = np.vstack(act_list)
    return all_activations

# --- Topological and Distance Analysis ---
def compute_neuron_distances(activations_dict, device, metric='euclidean', max_samples=7000, reduce_dims=None):
    """
    Computes neuron distances, correctly handling both MLP (2D) and CNN (4D) activations.
    """
    ic(f"Computing neuron distances with metric: '{metric}'...")
    layer_names = list(activations_dict.keys())
    
    processed_activations = []
    for name in layer_names:
        activation = activations_dict[name]
        # Check if activation is from a CNN (4D: N, C, H, W)
        if activation.ndim == 4:
            # Average over spatial dimensions (H, W) to get a per-channel activation
            # This gives a representative vector for each "neuron" (channel).
            # Shape becomes (N, C)
            activation = np.mean(activation, axis=(2, 3))
        processed_activations.append(activation)

    layer_labels, layer_indices, cumulative_neurons = [], [0], 0
    for i, name in enumerate(layer_names):
        num_neurons = processed_activations[i].shape[1]
        layer_labels.extend([name] * num_neurons)
        cumulative_neurons += num_neurons
        layer_indices.append(cumulative_neurons)
        
    # hstack now works correctly on the list of 2D arrays
    all_neurons_activation = np.hstack(processed_activations)
    # Transpose to get shape (num_neurons, num_samples)
    neurons_np = all_neurons_activation.T
    
    if neurons_np.shape[0] > max_samples:
        ic(f"Subsampling from {neurons_np.shape[0]} to {max_samples} neurons for distance calculation.")
        idx = np.random.choice(neurons_np.shape[0], max_samples, replace=False)
        neurons_np = neurons_np[idx]
        # Note: Subsampling neurons means the layer_labels and indices are now incorrect for the
        # returned distance matrix. This is acceptable if the goal is just the topology of the subsample.
        # For full-network analysis, max_samples should be >= total neurons.
        layer_labels, layer_indices = None, None # Invalidate these as they no longer match

    # pdist computes pairwise distances between rows
    distance_matrix = squareform(pdist(neurons_np, metric=metric))
    
    ic(f"Distance matrix shape: {distance_matrix.shape}")
    return distance_matrix, neurons_np, layer_labels, layer_indices


def run_persistent_homology(distance_matrix, maxdim=2, thresh=25.0):
    """Runs Ripser to compute persistence diagrams."""
    ic(f"Running persistent homology (maxdim={maxdim}, thresh={thresh})...")

    # KEY FIX: Replace metric='precomputed' with distance_matrix=True
    result = ripser(distance_matrix, distance_matrix=True, maxdim=maxdim, thresh=thresh)

    for dim in range(maxdim + 1):
        num_features = len(result['dgms'][dim])
        ic(f"Found {num_features} features in dimension {dim}.")

    return result

# --- Neuron Ablation and Evaluation ---

def identify_neuron_clusters(embedding, eps, min_samples=2):
    """Identifies neuron clusters using DBSCAN on an embedding."""
    ic(f"Identifying neuron clusters with eps={eps}, min_samples={min_samples}...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embedding)
    labels = clustering.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ic(f"Found {n_clusters} clusters.")
    
    # Group indices by cluster label (ignore noise points, label -1)
    clusters = [np.where(labels == i)[0] for i in range(n_clusters)]
    # Sort clusters by size (smallest first) for progressive removal
    clusters.sort(key=len)
    
    return clusters

def evaluate_model_performance(model, dataloader, device, masks=None):
    """Evaluates model accuracy and loss, correctly passing masks to the model."""
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs, masks=masks) 
            
            if isinstance(outputs, tuple):
                main_out, _ = outputs
            else:
                main_out = outputs
            
            if main_out is None: continue

            loss = criterion(main_out, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(main_out.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_preds / total_samples
    
    return {'accuracy': accuracy, 'loss': avg_loss}

def identify_by_distance(distance_matrix):
    """
    Creates an ordered list of ALL neurons to remove. The first half of the list
    consists of one neuron from each of the closest pairs, ordered by distance.
    The second half consists of their partners.
    """
    ic("Identifying neuron removal order via closest pairs...")
    dist_copy = distance_matrix.copy()
    np.fill_diagonal(dist_copy, np.inf)
    n_neurons = dist_copy.shape[0]

    # Create a list of all unique pairs and their distances
    pairs = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            pairs.append((dist_copy[i, j], i, j))
    
    # Sort all pairs by distance, from smallest to largest
    pairs.sort()

    first_half_removal = []
    second_half_removal = []
    processed = np.zeros(n_neurons, dtype=bool)

    for dist, i, j in pairs:
        if not processed[i] and not processed[j]:
            # Add one neuron to the "more redundant" list
            first_half_removal.append(i)
            # Add its partner to the "less redundant" list
            second_half_removal.append(j)
            # Mark both as having been paired
            processed[i] = True
            processed[j] = True

    # The partners of the least redundant pairs should be removed first among the partners
    second_half_removal.reverse()

    # The final order removes the most redundant half, then the least redundant half
    full_removal_order = first_half_removal + second_half_removal
    
    # Handle case where n_neurons is odd
    if len(full_removal_order) < n_neurons:
        unprocessed_neuron = np.where(~processed)[0][0]
        full_removal_order.append(unprocessed_neuron)

    ic(f"Generated a full removal order for {len(full_removal_order)} neurons.")
    return full_removal_order

def identify_by_knn_distance(distance_matrix, k, strategy):
    """
    Creates a removal order based on the distance to the k-th nearest neighbor.
    'strategy' can be 'core' (remove densest first) or 'outlier' (remove sparsest first).
    """
    ic(f"Identifying neuron removal order via k-NN distance (k={k}, strategy='{strategy}')...")
    
    # Sort each row to find the k-th smallest distance (k+1 because the 0-th is self-distance)
    if k + 1 >= distance_matrix.shape[1]:
        raise ValueError(f"k ({k}) is too large for the number of neurons ({distance_matrix.shape[1]})")
        
    k_dist = np.sort(distance_matrix, axis=1)[:, k]

    # Argsort gives the indices that would sort the array.
    # For 'core' removal, we want to remove neurons with the smallest k-dist first.
    # For 'outlier' removal, we remove those with the largest k-dist first.
    if strategy == 'core':
        # Ascending sort (smallest k-dist first)
        removal_order = np.argsort(k_dist)
    elif strategy == 'outlier':
        # Descending sort (largest k-dist first)
        removal_order = np.argsort(k_dist)[::-1]
    else:
        raise ValueError("k-NN strategy must be 'core' or 'outlier'")

    ic(f"Generated removal order for {len(removal_order)} neurons.")
    return removal_order.tolist()


def identify_by_homology_degree(distance_matrix, homology_result):
    """
    Creates a removal order based on degree centrality in a graph constructed
    at the characteristic radius of the most persistent H1 feature.
    """
    ic("Identifying neuron removal order via homology degree centrality...")
    
    h1_diagram = homology_result['dgms'][1]
    if len(h1_diagram) == 0:
        ic("No H1 features found. Cannot use this strategy.")
        # As a fallback, return a random ordering
        n_neurons = distance_matrix.shape[0]
        removal_order = list(range(n_neurons))
        random.shuffle(removal_order)
        return removal_order

    # Calculate persistence (death - birth) for all H1 features
    persistence = h1_diagram[:, 1] - h1_diagram[:, 0]
    
    # Find the feature with the highest persistence
    most_persistent_feature = h1_diagram[np.argmax(persistence)]
    birth, death = most_persistent_feature[0], most_persistent_feature[1]
    
    # Use the average of birth and death as the characteristic radius
    characteristic_radius = (birth + death) / 2.0
    ic(f"Most persistent H1 feature born at {birth:.2f}, died at {death:.2f}.")
    ic(f"Using characteristic radius: {characteristic_radius:.2f}")

    # Build a graph where edges exist if distance <= characteristic_radius
    adjacency_matrix = (distance_matrix <= characteristic_radius).astype(int)
    np.fill_diagonal(adjacency_matrix, 0) # No self-loops

    # Calculate degree centrality (number of connections for each neuron)
    degree_centrality = np.sum(adjacency_matrix, axis=1)

    # Hypothesis: Neurons with higher degree are more central to the topological feature.
    # We remove the most central/important neurons first to test their impact.
    # Descending sort by degree.
    removal_order = np.argsort(degree_centrality)[::-1]
    
    ic(f"Generated removal order for {len(removal_order)} neurons based on degree.")
    return removal_order.tolist()
    
# --- Visualization ---

def plot_distance_matrix(distance_matrix, layer_indices, layer_names, title, save_path=None):
    """Plots a heatmap of the neuron distance matrix."""
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
    """Plots persistence diagrams."""
    ic(f"Plotting persistence diagram: {title}")
    fig, axes = plt.subplots(1, max(2, len(result['dgms'])), figsize=(18, 5))
    dims = len(result['dgms'])

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
    """Calculates and plots Betti curves."""
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

def _get_tsne_embedding(neurons, perplexity, n_components=2, random_state=42):
    """Helper function to compute t-SNE embedding."""
    if neurons.shape[1] > 50:
      ic(f"Running PCA to reduce features for t-SNE (to 50 components).")
      pca = PCA(n_components=50, random_state=random_state)
      neurons_reduced = pca.fit_transform(neurons)
    else:
      neurons_reduced = neurons
    
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, init='pca', learning_rate='auto')
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
    return embedding # Return for reuse

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

def plot_performance_degradation(results, baseline_acc, strategy, save_path=None):
    """Plots accuracy vs. percentage of neurons removed."""
    ic(f"Plotting performance degradation for '{strategy}' strategy...")
    
    percentages = [r['percent_removed'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(percentages, accuracies, marker='', linestyle='-', label=f'{strategy} removal')
    ax.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline Acc ({baseline_acc:.2f}%)')
    
    ax.set_title('Model Performance vs. Neuron Ablation')
    ax.set_xlabel('Percentage of Neurons Removed (%)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_ylim(bottom=0, top=105)
    ax.set_xlim(left=-2, right=102)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        ic(f"Saved plot to: {save_path}")
    plt.close(fig)


