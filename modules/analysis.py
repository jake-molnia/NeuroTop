# modules/analysis.py
import torch
import numpy as np
from tqdm import tqdm
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import pandas as pd
from icecream import ic

from .training import evaluate

# --- Activation Extraction ---
def extract_activations(model, data_loader, layers_to_hook, max_samples, device, normalize_activations=True, normalization_method="standard"):
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
    
    ic("Raw activation shapes:")
    for name, acts in activations.items():
        ic(f"{name}: {acts.shape}")
        ic(f"{name} stats - min: {acts.min().item():.3f}, max: {acts.max().item():.3f}, mean: {acts.mean().item():.3f}")
    
    # Apply normalization if requested
    if normalize_activations:
        print(f"--- Normalizing activations using {normalization_method} method ---")
        ic(f"Normalization method: {normalization_method}")
        
        for name, acts in activations.items():
            ic(f"Normalizing layer {name} with shape {acts.shape}")
            if normalization_method == "standard":
                scaler = StandardScaler()
                # Transpose for scaling (neurons as features)
                scaled_acts_T = scaler.fit_transform(acts.T.cpu().numpy())
                activations[name] = torch.from_numpy(scaled_acts_T.T).float()
            elif normalization_method == "minmax":
                # Min-max normalization per neuron
                acts_min = acts.min(dim=0, keepdim=True)[0]
                acts_max = acts.max(dim=0, keepdim=True)[0]
                activations[name] = (acts - acts_min) / (acts_max - acts_min + 1e-8)
            elif normalization_method == "none":
                ic(f"Skipping normalization for {name}")
                continue
            else:
                raise ValueError(f"Unknown normalization method: {normalization_method}")
            
            ic(f"{name} normalized stats - min: {activations[name].min().item():.3f}, max: {activations[name].max().item():.3f}, mean: {activations[name].mean().item():.3f}")
    
    return activations

# --- Topological & Geometric Analysis ---
def get_distance_matrix(activations, metric='euclidean'):
    """Calculates the pairwise distance matrix between neurons."""
    # Ensure activations are on CPU for numpy operations
    activations_T = activations.T.cpu().numpy()
    ic(f"Computing distance matrix for {activations_T.shape[0]} neurons with {activations_T.shape[1]} samples")
    dist_array = pdist(activations_T, metric)
    dist_matrix = squareform(dist_array)
    ic(f"Distance matrix shape: {dist_matrix.shape}")
    ic(f"Distance matrix stats - min: {dist_matrix.min():.3f}, max: {dist_matrix.max():.3f}, mean: {dist_matrix.mean():.3f}")
    return dist_matrix

def calculate_betti_curves(dist_matrix, maxdim=1):
    """A wrapper for ripser to calculate persistence diagrams."""
    ic(f"Running Ripser with maxdim={maxdim} on {dist_matrix.shape[0]}x{dist_matrix.shape[1]} distance matrix")
    diagrams = ripser(dist_matrix, maxdim=maxdim, do_cocycles=True, distance_matrix=True)
    for dim in range(maxdim + 1):
        if dim < len(diagrams['dgms']):
            ic(f"H{dim} features found: {len(diagrams['dgms'][dim])}")
    return diagrams

# --- Ablation Strategies ---

def identify_by_random(activations, dist_matrix=None):
    """
    Generates a random ordering of neuron indices.
    'dist_matrix' is accepted for consistent function signature but not used.
    """
    num_neurons = activations.shape[1]
    ic(f"Generating random ordering for {num_neurons} neurons")
    return np.random.permutation(num_neurons)


def identify_by_knn_distance(activations, dist_matrix, k=5):
    """
    Computes the average distance to the k-nearest neighbors for each neuron
    and returns neuron indices sorted by this score.
    """
    ic(f"Computing k-NN distances with k={k}")
    # Sort each row to find the k-nearest neighbors and calculate the mean distance
    knn_distances = np.sort(dist_matrix, axis=1)[:, 1:k+1]
    mean_distances = np.mean(knn_distances, axis=1)
    ic(f"k-NN distance stats - min: {mean_distances.min():.3f}, max: {mean_distances.max():.3f}, mean: {mean_distances.mean():.3f}")
    # Return neuron indices sorted from most to least important (highest distance)
    return np.argsort(mean_distances)[::-1]


def identify_by_homology_degree(activations, dist_matrix, homology_dim=1):
    """
    Identifies neuron removal order based on degree centrality in a graph constructed
    at the characteristic radius of the most persistent H1 feature.
    This is the robust, graph-based method from the old, working script.
    """

    # Run Ripser to compute persistence diagrams up to the requested dimension.
    ic(f"Computing H{homology_dim} persistence diagrams")
    homology_result = ripser(dist_matrix, maxdim=homology_dim, distance_matrix=True)
    target_diagram = homology_result['dgms'][homology_dim]  

    # --- This is the crucial robustness check ---
    if len(target_diagram) == 0:
        print(f"\n!!! CRITICAL WARNING: No H{homology_dim} features were found in the data. !!!")
        print(f"!!! The H{homology_dim} topological structure is trivial. This strategy cannot be applied. !!!")
        n_neurons = dist_matrix.shape[0]
        # Create and return a random permutation of neuron indices
        random_order = list(range(n_neurons))
        random.shuffle(random_order)
        return random_order

    # --- Proceed if H1 features were found ---
    
    # Calculate persistence (death - birth) for all features in this dimension
    persistence = target_diagram[:, 1] - target_diagram[:, 0]
    
    # Find the feature with the highest persistence
    most_persistent_idx = np.argmax(persistence)
    birth, death = target_diagram[most_persistent_idx]
    
    # Use the average of birth and death as the characteristic radius for our graph
    characteristic_radius = (birth + death) / 2.0
    
    ic(f"Found {len(target_diagram)} H{homology_dim} features")
    ic(f"Most persistent feature: birth={birth:.3f}, death={death:.3f}, persistence={persistence[most_persistent_idx]:.3f}")
    ic(f"Characteristic radius for graph construction: {characteristic_radius:.3f}")

    # Build a graph where edges exist if distance is within the characteristic_radius
    adjacency_matrix = (dist_matrix <= characteristic_radius).astype(int)
    # A neuron is not connected to itself
    np.fill_diagonal(adjacency_matrix, 0) 

    # Calculate degree centrality (the number of connections for each neuron)
    degree_centrality = np.sum(adjacency_matrix, axis=1)

    ic(f"Graph properties: edges={np.sum(adjacency_matrix)//2}, max_degree={np.max(degree_centrality)}, mean_degree={np.mean(degree_centrality):.3f}")

    # The hypothesis is that neurons with higher degree are more central and important.
    # We sort the neuron indices in descending order by their degree.
    #removal_order = np.argsort(degree_centrality)[::-1]
    removal_order = np.argsort(degree_centrality) # inverted to show we find relevant clusters!
    return removal_order.tolist()

def identify_by_homology_persistence(activations, dist_matrix, homology_dim=1):
    """
    Ranks neurons by their centrality to all topological features, weighted by
    the persistence (lifetime) of those features. This method avoids fragile
    cocycles and uses a robust, graph-based approach.
    """
    print("--- Running robust homology-persistence strategy (weighted graph) ---")

    # This strategy is specifically designed for H1 features (loops).
    if homology_dim != 1:
        print(f"Warning: This homology-persistence strategy is designed for H1 (loops). Forcing homology_dim=1.")

    # Run Ripser to compute H1 persistence diagrams.
    ic("Computing H1 persistence diagrams for weighted strategy")
    homology_result = ripser(dist_matrix, maxdim=1, distance_matrix=True)
    h1_diagram = homology_result['dgms'][1]
    n_neurons = dist_matrix.shape[0]

    # --- Robustness Check: Handle the no-features case ---
    if len(h1_diagram) == 0:
        print("\n!!! CRITICAL WARNING: No H1 features (loops) were found in the data. !!!")
        print("!!! The topological structure is trivial. This strategy cannot be applied. !!!")
        print("!!! FALLING BACK TO A RANDOM NEURON ORDERING. !!!\n")
        # Create and return a random permutation of neuron indices
        random_order = list(range(n_neurons))
        random.shuffle(random_order)
        return random_order

    # --- Proceed if H1 features were found ---
    
    # Initialize a total importance score for each neuron
    total_neuron_importance = np.zeros(n_neurons)
    
    ic(f"Found {len(h1_diagram)} H1 features for weighted analysis")

    # Calculate persistence (lifetime) for all H1 features
    persistence = h1_diagram[:, 1] - h1_diagram[:, 0]
    ic(f"Persistence stats - min: {persistence.min():.3f}, max: {persistence.max():.3f}, mean: {persistence.mean():.3f}")
    
    # Iterate through every feature found
    for i in range(len(h1_diagram)):
        birth, death = h1_diagram[i]
        feature_persistence = persistence[i]
        
        # Skip features with zero or negative persistence
        if feature_persistence <= 0:
            ic(f"Skipping feature {i} with non-positive persistence: {feature_persistence}")
            continue
            
        # Define the characteristic radius for this specific feature
        characteristic_radius = (birth + death) / 2.0
        ic(f"Feature {i}: radius={characteristic_radius:.3f}, persistence={feature_persistence:.3f}")
        
        # Build a temporary graph for this feature
        adjacency_matrix = (dist_matrix <= characteristic_radius).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        
        # Calculate degree centrality within this feature's graph
        degree_centrality = np.sum(adjacency_matrix, axis=1)
        
        # Add to each neuron's total score, weighted by the feature's persistence
        total_neuron_importance += (degree_centrality * feature_persistence)

    ic(f"Final importance scores - min: {total_neuron_importance.min():.3f}, max: {total_neuron_importance.max():.3f}, mean: {total_neuron_importance.mean():.3f}")

    # Sort neurons by their final accumulated importance score
    removal_order = np.argsort(total_neuron_importance)[::-1]
    
    return removal_order.tolist()

def get_ablation_strategy(name):
    strategies = {
        'homology_degree': identify_by_homology_degree,
        'homology_persistence': identify_by_homology_persistence,
        'knn_distance': identify_by_knn_distance,
        'random': identify_by_random,
    }
    
    # Handle suffixed strategies like 'homology_degree_H0'
    if name.startswith('homology_degree'):
        return strategies['homology_degree']
    
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
    ic(f"Starting ablation test with {total_neurons_in_scope} neurons across {len(layer_shapes)} layers")
    
    # Ensure the original accuracy is calculated once with no masks
    _, original_acc = evaluate(model, test_loader, device, masks=None)
    results.append({'percent_removed': 0.0, 'accuracy': original_acc})
    ic(f"Baseline accuracy: {original_acc:.2f}%")

    for p in tqdm(percentages, desc=f"Ablating model globally via masking"):
        if p == 0: continue # Skip 0% as it's already recorded

        # Create fresh masks of all ones for each percentage step
        masks = {name: torch.ones(shape, device=device) for name, shape in layer_shapes.items()}
        
        num_to_remove = int(total_neurons_in_scope * (p / 100.0))
        global_indices_to_remove = global_neuron_order[:num_to_remove]
        ic(f"Removing {num_to_remove} neurons ({p}%)")
        
        # Use the map to turn off the correct neurons in the correct layers
        for global_idx in global_indices_to_remove:
            layer_name, local_neuron_idx = global_to_local_map[global_idx]
            
            # Ensure the layer is one we intend to mask and has a mask prepared
            if layer_name in masks:
                # This assumes the local_neuron_idx is a flat index for the layer's neurons
                # If layer shapes are > 2D, this might need unraveling.
                # Assuming layers are Linear (flat) or Conv (handled by flat indexing).
                mask_flat = masks[layer_name].view(-1)
                mask_flat[local_neuron_idx] = 0.0

        # Evaluate the model with the generated masks for this percentage
        _, acc = evaluate(model, test_loader, device, masks=masks)
        results.append({'percent_removed': p, 'accuracy': acc})
        ic(f"{p}% removed: accuracy = {acc:.2f}%")

    return pd.DataFrame(results)

# --- Visualization ---
def plot_persistence_diagram(diagrams, output_path):
    ic(f"Plotting persistence diagram to {output_path}")
    plot_diagrams(diagrams['dgms'], show=False)
    plt.title("Persistence Diagram")
    plt.savefig(output_path)
    plt.close()

def plot_tsne(activations, output_path):
    print("Generating t-SNE plot...")
    activations_T = activations.T.cpu().numpy()
    ic(f"t-SNE input shape: {activations_T.shape}")
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, metric='euclidean')
    tsne_results = tsne.fit_transform(activations_T)
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1])
    plt.title("t-SNE of Neurons (Global)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(output_path)
    plt.close()
    ic(f"t-SNE plot saved to {output_path}")

def plot_distance_matrix(dist_matrix, output_path):
    print("Generating distance matrix plot...")
    ic(f"Distance matrix plot shape: {dist_matrix.shape}")
    plt.figure(figsize=(10, 10))
    sns.heatmap(dist_matrix, cmap='viridis')
    plt.title("Neuron Distance Matrix (Global)")
    plt.savefig(output_path)
    plt.close()
    ic(f"Distance matrix plot saved to {output_path}")


def stratify_activations_by_context(model, data_loader, layers_to_hook, max_samples, device, normalize_activations=True, normalization_method="standard"):
    """
    Separates neuron activations based on whether the model's prediction for an input was correct or incorrect.
    Returns a dictionary of activation dictionaries, one for 'correct' and one for 'incorrect'.
    """
    model.to(device)
    model.eval()
    
    # --- Step 1: Get all predictions and labels first ---
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Getting Predictions for Stratification"):
            outputs = model(data.to(device))
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    correct_indices = (all_preds == all_labels).nonzero(as_tuple=True)[0]
    incorrect_indices = (all_preds != all_labels).nonzero(as_tuple=True)[0]

    ic(f"Found {len(correct_indices)} correct and {len(incorrect_indices)} incorrect predictions.")

    # --- Step 2: Extract activations using the indices ---
    activations = {layer: [] for layer in layers_to_hook}
    feature_layers = model.get_feature_layers()
    hooks = []

    def get_hook(name):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())
        return hook

    for layer_name in layers_to_hook:
        layer_to_hook = feature_layers[layer_name]
        hooks.append(layer_to_hook.register_forward_hook(get_hook(layer_name)))
    
    # Process the full dataset once
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Extracting Activations for Stratification"):
            model(data.to(device))
    
    for hook in hooks:
        hook.remove()
    
    # Concatenate all activations
    for name, acts in activations.items():
        activations[name] = torch.cat(acts, dim=0)

    # --- Step 3: Split activations by context ---
    stratified_activations = {'correct': {}, 'incorrect': {}}
    for name, acts in activations.items():
        stratified_activations['correct'][name] = acts[correct_indices]
        stratified_activations['incorrect'][name] = acts[incorrect_indices]

        if max_samples:
            stratified_activations['correct'][name] = stratified_activations['correct'][name][:max_samples]
            stratified_activations['incorrect'][name] = stratified_activations['incorrect'][name][:max_samples]
            
    # --- Step 4: Normalize within each context ---
    for context, acts_dict in stratified_activations.items():
        if normalize_activations:
            ic(f"Normalizing activations for context: {context}")
            for name, acts in acts_dict.items():
                if acts.shape[0] == 0: continue # Skip if no samples
                if normalization_method == "standard":
                    scaler = StandardScaler()
                    scaled_acts_T = scaler.fit_transform(acts.T.cpu().numpy())
                    stratified_activations[context][name] = torch.from_numpy(scaled_acts_T.T).float()
                # Add other normalization methods if needed
    
    ic("Stratified activation shapes:")
    ic(f"Correct: {[f'{k}: {v.shape}' for k,v in stratified_activations['correct'].items()]}")
    ic(f"Incorrect: {[f'{k}: {v.shape}' for k,v in stratified_activations['incorrect'].items()]}")
    
    return stratified_activations


def get_filtration_scales(dist_matrix, num_scales=10):
    """
    Analyzes the distance matrix to propose multiple filtration radii based on percentiles.
    """
    ic(f"Calculating {num_scales} filtration scales from distance matrix.")
    # Use only the upper triangle to avoid duplicates and diagonal zeros
    upper_triangle = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
    percentiles = np.linspace(5, 95, num_scales) # Go from 5% to 95%
    scales = np.percentile(upper_triangle, percentiles)
    ic(f"Proposed scales (radii): {[f'{s:.3f}' for s in scales]}")
    return scales


def classify_neuron_criticality(dist_matrix, scales):
    """
    Classifies neurons based on their degree centrality across multiple scales.
    Returns a DataFrame with neuron index, classification, and degree evolution.
    """
    num_neurons = dist_matrix.shape[0]
    degree_evolution = np.zeros((num_neurons, len(scales)))

    for i, scale in enumerate(tqdm(scales, desc="Analyzing scales")):
        adj_matrix = (dist_matrix <= scale).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        degree_evolution[:, i] = np.sum(adj_matrix, axis=1)
    
    # Normalize degree evolution for each neuron to be between 0 and 1
    max_degrees = np.max(degree_evolution, axis=1, keepdims=True)
    norm_degree_evolution = degree_evolution / (max_degrees + 1e-8)

    classifications = []
    for i in range(num_neurons):
        # Heuristic for classification:
        # 'Core': High degree even at small scales (e.g., >50% of its max degree in the first 30% of scales)
        # 'Redundant': Low degree throughout (e.g., never reaches 20% of the network's max degree)
        # 'Conditional': Everything else (becomes important at larger scales)
        
        is_core = np.any(norm_degree_evolution[i, :int(len(scales)*0.3)] > 0.5)
        is_redundant = np.all(degree_evolution[i, :] < (np.max(degree_evolution) * 0.2))
        
        if is_core:
            classifications.append('Core')
        elif is_redundant:
            classifications.append('Redundant')
        else:
            classifications.append('Conditional')
            
    results_df = pd.DataFrame({
        'neuron_id': range(num_neurons),
        'classification': classifications
    })
    # Add degree evolution data to the dataframe
    for i, scale in enumerate(scales):
        results_df[f'degree_at_scale_{i}'] = degree_evolution[:, i]
    
    ic(results_df['classification'].value_counts())
    return results_df


# --- [NEW] Phase 4: Hybrid Analysis & Fusion ---

def get_gradient_importance(model, data_loader, device, layer_name):
    """
    Calculates neuron importance based on the magnitude of gradients flowing back to it.
    Uses a simple L1-norm of gradients as the importance score.
    """
    model.to(device)
    model.eval()
    
    feature_layer = model.get_feature_layers()[layer_name]
    neuron_grads = None
    
    # Hook to capture gradients
    def grad_hook(grad):
        nonlocal neuron_grads
        # Sum gradients across the batch dimension
        neuron_grads = grad.abs().sum(0).cpu().numpy()
        
    # Register hook on the layer's weights
    weight_handle = feature_layer.weight.register_hook(grad_hook)
    
    ic(f"Calculating gradient importance for layer '{layer_name}'")
    
    # We only need one batch to get representative gradients
    data, labels = next(iter(data_loader))
    data, labels = data.to(device), labels.to(device)
    
    model.zero_grad()
    outputs = model(data)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    loss.backward()
    
    weight_handle.remove() # Clean up the hook
    
    # The hook populates neuron_grads. The shape might be (out_features, in_features).
    # We sum across the input features to get a single score per output neuron.
    if neuron_grads.ndim > 1:
        importance_scores = np.sum(neuron_grads, axis=1)
    else:
        importance_scores = neuron_grads
        
    ic(f"Gradient importance scores calculated for {len(importance_scores)} neurons.")
    return importance_scores


def fuse_importance_rankings(rankings_dict):
    """
    Combines multiple neuron ranking lists into a single consensus ranking using Borda Count.
    Args:
        rankings_dict: A dictionary where keys are method names and values are lists of neuron indices
                       sorted by importance (most important first).
    Returns:
        A list of neuron indices sorted by the fused importance.
    """
    ic(f"Fusing rankings from methods: {list(rankings_dict.keys())}")
    
    # Get all unique neuron IDs from all rankings
    all_neurons = set()
    for ranking in rankings_dict.values():
        all_neurons.update(ranking)
    all_neurons = list(all_neurons)
    num_neurons = len(all_neurons)

    # Borda Count: score is (num_neurons - rank). Higher score is better.
    borda_scores = {neuron: 0 for neuron in all_neurons}
    
    for method, ranking in rankings_dict.items():
        for i, neuron_id in enumerate(ranking):
            rank = i + 1
            score = num_neurons - rank
            borda_scores[neuron_id] += score
            
    # Sort neurons by their final Borda score in descending order
    fused_ranking_df = pd.DataFrame(list(borda_scores.items()), columns=['neuron_id', 'borda_score'])
    fused_ranking_df = fused_ranking_df.sort_values('borda_score', ascending=False)
    
    ic("Top 5 neurons in fused ranking:")
    ic(fused_ranking_df.head())
    
    return fused_ranking_df['neuron_id'].tolist()


# --- [NEW] Visualization for Advanced Analysis ---

def plot_criticality_distribution(criticality_df, output_path):
    """
    Plots a bar chart showing the distribution of neuron classifications.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='classification', data=criticality_df, palette='viridis', order=['Core', 'Conditional', 'Redundant'])
    plt.title("Distribution of Neuron Criticality", fontsize=16)
    plt.xlabel("Neuron Classification", fontsize=12)
    plt.ylabel("Number of Neurons", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    ic(f"Criticality distribution plot saved to {output_path}")

def plot_degree_evolution(criticality_df, scales, output_path, num_to_plot=10):
    """
    Plots the degree evolution for a sample of neurons from each class.
    """
    plt.figure(figsize=(12, 8))
    
    for classification in ['Core', 'Conditional', 'Redundant']:
        subset_df = criticality_df[criticality_df['classification'] == classification]
        if len(subset_df) == 0: continue
        
        sample_indices = subset_df.sample(n=min(num_to_plot, len(subset_df))).index
        
        degree_cols = [col for col in criticality_df.columns if 'degree_at_scale' in col]
        for idx in sample_indices:
            plt.plot(scales, criticality_df.loc[idx, degree_cols],
                     label=f"{classification} Neuron" if idx == sample_indices[0] else None, # Label only once per class
                     alpha=0.6)

    plt.title("Evolution of Neuron Degree Across Filtration Scales", fontsize=16)
    plt.xlabel("Filtration Radius (Scale)", fontsize=12)
    plt.ylabel("Neuron Degree (Number of Connections)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    ic(f"Degree evolution plot saved to {output_path}")

def plot_ranking_comparison(ranking1, ranking2, name1, name2, output_path):
    """
    Generates a scatter plot to compare two neuron importance rankings.
    """
    # Create a mapping from neuron_id to its rank for each list
    rank1_map = {neuron_id: i for i, neuron_id in enumerate(ranking1)}
    rank2_map = {neuron_id: i for i, neuron_id in enumerate(ranking2)}
    
    all_neurons = list(set(ranking1) | set(ranking2))
    
    plot_data = []
    for neuron_id in all_neurons:
        plot_data.append({
            'neuron_id': neuron_id,
            'rank1': rank1_map.get(neuron_id, len(all_neurons)),
            'rank2': rank2_map.get(neuron_id, len(all_neurons))
        })
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='rank1', y='rank2', alpha=0.7)
    # Add a y=x line for reference
    plt.plot([0, len(all_neurons)], [0, len(all_neurons)], color='r', linestyle='--', label='Perfect Agreement')
    
    plt.title(f"Comparison of Neuron Rankings: {name1} vs. {name2}", fontsize=16)
    plt.xlabel(f"Rank by {name1} Importance", fontsize=12)
    plt.ylabel(f"Rank by {name2} Importance", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    ic(f"Ranking comparison plot saved to {output_path}")
