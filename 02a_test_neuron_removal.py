# 02a_test_neuron_removal.py

import click
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from icecream import ic
import pickle
import random

from modules import (
    MLPnet,
    evaluate_model_performance,
    plot_performance_degradation,
    identify_neuron_clusters,
    _get_tsne_embedding,
    compute_neuron_distances,
    identify_by_distance,
    run_persistent_homology,
    identify_by_knn_distance,
    identify_by_homology_degree
)

"""
uv run 02a_test_neuron_removal.py \
    --weights-path outputs/weights/cifar10_mlp.pth \
    --activations-path outputs/activations/cifar10_mlp_train.npz \
    --removal-strategy homology-degree \
    --removal-steps 4120 \
    --output-dir outputs/ablation/chomo \
    --distance-metric "euclidean"
"""

def create_masks_from_indices(indices_to_remove, layer_indices, layer_names, layer_shapes):
    """
    Converts a flat list of neuron indices to a dictionary of layer-specific masks.
    """
    masks = {name: torch.ones(shape) for name, shape in layer_shapes.items()}
    
    for global_idx in indices_to_remove:
        # Find which layer this neuron belongs to
        for i, (start, end) in enumerate(zip(layer_indices, layer_indices[1:])):
            if start <= global_idx < end:
                layer_name = layer_names[i]
                local_idx = global_idx - start
                masks[layer_name][local_idx] = 0
                break
    return masks

@click.command()
# --- Input/Output Options ---
@click.option('--weights-path', type=click.Path(exists=True, path_type=Path), required=True, help='Path to saved model weights (.pth).')
@click.option('--activations-path', type=click.Path(exists=True, path_type=Path), required=True, help='Path to saved activations file (.npz) from 01b.')
@click.option('--output-dir', type=click.Path(path_type=Path), default='outputs/ablation', help='Directory to save analysis plots.')
@click.option('--cache-dir', type=click.Path(path_type=Path), default='outputs/analysis_cache', help='Directory to cache expensive computations.')

# --- Ablation Hyperparameters ---
@click.option('--removal-strategy', type=click.Choice([
    'random', 'cluster', 'distance', 'knn-core', 'knn-outlier', 'homology-degree'
]), default='knn-core', help='Strategy for choosing which neurons to remove.')
@click.option('--distance-metric', type=click.Choice(['euclidean', 'cosine']), default='euclidean', help='Metric for neuron distance calculation.')
@click.option('--k-neighbors', type=int, default=10, help='Value of k for k-NN distance strategy.')
@click.option('--removal-steps', type=int, default=50, help='Number of steps in the ablation experiment.')

# --- Hyperparameters for specific strategies ---
@click.option('--cluster-eps', type=float, default=0.5, help='DBSCAN epsilon for clustering (cluster strategy).')
@click.option('--perplexity', type=int, default=30, help='Perplexity for t-SNE (cluster strategy).')

def main(**kwargs):
    """
    Perform an ablation study by progressively removing neurons and measuring
    the impact on model performance. Caches expensive computations.
    """
    strategy = kwargs['removal_strategy']
    ic.configureOutput(prefix=f'{strategy} | ')
    
    # --- Setup ---
    output_dir = kwargs['output_dir']
    cache_dir = kwargs['cache_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    ic(f"Using device: {device}")
    
    # --- Data Loading ---
    ic("Loading CIFAR-10 test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- Load Model ---
    ic(f"Loading model weights from: {kwargs['weights_path']}")
    model = MLPnet().to(device)
    model.load_state_dict(torch.load(kwargs['weights_path'], map_location=device), strict=False)

    # --- Baseline Performance ---
    ic("Evaluating baseline performance (no neurons removed)...")
    baseline_performance = evaluate_model_performance(model, testloader, device)
    ic(baseline_performance)

    # --- Load Activations ---
    ic(f"Loading activations from: {kwargs['activations_path']}")
    activations_data = np.load(kwargs['activations_path'])
    activations_dict = {key: activations_data[key] for key in activations_data.files}
    layer_names = list(activations_dict.keys())
    layer_shapes = model.layer_shapes
    layer_indices = [0]
    cumulative_neurons = 0
    for name in layer_names:
        cumulative_neurons += layer_shapes[name]
        layer_indices.append(cumulative_neurons)

    # --- Identify Neurons to Remove ---
    ic(f"Preparing neuron removal order based on '{strategy}' strategy...")
    
    distance_matrix = None
    if strategy in ['distance', 'knn-core', 'knn-outlier', 'homology-degree']:
        dist_mat_path = cache_dir / f'distance_matrix_{kwargs["distance_metric"]}.npy'
        if dist_mat_path.exists():
            ic(f"Loading cached distance matrix from: {dist_mat_path}")
            distance_matrix = np.load(dist_mat_path)
        else:
            ic("Computing and caching distance matrix...")
            distance_matrix, _, _, _ = compute_neuron_distances(
                activations_dict, device=device, metric=kwargs['distance_metric']
            )
            np.save(dist_mat_path, distance_matrix)
            ic(f"Saved distance matrix to: {dist_mat_path}")

    if strategy == 'cluster':
        all_neurons_activation = np.hstack(list(activations_dict.values()))
        neurons_np = all_neurons_activation.T
        embedding = _get_tsne_embedding(neurons_np, perplexity=kwargs['perplexity'])
        clusters = identify_neuron_clusters(embedding, eps=kwargs['cluster_eps'])
        neurons_to_remove_ordered = [idx for cluster in clusters for idx in cluster]
        remaining_neurons = list(set(range(neurons_np.shape[0])) - set(neurons_to_remove_ordered))
        random.shuffle(remaining_neurons)
        neurons_to_remove_ordered.extend(remaining_neurons)

    elif strategy == 'distance':
        neurons_to_remove_ordered = identify_by_distance(distance_matrix)

    elif strategy in ['knn-core', 'knn-outlier']:
        neurons_to_remove_ordered = identify_by_knn_distance(
            distance_matrix, k=kwargs['k_neighbors'], strategy=strategy.split('-')[1]
        )
    elif strategy == 'homology-degree':
        homology_result = None
        homology_path = cache_dir / f'homology_result_{kwargs["distance_metric"]}.pkl'
        if homology_path.exists():
            ic(f"Loading cached homology result from: {homology_path}")
            with open(homology_path, 'rb') as f:
                homology_result = pickle.load(f)
        else:
            ic("Computing and caching homology result...")
            homology_result = run_persistent_homology(distance_matrix, maxdim=1)
            with open(homology_path, 'wb') as f:
                pickle.dump(homology_result, f)
            ic(f"Saved homology result to: {homology_path}")
        
        neurons_to_remove_ordered = identify_by_homology_degree(distance_matrix, homology_result)
    
    else: # random
        total_neurons = sum(layer_shapes.values())
        neurons_to_remove_ordered = list(range(total_neurons))
        random.shuffle(neurons_to_remove_ordered)

    # --- Run Ablation Study ---
    ic(f"Starting ablation study with {kwargs['removal_steps']} steps...")
    ablation_results = [{'percent_removed': 0, **baseline_performance}]

    for step in range(1, kwargs['removal_steps'] + 1):
        percent_to_remove = (step / kwargs['removal_steps']) * 100
        # Ensure we don't go out of bounds if the list is shorter
        num_to_remove = min(int(len(neurons_to_remove_ordered) * (percent_to_remove / 100)), len(neurons_to_remove_ordered))
        current_indices_to_remove = neurons_to_remove_ordered[:num_to_remove]
        
        ic(f"Step {step}/{kwargs['removal_steps']}: Removing {len(current_indices_to_remove)} neurons ({percent_to_remove:.1f}%)")

        masks = create_masks_from_indices(current_indices_to_remove, layer_indices, layer_names, layer_shapes)
        model.update_masks({name: mask.to(device) for name, mask in masks.items()})
        
        performance = evaluate_model_performance(model, testloader, device)
        ic(performance)
        
        result_log = {'percent_removed': percent_to_remove, **performance}
        ablation_results.append(result_log)

    # --- Plot and Save Results ---
    plot_filename = f'performance_degradation_{strategy}'
    if strategy in ['distance', 'knn-core', 'knn-outlier', 'homology-degree']:
        plot_filename += f'_{kwargs["distance_metric"]}'
    if strategy in ['knn-core', 'knn-outlier']:
        plot_filename += f'_k{kwargs["k_neighbors"]}'
    
    plot_path = output_dir / f'{plot_filename}.png'
    plot_performance_degradation(
        ablation_results, 
        baseline_performance['accuracy'], 
        strategy, 
        save_path=plot_path
    )
    ic(f"Saved final plot to: {plot_path}")

    ic("Ablation study complete.")

if __name__ == '__main__':
    main()
