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
    MLPnet, ResNetForCifar, VGGForCifar,
    evaluate_model_performance,
    compute_neuron_distances,
    run_persistent_homology,
    identify_by_homology_degree,
    plot_performance_degradation
)

def get_model(model_name, num_classes):
    """Helper function to instantiate the correct model."""
    if 'resnet' in model_name:
        return ResNetForCifar(resnet_type=model_name, num_classes=num_classes)
    elif 'vgg' in model_name:
        return VGGForCifar(vgg_type=model_name, num_classes=num_classes)
    elif model_name == 'MLPnet':
        return MLPnet(num_class=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def create_masks_from_indices(indices_to_remove, layer_indices, layer_names, layer_shapes, model):
    """
    Converts a flat list of neuron indices to a dictionary of layer-specific masks.
    """
    masks = {}
    device = model.get_device()
    
    for name, shape in layer_shapes.items():
        is_mlp_layer = isinstance(shape, int)
        mask_shape = (shape,) if is_mlp_layer else (1, shape, 1, 1)
        masks[name] = torch.ones(mask_shape, device=device)

    for global_idx in indices_to_remove:
        for i, (start, end) in enumerate(zip(layer_indices, layer_indices[1:])):
            if start <= global_idx < end:
                layer_name = layer_names[i]
                local_idx = global_idx - start
                
                if masks[layer_name].dim() == 4:
                    masks[layer_name][0, local_idx, :, :] = 0
                else:
                    masks[layer_name][local_idx] = 0
                break
    return masks

@click.command()
@click.option('--model-name', type=str, required=True)
@click.option('--dataset-name', type=str, required=True)
@click.option('--weights-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--activations-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--output-dir', type=click.Path(path_type=Path), default='outputs/ablation')
@click.option('--cache-dir', type=click.Path(path_type=Path), default='outputs/analysis_cache')
@click.option('--removal-strategy', type=str, default='homology-degree')
@click.option('--distance-metric', type=str, default='euclidean')
@click.option('--removal-steps', type=int, default=100)
def main(**kwargs):
    """Performs neuron ablation study."""
    strategy, model_name, dataset_name = kwargs['removal_strategy'], kwargs['model_name'], kwargs['dataset_name']
    ic.configureOutput(prefix=f'{model_name}/{dataset_name}/{strategy} | ')
    
    output_dir, cache_dir = kwargs['output_dir'], kwargs['cache_dir']
    output_dir.mkdir(parents=True, exist_ok=True); cache_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ic(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=512, shuffle=False)
    
    model = get_model(model_name, num_classes=10).to(device)
    model.load_state_dict(torch.load(kwargs['weights_path'], map_location=device))
    
    ic("Evaluating baseline performance...")
    baseline_performance = evaluate_model_performance(model, testloader, device)
    ic(baseline_performance)

    activations_data = np.load(kwargs['activations_path'])
    activations_dict = {key: activations_data[key] for key in activations_data.files}
    
    # --- KEY FIX: Get neuron counts from the model definition, not the activation file ---
    layer_shapes = model.layer_shapes
    layer_names = list(layer_shapes.keys())
    total_neurons = sum(layer_shapes.values())
    
    layer_indices = [0]
    cumulative = 0
    for name in layer_names:
        cumulative += layer_shapes[name]
        layer_indices.append(cumulative)
    # -------------------------------------------------------------------------------------

    ic(f"Total neurons for ablation: {total_neurons}")
    ic(f"Preparing neuron removal order via '{strategy}'...")
    
    dist_mat, _, _, _ = compute_neuron_distances(activations_dict, device=device, metric=kwargs['distance_metric'], max_samples=5000)
    homology_res = run_persistent_homology(dist_mat)
    neurons_to_remove_ordered = identify_by_homology_degree(dist_mat, homology_res)

    # Ensure removal order list is the same size as the total number of neurons
    if len(neurons_to_remove_ordered) < total_neurons:
        remaining = list(set(range(total_neurons)) - set(neurons_to_remove_ordered))
        random.shuffle(remaining)
        neurons_to_remove_ordered.extend(remaining)

    ic(f"Starting ablation study with {kwargs['removal_steps']} steps...")
    ablation_results = [{'percent_removed': 0, **baseline_performance}]
    
    removal_percentages = np.linspace(0, 100, kwargs['removal_steps'] + 1)

    for step, percent in enumerate(removal_percentages[1:], 1):
        num_to_remove = int(total_neurons * (percent / 100))
        indices = neurons_to_remove_ordered[:num_to_remove]
        
        ic(f"Step {step}/{kwargs['removal_steps']}: Removing {len(indices)} neurons ({percent:.1f}%)")
        
        masks = create_masks_from_indices(indices, layer_indices, layer_names, layer_shapes, model)
        performance = evaluate_model_performance(model, testloader, device, masks=masks)
        ic(performance)
        
        ablation_results.append({'percent_removed': percent, **performance})

    # Save results
    plot_filename_base = f'perf_degrad_{model_name}_{dataset_name}_{strategy}'
    plot_path = output_dir / f'{plot_filename_base}.png'
    plot_title = f'{model_name.upper()} on {dataset_name.upper()}\n({strategy} removal)'
    if 'plot_performance_degradation' in globals():
        plot_performance_degradation(ablation_results, baseline_performance['accuracy'], plot_title, save_path=plot_path)

    results_data_path = output_dir / f'{plot_filename_base}.pkl'
    with open(results_data_path, 'wb') as f: pickle.dump(ablation_results, f)
    ic("Ablation study complete.")

if __name__ == '__main__':
    main()

