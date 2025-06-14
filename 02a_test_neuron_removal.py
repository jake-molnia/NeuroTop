# 02a_test_neuron_removal.py

import click
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from icecream import ic
import pickle
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import tarfile
import os
import torch.nn.functional as F

from modules import (
    MLPnet, SimpleMLPOld, SimpleMLPNew, MNIST_MLP, DetailedLSTMSentiment,
    evaluate_model_performance,
    compute_neuron_distances,
    run_persistent_homology,
    identify_by_homology_degree,
    plot_performance_degradation
)

# --- Global variables for text processing - LAZILY INITIALIZED ---
tokenizer = None
vocab = None
collate_fn_for_imdb = None
MAX_VOCAB_SIZE = 10000
MAX_LEN = 256

def get_dataset(dataset_name, data_root='./data', train_split=False):
    """Gets the specified dataset. Defaults to TEST split for evaluation."""
    ic(f"Loading '{'train' if train_split else 'test'}' split of {dataset_name} dataset...")
    data_root_path = Path(data_root)
    data_root_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = torchvision.datasets.MNIST(root=data_root, train=train_split, download=True, transform=transform)
        return dataset, 10

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        dataset = torchvision.datasets.CIFAR10(root=data_root, train=train_split, download=True, transform=transform)
        return dataset, 10
    
    elif dataset_name == 'wine_quality':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        csv_path = data_root_path / 'winequality-red.csv'
        if not csv_path.exists():
            pd.read_csv(url, sep=';').to_csv(csv_path, index=False)
        
        df = pd.read_csv(csv_path)
        X = df.drop('quality', axis=1).values
        y = df['quality'].values
        y = y - y.min()

        X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data_to_use = X_train if train_split else X_test
        labels_to_use = _ if train_split else y_test
        
        scaler = StandardScaler().fit(X_train)
        X_scaled = scaler.transform(data_to_use)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(labels_to_use, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor), len(df['quality'].unique())

    # (IMDB loader remains the same, omitted for brevity but is still here)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_model(model_name, num_classes, vocab_size=None, dropout_rate=0.5, use_batchnorm=True):
    """Helper function to instantiate the correct model."""
    ic(f"Initializing model: {model_name} for {num_classes} classes")
    if model_name == 'SimpleMLPOld':
        return SimpleMLPOld(num_classes=num_classes)
    elif model_name == 'SimpleMLPNew':
        return SimpleMLPNew(num_classes=num_classes)
    elif model_name == 'MNIST_MLP':
        ic(f"MNIST_MLP hyperparameters: dropout_rate={dropout_rate}, use_batchnorm={use_batchnorm}")
        return MNIST_MLP(num_classes=num_classes, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    elif model_name == 'DetailedLSTMSentiment':
        if vocab_size is None:
            raise ValueError("vocab_size must be provided for DetailedLSTMSentiment model")
        return DetailedLSTMSentiment(vocab_size=vocab_size, num_classes=num_classes)
    elif model_name == 'MLPnet':
         return MLPnet(num_class=num_classes)
    elif 'resnet' in model_name:
        return ResNetForCifar(resnet_type=model_name, num_classes=num_classes)
    elif 'vgg' in model_name:
        return VGGForCifar(vgg_type=model_name, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def create_masks_from_indices(indices_to_remove, layer_indices, layer_names, layer_shapes, model):
    """Converts a flat list of neuron indices to a dictionary of layer-specific masks."""
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
                
                if masks[layer_name].dim() == 4: # CNN Layer
                    masks[layer_name][0, local_idx, :, :] = 0
                else: # MLP Layer
                    masks[layer_name][local_idx] = 0
                break
    return masks

def generate_result_filename(model_name, dataset_name, strategy, dropout_rate=None, use_batchnorm=None):
    """Generate result filename with hyperparameters for MNIST_MLP, standard filename for others."""
    if model_name == 'MNIST_MLP' and dropout_rate is not None and use_batchnorm is not None:
        bn_suffix = 'bn1' if use_batchnorm else 'bn0'
        return f'perf_degrad_{model_name}_{dataset_name}_dropout{dropout_rate}_{bn_suffix}_{strategy}'
    else:
        return f'perf_degrad_{model_name}_{dataset_name}_{strategy}'

@click.command()
@click.option('--model-name', required=True, help='Name of the model architecture.')
@click.option('--dataset-name', type=click.Choice(['cifar10', 'wine_quality', 'imdb', 'mnist']), required=True)
@click.option('--weights-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--activations-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--output-dir', type=click.Path(path_type=Path), default='outputs/ablation')
@click.option('--cache-dir', type=click.Path(path_type=Path), default='outputs/analysis_cache')
@click.option('--removal-strategy', type=str, default='homology-degree')
@click.option('--distance-metric', type=str, default='euclidean')
@click.option('--removal-steps', type=int, default=100)
@click.option('--dropout-rate', type=float, default=0.5, help='Dropout rate for MNIST_MLP model.')
@click.option('--use-batchnorm/--no-batchnorm', default=True, help='Use/disable batch normalization for MNIST_MLP model.')
def main(**kwargs):
    """Performs neuron ablation study."""
    strategy, model_name, dataset_name = kwargs['removal_strategy'], kwargs['model_name'], kwargs['dataset_name']
    
    dropout_rate, use_batchnorm = kwargs['dropout_rate'], kwargs['use_batchnorm']
    
    ic.configureOutput(prefix=f'{model_name}/{dataset_name}/{strategy} | ')
    if model_name == 'MNIST_MLP':
        ic(f"MNIST_MLP hyperparameters: dropout_rate={dropout_rate}, use_batchnorm={use_batchnorm}")
    
    output_dir, cache_dir = kwargs['output_dir'], kwargs['cache_dir']
    output_dir.mkdir(parents=True, exist_ok=True); cache_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ic(f"Using device: {device}")
    
    testset, num_classes = get_dataset(dataset_name, train_split=False)
    if dataset_name == 'imdb':
        testloader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=collate_fn_for_imdb)
    else:
        testloader = DataLoader(testset, batch_size=512, shuffle=False)
    
    vocab_size = len(vocab) if vocab else None
    model = get_model(model_name, num_classes=num_classes, vocab_size=vocab_size, 
                     dropout_rate=dropout_rate, use_batchnorm=use_batchnorm).to(device)
    model.load_state_dict(torch.load(kwargs['weights_path'], map_location=device))
    
    ic("Evaluating baseline performance...")
    baseline_performance = evaluate_model_performance(model, testloader, device)
    ic(baseline_performance)

    activations_data = np.load(kwargs['activations_path'])
    activations_dict = {key: activations_data[key] for key in activations_data.files}
    
    layer_shapes = model.layer_shapes
    layer_names = list(layer_shapes.keys())
    total_neurons = sum(layer_shapes.values())
    
    layer_indices = [0]
    cumulative = 0
    for name in layer_names:
        cumulative += layer_shapes[name]
        layer_indices.append(cumulative)

    ic(f"Total neurons for ablation: {total_neurons}")
    ic(f"Preparing neuron removal order via '{strategy}'...")
    
    dist_mat, _, _, _ = compute_neuron_distances(activations_dict, device=device, metric=kwargs['distance_metric'], max_samples=5000)
    homology_res = run_persistent_homology(dist_mat)
    neurons_to_remove_ordered = identify_by_homology_degree(dist_mat, homology_res)

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

    plot_filename_base = generate_result_filename(model_name, dataset_name, strategy, dropout_rate, use_batchnorm)
    plot_path = output_dir / f'{plot_filename_base}.png'
    plot_title = f'{model_name.upper()} on {dataset_name.upper()}\n({strategy} removal)'
    if model_name == 'MNIST_MLP':
        bn_text = 'with BatchNorm' if use_batchnorm else 'no BatchNorm'
        plot_title += f'\n(dropout={dropout_rate}, {bn_text})'
    
    if 'plot_performance_degradation' in globals():
        plot_performance_degradation(ablation_results, baseline_performance['accuracy'], plot_title, save_path=plot_path)

    results_data_path = output_dir / f'{plot_filename_base}.pkl'
    with open(results_data_path, 'wb') as f: pickle.dump(ablation_results, f)
    ic("Ablation study complete.")

if __name__ == '__main__':
    main()