# 01c_analyze_topology.py
# Usage: python 01c_analyze_topology.py --activations-path ... [OPTIONS]

import click
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from icecream import ic
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modules import (
    MLPnet, SimpleMLPOld, SimpleMLPNew, MNIST_MLP, DetailedLSTMSentiment,
    compute_neuron_distances, 
    run_persistent_homology,
    plot_distance_matrix,
    plot_persistence_diagram,
    plot_betti_curves,
    plot_neuron_embedding_2d,
    plot_neuron_embedding_3d
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

@click.command()
# --- Input/Output Options ---
@click.option('--activations-path', type=click.Path(exists=True, path_type=Path), required=True, help='Path to saved activations file (.npz).')
@click.option('--output-dir', type=click.Path(path_type=Path), default='outputs/plots', help='Directory to save analysis plots.')
@click.option('--model-name', type=str, default='MLPnet', help='Name of the model architecture.')
@click.option('--dataset-name', type=click.Choice(['cifar10', 'wine_quality', 'imdb', 'mnist']), required=True, help='Dataset name to determine model parameters.')
@click.option('--weights-path', type=click.Path(exists=True, path_type=Path), default=None, help='Path to model weights (optional, for model instantiation verification).')

# --- Model-specific Options ---
@click.option('--dropout-rate', type=float, default=0.5, help='Dropout rate for MNIST_MLP model.')
@click.option('--use-batchnorm/--no-batchnorm', default=True, help='Use/disable batch normalization for MNIST_MLP model.')

# --- Analysis Hyperparameters ---
@click.option('--distance-metric', type=click.Choice(['euclidean', 'cosine']), default='euclidean', help='Metric for neuron distance calculation.')
@click.option('--max-samples', type=int, default=5000, help='Max samples to use for distance calculation.')
@click.option('--reduce-dims', type=int, default=None, help='Reduce neuron dimensions to this number using PCA before distance computation.')
@click.option('--maxdim', type=int, default=2, help='Maximum homology dimension to compute.')
@click.option('--thresh', type=float, default=25.0, help='Filtration threshold for Ripser.')
@click.option('--perplexity', type=int, default=30, help='Perplexity for t-SNE algorithm.')

# --- Plotting Control ---
@click.option('--plot-distance-matrix', is_flag=True, help='Flag to plot the neuron distance matrix.')
@click.option('--plot-betti-curves', is_flag=True, help='Flag to plot the Betti curves.')
@click.option('--plot-embedding-3d', is_flag=True, help='Flag to plot the 3D t-SNE embedding.')

# --- Wandb Logging ---
@click.option('--wandb-project', default='neural-topology', help='Wandb project name.')
@click.option('--wandb-name', default=None, help='Wandb run name.')
@click.option('--log-to-wandb', is_flag=True, help='Flag to enable logging plots to Wandb.')

def main(**kwargs):
    """Perform topological analysis on saved activations with fine-grained control."""
    output_dir = kwargs['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = kwargs['model_name']
    dataset_name = kwargs['dataset_name']
    dropout_rate = kwargs['dropout_rate']
    use_batchnorm = kwargs['use_batchnorm']
    
    ic.configureOutput(prefix=f'{model_name}/{dataset_name} Analysis | ')
    if model_name == 'MNIST_MLP':
        ic(f"MNIST_MLP hyperparameters: dropout_rate={dropout_rate}, use_batchnorm={use_batchnorm}")
    
    # --- Device Setup ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ic(f"Using device: {device}")
    
    ic(f"Analyzing activations from: {kwargs['activations_path']}")
    ic(f"Saving plots to: {output_dir}")
    ic(f"Analysis Parameters: { {k: v for k, v in kwargs.items() if k not in ['output_dir', 'activations_path']} }")

    # Get dataset info to determine num_classes and vocab_size
    try:
        _, num_classes = get_dataset(dataset_name, train_split=False)
        vocab_size = len(vocab) if vocab else None
        
        # Instantiate model for layer shape information (optional verification)
        if kwargs['weights_path']:
            model = get_model(model_name, num_classes=num_classes, vocab_size=vocab_size, 
                             dropout_rate=dropout_rate, use_batchnorm=use_batchnorm).to(device)
            model.load_state_dict(torch.load(kwargs['weights_path'], map_location=device))
            ic(f"Model loaded successfully with {sum(model.layer_shapes.values())} total neurons")
        else:
            ic("No weights path provided, proceeding with activations analysis only")
    except Exception as e:
        ic(f"Could not load dataset/model (proceeding anyway): {e}")

    if kwargs['log_to_wandb']:
        wandb_config = kwargs.copy()
        wandb.init(project=kwargs['wandb_project'], name=kwargs['wandb_name'], config=wandb_config, job_type='analysis')
        ic("Logging to Wandb is enabled.")

    # Load Activations
    activations_data = np.load(kwargs['activations_path'])
    activations_dict = {key: activations_data[key] for key in activations_data.files}
    ic(f"Loaded activation layers: {list(activations_dict.keys())}")

    # Perform Core Analysis
    distance_matrix, neurons, layer_labels, layer_indices = compute_neuron_distances(
        activations_dict, 
        device=device,
        metric=kwargs['distance_metric'], 
        max_samples=kwargs['max_samples'],
        reduce_dims=kwargs['reduce_dims']
    )
    topology_result = run_persistent_homology(
        distance_matrix, 
        maxdim=kwargs['maxdim'], 
        thresh=kwargs['thresh']
    )

    # --- Generate filename base for consistent naming ---
    def generate_plot_filename_base():
        if model_name == 'MNIST_MLP':
            bn_suffix = 'bn1' if use_batchnorm else 'bn0'
            return f'{model_name}_{dataset_name}_dropout{dropout_rate}_{bn_suffix}'
        else:
            return f'{model_name}_{dataset_name}'

    filename_base = generate_plot_filename_base()

    # --- Generate, Save, and Log Plots ---
    plots_to_log = {}

    # 1. Persistence Diagram (default plot)
    diag_path = output_dir / f'persistence_diagram_{filename_base}_{kwargs["distance_metric"]}.png'
    plot_title = f"Persistence Diagram ({model_name}, {dataset_name.upper()}, {kwargs['distance_metric']})"
    if model_name == 'MNIST_MLP':
        bn_text = 'with BatchNorm' if use_batchnorm else 'no BatchNorm'
        plot_title += f'\n(dropout={dropout_rate}, {bn_text})'
    
    plot_persistence_diagram(
        topology_result,
        title=plot_title,
        save_path=diag_path
    )
    plots_to_log["persistence_diagram"] = wandb.Image(str(diag_path))

    # 2. 2D t-SNE Embedding (default plot)
    embed_2d_path = output_dir / f'neuron_embedding_2d_{filename_base}_perp{kwargs["perplexity"]}.png'
    embedding_title = f'{model_name.upper()} on {dataset_name.upper()}'
    if model_name == 'MNIST_MLP':
        bn_text = 'with BatchNorm' if use_batchnorm else 'no BatchNorm'
        embedding_title += f'\n(dropout={dropout_rate}, {bn_text})'
    
    plot_neuron_embedding_2d(
        neurons,
        layer_labels=layer_labels,
        model_name=embedding_title,
        perplexity=kwargs['perplexity'],
        save_path=embed_2d_path
    )
    plots_to_log["neuron_embedding_2d"] = wandb.Image(str(embed_2d_path))
    
    # 3. Distance Matrix
    if kwargs['plot_distance_matrix']:
        dist_mat_path = output_dir / f'distance_matrix_{filename_base}_{kwargs["distance_metric"]}.png'
        dist_title = f"Neuron Distance Matrix ({model_name}, {dataset_name.upper()}, {kwargs['distance_metric']})"
        if model_name == 'MNIST_MLP':
            bn_text = 'with BatchNorm' if use_batchnorm else 'no BatchNorm'
            dist_title += f'\n(dropout={dropout_rate}, {bn_text})'
        
        plot_distance_matrix(
            distance_matrix,
            layer_indices=layer_indices,
            layer_names=list(activations_dict.keys()),
            title=dist_title,
            save_path=dist_mat_path
        )
        plots_to_log["distance_matrix"] = wandb.Image(str(dist_mat_path))
    
    # 4. Betti Curves
    if kwargs['plot_betti_curves']:
        betti_path = output_dir / f'betti_curves_{filename_base}_{kwargs["distance_metric"]}.png'
        betti_title = f"Betti Curves ({model_name}, {dataset_name.upper()}, {kwargs['distance_metric']})"
        if model_name == 'MNIST_MLP':
            bn_text = 'with BatchNorm' if use_batchnorm else 'no BatchNorm'
            betti_title += f'\n(dropout={dropout_rate}, {bn_text})'
        
        plot_betti_curves(
            topology_result,
            thresh=kwargs['thresh'],
            title=betti_title,
            save_path=betti_path
        )
        plots_to_log["betti_curves"] = wandb.Image(str(betti_path))
        
    # 5. 3D t-SNE Embedding
    if kwargs['plot_embedding_3d']:
        embed_3d_path = output_dir / f'neuron_embedding_3d_{filename_base}_perp{kwargs["perplexity"]}.png'
        plot_neuron_embedding_3d(
            neurons,
            layer_labels=layer_labels,
            model_name=embedding_title,
            perplexity=kwargs['perplexity'],
            save_path=embed_3d_path
        )
        plots_to_log["neuron_embedding_3d"] = wandb.Image(str(embed_3d_path))

    # Log all generated plots to Wandb if enabled
    if kwargs['log_to_wandb']:
        ic(f"Logging {len(plots_to_log)} plots to Wandb...")
        wandb.log(plots_to_log)
        wandb.finish()

    ic("Analysis complete.")
    return

if __name__ == '__main__':
    main()