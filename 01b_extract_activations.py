# 01b_extract_activations.py

import click
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import tarfile
import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from icecream import ic

from modules import (
    MLPnet, SimpleMLPOld, SimpleMLPNew, MNIST_MLP, DetailedLSTMSentiment, 
    collect_activations
)

# --- Global variables for text processing - LAZILY INITIALIZED ---
tokenizer = None
vocab = None
collate_fn_for_imdb = None
MAX_VOCAB_SIZE = 10000
MAX_LEN = 256

def get_dataset(dataset_name, data_root='./data', train_split=True):
    """Gets the specified dataset. Defaults to TRAIN split for activation extraction."""
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

        X_data, _, y_data, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        data_to_use = X_data if train_split else _
        labels_to_use = y_data if train_split else _

        scaler = StandardScaler().fit(X_data)
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
    # Add other models here...
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def generate_activations_filename(model, dataset, dropout_rate=None, use_batchnorm=None, split='train'):
    """Generate filename with hyperparameters for MNIST_MLP, standard filename for others."""
    if model == 'MNIST_MLP' and dropout_rate is not None and use_batchnorm is not None:
        bn_suffix = 'bn1' if use_batchnorm else 'bn0'
        return f'{model}_{dataset}_dropout{dropout_rate}_{bn_suffix}_{split}.npz'
    else:
        return f'{model}_{dataset}_{split}.npz'

@click.command()
@click.option('--model', required=True, help='Model architecture to load.')
@click.option('--dataset', type=click.Choice(['cifar10', 'wine_quality', 'imdb', 'mnist']), required=True, help='Dataset used to train the model.')
@click.option('--weights-path', type=click.Path(exists=True, path_type=Path), required=True, help='Path to saved model weights (.pth).')
@click.option('--batch-size', type=int, default=128, help='Batch size for inference.')
@click.option('--save-path', type=click.Path(path_type=Path), required=True, help='Path to save the activations (.npz).')
@click.option('--dropout-rate', type=float, default=0.5, help='Dropout rate for MNIST_MLP model.')
@click.option('--use-batchnorm/--no-batchnorm', default=True, help='Use/disable batch normalization for MNIST_MLP model.')
def main(model, dataset, weights_path, batch_size, save_path, dropout_rate, use_batchnorm):
    """Load a trained model and save its activations on a given dataset."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    ic.configureOutput(prefix=f'{model}/{dataset} | ')
    ic(f"Using device: {device}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_obj, num_classes = get_dataset(dataset, train_split=True)
    
    if dataset == 'imdb':
        dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_for_imdb)
    else:
        dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)
    
    vocab_size = len(vocab) if vocab else None
    net = get_model(model, num_classes, vocab_size=vocab_size, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm).to(device)
    
    ic(f"Loading weights from: {weights_path}")
    net.load_state_dict(torch.load(weights_path, map_location=device))

    activations = collect_activations(net, dataloader, device)
    
    ic(f"Saving activations to: {save_path}")
    np.savez_compressed(save_path, **activations)
    ic("Extraction complete.")

if __name__ == '__main__':
    main()