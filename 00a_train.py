# 00a_train.py

import click
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import tarfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from icecream import ic

from modules import (
    MLPnet, SimpleMLPOld, SimpleMLPNew, MNIST_MLP, DetailedLSTMSentiment, 
    label_smooth
)

# --- Global variables for text processing - LAZILY INITIALIZED ---
tokenizer = None
vocab = None
collate_fn_for_imdb = None
MAX_VOCAB_SIZE = 10000
MAX_LEN = 256 

def get_dataset(dataset_name, data_root='./data'):
    """Gets the specified dataset, now handling images, tabular, and text."""
    ic(f"Loading {dataset_name} dataset...")
    data_root_path = Path(data_root)
    data_root_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform), 10
    
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        return torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform), 10
    
    elif dataset_name == 'wine_quality':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        csv_path = data_root_path / 'winequality-red.csv'
        if not csv_path.exists():
            pd.read_csv(url, sep=';').to_csv(csv_path, index=False)
        df = pd.read_csv(csv_path)
        X = df.drop('quality', axis=1).values
        y = df['quality'].values
        y = y - y.min() 
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor), len(df['quality'].unique())

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def train_model(model, trainloader, device, epochs, lr):
    ic(f"Starting training for {epochs} epochs on {device}...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss, correct_preds, total_samples = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(outputs, tuple): # Handles original MLPnet
                main_out, _ = outputs
                loss = criterion(outputs[0], labels) + 0.3 * criterion(outputs[1], labels)
            else:
                main_out = outputs
                loss = criterion(main_out, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(main_out.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct_preds / total_samples
        ic(f'Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

def generate_model_filename(dataset, model, dropout_rate=None, use_batchnorm=None):
    """Generate filename with hyperparameters for MNIST_MLP, standard filename for others."""
    if model == 'MNIST_MLP' and dropout_rate is not None and use_batchnorm is not None:
        bn_suffix = 'bn1' if use_batchnorm else 'bn0'
        return f'{dataset}_{model}_dropout{dropout_rate}_{bn_suffix}.pth'
    else:
        return f'{dataset}_{model}.pth'

@click.command()
@click.option('--model', required=True, help='Model architecture to train.')
@click.option('--dataset', type=click.Choice(['cifar10', 'wine_quality', 'imdb', 'mnist']), required=True, help='Dataset to use.')
@click.option('--epochs', type=int, default=15, help='Number of training epochs.')
@click.option('--lr', type=float, default=1e-3, help='Learning rate.')
@click.option('--batch-size', type=int, default=64, help='Training batch size.')
@click.option('--save-dir', type=click.Path(path_type=Path), default='outputs/weights', help='Directory to save weights.')
@click.option('--dropout-rate', type=float, default=0.5, help='Dropout rate for MNIST_MLP model.')
@click.option('--use-batchnorm/--no-batchnorm', default=True, help='Use/disable batch normalization for MNIST_MLP model.')
def main(model, dataset, epochs, lr, batch_size, save_dir, dropout_rate, use_batchnorm):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ic(f"Using device: {device}")
    
    # Generate appropriate filename
    filename = generate_model_filename(dataset, model, dropout_rate, use_batchnorm)
    save_path = save_dir / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_data, num_classes = get_dataset(dataset)
    
    if dataset == 'imdb':
        # Assumes collate_fn is set up during get_dataset
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_imdb)
    else:
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    ic(f"Initializing model: {model} for {num_classes} classes")
    if model == 'MNIST_MLP':
        ic(f"MNIST_MLP hyperparameters: dropout_rate={dropout_rate}, use_batchnorm={use_batchnorm}")
    
    net = None
    if model == 'SimpleMLPOld':
        net = SimpleMLPOld(num_classes=num_classes).to(device)
    elif model == 'SimpleMLPNew':
        net = SimpleMLPNew(num_classes=num_classes).to(device)
    elif model == 'MNIST_MLP':
        net = MNIST_MLP(num_classes=num_classes, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm).to(device)
    # Add other models here...
    else:
        raise ValueError(f"Unknown model type in training script: {model}")

    train_model(net, trainloader, device, epochs, lr)
    
    ic(f"Saving model weights to: {save_path}")
    torch.save(net.state_dict(), save_path)
    ic("Training complete.")

if __name__ == '__main__':
    main()