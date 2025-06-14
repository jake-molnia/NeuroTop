# 01b_extract_activations.py

import click
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from icecream import ic

# --- Import all models and the activation collector ---
from modules import MLPnet, ResNetForCifar, VGGForCifar, collect_activations

def get_dataset(dataset_name, data_root='./data', train_split=True):
    """Gets the specified dataset from torchvision for a given split."""
    ic(f"Loading '{'train' if train_split else 'test'}' split of {dataset_name} dataset...")
    
    # Normalization constants for different datasets
    DATASET_STATS = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        'fashion_mnist': ((0.2860,), (0.3530,))
    }
    
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    mean, std = DATASET_STATS[dataset_name]
    
    # Fashion-MNIST is grayscale, needs a different transform pipeline
    if dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Grayscale(num_output_channels=3), # Convert to 3 channels for CNNs
        ])
        dataset_class = torchvision.datasets.FashionMNIST
        return dataset_class(root=data_root, train=train_split, download=True, transform=transform)

    # Transforms for color datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if dataset_name == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
    elif dataset_name == 'svhn':
        split = 'train' if train_split else 'test'
        return torchvision.datasets.SVHN(root=data_root, split=split, download=True, transform=transform)
        
    return dataset_class(root=data_root, train=train_split, download=True, transform=transform)


def get_model(model_name, num_classes):
    """Helper function to instantiate the correct model."""
    ic(f"Initializing model: {model_name} for {num_classes} classes")
    if 'resnet' in model_name:
        return ResNetForCifar(resnet_type=model_name, num_classes=num_classes)
    elif 'vgg' in model_name:
        return VGGForCifar(vgg_type=model_name, num_classes=num_classes)
    elif model_name == 'MLPnet':
        return MLPnet(num_class=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


@click.command()
@click.option('--model', type=click.Choice(['MLPnet', 'resnet18', 'resnet34', 'vgg11_bn', 'vgg16_bn']), required=True, help='Model architecture to load.')
@click.option('--dataset', type=click.Choice(['cifar10', 'cifar100', 'svhn', 'fashion_mnist']), required=True, help='Dataset used to train the model.')
@click.option('--weights-path', type=click.Path(exists=True, path_type=Path), required=True, help='Path to saved model weights (.pth).')
@click.option('--batch-size', type=int, default=512, help='Batch size for inference.')
@click.option('--save-path', type=click.Path(path_type=Path), required=True, help='Path to save the activations (.npz).')
def main(model, dataset, weights_path, batch_size, save_path):
    """Load a trained model and save its activations on a given dataset."""

    # Setup
    device = torch.device('mps')
    ic.configureOutput(prefix=f'{model}/{dataset} | ')
    ic(f"Using device: {device}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Data Loading (extracting from the training split by default)
    dataset_obj = get_dataset(dataset, train_split=True)
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    num_classes = len(dataset_obj.classes) if hasattr(dataset_obj, 'classes') else 10

    # Model Initialization and Weight Loading
    net = get_model(model, num_classes).to(device)
    
    ic(f"Loading weights from: {weights_path}")
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # Activation Collection
    activations = collect_activations(net, dataloader, device)
    
    # Save Activations
    ic(f"Saving activations to: {save_path}")
    np.savez_compressed(save_path, **activations)
    ic("Extraction complete.")
    return

if __name__ == '__main__':
    main()
