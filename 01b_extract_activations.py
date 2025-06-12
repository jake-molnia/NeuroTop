# extract_activations.py

import click
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from icecream import ic

from modules import MLPnet, collect_activations

@click.command()
@click.option('--model', type=click.Choice(['MLPnet']), default='MLPnet', help='Model architecture to load.')
@click.option('--weights-path', type=click.Path(exists=True, path_type=Path), required=True, help='Path to saved model weights (.pth).')
@click.option('--dataset-split', type=click.Choice(['train', 'test']), default='train', help='Dataset split to use.')
@click.option('--batch-size', type=int, default=512, help='Batch size for inference.')
@click.option('--save-path', type=click.Path(path_type=Path), required=True, help='Path to save the activations (.npz).')
def main(model, weights_path, dataset_split, batch_size, save_path):
    """Load a trained model and save its activations on a given dataset."""

    # Setup
    device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    ic(f"Using device: {device}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Data Loading
    ic(f"Loading CIFAR-10 {dataset_split} data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    is_train = (dataset_split == 'train')
    dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model Initialization and Weight Loading
    ic(f"Initializing model: {model}")
    if model == 'MLPnet':
        net = MLPnet().to(device)
    else:
        raise ValueError(f"Unknown model type: {model}")
    
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
