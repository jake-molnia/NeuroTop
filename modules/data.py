"""
Dataset loading and preprocessing utilities.
Clean, focused data handling with proper logging.
"""

import torch
from torchvision import datasets, transforms
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def get_dataloaders(config: Dict[str, Any]) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and testing dataloaders from configuration."""
    
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    path = dataset_config['path']
    batch_size = dataset_config['batch_size']
    num_workers = dataset_config.get('num_workers', 4)
    
    logger.info(f"Loading {dataset_name} dataset from {path}")
    
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
        
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created dataloaders: train={len(train_loader)} batches, test={len(test_loader)} batches")
    logger.info(f"Dataset sizes: train={len(train_dataset)}, test={len(test_dataset)}")
    
    return train_loader, test_loader

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get dataset metadata."""
    
    info = {
        'MNIST': {
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'input_size': 784
        },
        'FashionMNIST': {
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'input_size': 784
        },
        'CIFAR10': {
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'input_size': 3072
        }
    }
    
    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return info[dataset_name]