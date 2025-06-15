# modules/data.py
import torch
from torchvision import datasets, transforms
from icecream import ic

def get_dataloaders(config):
    """
    Creates and returns the training and testing dataloaders.
    """
    dataset_name = config['dataset']['name']
    path = config['dataset']['path']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

    ic(f"Loading dataset: {dataset_name}")
    ic(f"Dataset config: batch_size={batch_size}, num_workers={num_workers}, path={path}")

    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        
        ic(f"MNIST dataset sizes: train={len(train_dataset)}, test={len(test_dataset)}")
        
    # Add other datasets like CIFAR10 here if needed
    # elif dataset_name == 'CIFAR10':
    #     ...
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"Loaded {dataset_name} dataset. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    ic(f"Batch sizes: train_loader={len(train_loader)}, test_loader={len(test_loader)}")
    
    return train_loader, test_loader