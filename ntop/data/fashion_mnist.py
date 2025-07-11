import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional
from ..core.models import simple_mlp

def get_fashion_mnist_loaders(batch_size: int = 64, train_size: Optional[int] = None,
                             test_size: Optional[int] = None, data_dir: str = './data') -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    
    if train_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:train_size]
        train_dataset = Subset(train_dataset, train_indices)
    
    if test_size is not None:
        test_indices = torch.randperm(len(test_dataset))[:test_size]
        test_dataset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_simple_fashion_mnist_model(hidden_dims: list = [256, 128], dropout: float = 0.2) -> nn.Module:
    return simple_mlp(input_dim=784, hidden_dims=hidden_dims, output_dim=10, dropout=dropout)

def get_quick_fashion_mnist(batch_size: int = 64, n_samples: int = 1000) -> Tuple[DataLoader, DataLoader]:
    train_loader, test_loader = get_fashion_mnist_loaders(
        batch_size=batch_size,
        train_size=n_samples,
        test_size=n_samples // 5
    )
    return train_loader, test_loader

class FashionMNISTConfig:
    def __init__(self, hidden_dims: list = [256, 128], batch_size: int = 64,
                 train_size: Optional[int] = None, test_size: Optional[int] = None,
                 lr: float = 1e-3, epochs: int = 20, dropout: float = 0.2):
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
    
    def create_model(self) -> nn.Module:
        return create_simple_fashion_mnist_model(self.hidden_dims, self.dropout)
    
    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return get_fashion_mnist_loaders(self.batch_size, self.train_size, self.test_size)

# Fashion-MNIST class names for visualization
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def get_class_name(class_idx: int) -> str:
    return FASHION_MNIST_CLASSES[class_idx] if 0 <= class_idx < len(FASHION_MNIST_CLASSES) else f"Class {class_idx}"