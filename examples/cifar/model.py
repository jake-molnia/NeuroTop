"""Shared model, dataset, and training utilities for CIFAR-10 MLP experiments.

Imported by both ex1_training.py and ex2_pruning.py so that no logic is
duplicated between the two scripts.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


# ─── Shared constants ─────────────────────────────────────────────────────────

NUM_CLASSES = 10
INPUT_DIM = 32 * 32 * 3   # flattened CIFAR-10 image
HIDDEN_DIMS = [512, 256, 128]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ─── Model ────────────────────────────────────────────────────────────────────

class CifarMLP(nn.Module):
    """Simple feed-forward MLP for CIFAR-10 classification.

    Architecture:
        flatten → Linear+ReLU (×3) → Linear (logits)

    All intermediate Linear layers are hooked by ntop for RF analysis.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] | None = None,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        hidden_dims = hidden_dims or list(HIDDEN_DIMS)
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``FloatTensor[B, 3, 32, 32]`` — CIFAR-10 images.

        Returns:
            ``FloatTensor[B, NUM_CLASSES]`` — unnormalised class logits.
        """
        return self.net(x.flatten(1))


# ─── Data utilities ───────────────────────────────────────────────────────────

def get_loaders(
    batch_size: int = 256,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for CIFAR-10.

    Args:
        batch_size: Mini-batch size for both loaders.
        data_dir: Root directory for dataset download / cache.

    Returns:
        ``(train_loader, test_loader)``
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ─── Training helpers ─────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate classification accuracy and mean cross-entropy loss.

    Returns:
        ``(accuracy, mean_loss)`` — accuracy in ``[0, 1]``, loss per sample.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            total += len(targets)
    return correct / total, total_loss / total


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns:
        ``(accuracy, mean_loss)`` — train accuracy in ``[0, 1]`` and mean loss
        per sample for the epoch.
    """
    model.train()
    correct = total = 0
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            total += len(targets)
    return correct / total, total_loss / total
