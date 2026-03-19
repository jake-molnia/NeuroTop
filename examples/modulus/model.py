"""Shared model, dataset, and training utilities for modular arithmetic experiments.

Imported by both ex1_grokking.py and ex2_pruning.py so that no logic is
duplicated between the two scripts.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─── Shared constants ─────────────────────────────────────────────────────────

MODULUS = 113
D_MODEL = 128
N_HEADS = 4
MLP_EXPANSION = 4   # MLP hidden dim = MLP_EXPANSION * D_MODEL


# ─── Model ────────────────────────────────────────────────────────────────────

class ModularArithmeticModel(nn.Module):
    """Small transformer for modular addition: predicts (x + y) mod p.

    Architecture:
        embedding → multi-head self-attention → dense projection
        → ReLU MLP → unembedding

    Token IDs are 0 … modulus-1 for digits and ``modulus`` for the separator.
    Input sequences have the form ``[a, b, separator]``; the last token's
    representation is used as the sequence output.
    """

    def __init__(self, modulus: int, d_model: int = D_MODEL):
        super().__init__()
        self.modulus = modulus
        self.embed = nn.Embedding(modulus + 1, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads=N_HEADS,
                                               batch_first=True)
        self.attention_proj = nn.Linear(d_model, d_model)
        self.mlp_up = nn.Linear(d_model, MLP_EXPANSION * d_model)
        self.mlp_down = nn.Linear(MLP_EXPANSION * d_model, d_model)
        self.unembed = nn.Linear(d_model, modulus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``LongTensor[B, 3]`` — token sequences ``[a, b, separator]``.

        Returns:
            ``FloatTensor[B, modulus]`` — unnormalised class logits.
        """
        h = self.embed(x)                                 # [B, 3, d_model]
        attn_out, _ = self.attention(h, h, h)
        h = self.attention_proj(attn_out)[:, -1, :]       # last token: [B, d_model]
        h = self.mlp_down(torch.relu(self.mlp_up(h)))
        return self.unembed(h)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ModularAdditionDataset(Dataset):
    """All ``(x, y)`` pairs for modular addition: label = ``(x + y) mod p``."""

    def __init__(self, modulus: int):
        pairs = [
            (torch.tensor([x, y, modulus]),
             torch.tensor((x + y) % modulus))
            for x in range(modulus)
            for y in range(modulus)
        ]
        np.random.shuffle(pairs)
        self.data = pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─── Data utilities ───────────────────────────────────────────────────────────

def get_loaders(
    modulus: int,
    train_split: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for modular addition.

    Args:
        modulus: p for (x + y) mod p.
        train_split: Fraction of pairs used for training.
        batch_size: Mini-batch size for both loaders.

    Returns:
        ``(train_loader, test_loader)``
    """
    dataset = ModularAdditionDataset(modulus)
    split = int(len(dataset) * train_split)
    train_set = torch.utils.data.Subset(dataset, range(split))
    test_set = torch.utils.data.Subset(dataset, range(split, len(dataset)))
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
