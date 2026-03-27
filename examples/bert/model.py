"""Shared model, dataset, and training utilities for BERT/GLUE experiments.

Imported by both ex1_training.py and ex2_pruning.py so that no logic is
duplicated between the two scripts.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


# ─── Shared constants ─────────────────────────────────────────────────────────

GLUE_NUM_LABELS = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "stsb": 1, "rte": 2}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data utilities ───────────────────────────────────────────────────────────

def get_loaders(
    dataset_name: str,
    subset_size: int,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
) -> tuple[DataLoader, DataLoader]:
    """Load a GLUE task and return ``(train_loader, val_loader)``.

    Args:
        dataset_name: GLUE task key, e.g. ``'cola'``, ``'sst2'``.
        subset_size: Maximum number of training samples to use.
            Validation uses ``subset_size // 4`` samples.
        tokenizer: HuggingFace tokenizer compatible with the model.
        batch_size: Batch size for both loaders.
        max_length: Tokeniser maximum sequence length.

    Returns:
        ``(train_loader, val_loader)``
    """
    dataset = load_dataset("glue", dataset_name)
    text_cols = (
        ["sentence"] if dataset_name in ["cola", "sst2"]
        else ["sentence1", "sentence2"]
    )

    def tokenize(examples):
        return tokenizer(
            *[examples[c] for c in text_cols],
            truncation=True, padding=False, max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize, batched=True,
        remove_columns=[c for c in dataset["train"].column_names if c != "label"],
    )
    tokenized = tokenized.rename_column("label", "labels").with_format("torch")

    train_data = tokenized["train"].select(
        range(min(subset_size, len(tokenized["train"])))
    )
    val_key = "validation" if "validation" in tokenized else "test"
    val_data = tokenized[val_key].select(
        range(min(subset_size // 4, len(tokenized[val_key])))
    )

    collator = DataCollatorWithPadding(tokenizer)
    return (
        DataLoader(train_data, batch_size=batch_size, collate_fn=collator, shuffle=True),
        DataLoader(val_data, batch_size=batch_size, collate_fn=collator),
    )


# ─── Model ────────────────────────────────────────────────────────────────────

def load_or_train_model(
    dataset_name: str,
    model_name: str,
    subset_size: int,
    models_dir: str,
    train_epochs: int,
) -> tuple[nn.Module, object]:
    """Load a fine-tuned checkpoint from disk, or train one from scratch.

    Checkpoints are saved to ``{models_dir}/{model_name}_{dataset_name}`` after
    training so subsequent runs skip retraining.

    Args:
        dataset_name: GLUE task name.
        model_name: HuggingFace model identifier.
        subset_size: Training sample limit.
        models_dir: Root directory for model checkpoints.
        train_epochs: Number of fine-tuning epochs when training from scratch.

    Returns:
        ``(model, tokenizer)``
    """
    model_path = os.path.join(
        models_dir, f"{model_name.replace('/', '_')}_{dataset_name}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return (
            AutoModelForSequenceClassification.from_pretrained(model_path).to(device),
            tokenizer,
        )

    print(f"Training {model_name} on {dataset_name} for {train_epochs} epochs ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=GLUE_NUM_LABELS[dataset_name]
    ).to(device)
    train_loader, _ = get_loaders(dataset_name, subset_size, tokenizer)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(train_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_epochs}"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}")

    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Saved checkpoint to {model_path}")
    return model, tokenizer


# ─── Training helpers ─────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, float]:
    """Evaluate classification accuracy and mean cross-entropy loss.

    Returns:
        ``(accuracy, mean_loss)`` — accuracy in ``[0, 1]``, loss per sample.
    """
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == batch["labels"]).sum().item()
            total_loss += outputs.loss.item() * batch["labels"].size(0)
            total += batch["labels"].size(0)
    return correct / total, total_loss / total


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns:
        ``(accuracy, mean_loss)`` — train accuracy in ``[0, 1]`` and mean loss
        per sample for the epoch.
    """
    model.train()
    correct = total = 0
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predictions = outputs.logits.argmax(-1)
            total_loss += loss.item() * batch["labels"].size(0)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return correct / total, total_loss / total


def count_nonzero_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_params, nonzero_params)`` for all parameters."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero
