"""BERT pruning experiment using RF-based neuron importance.

Loads (or trains from scratch) a BERT model on a GLUE classification task,
computes H0 persistence (RF) scores across all Linear layers, and evaluates
structured pruning at 30 %, 50 %, and 70 % sparsity.  Each sparsity level is
compared against a random-selection baseline.

Pruning strategy
----------------
All neurons across all Linear layers are ranked by their RF score.  The
``--pruning-method rf`` variant prunes the *lowest*-RF neurons first (they
are topologically least important); ``--pruning-method random`` selects
uniformly at random for comparison.  After pruning, the model is fine-tuned
for ``--finetune-epochs`` epochs to recover accuracy.

Usage
-----
    python 01_bert.py \\
        --dataset cola \\
        --model-name bert-base-uncased \\
        --pruning-method rf \\
        --results-csv results/bert_rf.csv
"""

import copy
import csv
import os

import click
import numpy as np
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

from ntop.monitoring import collect_over_loader, analyze


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of output labels per GLUE task
GLUE_NUM_LABELS = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "stsb": 1, "rte": 2}


# ─── Data ─────────────────────────────────────────────────────────────────────

def prepare_dataloaders(
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

    print(f"Training {model_name} on {dataset_name} for {train_epochs} epochs …")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=GLUE_NUM_LABELS[dataset_name]
    ).to(device)
    train_loader, _ = prepare_dataloaders(dataset_name, subset_size, tokenizer)
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


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Return classification accuracy (%) on a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(**batch).logits.argmax(-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return 100 * correct / total


def count_nonzero_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_params, nonzero_params)`` for all parameters."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


# ─── Pruning ──────────────────────────────────────────────────────────────────

def select_neurons_to_prune(
    rf_scores: dict[str, np.ndarray],
    target_pct: float,
    method: str,
) -> dict[str, list[int]]:
    """Select neurons to prune to reach a target global sparsity.

    Neurons are selected globally across all layers (not per-layer), so the
    pruning budget is distributed according to RF importance.

    Args:
        rf_scores: ``layer_name -> np.ndarray[N_neurons]`` of RF importance.
        target_pct: Percentage of total neurons to prune (0–100).
        method: ``'rf'`` — prune lowest-RF neurons first;
                ``'random'`` — uniform random selection.

    Returns:
        ``layer_name -> list[neuron_index]`` mapping of neurons to zero out.
    """
    all_neurons = [
        (layer_name, idx, float(score))
        for layer_name, scores in rf_scores.items()
        for idx, score in enumerate(scores)
    ]
    total = len(all_neurons)
    target_count = int(total * target_pct / 100)
    print(f"  Total neurons: {total:,}  |  Pruning ({target_pct}%): {target_count:,}")

    if method == "rf":
        all_neurons.sort(key=lambda t: t[2])   # lowest RF first
    elif method == "random":
        np.random.shuffle(all_neurons)
    else:
        raise ValueError(f"Unknown pruning method: {method!r}. Use 'rf' or 'random'.")

    pruned: dict[str, list[int]] = {}
    for layer_name, idx, _ in all_neurons[:target_count]:
        pruned.setdefault(layer_name, []).append(idx)
    return pruned


def zero_neurons(model: nn.Module, neurons_by_layer: dict[str, list[int]]) -> None:
    """Zero the weights (and biases) of specific output neurons in Linear layers.

    This is a hard, irreversible pruning step: the nominated neurons are
    permanently silenced by setting their weight rows to zero.

    Args:
        model: Model to prune in-place.
        neurons_by_layer: ``layer_name -> list[neuron_index]``.
    """
    modules = dict(model.named_modules())
    with torch.no_grad():
        for layer_name, indices in neurons_by_layer.items():
            mod = modules.get(layer_name)
            if not isinstance(mod, nn.Linear) or not indices:
                continue
            idx = torch.tensor(indices, dtype=torch.long)
            mod.weight.data[idx] = 0.0
            if mod.bias is not None:
                mod.bias.data[idx] = 0.0


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--dataset", required=True,
              type=click.Choice(list(GLUE_NUM_LABELS)),
              help="GLUE task name.")
@click.option("--model-name", required=True,
              type=click.Choice(["bert-base-uncased", "bert-large-uncased"]),
              help="HuggingFace model identifier.")
@click.option("--pruning-method", required=True,
              type=click.Choice(["random", "rf"]),
              help="Neuron selection strategy.")
@click.option("--results-csv", required=True,
              help="Path to append results CSV rows.")
@click.option("--models-dir", default="./trained_models",
              show_default=True)
@click.option("--train-epochs", default=50, show_default=True,
              help="Fine-tuning epochs when training from scratch.")
@click.option("--finetune-epochs", default=10, show_default=True,
              help="Recovery fine-tuning epochs after pruning.")
@click.option("--subset-size", default=5000, show_default=True,
              help="Maximum training samples.")
@click.option("--max-samples", default=512, show_default=True,
              help="Activation samples for RF computation.")
def run_experiment(
    dataset, model_name, pruning_method, results_csv, models_dir,
    train_epochs, finetune_epochs, subset_size, max_samples,
):
    """Prune BERT at 30 %, 50 %, and 70 % sparsity and record results.

    For each sparsity level: evaluates pre-pruning → applies neuron zeroing
    → fine-tunes → records (baseline, post-prune, post-finetune, Δ,
    compression ratio) to the CSV.
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  Dataset: {dataset}  |  Method: {pruning_method}")
    print(f"{'='*60}\n")

    model, tokenizer = load_or_train_model(
        dataset, model_name, subset_size, models_dir, train_epochs
    )
    train_loader, val_loader = prepare_dataloaders(dataset, subset_size, tokenizer)

    baseline_acc = evaluate(model, val_loader)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    # Compute RF scores once on the unmodified model
    print("Computing RF scores …")
    acts = collect_over_loader(model, val_loader, max_samples=max_samples)
    rf_scores = analyze(acts)

    original_total, _ = count_nonzero_parameters(model)

    for prune_pct in [30, 50, 70]:
        print(f"\n{'='*60}")
        print(f"Pruning {prune_pct} %")
        print(f"{'='*60}")

        # Work on a fresh copy so each sparsity level starts from the baseline
        pruned_model = copy.deepcopy(model)
        neurons_to_prune = select_neurons_to_prune(rf_scores, prune_pct, pruning_method)
        zero_neurons(pruned_model, neurons_to_prune)

        pre_ft_acc = evaluate(pruned_model, val_loader)
        print(f"After pruning (before fine-tune): {pre_ft_acc:.2f}%"
              f"  (Δ {baseline_acc - pre_ft_acc:.2f}%)")

        # Fine-tune to recover accuracy lost from weight zeroing
        print(f"Fine-tuning for {finetune_epochs} epoch(s) …")
        ft_optimizer = optim.AdamW(pruned_model.parameters(), lr=2e-5)
        for epoch in range(finetune_epochs):
            pruned_model.train()
            for batch in tqdm(train_loader,
                              desc=f"  FT epoch {epoch + 1}/{finetune_epochs}"):
                ft_optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                pruned_model(**batch).loss.backward()
                ft_optimizer.step()

        final_acc = evaluate(pruned_model, val_loader)
        _, pruned_nonzero = count_nonzero_parameters(pruned_model)
        params_retained_pct = pruned_nonzero / original_total * 100
        params_removed = original_total - pruned_nonzero

        print(f"After fine-tune:  {final_acc:.2f}%  (Δ {baseline_acc - final_acc:.2f}%)")
        print(f"Parameters: {pruned_nonzero:,} / {original_total:,}"
              f"  ({params_retained_pct:.1f}% retained)")

        # Append to results CSV
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
        file_exists = os.path.isfile(results_csv) and os.path.getsize(results_csv) > 0
        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Model", "Method", "Prune%", "Baseline",
                    "AfterPrune", "AfterFT", "Delta",
                    "ParamsRetained%", "ParamsRemoved",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "Model": model_name.split("-")[1].title(),
                "Method": pruning_method,
                "Prune%": prune_pct,
                "Baseline": f"{baseline_acc:.2f}",
                "AfterPrune": f"{pre_ft_acc:.2f}",
                "AfterFT": f"{final_acc:.2f}",
                "Delta": f"{baseline_acc - final_acc:.2f}",
                "ParamsRetained%": f"{params_retained_pct:.1f}%",
                "ParamsRemoved": f"{params_removed:,}",
            })

    print(f"\nResults saved to {results_csv}")


if __name__ == "__main__":
    run_experiment()
