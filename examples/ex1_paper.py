"""Grokking experiment with topological analysis.

Trains a small transformer on modular addition — (x + y) mod p — and tracks
how H0 persistence (RF) scores evolve through the memorization → generalization
(grokking) transition.

Every ``ANALYSIS_INTERVAL`` epochs the model is paused, activations are
collected over the test set, and RF scores are computed. Distribution plots
and persistence diagrams are written to ``plots/``. Summary metrics are saved
to ``grokking_results.csv``.

After training, a brief statistical analysis of the memorization phase is
printed (Spearman correlation between epoch and RF mean, changes in percentiles).

Usage
-----
    python ex1_paper.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from ntop.monitoring import collect_over_loader, analyze
from ntop.plots import plot_rf_distribution, plot_persistence_scatter


# ─── Hyperparameters ──────────────────────────────────────────────────────────

MODULUS = 113          # p for (x + y) mod p
EPOCHS = 10_000
BATCH_SIZE = 256
TRAIN_SPLIT = 0.7
LR = 1e-4
WEIGHT_DECAY = 0.1
D_MODEL = 128
N_HEADS = 4
MLP_EXPANSION = 4      # MLP hidden dim = MLP_EXPANSION * D_MODEL
ANALYSIS_INTERVAL = 100
MAX_SAMPLES = 500      # activations samples for RF estimation


# ─── Model ────────────────────────────────────────────────────────────────────

class ModularArithmeticModel(nn.Module):
    """Small transformer for modular addition: predicts (x + y) mod p.

    Architecture:
        embedding → single-head self-attention → dense projection
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


# ─── Live loss plot ────────────────────────────────────────────────────────────

class LiveLossPlot:
    """Interactive matplotlib window that updates train/test loss during training."""

    def __init__(self, title: str = "Training Progress"):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle(title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_yscale("log")
        self.ax.grid(True)
        self._lines: dict = {}
        self._data: dict = defaultdict(list)
        self._steps: dict = defaultdict(list)

    def update(self, metric: str, value: float, step: int) -> None:
        """Add one data point and redraw the plot."""
        self._data[metric].append(value)
        self._steps[metric].append(step)
        if metric not in self._lines:
            (line,) = self.ax.plot([], [], label=metric)
            self._lines[metric] = line
            self.ax.legend()
        self._lines[metric].set_data(self._steps[metric], self._data[metric])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save(self, path: str = "training_progress.png") -> None:
        """Save the final figure to disk."""
        plt.ioff()
        self.fig.savefig(path)
        print(f"Training plot saved to {path}")

    def close(self) -> None:
        plt.close(self.fig)


# ─── Helpers ──────────────────────────────────────────────────────────────────

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


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_grokking_experiment() -> None:
    """Train on modular addition and track RF score evolution through grokking.

    Writes per-epoch plots to ``plots/``, summary metrics to
    ``grokking_results.csv``, and RF history to ``rf_history.npz``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModularArithmeticModel(MODULUS).to(device)

    dataset = ModularAdditionDataset(MODULUS)
    split = int(len(dataset) * TRAIN_SPLIT)
    train_set = torch.utils.data.Subset(dataset, range(split))
    test_set = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"Modulus p={MODULUS}  |  train={len(train_set)}  |  test={len(test_set)}")

    live_plot = LiveLossPlot("Grokking: Train/Test Loss")
    results = []
    rf_history: list[dict] = []
    checkpoint_epochs: list[int] = []

    for epoch in range(EPOCHS):
        # ── Training step ────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += loss.item() * len(targets)
                train_correct += (logits.argmax(1) == targets).sum().item()
                train_total += len(targets)

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total

        # ── Evaluation and topology analysis ─────────────────────────────────
        if epoch % ANALYSIS_INTERVAL == 0 or epoch == EPOCHS - 1:
            test_acc, test_loss_avg = evaluate(model, test_loader, device)

            live_plot.update("train_loss", train_loss_avg, epoch)
            live_plot.update("test_loss", test_loss_avg, epoch)

            print(f"\nEpoch {epoch:5d}: "
                  f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
                  f"train_loss={train_loss_avg:.4f}  test_loss={test_loss_avg:.4f}")

            try:
                acts = collect_over_loader(model, test_loader,
                                           max_samples=MAX_SAMPLES, verbose=False)
                rf_scores = analyze(acts)
                all_rf = np.concatenate(list(rf_scores.values()))

                plot_rf_distribution(rf_scores, epoch)
                plot_persistence_scatter(rf_scores, epoch)

                rf_history.append({k: v.copy() for k, v in rf_scores.items()})
                checkpoint_epochs.append(epoch)

                results.append({
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loss": train_loss_avg,
                    "test_loss": test_loss_avg,
                    "rf_mean": float(np.mean(all_rf)),
                    "rf_max": float(np.max(all_rf)),
                    "rf_std": float(np.std(all_rf)),
                    "rf_p99": float(np.percentile(all_rf, 99)),
                    "rf_p1": float(np.percentile(all_rf, 1)),
                })
            except Exception as exc:
                print(f"  Topology analysis failed at epoch {epoch}: {exc}")
        else:
            # Update live plot with train loss on non-analysis epochs
            live_plot.update("train_loss", train_loss_avg, epoch)

    # ── Save outputs ──────────────────────────────────────────────────────────
    live_plot.save("training_progress.png")
    live_plot.close()

    np.savez(
        "rf_history.npz",
        epochs=np.array(checkpoint_epochs),
        **{f"epoch_{e}": np.concatenate(list(s.values()))
           for e, s in zip(checkpoint_epochs, rf_history)},
    )

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("grokking_results.csv", index=False)
    print("\nSaved: grokking_results.csv, rf_history.npz, plots/")

    # ── Grokking analysis ────────────────────────────────────────────────────
    print("\n=== GROKKING ANALYSIS ===")
    if df.empty or "test_acc" not in df.columns:
        return

    grok_rows = df[df["test_acc"] > 0.95]
    if grok_rows.empty:
        print("No grokking observed (test accuracy never exceeded 0.95).")
        return

    grok_epoch = int(grok_rows.iloc[0]["epoch"])
    print(f"Grokking occurred at epoch {grok_epoch}.")

    # Memorization phase: model fits training data but hasn't generalised yet
    pre_grok = df[df["epoch"] < grok_epoch]
    memorization = pre_grok[
        (pre_grok["train_acc"] > 0.95) & (pre_grok["test_acc"] < 0.5)
    ]

    if len(memorization) > 1:
        from scipy.stats import spearmanr

        print("\n=== MEMORIZATION PHASE ===")
        rf_start = memorization["rf_mean"].iloc[0]
        rf_end = memorization["rf_mean"].iloc[-1]
        pct_change = (rf_end - rf_start) / rf_start * 100
        print(f"RF mean:  {rf_start:.4f} → {rf_end:.4f}  ({pct_change:+.1f}%)")

        p99_start = memorization["rf_p99"].iloc[0]
        p99_end = memorization["rf_p99"].iloc[-1]
        p99_change = (p99_end - p99_start) / p99_start * 100
        print(f"RF p99:   {p99_start:.4f} → {p99_end:.4f}  ({p99_change:+.1f}%)")

        corr, pval = spearmanr(memorization["epoch"], memorization["rf_mean"])
        print(f"Spearman(epoch, RF_mean) during memorization: r={corr:.3f}, p={pval:.3f}")


if __name__ == "__main__":
    run_grokking_experiment()
