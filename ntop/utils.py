"""Shared utilities for all ntop experiment scripts.

Provides checkpoint save/load and a comprehensive suite of publication-ready
plot functions used by examples/modulus/ex1_grokking.py and ex2_pruning.py.

All plot functions:
  - accept data and an ``out_dir`` string
  - save one PNG to ``out_dir`` and return nothing
  - create ``out_dir`` if it does not exist
  - never call ``plt.show()`` or ``plt.ion()``
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import torch

plt.style.use("seaborn-v0_8-whitegrid")


# ─── Checkpoint utilities ─────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, path: str) -> None:
    """Save model weights, optimizer state, and epoch to a ``.pt`` file."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path: str) -> int:
    """Load checkpoint from ``path``.

    Returns:
        The saved epoch number, or 0 if the file does not exist.
    """
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _grok_epoch(results: list[dict]) -> int | None:
    """Return the first epoch where test_acc > 0.95, or None."""
    for r in results:
        if r.get("test_acc", 0.0) > 0.95:
            return r["epoch"]
    return None


def _rf_global(rf_snapshot: dict) -> np.ndarray:
    """Concatenate all neuron RF scores from a snapshot into a 1-D array."""
    return np.concatenate(list(rf_snapshot.values()))


def _epoch_colors(n: int) -> list:
    """Return n colors from viridis: early=dark (0), late=light (1)."""
    cmap = cm.viridis
    if n == 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


def _subsample_indices(n: int, max_n: int = 10) -> list[int]:
    """Return up to max_n evenly spaced indices into a sequence of length n."""
    if n <= max_n:
        return list(range(n))
    return [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]


def _nearest_idx(checkpoint_epochs: list[int], target: int) -> int:
    """Return the index of the checkpoint epoch closest to target."""
    return min(range(len(checkpoint_epochs)),
               key=lambda i: abs(checkpoint_epochs[i] - target))


# ─── Grokking plots ───────────────────────────────────────────────────────────

def plot_loss_curves(results: list[dict], out_dir: str) -> None:
    """Train and test loss over epochs (log scale).

    Draws a vertical dashed line at the grokking epoch if test_acc > 0.95
    is first observed in ``results``.

    Output: ``loss_curves.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = [r["epoch"] for r in results]
    train_loss = [r["train_loss"] for r in results]
    test_loss = [r["test_loss"] for r in results]
    grok = _grok_epoch(results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train loss", linewidth=1.5)
    ax.plot(epochs, test_loss, label="Test loss", linewidth=1.5)
    if grok is not None:
        ax.axvline(grok, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Grokking (epoch {grok})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Train / Test Loss")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_curves(results: list[dict], out_dir: str) -> None:
    """Train and test accuracy over epochs (linear scale).

    Draws a vertical dashed line at the grokking epoch.

    Output: ``accuracy_curves.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = [r["epoch"] for r in results]
    train_acc = [r["train_acc"] for r in results]
    test_acc = [r["test_acc"] for r in results]
    grok = _grok_epoch(results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, label="Train accuracy", linewidth=1.5)
    ax.plot(epochs, test_acc, label="Test accuracy", linewidth=1.5)
    if grok is not None:
        ax.axvline(grok, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Grokking (epoch {grok})")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train / Test Accuracy")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curves.png"), dpi=150)
    plt.close(fig)


def plot_generalization_gap(results: list[dict], out_dir: str) -> None:
    """Generalization gap (test_loss − train_loss) over epochs.

    The gap closes sharply at the grokking transition.

    Output: ``generalization_gap.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = [r["epoch"] for r in results]
    gap = [r["test_loss"] - r["train_loss"] for r in results]
    grok = _grok_epoch(results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, gap, color="steelblue", linewidth=1.5)
    if grok is not None:
        ax.axvline(grok, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Grokking (epoch {grok})")
        ax.legend(frameon=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test loss − Train loss")
    ax.set_title("Generalization Gap")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "generalization_gap.png"), dpi=150)
    plt.close(fig)


def plot_rf_kde(
    rf_history: list[dict],
    checkpoint_epochs: list[int],
    out_dir: str,
    grokking_epoch: int | None = None,
) -> None:
    """Global RF score KDE curves, subsampled to ~10 checkpoints, log density.

    - Subsamples to at most 10 evenly spaced checkpoints to avoid overplotting.
    - X-axis clipped to the p99 of the final checkpoint's RF values.
    - Y-axis log scale so low-density tails remain visible.
    - If ``grokking_epoch`` is given, the nearest checkpoint is highlighted in
      crimson at full opacity with a thicker line.
    - Early checkpoints are dark (viridis), late are light.

    Output: ``rf_kde_global.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    indices = _subsample_indices(len(checkpoint_epochs))
    sub_history = [rf_history[i] for i in indices]
    sub_epochs  = [checkpoint_epochs[i] for i in indices]
    colors      = _epoch_colors(len(sub_epochs))

    # x-axis upper bound: p99 of final checkpoint
    final_vals = _rf_global(rf_history[-1])
    final_vals = final_vals[np.isfinite(final_vals)]
    x_max = float(np.percentile(final_vals, 99)) if len(final_vals) else 1.0
    xs = np.linspace(0, x_max, 400)

    grok_idx = (_nearest_idx(sub_epochs, grokking_epoch)
                if grokking_epoch is not None else None)

    fig, ax = plt.subplots(figsize=(7, 7))
    for i, (snap, epoch) in enumerate(zip(sub_history, sub_epochs)):
        vals = _rf_global(snap)
        vals = vals[np.isfinite(vals) & (vals <= x_max * 1.05)]
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        density = np.maximum(kde(xs), 1e-10)   # floor for log scale

        is_grok = (i == grok_idx)
        ax.plot(xs, density,
                color="crimson" if is_grok else colors[i],
                linewidth=2.2 if is_grok else 1.1,
                alpha=1.0 if is_grok else 0.75,
                zorder=10 if is_grok else 2,
                label=f"epoch {epoch} (grokking)" if is_grok else None)

    ax.set_yscale("log")
    ax.set_xlim(0, x_max)
    ax.set_xlabel("RF score", fontsize=13)
    ax.set_ylabel("Density (log)", fontsize=13)
    ax.set_title("RF Score Distribution — all layers", fontsize=14)
    ax.tick_params(labelsize=11)

    sm = plt.cm.ScalarMappable(
        cmap=cm.viridis,
        norm=plt.Normalize(vmin=sub_epochs[0], vmax=sub_epochs[-1]),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Epoch")

    if grok_idx is not None:
        ax.legend(frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_kde_global.png"), dpi=150)
    plt.close(fig)


def plot_rf_kde_per_layer(
    rf_history: list[dict],
    checkpoint_epochs: list[int],
    out_dir: str,
    grokking_epoch: int | None = None,
) -> None:
    """Per-layer RF KDE curves, subsampled to ~10 checkpoints, log density.

    Same rendering rules as ``plot_rf_kde`` applied per layer.

    Output: ``rf_kde_{layer_name}.png`` per layer.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not rf_history:
        return
    layer_names = list(rf_history[0].keys())

    indices     = _subsample_indices(len(checkpoint_epochs))
    sub_history = [rf_history[i] for i in indices]
    sub_epochs  = [checkpoint_epochs[i] for i in indices]
    colors      = _epoch_colors(len(sub_epochs))
    grok_idx    = (_nearest_idx(sub_epochs, grokking_epoch)
                   if grokking_epoch is not None else None)

    for layer in layer_names:
        # x-axis bound: p99 of this layer's final checkpoint
        final_layer_vals = rf_history[-1].get(layer, np.array([]))
        final_layer_vals = final_layer_vals[np.isfinite(final_layer_vals)]
        x_max = (float(np.percentile(final_layer_vals, 99))
                 if len(final_layer_vals) else 1.0)
        xs = np.linspace(0, x_max, 400)

        fig, ax = plt.subplots(figsize=(7, 7))
        for i, (snap, epoch) in enumerate(zip(sub_history, sub_epochs)):
            vals = snap.get(layer, np.array([]))
            vals = vals[np.isfinite(vals) & (vals <= x_max * 1.05)]
            if len(vals) < 2:
                continue
            kde = gaussian_kde(vals)
            density = np.maximum(kde(xs), 1e-10)

            is_grok = (i == grok_idx)
            ax.plot(xs, density,
                    color="crimson" if is_grok else colors[i],
                    linewidth=2.2 if is_grok else 1.1,
                    alpha=1.0 if is_grok else 0.75,
                    zorder=10 if is_grok else 2,
                    label=f"epoch {epoch} (grokking)" if is_grok else None)

        ax.set_yscale("log")
        ax.set_xlim(0, x_max)
        ax.set_xlabel("RF score", fontsize=13)
        ax.set_ylabel("Density (log)", fontsize=13)
        ax.set_title(f"RF Score Distribution — {layer}", fontsize=14)
        ax.tick_params(labelsize=11)

        sm = plt.cm.ScalarMappable(
            cmap=cm.viridis,
            norm=plt.Normalize(vmin=sub_epochs[0], vmax=sub_epochs[-1]),
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Epoch")

        if grok_idx is not None:
            ax.legend(frameon=True, fontsize=10)

        fig.tight_layout()
        safe_name = layer.replace(".", "_").replace("/", "_")
        fig.savefig(os.path.join(out_dir, f"rf_kde_{safe_name}.png"), dpi=150)
        plt.close(fig)


def plot_rf_percentile_evolution(
    rf_history: list[dict],
    checkpoint_epochs: list[int],
    out_dir: str,
    grokking_epoch: int | None = None,
) -> None:
    """Global RF percentile bands (p5/p25/p50/p75/p95) over checkpoints.

    Output: ``rf_percentiles_global.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    pcts = {5: [], 25: [], 50: [], 75: [], 95: []}
    for snap in rf_history:
        vals = _rf_global(snap)
        for p in pcts:
            pcts[p].append(float(np.percentile(vals, p)))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {5: "p5", 25: "p25", 50: "p50 (median)", 75: "p75", 95: "p95"}
    styles = {5: ":", 25: "--", 50: "-", 75: "--", 95: ":"}
    palette = {5: "#2166ac", 25: "#74add1", 50: "#313695",
               75: "#f46d43", 95: "#d73027"}
    for p, vals in pcts.items():
        ax.plot(checkpoint_epochs, vals, linestyle=styles[p],
                color=palette[p], linewidth=1.5, label=labels[p])

    if grokking_epoch is not None:
        ax.axvline(grokking_epoch, color="crimson", linestyle="--",
                   linewidth=1.2, label=f"Grokking (epoch {grokking_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RF score")
    ax.set_title("RF Percentile Evolution (all layers)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_percentiles_global.png"), dpi=150)
    plt.close(fig)


def plot_rf_percentile_evolution_per_layer(
    rf_history: list[dict],
    checkpoint_epochs: list[int],
    layer_names: list[str],
    out_dir: str,
    grokking_epoch: int | None = None,
) -> None:
    """Per-layer RF percentile evolution, one PNG per layer.

    Output: ``rf_percentiles_{layer_name}.png`` per layer.
    """
    os.makedirs(out_dir, exist_ok=True)
    styles = {5: ":", 25: "--", 50: "-", 75: "--", 95: ":"}
    palette = {5: "#2166ac", 25: "#74add1", 50: "#313695",
               75: "#f46d43", 95: "#d73027"}
    labels = {5: "p5", 25: "p25", 50: "p50", 75: "p75", 95: "p95"}

    for layer in layer_names:
        pcts = {5: [], 25: [], 50: [], 75: [], 95: []}
        for snap in rf_history:
            vals = snap.get(layer, np.array([]))
            for p in pcts:
                pcts[p].append(float(np.percentile(vals, p)) if len(vals) > 0 else 0.0)

        fig, ax = plt.subplots(figsize=(8, 5))
        for p, vals in pcts.items():
            ax.plot(checkpoint_epochs, vals, linestyle=styles[p],
                    color=palette[p], linewidth=1.5, label=labels[p])
        if grokking_epoch is not None:
            ax.axvline(grokking_epoch, color="crimson", linestyle="--",
                       linewidth=1.2, label=f"Grokking (epoch {grokking_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RF score")
        ax.set_title(f"RF Percentile Evolution — {layer}")
        ax.legend(frameon=True)
        fig.tight_layout()
        safe_name = layer.replace(".", "_").replace("/", "_")
        fig.savefig(os.path.join(out_dir, f"rf_percentiles_{safe_name}.png"), dpi=150)
        plt.close(fig)


def plot_rf_heatmap(
    rf_history: list[dict],
    checkpoint_epochs: list[int],
    out_dir: str,
    grokking_epoch: int | None = None,
) -> None:
    """Heatmap of mean RF score per layer over checkpoints.

    Rows = layers, columns = checkpoint epochs. Shows which layers change
    topology first across training.

    Output: ``rf_heatmap.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    if not rf_history:
        return
    layer_names = list(rf_history[0].keys())

    # matrix: (n_layers, n_checkpoints)
    matrix = np.zeros((len(layer_names), len(checkpoint_epochs)))
    for j, snap in enumerate(rf_history):
        for i, layer in enumerate(layer_names):
            vals = snap.get(layer, np.array([0.0]))
            matrix[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(8, len(checkpoint_epochs) * 0.4),
                                    max(4, len(layer_names) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, label="Mean RF score")

    ax.set_xticks(range(len(checkpoint_epochs)))
    ax.set_xticklabels(checkpoint_epochs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=8)
    if grokking_epoch is not None:
        grok_col = _nearest_idx(checkpoint_epochs, grokking_epoch)
        ax.axvline(grok_col, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Grokking (epoch {grokking_epoch})")
        ax.legend(frameon=True, fontsize=8, loc="upper left")

    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("Layer")
    ax.set_title("Mean RF Score per Layer over Training")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_rf_change_rate(
    rf_history: list[dict],
    checkpoint_epochs: list[int],
    out_dir: str,
    grokking_epoch: int | None = None,
) -> None:
    """Global mean RF score change (delta) between consecutive checkpoints.

    Expected to spike at or just before the grokking transition.

    Output: ``rf_change_rate.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    mean_rf = [float(np.mean(_rf_global(snap))) for snap in rf_history]
    deltas = [mean_rf[i] - mean_rf[i - 1] for i in range(1, len(mean_rf))]
    epochs = checkpoint_epochs[1:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, deltas, color="darkorange", linewidth=1.5, marker="o",
            markersize=3)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    if grokking_epoch is not None:
        ax.axvline(grokking_epoch, color="crimson", linestyle="--",
                   linewidth=1.2, label=f"Grokking (epoch {grokking_epoch})")
        ax.legend(frameon=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ mean RF score")
    ax.set_title("RF Change Rate (global)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_change_rate.png"), dpi=150)
    plt.close(fig)


# ─── Pruning plots ────────────────────────────────────────────────────────────

def plot_gate_evolution(
    gate_stats_history: list[dict],
    out_dir: str,
) -> None:
    """Tau and temperature trajectories per layer across gate training epochs.

    Output: ``gate_evolution.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    if not gate_stats_history:
        return

    # Detect layer names (all keys that are not 'cycle' or 'epoch')
    layer_names = [k for k in gate_stats_history[0] if k not in ("cycle", "epoch")]
    n_layers = len(layer_names)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(n_layers, 1,
                              figsize=(10, 3 * n_layers),
                              squeeze=False)

    x = list(range(len(gate_stats_history)))
    for i, layer in enumerate(layer_names):
        ax = axes[i, 0]
        taus = [entry[layer]["tau"] for entry in gate_stats_history]
        temps = [entry[layer]["temp"] for entry in gate_stats_history]

        # Mark cycle boundaries
        cycle_ids = [entry["cycle"] for entry in gate_stats_history]
        boundaries = [j for j in range(1, len(cycle_ids))
                      if cycle_ids[j] != cycle_ids[j - 1]]
        for b in boundaries:
            ax.axvline(b, color="grey", linestyle=":", linewidth=0.8)

        ax2 = ax.twinx()
        ax.plot(x, taus, color="steelblue", linewidth=1.5, label="τ (tau)")
        ax2.plot(x, temps, color="darkorange", linewidth=1.5,
                 linestyle="--", label="temp")
        ax.set_ylabel("τ", color="steelblue")
        ax2.set_ylabel("temp", color="darkorange")
        ax.set_title(layer)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, frameon=True, fontsize=8)

    axes[-1, 0].set_xlabel("Gate training step")
    fig.suptitle("Gate Evolution (τ and temperature per layer)", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate_evolution.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_sparsity_evolution(
    sparsity_history: list[dict],
    out_dir: str,
) -> None:
    """Sparsity percentage over gate training steps, with cycle boundary markers.

    Output: ``sparsity_evolution.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    x = list(range(len(sparsity_history)))
    sparsity = [entry["sparsity"] * 100.0 for entry in sparsity_history]

    cycle_ids = [entry["cycle"] for entry in sparsity_history]
    boundaries = [j for j in range(1, len(cycle_ids))
                  if cycle_ids[j] != cycle_ids[j - 1]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, sparsity, color="steelblue", linewidth=1.5)
    for b in boundaries:
        ax.axvline(b, color="crimson", linestyle="--", linewidth=1.0,
                   label="Cycle boundary" if b == boundaries[0] else None)
    if boundaries:
        ax.legend(frameon=True)
    ax.set_xlabel("Gate training step")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Sparsity Evolution")
    ax.set_ylim(-2, 102)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sparsity_evolution.png"), dpi=150)
    plt.close(fig)


def plot_pruning_rf_overlay(
    rf_before: dict,
    rf_after: dict,
    out_dir: str,
    label: str,
) -> None:
    """Before vs after pruning KDE overlay (global across all layers).

    Args:
        rf_before: ``{layer_name: np.ndarray}`` before pruning.
        rf_after: ``{layer_name: np.ndarray}`` after pruning.
        out_dir: Directory to save the PNG.
        label: Appended to filename to distinguish cycles.

    Output: ``rf_pruning_overlay_{label}.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    before_vals = np.concatenate(list(rf_before.values()))
    after_vals = np.concatenate(list(rf_after.values()))

    fig, ax = plt.subplots(figsize=(8, 5))
    for vals, color, name in [
        (before_vals, "#2166ac", "Before pruning"),
        (after_vals, "#d73027", "After pruning"),
    ]:
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(xs, kde(xs), color=color, linewidth=1.8, label=name)

    ax.set_xlabel("RF score")
    ax.set_ylabel("Density")
    ax.set_title(f"RF Distribution Before/After Pruning — {label}")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"rf_pruning_overlay_{label}.png"), dpi=150)
    plt.close(fig)


def plot_pruning_accuracy(
    cycle_results: list[dict],
    out_dir: str,
) -> None:
    """Accuracy before prune, post prune, and post fine-tune across cycles.

    Output: ``pruning_accuracy.png``
    """
    os.makedirs(out_dir, exist_ok=True)
    cycles = [r["cycle"] for r in cycle_results]
    before = [r["acc_before_prune"] * 100 for r in cycle_results]
    post_prune = [r["acc_post_prune"] * 100 for r in cycle_results]
    post_ft = [r["acc_post_finetune"] * 100 for r in cycle_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cycles, before, marker="o", linewidth=1.5, label="Before prune")
    ax.plot(cycles, post_prune, marker="s", linewidth=1.5, label="Post prune")
    ax.plot(cycles, post_ft, marker="^", linewidth=1.5, label="Post fine-tune")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Across Pruning Cycles")
    ax.set_ylim(-2, 102)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pruning_accuracy.png"), dpi=150)
    plt.close(fig)
