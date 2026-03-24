"""RF-gated pruning experiment on the modular addition transformer.

Usage
-----
    uv run -m examples.modulus.ex2_pruning
    uv run -m examples.modulus.ex2_pruning --prune-cycles 4 --finetune-epochs 30
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import click

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich import box

from .model import ModularArithmeticModel, get_loaders, evaluate, train_epoch
from ntop.monitoring import collect_over_loader, analyze
from ntop.utils import (
    save_checkpoint, load_checkpoint,
    plot_pruning_rf_overlay, plot_pruning_accuracy,
    plot_accuracy_curves,
    plot_rf_kde, plot_rf_kde_per_layer,
    plot_rf_percentile_evolution, plot_rf_percentile_evolution_per_layer,
    plot_rf_heatmap, plot_rf_change_rate,
)

console = Console()


# ─── Pruning utilities ────────────────────────────────────────────────────────

def rf_gates(rf_scores: dict, device: torch.device, temp: float = 0.1) -> dict:
    gates = {}
    for name, scores in rf_scores.items():
        rf = torch.tensor(scores, dtype=torch.float32, device=device)
        rf_norm = (rf - rf.min()) / (rf.max() - rf.min() + 1e-8)
        tau = rf_norm.median()
        gates[name] = torch.sigmoid((rf_norm - tau) / temp)
    return gates


def otsu_threshold(vals: np.ndarray) -> float:
    if vals.max() - vals.min() < 1e-6:
        return float(vals.mean())
    bins = np.linspace(vals.min(), vals.max(), 256)
    best_t, best_var = bins[0], -1.0
    for t in bins:
        w0, w1 = (vals < t).mean(), (vals >= t).mean()
        if w0 == 0 or w1 == 0:
            continue
        var = w0 * w1 * (vals[vals < t].mean() - vals[vals >= t].mean()) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return float(best_t)


def hard_prune(model: nn.Module, gates: dict, prune_mask: dict) -> int:
    """Zero out dead neurons and update prune_mask. Returns count of newly pruned neurons."""
    modules = dict(model.named_modules())
    total = 0
    for name, g in gates.items():
        mod = modules.get(name)
        if not isinstance(mod, nn.Linear):
            continue
        vals = g.detach().cpu().numpy()
        cut = otsu_threshold(vals)
        dead = torch.tensor(vals < cut)
        if not dead.any():
            continue
        idx = dead.nonzero(as_tuple=True)[0]
        existing_mask = prune_mask.get(name, torch.ones(mod.weight.shape[0], dtype=torch.bool))
        new_idx = idx[existing_mask[idx]]  # only count newly pruned
        existing_mask[idx] = False
        prune_mask[name] = existing_mask
        with torch.no_grad():
            mod.weight.data[idx] = 0.0
            if mod.bias is not None:
                mod.bias.data[idx] = 0.0
        total += len(new_idx)
    return total


def apply_prune_mask(model: nn.Module, prune_mask: dict) -> None:
    """Re-zero any weights that fine-tuning restored."""
    modules = dict(model.named_modules())
    for name, mask in prune_mask.items():
        mod = modules.get(name)
        if mod is None:
            continue
        with torch.no_grad():
            mod.weight.data[~mask] = 0.0
            if mod.bias is not None:
                mod.bias.data[~mask] = 0.0


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--modulus",         default=113,  show_default=True)
@click.option("--prune-cycles",    default=6,    show_default=True)
@click.option("--finetune-epochs", default=50,   show_default=True)
@click.option("--train-epochs",    default=100,  show_default=True)
@click.option("--gate-temp",       default=0.1,  show_default=True)
@click.option("--max-samples",     default=512,  show_default=True)
@click.option("--checkpoint",      default="outputs/modulus/grokking_checkpoint.pt", show_default=True)
@click.option("--out-dir",         default="outputs/modulus", show_default=True)
def main(modulus, prune_cycles, finetune_epochs, train_epochs,
         gate_temp, max_samples, checkpoint, out_dir):
    """RF-gated iterative pruning on the modular addition transformer."""

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots_pruning")
    os.makedirs(plots_dir, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    console.print(f"[dim]p={modulus}  cycles={prune_cycles}  device={device}[/]")

    # ── Model + data ──────────────────────────────────────────────────────────
    model        = ModularArithmeticModel(modulus).to(device)
    optimizer    = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    train_loader, test_loader = get_loaders(modulus, train_split=0.7, batch_size=256)

    # ── Load or train ─────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(model, optimizer, checkpoint)
    if start_epoch > train_epochs:
        console.print(f"[dim][green]✓[/] loaded checkpoint {checkpoint} (epoch {start_epoch})[/]\n")
    else:
        console.print(f"[yellow]no checkpoint {checkpoint} — training {train_epochs} epochs[/]")
        with Progress(
            TextColumn("[dim]{task.description}[/]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as p:
            task = p.add_task("pretrain", total=train_epochs)
            for _ in range(train_epochs):
                train_epoch(model, train_loader, optimizer, criterion, device)
                p.advance(task)
        save_checkpoint(model, optimizer, train_epochs, checkpoint)
        console.print()

    # ── Initial RF ────────────────────────────────────────────────────────────
    with console.status("[dim]computing RF scores...[/]"):
        acts      = collect_over_loader(model, test_loader, max_samples=max_samples, verbose=False)
        rf_scores = analyze(acts)

    total_neurons = sum(len(v) for v in rf_scores.values())
    console.print(f"[dim]{total_neurons} neurons  {len(rf_scores)} layers[/]\n")

    acc_before, _  = evaluate(model, test_loader, device)
    cycle_results: list[dict] = []
    rf_snapshots:  list[dict] = []
    total_pruned = 0
    prune_mask: dict = {}

    # rf_history / checkpoint_cycles for grokking-style RF plots
    # cycle 0 = initial (pre-prune), then one entry per cycle after fine-tune
    rf_history: list[dict]  = [{k: v.copy() for k, v in rf_scores.items()}]
    checkpoint_cycles: list[int] = [0]

    # ── Cycle loop ────────────────────────────────────────────────────────────
    for cycle in range(prune_cycles):

        # 1. RF gates + hard prune
        gates          = rf_gates(rf_scores, device, temp=gate_temp)
        rf_before      = {k: v.copy() for k, v in rf_scores.items()}
        neurons_pruned = hard_prune(model, gates, prune_mask)
        total_pruned  += neurons_pruned
        acc_post_prune, _ = evaluate(model, test_loader, device)
        sparsity       = total_pruned / total_neurons

        # 2. fine-tune the pruned model (re-apply mask each step so weights stay zeroed)
        ft_opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"[bold]cycle {cycle+1}/{prune_cycles}[/]"
                f"  [dim]pruned[/] [bold]{neurons_pruned}[/] [dim]({sparsity:.1%})[/]"
                f"  [red]{acc_post_prune:.3f}[/]",
                total=finetune_epochs,
            )
            for _ in range(finetune_epochs):
                train_epoch(model, train_loader, ft_opt, criterion, device)
                apply_prune_mask(model, prune_mask)
                test_acc, _ = evaluate(model, test_loader, device)
                col = "green" if test_acc >= acc_before - 0.05 else "yellow"
                progress.update(
                    task_id, advance=1,
                    description=(
                        f"[bold]cycle {cycle+1}/{prune_cycles}[/]"
                        f"  [dim]pruned[/] [bold]{neurons_pruned}[/] [dim]({sparsity:.1%})[/]"
                        f"  [{col}]{test_acc:.3f}[/]"
                    ),
                )

        # 3. record
        acc_post_ft, _ = evaluate(model, test_loader, device)
        col = "green" if acc_post_ft >= acc_before - 0.05 else "red"
        console.print(
            f"  [dim]↳[/] {acc_before:.3f}"
            f" [dim]→ prune →[/] [red]{acc_post_prune:.3f}[/]"
            f" [dim]→ tune →[/] [{col}]{acc_post_ft:.3f}[/]\n"
        )

        # 4. refresh RF
        with console.status("[dim]refreshing RF...[/]", spinner="dots"):
            acts      = collect_over_loader(model, test_loader, max_samples=max_samples, verbose=False)
            rf_scores = analyze(acts)

        rf_after = {k: v.copy() for k, v in rf_scores.items()}
        rf_snapshots.append({"cycle": cycle, "rf_before": rf_before, "rf_after": rf_after})
        rf_history.append(rf_after)
        checkpoint_cycles.append(cycle + 1)

        all_rf = np.concatenate(list(rf_scores.values()))
        cycle_results.append({
            "epoch":             cycle + 1,   # alias so plot_accuracy_curves works
            "cycle":             cycle,
            "train_acc":         acc_before,  # accuracy before this cycle's prune
            "test_acc":          acc_post_ft,
            "acc_before_prune":  acc_before,
            "acc_post_prune":    acc_post_prune,
            "acc_post_finetune": acc_post_ft,
            "sparsity":          sparsity,
            "neurons_pruned":    neurons_pruned,
            "rf_mean":           float(np.mean(all_rf)),
        })
        acc_before = acc_post_ft

    # ── Save + plot ───────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "pruning_results.csv")
    pd.DataFrame(cycle_results).to_csv(csv_path, index=False)

    layer_names = list(rf_history[0].keys())

    with console.status("[dim]saving plots...[/]"):
        plot_pruning_accuracy(cycle_results, plots_dir)
        for snap in rf_snapshots:
            plot_pruning_rf_overlay(
                snap["rf_before"], snap["rf_after"],
                out_dir=plots_dir, label=f"cycle_{snap['cycle']}",
            )

        # grokking-style RF plots (cycle axis instead of epoch axis)
        plot_accuracy_curves(cycle_results, plots_dir)
        plot_rf_kde(rf_history, checkpoint_cycles, plots_dir)
        plot_rf_kde_per_layer(rf_history, checkpoint_cycles, plots_dir)
        plot_rf_percentile_evolution(rf_history, checkpoint_cycles, plots_dir)
        plot_rf_percentile_evolution_per_layer(
            rf_history, checkpoint_cycles, layer_names, plots_dir,
        )
        plot_rf_heatmap(rf_history, checkpoint_cycles, plots_dir)
        plot_rf_change_rate(rf_history, checkpoint_cycles, plots_dir)

    console.print(f"[green]✓[/] results → [underline]{csv_path}[/]")
    console.print(f"[green]✓[/] plots   → [underline]{plots_dir}[/]\n")


if __name__ == "__main__":
    main()