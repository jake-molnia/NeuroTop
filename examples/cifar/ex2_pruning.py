"""RF-gated pruning experiment on the CIFAR-10 MLP.

Usage
-----
    uv run -m examples.cifar.ex2_pruning
    uv run -m examples.cifar.ex2_pruning --prune-cycles 4 --finetune-epochs 10
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

from .model import CifarMLP, get_loaders, evaluate, train_epoch
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

def rf_gates(rf_scores: dict, device: torch.device, temp: float = 0.1,
             quantile: float = 0.25) -> dict:
    """Compute soft gates from RF scores.

    Args:
        rf_scores: Per-layer RF scores from ``analyze()``.
        device: Torch device.
        temp: Sigmoid temperature (lower = sharper gate).
        quantile: Threshold quantile — only neurons below this percentile of
            normalised RF are candidates for pruning.  Default 0.25 (25th
            percentile) is much more conservative than the old median.
    """
    gates = {}
    for name, scores in rf_scores.items():
        rf = torch.tensor(scores, dtype=torch.float32, device=device)
        rf_norm = (rf - rf.min()) / (rf.max() - rf.min() + 1e-8)
        tau = rf_norm.quantile(quantile)
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

def hard_prune(
    model: nn.Module,
    gates: dict,
    prune_mask: dict,
    max_prune_fraction: float,
    max_prune_ratio: float = 0.20,
    exclude_layers: set[str] | None = None,
) -> int:
    """Zero out the lowest-gate candidates with global and per-layer caps."""
    modules = dict(model.named_modules())
    exclude_layers = exclude_layers or set()
    candidates: list[tuple[float, str, int]] = []
    total_neurons = sum(
        modules[name].weight.shape[0]
        for name in gates
        if name not in exclude_layers and isinstance(modules.get(name), nn.Linear)
    )
    max_new = max(1, int(np.ceil(total_neurons * max_prune_fraction)))

    for name, g in gates.items():
        if name in exclude_layers:
            continue
        mod = modules.get(name)
        if not isinstance(mod, nn.Linear):
            continue
        vals = g.detach().cpu().numpy()
        cut = otsu_threshold(vals)
        existing_mask = prune_mask.get(name, torch.ones(mod.weight.shape[0], dtype=torch.bool))
        for idx in np.flatnonzero(vals < cut):
            if existing_mask[idx]:
                candidates.append((float(vals[idx]), name, int(idx)))

    selected = candidates[:max_new] if len(candidates) <= max_new else sorted(candidates)[:max_new]
    selected_by_layer: dict[str, list[int]] = {}
    for _, name, idx in selected:
        selected_by_layer.setdefault(name, []).append(idx)

    total = 0
    for name, indices in selected_by_layer.items():
        mod = modules[name]
        idx = torch.tensor(indices, dtype=torch.long)
        existing_mask = prune_mask.get(name, torch.ones(mod.weight.shape[0], dtype=torch.bool))
        new_idx = idx[existing_mask[idx]]
        alive_count = existing_mask.sum().item()
        max_kill = max(1, int(np.ceil(alive_count * max_prune_ratio)))
        if len(new_idx) > max_kill:
            gate_vals = gates[name][new_idx.to(gates[name].device)]
            _, keep_order = gate_vals.sort()
            new_idx = new_idx[keep_order[:max_kill].cpu()]
        existing_mask[new_idx] = False
        prune_mask[name] = existing_mask
        device_idx = new_idx.to(mod.weight.device)
        with torch.no_grad():
            mod.weight.data[device_idx] = 0.0
            if mod.bias is not None:
                mod.bias.data[device_idx] = 0.0
        total += len(new_idx)
    return total


def apply_prune_mask(model: nn.Module, prune_mask: dict) -> None:
    """Re-zero any weights that fine-tuning restored."""
    modules = dict(model.named_modules())
    for name, mask in prune_mask.items():
        mod = modules.get(name)
        if mod is None:
            continue
        device_mask = mask.to(mod.weight.device)
        with torch.no_grad():
            mod.weight.data[~device_mask] = 0.0
            if mod.bias is not None:
                mod.bias.data[~device_mask] = 0.0


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--prune-cycles",    default=6,    show_default=True)
@click.option("--finetune-epochs", default=20,   show_default=True)
@click.option("--train-epochs",    default=50,   show_default=True)
@click.option("--gate-temp",       default=0.1,  show_default=True)
@click.option("--gate-quantile",   default=0.25, show_default=True,
              help="RF quantile threshold for pruning (lower = more conservative)")
@click.option("--max-prune-ratio", default=0.20, show_default=True,
              help="Max fraction of alive neurons to prune per layer per cycle")
@click.option("--max-samples",     default=512,  show_default=True)
@click.option("--batch-size",      default=256,  show_default=True)
@click.option("--max-prune-fraction", type=click.FloatRange(0.0, 1.0, min_open=True),
              default=0.05, show_default=True,
              help="Maximum fraction of RF-tracked neurons to newly prune per cycle")
@click.option("--include-output-layer", is_flag=True,
              help="Allow pruning classifier output logits; disabled by default")
@click.option("--max-accuracy-drop", type=click.FloatRange(0.0, 1.0),
              default=0.02, show_default=True,
              help="Fail if final fine-tuned accuracy drops more than this from baseline")
@click.option("--min-final-sparsity", type=click.FloatRange(0.0, 1.0),
              default=0.0, show_default=True,
              help="Fail if final prunable-neuron sparsity is below this value")
@click.option("--checkpoint",      default="outputs/cifar/training_checkpoint.pt", show_default=True)
@click.option("--out-dir",         default="outputs/cifar", show_default=True)
def main(prune_cycles, finetune_epochs, train_epochs,
         gate_temp, gate_quantile, max_prune_ratio,
         max_samples, batch_size, max_prune_fraction,
         include_output_layer, max_accuracy_drop, min_final_sparsity,
         checkpoint, out_dir):
    """RF-gated iterative pruning on the CIFAR-10 MLP."""

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots_pruning")
    os.makedirs(plots_dir, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    console.print(
        f"[dim]CIFAR-10 MLP  cycles={prune_cycles}  max_prune={max_prune_fraction:.1%}"
        f"  device={device}[/]"
    )

    # ── Model + data ──────────────────────────────────────────────────────────
    model        = CifarMLP().to(device)
    optimizer    = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    train_loader, test_loader = get_loaders(batch_size)

    # ── Load or train ─────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(model, optimizer, checkpoint)
    if start_epoch > train_epochs:
        console.print(f"[dim][green]\u2713[/] loaded checkpoint {checkpoint} (epoch {start_epoch})[/]\n")
    else:
        console.print(f"[yellow]no checkpoint {checkpoint} \u2014 training {train_epochs} epochs[/]")
        pretrain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_epochs,
        )
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
                pretrain_sched.step()
                p.advance(task)
        save_checkpoint(model, optimizer, train_epochs, checkpoint)
        console.print()

    # ── Initial RF ────────────────────────────────────────────────────────────
    with console.status("[dim]computing RF scores...[/]"):
        acts      = collect_over_loader(model, test_loader, max_samples=max_samples, verbose=False)
        rf_scores = analyze(acts)

    exclude_layers = set() if include_output_layer else {"net.6"}
    prunable_rf_scores = {
        name: vals for name, vals in rf_scores.items()
        if name not in exclude_layers
    }
    total_neurons = sum(len(v) for v in prunable_rf_scores.values())
    console.print(
        f"[dim]{total_neurons} prunable neurons  {len(prunable_rf_scores)} layers"
        f"  excluded={sorted(exclude_layers) or 'none'}[/]\n"
    )

    acc_before, _  = evaluate(model, test_loader, device)
    baseline_acc = acc_before
    cycle_results: list[dict] = []
    rf_snapshots:  list[dict] = []
    total_pruned = 0
    prune_mask: dict = {}

    # rf_history / checkpoint_cycles for RF plots
    # cycle 0 = initial (pre-prune), then one entry per cycle after fine-tune
    rf_history: list[dict]  = [{k: v.copy() for k, v in rf_scores.items()}]
    checkpoint_cycles: list[int] = [0]

    # ── Cycle loop ────────────────────────────────────────────────────────────
    for cycle in range(prune_cycles):

        # 1. RF gates + hard prune
        gates          = rf_gates(rf_scores, device, temp=gate_temp, quantile=gate_quantile)
        rf_before      = {k: v.copy() for k, v in rf_scores.items()}
        neurons_pruned = hard_prune(
            model, gates, prune_mask, max_prune_fraction,
            max_prune_ratio=max_prune_ratio, exclude_layers=exclude_layers,
        )
        total_pruned  += neurons_pruned
        acc_post_prune, _ = evaluate(model, test_loader, device)
        sparsity       = total_pruned / total_neurons

        # 2. fine-tune the pruned model (re-apply mask each step so weights stay zeroed)
        ft_opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
        ft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            ft_opt, T_max=finetune_epochs,
        )

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
                ft_sched.step()
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
            f"  [dim]\u21b3[/] {acc_before:.3f}"
            f" [dim]\u2192 prune \u2192[/] [red]{acc_post_prune:.3f}[/]"
            f" [dim]\u2192 tune \u2192[/] [{col}]{acc_post_ft:.3f}[/]\n"
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

        # RF plots (cycle axis instead of epoch axis)
        plot_accuracy_curves(cycle_results, plots_dir)
        plot_rf_kde(rf_history, checkpoint_cycles, plots_dir)
        plot_rf_kde_per_layer(rf_history, checkpoint_cycles, plots_dir)
        plot_rf_percentile_evolution(rf_history, checkpoint_cycles, plots_dir)
        plot_rf_percentile_evolution_per_layer(
            rf_history, checkpoint_cycles, layer_names, plots_dir,
        )
        plot_rf_heatmap(rf_history, checkpoint_cycles, plots_dir)
        plot_rf_change_rate(rf_history, checkpoint_cycles, plots_dir)

    console.print(f"[green]\u2713[/] results \u2192 [underline]{csv_path}[/]")
    console.print(f"[green]\u2713[/] plots   \u2192 [underline]{plots_dir}[/]\n")

    if not cycle_results:
        raise click.ClickException("No pruning cycles ran; cannot validate pruning result.")

    final_acc = cycle_results[-1]["acc_post_finetune"]
    final_sparsity = cycle_results[-1]["sparsity"]
    allowed_acc = baseline_acc - max_accuracy_drop
    if final_acc < allowed_acc:
        raise click.ClickException(
            f"Final accuracy {final_acc:.4f} is below allowed floor {allowed_acc:.4f} "
            f"(baseline={baseline_acc:.4f}, max_drop={max_accuracy_drop:.4f})."
        )
    if final_sparsity < min_final_sparsity:
        raise click.ClickException(
            f"Final sparsity {final_sparsity:.4f} is below required "
            f"{min_final_sparsity:.4f}."
        )
    console.print(
        f"[green]\u2713[/] validation passed  final_acc={final_acc:.4f}"
        f"  baseline={baseline_acc:.4f}  sparsity={final_sparsity:.1%}"
    )


if __name__ == "__main__":
    main()
