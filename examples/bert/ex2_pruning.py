"""RF-gated pruning experiment on BERT for GLUE tasks.

Usage
-----
    uv run -m examples.bert.ex2_pruning --dataset cola --model-name bert-base-uncased
    uv run -m examples.bert.ex2_pruning --help
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

from .model import (
    GLUE_NUM_LABELS,
    device,
    load_or_train_model,
    get_loaders,
    evaluate,
    train_epoch,
)
from ntop.monitoring import collect_over_loader, analyze
from ntop.utils import (
    plot_pruning_rf_overlay, plot_pruning_accuracy,
    plot_accuracy_curves,
    plot_rf_kde, plot_rf_kde_per_layer,
    plot_rf_percentile_evolution, plot_rf_percentile_evolution_per_layer,
    plot_rf_heatmap, plot_rf_change_rate,
)

console = Console()


# ─── Pruning utilities ────────────────────────────────────────────────────────

def rf_gates(rf_scores: dict, dev: torch.device, temp: float = 0.1) -> dict:
    gates = {}
    for name, scores in rf_scores.items():
        rf = torch.tensor(scores, dtype=torch.float32, device=dev)
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


def _rank01(values: np.ndarray) -> np.ndarray:
    """Return stable ascending ranks scaled to [0, 1]."""
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / (len(values) - 1)


def activation_importance(acts: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Mean absolute activation per neuron from collected hook outputs."""
    return {
        name: tensor.detach().float().abs().mean(dim=0).cpu().numpy()
        for name, tensor in acts.items()
    }


def hard_prune(
    model: nn.Module,
    gates: dict,
    rf_scores: dict,
    act_scores: dict,
    prune_mask: dict,
    max_prune_fraction: float,
    exclude_layers: set[str] | None = None,
    method: str = "rf",
    rng: np.random.Generator | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> int:
    """Zero out selected candidates, capped to a per-cycle budget."""
    modules = dict(model.named_modules())
    exclude_layers = exclude_layers or set()
    rng = rng or np.random.default_rng()
    candidates: list[tuple[float, str, int]] = []
    hybrid_candidates: list[tuple[float, float, float, float, str, int]] = []
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
        if method == "rf":
            candidate_indices = np.flatnonzero(vals < cut)
        elif method == "random":
            candidate_indices = np.flatnonzero(existing_mask.numpy())
        elif method in {"hybrid", "hybrid_no_rf"}:
            candidate_indices = np.flatnonzero(existing_mask.numpy())
        elif method == "hybrid_rf_filter":
            candidate_indices = np.flatnonzero(vals < cut)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        if method in {"hybrid", "hybrid_no_rf", "hybrid_rf_filter"}:
            candidate_indices = np.array(
                [idx for idx in candidate_indices if existing_mask[idx]], dtype=int,
            )
            if len(candidate_indices) == 0:
                continue
            weight = mod.weight.detach().abs().mean(dim=1).cpu().numpy()
            if mod.bias is not None:
                weight = weight + mod.bias.detach().abs().cpu().numpy()
            act = act_scores.get(name, np.zeros(mod.weight.shape[0]))
            savings = float(mod.weight.shape[1] + (1 if mod.bias is not None else 0))
            for idx in candidate_indices:
                hybrid_candidates.append((
                    float(rf_scores[name][idx]),
                    float(weight[idx]),
                    float(act[idx]),
                    savings,
                    name,
                    int(idx),
                ))
            continue
        for idx in candidate_indices:
            if existing_mask[idx]:
                score = float(vals[idx]) if method == "rf" else float(rng.random())
                candidates.append((score, name, int(idx)))

    if method in {"hybrid", "hybrid_no_rf", "hybrid_rf_filter"} and hybrid_candidates:
        raw = np.array([row[:4] for row in hybrid_candidates], dtype=float)
        rf_rank = _rank01(raw[:, 0])
        weight_rank = _rank01(raw[:, 1])
        act_rank = _rank01(raw[:, 2])
        savings_rank = _rank01(raw[:, 3])
        scores = alpha * weight_rank + beta * act_rank - gamma * savings_rank
        if method == "hybrid":
            scores = scores + rf_rank
        for score, row in zip(scores, hybrid_candidates):
            candidates.append((float(score), row[4], row[5]))

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
        existing_mask[new_idx] = False
        prune_mask[name] = existing_mask
        device_idx = new_idx.to(mod.weight.device)
        with torch.no_grad():
            mod.weight.data[device_idx] = 0.0
            if mod.bias is not None:
                mod.bias.data[device_idx] = 0.0
        total += len(new_idx)
    return total


def count_nonzero_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and nonzero parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


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
@click.option("--dataset",         required=True, type=click.Choice(list(GLUE_NUM_LABELS)),
              help="GLUE task name")
@click.option("--model-name",      required=True,
              type=click.Choice(["bert-base-uncased", "bert-large-uncased"]),
              help="HuggingFace model identifier")
@click.option("--prune-cycles",    default=6,    show_default=True)
@click.option("--finetune-epochs", default=3,    show_default=True)
@click.option("--train-epochs",    default=50,   show_default=True)
@click.option("--gate-temp",       default=0.1,  show_default=True)
@click.option("--max-samples",     default=512,  show_default=True)
@click.option("--subset-size",     default=5000, show_default=True)
@click.option("--batch-size",      default=16,   show_default=True)
@click.option("--max-prune-fraction", type=click.FloatRange(0.0, 1.0, min_open=True),
              default=0.05, show_default=True,
              help="Maximum fraction of RF-tracked neurons to newly prune per cycle")
@click.option("--pruning-method", type=click.Choice(["rf", "random", "hybrid", "hybrid_no_rf", "hybrid_rf_filter"]),
              default="rf", show_default=True)
@click.option("--alpha", default=1.0, show_default=True,
              help="Hybrid weight for normalized weight-magnitude rank")
@click.option("--beta", default=1.0, show_default=True,
              help="Hybrid weight for normalized activation-importance rank")
@click.option("--gamma", default=1.0, show_default=True,
              help="Hybrid reward for normalized parameter-savings rank")
@click.option("--include-output-layer", is_flag=True,
              help="Allow pruning classifier output logits; disabled by default")
@click.option("--max-accuracy-drop", type=click.FloatRange(0.0, 1.0),
              default=0.02, show_default=True,
              help="Fail if final fine-tuned accuracy drops more than this from baseline")
@click.option("--min-final-sparsity", type=click.FloatRange(0.0, 1.0),
              default=0.0, show_default=True,
              help="Fail if final prunable-neuron sparsity is below this value")
@click.option("--seed",            default=0,    show_default=True)
@click.option("--models-dir",      default="./trained_models", show_default=True)
@click.option("--out-dir",         default="outputs/bert", show_default=True)
def main(dataset, model_name, prune_cycles, finetune_epochs, train_epochs,
         gate_temp, max_samples, subset_size, batch_size, max_prune_fraction,
         pruning_method, alpha, beta, gamma,
         include_output_layer, max_accuracy_drop, min_final_sparsity, seed,
         models_dir, out_dir):
    """RF-gated iterative pruning on BERT for a GLUE task."""

    out_dir = os.path.join(out_dir, f"{dataset}_{model_name.replace('/', '_')}")
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots_pruning")
    os.makedirs(plots_dir, exist_ok=True)

    console.print(
        f"[dim]BERT pruning  model={model_name}  dataset={dataset}"
        f"  method={pruning_method}  cycles={prune_cycles}  max_prune={max_prune_fraction:.1%}"
        f"  alpha={alpha:g}  beta={beta:g}  gamma={gamma:g}  seed={seed}  device={device}[/]"
    )
    rng = np.random.default_rng(seed)

    # ── Model + data ──────────────────────────────────────────────────────────
    model, tokenizer = load_or_train_model(
        dataset, model_name, subset_size, models_dir, train_epochs, seed=seed,
    )
    train_loader, val_loader = get_loaders(
        dataset, subset_size, tokenizer, batch_size, seed=seed,
    )

    # ── Initial RF ────────────────────────────────────────────────────────────
    with console.status("[dim]computing RF scores...[/]"):
        acts      = collect_over_loader(model, val_loader, max_samples=max_samples, verbose=False)
        rf_scores = analyze(acts)
        act_scores = activation_importance(acts)

    exclude_layers = set() if include_output_layer else {"classifier"}
    prunable_rf_scores = {
        name: vals for name, vals in rf_scores.items()
        if name not in exclude_layers
    }
    total_neurons = sum(len(v) for v in prunable_rf_scores.values())
    console.print(
        f"[dim]{total_neurons} prunable neurons  {len(prunable_rf_scores)} layers"
        f"  excluded={sorted(exclude_layers) or 'none'}[/]\n"
    )

    acc_before, _  = evaluate(model, val_loader)
    baseline_acc = acc_before
    total_params, _ = count_nonzero_parameters(model)
    cycle_results: list[dict] = []
    rf_snapshots:  list[dict] = []
    total_pruned = 0
    prune_mask: dict = {}

    rf_history: list[dict]  = [{k: v.copy() for k, v in rf_scores.items()}]
    checkpoint_cycles: list[int] = [0]

    # ── Cycle loop ────────────────────────────────────────────────────────────
    for cycle in range(prune_cycles):

        # 1. RF gates + hard prune
        gates          = rf_gates(rf_scores, device, temp=gate_temp)
        rf_before      = {k: v.copy() for k, v in rf_scores.items()}
        neurons_pruned = hard_prune(
            model, gates, rf_scores, act_scores, prune_mask,
            max_prune_fraction, exclude_layers,
            method=pruning_method, rng=rng, alpha=alpha, beta=beta, gamma=gamma,
        )
        total_pruned  += neurons_pruned
        acc_post_prune, _ = evaluate(model, val_loader)
        sparsity       = total_pruned / total_neurons

        # 2. fine-tune the pruned model
        ft_opt = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

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
                train_epoch(model, train_loader, ft_opt)
                apply_prune_mask(model, prune_mask)
                test_acc, _ = evaluate(model, val_loader)
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
        acc_post_ft, _ = evaluate(model, val_loader)
        _, nonzero_params = count_nonzero_parameters(model)
        parameter_sparsity = 1.0 - (nonzero_params / total_params)
        col = "green" if acc_post_ft >= acc_before - 0.05 else "red"
        console.print(
            f"  [dim]\u21b3[/] {acc_before:.3f}"
            f" [dim]\u2192 prune \u2192[/] [red]{acc_post_prune:.3f}[/]"
            f" [dim]\u2192 tune \u2192[/] [{col}]{acc_post_ft:.3f}[/]\n"
        )

        # 4. refresh RF
        with console.status("[dim]refreshing RF...[/]", spinner="dots"):
            acts      = collect_over_loader(model, val_loader, max_samples=max_samples, verbose=False)
            rf_scores = analyze(acts)
            act_scores = activation_importance(acts)

        rf_after = {k: v.copy() for k, v in rf_scores.items()}
        rf_snapshots.append({"cycle": cycle, "rf_before": rf_before, "rf_after": rf_after})
        rf_history.append(rf_after)
        checkpoint_cycles.append(cycle + 1)

        all_rf = np.concatenate(list(rf_scores.values()))
        cycle_results.append({
            "epoch":             cycle + 1,
            "cycle":             cycle,
            "train_acc":         acc_before,
            "test_acc":          acc_post_ft,
            "acc_before_prune":  acc_before,
            "acc_post_prune":    acc_post_prune,
            "acc_post_finetune": acc_post_ft,
            "sparsity":          sparsity,
            "neurons_pruned":    neurons_pruned,
            "parameter_sparsity": parameter_sparsity,
            "nonzero_params":     nonzero_params,
            "total_params":       total_params,
            "pruning_method":     pruning_method,
            "alpha":              alpha,
            "beta":               beta,
            "gamma":              gamma,
            "rf_mean":           float(np.mean(all_rf)),
        })
        acc_before = acc_post_ft

    # ── Save + plot ───────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "pruning_results.csv")
    pd.DataFrame(cycle_results).to_csv(csv_path, index=False)

    layer_names = list(rf_history[0].keys())

    try:
        with console.status("[dim]saving plots...[/]"):
            plot_pruning_accuracy(cycle_results, plots_dir)
            for snap in rf_snapshots:
                plot_pruning_rf_overlay(
                    snap["rf_before"], snap["rf_after"],
                    out_dir=plots_dir, label=f"cycle_{snap['cycle']}",
                )

            plot_accuracy_curves(cycle_results, plots_dir)
            plot_rf_kde(rf_history, checkpoint_cycles, plots_dir)
            plot_rf_kde_per_layer(rf_history, checkpoint_cycles, plots_dir)
            plot_rf_percentile_evolution(rf_history, checkpoint_cycles, plots_dir)
            plot_rf_percentile_evolution_per_layer(
                rf_history, checkpoint_cycles, layer_names, plots_dir,
            )
            plot_rf_heatmap(rf_history, checkpoint_cycles, plots_dir)
            plot_rf_change_rate(rf_history, checkpoint_cycles, plots_dir)
    except Exception as exc:
        console.print(f"[yellow]warning: plot generation failed: {exc}[/]")

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
