"""CIFAR-10 MLP training with topological analysis.

Trains a simple MLP on CIFAR-10 and tracks how RF scores evolve throughout
training.

Usage
-----
    python -m examples.cifar.ex1_training [OPTIONS]
    python -m examples.cifar.ex1_training --help
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import click

from .model import (
    CifarMLP,
    get_loaders,
    evaluate,
    train_epoch,
)
from ntop.monitoring import collect_over_loader, analyze
from ntop.utils import (
    save_checkpoint, load_checkpoint,
    plot_loss_curves, plot_accuracy_curves, plot_generalization_gap,
    plot_rf_kde, plot_rf_kde_per_layer,
    plot_rf_percentile_evolution, plot_rf_percentile_evolution_per_layer,
    plot_rf_heatmap, plot_rf_change_rate,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--epochs",              default=100,   show_default=True, help="Total training epochs")
@click.option("--batch-size",          default=256,   show_default=True, help="Mini-batch size")
@click.option("--lr",                  default=1e-3,  show_default=True, help="AdamW learning rate")
@click.option("--weight-decay",        default=5e-3,  show_default=True, help="AdamW weight decay")
@click.option("--analysis-interval",   default=5,     show_default=True, help="Epochs between RF analyses")
@click.option("--max-samples",         default=500,   show_default=True, help="Activation samples for RF")
@click.option("--checkpoint-interval", default=20,    show_default=True, help="Epochs between checkpoints")
@click.option("--out-dir",             default="outputs/cifar", show_default=True,
              help="Root output directory")
def main(epochs, batch_size, lr, weight_decay,
         analysis_interval, max_samples, checkpoint_interval, out_dir):
    """Train an MLP on CIFAR-10 and track RF topology throughout training."""

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "training_checkpoint.pt")
    plots_dir       = os.path.join(out_dir, "plots_training")
    results_csv     = os.path.join(out_dir, "training_results.csv")
    rf_npz          = os.path.join(out_dir, "training_rf_history.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    click.echo(f"CIFAR-10 MLP training  epochs={epochs:,}  device={device}")
    click.echo(click.style(f"  output: {out_dir}", dim=True))
    click.echo()

    model     = CifarMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    if start_epoch > 0:
        click.echo(click.style(f"Resuming from epoch {start_epoch}.", dim=True))
        # Advance scheduler to correct position after resume
        for _ in range(start_epoch):
            scheduler.step()
    train_loader, test_loader = get_loaders(batch_size)

    results: list[dict]       = []
    rf_history: list[dict]    = []
    checkpoint_epochs: list[int] = []

    for epoch in range(start_epoch, epochs):
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()

        if epoch % checkpoint_interval == 0 and epoch > 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

        if epoch % analysis_interval == 0 or epoch == epochs - 1:
            test_acc, test_loss = evaluate(model, test_loader, device)

            rf_mean = None
            try:
                acts      = collect_over_loader(model, test_loader,
                                                max_samples=max_samples, verbose=False)
                rf_scores = analyze(acts)
                all_rf    = np.concatenate(list(rf_scores.values()))
                rf_mean   = float(np.mean(all_rf))

                rf_history.append({k: v.copy() for k, v in rf_scores.items()})
                checkpoint_epochs.append(epoch)
                results.append({
                    "epoch":      epoch,
                    "train_acc":  train_acc,
                    "test_acc":   test_acc,
                    "train_loss": train_loss,
                    "test_loss":  test_loss,
                    "rf_mean":    rf_mean,
                    "rf_max":     float(np.max(all_rf)),
                    "rf_std":     float(np.std(all_rf)),
                    "rf_p99":     float(np.percentile(all_rf, 99)),
                    "rf_p1":      float(np.percentile(all_rf, 1)),
                })
            except Exception as exc:
                click.echo(click.style(f"warning: RF analysis failed at epoch {epoch}: {exc}",
                                       fg="yellow"), err=True)

            rf_str = f"  rf={rf_mean:.4f}" if rf_mean is not None else ""
            click.echo(
                click.style(f"epoch {epoch:>6}", dim=True)
                + f"  train={train_acc:.3f}"
                + f"  test={test_acc:.3f}"
                + click.style(f"  loss={train_loss:.4f}/{test_loss:.4f}{rf_str}", dim=True)
            )

    # ── Save outputs ───────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, epochs, checkpoint_path)

    if rf_history:
        np.savez(
            rf_npz,
            epochs=np.array(checkpoint_epochs),
            **{f"epoch_{e}": np.concatenate(list(s.values()))
               for e, s in zip(checkpoint_epochs, rf_history)},
        )

    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    if rf_history:
        click.echo(click.style("\nGenerating plots...", dim=True))
        layer_names = list(rf_history[0].keys())

        plot_loss_curves(results, plots_dir)
        plot_accuracy_curves(results, plots_dir)
        plot_generalization_gap(results, plots_dir)
        plot_rf_kde(rf_history, checkpoint_epochs, plots_dir)
        plot_rf_kde_per_layer(rf_history, checkpoint_epochs, plots_dir)
        plot_rf_percentile_evolution(rf_history, checkpoint_epochs, plots_dir)
        plot_rf_percentile_evolution_per_layer(
            rf_history, checkpoint_epochs, layer_names, plots_dir,
        )
        plot_rf_heatmap(rf_history, checkpoint_epochs, plots_dir)
        plot_rf_change_rate(rf_history, checkpoint_epochs, plots_dir)
        click.echo(f"Saved plots to {plots_dir}/")

    click.echo(f"Saved results to {results_csv}")


if __name__ == "__main__":
    main()
