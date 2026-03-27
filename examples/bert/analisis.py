"""Overlay RF mean and test loss on a dual-axis plot.

Usage:
    uv run -m examples.bert.analisis --dataset cola --model-name bert-base-uncased
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset",    required=True, help="GLUE task name (e.g. cola, sst2)")
@click.option("--model-name", required=True, help="Model identifier (e.g. bert-base-uncased)")
@click.option("--out-dir",    default="outputs/bert", show_default=True)
def main(dataset, model_name, out_dir):
    """Plot RF mean vs test loss for a BERT/GLUE training run."""

    run_dir = os.path.join(out_dir, f"{dataset}_{model_name.replace('/', '_')}")
    csv_path = os.path.join(run_dir, "training_results.csv")
    out_path = os.path.join(run_dir, "plots_training", "rf_vs_test_loss.png")

    if not os.path.exists(csv_path):
        print(f"Results CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Mean RF score", fontsize=13, color="#e07b39")
    ax1.plot(df["epoch"], df["rf_mean"], color="#e07b39", linewidth=1.8, label="RF mean")
    ax1.tick_params(axis="y", labelcolor="#e07b39")
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax2 = ax1.twinx()
    ax2.set_ylabel("Test loss", fontsize=13, color="#4c8be0")
    ax2.plot(df["epoch"], df["test_loss"], color="#4c8be0", linewidth=1.8,
             linestyle="--", label="Test loss")
    ax2.tick_params(axis="y", labelcolor="#4c8be0")
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper right")

    ax1.set_title(f"RF Mean vs Test Loss ({model_name} / {dataset})",
                  fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
