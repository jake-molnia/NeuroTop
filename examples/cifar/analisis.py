"""Overlay RF mean and test loss on a dual-axis plot.

Usage:
    uv run -m examples.cifar.analisis
"""

import os
from pathlib import Path

import click
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DEFAULT_CSV = Path("outputs/cifar/training_results.csv")
DEFAULT_OUT = Path("outputs/cifar/plots_training/rf_vs_test_loss.png")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_CSV,
    show_default=True,
    help="Path to the training results CSV.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUT,
    show_default=True,
    help="Path to save the plot image.",
)
def main(csv_path: Path, out_path: Path) -> None:
    if not csv_path.is_file():
        raise click.ClickException(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Mean RF score", fontsize=13, color="#e07b39")
    ax1.plot(df["epoch"], df["rf_mean"], color="#e07b39", linewidth=1.8, label="RF mean")
    ax1.tick_params(axis="y", labelcolor="#e07b39")
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax2 = ax1.twinx()
    ax2.set_ylabel("Test loss", fontsize=13, color="#4c8be0")
    ax2.plot(
        df["epoch"],
        df["test_loss"],
        color="#4c8be0",
        linewidth=1.8,
        linestyle="--",
        label="Test loss",
    )
    ax2.tick_params(axis="y", labelcolor="#4c8be0")
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper right")

    ax1.set_title("RF Mean vs Test Loss (CIFAR-10 MLP)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Saved {out_path}")


if __name__ == "__main__":
    main()
