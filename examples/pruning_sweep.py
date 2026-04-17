"""Run a small max-prune-fraction sweep and summarize pruning outcomes.

Usage
-----
    uv run -m examples.pruning_sweep --example modulus --help
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
import pandas as pd


def _fraction_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _run(cmd: list[str]) -> None:
    click.echo(" ".join(cmd))
    subprocess.run(cmd, check=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--example", "example_name",
              type=click.Choice(["modulus", "cifar", "bert"]), required=True)
@click.option("--fractions", default="0.02,0.05,0.10", show_default=True,
              help="Comma-separated max-prune-fraction values.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path),
              required=True, help="Directory that will contain one run per fraction.")
@click.option("--checkpoint", type=click.Path(dir_okay=False),
              help="Checkpoint path; required for modulus and CIFAR, ignored by BERT.")
@click.option("--prune-cycles", default=6, show_default=True)
@click.option("--finetune-epochs", default=10, show_default=True)
@click.option("--train-epochs", default=100, show_default=True)
@click.option("--max-samples", default=256, show_default=True)
@click.option("--subset-size", default=1000, show_default=True,
              help="BERT train subset size; ignored by modulus and CIFAR.")
@click.option("--batch-size", default=256, show_default=True,
              help="Batch size for CIFAR and BERT; ignored by modulus.")
@click.option("--modulus", default=113, show_default=True,
              help="Modulus value; ignored by CIFAR.")
@click.option("--dataset", default="cola", show_default=True,
              help="GLUE dataset; BERT only.")
@click.option("--model-name", default="bert-base-uncased", show_default=True,
              help="HuggingFace model name; BERT only.")
@click.option("--models-dir", default="./trained_models", show_default=True,
              help="Model checkpoint directory; BERT only.")
@click.option("--seed", default=0, show_default=True,
              help="Seed for examples that support deterministic loaders.")
@click.option("--max-accuracy-drop", default=0.02, show_default=True)
@click.option("--min-final-sparsity", default=0.0, show_default=True)
def main(
    example_name: str,
    fractions: str,
    out_dir: Path,
    checkpoint: str,
    prune_cycles: int,
    finetune_epochs: int,
    train_epochs: int,
    max_samples: int,
    subset_size: int,
    batch_size: int,
    modulus: int,
    dataset: str,
    model_name: str,
    models_dir: str,
    seed: int,
    max_accuracy_drop: float,
    min_final_sparsity: float,
) -> None:
    values = [float(part.strip()) for part in fractions.split(",") if part.strip()]
    if not values:
        raise click.ClickException("At least one fraction is required.")
    if example_name in {"modulus", "cifar"} and not checkpoint:
        raise click.ClickException("--checkpoint is required for modulus and CIFAR.")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for value in values:
        run_dir = out_dir / f"max_prune_{_fraction_label(value)}"
        module = f"examples.{example_name}.ex2_pruning"
        cmd = [
            sys.executable, "-m", module,
            "--prune-cycles", str(prune_cycles),
            "--finetune-epochs", str(finetune_epochs),
            "--train-epochs", str(train_epochs),
            "--max-samples", str(max_samples),
            "--max-prune-fraction", str(value),
            "--max-accuracy-drop", str(max_accuracy_drop),
            "--min-final-sparsity", str(min_final_sparsity),
            "--out-dir", os.fspath(run_dir),
        ]
        if example_name == "modulus":
            cmd.extend([
                "--checkpoint", checkpoint,
                "--modulus", str(modulus),
                "--seed", str(seed),
            ])
        elif example_name == "cifar":
            cmd.extend([
                "--checkpoint", checkpoint,
                "--batch-size", str(batch_size),
            ])
        else:
            cmd.extend([
                "--dataset", dataset,
                "--model-name", model_name,
                "--subset-size", str(subset_size),
                "--batch-size", str(batch_size),
                "--models-dir", models_dir,
                "--seed", str(seed),
            ])

        status = "passed"
        error = ""
        try:
            _run(cmd)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error = f"exit code {exc.returncode}"

        csv_path = run_dir / "pruning_results.csv"
        if not csv_path.exists():
            matches = sorted(run_dir.glob("*/pruning_results.csv"))
            if len(matches) == 1:
                csv_path = matches[0]
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                first = df.iloc[0]
                last = df.iloc[-1]
                rows.append({
                    "example": example_name,
                    "max_prune_fraction": value,
                    "status": status,
                    "error": error,
                    "baseline_acc": float(first["acc_before_prune"]),
                    "final_acc": float(last["acc_post_finetune"]),
                    "acc_delta": float(last["acc_post_finetune"] - first["acc_before_prune"]),
                    "min_post_prune_acc": float(df["acc_post_prune"].min()),
                    "final_sparsity": float(last["sparsity"]),
                    "total_pruned": int(df["neurons_pruned"].sum()),
                    "results_csv": os.fspath(csv_path),
                })
                continue

        if status == "passed":
            status = "failed"
            error = "missing or empty pruning_results.csv"
        rows.append({
            "example": example_name,
            "max_prune_fraction": value,
            "status": status,
            "error": error,
            "baseline_acc": None,
            "final_acc": None,
            "acc_delta": None,
            "min_post_prune_acc": None,
            "final_sparsity": None,
            "total_pruned": None,
            "results_csv": os.fspath(csv_path),
        })

    summary_path = out_dir / "sweep_summary.csv"
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    click.echo(f"\nSaved summary to {summary_path}")
    click.echo(summary.to_string(index=False))

    if (summary["status"] == "failed").any():
        raise click.ClickException("One or more sweep runs failed validation.")


if __name__ == "__main__":
    main()
