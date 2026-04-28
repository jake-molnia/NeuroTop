"""Run final pruning experiments with provenance and combined summaries."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import click
import pandas as pd

from examples.bert.model import GLUE_CLASSIFICATION_TASKS, GLUE_TASKS


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _run(cmd: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    return proc.returncode


def _capture(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT)
    except FileNotFoundError as exc:
        return f"{cmd[0]} not found: {exc}\n"
    except subprocess.CalledProcessError as exc:
        return exc.output


def _write_provenance(out_dir: Path, cwd: Path, command_text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    git_status = _capture(["git", "status", "--short"], cwd)
    (out_dir / "command.txt").write_text(command_text + "\n")
    (out_dir / "git_commit.txt").write_text(_capture(["git", "rev-parse", "HEAD"], cwd))
    (out_dir / "git_status.txt").write_text(git_status)
    (out_dir / "environment.txt").write_text(
        _capture([sys.executable, "-m", "pip", "freeze"], cwd)
    )
    device = [
        f"platform={platform.platform()}",
        f"python={sys.version}",
    ]
    device.append(_capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], cwd))
    (out_dir / "device.txt").write_text("\n".join(device))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--example", type=click.Choice(["modulus", "cifar", "bert"]), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--methods", default="rf,random", show_default=True)
@click.option("--seeds", default="0,1,2", show_default=True,
              help="Comma-separated pruning/fine-tune seeds. Use seed-specific checkpoints/model dirs for independent training replicates.")
@click.option("--fractions", required=True)
@click.option("--alpha", default=1.0, show_default=True)
@click.option("--beta", default=1.0, show_default=True)
@click.option("--gamma", default=1.0, show_default=True)
@click.option("--checkpoint", type=click.Path(dir_okay=False))
@click.option("--prune-cycles", default=4, show_default=True)
@click.option("--finetune-epochs", default=2, show_default=True)
@click.option("--train-epochs", default=3, show_default=True)
@click.option("--max-samples", default=512, show_default=True)
@click.option("--subset-size", default=5000, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
@click.option("--modulus", default=113, show_default=True)
@click.option("--dataset", default="cola", show_default=True)
@click.option("--datasets", default=None,
              help="Comma-separated GLUE task list for BERT. Overrides --dataset.")
@click.option("--model-name", default="bert-base-uncased", show_default=True)
@click.option("--models-dir", default="./trained_models", show_default=True)
@click.option("--max-accuracy-drop", default=0.02, show_default=True)
@click.option("--min-final-sparsity", default=0.0, show_default=True)
def main(
    example: str,
    out_dir: Path,
    methods: str,
    seeds: str,
    fractions: str,
    alpha: float,
    beta: float,
    gamma: float,
    checkpoint: str | None,
    prune_cycles: int,
    finetune_epochs: int,
    train_epochs: int,
    max_samples: int,
    subset_size: int,
    batch_size: int,
    modulus: int,
    dataset: str,
    datasets: str | None,
    model_name: str,
    models_dir: str,
    max_accuracy_drop: float,
    min_final_sparsity: float,
) -> None:
    cwd = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []

    dataset_values = [dataset]
    if datasets:
        dataset_values = _split_csv(datasets)
        if example != "bert":
            raise click.ClickException("--datasets is only supported for --example bert.")
    if example == "bert":
        unknown = sorted(set(dataset_values).difference(GLUE_TASKS))
        if unknown:
            raise click.ClickException(f"Unknown GLUE task(s): {','.join(unknown)}")
        regression = [
            name for name in dataset_values
            if name not in GLUE_CLASSIFICATION_TASKS
        ]
        if regression:
            raise click.ClickException(
                "final_experiments currently reports accuracy; run regression "
                f"GLUE tasks separately: {','.join(regression)}"
            )

    multiple_datasets = example == "bert" and len(dataset_values) > 1

    for dataset_name in dataset_values:
        dataset_prefix = dataset_name if multiple_datasets else None
        dataset_root = out_dir / dataset_name if dataset_prefix else out_dir
        for method in _split_csv(methods):
            for seed in _split_csv(seeds):
                run_dir = dataset_root / example / method / f"seed_{seed}"
                cmd = [
                    sys.executable, "-m", "examples.pruning_sweep",
                    "--example", example,
                    "--fractions", fractions,
                    "--out-dir", os.fspath(run_dir),
                    "--prune-cycles", str(prune_cycles),
                    "--finetune-epochs", str(finetune_epochs),
                    "--train-epochs", str(train_epochs),
                    "--max-samples", str(max_samples),
                    "--subset-size", str(subset_size),
                    "--batch-size", str(batch_size),
                    "--modulus", str(modulus),
                    "--dataset", dataset_name,
                    "--model-name", model_name,
                    "--models-dir", models_dir,
                    "--seed", seed,
                    "--pruning-method", method,
                    "--alpha", str(alpha),
                    "--beta", str(beta),
                    "--gamma", str(gamma),
                    "--max-accuracy-drop", str(max_accuracy_drop),
                    "--min-final-sparsity", str(min_final_sparsity),
                ]
                if checkpoint:
                    cmd.extend(["--checkpoint", checkpoint])

                _write_provenance(run_dir, cwd, " ".join(cmd))
                status = _run(cmd, cwd, run_dir / "run.log")

                summary_path = run_dir / "sweep_summary.csv"
                if summary_path.exists():
                    df = pd.read_csv(summary_path)
                    if example == "bert":
                        df["dataset"] = dataset_name
                        df["model_name"] = model_name
                    df["pruning_seed"] = int(seed)
                    df["run_status"] = "passed" if status == 0 else "failed"
                    rows.append(df)
                else:
                    rows.append(pd.DataFrame([{
                        "example": example,
                        "dataset": dataset_name if example == "bert" else "",
                        "model_name": model_name if example == "bert" else "",
                        "pruning_method": method,
                        "pruning_seed": int(seed),
                        "run_status": "failed",
                        "error": "missing sweep_summary.csv",
                    }]))

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined_name = f"{example}_glue_combined_summary.csv" if multiple_datasets else f"{example}_combined_summary.csv"
    combined_path = out_dir / combined_name
    combined.to_csv(combined_path, index=False)
    if multiple_datasets and not combined.empty:
        for dataset_name, dataset_df in combined.groupby("dataset", dropna=False):
            dataset_path = out_dir / str(dataset_name) / f"{example}_{dataset_name}_combined_summary.csv"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_df.to_csv(dataset_path, index=False)
    click.echo(f"Saved combined summary to {combined_path}")
    if not combined.empty:
        click.echo(combined.to_string(index=False))
    failed_status = (
        "status" in combined.columns and (combined["status"] == "failed").any()
    )
    failed_run = (
        "run_status" in combined.columns and (combined["run_status"] == "failed").any()
    )
    if not combined.empty and (failed_status or failed_run):
        raise click.ClickException("One or more final experiment runs failed validation.")


if __name__ == "__main__":
    main()
