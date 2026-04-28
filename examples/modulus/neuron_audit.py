"""Export neuron-level RF and Fourier audit data for modular addition."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from examples.modulus.model import MODULUS, ModularArithmeticModel
from ntop.monitoring import analyze


def _load_model(checkpoint: Path, modulus: int, device: torch.device) -> ModularArithmeticModel:
    model = ModularArithmeticModel(modulus).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _collect_mlp_up(
    model: ModularArithmeticModel,
    modulus: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.repeat(np.arange(modulus), modulus)
    ys = np.tile(np.arange(modulus), modulus)
    inputs = torch.tensor(
        np.stack([xs, ys, np.full_like(xs, modulus)], axis=1),
        dtype=torch.long,
        device=device,
    )
    captured: dict[str, torch.Tensor] = {}

    def hook(_, __, output):
        captured["mlp_up"] = torch.relu(output.detach()).cpu()

    handle = model.mlp_up.register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    handle.remove()
    return captured["mlp_up"].numpy(), xs, ys


def _fourier_summary(grid: np.ndarray) -> dict[str, float | int]:
    centered = grid - grid.mean()
    spectrum = np.fft.fft2(centered)
    magnitude = np.abs(spectrum)
    magnitude[0, 0] = 0.0

    flat_idx = int(np.argmax(magnitude))
    freq_x, freq_y = np.unravel_index(flat_idx, magnitude.shape)
    coeff = spectrum[freq_x, freq_y] / grid.size
    total_power = float(np.square(magnitude).sum())
    peak_power = float(magnitude[freq_x, freq_y] ** 2)

    return {
        "fourier_freq_x": int(freq_x),
        "fourier_freq_y": int(freq_y),
        "fourier_real": float(np.real(coeff)),
        "fourier_imag": float(np.imag(coeff)),
        "fourier_amplitude": float(np.abs(coeff)),
        "fourier_phase": float(np.angle(coeff)),
        "fourier_peak_power_share": peak_power / total_power if total_power else 0.0,
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--checkpoint", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--modulus", default=MODULUS, show_default=True)
@click.option("--top-k", default=32, show_default=True)
def main(checkpoint: Path, out_dir: Path, modulus: int, top_k: int) -> None:
    """Write full and top-k CSV tables for the modulus transformer's MLP neurons."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(checkpoint, modulus, device)
    acts, xs, ys = _collect_mlp_up(model, modulus, device)
    rf = analyze({"mlp_up": torch.from_numpy(acts)})["mlp_up"]
    order = np.argsort(rf)
    percentile = np.empty_like(rf, dtype=float)
    percentile[order] = (np.arange(len(rf)) + 1) / len(rf)

    rows: list[dict[str, float | int]] = []
    for neuron_idx in range(acts.shape[1]):
        grid = np.zeros((modulus, modulus), dtype=float)
        grid[xs, ys] = acts[:, neuron_idx]
        values = acts[:, neuron_idx]
        row: dict[str, float | int] = {
            "layer": "mlp_up",
            "neuron": neuron_idx,
            "rf": float(rf[neuron_idx]),
            "rf_percentile": float(percentile[neuron_idx]),
            "activation_mean": float(values.mean()),
            "activation_std": float(values.std()),
            "activation_min": float(values.min()),
            "activation_max": float(values.max()),
            "activation_nonzero_fraction": float(np.mean(values > 0)),
        }
        row.update(_fourier_summary(grid))
        rows.append(row)

    full = pd.DataFrame(rows).sort_values("rf", ascending=False)
    full["rf_rank_desc"] = np.arange(1, len(full) + 1)
    full_path = out_dir / "modulus_neuron_audit.csv"
    top_path = out_dir / f"modulus_neuron_audit_top{top_k}.csv"
    full.to_csv(full_path, index=False)
    full.head(top_k).to_csv(top_path, index=False)
    click.echo(f"Saved {full_path}")
    click.echo(f"Saved {top_path}")


if __name__ == "__main__":
    main()
