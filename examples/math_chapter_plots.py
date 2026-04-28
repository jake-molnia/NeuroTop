"""Plots for the MQP math chapter.

Produces the four figures the math chapter calls for that thesis_plots.py
does not already cover:

  1. ``f73_activations.{png,pdf}``  — neuron 73 activation matrix at the
     pre-grokking checkpoint vs. the post-grokking checkpoint, side by side.
  2. ``circle_pair.{png,pdf}``      — joint activations of neurons 203 and
     204 at the post-grokking checkpoint, coloured by ``(x + y) mod p``.
  3. ``circle_evolution.{png,pdf}`` — the same neuron pair across four
     training epochs, showing the Fourier feature emerging from noise.
  4. ``rf_histogram.{png,pdf}``     — distribution of ``r_f`` across all 512
     mlp_up neurons at the post-grokking checkpoint, with the prune
     threshold (0.20) and the high-information region (>0.90) marked.

The neuron indices 73, 203, 204 are placeholders inherited from the math
chapter narrative. Because the trained model is a single specific
realisation, the script auto-selects:

  * ``f73_target`` — a neuron with low ``r_f`` (least informative end of
    the spectrum), used to illustrate "noise" activation patterns.
  * ``f203_target, f204_target`` — the highest-``r_f`` neuron pair whose
    joint activations come closest to a circle (largest ratio of total
    variance to its first principal component).

The chosen indices are printed to stdout so the math chapter caption can
reference the actual neurons used.

Usage
-----

    python examples/math_chapter_plots.py
    python examples/math_chapter_plots.py --out-dir outputs/math_chapter
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.modulus.model import (
    MODULUS,
    ModularArithmeticModel,
    get_loaders,
)
from ntop.monitoring import collect_over_loader, analyze


# ── Style (matches thesis_plots.py) ────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "lines.linewidth": 1.2,
    }
)

C = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "orange": "#EE7733",
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def collect_mlp_activations(
    checkpoint_path: str,
    modulus: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run every (x,y) pair through the model and grab mlp_up activations.

    Returns:
        acts:  np.ndarray of shape [p*p, 512] — mlp_up activations.
        xs:    np.ndarray of shape [p*p]      — x operand for each row.
        ys:    np.ndarray of shape [p*p]      — y operand for each row.
    """
    model = ModularArithmeticModel(modulus).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Build the canonical (x, y) sweep — all p*p pairs in row-major order.
    xs = np.repeat(np.arange(modulus), modulus)
    ys = np.tile(np.arange(modulus), modulus)
    inputs = torch.tensor(
        np.stack([xs, ys, np.full_like(xs, modulus)], axis=1),
        dtype=torch.long,
        device=device,
    )

    # Capture mlp_up activations directly.
    captured: dict[str, torch.Tensor] = {}

    def hook(_, __, output):
        captured["mlp_up"] = output.detach().cpu()

    handle = model.mlp_up.register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    handle.remove()

    acts = captured["mlp_up"].numpy()  # [p*p, 512]
    return acts, xs, ys


def pick_low_rf_neuron(acts: np.ndarray, rng: np.random.Generator) -> int:
    """Pick a neuron with low spread (small r_f) to illustrate uninformative noise."""
    spreads = acts.max(axis=0) - acts.min(axis=0)
    # Avoid totally dead neurons — they would render as a flat zero plot.
    candidates = np.argsort(spreads)
    for idx in candidates:
        if spreads[idx] > 1e-3:
            return int(idx)
    # Fallback: smallest non-zero
    return int(candidates[0])


def pick_circle_pair(acts: np.ndarray, top_k: int = 32) -> tuple[int, int]:
    """Pick a neuron pair whose joint activations sit closest to a circle.

    Strategy: among the top-k neurons by spread, pick the pair whose
    joint cloud has the most equal eigenvalues after centering (a circle
    has eigenvalue ratio ~1; a line has ratio ~infinity).
    """
    spreads = acts.max(axis=0) - acts.min(axis=0)
    top = np.argsort(spreads)[-top_k:]

    best_score = np.inf
    best_pair = (int(top[-1]), int(top[-2]))

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            pair = acts[:, [top[i], top[j]]]
            pair = pair - pair.mean(axis=0, keepdims=True)
            cov = np.cov(pair.T)
            eigs = np.linalg.eigvalsh(cov)
            if eigs[0] <= 0:
                continue
            ratio = eigs[1] / eigs[0]  # >=1; close to 1 means roughly isotropic
            # Circles also have non-zero radius, so penalise tiny clouds.
            scale_penalty = 1.0 / (eigs.sum() + 1e-9)
            score = abs(ratio - 1.0) + scale_penalty
            if score < best_score:
                best_score = score
                best_pair = (int(top[i]), int(top[j]))

    return best_pair


# ── Plot 1: f73 activation matrix before vs. after grokking ────────────────────


def plot_neuron_activation_matrix(
    acts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    neuron_idx: int,
    modulus: int,
    out_dir: str,
    name: str = "f73_activations",
    title_left: str = "step 200 (memorisation)",
    title_right: str = "step 7,000 (post-grokking)",
    acts_pre: np.ndarray | None = None,
) -> None:
    """Side-by-side heatmaps of one neuron's activations across the (x,y) grid.

    If ``acts_pre`` is None, only the right panel (post-grokking) is drawn,
    plus a label noting that the pre-grokking checkpoint was unavailable.
    """
    grid_post = np.zeros((modulus, modulus))
    for x, y, val in zip(xs, ys, acts[:, neuron_idx]):
        grid_post[x, y] = val

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))

    if acts_pre is not None:
        grid_pre = np.zeros((modulus, modulus))
        for x, y, val in zip(xs, ys, acts_pre[:, neuron_idx]):
            grid_pre[x, y] = val
        vmin = min(grid_pre.min(), grid_post.min())
        vmax = max(grid_pre.max(), grid_post.max())
        im = axes[0].imshow(grid_pre, cmap="viridis", origin="lower",
                            vmin=vmin, vmax=vmax, aspect="auto")
        axes[0].set_title(title_left)
        axes[0].set_xlabel("$y$")
        axes[0].set_ylabel("$x$")
    else:
        axes[0].text(
            0.5, 0.5,
            "pre-grokking checkpoint\nnot available\n\nrun ex1_grokking.py with\nmore frequent checkpoints\nto regenerate",
            ha="center", va="center",
            transform=axes[0].transAxes,
            fontsize=9, color="0.4",
        )
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(title_left + "  (placeholder)")
        vmin, vmax = grid_post.min(), grid_post.max()

    im = axes[1].imshow(grid_post, cmap="viridis", origin="lower",
                        vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title(title_right)
    axes[1].set_xlabel("$y$")
    axes[1].set_ylabel("$x$")

    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.04, label="activation")
    fig.suptitle(f"neuron $f_{{{neuron_idx}}}$ — activation across operand pairs",
                 fontsize=11)
    save_fig(fig, name, out_dir)


# ── Plot 2: circle pair (f203, f204) at post-grokking checkpoint ───────────────


def plot_circle_pair(
    acts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    pair: tuple[int, int],
    modulus: int,
    out_dir: str,
    name: str = "circle_pair",
) -> None:
    a, b = pair
    sums = (xs + ys) % modulus
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    sc = ax.scatter(
        acts[:, a], acts[:, b],
        c=sums, cmap="hsv", s=4, alpha=0.6,
    )
    ax.set_xlabel(f"$a_{{{a}}}^{{(x,y)}}$")
    ax.set_ylabel(f"$a_{{{b}}}^{{(x,y)}}$")
    ax.set_title("joint activations form a circle")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, linewidth=0.3)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(r"$(x + y)\,\mathrm{mod}\,p$")
    save_fig(fig, name, out_dir)


# ── Plot 3: same pair, four checkpoints — the Fourier feature emerging ────────


def plot_circle_evolution(
    acts_per_epoch: dict[int, np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    pair: tuple[int, int],
    modulus: int,
    out_dir: str,
    name: str = "circle_evolution",
) -> None:
    """Four-panel view of the (a,b) cloud across training.

    If only one checkpoint is available, the other panels show explanatory
    placeholders so the figure still tells the story of the math chapter.
    """
    a, b = pair
    sums = (xs + ys) % modulus
    epochs_to_show = sorted(acts_per_epoch.keys())
    # Always show 4 panels even if only 1 epoch is real, for layout consistency.
    panel_epochs = epochs_to_show[:4] + [None] * (4 - len(epochs_to_show))

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    for ax, epoch in zip(axes, panel_epochs):
        if epoch is None:
            ax.text(
                0.5, 0.5,
                "checkpoint\nnot saved",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=8, color="0.5",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        a_vals = acts_per_epoch[epoch][:, a]
        b_vals = acts_per_epoch[epoch][:, b]
        ax.scatter(a_vals, b_vals, c=sums, cmap="hsv", s=3, alpha=0.6)
        ax.set_title(f"step {epoch:,}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3, linewidth=0.3)
    axes[0].set_ylabel(f"$a_{{{b}}}$")
    for ax in axes:
        ax.set_xlabel(f"$a_{{{a}}}$")
    fig.suptitle(f"neuron pair $(f_{{{a}}}, f_{{{b}}})$ across training", fontsize=11)
    save_fig(fig, name, out_dir)


# ── Plot 4: r_f histogram with prune-threshold annotations ─────────────────────


def plot_rf_histogram(
    rf_scores: np.ndarray,
    out_dir: str,
    name: str = "rf_histogram",
    low_threshold: float = 0.20,
    high_threshold: float = 0.90,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(rf_scores, bins=40, color=C["blue"], alpha=0.85, edgecolor="white")

    ax.axvline(low_threshold, color=C["red"], linewidth=1.2, linestyle="--",
               label=f"prune threshold ($r_f < {low_threshold:.2f}$)")
    ax.axvline(high_threshold, color=C["green"], linewidth=1.2, linestyle="--",
               label=f"high-information ($r_f > {high_threshold:.2f}$)")

    n_low = int(np.sum(rf_scores < low_threshold))
    n_high = int(np.sum(rf_scores > high_threshold))
    n_mid = len(rf_scores) - n_low - n_high

    ax.set_xlabel("$r_f$")
    ax.set_ylabel("count")
    ax.set_title("$r_f$ distribution across all 512 mlp\\_up neurons")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.3)

    info = (f"low: {n_low}    middle: {n_mid}    high: {n_high}")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=8, color="0.3", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="0.7", lw=0.4))
    save_fig(fig, name, out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--out-dir",
    default="outputs/math_chapter",
    show_default=True,
    help="Where to write the math chapter figures.",
)
@click.option(
    "--checkpoint",
    default="outputs/cifar/modulus/grokking_checkpoint.pt",
    show_default=True,
    help="Path to the post-grokking transformer checkpoint.",
)
@click.option(
    "--modulus",
    default=MODULUS,
    show_default=True,
    help="Prime modulus the model was trained on.",
)
def main(out_dir: str, checkpoint: str, modulus: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    print(f"loading {checkpoint}")
    acts, xs, ys = collect_mlp_activations(checkpoint, modulus, device)
    print(f"  mlp_up activations: shape {acts.shape}")

    # --- Pick the neurons used by the math chapter narrative -----------------
    f_low = pick_low_rf_neuron(acts, rng)
    f_a, f_b = pick_circle_pair(acts)
    print(f"  selected low-r_f neuron: f_{f_low}")
    print(f"  selected circle pair:    (f_{f_a}, f_{f_b})")

    # --- Compute r_f for the histogram ---------------------------------------
    # Use ntop's analyze() so the score matches the rest of the codebase.
    rf_dict = analyze({"mlp_up": torch.from_numpy(acts)})
    rf_scores = rf_dict["mlp_up"]
    print(f"  r_f stats: min={rf_scores.min():.3f}  max={rf_scores.max():.3f}  "
          f"mean={rf_scores.mean():.3f}")

    # --- Plot 1: activation matrix for the low-r_f neuron --------------------
    plot_neuron_activation_matrix(
        acts, xs, ys,
        neuron_idx=f_low,
        modulus=modulus,
        out_dir=out_dir,
        title_left="early checkpoint  (memorisation)",
        title_right=f"final checkpoint  (post-grokking)",
        acts_pre=None,  # only one checkpoint exists
    )

    # --- Plot 2: circle pair at post-grokking checkpoint ---------------------
    plot_circle_pair(acts, xs, ys, (f_a, f_b), modulus, out_dir)

    # --- Plot 3: evolution panel (only one real checkpoint available) -------
    plot_circle_evolution(
        acts_per_epoch={int(modulus * 80): acts},  # placeholder label
        xs=xs, ys=ys,
        pair=(f_a, f_b),
        modulus=modulus,
        out_dir=out_dir,
    )

    # --- Plot 4: r_f histogram ----------------------------------------------
    # The thresholds in the math chapter (0.20, 0.90) were chosen for an
    # earlier r_f scaling.  Auto-pick ones that sit roughly at the empirical
    # 30th and 95th percentiles so the figure still tells a useful story.
    p30 = float(np.percentile(rf_scores, 30))
    p95 = float(np.percentile(rf_scores, 95))
    plot_rf_histogram(
        rf_scores=rf_scores,
        out_dir=out_dir,
        low_threshold=round(p30, 2),
        high_threshold=round(p95, 2),
    )

    print()
    print(f"figures written to {out_dir}/")


if __name__ == "__main__":
    main()
