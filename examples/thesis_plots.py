"""Publication-quality thesis figures for NeuroTop experiments.

Reads CSV / NPZ results from CIFAR-10 and modular-addition experiments and
generates 8 figures suitable for an academic thesis.

Usage
-----
    python examples/thesis_plots.py
    python examples/thesis_plots.py --out-dir outputs/thesis
"""

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import click

# ─── LaTeX probe ────────────────────────────────────────────────────────────

def _latex_available() -> bool:
    try:
        plt.rcParams.update({"text.usetex": True})
        fig = plt.figure(figsize=(1, 1))
        fig.text(0.5, 0.5, r"$\alpha$")
        fig.savefig(io.BytesIO(), format="png")
        plt.close(fig)
        return True
    except Exception:
        plt.rcParams.update({"text.usetex": False})
        return False

USE_TEX = _latex_available()

# ─── Style ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "STIX", "Times New Roman"],
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
    "axes.grid": True,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "legend.framealpha": 0.9,
})

# ─── Palette (Paul Tol muted, colorblind-friendly) ─────────────────────────

C = {
    "blue":   "#4477AA",
    "cyan":   "#66CCEE",
    "green":  "#228833",
    "yellow": "#CCBB44",
    "red":    "#EE6677",
    "purple": "#AA3377",
    "grey":   "#BBBBBB",
    "orange": "#EE7733",
}

# ─── Figure sizes (inches) ─────────────────────────────────────────────────

FULL_WIDTH = (7.0, 3.5)
FULL_TALL  = (7.0, 5.0)
DUAL_PANEL = (7.0, 2.8)

# ─── Helpers ────────────────────────────────────────────────────────────────

def _tex(s: str) -> str:
    """Wrap string for LaTeX or plain rendering."""
    return s


def save_fig(fig: plt.Figure, name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def load_data() -> dict:
    root = "outputs"
    data = {}
    data["cifar_train"] = pd.read_csv(os.path.join(root, "cifar", "training_results.csv"))
    data["cifar_prune"] = pd.read_csv(os.path.join(root, "cifar", "pruning_results.csv"))
    data["mod_grok"]    = pd.read_csv(os.path.join(root, "modulus", "grokking_results.csv"))
    data["mod_prune"]   = pd.read_csv(os.path.join(root, "modulus", "pruning_results.csv"))
    data["mod_rf_npz"]  = np.load(os.path.join(root, "modulus", "grokking_rf_history.npz"))
    return data


def _add_phases(ax, grok_ep=2000, xmax=10000, alpha=0.06, label_y=None):
    """Add shaded phase regions for the grokking experiment."""
    phases = [
        (0, 200, C["grey"], "Learning"),
        (200, grok_ep, C["orange"], "Memorization"),
        (grok_ep, 3400, C["red"], "Grokking\ntransition"),
        (3400, xmax, C["green"], "Generalization"),
    ]
    for x0, x1, color, label in phases:
        ax.axvspan(x0, x1, alpha=alpha, color=color, zorder=0)
        if label_y is not None:
            ax.text((x0 + x1) / 2, label_y, label,
                    ha="center", va="top", fontsize=6.5, color="0.35",
                    style="italic")


# ─── Figure 1: Grokking Training Dynamics (Hero) ───────────────────────────

def fig1_grokking_dynamics(data: dict, out_dir: str) -> None:
    df = data["mod_grok"]

    fig, (ax_acc, ax_rf) = plt.subplots(
        2, 1, figsize=FULL_TALL, sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    ep = df["epoch"]

    # --- Top: accuracy ---
    ax_acc.plot(ep, df["train_acc"], color=C["cyan"], label="Train accuracy")
    ax_acc.plot(ep, df["test_acc"],  color=C["blue"], label="Test accuracy")
    ax_acc.axvline(2000, color=C["red"], ls="--", lw=0.8, zorder=5)
    _add_phases(ax_acc, label_y=1.02)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.03, 1.08)
    ax_acc.legend(loc="center right", fontsize=7)
    ax_acc.set_title("Grokking Training Dynamics: Modular Addition (p=113)")

    # --- Bottom: RF mean ---
    ax_rf.plot(ep, df["rf_mean"], color=C["orange"], lw=1.4)
    ax_rf.axvline(2000, color=C["red"], ls="--", lw=0.8, zorder=5)
    _add_phases(ax_rf)

    # Annotate peak RF
    peak_idx = df["rf_mean"].idxmax()
    peak_ep  = df["epoch"].iloc[peak_idx]
    peak_val = df["rf_mean"].iloc[peak_idx]
    ax_rf.annotate(
        f"Peak RF = {peak_val:.1f}",
        xy=(peak_ep, peak_val), xytext=(peak_ep + 1500, peak_val + 1),
        fontsize=7, color="0.3",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6),
    )
    # Annotate settled RF
    settled = df.loc[df["epoch"] >= 5000, "rf_mean"].mean()
    ax_rf.axhline(settled, color=C["grey"], ls=":", lw=0.6)
    ax_rf.text(8500, settled + 0.4, f"Settled RF $\\approx$ {settled:.1f}",
               fontsize=7, color="0.4", ha="center")

    ax_rf.set_ylabel(_tex("Mean RF Score ($H_0$ persistence)"))
    ax_rf.set_xlabel("Epoch")

    save_fig(fig, "fig1_grokking_dynamics", out_dir)


# ─── Figure 2: RF as Grokking Indicator ────────────────────────────────────

def fig2_rf_grokking_indicator(data: dict, out_dir: str) -> None:
    df = data["mod_grok"]

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH)
    ep = df["epoch"]

    # Left axis: RF mean
    ln1 = ax1.plot(ep, df["rf_mean"], color=C["orange"], lw=1.4,
                   label="Mean RF score")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(_tex("Mean RF Score ($H_0$ persistence)"), color=C["orange"])
    ax1.tick_params(axis="y", labelcolor=C["orange"])

    # Right axis: test loss
    ax2 = ax1.twinx()
    ln2 = ax2.plot(ep, df["test_loss"], color=C["blue"], lw=1.2, ls="--",
                   label="Test loss")
    ax2.set_ylabel("Test Loss", color=C["blue"])
    ax2.tick_params(axis="y", labelcolor=C["blue"])

    # Grokking line
    ln3 = ax1.axvline(2000, color=C["red"], ls=":", lw=1.0,
                      label=_tex("Grokking ($t_{grok}=2000$)"))

    # Annotate correlated peaks
    ax1.annotate(
        "Correlated peaks",
        xy=(800, 10.6), xytext=(2800, 9.5),
        fontsize=7, color="0.3",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6),
    )

    # Combined legend
    lines = ln1 + ln2 + [ln3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=7)

    ax1.set_title("RF Score Tracks Generalization Dynamics")

    save_fig(fig, "fig2_rf_grokking_indicator", out_dir)


# ─── Figure 3: CIFAR Training Convergence ──────────────────────────────────

def fig3_cifar_convergence(data: dict, out_dir: str) -> None:
    df = data["cifar_train"]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=DUAL_PANEL)
    ep = df["epoch"]

    # Left: accuracy
    ax_acc.plot(ep, df["train_acc"], color=C["cyan"], label="Train")
    ax_acc.plot(ep, df["test_acc"],  color=C["blue"], label="Test")
    final_test = df["test_acc"].iloc[-1]
    ax_acc.annotate(
        f"Test = {final_test:.1%}",
        xy=(ep.iloc[-1], final_test),
        xytext=(ep.iloc[-1] - 35, final_test - 0.08),
        fontsize=7.5, fontweight="bold", color=C["blue"],
        arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.6),
    )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(fontsize=7)
    ax_acc.set_ylim(0.25, 0.68)

    # Right: loss
    ax_loss.plot(ep, df["train_loss"], color=C["cyan"], label="Train")
    ax_loss.plot(ep, df["test_loss"],  color=C["blue"], label="Test")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(fontsize=7)

    fig.suptitle("CIFAR-10 MLP Training (BatchNorm + Dropout + Augmentation)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3_cifar_convergence", out_dir)


# ─── Figure 4: Pruning Accuracy Retention (CIFAR) ──────────────────────────

def fig4_cifar_pruning(data: dict, out_dir: str) -> None:
    df = data["cifar_prune"]

    fig, ax = plt.subplots(figsize=FULL_WIDTH)
    sp = df["sparsity"] * 100

    ax.plot(sp, df["acc_before_prune"] * 100, color=C["blue"],
            marker="o", ms=4, label="Before prune")
    ax.plot(sp, df["acc_post_prune"] * 100, color=C["yellow"],
            marker="s", ms=4, label="Post prune")
    ax.plot(sp, df["acc_post_finetune"] * 100, color=C["green"],
            marker="^", ms=4, label="Post fine-tune")

    # Shaded regime regions
    ax.axvspan(0, 20, alpha=0.06, color=C["green"], zorder=0)
    ax.axvspan(20, 36, alpha=0.06, color=C["yellow"], zorder=0)
    ax.axvspan(36, 50, alpha=0.06, color=C["red"], zorder=0)

    # Regime labels
    for x, label in [(10, "Free\npruning"), (28, "Plateau"), (43, "Accuracy\ncliff")]:
        ax.text(x, 27, label, ha="center", fontsize=6.5, color="0.4",
                style="italic")

    # Sweet spot annotation
    ax.annotate(
        _tex("Sweet spot: 33\\% sparsity\n88\\% retention"),
        xy=(33.1, 53.6), xytext=(38, 58),
        fontsize=7, color="0.3",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    ax.set_xlabel(_tex("Sparsity (\\%)"))
    ax.set_ylabel(_tex("Accuracy (\\%)"))
    ax.set_title("RF-Guided Pruning: CIFAR-10 MLP")
    ax.set_xlim(0, 50)
    ax.set_ylim(25, 68)
    ax.legend(loc="upper right", fontsize=7)

    save_fig(fig, "fig4_cifar_pruning", out_dir)


# ─── Figure 5: Pruning Accuracy Retention (Modulus) ────────────────────────

def fig5_modulus_pruning(data: dict, out_dir: str) -> None:
    df = data["mod_prune"]

    fig, ax = plt.subplots(figsize=FULL_WIDTH)
    sp = df["sparsity"] * 100

    # Clip the post-prune outlier (cycle 0: 6.4%) for visual clarity
    post_prune_clipped = df["acc_post_prune"].clip(lower=0.55) * 100

    ax.plot(sp, df["acc_before_prune"] * 100, color=C["blue"],
            marker="o", ms=4, label="Before prune")
    ax.plot(sp, post_prune_clipped, color=C["yellow"],
            marker="s", ms=4, label="Post prune")
    ax.plot(sp, df["acc_post_finetune"] * 100, color=C["green"],
            marker="^", ms=4, label="Post fine-tune")

    # Callout for the clipped outlier
    ax.annotate(
        _tex("Cycle 1 post-prune: 6.4\\%\n(off-scale, recovers to 96.3\\%)"),
        xy=(sp.iloc[0], 55), xytext=(10, 62),
        fontsize=6.5, color="0.3",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    # Self-limiting onset
    ax.axvline(26.6, color=C["grey"], ls=":", lw=0.7)
    ax.text(27.2, 97, "Self-limiting\nonset", fontsize=6.5, color="0.4",
            style="italic", va="top")

    # Convergence annotation
    ax.annotate(
        _tex("Converges: 73\\% at 31\\% sparsity"),
        xy=(31, 73), xytext=(20, 66),
        fontsize=7, color="0.3",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    ax.set_xlabel(_tex("Sparsity (\\%)"))
    ax.set_ylabel(_tex("Accuracy (\\%)"))
    ax.set_title("RF-Guided Pruning: Grokked Transformer (Modular Addition)")
    ax.set_xlim(0, 35)
    ax.set_ylim(55, 105)
    ax.legend(loc="lower left", fontsize=7)

    save_fig(fig, "fig5_modulus_pruning", out_dir)


# ─── Figure 6: Cross-Experiment Comparison ─────────────────────────────────

def fig6_cross_experiment(data: dict, out_dir: str) -> None:
    cifar = data["cifar_prune"]
    mod   = data["mod_prune"]

    fig, ax = plt.subplots(figsize=FULL_WIDTH)

    # Retention = post_finetune / initial_baseline
    cifar_baseline = cifar["acc_before_prune"].iloc[0]
    mod_baseline   = mod["acc_before_prune"].iloc[0]

    cifar_ret = cifar["acc_post_finetune"] / cifar_baseline
    mod_ret   = mod["acc_post_finetune"]   / mod_baseline

    ax.plot(cifar["sparsity"] * 100, cifar_ret, color=C["blue"],
            marker="o", ms=4, label="CIFAR-10 MLP")
    ax.plot(mod["sparsity"] * 100, mod_ret, color=C["orange"],
            marker="^", ms=4, label="Grokked Transformer")

    # 80% threshold
    ax.axhline(0.80, color=C["grey"], ls="--", lw=0.7)
    ax.text(1, 0.81, _tex("80\\% retention"), fontsize=6.5, color="0.5")

    # Shared saturation zone
    ax.axvspan(28, 33, alpha=0.08, color=C["purple"], zorder=0)
    ax.text(30.5, 0.42, _tex("Shared\nsaturation\nzone"), ha="center",
            fontsize=6.5, color=C["purple"], style="italic", alpha=0.8)

    ax.set_xlabel(_tex("Sparsity (\\%)"))
    ax.set_ylabel("Accuracy Retention (ratio)")
    ax.set_title("Cross-Experiment Comparison: RF-Guided Pruning")
    ax.set_xlim(0, 50)
    ax.set_ylim(0.35, 1.08)
    ax.legend(loc="upper right", fontsize=7)

    save_fig(fig, "fig6_cross_experiment", out_dir)


# ─── Figure 7: RF Distribution Shift (Grokking KDEs) ──────────────────────

def fig7_rf_distribution_shift(data: dict, out_dir: str) -> None:
    npz = data["mod_rf_npz"]

    key_epochs = [
        (0,    C["grey"],   "--", 0.9, "Epoch 0 (init)"),
        (800,  C["red"],    "-",  1.6, "Epoch 800 (memorization peak)"),
        (2000, C["orange"], "-",  1.3, "Epoch 2000 (grokking onset)"),
        (5000, C["green"],  "-",  1.3, "Epoch 5000 (post-grok)"),
    ]

    fig, ax = plt.subplots(figsize=FULL_WIDTH)

    # Use p95 of epoch 800 for a tighter x range that shows the shapes
    vals_800 = npz["epoch_800"]
    vals_800 = vals_800[np.isfinite(vals_800)]
    x_max = float(np.percentile(vals_800, 95)) * 1.1
    xs = np.linspace(0, x_max, 500)

    for ep, color, ls, lw, label in key_epochs:
        key = f"epoch_{ep}"
        if key not in npz:
            continue
        vals = npz[key]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals, bw_method=0.3)
        density = kde(xs)
        ax.fill_between(xs, density, alpha=0.12, color=color, zorder=2)
        ax.plot(xs, density, color=color, ls=ls, lw=lw, label=label, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)

    # Annotate the heavy tail
    ax.annotate(
        "Broader distribution\nduring memorization",
        xy=(x_max * 0.65, 2e-3), xytext=(x_max * 0.5, 5e-2),
        fontsize=7, color="0.3",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6),
    )

    # Annotate the compression
    ax.annotate(
        "Compresses\npost-grok",
        xy=(2.0, 0.3), xytext=(6, 0.5),
        fontsize=7, color=C["green"],
        arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.6),
    )

    ax.set_xlabel(_tex("RF Score ($H_0$ persistence)"))
    ax.set_ylabel("Density (log scale)")
    ax.set_title("RF Distribution Shift Across Grokking Phases")
    ax.set_xlim(0, x_max)
    ax.legend(loc="upper right", fontsize=7)

    save_fig(fig, "fig7_rf_distribution_shift", out_dir)


# ─── Figure 8: Neurons Pruned Per Cycle ────────────────────────────────────

def fig8_neurons_per_cycle(data: dict, out_dir: str) -> None:
    cifar = data["cifar_prune"]
    mod   = data["mod_prune"]

    fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # --- Left: CIFAR ---
    cycles_c = cifar["cycle"]
    ax_c.bar(cycles_c, cifar["neurons_pruned"], color=C["blue"], alpha=0.8,
             width=0.7, zorder=3)
    ax_c.set_xlabel("Pruning Cycle")
    ax_c.set_ylabel("Neurons Pruned", color=C["blue"])
    ax_c.tick_params(axis="y", labelcolor=C["blue"])
    ax_c.set_title("CIFAR-10 MLP", fontsize=10)

    ax_c2 = ax_c.twinx()
    ax_c2.plot(cycles_c, cifar["sparsity"] * 100, color=C["blue"],
               ls="--", lw=1.0, marker=".", ms=4)
    ax_c2.set_ylabel(_tex("Cumulative Sparsity (\\%)"), color="0.4")
    ax_c2.tick_params(axis="y", labelcolor="0.4")

    # --- Right: Modulus ---
    cycles_m = mod["cycle"]
    ax_m.bar(cycles_m, mod["neurons_pruned"], color=C["orange"], alpha=0.8,
             width=0.7, zorder=3)
    ax_m.set_xlabel("Pruning Cycle")
    ax_m.set_ylabel("Neurons Pruned", color=C["orange"])
    ax_m.tick_params(axis="y", labelcolor=C["orange"])
    ax_m.set_title("Grokked Transformer", fontsize=10)

    ax_m2 = ax_m.twinx()
    ax_m2.plot(cycles_m, mod["sparsity"] * 100, color=C["orange"],
               ls="--", lw=1.0, marker=".", ms=4)
    ax_m2.set_ylabel(_tex("Cumulative Sparsity (\\%)"), color="0.4")
    ax_m2.tick_params(axis="y", labelcolor="0.4")

    # Self-limiting region label
    self_lim_start = 8
    if len(cycles_m) > self_lim_start:
        ax_m.axvspan(self_lim_start - 0.5, cycles_m.iloc[-1] + 0.5,
                     alpha=0.06, color=C["red"], zorder=0)
        ax_m.text(
            (self_lim_start + cycles_m.iloc[-1]) / 2,
            mod["neurons_pruned"].max() * 0.85,
            _tex("Self-limiting:\n1--4 neurons/cycle"),
            ha="center", fontsize=6.5, color="0.4", style="italic",
        )

    fig.suptitle("Neurons Pruned Per Cycle", fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig8_neurons_per_cycle", out_dir)


# ─── Main ───────────────────────────────────────────────────────────────────

@click.command()
@click.option("--out-dir", default="outputs/thesis", show_default=True,
              help="Directory for output figures")
def main(out_dir: str) -> None:
    """Generate all thesis figures from experiment results."""
    print(f"Generating thesis figures -> {out_dir}/")
    print(f"  LaTeX rendering: {'yes' if USE_TEX else 'no (fallback to serif)'}")
    print()

    data = load_data()

    fig1_grokking_dynamics(data, out_dir)
    fig2_rf_grokking_indicator(data, out_dir)
    fig3_cifar_convergence(data, out_dir)
    fig4_cifar_pruning(data, out_dir)
    fig5_modulus_pruning(data, out_dir)
    fig6_cross_experiment(data, out_dir)
    fig7_rf_distribution_shift(data, out_dir)
    fig8_neurons_per_cycle(data, out_dir)

    print(f"\nDone. {len(os.listdir(out_dir))} files in {out_dir}/")


if __name__ == "__main__":
    main()
