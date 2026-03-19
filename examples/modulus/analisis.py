"""Overlay RF mean and test loss on a dual-axis plot.

Usage:
    uv run -m examples.modulus.analisis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV = "outputs/modulus/grokking_results.csv"
OUT = "outputs/modulus/plots_grokking/rf_vs_test_loss.png"

df = pd.read_csv(CSV)

grok_rows = df[df["test_acc"] > 0.95]
grok_epoch = int(grok_rows.iloc[0]["epoch"]) if not grok_rows.empty else None

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

if grok_epoch is not None:
    ax1.axvline(grok_epoch, color="#b04060", linewidth=1.2,
                linestyle=":", label=f"Grokking (epoch {grok_epoch})")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper right")

ax1.set_title("RF Mean vs Test Loss", fontsize=14, fontweight="bold")
fig.tight_layout()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}")