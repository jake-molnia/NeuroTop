"""
ntop — Topological analysis and pruning of neural networks.

Core idea: treat each neuron's activation distribution as a 1-D point cloud.
The H0 persistence (largest gap between sorted activations) gives a scalar
importance score called the RF (receptive-field) score. High-RF neurons sit
at topological boundaries and are more important to preserve during pruning.

Public API
----------
collect_activations   — context manager to register forward hooks and collect
                        layer activations during a forward pass.
collect_over_loader   — convenience wrapper: run a DataLoader and collect
                        activations up to max_samples.
analyze               — compute H0 persistence (RF) scores from an activation dict.
compute_rf            — lower-level RF computation (called by analyze).
GatedPruning          — differentiable pruning with learned per-layer soft gates.
LayerwiseThresholds   — nn.Module holding per-layer tau/temperature parameters.
"""

__version__ = "0.0.0a1"

from .monitoring import collect_activations, collect_over_loader, analyze
from .analysis import compute_rf
from .gating import GatedPruning, LayerwiseThresholds
from .utils import (
    save_checkpoint,
    load_checkpoint,
    plot_loss_curves,
    plot_accuracy_curves,
    plot_generalization_gap,
    plot_rf_kde,
    plot_rf_kde_per_layer,
    plot_rf_percentile_evolution,
    plot_rf_percentile_evolution_per_layer,
    plot_rf_heatmap,
    plot_rf_change_rate,
    plot_gate_evolution,
    plot_sparsity_evolution,
    plot_pruning_rf_overlay,
    plot_pruning_accuracy,
)

__all__ = [
    "collect_activations",
    "collect_over_loader",
    "analyze",
    "compute_rf",
    "GatedPruning",
    "LayerwiseThresholds",
    "save_checkpoint",
    "load_checkpoint",
    "plot_loss_curves",
    "plot_accuracy_curves",
    "plot_generalization_gap",
    "plot_rf_kde",
    "plot_rf_kde_per_layer",
    "plot_rf_percentile_evolution",
    "plot_rf_percentile_evolution_per_layer",
    "plot_rf_heatmap",
    "plot_rf_change_rate",
    "plot_gate_evolution",
    "plot_sparsity_evolution",
    "plot_pruning_rf_overlay",
    "plot_pruning_accuracy",
]
