"""Differentiable gated pruning with per-layer learned thresholds.

Soft gates are computed from RF scores via a per-layer sigmoid:

    gate_i = sigmoid((log1p(rf_i) - τ_l) / exp(log_temp_l))

During gate training the model weights are frozen; only ``τ`` and
``log_temp`` are updated through a composite loss that balances sparsity,
topology preservation, and gate polarisation.  Neurons whose gate falls
below a hard threshold are then permanently zeroed (hard prune) and masked
after every subsequent optimiser step.

Typical workflow::

    acts = collect_over_loader(model, loader)
    rf_scores = analyze(acts)

    gated = GatedPruning(model, device, rf_scores)
    gate_optimizer = torch.optim.Adam(gated.thresholds.parameters(), lr=1e-3)

    for epoch in range(gate_epochs):
        gated.compute_gates(rf_scores)
        gated.apply_gates()          # register forward hooks
        for batch in loader:
            gate_optimizer.zero_grad()
            task_loss = model(**batch).loss
            loss = gated.compute_loss(task_loss, rf_scores)
            loss.backward()
            gate_optimizer.step()

    pruned = gated.hard_prune(threshold=0.5)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any


class LayerwiseThresholds(nn.Module):
    """
    Per-layer learned threshold (τ) and temperature (log_temp).

    gate_i = sigmoid((rf_i - τ_l) / exp(log_temp_l))

    τ is initialized to the mean RF score of the layer so the sigmoid
    starts centered on the actual data distribution.
    """

    def __init__(self, layer_names: List[str], rf_scores: Dict[str, np.ndarray]):
        super().__init__()
        # nn.ParameterDict doesn't allow '.' in keys
        self._key = lambda n: n.replace('.', '__')

        taus, log_temps = {}, {}
        for name in layer_names:
            key = self._key(name)
            init_tau = float(np.log1p(rf_scores[name]).mean()) if name in rf_scores and np.log1p(rf_scores[name]).std() > 1e-8 else 0.0
            taus[key] = nn.Parameter(torch.tensor(init_tau))
            log_temps[key] = nn.Parameter(torch.tensor(0.0))  # temp = exp(0) = 1.0

        self.tau = nn.ParameterDict(taus)
        self.log_temp = nn.ParameterDict(log_temps)

    def forward(self, layer_name: str, rf_tensor: torch.Tensor) -> torch.Tensor:
        key = self._key(layer_name)
        tau = self.tau[key]
        temp = torch.exp(self.log_temp[key]).clamp(min=1e-4)
        return torch.sigmoid((rf_tensor - tau) / temp)

    def stats(self) -> Dict[str, Dict[str, float]]:
        return {
            k.replace('__', '.'): {
                'tau': self.tau[k].item(),
                'temp': torch.exp(self.log_temp[k]).item(),
            }
            for k in self.tau
        }


class GatedPruning:
    """
    Differentiable gating over neurons using per-layer learned thresholds.

    Workflow:
        1. Compute RF scores (from ntop.monitoring.analyze)
        2. gated = GatedPruning(model, device, rf_scores)
        3. In training loop: gated.apply_gates(); loss = gated.compute_loss(task_loss, rf_scores)
        4. Periodically: gated.hard_prune(threshold)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        rf_scores: Dict[str, np.ndarray],
        lambda_sparse: float = 0.01,
        lambda_topo: float = 0.1,
        lambda_polar: float = 0.01,
    ):
        self.model = model
        self.device = device
        self.lambda_sparse = lambda_sparse
        self.lambda_topo = lambda_topo
        self.lambda_polar = lambda_polar

        # Only gate Linear layers that have RF scores
        modules = dict(model.named_modules())
        self.layer_names = [
            name for name, mod in modules.items()
            if isinstance(mod, nn.Linear) and name in rf_scores
        ]

        self.thresholds = LayerwiseThresholds(self.layer_names, rf_scores).to(device)
        self.gates: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        # Permanent binary mask — zeroed neurons stay zeroed
        self._dead: Dict[str, torch.Tensor] = {}

    def compute_gates(self, rf_scores: Dict[str, np.ndarray]):
        self.gates.clear()
        for name in self.layer_names:
            if name not in rf_scores:
                continue
            rf = torch.tensor(
                np.log1p(rf_scores[name]), dtype=torch.float32, device=self.device
            )
            g = self.thresholds(name, rf)
            if name in self._dead:
                g = g * self._dead[name].to(self.device)
            self.gates[name] = g

    def apply_gates(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        modules = dict(self.model.named_modules())
        for name, g in self.gates.items():
            if name not in modules:
                continue
            # Detach so the hook doesn't hold a reference into the computation
            # graph — hooks are pure forward masking, gradients flow through
            # compute_loss instead.
            gate = g.detach()

            def hook(_, __, output, _gate=gate):
                shape = (1, 1, -1) if output.dim() == 3 else (1, -1)
                return output * _gate.view(*shape)

            self._hooks.append(modules[name].register_forward_hook(hook))

    def hard_prune(self, threshold: float = None) -> Dict[str, torch.Tensor]:
        """Zero weights of neurons below the per-layer Otsu threshold.

        Uses Otsu's method on each layer's gate values to find the natural
        binary split — no fixed threshold needed. Falls back to the midpoint
        if the gate distribution is degenerate (all values identical).

        The ``threshold`` argument is kept for API compatibility but ignored.
        """
        def otsu_threshold(values: np.ndarray) -> float:
            if values.max() - values.min() < 1e-6:
                return values.mean()
            bins = np.linspace(values.min(), values.max(), 256)
            best_t, best_var = bins[0], -1.0
            for t in bins:
                w0 = (values < t).mean()
                w1 = 1.0 - w0
                if w0 == 0 or w1 == 0:
                    continue
                var = w0 * w1 * (values[values < t].mean() - values[values >= t].mean()) ** 2
                if var > best_var:
                    best_var, best_t = var, t
            return best_t

        modules = dict(self.model.named_modules())
        pruned_indices: Dict[str, torch.Tensor] = {}

        for name, g in self.gates.items():
            if name not in modules:
                continue
            mod = modules[name]
            if not isinstance(mod, nn.Linear):
                continue

            gate_vals = g.detach().cpu().numpy()
            cut = otsu_threshold(gate_vals)
            dead = g.detach() < cut

            if not dead.any():
                continue

            if name not in self._dead:
                self._dead[name] = torch.ones(g.shape[0])
            self._dead[name][dead.cpu()] = 0.0

            idx = dead.nonzero(as_tuple=True)[0]
            with torch.no_grad():
                mod.weight.data[idx] = 0.0
                if mod.bias is not None:
                    mod.bias.data[idx] = 0.0
            pruned_indices[name] = idx

        return pruned_indices

    def compute_loss(
        self, task_loss: torch.Tensor, rf_scores: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        # Recompute gates fresh here so gradients flow to tau/temp each batch.
        # Hooks use detached epoch-start gates for forward masking; this is the
        # only place where tau/temp receive gradients.
        live_gates: Dict[str, torch.Tensor] = {}
        for name in self.layer_names:
            if name not in rf_scores:
                continue
            rf = torch.tensor(
                np.log1p(rf_scores[name]), dtype=torch.float32, device=self.device
            )
            g = self.thresholds(name, rf)
            if name in self._dead:
                g = g * self._dead[name].to(self.device)
            live_gates[name] = g

        loss = task_loss
        total_neurons = sum(len(g) for g in live_gates.values())
        if total_neurons == 0:
            return loss

        # All terms normalized by neuron count so lambdas are scale-invariant
        # Sparsity: mean gate value — penalize keeping neurons active
        sparsity = sum(g.sum() for g in live_gates.values()) / total_neurons
        loss = loss + self.lambda_sparse * sparsity

        # Topology: penalize pruning high-RF neurons, normalized by total RF mass
        topo = torch.tensor(0.0, device=self.device)
        for name, g in live_gates.items():
            if name in rf_scores:
                rf = torch.tensor(np.log1p(rf_scores[name]), dtype=torch.float32, device=self.device)
                topo = topo + ((1 - g) * rf).sum()
        topo = topo / total_neurons
        loss = loss + self.lambda_topo * topo

        # Polarization: push gates to 0 or 1, normalized
        polar = sum((g * (1 - g)).sum() for g in live_gates.values()) / total_neurons
        loss = loss + self.lambda_polar * polar

        return loss

    def sparsity(self) -> Dict[str, Any]:
        total = sum(len(g) for g in self.gates.values())
        active = sum((g >= 0.5).sum().item() for g in self.gates.values())
        return {
            'total': total,
            'active': active,
            'sparsity': 1.0 - active / total if total > 0 else 0.0,
        }

    def apply_dead_mask(self):
        """Re-zero weights for permanently pruned neurons after an optimizer step."""
        modules = dict(self.model.named_modules())
        for name, mask in self._dead.items():
            if name not in modules:
                continue
            mod = modules[name]
            if not isinstance(mod, nn.Linear):
                continue
            dead = mask == 0.0
            if dead.any():
                with torch.no_grad():
                    mod.weight.data[dead] = 0.0
                    if mod.bias is not None:
                        mod.bias.data[dead] = 0.0

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()