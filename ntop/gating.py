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
            init_tau = float(rf_scores[name].mean()) if name in rf_scores and rf_scores[name].std() > 1e-8 else 0.0
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
            gate = g  # capture

            def hook(_, __, output, _gate=gate):
                shape = (1, 1, -1) if output.dim() == 3 else (1, -1)
                return output * _gate.view(*shape)

            self._hooks.append(modules[name].register_forward_hook(hook))

    def hard_prune(self, threshold: float = 0.5) -> int:
        """Zero weights of neurons with gate < threshold. Permanent."""
        modules = dict(self.model.named_modules())
        newly_pruned = 0
        for name, g in self.gates.items():
            if name not in modules:
                continue
            mod = modules[name]
            if not isinstance(mod, nn.Linear):
                continue
            dead = g.detach() < threshold
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
            newly_pruned += len(idx)
        return newly_pruned

    def compute_loss(
        self, task_loss: torch.Tensor, rf_scores: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        loss = task_loss
        total_neurons = sum(len(g) for g in self.gates.values())
        if total_neurons == 0:
            return loss

        # All terms normalized by neuron count so lambdas are scale-invariant
        # Sparsity: mean gate value — penalize keeping neurons active
        sparsity = sum(g.sum() for g in self.gates.values()) / total_neurons
        loss = loss + self.lambda_sparse * sparsity

        # Topology: penalize pruning high-RF neurons, normalized by total RF mass
        topo = torch.tensor(0.0, device=self.device)
        for name, g in self.gates.items():
            if name in rf_scores:
                rf = torch.tensor(rf_scores[name], dtype=torch.float32, device=self.device)
                topo = topo + ((1 - g) * rf).sum()
        topo = topo / total_neurons
        loss = loss + self.lambda_topo * topo

        # Polarization: push gates to 0 or 1, normalized
        polar = sum((g * (1 - g)).sum() for g in self.gates.values()) / total_neurons
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

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()