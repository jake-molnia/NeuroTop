import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any


class LayerwiseThresholdGating(nn.Module):
    """
    Per-layer differentiable gating via learned threshold and temperature.

    For each neuron i in layer l:
        gate_i = sigmoid( (rf_i - τ_l) / t_l )

    τ_l (threshold): where the sigmoid is centered — neurons with rf > τ survive.
    t_l (temperature): controls sharpness. High t → soft/uniform. Low t → hard binary.

    Both are learned per-layer so each layer can independently decide
    how aggressively to prune and at what RF cutoff.
    """

    def __init__(self, layer_names: List[str],
                 init_threshold: float = 0.0,
                 init_temp: float = 1.0):
        super().__init__()
        # nn.ParameterDict keys cannot contain '.' so we replace with '__'
        self.tau = nn.ParameterDict({
            self._key(name): nn.Parameter(torch.tensor(init_threshold))
            for name in layer_names
        })
        # Store log(temp) so temp = exp(log_temp) is always positive
        self.log_temp = nn.ParameterDict({
            self._key(name): nn.Parameter(torch.tensor(float(np.log(init_temp))))
            for name in layer_names
        })

    def _key(self, layer_name: str) -> str:
        return layer_name.replace('.', '__')

    def forward(self, layer_name: str, rf_scores: torch.Tensor) -> torch.Tensor:
        key = self._key(layer_name)
        if key not in self.tau:
            return torch.ones(rf_scores.shape[0], device=rf_scores.device)
        tau = self.tau[key]
        temp = torch.exp(self.log_temp[key]).clamp(min=1e-4)
        return torch.sigmoid((rf_scores - tau) / temp)

    def layer_stats(self) -> Dict[str, Dict[str, float]]:
        """Return current τ and t for each layer — useful for logging."""
        stats = {}
        for key in self.tau:
            layer_name = key.replace('__', '.')
            stats[layer_name] = {
                'tau': self.tau[key].item(),
                'temp': torch.exp(self.log_temp[key]).item(),
            }
        return stats


class GatedPruning:
    def __init__(self, model: nn.Module, device: torch.device,
                 lambda_sparse: float = 0.01,
                 lambda_topo: float = 0.1,
                 lambda_polar: float = 0.01):
        self.model = model
        self.device = device
        self.lambda_sparse = lambda_sparse
        self.lambda_topo = lambda_topo
        self.lambda_polar = lambda_polar

        # Populated on first compute_gates call once we know which layers have RF scores
        self.thresholds: Optional[LayerwiseThresholdGating] = None
        self.gates: Dict[str, torch.Tensor] = {}
        self.gate_hooks: List = []
        # Persistent binary mask — neurons zeroed by hard_prune stay dead
        self.pruned_mask: Dict[str, torch.Tensor] = {}

    def _init_thresholds(self, layer_names: List[str], rf_state: Dict[str, Any]):
        """
        Initialize per-layer τ to the mean RF score of that layer so the sigmoid
        starts centered on the actual data distribution rather than at 0.
        """
        rf_values = self._extract_rf(rf_state)
        self.thresholds = LayerwiseThresholdGating(layer_names).to(self.device)

        with torch.no_grad():
            for name in layer_names:
                key = self.thresholds._key(name)
                if key not in self.thresholds.tau:
                    continue
                if name in rf_values:
                    layer_rf = rf_values[name]
                    if isinstance(layer_rf, dict) and 'rf_0' in layer_rf:
                        vals = np.array(layer_rf['rf_0'])
                        init_tau = float(vals.mean()) if vals.std() > 1e-8 else 0.0
                        self.thresholds.tau[key].fill_(init_tau)

        print(f"Initialized per-layer thresholds for {len(layer_names)} layers.")

    def _extract_rf(self, rf_state: Dict[str, Any]) -> Dict:
        rf_values = rf_state.get('rf_values', {})
        if not rf_values:
            for comp_data in rf_state.get('by_components', {}).values():
                rf_values.update(comp_data.get('rf_values', {}))
        return rf_values

    def compute_gates(self, rf_state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        rf_values = self._extract_rf(rf_state)

        # Collect valid layers (Linear with RF scores)
        valid_layers = []
        for layer_name, layer_rf in rf_values.items():
            if not (isinstance(layer_rf, dict) and 'rf_0' in layer_rf):
                continue
            try:
                module = dict(self.model.named_modules())[layer_name]
                if isinstance(module, nn.Linear):
                    valid_layers.append(layer_name)
            except KeyError:
                continue

        # Lazy init on first call
        if self.thresholds is None:
            self._init_thresholds(valid_layers, rf_state)

        self.gates.clear()
        for layer_name in valid_layers:
            layer_rf = rf_values[layer_name]
            rf_tensor = torch.tensor(
                np.array(layer_rf['rf_0']), dtype=torch.float32, device=self.device
            )
            rf_tensor = torch.log1p(rf_tensor)
            gates = self.thresholds(layer_name, rf_tensor)

            # Zero out permanently pruned neurons
            if layer_name in self.pruned_mask:
                gates = gates * self.pruned_mask[layer_name].to(self.device)

            self.gates[layer_name] = gates

        return self.gates

    def apply_gates(self, hard_threshold: Optional[float] = None):
        for hook in self.gate_hooks:
            hook.remove()
        self.gate_hooks.clear()

        def make_hook(layer_name, gates, threshold):
            def hook(module, input, output):
                g = gates.clone()
                if threshold is not None:
                    g = (g >= threshold).float()
                if output.dim() == 3:
                    g = g.view(1, 1, -1)
                elif output.dim() == 2:
                    g = g.view(1, -1)
                return output * g
            return hook

        for layer_name, gates in self.gates.items():
            try:
                module = dict(self.model.named_modules())[layer_name]
                h = module.register_forward_hook(make_hook(layer_name, gates, hard_threshold))
                self.gate_hooks.append(h)
            except KeyError:
                continue

    def hard_prune(self, threshold: float) -> int:
        """
        Permanently zero weights for neurons with gate < threshold.
        Updates pruned_mask so compute_gates keeps them dead going forward.
        """
        newly_pruned = 0
        for layer_name, gates in self.gates.items():
            try:
                module = dict(self.model.named_modules())[layer_name]
                if not isinstance(module, nn.Linear):
                    continue
                dead = (gates.detach() < threshold)
                if layer_name not in self.pruned_mask:
                    self.pruned_mask[layer_name] = torch.ones(
                        gates.shape[0], dtype=torch.float32
                    )
                self.pruned_mask[layer_name][dead.cpu()] = 0.0
                indices = dead.nonzero(as_tuple=True)[0].cpu()
                with torch.no_grad():
                    module.weight.data[indices] = 0.0
                    if module.bias is not None:
                        module.bias.data[indices] = 0.0
                newly_pruned += len(indices)
            except KeyError:
                continue
        return newly_pruned

    def compute_loss(self, task_loss: torch.Tensor, rf_state: Dict[str, Any]) -> torch.Tensor:
        total = task_loss

        # Sparsity: penalize active gates (push toward pruning)
        sparsity = sum(g.sum() for g in self.gates.values())
        total = total + self.lambda_sparse * sparsity

        # Topology: penalize pruning high-RF neurons
        rf_values = self._extract_rf(rf_state)
        topo = torch.tensor(0.0, device=self.device)
        for layer_name, gates in self.gates.items():
            if layer_name in rf_values:
                layer_rf = rf_values[layer_name]
                if isinstance(layer_rf, dict) and 'rf_0' in layer_rf:
                    rf_t = torch.tensor(
                        np.array(layer_rf['rf_0']), dtype=torch.float32, device=self.device
                    )
                    topo = topo + ((1 - gates) * rf_t).sum()
        total = total + self.lambda_topo * topo

        # Polarization: push gates toward 0 or 1, away from 0.5
        polar = sum((g * (1 - g)).sum() for g in self.gates.values())
        total = total + self.lambda_polar * polar

        return total

    def get_sparsity_stats(self) -> Dict[str, Any]:
        total = sum(len(g) for g in self.gates.values())
        active = sum((g >= 0.5).sum().item() for g in self.gates.values())
        return {
            'total_neurons': total,
            'active_neurons': active,
            'sparsity': 1.0 - (active / total) if total > 0 else 0.0,
        }

    def remove_hooks(self):
        for hook in self.gate_hooks:
            hook.remove()
        self.gate_hooks.clear()