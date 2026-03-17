import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Type, Union
from contextlib import contextmanager
from .analysis import compute_rf


LayerFilter = Union[
    Type[nn.Module],           # e.g. nn.Linear — hooks all layers of this type
    Callable[[str, nn.Module], bool],  # arbitrary predicate
    List[str],                 # explicit layer names
]


def _make_filter(layer_filter: LayerFilter) -> Callable[[str, nn.Module], bool]:
    if isinstance(layer_filter, list):
        names = set(layer_filter)
        return lambda name, _: name in names
    if isinstance(layer_filter, type):
        return lambda _, module: isinstance(module, layer_filter)
    return layer_filter


@contextmanager
def collect_activations(
    model: nn.Module,
    layer_filter: LayerFilter = nn.Linear,
    device: Optional[torch.device] = None,
):
    """
    Context manager that registers forward hooks and yields an activation collector.

    Usage:
        with collect_activations(model, nn.Linear) as acts:
            model(x)
        # acts is now dict[str, Tensor[N_samples, N_neurons]]

    The primary API for ntop is just a dict[str, Tensor] — you can build
    that however you want (hooks, manual forward, saved activations, etc.)
    and pass it directly to compute_rf().
    """
    should_hook = _make_filter(layer_filter)
    store: Dict[str, List[torch.Tensor]] = {}
    hooks = []

    def make_hook(name):
        def hook(_, __, output):
            out = output[0] if isinstance(output, tuple) else output
            if not isinstance(out, torch.Tensor):
                return
            # Collapse sequence dim if present: [B, T, D] -> [B, D]
            if out.dim() == 3:
                out = out.mean(dim=1)
            store.setdefault(name, []).append(out.detach().cpu())
        return hook

    for name, module in model.named_modules():
        if should_hook(name, module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"Registered hooks on {len(hooks)} layers.")

    try:
        yield store
    finally:
        for h in hooks:
            h.remove()

    # Concatenate accumulated batches in-place
    for name in list(store.keys()):
        store[name] = torch.cat(store[name], dim=0)


def collect_over_loader(
    model: nn.Module,
    loader,
    layer_filter: LayerFilter = nn.Linear,
    max_samples: int = 512,
    input_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience: run model over a dataloader and collect activations.

    Args:
        model: any nn.Module
        loader: DataLoader yielding tensors or dicts
        layer_filter: which layers to hook (type, predicate, or name list)
        max_samples: stop accumulating after this many samples
        input_key: if batches are dicts, which key to use as input.
                   If None and batch is a dict, passes entire dict as kwargs.

    Returns:
        dict[str, Tensor[N_samples, N_neurons]]
    """
    device = next(model.parameters()).device

    with collect_activations(model, layer_filter) as store:
        model.eval()
        seen = 0
        with torch.no_grad():
            for batch in loader:
                if hasattr(batch, 'items'):
                    if input_key:
                        model(batch[input_key].to(device))
                    else:
                        model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()})
                elif isinstance(batch, (list, tuple)):
                    model(batch[0].to(device))
                else:
                    model(batch.to(device))

                if store:
                    first = next(iter(store.values()))
                    seen = sum(t.shape[0] for t in first) if isinstance(first, list) else first.shape[0]
                    if seen >= max_samples:
                        break

    # Trim to max_samples
    for name in store:
        store[name] = store[name][:max_samples]

    print(f"Collected {next(iter(store.values())).shape[0]} samples "
          f"across {len(store)} layers.")
    return store


def analyze(
    activations: Dict[str, torch.Tensor],
    max_samples: Optional[int] = None,
) -> Dict[str, "np.ndarray"]:  # noqa: F821
    """
    Compute RF scores directly from an activation dict.
    This is the primary ntop API — no model needed.

    Args:
        activations: layer_name -> Tensor[N_samples, N_neurons]
        max_samples: subsample if needed

    Returns:
        layer_name -> np.ndarray[N_neurons] of H0 persistence scores
    """
    return compute_rf(activations, max_samples=max_samples)