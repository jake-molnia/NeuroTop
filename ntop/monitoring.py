"""Activation collection and RF score analysis.

Provides utilities to hook into a model's forward pass and gather
per-layer activations, then pass those activations to compute_rf().

Typical usage
-------------
    from ntop.monitoring import collect_over_loader, analyze

    acts = collect_over_loader(model, val_loader)
    rf_scores = analyze(acts)   # dict[layer_name -> np.ndarray[N_neurons]]
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Callable, Type, Union
from contextlib import contextmanager
from .analysis import compute_rf


LayerFilter = Union[
    Type[nn.Module],                       # e.g. nn.Linear — hooks all layers of this type
    Callable[[str, nn.Module], bool],       # arbitrary predicate over (name, module)
    List[str],                             # explicit list of layer names
]


def _make_filter(layer_filter: LayerFilter) -> Callable[[str, nn.Module], bool]:
    """Convert a LayerFilter to a (name, module) -> bool predicate."""
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
    verbose: bool = True,
):
    """Context manager that registers forward hooks and yields an activation store.

    The store is a dict mapping layer names to lists of activation tensors.
    After the ``with`` block exits the lists are concatenated in-place so each
    entry becomes a single ``Tensor[N_samples, N_neurons]``.

    3-D outputs (e.g. transformer hidden states ``[B, T, D]``) are averaged
    over the sequence dimension before accumulation.

    Args:
        model: Any ``nn.Module``.
        layer_filter: Which layers to hook — a module type (hooks all layers of
            that type), a ``(name, module) -> bool`` predicate, or an explicit
            list of layer names.
        verbose: If True, print the number of hooked layers.

    Yields:
        store: ``dict[str, list[Tensor]]`` during the ``with`` block;
               ``dict[str, Tensor[N_samples, N_neurons]]`` after exit.

    Example::

        with collect_activations(model, nn.Linear) as acts:
            model(x)
        # acts is now dict[str, Tensor[N_samples, N_neurons]]
        rf_scores = analyze(acts)
    """
    should_hook = _make_filter(layer_filter)
    store: Dict[str, Any] = {}  # list[Tensor] during collection, Tensor after exit
    hooks = []

    def make_hook(name):
        def hook(_, __, output):
            out = output[0] if isinstance(output, tuple) else output
            if not isinstance(out, torch.Tensor):
                return
            # Collapse sequence dimension: [B, T, D] -> [B, D]
            if out.dim() == 3:
                out = out.mean(dim=1)
            store.setdefault(name, []).append(out.detach().cpu())
        return hook

    for name, module in model.named_modules():
        if should_hook(name, module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    if verbose:
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
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Run a model over a DataLoader and collect layer activations.

    Stops after accumulating ``max_samples`` samples and trims any excess.

    Args:
        model: Any ``nn.Module``.
        loader: A DataLoader yielding tensors, lists/tuples, or dicts.
        layer_filter: Which layers to hook (see :func:`collect_activations`).
        max_samples: Stop accumulating after this many samples.
        input_key: When batches are dicts, use this key as the model input.
            If ``None`` and the batch is a dict, all items are passed as kwargs.
        verbose: If True, print collection progress.

    Returns:
        ``dict[layer_name -> Tensor[N_samples, N_neurons]]``
    """
    device = next(model.parameters()).device

    with collect_activations(model, layer_filter, verbose=verbose) as store:
        model.eval()
        seen = 0
        with torch.no_grad():
            for batch in loader:
                if hasattr(batch, "items"):
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
                    seen = (sum(t.shape[0] for t in first)
                            if isinstance(first, list) else first.shape[0])
                    if seen >= max_samples:
                        break

    # Trim to max_samples
    for name in store:
        store[name] = store[name][:max_samples]

    if verbose:
        n_samples = next(iter(store.values())).shape[0]
        print(f"Collected {n_samples} samples across {len(store)} layers.")

    return store


def analyze(
    activations: Dict[str, torch.Tensor],
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute H0 persistence (RF) scores from a layer activation dict.

    This is the primary ntop analysis entry point. No model access needed —
    pass any activation dict (from hooks, manual forward passes, saved files,
    etc.) and get back per-neuron importance scores.

    Args:
        activations: ``layer_name -> Tensor[N_samples, N_neurons]``
        max_samples: Randomly subsample rows before computing if provided.

    Returns:
        ``layer_name -> np.ndarray[N_neurons]`` of H0 persistence scores.
        Neurons with zero variance (dead neurons) receive a score of 0.
    """
    return compute_rf(activations, max_samples=max_samples)
