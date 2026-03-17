import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional
import torch


def _h0_persistence(activations: jnp.ndarray) -> jnp.ndarray:
    """
    Closed-form H0 persistence for 1D point clouds, vectorized over all neurons.

    For each neuron (column), treat its N_samples scalar activations as points in R^1.
    H0 persistence = largest gap between consecutive sorted values.
    H1 is always empty in R^1 so we skip ripser entirely.

    Args:
        activations: [N_samples, N_neurons]
    Returns:
        rf_0: [N_neurons] — H0 persistence score per neuron
    """
    sorted_acts = jnp.sort(activations, axis=0)          # [N_samples, N_neurons]
    gaps = jnp.diff(sorted_acts, axis=0)                 # [N_samples-1, N_neurons]
    return jnp.max(gaps, axis=0)                         # [N_neurons]


_h0_persistence_jit = jax.jit(_h0_persistence)


def compute_rf(
    activations: Dict[str, torch.Tensor],
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-neuron H0 persistence (RF score) for a dict of activations.

    Args:
        activations: layer_name -> Tensor[N_samples, N_neurons]
        max_samples: if set, subsample rows randomly before computing

    Returns:
        layer_name -> np.ndarray[N_neurons] of H0 persistence scores
    """
    results = {}
    for name, acts in activations.items():
        if acts.dim() > 2:
            acts = acts.flatten(1)

        arr = acts.cpu().float().numpy()

        if max_samples is not None and arr.shape[0] > max_samples:
            idx = np.random.choice(arr.shape[0], max_samples, replace=False)
            arr = arr[idx]

        if arr.shape[0] < 2:
            results[name] = np.zeros(arr.shape[1])
            continue

        # Filter out dead neurons (zero variance) — score stays 0
        var = arr.var(axis=0)
        active = var > 1e-8

        scores = np.zeros(arr.shape[1])
        if active.any():
            x = jnp.array(arr[:, active])
            scores[active] = np.array(_h0_persistence_jit(x))

        results[name] = scores

    return results