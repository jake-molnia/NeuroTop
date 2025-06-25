"""
Pure activation extraction and processing.
No side effects - just data transformation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def extract_activations(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    layers_to_hook: List[str],
    max_samples: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """Extract raw activations from specified layers during forward pass."""
    
    model.to(device)
    model.eval()
    
    activations = {layer: [] for layer in layers_to_hook}
    feature_layers = model.get_feature_layers()
    hooks = []

    def get_hook(name: str):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())
        return hook

    # Register hooks
    for layer_name in layers_to_hook:
        if layer_name not in feature_layers:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        layer = feature_layers[layer_name]
        hooks.append(layer.register_forward_hook(get_hook(layer_name)))

    # Extract activations
    num_samples = 0
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Extracting activations"):
            model(data.to(device))
            num_samples += data.size(0)
            if max_samples and num_samples >= max_samples:
                break

    # Cleanup hooks
    for hook in hooks:
        hook.remove()

    # Concatenate and truncate
    for name, acts in activations.items():
        activations[name] = torch.cat(acts, dim=0)
        if max_samples:
            activations[name] = activations[name][:max_samples]
    
    logger.info(f"Extracted activations: {_get_activation_summary(activations)}")
    return activations

def normalize_activations(
    activations: Dict[str, torch.Tensor],
    method: str = "standard"
) -> Dict[str, torch.Tensor]:
    """Apply normalization to activation tensors."""
    
    if method == "none":
        return activations
    
    normalized = {}
    for name, acts in activations.items():
        if method == "standard":
            scaler = StandardScaler()
            # Normalize across samples (rows), features are columns
            scaled_acts = scaler.fit_transform(acts.T.numpy()).T
            normalized[name] = torch.from_numpy(scaled_acts).float()
        elif method == "minmax":
            scaler = MinMaxScaler()
            scaled_acts = scaler.fit_transform(acts.T.numpy()).T
            normalized[name] = torch.from_numpy(scaled_acts).float()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    logger.info(f"Applied {method} normalization")
    return normalized

def stratify_activations_by_correctness(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    layers_to_hook: List[str],
    max_samples: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract activations stratified by prediction correctness.
    Returns {'correct': {layer: activations}, 'incorrect': {layer: activations}}
    """
    
    model.to(device)
    model.eval()
    
    # Get predictions first
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in data_loader:
            outputs = model(data.to(device))
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    correct_mask = (all_preds == all_labels)
    
    logger.info(f"Found {correct_mask.sum()} correct, {(~correct_mask).sum()} incorrect predictions")
    
    # Extract all activations
    activations = extract_activations(model, data_loader, layers_to_hook, max_samples, device)
    
    # Stratify by correctness
    stratified = {'correct': {}, 'incorrect': {}}
    for name, acts in activations.items():
        stratified['correct'][name] = acts[correct_mask]
        stratified['incorrect'][name] = acts[~correct_mask]
        
        if max_samples:
            stratified['correct'][name] = stratified['correct'][name][:max_samples]
            stratified['incorrect'][name] = stratified['incorrect'][name][:max_samples]
    
    return stratified

def concatenate_layer_activations(
    activations: Dict[str, torch.Tensor],
    layers: List[str]
) -> Tuple[torch.Tensor, List[Tuple[str, int]]]:
    """
    Concatenate activations from multiple layers into single tensor.
    Returns (concatenated_tensor, global_to_local_mapping)
    """
    
    concat_list = []
    global_to_local_map = []
    
    for layer_name in layers:
        if layer_name not in activations:
            logger.warning(f"Layer {layer_name} not found, skipping")
            continue
            
        layer_acts = activations[layer_name]
        concat_list.append(layer_acts)
        
        # Build mapping from global neuron index to (layer, local_index)
        for local_idx in range(layer_acts.shape[1]):
            global_to_local_map.append((layer_name, local_idx))
    
    concatenated = torch.cat(concat_list, dim=1)
    logger.info(f"Concatenated {len(layers)} layers into {concatenated.shape[1]} neurons")
    
    return concatenated, global_to_local_map

def _get_activation_summary(activations: Dict[str, torch.Tensor]) -> str:
    """Helper to create activation summary string."""
    summary = []
    for name, acts in activations.items():
        summary.append(f"{name}: {acts.shape}")
    return ", ".join(summary)