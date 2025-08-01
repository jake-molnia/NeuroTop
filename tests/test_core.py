import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ntop.core.models import simple_mlp
from ntop.core.activations import extract_activations
from ntop.analysis.distances import compute_distances
from ntop.analysis.persistence import compute_persistence, extract_betti_numbers

def test_model_creation():
    model = simple_mlp(input_dim=784, hidden_dims=[64, 32], output_dim=10)
    assert isinstance(model, nn.Module)
    assert len(model.layers) == 3
    print("✓ Model creation test passed")

def test_activation_extraction():
    model = simple_mlp(input_dim=10, hidden_dims=[8, 6], output_dim=2)
    
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=5)
    
    activations = extract_activations(model, dataloader)
    
    assert len(activations) > 0
    assert 'layer_0' in activations
    assert activations['layer_0'].shape[0] == 20
    assert activations['layer_0'].shape[1] == 8
    print("✓ Activation extraction test passed")

def test_distance_computation():
    activations = torch.randn(10, 5)
    
    for metric in ['euclidean', 'cosine', 'correlation']:
        dist_matrix = compute_distances(activations, metric=metric)
        assert dist_matrix.shape == (10, 10)
        assert np.allclose(np.diag(dist_matrix), 0, atol=1e-10)
        print(f"✓ Distance computation test passed for {metric}")

def test_persistence_computation():
    np.random.seed(42)
    dist_matrix = np.random.rand(10, 10)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)
    
    result = compute_persistence(dist_matrix, max_dim=1)
    assert 'dgms' in result
    assert len(result['dgms']) >= 2
    
    betti = extract_betti_numbers(result)
    assert isinstance(betti, dict)
    assert 0 in betti
    print("✓ Persistence computation test passed")

def test_end_to_end():
    model = simple_mlp(input_dim=20, hidden_dims=[16, 12], output_dim=3)
    
    x = torch.randn(50, 20)
    y = torch.randint(0, 3, (50,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    activations = extract_activations(model, dataloader)
    
    layer_acts = activations['layer_0']
    dist_matrix = compute_distances(layer_acts, metric='euclidean')
    
    persistence = compute_persistence(dist_matrix, max_dim=1)
    betti = extract_betti_numbers(persistence)
    
    assert len(betti) >= 2
    assert all(isinstance(v, (int, np.integer)) for v in betti.values())
    print("✓ End-to-end test passed")

if __name__ == "__main__":
    test_model_creation()
    test_activation_extraction()
    test_distance_computation()
    test_persistence_computation()
    test_end_to_end()
    print("\n🎉 All tests passed!")