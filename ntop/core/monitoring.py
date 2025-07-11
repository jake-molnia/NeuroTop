import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from ..analysis.distances import compute_distances
from ..analysis.persistence import PersistenceAnalyzer
from ..core.activations import extract_activations
from torch.utils.data import DataLoader

class TopologyMonitor:
    def __init__(self, layers: List[str], distance_metric: str = 'euclidean',
                 frequency: int = 5, max_dim: int = 2, max_samples: int = 500):
        self.layers = layers
        self.distance_metric = distance_metric
        self.frequency = frequency
        self.max_dim = max_dim
        self.max_samples = max_samples
        
        self.analyzer = PersistenceAnalyzer(max_dim=max_dim)
        self.snapshots = []
        self.epochs = []
        
    def should_capture(self, epoch: int) -> bool:
        return (epoch + 1) % self.frequency == 0
    
    def capture_snapshot(self, model: nn.Module, epoch: int, dataloader: DataLoader):
        if not self.should_capture(epoch):
            return
            
        model.eval()
        with torch.no_grad():
            activations = extract_activations(model, dataloader, self.layers, 
                                            max_batches=self.max_samples // dataloader.batch_size)
        
        snapshot = {'epoch': epoch, 'layers': {}}
        
        for layer_name, layer_acts in activations.items():
            if len(layer_acts) > self.max_samples:
                indices = torch.randperm(len(layer_acts))[:self.max_samples]
                layer_acts = layer_acts[indices]
            
            dist_matrix = compute_distances(layer_acts, self.distance_metric)
            analysis = self.analyzer.analyze(dist_matrix)
            
            snapshot['layers'][layer_name] = {
                'distance_matrix': dist_matrix,
                'analysis': analysis,
                'activations_shape': layer_acts.shape
            }
        
        self.snapshots.append(snapshot)
        self.epochs.append(epoch)
        
        print(f"  Topology captured at epoch {epoch+1}")
    
    def get_evolution_data(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {'epochs': [], 'layers': {}}
        
        evolution = {'epochs': self.epochs.copy(), 'layers': {}}
        
        for layer_name in self.layers:
            layer_evolution = {
                'betti_numbers': [],
                'entropy': [],
                'feature_counts': [],
                'analyses': []
            }
            
            for snapshot in self.snapshots:
                if layer_name in snapshot['layers']:
                    analysis = snapshot['layers'][layer_name]['analysis']
                    layer_evolution['betti_numbers'].append(analysis['betti_numbers'])
                    layer_evolution['entropy'].append(analysis['entropy'])
                    layer_evolution['feature_counts'].append({
                        dim: len(intervals) for dim, intervals in analysis['intervals'].items()
                    })
                    layer_evolution['analyses'].append(analysis)
            
            evolution['layers'][layer_name] = layer_evolution
        
        return evolution
    
    def clear_history(self):
        self.snapshots = []
        self.epochs = []
        self.analyzer.results = []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {}
        
        evolution = self.get_evolution_data()
        summary = {
            'total_epochs_captured': len(self.epochs),
            'capture_frequency': self.frequency,
            'layers_monitored': list(evolution['layers'].keys()),
            'distance_metric': self.distance_metric
        }
        
        for layer_name, layer_data in evolution['layers'].items():
            if layer_data['betti_numbers']:
                final_betti = layer_data['betti_numbers'][-1]
                summary[f'{layer_name}_final_betti'] = final_betti
        
        return summary