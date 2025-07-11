import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform

class EvolutionTracker:
    def __init__(self):
        self.timeline = []
        
    def add_snapshot(self, analysis_result: Dict, epoch: int):
        snapshot = {
            'epoch': epoch,
            'analysis': analysis_result
        }
        self.timeline.append(snapshot)
    
    def compute_betti_evolution(self) -> Dict[int, List[int]]:
        evolution = {}
        if not self.timeline:
            return evolution
            
        max_dim = max(max(snap['analysis']['betti_numbers'].keys()) for snap in self.timeline)
        
        for dim in range(max_dim + 1):
            evolution[dim] = []
            for snapshot in self.timeline:
                betti = snapshot['analysis']['betti_numbers'].get(dim, 0)
                evolution[dim].append(betti)
        
        return evolution
    
    def detect_critical_transitions(self, threshold: float = 0.5) -> List[Dict]:
        betti_evolution = self.compute_betti_evolution()
        transitions = []
        
        for dim, values in betti_evolution.items():
            if len(values) < 3:
                continue
                
            values_array = np.array(values)
            diff = np.abs(np.diff(values_array))
            
            if len(diff) == 0:
                continue
                
            mean_change = np.mean(diff)
            std_change = np.std(diff)
            
            if std_change == 0:
                continue
                
            significant_changes = diff > (mean_change + threshold * std_change)
            
            change_indices = np.where(significant_changes)[0]
            
            for idx in change_indices:
                epoch_idx = idx + 1
                if epoch_idx < len(self.timeline):
                    transitions.append({
                        'epoch': self.timeline[epoch_idx]['epoch'],
                        'dimension': dim,
                        'change_magnitude': diff[idx],
                        'before_value': values[idx],
                        'after_value': values[idx + 1]
                    })
        
        return sorted(transitions, key=lambda x: x['epoch'])
    
    def compute_feature_stability(self) -> Dict[int, float]:
        stability = {}
        betti_evolution = self.compute_betti_evolution()
        
        for dim, values in betti_evolution.items():
            if len(values) <= 1:
                stability[dim] = 1.0
                continue
                
            values_array = np.array(values)
            if len(np.unique(values_array)) == 1:
                stability[dim] = 1.0
            else:
                cv = np.std(values_array) / (np.mean(values_array) + 1e-8)
                stability[dim] = 1.0 / (1.0 + cv)
        
        return stability
    
    def get_learning_phases(self, min_phase_length: int = 3) -> List[Dict]:
        if len(self.timeline) < min_phase_length * 2:
            return []
            
        transitions = self.detect_critical_transitions()
        transition_epochs = [t['epoch'] for t in transitions]
        
        phases = []
        start_epoch = self.timeline[0]['epoch'] if self.timeline else 0
        
        for transition_epoch in transition_epochs:
            if transition_epoch - start_epoch >= min_phase_length:
                phases.append({
                    'start_epoch': start_epoch,
                    'end_epoch': transition_epoch,
                    'length': transition_epoch - start_epoch
                })
                start_epoch = transition_epoch
        
        if self.timeline and self.timeline[-1]['epoch'] - start_epoch >= min_phase_length:
            phases.append({
                'start_epoch': start_epoch,
                'end_epoch': self.timeline[-1]['epoch'],
                'length': self.timeline[-1]['epoch'] - start_epoch
            })
        
        return phases

def compute_topology_distance(analysis1: Dict, analysis2: Dict, method: str = 'betti') -> float:
    if method == 'betti':
        betti1 = analysis1['betti_numbers']
        betti2 = analysis2['betti_numbers']
        
        all_dims = set(betti1.keys()) | set(betti2.keys())
        diff = 0.0
        
        for dim in all_dims:
            val1 = betti1.get(dim, 0)
            val2 = betti2.get(dim, 0)
            diff += abs(val1 - val2)
        
        return diff
    
    elif method == 'entropy':
        entropy1 = analysis1['entropy']
        entropy2 = analysis2['entropy']
        
        all_dims = set(entropy1.keys()) | set(entropy2.keys())
        diff = 0.0
        
        for dim in all_dims:
            val1 = entropy1.get(dim, 0)
            val2 = entropy2.get(dim, 0)
            diff += abs(val1 - val2)
        
        return diff
    
    else:
        raise ValueError(f"Unknown distance method: {method}")

def detect_phase_transitions(betti_evolution: Dict[int, List[int]], 
                           sensitivity: float = 1.0) -> List[int]:
    all_transitions = []
    
    for dim, values in betti_evolution.items():
        if len(values) < 3:
            continue
            
        values_array = np.array(values)
        diff = np.abs(np.diff(values_array))
        
        if len(diff) == 0 or np.std(diff) == 0:
            continue
            
        threshold = np.mean(diff) + sensitivity * np.std(diff)
        peaks, _ = find_peaks(diff, height=threshold)
        
        all_transitions.extend(peaks + 1)
    
    return sorted(list(set(all_transitions)))

def compute_persistence_entropy_evolution(tracker: EvolutionTracker) -> Dict[int, List[float]]:
    entropy_evolution = {}
    
    if not tracker.timeline:
        return entropy_evolution
        
    max_dim = max(max(snap['analysis']['entropy'].keys()) for snap in tracker.timeline)
    
    for dim in range(max_dim + 1):
        entropy_evolution[dim] = []
        for snapshot in tracker.timeline:
            entropy = snapshot['analysis']['entropy'].get(dim, 0.0)
            entropy_evolution[dim].append(entropy)
    
    return entropy_evolution

def compute_topology_trajectory(evolution_data: Dict) -> Dict[str, np.ndarray]:
    trajectories = {}
    
    for layer_name, layer_data in evolution_data['layers'].items():
        betti_numbers = layer_data['betti_numbers']
        
        if not betti_numbers:
            continue
            
        trajectory_matrix = []
        
        for betti_dict in betti_numbers:
            max_dim = max(betti_dict.keys()) if betti_dict else 0
            row = [betti_dict.get(dim, 0) for dim in range(max_dim + 1)]
            trajectory_matrix.append(row)
        
        trajectories[layer_name] = np.array(trajectory_matrix)
    
    return trajectories