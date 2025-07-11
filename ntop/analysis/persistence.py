import numpy as np
from typing import Dict, List, Tuple, Optional
from ripser import ripser

def compute_persistence(distance_matrix: np.ndarray, 
                       max_dim: int = 2) -> Dict[str, np.ndarray]:
    result = ripser(distance_matrix, maxdim=max_dim, distance_matrix=True)
    return result

def extract_betti_numbers(persistence_result: Dict) -> Dict[int, int]:
    betti = {}
    diagrams = persistence_result['dgms']
    
    for dim, diagram in enumerate(diagrams):
        if len(diagram) == 0:
            betti[dim] = 0
        else:
            infinite_features = np.isinf(diagram[:, 1])
            betti[dim] = np.sum(infinite_features)
    
    return betti

def get_persistence_intervals(persistence_result: Dict, 
                            dim: int = 1) -> np.ndarray:
    diagrams = persistence_result['dgms']
    if dim >= len(diagrams):
        return np.array([]).reshape(0, 2)
    
    diagram = diagrams[dim]
    finite_features = ~np.isinf(diagram[:, 1])
    return diagram[finite_features]

def compute_persistence_entropy(intervals: np.ndarray) -> float:
    if len(intervals) == 0:
        return 0.0
    
    lifespans = intervals[:, 1] - intervals[:, 0]
    lifespans = lifespans[lifespans > 0]
    
    if len(lifespans) == 0:
        return 0.0
    
    p = lifespans / lifespans.sum()
    return -np.sum(p * np.log(p + 1e-10))

def get_longest_lived_features(intervals: np.ndarray, 
                              n_features: int = 5) -> np.ndarray:
    if len(intervals) == 0:
        return np.array([]).reshape(0, 2)
    
    lifespans = intervals[:, 1] - intervals[:, 0]
    top_indices = np.argsort(lifespans)[-n_features:]
    return intervals[top_indices]

class PersistenceAnalyzer:
    def __init__(self, max_dim: int = 2):
        self.max_dim = max_dim
        self.results = []
    
    def analyze(self, distance_matrix: np.ndarray) -> Dict:
        persistence = compute_persistence(distance_matrix, self.max_dim)
        betti = extract_betti_numbers(persistence)
        
        analysis = {
            'persistence': persistence,
            'betti_numbers': betti,
            'intervals': {},
            'entropy': {}
        }
        
        for dim in range(self.max_dim + 1):
            intervals = get_persistence_intervals(persistence, dim)
            analysis['intervals'][dim] = intervals
            analysis['entropy'][dim] = compute_persistence_entropy(intervals)
        
        return analysis
    
    def add_snapshot(self, distance_matrix: np.ndarray, epoch: int = 0):
        analysis = self.analyze(distance_matrix)
        analysis['epoch'] = epoch
        self.results.append(analysis)
    
    def get_betti_evolution(self) -> Dict[int, List[int]]:
        evolution = {}
        for dim in range(self.max_dim + 1):
            evolution[dim] = [r['betti_numbers'].get(dim, 0) for r in self.results]
        return evolution