import numpy as np
from typing import Dict, List, Tuple, Optional
from ripser import ripser
from scipy.spatial.distance import directed_hausdorff

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

def compare_persistence_diagrams(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    if len(dgm1) == 0 or len(dgm2) == 0:
        return float('inf')
    
    hausdorff_dist = max(
        directed_hausdorff(dgm1, dgm2)[0],
        directed_hausdorff(dgm2, dgm1)[0]
    )
    return hausdorff_dist

def compute_diagram_distance(dgm1: np.ndarray, dgm2: np.ndarray, method: str = 'hausdorff') -> float:
    if method == 'hausdorff':
        return compare_persistence_diagrams(dgm1, dgm2)
    elif method == 'bottleneck':
        try:
            from persim import bottleneck
            return bottleneck(dgm1, dgm2)
        except ImportError:
            return compare_persistence_diagrams(dgm1, dgm2)
    else:
        raise ValueError(f"Unknown distance method: {method}")

def extract_persistent_features(dgm: np.ndarray, min_lifetime: float = 0.1) -> np.ndarray:
    if len(dgm) == 0:
        return np.array([]).reshape(0, 2)
    
    lifespans = dgm[:, 1] - dgm[:, 0]
    persistent_mask = lifespans >= min_lifetime
    return dgm[persistent_mask]

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
    
    def compare_with_previous(self, current_analysis: Dict) -> Dict:
        if not self.results:
            return {}
        
        prev_analysis = self.results[-1]
        comparison = {}
        
        for dim in range(self.max_dim + 1):
            curr_dgm = current_analysis['intervals'].get(dim, np.array([]).reshape(0, 2))
            prev_dgm = prev_analysis['intervals'].get(dim, np.array([]).reshape(0, 2))
            
            comparison[dim] = {
                'distance': compute_diagram_distance(curr_dgm, prev_dgm),
                'betti_change': current_analysis['betti_numbers'].get(dim, 0) - prev_analysis['betti_numbers'].get(dim, 0),
                'entropy_change': current_analysis['entropy'].get(dim, 0) - prev_analysis['entropy'].get(dim, 0)
            }
        
        return comparison