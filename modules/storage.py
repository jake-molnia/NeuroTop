"""
Local storage and caching for temporal neural topology analysis.
Efficient HDF5/PyTorch storage with automatic cleanup.
"""

import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
import json
import shutil
from collections import defaultdict

logger = logging.getLogger(__name__)

class TemporalActivationStore:
    """Efficient local storage for temporal activation data."""
    
    def __init__(
        self, 
        cache_dir: str = "./flow_cache",
        max_size_gb: float = 20.0,
        compression_level: int = 6
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_gb = max_size_gb
        self.compression_level = compression_level
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'experiments': {},
            'total_size_mb': 0,
            'last_cleanup': time.time()
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def store_temporal_activations(
        self,
        experiment_id: str,
        epoch: int,
        activations: Dict[str, torch.Tensor],
        model_state: Optional[Dict] = None
    ) -> str:
        """Store activations for specific epoch."""
        exp_dir = self.cache_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Store activations as HDF5
        h5_file = exp_dir / f"activations_epoch_{epoch:04d}.h5"
        
        with h5py.File(h5_file, 'w') as f:
            for layer_name, acts in activations.items():
                f.create_dataset(
                    layer_name, 
                    data=acts.cpu().numpy(),
                    compression='gzip',
                    compression_opts=self.compression_level
                )
            
            # Store metadata
            f.attrs['epoch'] = epoch
            f.attrs['timestamp'] = time.time()
            f.attrs['experiment_id'] = experiment_id
        
        # Store model state if provided
        if model_state is not None:
            torch.save(model_state, exp_dir / f"model_epoch_{epoch:04d}.pt")
        
        # Update metadata
        file_size_mb = h5_file.stat().st_size / (1024 * 1024)
        if experiment_id not in self.metadata['experiments']:
            self.metadata['experiments'][experiment_id] = {
                'epochs': [],
                'total_size_mb': 0,
                'created': time.time()
            }
        
        self.metadata['experiments'][experiment_id]['epochs'].append(epoch)
        self.metadata['experiments'][experiment_id]['total_size_mb'] += file_size_mb
        self.metadata['total_size_mb'] += file_size_mb
        
        self._save_metadata()
        self._check_cache_size()
        
        logger.debug(f"Stored activations for {experiment_id} epoch {epoch} ({file_size_mb:.1f}MB)")
        return str(h5_file)
    
    def load_temporal_activations(
        self,
        experiment_id: str,
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """Load activations for specific epoch."""
        h5_file = self.cache_dir / experiment_id / f"activations_epoch_{epoch:04d}.h5"
        
        if not h5_file.exists():
            raise FileNotFoundError(f"No activations found for {experiment_id} epoch {epoch}")
        
        activations = {}
        with h5py.File(h5_file, 'r') as f:
            for layer_name in f.keys():
                activations[layer_name] = torch.from_numpy(f[layer_name][:]).float()
        
        logger.debug(f"Loaded activations for {experiment_id} epoch {epoch}")
        return activations
    
    def get_available_epochs(self, experiment_id: str) -> List[int]:
        """Get list of available epochs for experiment."""
        if experiment_id in self.metadata['experiments']:
            return sorted(self.metadata['experiments'][experiment_id]['epochs'])
        return []
    
    def _check_cache_size(self):
        """Check cache size and cleanup if necessary."""
        total_size_gb = self.metadata['total_size_mb'] / 1024
        
        if total_size_gb > self.max_size_gb:
            logger.info(f"Cache size ({total_size_gb:.1f}GB) exceeds limit ({self.max_size_gb}GB), cleaning up...")
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """LRU cleanup of cache."""
        # Sort experiments by creation time (oldest first)
        exp_items = [(exp_id, data) for exp_id, data in self.metadata['experiments'].items()]
        exp_items.sort(key=lambda x: x[1]['created'])
        
        target_size_gb = self.max_size_gb * 0.8  # Clean to 80% of limit
        current_size_gb = self.metadata['total_size_mb'] / 1024
        
        for exp_id, exp_data in exp_items:
            if current_size_gb <= target_size_gb:
                break
            
            # Remove entire experiment
            exp_dir = self.cache_dir / exp_id
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
                current_size_gb -= exp_data['total_size_mb'] / 1024
                del self.metadata['experiments'][exp_id]
                logger.info(f"Removed experiment {exp_id} from cache")
        
        # Update total size
        self.metadata['total_size_mb'] = current_size_gb * 1024
        self.metadata['last_cleanup'] = time.time()
        self._save_metadata()

class FlowArtifactStore:
    """Store computed flow analysis results."""
    
    def __init__(self, base_store: TemporalActivationStore):
        self.base_store = base_store
    
    def store_flow_analysis(
        self,
        experiment_id: str,
        epoch: int,
        analysis_results: Dict[str, Any]
    ):
        """Store flow analysis results."""
        exp_dir = self.base_store.cache_dir / experiment_id
        flow_dir = exp_dir / "flow_analysis"
        flow_dir.mkdir(exist_ok=True)
        
        # Store different types of results
        epoch_file = flow_dir / f"flow_epoch_{epoch:04d}.pt"
        
        # Prepare data for storage
        storage_data = {}
        
        for key, value in analysis_results.items():
            if isinstance(value, np.ndarray):
                storage_data[key] = torch.from_numpy(value)
            elif isinstance(value, pd.DataFrame):
                storage_data[key] = value.to_dict()
            elif isinstance(value, dict):
                storage_data[key] = value
            else:
                storage_data[key] = value
        
        torch.save(storage_data, epoch_file)
        logger.debug(f"Stored flow analysis for {experiment_id} epoch {epoch}")
    
    def load_flow_analysis(
        self,
        experiment_id: str,
        epoch: int
    ) -> Dict[str, Any]:
        """Load flow analysis results."""
        exp_dir = self.base_store.cache_dir / experiment_id
        flow_dir = exp_dir / "flow_analysis"
        epoch_file = flow_dir / f"flow_epoch_{epoch:04d}.pt"
        
        if not epoch_file.exists():
            raise FileNotFoundError(f"No flow analysis found for {experiment_id} epoch {epoch}")
        
        # The fix is to add `weights_only=False` to the torch.load call.
        # This tells PyTorch that you trust the source of the file and allows it
        # to load complex Python objects like NumPy arrays and Pandas DataFrames.
        storage_data = torch.load(epoch_file, weights_only=False)
        
        # Convert back to original types
        results = {}
        for key, value in storage_data.items():
            # This part of your code seems to have a logic issue for converting
            # DataFrames back. A more robust way is to check the type of `value`,
            # which is a dictionary after being saved.
            if isinstance(value, dict) and key.endswith('_dataframe'):
                results[key.replace('_dataframe', '')] = pd.DataFrame.from_dict(value)
            elif isinstance(value, torch.Tensor):
                results[key] = value
            else:
                # Handle other potential types like dictionaries that are not dataframes
                results[key] = value
        
        return results

class TemporalExperimentManager:
    """Manage temporal experiments with unified storage."""
    
    def __init__(
        self,
        cache_dir: str = "./flow_cache",
        max_size_gb: float = 20.0
    ):
        self.activation_store = TemporalActivationStore(cache_dir, max_size_gb)
        self.flow_store = FlowArtifactStore(self.activation_store)
        self.active_experiments = {}
    
    def start_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any]
    ) -> str:
        """Start a new temporal experiment."""
        # Store experiment config
        exp_dir = self.activation_store.cache_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.active_experiments[experiment_id] = {
            'config': config,
            'start_time': time.time(),
            'epochs_captured': []
        }
        
        logger.info(f"Started temporal experiment: {experiment_id}")
        return experiment_id
    
    def capture_epoch(
        self,
        experiment_id: str,
        epoch: int,
        activations: Dict[str, torch.Tensor],
        model_state: Optional[Dict] = None,
        run_flow_analysis: bool = True
    ):
        """Capture activations and optionally run flow analysis for epoch."""
        # Store activations
        self.activation_store.store_temporal_activations(
            experiment_id, epoch, activations, model_state
        )
        
        # Run flow analysis if requested
        if run_flow_analysis:
            from .topology import (
                compute_distance_matrix, 
                compute_wasserstein_distance_matrix,
                compute_persistence_diagrams,
                classify_neuron_criticality,
                get_filtration_scales,
                compute_total_energy_functional
            )
            from .activations import concatenate_layer_activations
            
            # Get config for this experiment
            config = self.active_experiments[experiment_id]['config']
            layers = config.get('analysis', {}).get('activation_extraction', {}).get('layers', [])
            
            # Concatenate activations
            concatenated_acts, global_to_local_map = concatenate_layer_activations(activations, layers)
            
            # Compute analysis
            use_wasserstein = config.get('flow_analysis', {}).get('wasserstein_enabled', False)
            
            if use_wasserstein:
                dist_matrix = compute_wasserstein_distance_matrix(concatenated_acts)
            else:
                dist_matrix = compute_distance_matrix(concatenated_acts)
            
            persistence = compute_persistence_diagrams(dist_matrix)
            scales = get_filtration_scales(dist_matrix)
            criticality = classify_neuron_criticality(dist_matrix, scales)
            energies = compute_total_energy_functional(concatenated_acts)
            
            analysis_results = {
                'distance_matrix': dist_matrix,
                'persistence': persistence,
                'criticality_dataframe': criticality,  # Mark as DataFrame
                'energies': energies,
                'global_to_local_map': global_to_local_map
            }
            
            self.flow_store.store_flow_analysis(experiment_id, epoch, analysis_results)
        
        # Update experiment tracking
        if experiment_id in self.active_experiments:
            self.active_experiments[experiment_id]['epochs_captured'].append(epoch)
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary of experiment."""
        if experiment_id not in self.activation_store.metadata['experiments']:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_metadata = self.activation_store.metadata['experiments'][experiment_id]
        available_epochs = self.activation_store.get_available_epochs(experiment_id)
        
        # Load config if available
        config_file = self.activation_store.cache_dir / experiment_id / "config.json"
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        return {
            'experiment_id': experiment_id,
            'config': config,
            'available_epochs': available_epochs,
            'total_size_mb': exp_metadata['total_size_mb'],
            'created': exp_metadata['created'],
            'num_epochs': len(available_epochs)
        }
    
    def cleanup_experiment(self, experiment_id: str):
        """Clean up specific experiment."""
        exp_dir = self.activation_store.cache_dir / experiment_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            
        if experiment_id in self.activation_store.metadata['experiments']:
            exp_size = self.activation_store.metadata['experiments'][experiment_id]['total_size_mb']
            self.activation_store.metadata['total_size_mb'] -= exp_size
            del self.activation_store.metadata['experiments'][experiment_id]
            self.activation_store._save_metadata()
        
        if experiment_id in self.active_experiments:
            del self.active_experiments[experiment_id]
        
        logger.info(f"Cleaned up experiment: {experiment_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size_gb = self.activation_store.metadata['total_size_mb'] / 1024
        num_experiments = len(self.activation_store.metadata['experiments'])
        
        # Calculate oldest/newest experiments
        if num_experiments > 0:
            creation_times = [exp['created'] for exp in self.activation_store.metadata['experiments'].values()]
            oldest = min(creation_times)
            newest = max(creation_times)
        else:
            oldest = newest = time.time()
        
        return {
            'total_size_gb': total_size_gb,
            'max_size_gb': self.activation_store.max_size_gb,
            'utilization_pct': (total_size_gb / self.activation_store.max_size_gb) * 100,
            'num_experiments': num_experiments,
            'oldest_experiment_age_hours': (time.time() - oldest) / 3600,
            'cache_directory': str(self.activation_store.cache_dir)
        }