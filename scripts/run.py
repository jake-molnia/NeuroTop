#!/usr/bin/env python3
"""
Unified flow experiment runner.
Complete pipeline: train → analyze → visualize → report.
Streamlined to avoid redundant computation.
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent))

import modules as ntf
from modules.storage import TemporalExperimentManager
from modules.utils import Timer, set_seed, get_device
from modules.training import FlowAwareTrainer

logger = logging.getLogger(__name__)

class FlowExperimentRunner:
    """Unified runner for complete flow analysis experiments."""
    
    def __init__(self, config_path: str):
        self.config = ntf.load_config(config_path)
        self.experiment_id = self._generate_experiment_id()
        self.temporal_manager = None
        self.exp_logger = None
        
        # Setup
        set_seed(self.config.get('seed', 42))
        self.device = get_device()
        
        # Initialize storage if temporal analysis enabled
        temporal_config = self.config.get('temporal_analysis', {})
        if temporal_config.get('enabled', False):
            storage_config = self.config.get('storage', {})
            cache_dir = storage_config.get('local_cache_dir', './flow_cache')
            max_size_gb = storage_config.get('max_cache_size_gb', 20.0)
            
            self.temporal_manager = TemporalExperimentManager(cache_dir, max_size_gb)
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self.config.get('model', {}).get('name', 'model')
        dataset_name = self.config.get('dataset', {}).get('name', 'dataset')
        return f"{dataset_name}_{model_name}_{timestamp}"
    
    def run_single_experiment(self) -> Dict[str, Any]:
        """Run complete single model experiment."""
        
        logger.info(f"Starting flow experiment: {self.experiment_id}")
        
        # Initialize experiment logging
        self._setup_experiment_logging()
        
        try:
            # Combined training and analysis phase
            with Timer("Training and Analysis Phase"):
                trained_model, flow_results = self._run_training_and_analysis()
            
            # Visualization phase
            with Timer("Visualization Phase"):
                visualizations = self._run_visualization_phase(flow_results)
            
            # Final reporting
            summary = self._generate_experiment_summary(flow_results, visualizations)
            
            logger.info("Flow experiment completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            if self.exp_logger:
                self.exp_logger.finish()
    
    def _setup_experiment_logging(self):
        """Setup wandb experiment logging."""
        
        logging_config = self.config.get('logging', {})
        project_name = logging_config.get('project', 'neural-topology-flow')
        tags = ['flow-analysis'] + self.config.get('tags', [])
        
        self.exp_logger = ntf.init_experiment(
            project_name=project_name,
            experiment_name=self.experiment_id,
            config=self.config,
            tags=tags
        )
    
    def _run_training_and_analysis(self) -> tuple[torch.nn.Module, Dict[str, Any]]:
        """Run training with integrated analysis - no redundant computation."""
        
        # Load data once
        train_loader, test_loader = ntf.get_dataloaders(self.config)
        
        # Create model
        model = ntf.get_model(self.config['model'])
        
        # Start temporal experiment if enabled
        if self.temporal_manager:
            self.temporal_manager.start_experiment(self.experiment_id, self.config)
        
        # Train model (with temporal capture if enabled)
        if self.config.get('temporal_analysis', {}).get('enabled', False):
            trainer = FlowAwareTrainer(
                model, self.config, self.temporal_manager, self.experiment_id
            )
            trained_model = trainer.train_with_flow_monitoring(
                train_loader, test_loader, self.device, self.exp_logger
            )
            
            # Use captured temporal data for analysis
            available_epochs = self.temporal_manager.get_experiment_summary(self.experiment_id)['available_epochs']
            if available_epochs:
                flow_results = self._analyze_temporal_data(available_epochs)
            else:
                logger.warning("No temporal data captured, running final analysis")
                flow_results = self._analyze_final_state(trained_model, test_loader)
        else:
            # Standard training
            trained_model = ntf.training.train_and_evaluate(
                self.config, model, train_loader, test_loader, self.device, self.exp_logger
            )
            
            # Single point analysis on trained model
            flow_results = self._analyze_final_state(trained_model, test_loader)
        
        # Save model
        if self.exp_logger:
            model_name = self.config['model'].get('name', 'model')
            self.exp_logger.save_model_artifact(trained_model, model_name)
        
        return trained_model, flow_results
    
    def _analyze_temporal_data(self, epochs: List[int]) -> Dict[str, Any]:
        """Analyze captured temporal data."""
        
        logger.info(f"Analyzing temporal data for {len(epochs)} epochs")
        
        temporal_results = {}
        flow_analyses = {}
        
        # Load all cached data
        for epoch in epochs:
            try:
                # Load flow analysis from cache
                flow_data = self.temporal_manager.flow_store.load_flow_analysis(
                    self.experiment_id, epoch
                )
                temporal_results[epoch] = flow_data
                
                # Load activations for inter-layer flow analysis
                activations = self.temporal_manager.activation_store.load_temporal_activations(
                    self.experiment_id, epoch
                )
                
                # Compute inter-layer flows once per epoch
                layer_names = list(activations.keys())
                epoch_flows = {}
                
                for i in range(len(layer_names) - 1):
                    source_layer = layer_names[i]
                    target_layer = layer_names[i + 1]
                    
                    flow_analysis = ntf.topology.compute_information_flow_between_layers(
                        activations[source_layer], activations[target_layer]
                    )
                    
                    epoch_flows[f"{source_layer}_to_{target_layer}"] = flow_analysis
                
                flow_analyses[epoch] = epoch_flows
                
            except Exception as e:
                logger.warning(f"Could not load data for epoch {epoch}: {e}")
        
        # Create temporal tracker for evolution analysis
        tracker = ntf.topology.TemporalTopologyTracker()
        tracker.snapshots = [temporal_results[epoch] for epoch in sorted(temporal_results.keys())]
        tracker.timestamps = sorted(temporal_results.keys())
        
        evolution = tracker.compute_flow_evolution()
        
        return {
            'temporal_results': temporal_results,
            'flow_analyses': flow_analyses,
            'evolution': evolution,
            'epochs_analyzed': sorted(temporal_results.keys())
        }
    
    def _analyze_final_state(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Analyze final trained model state - extract activations once and reuse."""
        
        logger.info("Running final state analysis")
        
        # Extract configuration
        analysis_config = self.config['analysis']['activation_extraction']
        layers = analysis_config['layers']
        max_samples = analysis_config.get('max_samples')
        normalize_method = analysis_config.get('normalization_method', 'standard')
        
        # Extract activations once
        raw_activations = ntf.extract_activations(
            model, test_loader, layers, max_samples, self.device
        )
        
        # Normalize once
        if normalize_method != 'none':
            activations = ntf.normalize_activations(raw_activations, normalize_method)
        else:
            activations = raw_activations
        
        # Concatenate for global analysis
        concatenated_acts, global_to_local_map = ntf.concatenate_layer_activations(activations, layers)
        
        # Determine analysis type
        flow_config = self.config.get('flow_analysis', {})
        use_wasserstein = flow_config.get('wasserstein_enabled', False)
        
        # Compute distance matrix once
        if use_wasserstein:
            dist_matrix = ntf.topology.compute_wasserstein_distance_matrix(concatenated_acts)
        else:
            dist_matrix = ntf.topology.compute_distance_matrix(concatenated_acts)
        
        # Reuse distance matrix for all topology analysis
        persistence = ntf.topology.compute_persistence_diagrams(dist_matrix)
        scales = ntf.topology.get_filtration_scales(dist_matrix)
        criticality = ntf.topology.classify_neuron_criticality(dist_matrix, scales)
        energies = ntf.topology.compute_total_energy_functional(concatenated_acts)
        
        # Inter-layer flow analysis - reuse activations
        flow_analyses = {}
        for i in range(len(layers) - 1):
            source_layer = layers[i]
            target_layer = layers[i + 1]
            
            flow_analysis = ntf.topology.compute_information_flow_between_layers(
                activations[source_layer], activations[target_layer]
            )
            
            flow_analyses[f"{source_layer}_to_{target_layer}"] = flow_analysis
        
        return {
            'single_point': {
                'distance_matrix': dist_matrix,
                'persistence': persistence,
                'criticality': criticality,
                'energies': energies,
                'global_to_local_map': global_to_local_map
            },
            'flow_analyses': flow_analyses,
            'activations': activations
        }
    
    def _run_visualization_phase(self, flow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualizations from computed results."""
        
        logger.info("Generating flow visualizations")
        
        plots = {}
        
        # Temporal plots if we have temporal data
        if 'temporal_results' in flow_results:
            plots['flow_evolution'] = ntf.visualization.plot_wasserstein_flow_evolution(
                flow_results['temporal_results']
            )
        
        # Single point plots
        if 'single_point' in flow_results:
            single_data = flow_results['single_point']
            
            plots['persistence'] = ntf.visualization.plot_persistence_diagram(
                single_data['persistence']
            )
            plots['criticality'] = ntf.visualization.plot_criticality_distribution(
                single_data['criticality']
            )
        
        # Flow analysis plots
        if 'flow_analyses' in flow_results:
            if isinstance(flow_results['flow_analyses'], dict):
                flow_data = flow_results['flow_analyses']
            else:
                # Multiple epochs - use latest
                epochs = sorted(flow_results['flow_analyses'].keys())
                flow_data = flow_results['flow_analyses'][epochs[-1]]
            
            plots['bottlenecks'] = ntf.visualization.plot_information_bottlenecks(flow_data)
            
            # Transport matrix for first flow
            first_flow = list(flow_data.values())[0]
            plots['transport_matrix'] = ntf.visualization.plot_transport_matrix_heatmap(
                first_flow['transport_plan']
            )
        
        # Log all plots to wandb
        if self.exp_logger:
            for plot_name, figure in plots.items():
                self.exp_logger.log_figure(figure, f'flow/{plot_name}')
        
        return plots
    
    def _generate_experiment_summary(self, flow_results: Dict[str, Any], visualizations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate experiment summary."""
        
        summary = {
            'experiment_id': self.experiment_id,
            'status': 'completed',
            'visualizations': list(visualizations.keys())
        }
        
        # Add metrics
        if 'single_point' in flow_results:
            energies = flow_results['single_point']['energies']
            summary['final_energy'] = energies['total']
        
        if 'temporal_results' in flow_results:
            summary['epochs_analyzed'] = len(flow_results['temporal_results'])
            
            if 'evolution' in flow_results:
                evolution = flow_results['evolution']
                summary['energy_trend'] = evolution['energy_evolution'][-1] - evolution['energy_evolution'][0]
        
        # Cache statistics
        if self.temporal_manager:
            cache_stats = self.temporal_manager.get_cache_stats()
            summary['cache_stats'] = cache_stats
        
        # Log summary to wandb
        if self.exp_logger:
            summary_metrics = {'experiment/status': 'completed'}
            
            if 'final_energy' in summary:
                summary_metrics['experiment/final_energy'] = summary['final_energy']
            if 'epochs_analyzed' in summary:
                summary_metrics['experiment/epochs_analyzed'] = summary['epochs_analyzed']
            
            self.exp_logger.log_metrics(summary_metrics)
        
        logger.info(f"Experiment summary generated for {self.experiment_id}")
        return summary

def main():
    parser = argparse.ArgumentParser(description="Unified flow experiment runner")
    parser.add_argument('config', help="Path to configuration file")
    parser.add_argument('--name', help="Override experiment name")
    parser.add_argument('--tags', nargs='*', help="Additional tags")
    parser.add_argument('--dry-run', action='store_true', help="Validate config without running")
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config = ntf.load_config(args.config)
    
    if args.dry_run:
        ntf.validate_config(config)
        logger.info("Configuration validated successfully")
        return
    
    # Add custom tags if provided
    if args.tags:
        config.setdefault('tags', []).extend(args.tags)
    
    # Override experiment name if provided
    if args.name:
        config['experiment_name'] = args.name
    
    # Create and run experiment
    runner = FlowExperimentRunner(args.config)
    
    try:
        result = runner.run_single_experiment()
        print(f"\nExperiment completed: {result['experiment_id']}")
            
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()