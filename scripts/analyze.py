#!/usr/bin/env python3
"""
Pure topological analysis script.
Loads trained models and performs topological analysis of their representations.
"""

import sys
import torch
import argparse
import logging
import wandb
from pathlib import Path
from typing import Dict, Any, Optional
sys.path.append(str(Path(__file__).parent.parent))

import modules as ntf
from modules.utils import Timer
from modules.visualization import create_experiment_summary_plots

logger = logging.getLogger(__name__)

def load_model_from_wandb(
    project_name: str,
    run_id: str,
    artifact_name: str,
    model_config: Dict[str, Any],
    device
) -> ntf.models.BaseModel:
    """Load a trained model from wandb artifacts."""
    
    # Initialize wandb API
    api = wandb.Api()
    run = api.run(f"{project_name}/{run_id}")
    
    # Get artifact using the API directly (not from run)
    try:
        # Try getting the artifact by full name first
        full_artifact_name = f"{project_name}/{artifact_name}:latest"
        artifact = api.artifact(full_artifact_name)
    except:
        # Fallback: search through run's artifacts
        artifacts = run.logged_artifacts()
        artifact = None
        for art in artifacts:
            if artifact_name in art.name:
                artifact = art
                break
        
        if artifact is None:
            # List available artifacts for debugging
            available = [art.name for art in artifacts]
            raise ValueError(f"Artifact '{artifact_name}' not found. Available artifacts: {available}")
    
    # Download artifact
    artifact_dir = artifact.download()
    
    # Load model
    model = ntf.get_model(model_config)
    
    # Find the model file (it might have a different name)
    model_files = list(Path(artifact_dir).glob("*.pth"))
    if not model_files:
        raise ValueError(f"No .pth files found in artifact {artifact_name}")
    
    model_path = model_files[0]  # Use the first .pth file found
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from wandb: {artifact_name}")
    return model

def analyze_model_topology(
    model: ntf.models.BaseModel,
    config: Dict[str, Any],
    exp_logger: Optional[ntf.ExperimentLogger] = None
) -> Dict[str, Any]:
    """Perform complete topological analysis on a trained model."""
    
    device = model.get_device()
    analysis_config = config['analysis']
    
    # Load data for activation extraction
    _, test_loader = ntf.get_dataloaders(config)
    
    # Extract activations
    logger.info("Extracting activations...")
    with Timer("Activation extraction"):
        raw_activations = ntf.extract_activations(
            model=model,
            data_loader=test_loader,
            layers_to_hook=analysis_config['activation_extraction']['layers'],
            max_samples=analysis_config['activation_extraction'].get('max_samples'),
            device=device
        )
    
    # Normalize activations
    normalize_method = analysis_config['activation_extraction'].get('normalization_method', 'standard')
    if normalize_method != 'none':
        logger.info(f"Normalizing activations using {normalize_method}")
        activations = ntf.normalize_activations(raw_activations, normalize_method)
    else:
        activations = raw_activations
    
    # Concatenate layers for global analysis
    logger.info("Concatenating layer activations...")
    concatenated_acts, global_to_local_map = ntf.concatenate_layer_activations(
        activations, analysis_config['activation_extraction']['layers']
    )
    
    # Compute distance matrix
    logger.info("Computing distance matrix...")
    with Timer("Distance matrix computation"):
        distance_matrix = ntf.compute_distance_matrix(concatenated_acts)
    
    # Compute topological features
    logger.info("Computing persistence diagrams...")
    with Timer("Persistence computation"):
        persistence_diagrams = ntf.compute_persistence_diagrams(
            distance_matrix, max_dimension=2
        )
    
    # Classify neuron criticality
    logger.info("Classifying neuron criticality...")
    with Timer("Criticality classification"):
        scales = ntf.topology.get_filtration_scales(distance_matrix, num_scales=15)
        criticality_df = ntf.classify_neuron_criticality(distance_matrix, scales)
    
    # Create visualizations if enabled
    plots = {}
    viz_config = analysis_config.get('visualizations', {})
    
    if viz_config.get('tsne_plot', False):
        logger.info("Creating t-SNE plot...")
        plots['tsne'] = ntf.visualization.plot_tsne_embedding(
            concatenated_acts.numpy(), title="Neuron t-SNE Embedding"
        )
    
    if viz_config.get('distance_matrix', False):
        logger.info("Creating distance matrix plot...")
        plots['distance_matrix'] = ntf.visualization.plot_distance_matrix(
            distance_matrix, title="Neuron Distance Matrix"
        )
    
    if viz_config.get('persistence_diagram', False):
        logger.info("Creating persistence diagram...")
        plots['persistence'] = ntf.visualization.plot_persistence_diagram(
            persistence_diagrams, title="Persistence Diagram"
        )
    
    # Criticality plots
    plots['criticality_dist'] = ntf.visualization.plot_criticality_distribution(
        criticality_df, title="Neuron Criticality Distribution"
    )
    
    plots['degree_evolution'] = ntf.visualization.plot_degree_evolution(
        criticality_df, scales, title="Degree Evolution Across Scales"
    )
    
    # Log to wandb if logger provided
    if exp_logger:
        # Log criticality metrics
        criticality_counts = criticality_df['classification'].value_counts()
        metrics = {
            f'analysis/criticality_{k.lower()}': v 
            for k, v in criticality_counts.items()
        }
        metrics['analysis/total_neurons'] = len(criticality_df)
        exp_logger.log_metrics(metrics)
        
        # Log data artifacts
        exp_logger.log_dataframe(criticality_df, 'neuron_criticality')
        exp_logger.save_tensor_artifact(
            concatenated_acts, 'activations',
            metadata={'normalization': normalize_method}
        )
        
        # Log plots
        for plot_name, figure in plots.items():
            exp_logger.log_figure(figure, f'analysis/{plot_name}')
    
    # Return analysis results
    results = {
        'activations': concatenated_acts,
        'distance_matrix': distance_matrix,
        'persistence_diagrams': persistence_diagrams,
        'criticality_df': criticality_df,
        'global_to_local_map': global_to_local_map,
        'plots': plots,
        'scales': scales
    }
    
    logger.info("Topological analysis completed")
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze trained models topologically")
    parser.add_argument('config', help="Path to analysis configuration file")
    parser.add_argument('--model-run', required=True, help="Wandb run ID containing the trained model")
    parser.add_argument('--model-artifact', required=True, help="Name of the model artifact")
    parser.add_argument('--project', default='neural-topology-training', help="Wandb project name for model")
    parser.add_argument('--name', help="Override experiment name")
    parser.add_argument('--tags', nargs='*', help="Additional tags")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ntf.load_config(args.config)
    
    # Setup
    ntf.set_seed(config.get('seed', 42))
    device = ntf.get_device()
    
    # Create experiment name
    model_name = config['model'].get('name', 'model')
    exp_name = args.name or f"analysis_{model_name}_{args.model_run[:8]}"
    
    # Initialize experiment logger
    tags = ['analysis'] + config.get('tags', [])
    if args.tags:
        tags.extend(args.tags)
    
    exp_logger = ntf.init_experiment(
        project_name="neural-topology-analysis",
        experiment_name=exp_name,
        config=config,
        tags=tags,
        group='topological_analysis'
    )
    
    try:
        # Load trained model
        logger.info(f"Loading model from run: {args.model_run}")
        model = load_model_from_wandb(
            project_name=args.project,
            run_id=args.model_run,
            artifact_name=args.model_artifact,
            model_config=config['model'],
            device=device
        )
        
        # Perform analysis
        results = analyze_model_topology(model, config, exp_logger)
        
        # Save analysis results as artifact
        exp_logger.save_results_artifact(
            results['criticality_df'], 
            f"analysis_{model_name}",
            metadata={
                'model_run': args.model_run,
                'model_artifact': args.model_artifact,
                'total_neurons': len(results['criticality_df'])
            }
        )
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    finally:
        exp_logger.finish()

if __name__ == '__main__':
    main()