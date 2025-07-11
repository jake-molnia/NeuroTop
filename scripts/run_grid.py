#!/usr/bin/env python3
"""
Simple grid runner: Train models + Run analysis
Just chains training and analysis for multiple models.
"""

import sys
import argparse
import logging
import wandb
import yaml
from pathlib import Path
from typing import Dict, Any, List
sys.path.append(str(Path(__file__).parent.parent))

import modules as ntf
def train_single_model(config, exp_logger=None):
    """Train a single model configuration."""
    
    # Setup
    ntf.set_seed(config.get('seed', 42))
    device = ntf.get_device()
    
    # Load data
    train_loader, test_loader = ntf.get_dataloaders(config)
    
    # Create model
    model = ntf.get_model(config['model'])
    model_name = config['model'].get('name', 'model')
    
    logger.info(f"Training model: {model_name}")
    
    # Training
    with ntf.Timer(f"Training {model_name}"):
        trained_model = ntf.training.train_and_evaluate(
            config, model, train_loader, test_loader, device, exp_logger
        )
    
    # Save model as artifact
    if exp_logger:
        model_metadata = {
            'model_name': model_name,
            'architecture': config['model']['architecture'],
            'dataset': config['dataset']['name'],
            'final_accuracy': 'logged_during_training'
        }
        exp_logger.save_model_artifact(trained_model, model_name, model_metadata)
        logger.info(f"Saved model artifact: {model_name}")
    
    return trained_model

def analyze_model_topology(model, config, exp_logger=None):
    """Perform complete topological analysis on a trained model."""
    
    device = model.get_device()
    analysis_config = config['analysis']
    
    # Load data for activation extraction
    _, test_loader = ntf.get_dataloaders(config)
    
    # Extract activations
    logger.info("Extracting activations...")
    with ntf.Timer("Activation extraction"):
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
    with ntf.Timer("Distance matrix computation"):
        distance_matrix = ntf.compute_distance_matrix(concatenated_acts)
    
    # Compute topological features
    logger.info("Computing persistence diagrams...")
    with ntf.Timer("Persistence computation"):
        persistence_diagrams = ntf.compute_persistence_diagrams(
            distance_matrix, max_dimension=2
        )
    
    # Classify neuron criticality
    logger.info("Classifying neuron criticality...")
    with ntf.Timer("Criticality classification"):
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

logger = logging.getLogger(__name__)

def run_train_analyze_grid(grid_config: Dict[str, Any]) -> Dict[str, Any]:
    """Train all models and analyze each one."""
    
    base_config = grid_config['base_config']
    models = grid_config['models']
    
    logger.info(f"Starting train+analyze grid: {len(models)} models")
    
    results = {
        'completed': [],
        'failed': []
    }
    
    for i, model_spec in enumerate(models):
        model_name = model_spec.get('name', f'model_{i}')
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING MODEL {i+1}/{len(models)}: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create model-specific config
            config = ntf.config.merge_configs(base_config, {'model': model_spec})
            
            # Validate the merged config (now it has all required sections at top level)
            ntf.validate_config(config)
            
            # STEP 1: Train the model
            logger.info(f"🚀 Training {model_name}...")
            
            train_exp_name = f"train_{model_name}"
            train_logger = ntf.init_experiment(
                project_name="neural-topology-grid",
                experiment_name=train_exp_name,
                config=config,
                tags=['grid', 'training', model_name],
                group='model_training'
            )
            
            model = train_single_model(config, train_logger)
            train_run_id = train_logger.run.id
            train_logger.finish()
            
            logger.info(f"✅ Training completed: {model_name}")
            
            # STEP 2: Analyze the model
            logger.info(f"🔍 Analyzing {model_name}...")
            
            analyze_exp_name = f"analyze_{model_name}"
            analyze_logger = ntf.init_experiment(
                project_name="neural-topology-grid",
                experiment_name=analyze_exp_name,
                config=config,
                tags=['grid', 'analysis', model_name],
                group='topological_analysis'
            )
            
            analysis_results = analyze_model_topology(model, config, analyze_logger)
            analyze_run_id = analyze_logger.run.id
            analyze_logger.finish()
            
            logger.info(f"✅ Analysis completed: {model_name}")
            
            # Store success
            results['completed'].append({
                'model_name': model_name,
                'train_run_id': train_run_id,
                'analyze_run_id': analyze_run_id,
                'config': config
            })
            
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {e}")
            results['failed'].append({
                'model_name': model_name,
                'error': str(e)
            })
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("GRID EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Completed: {len(results['completed'])}/{len(models)}")
    logger.info(f"❌ Failed: {len(results['failed'])}")
    
    if results['completed']:
        logger.info("\nSuccessful experiments:")
        for result in results['completed']:
            logger.info(f"  - {result['model_name']}: train={result['train_run_id'][:8]}, analyze={result['analyze_run_id'][:8]}")
    
    if results['failed']:
        logger.info("\nFailed experiments:")
        for failure in results['failed']:
            logger.info(f"  - {failure['model_name']}: {failure['error']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run train+analyze grid experiment")
    parser.add_argument('config', help="Path to grid configuration file")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be run")
    
    args = parser.parse_args()
    
    # Load configuration directly (bypass validation for grid configs)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate it's a grid config (don't validate the full config, just check structure)
    if 'base_config' not in config or 'models' not in config:
        logger.error("Configuration must be a grid config with 'base_config' and 'models' sections")
        sys.exit(1)
    
    # Show what will be run
    models = config['models']
    logger.info(f"Grid configuration loaded:")
    logger.info(f"  - Dataset: {config['base_config']['dataset']['name']}")
    logger.info(f"  - Models: {len(models)}")
    for i, model in enumerate(models):
        logger.info(f"    {i+1}. {model.get('name', f'model_{i}')}")
    
    if args.dry_run:
        logger.info("\nDry run mode - not executing")
        return
    
    # Run the grid
    logger.info(f"\nStarting execution...")
    results = run_train_analyze_grid(config)
    
    # Final summary
    if results['completed']:
        logger.info(f"\n🎉 Grid experiment completed successfully!")
        logger.info(f"Check your wandb project 'neural-topology-grid' for results")

if __name__ == '__main__':
    main()