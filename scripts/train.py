#!/usr/bin/env python3
"""
Pure model training script.
Trains models and saves them as wandb artifacts for later analysis.
"""

import sys
import argparse
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import modules as ntf
from modules.training import train_and_evaluate
from modules.utils import Timer, validate_experiment_setup

logger = logging.getLogger(__name__)

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
    with Timer(f"Training {model_name}"):
        trained_model = train_and_evaluate(
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

def train_from_grid_config(grid_config):
    """Train multiple models from grid search configuration."""
    
    base_config = grid_config['base_config']
    models = grid_config['models']
    
    logger.info(f"Training {len(models)} models from grid configuration")
    
    results = []
    
    for model_spec in models:
        # Create specific config for this model
        config = ntf.merge_configs(base_config, {'model': model_spec})
        
        # Create experiment name
        model_name = model_spec.get('name', 'model')
        exp_name = f"train_{config['dataset']['name']}_{model_name}"
        
        # Initialize experiment logger
        exp_logger = ntf.init_experiment(
            project_name="neural-topology-training",
            experiment_name=exp_name,
            config=config,
            tags=['training'] + config.get('tags', []),
            group='model_training'
        )
        
        try:
            # Validate setup
            validate_experiment_setup(config)
            
            # Train model
            model = train_single_model(config, exp_logger)
            
            results.append({
                'model_name': model_name,
                'experiment_name': exp_name,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            results.append({
                'model_name': model_name,
                'experiment_name': exp_name,
                'status': 'failed',
                'error': str(e)
            })
        
        finally:
            exp_logger.finish()
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    logger.info(f"Training completed: {successful}/{len(results)} models successful")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train neural network models")
    parser.add_argument('config', help="Path to configuration file")
    parser.add_argument('--name', help="Override experiment name")
    parser.add_argument('--tags', nargs='*', help="Additional tags for the experiment")
    parser.add_argument('--dry-run', action='store_true', help="Validate config without training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ntf.load_config(args.config)
    
    # Handle different config types
    if 'base_config' in config and 'models' in config:
        # Grid search configuration
        if args.dry_run:
            logger.info("Dry run: Grid search configuration validated")
            return
        
        train_from_grid_config(config)
    
    else:
        # Single model configuration
        if args.dry_run:
            ntf.validate_config(config)
            validate_experiment_setup(config)
            logger.info("Dry run: Single model configuration validated")
            return
        
        # Create experiment name
        exp_name = args.name or ntf.get_experiment_name(config)
        
        # Add custom tags
        tags = ['training'] + config.get('tags', [])
        if args.tags:
            tags.extend(args.tags)
        
        # Initialize experiment
        exp_logger = ntf.init_experiment(
            project_name="neural-topology-training",
            experiment_name=exp_name,
            config=config,
            tags=tags
        )
        
        try:
            validate_experiment_setup(config)
            train_single_model(config, exp_logger)
        finally:
            exp_logger.finish()

if __name__ == '__main__':
    main()