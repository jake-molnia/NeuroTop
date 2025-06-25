#!/usr/bin/env python3
"""
Pure ablation testing script.
Loads trained models and analysis results to perform neuron ablation experiments.
"""

import sys
import argparse
import logging
import wandb
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
sys.path.append(str(Path(__file__).parent.parent))

from .. import modules as ntf
from ..modules.utils import Timer

logger = logging.getLogger(__name__)

def load_analysis_results_from_wandb(
    project_name: str,
    run_id: str,
    artifact_name: str
) -> pd.DataFrame:
    """Load analysis results from wandb artifacts."""
    
    api = wandb.Api()
    run = api.run(f"{project_name}/{run_id}")
    
    # Download analysis artifact
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()
    
    # Load results
    results_path = Path(artifact_dir) / f"{artifact_name}_results.csv"
    results_df = pd.read_csv(results_path)
    
    logger.info(f"Loaded analysis results: {results_df.shape[0]} neurons")
    return results_df

def run_ablation_strategy(
    model: ntf.models.BaseModel,
    config: Dict[str, Any],
    strategy_config: Dict[str, Any],
    analysis_results: Optional[pd.DataFrame] = None,
    exp_logger: Optional[ntf.ExperimentLogger] = None
) -> pd.DataFrame:
    """Run a single ablation strategy."""
    
    strategy_name = strategy_config['strategy']
    strategy_params = strategy_config.get('params', {})
    
    logger.info(f"Running ablation strategy: {strategy_name}")
    
    device = model.get_device()
    analysis_config = config['analysis']
    
    # Load data
    _, test_loader = ntf.get_dataloaders(config)
    
    # Extract activations (needed for distance matrix)
    logger.info("Extracting activations for ablation...")
    with Timer("Activation extraction"):
        raw_activations = ntf.extract_activations(
            model=model,
            data_loader=test_loader,
            layers_to_hook=analysis_config['activation_extraction']['layers'],
            max_samples=analysis_config['activation_extraction'].get('max_samples'),
            device=device
        )
    
    # Normalize if configured
    normalize_method = analysis_config['activation_extraction'].get('normalization_method', 'standard')
    if normalize_method != 'none':
        activations = ntf.normalize_activations(raw_activations, normalize_method)
    else:
        activations = raw_activations
    
    # Concatenate for global analysis
    concatenated_acts, global_to_local_map = ntf.concatenate_layer_activations(
        activations, analysis_config['activation_extraction']['layers']
    )
    
    # Compute distance matrix
    logger.info("Computing distance matrix...")
    with Timer("Distance matrix computation"):
        distance_matrix = ntf.compute_distance_matrix(concatenated_acts)
    
    # Run ablation analysis to get neuron ranking
    logger.info(f"Computing {strategy_name} neuron ranking...")
    with Timer(f"{strategy_name} ranking"):
        neuron_ranking = ntf.run_ablation_analysis(
            activations=concatenated_acts,
            distance_matrix=distance_matrix,
            strategy_name=strategy_name,
            strategy_params=strategy_params
        )
    
    # Test ablation performance
    removal_percentages = list(range(0, 101, strategy_config.get('step', 5)))
    
    logger.info("Testing ablation performance...")
    with Timer("Ablation testing"):
        results_df = ntf.test_ablation_performance(
            model=model,
            test_loader=test_loader,
            neuron_ranking=neuron_ranking,
            global_to_local_map=global_to_local_map,
            removal_percentages=removal_percentages,
            device=device
        )
    
    # Add strategy info to results
    results_df['strategy'] = strategy_name
    results_df['model_name'] = config['model'].get('name', 'model')
    
    # Create ablation curve plot
    ablation_plot = ntf.visualization.plot_ablation_curves(
        results_df, 
        title=f"Ablation Performance: {strategy_name}"
    )
    
    # Log results
    if exp_logger:
        # Log summary metrics
        baseline_acc = results_df[results_df['percent_removed'] == 0]['accuracy'].iloc[0]
        final_acc = results_df[results_df['percent_removed'] == results_df['percent_removed'].max()]['accuracy'].iloc[0]
        degradation = baseline_acc - final_acc
        
        metrics = {
            f'ablation/{strategy_name}/baseline_accuracy': baseline_acc,
            f'ablation/{strategy_name}/final_accuracy': final_acc,
            f'ablation/{strategy_name}/total_degradation': degradation,
            f'ablation/{strategy_name}/degradation_per_10pct': degradation / 10.0,
            f'ablation/{strategy_name}/neurons_analyzed': len(neuron_ranking)
        }
        exp_logger.log_metrics(metrics)
        
        # Log results table and plot
        exp_logger.log_dataframe(results_df, f'ablation_{strategy_name}_results')
        exp_logger.log_figure(ablation_plot, f'ablation/{strategy_name}_curve')
        
        # Save as artifact
        exp_logger.save_results_artifact(
            results_df, 
            f"ablation_{strategy_name}",
            metadata={
                'strategy': strategy_name,
                'strategy_params': strategy_params,
                'baseline_accuracy': baseline_acc,
                'final_accuracy': final_acc
            }
        )
    
    logger.info(f"Ablation strategy {strategy_name} completed")
    return results_df

def run_ablation_suite(
    model: ntf.models.BaseModel,
    config: Dict[str, Any],
    strategies: List[Dict[str, Any]],
    analysis_results: Optional[pd.DataFrame] = None,
    exp_logger: Optional[ntf.ExperimentLogger] = None
) -> List[pd.DataFrame]:
    """Run multiple ablation strategies."""
    
    logger.info(f"Running ablation suite with {len(strategies)} strategies")
    
    all_results = []
    
    for strategy_config in strategies:
        strategy_name = strategy_config['strategy']
        
        try:
            results_df = run_ablation_strategy(
                model=model,
                config=config,
                strategy_config=strategy_config,
                analysis_results=analysis_results,
                exp_logger=exp_logger
            )
            all_results.append(results_df)
            
        except Exception as e:
            logger.error(f"Ablation strategy {strategy_name} failed: {e}")
            continue
    
    # Create comparison plot if multiple strategies
    if len(all_results) > 1 and exp_logger:
        combined_df = pd.concat(all_results, ignore_index=True)
        comparison_plot = ntf.visualization.plot_ablation_curves(
            combined_df, 
            group_by='strategy',
            title="Ablation Strategy Comparison"
        )
        exp_logger.log_figure(comparison_plot, 'ablation/strategy_comparison')
    
    successful = len(all_results)
    logger.info(f"Ablation suite completed: {successful}/{len(strategies)} strategies successful")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Test neuron ablation strategies")
    parser.add_argument('config', help="Path to ablation configuration file")
    parser.add_argument('--model-run', required=True, help="Wandb run ID containing the trained model")
    parser.add_argument('--model-artifact', required=True, help="Name of the model artifact")
    parser.add_argument('--analysis-run', help="Wandb run ID containing analysis results (optional)")
    parser.add_argument('--analysis-artifact', help="Name of the analysis artifact (optional)")
    parser.add_argument('--model-project', default='neural-topology-training', help="Wandb project for model")
    parser.add_argument('--analysis-project', default='neural-topology-analysis', help="Wandb project for analysis")
    parser.add_argument('--strategy', help="Run only specific strategy (if not provided, runs all)")
    parser.add_argument('--name', help="Override experiment name")
    parser.add_argument('--tags', nargs='*', help="Additional tags")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ntf.load_config(args.config)
    
    # Setup
    ntf.set_seed(config.get('seed', 42))
    device = ntf.get_device()
    
    # Load analysis results if provided
    analysis_results = None
    if args.analysis_run and args.analysis_artifact:
        logger.info("Loading analysis results...")
        analysis_results = load_analysis_results_from_wandb(
            project_name=args.analysis_project,
            run_id=args.analysis_run,
            artifact_name=args.analysis_artifact
        )
    
    # Create experiment name
    model_name = config['model'].get('name', 'model')
    exp_name = args.name or f"ablation_{model_name}_{args.model_run[:8]}"
    
    # Initialize experiment logger
    tags = ['ablation'] + config.get('tags', [])
    if args.tags:
        tags.extend(args.tags)
    
    exp_logger = ntf.init_experiment(
        project_name="neural-topology-ablation",
        experiment_name=exp_name,
        config=config,
        tags=tags,
        group='ablation_testing'
    )
    
    try:
        # Load trained model (same as in analyze.py)
        from analyze import load_model_from_wandb
        logger.info(f"Loading model from run: {args.model_run}")
        model = load_model_from_wandb(
            project_name=args.model_project,
            run_id=args.model_run,
            artifact_name=args.model_artifact,
            model_config=config['model'],
            device=device
        )
        
        # Determine which strategies to run
        if 'ablations' in config:
            # Grid config with multiple strategies
            strategies = config['ablations']
        elif 'ablation' in config:
            # Single strategy config
            strategies = [config['ablation']]
        else:
            raise ValueError("No ablation strategies found in configuration")
        
        # Filter by specific strategy if requested
        if args.strategy:
            strategies = [s for s in strategies if s['strategy'] == args.strategy]
            if not strategies:
                raise ValueError(f"Strategy '{args.strategy}' not found in configuration")
        
        # Run ablation experiments
        if len(strategies) == 1:
            results = run_ablation_strategy(
                model=model,
                config=config,
                strategy_config=strategies[0],
                analysis_results=analysis_results,
                exp_logger=exp_logger
            )
        else:
            results_list = run_ablation_suite(
                model=model,
                config=config,
                strategies=strategies,
                analysis_results=analysis_results,
                exp_logger=exp_logger
            )
        
        logger.info("Ablation experiments completed successfully")
        
    except Exception as e:
        logger.error(f"Ablation experiments failed: {e}")
        raise
    finally:
        exp_logger.finish()

if __name__ == '__main__':
    main()