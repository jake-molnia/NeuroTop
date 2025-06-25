#!/usr/bin/env python3
"""
Comparison and visualization script.
Loads multiple experiment results and creates comprehensive comparison plots.
"""

import sys
import argparse
import logging
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
sys.path.append(str(Path(__file__).parent.parent))

from .. import modules as ntf
from ..modules.utils import Timer

logger = logging.getLogger(__name__)

def fetch_runs_from_wandb(
    project_name: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100
) -> List[wandb.Run]:
    """Fetch runs from wandb project with optional filtering."""
    
    api = wandb.Api()
    
    # Build filter string
    filter_string = ""
    if filters:
        filter_parts = []
        for key, value in filters.items():
            if isinstance(value, list):
                filter_parts.append(f"{key} in {value}")
            else:
                filter_parts.append(f"{key} = '{value}'")
        filter_string = " and ".join(filter_parts)
    
    runs = api.runs(project_name, filters=filter_string if filter_string else {})
    
    logger.info(f"Found {len(runs)} runs in {project_name}")
    return list(runs)[:limit]

def load_ablation_results_from_runs(
    runs: List[wandb.Run],
    artifact_pattern: str = "ablation_"
) -> pd.DataFrame:
    """Load ablation results from multiple wandb runs."""
    
    all_results = []
    
    for run in runs:
        try:
            # Find ablation artifacts
            artifacts = [a for a in run.logged_artifacts() if artifact_pattern in a.name]
            
            for artifact in artifacts:
                # Download and load
                artifact_dir = artifact.download()
                csv_files = list(Path(artifact_dir).glob("*.csv"))
                
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    
                    # Add run metadata
                    df['run_id'] = run.id
                    df['run_name'] = run.name
                    df['created_at'] = run.created_at
                    
                    # Add config information
                    for key, value in run.config.items():
                        if isinstance(value, (str, int, float, bool)):
                            df[f'config_{key}'] = value
                    
                    all_results.append(df)
                    
        except Exception as e:
            logger.warning(f"Could not load results from run {run.id}: {e}")
            continue
    
    if not all_results:
        raise ValueError("No ablation results could be loaded")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"Loaded results from {len(all_results)} artifacts, {len(combined_df)} total rows")
    
    return combined_df

def create_comprehensive_comparison(
    results_df: pd.DataFrame,
    exp_logger: Optional[ntf.ExperimentLogger] = None
) -> Dict[str, Any]:
    """Create comprehensive comparison analysis."""
    
    plots = {}
    summary_stats = {}
    
    # 1. Strategy comparison across all models
    if 'strategy' in results_df.columns:
        logger.info("Creating strategy comparison plots...")
        
        # Main comparison plot
        plots['strategy_comparison'] = ntf.visualization.plot_ablation_curves(
            results_df,
            group_by='strategy',
            title="Ablation Strategy Comparison (All Models)"
        )
        
        # Performance summary by strategy
        strategy_summary = []
        for strategy in results_df['strategy'].unique():
            strategy_data = results_df[results_df['strategy'] == strategy]
            
            # Calculate metrics at different removal percentages
            for percent in [50, 75, 100]:
                subset = strategy_data[strategy_data['percent_removed'] == percent]
                if not subset.empty:
                    strategy_summary.append({
                        'strategy': strategy,
                        'percent_removed': percent,
                        'mean_accuracy': subset['accuracy'].mean(),
                        'std_accuracy': subset['accuracy'].std(),
                        'n_models': len(subset)
                    })
        
        summary_stats['strategy_performance'] = pd.DataFrame(strategy_summary)
    
    # 2. Model comparison across strategies
    if 'model_name' in results_df.columns:
        logger.info("Creating model comparison plots...")
        
        plots['model_comparison'] = ntf.visualization.plot_ablation_curves(
            results_df,
            group_by='model_name',
            title="Model Comparison (All Strategies)"
        )
        
        # Model robustness analysis
        model_robustness = []
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            
            # Calculate area under curve (robustness metric)
            for strategy in model_data['strategy'].unique():
                subset = model_data[model_data['strategy'] == strategy]
                if len(subset) > 1:
                    # Simple AUC approximation
                    auc = np.trapz(subset['accuracy'], subset['percent_removed'])
                    model_robustness.append({
                        'model': model,
                        'strategy': strategy,
                        'auc': auc,
                        'final_accuracy': subset[subset['percent_removed'] == subset['percent_removed'].max()]['accuracy'].iloc[0]
                    })
        
        summary_stats['model_robustness'] = pd.DataFrame(model_robustness)
    
    # 3. Performance degradation analysis
    logger.info("Analyzing performance degradation patterns...")
    
    degradation_analysis = []
    for _, group_df in results_df.groupby(['model_name', 'strategy']):
        if len(group_df) > 1:
            baseline = group_df[group_df['percent_removed'] == 0]['accuracy'].iloc[0]
            
            # Find critical points
            critical_points = {}
            for threshold in [90, 80, 70, 60, 50]:
                degraded = group_df[group_df['accuracy'] <= threshold]
                if not degraded.empty:
                    critical_points[f'accuracy_{threshold}'] = degraded['percent_removed'].min()
            
            degradation_analysis.append({
                'model': group_df['model_name'].iloc[0],
                'strategy': group_df['strategy'].iloc[0],
                'baseline_accuracy': baseline,
                **critical_points
            })
    
    summary_stats['degradation_analysis'] = pd.DataFrame(degradation_analysis)
    
    # 4. Statistical significance testing
    if len(results_df['strategy'].unique()) > 1:
        logger.info("Performing statistical analysis...")
        
        # Compare strategies at key removal percentages
        from scipy import stats
        stat_results = []
        
        for percent in [25, 50, 75]:
            subset = results_df[results_df['percent_removed'] == percent]
            strategies = subset['strategy'].unique()
            
            if len(strategies) >= 2:
                for i, strat1 in enumerate(strategies):
                    for strat2 in strategies[i+1:]:
                        data1 = subset[subset['strategy'] == strat1]['accuracy']
                        data2 = subset[subset['strategy'] == strat2]['accuracy']
                        
                        if len(data1) > 1 and len(data2) > 1:
                            t_stat, p_value = stats.ttest_ind(data1, data2)
                            stat_results.append({
                                'percent_removed': percent,
                                'strategy1': strat1,
                                'strategy2': strat2,
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
        
        summary_stats['statistical_tests'] = pd.DataFrame(stat_results)
    
    # Log everything if logger provided
    if exp_logger:
        # Log plots
        for plot_name, figure in plots.items():
            exp_logger.log_figure(figure, f'comparison/{plot_name}')
        
        # Log summary statistics
        for stat_name, df in summary_stats.items():
            if not df.empty:
                exp_logger.log_dataframe(df, f'summary_{stat_name}')
        
        # Log overall metrics
        overall_metrics = {
            'comparison/total_experiments': len(results_df['run_id'].unique()),
            'comparison/total_strategies': len(results_df['strategy'].unique()),
            'comparison/total_models': len(results_df['model_name'].unique()) if 'model_name' in results_df.columns else 1
        }
        exp_logger.log_metrics(overall_metrics)
    
    return {
        'plots': plots,
        'summary_stats': summary_stats,
        'raw_data': results_df
    }

def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument('--project', required=True, help="Wandb project to analyze")
    parser.add_argument('--filter-tags', nargs='*', help="Filter runs by tags")
    parser.add_argument('--filter-models', nargs='*', help="Filter runs by model names")
    parser.add_argument('--filter-strategies', nargs='*', help="Filter by ablation strategies")
    parser.add_argument('--limit', type=int, default=100, help="Maximum number of runs to analyze")
    parser.add_argument('--name', help="Experiment name for comparison results")
    parser.add_argument('--output-dir', help="Local directory to save plots (optional)")
    
    args = parser.parse_args()
    
    # Build filters
    filters = {}
    if args.filter_tags:
        filters['tags'] = args.filter_tags
    if args.filter_models:
        filters['config.model_name'] = args.filter_models
    
    # Fetch runs
    logger.info(f"Fetching runs from {args.project}...")
    runs = fetch_runs_from_wandb(args.project, filters, args.limit)
    
    if not runs:
        logger.error("No runs found matching criteria")
        return
    
    # Load results
    logger.info("Loading ablation results...")
    with Timer("Loading results"):
        results_df = load_ablation_results_from_runs(runs)
    
    # Filter by strategies if specified
    if args.filter_strategies:
        initial_count = len(results_df)
        results_df = results_df[results_df['strategy'].isin(args.filter_strategies)]
        logger.info(f"Filtered by strategies: {initial_count} -> {len(results_df)} results")
    
    # Create experiment for comparison results
    exp_name = args.name or f"comparison_{args.project}_{len(runs)}runs"
    
    exp_logger = ntf.init_experiment(
        project_name="neural-topology-comparisons",
        experiment_name=exp_name,
        config={
            'source_project': args.project,
            'n_runs_analyzed': len(runs),
            'filters': filters
        },
        tags=['comparison', 'analysis']
    )
    
    try:
        # Create comprehensive comparison
        logger.info("Creating comprehensive comparison analysis...")
        with Timer("Comparison analysis"):
            comparison_results = create_comprehensive_comparison(results_df, exp_logger)
        
        # Save local copies if requested
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save plots
            for plot_name, figure in comparison_results['plots'].items():
                ntf.visualization.save_figure(
                    figure, 
                    output_path / f"{plot_name}.png",
                    close=False
                )
            
            # Save summary statistics
            for stat_name, df in comparison_results['summary_stats'].items():
                df.to_csv(output_path / f"{stat_name}.csv", index=False)
            
            logger.info(f"Saved results to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("COMPARISON ANALYSIS SUMMARY")
        print("="*60)
        print(f"Analyzed {len(runs)} runs from {args.project}")
        print(f"Total data points: {len(results_df)}")
        print(f"Unique strategies: {list(results_df['strategy'].unique())}")
        if 'model_name' in results_df.columns:
            print(f"Unique models: {list(results_df['model_name'].unique())}")
        print(f"Results saved to wandb: {exp_logger.run.url}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Comparison analysis failed: {e}")
        raise
    finally:
        exp_logger.finish()

if __name__ == '__main__':
    main()