#!/usr/bin/env python3
"""
Cache management utilities for temporal flow analysis.
Clean, simple cache operations following tinygrad philosophy.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.storage import TemporalExperimentManager

logger = logging.getLogger(__name__)

def list_experiments(cache_dir: str) -> List[Dict]:
    """List all experiments in cache."""
    
    manager = TemporalExperimentManager(cache_dir)
    experiments = []
    
    for exp_id in manager.activation_store.metadata['experiments'].keys():
        try:
            summary = manager.get_experiment_summary(exp_id)
            experiments.append(summary)
        except Exception as e:
            logger.warning(f"Could not get summary for {exp_id}: {e}")
    
    return experiments

def show_experiment_details(cache_dir: str, experiment_id: str):
    """Show detailed information about specific experiment."""
    
    manager = TemporalExperimentManager(cache_dir)
    
    try:
        summary = manager.get_experiment_summary(experiment_id)
        
        print(f"\nExperiment: {experiment_id}")
        print("=" * 60)
        print(f"Created: {summary['created']}")
        print(f"Size: {summary['total_size_mb']:.1f} MB")
        print(f"Epochs: {len(summary['available_epochs'])}")
        print(f"Available epochs: {summary['available_epochs']}")
        
        # Show config if available
        if summary['config']:
            print(f"\nConfiguration:")
            print(f"  Dataset: {summary['config'].get('dataset', {}).get('name', 'Unknown')}")
            print(f"  Model: {summary['config'].get('model', {}).get('name', 'Unknown')}")
            
            temporal_config = summary['config'].get('temporal_analysis', {})
            if temporal_config.get('enabled', False):
                print(f"  Temporal capture epochs: {temporal_config.get('capture_epochs', [])}")
        
        # Check for flow analysis data
        exp_dir = Path(cache_dir) / experiment_id / "flow_analysis"
        if exp_dir.exists():
            flow_files = list(exp_dir.glob("*.pt"))
            print(f"  Flow analysis files: {len(flow_files)}")
        
        print()
        
    except Exception as e:
        print(f"Error getting details for {experiment_id}: {e}")

def cleanup_experiment(cache_dir: str, experiment_id: str):
    """Remove specific experiment from cache."""
    
    manager = TemporalExperimentManager(cache_dir)
    
    try:
        # Get size before cleanup
        summary = manager.get_experiment_summary(experiment_id)
        size_mb = summary['total_size_mb']
        
        # Cleanup
        manager.cleanup_experiment(experiment_id)
        
        print(f"Removed experiment {experiment_id} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"Error cleaning up {experiment_id}: {e}")

def compress_cache(cache_dir: str):
    """Compress cache data (placeholder for future compression features)."""
    
    manager = TemporalExperimentManager(cache_dir)
    cache_stats = manager.get_cache_stats()
    
    print(f"Cache compression not yet implemented")
    print(f"Current cache size: {cache_stats['total_size_gb']:.2f} GB")

def show_cache_stats(cache_dir: str):
    """Show cache statistics and health."""
    
    manager = TemporalExperimentManager(cache_dir)
    stats = manager.get_cache_stats()
    
    print(f"\nCache Statistics")
    print("=" * 40)
    print(f"Directory: {stats['cache_directory']}")
    print(f"Total size: {stats['total_size_gb']:.2f} GB")
    print(f"Max size: {stats['max_size_gb']:.2f} GB")
    print(f"Utilization: {stats['utilization_pct']:.1f}%")
    print(f"Experiments: {stats['num_experiments']}")
    
    if stats['num_experiments'] > 0:
        print(f"Oldest experiment: {stats['oldest_experiment_age_hours']:.1f} hours ago")
    
    # Warning if cache is getting full
    if stats['utilization_pct'] > 80:
        print(f"\nWARNING: Cache is {stats['utilization_pct']:.1f}% full")
        print("Consider cleaning up old experiments")
    
    print()

def export_experiment_data(
    cache_dir: str, 
    experiment_id: str, 
    output_dir: str,
    format: str = 'csv'
):
    """Export experiment data to external format."""
    
    manager = TemporalExperimentManager(cache_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        summary = manager.get_experiment_summary(experiment_id)
        available_epochs = summary['available_epochs']
        
        print(f"Exporting {experiment_id} data for {len(available_epochs)} epochs...")
        
        if format == 'csv':
            # Export flow analysis results as CSV
            for epoch in available_epochs:
                try:
                    flow_data = manager.flow_store.load_flow_analysis(experiment_id, epoch)
                    
                    # Export criticality data
                    if 'criticality' in flow_data:
                        criticality_df = flow_data['criticality']
                        csv_path = output_path / f"{experiment_id}_criticality_epoch_{epoch:04d}.csv"
                        criticality_df.to_csv(csv_path, index=False)
                    
                    # Export energy data
                    if 'energies' in flow_data:
                        energies = flow_data['energies']
                        energy_path = output_path / f"{experiment_id}_energies_epoch_{epoch:04d}.json"
                        with open(energy_path, 'w') as f:
                            json.dump(energies, f, indent=2)
                    
                except Exception as e:
                    logger.warning(f"Could not export epoch {epoch}: {e}")
        
        elif format == 'json':
            # Export as JSON
            export_data = {
                'experiment_id': experiment_id,
                'summary': summary,
                'epochs': {}
            }
            
            for epoch in available_epochs:
                try:
                    flow_data = manager.flow_store.load_flow_analysis(experiment_id, epoch)
                    
                    # Convert non-serializable data
                    serializable_data = {}
                    for key, value in flow_data.items():
                        if hasattr(value, 'to_dict'):  # DataFrame
                            serializable_data[key] = value.to_dict()
                        elif hasattr(value, 'tolist'):  # NumPy array
                            serializable_data[key] = value.tolist()
                        else:
                            serializable_data[key] = value
                    
                    export_data['epochs'][str(epoch)] = serializable_data
                    
                except Exception as e:
                    logger.warning(f"Could not export epoch {epoch}: {e}")
            
            json_path = output_path / f"{experiment_id}_export.json"
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        print(f"Export completed to {output_path}")
        
    except Exception as e:
        print(f"Export failed: {e}")

def validate_cache_integrity(cache_dir: str):
    """Validate cache integrity and consistency."""
    
    manager = TemporalExperimentManager(cache_dir)
    cache_path = Path(cache_dir)
    
    print("Validating cache integrity...")
    
    issues = []
    
    # Check metadata consistency
    metadata = manager.activation_store.metadata
    
    for exp_id, exp_data in metadata['experiments'].items():
        exp_dir = cache_path / exp_id
        
        # Check if experiment directory exists
        if not exp_dir.exists():
            issues.append(f"Missing directory for experiment {exp_id}")
            continue
        
        # Check if epoch files exist
        for epoch in exp_data['epochs']:
            h5_file = exp_dir / f"activations_epoch_{epoch:04d}.h5"
            if not h5_file.exists():
                issues.append(f"Missing activation file for {exp_id} epoch {epoch}")
        
        # Check config file
        config_file = exp_dir / "config.json"
        if not config_file.exists():
            issues.append(f"Missing config file for {exp_id}")
    
    # Check for orphaned files
    for exp_dir in cache_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['metadata.json']:
            if exp_dir.name not in metadata['experiments']:
                issues.append(f"Orphaned experiment directory: {exp_dir.name}")
    
    if issues:
        print(f"\nFound {len(issues)} integrity issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Cache integrity check passed ✓")

def main():
    parser = argparse.ArgumentParser(description="Cache management utilities")
    parser.add_argument('--cache-dir', default='./flow_cache', help="Cache directory path")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Show experiment details
    show_parser = subparsers.add_parser('show', help='Show experiment details')
    show_parser.add_argument('experiment_id', help='Experiment ID to show')
    
    # Cleanup experiment
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove experiment from cache')
    cleanup_parser.add_argument('experiment_id', help='Experiment ID to remove')
    
    # Cache statistics
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    
    # Export experiment data
    export_parser = subparsers.add_parser('export', help='Export experiment data')
    export_parser.add_argument('experiment_id', help='Experiment ID to export')
    export_parser.add_argument('output_dir', help='Output directory')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Export format')
    
    # Validate cache
    validate_parser = subparsers.add_parser('validate', help='Validate cache integrity')
    
    # Compress cache
    compress_parser = subparsers.add_parser('compress', help='Compress cache data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Ensure cache directory exists
    cache_path = Path(args.cache_dir)
    if not cache_path.exists() and args.command != 'stats':
        print(f"Cache directory {args.cache_dir} does not exist")
        return
    
    # Execute commands
    if args.command == 'list':
        experiments = list_experiments(args.cache_dir)
        
        if not experiments:
            print("No experiments found in cache")
        else:
            print(f"\nFound {len(experiments)} experiments:")
            print("-" * 80)
            print(f"{'Experiment ID':<40} {'Size (MB)':<10} {'Epochs':<8} {'Age (hrs)':<10}")
            print("-" * 80)
            
            import time
            current_time = time.time()
            
            for exp in experiments:
                age_hours = (current_time - exp['created']) / 3600
                print(f"{exp['experiment_id']:<40} {exp['total_size_mb']:<10.1f} "
                     f"{exp['num_epochs']:<8} {age_hours:<10.1f}")
    
    elif args.command == 'show':
        show_experiment_details(args.cache_dir, args.experiment_id)
    
    elif args.command == 'cleanup':
        cleanup_experiment(args.cache_dir, args.experiment_id)
    
    elif args.command == 'stats':
        show_cache_stats(args.cache_dir)
    
    elif args.command == 'export':
        export_experiment_data(
            args.cache_dir, 
            args.experiment_id, 
            args.output_dir,
            args.format
        )
    
    elif args.command == 'validate':
        validate_cache_integrity(args.cache_dir)
    
    elif args.command == 'compress':
        compress_cache(args.cache_dir)

if __name__ == '__main__':
    main()