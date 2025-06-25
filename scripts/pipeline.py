#!/usr/bin/env python3
"""
Pipeline orchestration script.
Runs the complete neural topology analysis pipeline: train -> analyze -> ablate -> compare.
"""

import os
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
sys.path.append(str(Path(__file__).parent.parent))

import modules as ntf

logger = logging.getLogger(__name__)

class PipelineStage:
    """Represents a single stage in the analysis pipeline."""
    
    def __init__(
        self, 
        name: str, 
        script: str, 
        args: List[str], 
        depends_on: Optional[List[str]] = None
    ):
        self.name = name
        self.script = script
        self.args = args
        self.depends_on = depends_on or []
        self.start_time = None
        self.end_time = None
        self.return_code = None
        self.output = None
        self.error = None

    def run(self) -> bool:
        """Execute this pipeline stage."""
        
        logger.info(f"Starting pipeline stage: {self.name}")
        self.start_time = time.time()
        
        # Build command
        script_path = str(Path(__file__).parent / self.script)
        cmd = [sys.executable, script_path] + self.args
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            self.return_code = result.returncode
            self.output = result.stdout
            self.error = result.stderr
            
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            
            if self.return_code == 0:
                logger.info(f"Stage {self.name} completed successfully in {duration:.1f}s")
                return True
            else:
                logger.error(f"Stage {self.name} failed with return code {self.return_code}")
                if self.error:
                    logger.error(f"Error output: {self.error}")
                return False
                
        except Exception as e:
            self.end_time = time.time()
            logger.error(f"Stage {self.name} failed with exception: {e}")
            return False

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class Pipeline:
    """Manages the complete analysis pipeline."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = ntf.load_config(config_path)
        self.stages = []
        self.results = {}
        
    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        
    def run(self, start_from: Optional[str] = None, stop_at: Optional[str] = None) -> bool:
        """Run the complete pipeline."""
        
        logger.info(f"Starting pipeline with {len(self.stages)} stages")
        
        # Find start/stop indices
        start_idx = 0
        stop_idx = len(self.stages)
        
        if start_from:
            start_idx = next((i for i, s in enumerate(self.stages) if s.name == start_from), 0)
        if stop_at:
            stop_idx = next((i+1 for i, s in enumerate(self.stages) if s.name == stop_at), len(self.stages))
        
        # Run stages
        success = True
        for i in range(start_idx, stop_idx):
            stage = self.stages[i]
            
            # Check dependencies
            for dep in stage.depends_on:
                if dep not in self.results or not self.results[dep]:
                    logger.error(f"Stage {stage.name} depends on {dep}, but {dep} was not successful")
                    success = False
                    break
            
            if not success:
                break
            
            # Run the stage
            stage_success = stage.run()
            self.results[stage.name] = stage_success
            
            if not stage_success:
                logger.error(f"Pipeline failed at stage: {stage.name}")
                success = False
                break
        
        # Log summary
        self._log_summary()
        
        return success
    
    def _log_summary(self) -> None:
        """Log pipeline execution summary."""
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        total_time = 0
        for stage in self.stages:
            status = "✓" if self.results.get(stage.name, False) else "✗"
            duration = stage.duration or 0
            total_time += duration
            
            print(f"{status} {stage.name:20} {duration:6.1f}s")
        
        print("-" * 60)
        print(f"Total execution time: {total_time:.1f}s")
        
        successful = sum(1 for result in self.results.values() if result)
        print(f"Successful stages: {successful}/{len(self.results)}")
        print("="*60)

def create_full_pipeline(config_path: str, **kwargs) -> Pipeline:
    """Create a complete train->analyze->ablate->compare pipeline."""
    
    pipeline = Pipeline(config_path)
    config = ntf.load_config(config_path)
    
    # Determine if this is a grid search or single model
    is_grid = 'base_config' in config and 'models' in config
    
    # Stage 1: Training
    train_args = [config_path]
    if kwargs.get('train_name'):
        train_args.extend(['--name', kwargs['train_name']])
    if kwargs.get('tags'):
        train_args.extend(['--tags'] + kwargs['tags'])
    
    pipeline.add_stage(PipelineStage(
        name="train",
        script="train.py",
        args=train_args
    ))
    
    # For grid search, we need to handle multiple models
    if is_grid:
        models = config['models']
        strategies = config.get('ablations', [])
        
        for model_spec in models:
            model_name = model_spec.get('name', 'model')
            
            # We would need model run IDs from training stage
            # This is a limitation of the current design - we'd need to parse training output
            # or use wandb API to find the latest runs
            
            logger.warning("Grid search pipeline not fully implemented - use individual scripts")
            
    else:
        # Single model pipeline
        model_name = config['model'].get('name', 'model')
        
        # Note: In a real implementation, we'd need to capture the model run ID
        # from the training stage and pass it to subsequent stages
        # For now, this serves as a template
        
        # Stage 2: Analysis
        pipeline.add_stage(PipelineStage(
            name="analyze",
            script="analyze.py",
            args=[
                config_path,
                "--model-run", "PLACEHOLDER_RUN_ID",  # Would be filled from train stage
                "--model-artifact", f"{model_name}_model"
            ],
            depends_on=["train"]
        ))
        
        # Stage 3: Ablation
        pipeline.add_stage(PipelineStage(
            name="ablate",
            script="ablate.py", 
            args=[
                config_path,
                "--model-run", "PLACEHOLDER_RUN_ID",
                "--model-artifact", f"{model_name}_model",
                "--analysis-run", "PLACEHOLDER_ANALYSIS_RUN_ID",
                "--analysis-artifact", f"analysis_{model_name}"
            ],
            depends_on=["analyze"]
        ))
    
    return pipeline

def create_compare_pipeline(project: str, **kwargs) -> Pipeline:
    """Create a comparison-only pipeline."""
    
    pipeline = Pipeline("dummy_config.yaml")  # Won't be used
    
    compare_args = ["--project", project]
    if kwargs.get('filter_tags'):
        compare_args.extend(["--filter-tags"] + kwargs['filter_tags'])
    if kwargs.get('filter_models'):
        compare_args.extend(["--filter-models"] + kwargs['filter_models'])
    if kwargs.get('filter_strategies'):
        compare_args.extend(["--filter-strategies"] + kwargs['filter_strategies'])
    if kwargs.get('output_dir'):
        compare_args.extend(["--output-dir", kwargs['output_dir']])
    
    pipeline.add_stage(PipelineStage(
        name="compare",
        script="compare.py",
        args=compare_args
    ))
    
    return pipeline

def main():
    parser = argparse.ArgumentParser(description="Run neural topology analysis pipeline")
    parser.add_argument('mode', choices=['full', 'compare'], help="Pipeline mode")
    
    # Full pipeline options
    parser.add_argument('--config', help="Configuration file (required for full mode)")
    parser.add_argument('--start-from', help="Start pipeline from specific stage")
    parser.add_argument('--stop-at', help="Stop pipeline at specific stage")
    parser.add_argument('--train-name', help="Override training experiment name")
    
    # Compare pipeline options
    parser.add_argument('--project', help="Wandb project to compare (required for compare mode)")
    parser.add_argument('--filter-tags', nargs='*', help="Filter by tags")
    parser.add_argument('--filter-models', nargs='*', help="Filter by models")
    parser.add_argument('--filter-strategies', nargs='*', help="Filter by strategies")
    parser.add_argument('--output-dir', help="Local output directory")
    
    # General options
    parser.add_argument('--tags', nargs='*', help="Additional tags")
    parser.add_argument('--dry-run', action='store_true', help="Show pipeline without running")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'full' and not args.config:
        parser.error("--config is required for full pipeline mode")
    if args.mode == 'compare' and not args.project:
        parser.error("--project is required for compare pipeline mode")
    
    # Create pipeline
    if args.mode == 'full':
        pipeline = create_full_pipeline(
            args.config,
            train_name=args.train_name,
            tags=args.tags
        )
    else:
        pipeline = create_compare_pipeline(
            args.project,
            filter_tags=args.filter_tags,
            filter_models=args.filter_models,
            filter_strategies=args.filter_strategies,
            output_dir=args.output_dir
        )
    
    # Show pipeline structure
    print("Pipeline stages:")
    for i, stage in enumerate(pipeline.stages):
        deps = f" (depends on: {', '.join(stage.depends_on)})" if stage.depends_on else ""
        print(f"  {i+1}. {stage.name}{deps}")
    
    if args.dry_run:
        print("\nDry run mode - pipeline not executed")
        return
    
    # Run pipeline
    print(f"\nStarting {args.mode} pipeline...")
    success = pipeline.run(start_from=args.start_from, stop_at=args.stop_at)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()