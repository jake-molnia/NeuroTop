"""
Wandb integration for experiment tracking.
Replaces all local file caching with cloud-based experiment management.
"""

import wandb
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """
    Centralized experiment logging with wandb.
    Handles initialization, metric logging, artifact management, and cleanup.
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.tags = tags or []
        self.group = group
        self.notes = notes
        self.run = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize wandb run with experiment configuration."""
        
        if self._initialized:
            logger.warning("Experiment already initialized")
            return
        
        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=self.config,
            tags=self.tags,
            group=self.group,
            notes=self.notes,
            reinit=True
        )
        
        self._initialized = True
        logger.info(f"Initialized wandb run: {self.experiment_name}")
        logger.info(f"Run URL: {self.run.url}")
    
    def log_metrics(
        self, 
        metrics: Dict[str, Union[float, int]], 
        step: Optional[int] = None
    ) -> None:
        """Log metrics to wandb."""
        
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        wandb.log(metrics, step=step)
        logger.debug(f"Logged metrics: {list(metrics.keys())}")
    
    def log_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        step: Optional[int] = None
    ) -> None:
        """Log pandas DataFrame as wandb Table."""
        
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        table = wandb.Table(dataframe=df)
        wandb.log({name: table}, step=step)
        logger.debug(f"Logged dataframe '{name}' with shape {df.shape}")
    
    def log_figure(
        self,
        figure: plt.Figure,
        name: str,
        step: Optional[int] = None,
        close_figure: bool = True
    ) -> None:
        """Log matplotlib figure to wandb."""
        
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        wandb.log({name: wandb.Image(figure)}, step=step)
        
        if close_figure:
            plt.close(figure)
        
        logger.debug(f"Logged figure: {name}")
    
    def save_model_artifact(
        self,
        model: torch.nn.Module,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model as wandb artifact."""
        
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        # Create artifact
        artifact = wandb.Artifact(
            name=f"{name}_model",
            type="model",
            metadata=metadata or {}
        )
        
        # Save model state dict to temporary file
        model_path = f"{name}_model.pth"
        torch.save(model.state_dict(), model_path)
        artifact.add_file(model_path)
        
        # Log artifact
        wandb.log_artifact(artifact)
        
        # Cleanup temporary file
        Path(model_path).unlink()
        
        logger.info(f"Saved model artifact: {name}")
    
    def save_tensor_artifact(
        self,
        tensor: torch.Tensor,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save tensor as wandb artifact."""
        
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        artifact = wandb.Artifact(
            name=f"{name}_tensor",
            type="tensor_data",
            metadata=metadata or {}
        )
        
        # Save tensor
        tensor_path = f"{name}_tensor.pt"
        torch.save(tensor, tensor_path)
        artifact.add_file(tensor_path)
        
        wandb.log_artifact(artifact)
        
        # Cleanup
        Path(tensor_path).unlink()
        
        logger.debug(f"Saved tensor artifact: {name}")
    
    def save_results_artifact(
        self,
        results_df: pd.DataFrame,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save results DataFrame as wandb artifact."""
        
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        artifact = wandb.Artifact(
            name=f"{name}_results",
            type="results",
            metadata=metadata or {}
        )
        
        # Save as CSV
        csv_path = f"{name}_results.csv"
        results_df.to_csv(csv_path, index=False)
        artifact.add_file(csv_path)
        
        wandb.log_artifact(artifact)
        
        # Cleanup
        Path(csv_path).unlink()
        
        logger.info(f"Saved results artifact: {name} with shape {results_df.shape}")
    
    def log_system_metrics(self) -> None:
        """Log system resource usage."""
        
        if not self._initialized:
            return
        
        try:
            import psutil
            
            metrics = {
                'system/cpu_percent': psutil.cpu_percent(),
                'system/memory_percent': psutil.virtual_memory().percent,
                'system/memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
            
            # GPU metrics if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    metrics[f'system/gpu_{i}_memory_allocated_gb'] = memory_allocated
                    metrics[f'system/gpu_{i}_memory_reserved_gb'] = memory_reserved
            
            self.log_metrics(metrics)
            
        except ImportError:
            logger.debug("psutil not available for system metrics")
    
    def finish(self) -> None:
        """Clean up and finish wandb run."""
        
        if self._initialized and self.run is not None:
            wandb.finish()
            self._initialized = False
            logger.info("Finished wandb run")

# Convenience functions for quick logging

def init_experiment(
    project_name: str,
    experiment_name: str,
    config: Dict[str, Any],
    **kwargs
) -> ExperimentLogger:
    """Initialize experiment logger with configuration."""
    
    exp_logger = ExperimentLogger(project_name, experiment_name, config, **kwargs)
    exp_logger.initialize()
    return exp_logger

def log_training_progress(
    exp_logger: ExperimentLogger,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    lr: Optional[float] = None
) -> None:
    """Log training metrics for a single epoch."""
    
    metrics = {
        'train/loss': train_loss,
        'train/accuracy': train_acc,
        'train/epoch': epoch
    }
    
    if val_loss is not None:
        metrics['val/loss'] = val_loss
    if val_acc is not None:
        metrics['val/accuracy'] = val_acc
    if lr is not None:
        metrics['train/learning_rate'] = lr
    
    exp_logger.log_metrics(metrics, step=epoch)

def log_ablation_results(
    exp_logger: ExperimentLogger,
    results_df: pd.DataFrame,
    strategy_name: str,
    model_name: str
) -> None:
    """Log ablation experiment results."""
    
    # Log summary metrics
    final_acc = results_df[results_df['percent_removed'] == results_df['percent_removed'].max()]['accuracy'].iloc[0]
    baseline_acc = results_df[results_df['percent_removed'] == 0]['accuracy'].iloc[0]
    degradation = baseline_acc - final_acc
    
    summary_metrics = {
        f'ablation/{strategy_name}/baseline_accuracy': baseline_acc,
        f'ablation/{strategy_name}/final_accuracy': final_acc,
        f'ablation/{strategy_name}/total_degradation': degradation,
        f'ablation/{strategy_name}/degradation_per_10pct': degradation / 10.0
    }
    
    exp_logger.log_metrics(summary_metrics)
    
    # Log full results
    exp_logger.log_dataframe(results_df, f'ablation_{strategy_name}_results')
    exp_logger.save_results_artifact(results_df, f'{model_name}_{strategy_name}')

def create_run_from_config(config: Dict[str, Any]) -> ExperimentLogger:
    """Create experiment logger from configuration dictionary."""
    
    # Extract wandb-specific config
    run_config = config.get('logging', {})
    
    project_name = run_config.get('project', 'neural-topology')
    experiment_name = run_config.get('name', 'experiment')
    tags = run_config.get('tags', [])
    group = run_config.get('group', None)
    
    # Create and initialize logger
    exp_logger = ExperimentLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        config=config,
        tags=tags,
        group=group
    )
    
    exp_logger.initialize()
    return exp_logger